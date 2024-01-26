import os
import sys

import requests
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms


def register_with_discovery():
    # register with the dml-discover service and get the master address, port and rank
    response = requests.post("http://dml-discovery-service:5000/register")
    if response.status_code != 200:
        raise Exception("Failed to register with dml-discover")
    return response.json()["master_addr"], response.json()["master_port"], response.json()["rank"]


def setup():
    world_size = int(os.environ['WORLD_SIZE'])

    # Assuming you have a discovery service that provides these details
    master_addr, master_port, rank = register_with_discovery()  # Replace with your discovery mechanism

    print("Master Address", master_addr, "Master Port", master_port, "Rank", rank, "World Size", world_size)

    os.environ['MASTER_ADDR'] = str(master_addr)
    os.environ['MASTER_PORT'] = str(master_port)

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    return world_size, rank


class MNISTNet(nn.Module):
    def __init__(self):
        super(MNISTNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3)
        # self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(800, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
#        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 800)
        x = F.relu(self.fc1(x))
        # x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def train(lr=0.001):
    print("Setting up")

    world_size, rank = setup()

    print("Rank", rank, "World Size", world_size)

    # Load MNIST dataset.
    transform = transforms.Compose([
        transforms.ToTensor(),
        # add rotation
        transforms.RandomRotation(10),
        ]) #, transforms.Normalize((0.1307,), (0.3081,))])
    dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=64)

    # Create model, move it to GPU with DDP.
    # model = MNISTNet().to(rank)
    # model = DDP(model, device_ids=[rank])
    model = MNISTNet()
    model = DDP(model)

    # Loss function and optimizer.
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print("Starting training")

    # Training loop.
    for epoch in range(1, 100):
        model.train()
        for batch_idx, (data, target) in enumerate(dataloader):
            # data, target = data.to(rank), target.to(rank)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % 10 == 0:
                print(f"Rank {rank}, Epoch {epoch}, Batch {batch_idx}, Loss {loss.item()}")


if __name__ == "__main__":
    train(float(os.environ.get('LEARNING_RATE', 0.001)))
    sys.exit(0)

