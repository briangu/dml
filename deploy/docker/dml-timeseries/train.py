import os
import time

import requests
import simdjson as json
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dataset import StreamingTokenDataset
from model import TimeSeriesTransformer
from server import (ServerThread, all_partitions_completed, app,
                    ask_for_config, populate_partitions)
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler


def register_with_discovery():
    """ register with the dml-discover service and get the master address and worker rank """
    url = "http://dml-discovery-service:5000/register"
    headers = {'Content-Type': 'application/json'}

    job_id = os.environ.get('JOB_UUID', 'dml-mnist-job')
    print(f"Current Job ID: {job_id}")

    request = {"job_id": job_id}
    response = requests.post(url, json=request, headers=headers)
    if response.status_code != 200:
        print(response.status_code, response.reason, response.text)
        raise Exception("Failed to register with dml-discover")
    return response.json()["master_addr"], response.json()["rank"]


def setup():
    global server

    world_size = int(os.environ['WORLD_SIZE'])

    if os.environ.get('MASTER_ADDR', None) is None:
        master_addr, rank = register_with_discovery()
    else:
        master_addr = os.environ['MASTER_ADDR']
        rank = int(os.environ['RANK'])
    pytorch_port = int(os.environ.get('PYTORCH_PORT', 8890))
    config_port = int(os.environ.get('CONFIG_PORT', 8891))

    output_path = os.environ['OUTPUT_PATH']

    if rank == 0:
        print("START: creating partitions")
        populate_partitions(os.environ['INPUT_PATH'])
        print("STOP: created partitions")
        print("starting config server")
        server = ServerThread(app)
        server.start()

    # set the environment variables for the process group
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = str(pytorch_port)
    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    return world_size, rank, master_addr, pytorch_port, config_port, output_path


def train(world_size, rank, master_addr, config_port, output_path, config):
    lr=config['lr']
    num_epochs=config['num_epochs']
    batch_size=config['batch_size']
    sequence_length=config['sequence_length']
    embed_size=config['embed_size']
    num_layers=config['num_layers']
    heads=config['heads']
    forward_expansion=config['forward_expansion']
    dropout=config['dropout']
    vocab_size=config['vocab_size']
    max_length=config['max_length']

    # Device configuration
    device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')

    # Data Loader
    dataset = StreamingTokenDataset(master_addr, config_port, sequence_length, vocab_size)
    dataloader = DataLoader(dataset, batch_size=batch_size)

    # Create model, move it to GPU with DDP (if using GPUs)
    model = TimeSeriesTransformer(embed_size, num_layers, heads, forward_expansion, dropout, max_length, vocab_size).to(device)
    model = DDP(model)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr * world_size)  # Scale LR

    print("Starting training")

    # Training loop
    for epoch in range(1, num_epochs):
        model.train()
        # sampler.set_epoch(epoch)  # Set epoch for sampler
        for batch_idx, (data, target) in enumerate(dataloader):
            # target = target.unsqueeze(1).expand(-1, sequence_length)
            print(f"Epoch {epoch}, Batch {batch_idx}, Data {data.shape}, Target {target.shape}")
            # data = torch.stack(data, dim=1)
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            # print(f"Output {output.shape}, Target {target.shape}")
            # output = output.transpose(1, 2)
            # target = target.view(-1)
            print(f"Output {output.shape}, Target {target.shape}")
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss {loss.item()}")

        if rank == 0:
            print(f"Epoch {epoch}, Loss {loss.item()}")
            # save the model
            torch.save(model.state_dict(), f"{output_path}/model_{epoch}.pt")
            # if all_partitions_completed():
            #     print("All partitions completed")
            #     break

    if rank == 0:
        print("Training completed")
        torch.save(model.state_dict(), f"{output_path}/model_final.pt")


if __name__ == "__main__":
    print("Setting up")
    world_size, rank, master_addr, pytorch_port, config_port, output_path = setup()
    print("Getting config")
    config = ask_for_config(master_addr, config_port)
    # print all the config variables
    print("Output Path", output_path,
          "Master Address", master_addr,
          "PyTorch Port", pytorch_port,
          "Config Port", config_port,
          "Rank", rank,
          "World Size", world_size,
          "Config", config)

    train(world_size, rank, master_addr, config_port, output_path, config)
