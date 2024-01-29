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
from model import Encoder
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
        populate_partitions(os.environ['INPUT_PATH'])
        server = ServerThread(app)
        server.start()

    # set the environment variables for the process group
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = str(pytorch_port)

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
    src_vocab_size=config['src_vocab_size']
    max_length=config['max_length']

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data Loader
    dataset = StreamingTokenDataset(master_addr, config_port, sequence_length)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)

    # Create model, move it to GPU with DDP (if using GPUs)
    model = Encoder(src_vocab_size, embed_size, num_layers, heads, rank, forward_expansion, dropout, max_length).to(device)
    model = DDP(model, device_ids=[rank])

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr * world_size)  # Scale LR

    print("Starting training")

    # Training loop
    for epoch in range(1, num_epochs):
        model.train()
        sampler.set_epoch(epoch)  # Set epoch for sampler
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            if rank == 0 and batch_idx % 10 == 0:  # Print on one process
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss {loss.item()}")

    # TODO: Save model
    # output_path


if __name__ == "__main__":
    world_size, rank, master_addr, pytorch_port, config_port, output_path = setup()

    config = ask_for_config(master_addr, pytorch_port)
    print("Setting up")
    world_size, rank = setup()
    # print all the config variables
    print("Output Path", output_path,
          "Master Address", master_addr,
          "Master Port", pytorch_port,
          "Rank", rank,
          "World Size", world_size,
          "Config", config)

    # if rank == 0:
    #     while not all_partitions_completed():
    #         time.sleep(1)
    #     print("All partitions processed")
    # else:
    train(world_size, rank, master_addr, config_port, output_path, config)
