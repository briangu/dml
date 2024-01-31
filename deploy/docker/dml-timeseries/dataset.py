import json

import numpy as np
import requests
import torch
from torch.utils.data import IterableDataset
import pandas as pd


def ask_for_partition(master_addr, master_port):
    # ask the master for a partition
    url = f"http://{master_addr}:{master_port}/partition"
    response = requests.get(url)
    if response.status_code != 200:
        print("Failed to get partition:", response.status_code, response.reason)
        raise Exception("Failed to get partition")
    return response.json()['partition_id'], response.json()['files']


def report_partition_completion(master_addr, master_port, partition_id):
    # report the completion of the partition to the master
    url = f"http://{master_addr}:{master_port}/partition"
    response = requests.post(url, json={'partition_id': partition_id})
    if response.status_code != 200:
        print("Failed to report partition completion:", response.status_code, response.reason)
        raise Exception("Failed to report partition completion")


def quantize_time_series(data, window_size, vocab_size):
    quantized_data = []
    targets = []

    for i in range(len(data) - window_size):
        window = data[i:i + window_size]
        next_value = data[i + window_size]

        # Scaling based on the window's min and max
        min_val, max_val = np.min(window), np.max(window)
        range_val = max_val - min_val
        window_scaled = (window - min_val) / range_val
        next_value_scaled = (next_value - min_val) / range_val

        # Quantization
        quantized_window = np.digitize(window_scaled, np.linspace(0, 1, vocab_size-1), right=False)# - 1
        quantized_target = np.digitize([next_value_scaled], np.linspace(0, 1, vocab_size-1), right=False)[0]# - 1

        # verify that all values in quantized_target are within the range of vocab_size
        assert np.all(0 <= quantized_window) and np.all(quantized_window < vocab_size), f"quantized_window: {quantized_window} not in range [0, {vocab_size})"
        assert 0 <= quantized_target < vocab_size, f"quantized_target: {quantized_target} not in range [0, {vocab_size})"

        quantized_data.append(quantized_window.tolist())
        targets.append(quantized_target)

    return torch.tensor(quantized_data), torch.tensor(targets)


class StreamingTokenDataset(IterableDataset):
    def __init__(self, master_addr, master_port, sequence_length, vocab_size):
        self.master_addr = master_addr
        self.master_port = master_port
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size

    def _load_file(self, file_path):
        # read the pickled pandas file
        raw_data = pd.read_pickle(file_path)
        raw_data = raw_data['close'].values

        quantized_data, targets = quantize_time_series(raw_data, self.sequence_length, self.vocab_size)

        for i in range(len(quantized_data)):
            yield quantized_data[i], targets[i]

    def __iter__(self):
        partition_id, files = ask_for_partition(self.master_addr, self.master_port)

        while partition_id is not None:
            print("Partition ID", partition_id, "Files", files)

            for file_path in files:
                yield from self._load_file(file_path)

            report_partition_completion(self.master_addr, self.master_port, partition_id)
            partition_id, files = ask_for_partition(self.master_addr, self.master_port)
