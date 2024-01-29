import json
from torch.utils.data import IterableDataset
import requests

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


class StreamingTokenDataset(IterableDataset):
    def __init__(self, master_addr, master_port, sequence_length):
        self.master_addr = master_addr
        self.master_port = master_port
        self.sequence_length = sequence_length

    def load_file(self, file_path):
        with open(file_path, 'r') as file:
            # Load the JSON data
            for line in file:
                data = json.loads(line)
                # Assuming each line in the file is a JSON array of tokens
                for i in range(len(data) - self.sequence_length):
                    yield data[i:i + self.sequence_length], data[i + self.sequence_length]

    def __iter__(self):
        # Ask the master for a partition
        partition_id, files = ask_for_partition(self.master_addr, self.master_port)
        print("Partition ID", partition_id, "Files", files)

        # while there are partitions to process
        while partition_id is not None:
            for file in files:
                yield from self.load_file(file)
            report_partition_completion(self.master_addr, self.master_port, partition_id)
            # Ask the master for a partition
            partition_id, files = ask_for_partition(self.master_addr, self.master_port)
            print("Partition ID", partition_id, "Files", files)
