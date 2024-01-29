import os
import sys
import threading
import time

import requests
import tiktoken
from flask import Flask, jsonify
from flask import request as flask_request
from werkzeug.serving import make_server


class ServerThread(threading.Thread):

    def __init__(self, app):
        super().__init__()
        self.server = make_server('0.0.0.0', int(os.environ['CONFIG_PORT']), app)
        self.ctx = app.app_context()
        self.ctx.push()

    def run(self):
        self.server.serve_forever()

    def shutdown(self):
        self.server.shutdown()

server = None


app = Flask(__name__)

STATUS_UNPROCESSED = 0
STATUS_PROCESSING = 1
STATUS_PROCESSED = 2

PARTITION_TTL = 30

# structure: {partition_id: {status: 0/1/2, offset: [start seek, end seek]}, timestamp: None}
# timestamp is the time the partition was marked as processing
partitions = {}

def is_partition_expired(partition):
    return partition['status'] == STATUS_PROCESSING and (time.time() - partition['timestamp']) > PARTITION_TTL


vocab_size = None


@app.route('/config', methods=['GET'])
def get_config():
    global vocab_size
    if vocab_size is None:
        enc = tiktoken.encoding_for_model("gpt-4")
        vocab_size = enc.n_vocab

    config = {
        'lr': 0.001,
        'num_epochs': 100,
        'batch_size': 64,
        'sequence_length': 128,
        'embed_size': 256,
        'num_layers': 8,
        'heads': 8,
        'forward_expansion': 4,
        'dropout': 0.1,
        'max_length': 100,
        'vocab_size': vocab_size,
    }

    return jsonify(config)


@app.route('/partition', methods=['GET'])
def get_partition():
    # get next unprocessed partition, mark it as processing with a TTL and return it.
    # if we encounter a stale partition, we mark it as unprocessed and return it
    now = time.time()
    for partition_id, partition in partitions.items():
        if partition['status'] == STATUS_UNPROCESSED or is_partition_expired(partition):
            partition['status'] = STATUS_PROCESSING
            partition['timestamp'] = now
            return jsonify(partition_id=partition_id, files=partition['files'])
    # TODO: send semaphore that indicates we are waiting for a partition to be available
    #           and if all are closed we send the sentinel that the work is done
    return jsonify(partition_id=None, files=None)


@app.route('/partition', methods=['POST'])
def update_partition():
    # update the partition status to processed
    partition_id = flask_request.json['partition_id']
    partitions[partition_id]['status'] = STATUS_PROCESSED
    return jsonify(message="Partition updated successfully"), 200


def populate_partitions(input_path):
    # enumerate all the files in the input path and create a partition for each one
    for root, _, files in os.walk(input_path):
        for partition_id, file in enumerate(files):
            file_path = os.path.join(root, file)
            partitions[partition_id] = {'status': STATUS_UNPROCESSED, 'files': [file_path]}

def all_partitions_completed():
    # check if all partitions are processed
    for partition in partitions.values():
        if partition['status'] != STATUS_PROCESSED:
            return False
    return True


def ask_for_config(master_addr, master_port):
    # ask the master for the config
    url = f"http://{master_addr}:{master_port}/config"
    response = requests.get(url)
    if response.status_code != 200:
        print("Failed to get config:", response.status_code, response.reason)
        raise Exception("Failed to get config")
    return response.json()


if __name__ == '__main__':
    populate_partitions(sys.argv[1])
    print("Partitions:", len(partitions))
    server = ServerThread(app)
    server.start()
