import os
import sys
import threading
import time

import requests
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


@app.route('/config', methods=['GET'])
def get_config():
    return(jsonify({
        'lr': float(os.environ.get('TRAIN_LR', 0.0001)),
        'num_epochs': int(os.environ.get('TRAIN_NUM_EPOCHS', 10)),
        'batch_size': int(os.environ.get('TRAIN_BATCH_SIZE', 64)),
        'sequence_length': int(os.environ.get('MODEL_SEQUENCE_LENGTH', 128)),
        'embed_size': int(os.environ.get('MODEL_EMBED_SIZE', 512)),
        'num_layers': int(os.environ.get('MODEL_NUM_LAYERS', 8)),
        'heads': int(os.environ.get('MODEL_NUM_HEADS', 8)),
        'forward_expansion': int(os.environ.get('MODEL_FORWARD_EXPANSION', 4)),
        'dropout': float(os.environ.get('MODEL_DROPOUT', 0.1)),
        'max_length': int(os.environ.get('MODEL_SEQUENCE_LENGTH', 128)), # same as sequence_length?
        'vocab_size': int(os.environ.get('MODEL_QUANTIZE_BUCKETS', 256))
    }))


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


def ask_for_config(master_addr, config_port):
    # ask the master for the config
    url = f"http://{master_addr}:{config_port}/config"
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
