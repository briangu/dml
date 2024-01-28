import json
import os
import threading
import time

import requests
import tiktoken
from flask import Flask, jsonify
from flask import request as flask_request
from werkzeug.serving import make_server

# https://stackoverflow.com/questions/15562446/how-to-stop-flask-application-without-using-ctrl-c
class ServerThread(threading.Thread):

    def __init__(self, app):
        super().__init__()
        self.server = make_server('0.0.0.0', int(os.environ.get('PORT', 8889)), app)
        self.ctx = app.app_context()
        self.ctx.push()

    def run(self):
        self.server.serve_forever()

    def shutdown(self):
        self.server.shutdown()


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

@app.route('/partition', methods=['GET'])
def get_partition():
    # get next unprocessed partition, mark it as processing with a TTL and return it.
    # if we encounter a stale partition, we mark it as unprocessed and return it
    now = time.time()
    for partition_id, partition in partitions.items():
        if partition['status'] == STATUS_UNPROCESSED or is_partition_expired(partition):
            partition['status'] = STATUS_PROCESSING
            partition['timestamp'] = now
            return jsonify(partition_id=partition_id, offset=partition['offset'])
    # send the sentinel that the work is done
    return jsonify(partition_id=None, offset=[None, None])

@app.route('/partition', methods=['POST'])
def update_partition():
    # update the partition status to processed
    partition_id = flask_request.json['partition_id']
    partitions[partition_id]['status'] = STATUS_PROCESSED
    return jsonify(message="Partition updated successfully"), 200


def populate_partitions(input_file_path, max_chunk_size):
    file_size = os.path.getsize(input_file_path)

    # Calculate the size of each chunk
    chunk_count = file_size // max_chunk_size + 1

    # Calculate the starting position of the chunk for this worker
    for i in range(chunk_count):
        start_pos = i * max_chunk_size
        chunk_size = min((i+1)*max_chunk_size, file_size) - start_pos
        partitions[i] = {'status': STATUS_UNPROCESSED, 'offset': [start_pos, chunk_size]}


def register_with_discovery():
    """ register with the dml-discover service and get the master address and worker rank """
    url = "http://dml-discovery-service:5000/register"
    headers = {'Content-Type': 'application/json'}

    job_id = os.environ.get('JOB_UUID', 'dml-tokens')
    print(f"Current Job ID: {job_id}")

    response = requests.post(url, json={"job_id": job_id}, headers=headers)
    if response.status_code != 200:
        print(response.status_code, response.reason, response.text)
        raise Exception("Failed to register with dml-discover")
    return response.json()["master_addr"], response.json()["rank"]


def setup():
    world_size = int(os.environ['WORLD_SIZE'])

    if os.environ.get('MASTER_ADDR', None) is None:
        master_addr, rank = register_with_discovery()
    else:
        master_addr = os.environ['MASTER_ADDR']
        rank = int(os.environ['RANK'])
    master_port = int(os.environ.get('MASTER_PORT', 8889))

    input_file_path = os.environ['INPUT_FILE_PATH']
    output_path = os.environ['OUTPUT_PATH']

    # print all the config variables
    print("Input File Path", input_file_path,
          "Output Path", output_path,
          "Master Address", master_addr,
          "Master Port", master_port,
          "Rank", rank,
          "World Size", world_size)

    return world_size, rank, master_addr, master_port, input_file_path, output_path


def ask_for_partition(master_addr, master_port):
    # ask the master for a partition
    url = f"http://{master_addr}:{master_port}/partition"
    response = requests.get(url)
    if response.status_code != 200:
        print("Failed to get partition:", response.status_code, response.reason)
        raise Exception("Failed to get partition")
    return response.json()['partition_id'], response.json()['offset']


def report_partition_completion(master_addr, master_port, partition_id):
    # report the completion of the partition to the master
    url = f"http://{master_addr}:{master_port}/partition"
    response = requests.post(url, json={'partition_id': partition_id})
    if response.status_code != 200:
        print("Failed to report partition completion:", response.status_code, response.reason)
        raise Exception("Failed to report partition completion")


def trim_chunk(chunk_data):
    # skip to the first space to ensure we have a complete word
    i = 0
    while chunk_data[i] != ' ':
        i += 1
    chunk_data = chunk_data[i:]

    # skip the reverse to the first space to ensure we have a complete word
    i = -1
    while chunk_data[i] != ' ':
        i -= 1
    chunk_data = chunk_data[:i+1]

    return chunk_data


def process(world_size, rank, master_addr, master_port, input_file_path, output_path):

    # get the next partition to process
    partition_id, (start_pos, chunk_size) = ask_for_partition(master_addr, master_port)

    while partition_id is not None:
        # if the partition is None, it means the work is done
        if partition_id is None:
            return

        output_file_path = os.path.join(output_path, f"{partition_id}.json")

        # Open the file and seek to the start position of the chunk
        with open(input_file_path, 'r') as f:
            f.seek(start_pos)
            offset = 0
            completed = False
            while not completed:
                f.seek(start_pos+offset)
                try:
                    data = f.read(chunk_size)
                    chunk_data = trim_chunk(data)
                    process_chunk(chunk_data, output_file_path)
                    report_partition_completion(master_addr, master_port, partition_id)
                    completed = True
                except:
                    print("Failed to process chunk, retrying")
                    offset += 1
                    chunk_size -= 1

        partition_id, (start_pos, chunk_size) = ask_for_partition(master_addr, master_port)


def process_chunk(chunk_data, output_file_path):
    # tokenize the chunk
    enc = tiktoken.encoding_for_model("gpt-4")
    tokens = enc.encode(chunk_data)

    # write the tokens as JSON to the output file
    with open(output_file_path, 'w') as f:
        json.dump(tokens, f)


if __name__ == "__main__":
    populate_partitions(os.environ['INPUT_FILE_PATH'], int(os.environ.get('MAX_CHUNK_SIZE', 1024*1024*2)))
    print(partitions)

    # Start Flask server in a new thread
    server = ServerThread(app)
    server.start()

    process(*setup())

    print("Done")

    # signal the Flask server to stop
    requests.post("http://localhost:8889/shutdown")

    server.shutdown()
    server.join()
