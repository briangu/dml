# import json
import os
import threading
import time

import requests
import tiktoken
from flask import Flask, jsonify
from flask import request as flask_request
from werkzeug.serving import make_server
import simdjson as json

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
    # TODO: send semaphore that indicates we are waiting for a partition to be available
    #           and if all are closed we send the sentinel that the work is done
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
    global server

    world_size = int(os.environ['WORLD_SIZE'])

    if os.environ.get('MASTER_ADDR', None) is None:
        master_addr, rank = register_with_discovery()
    else:
        master_addr = os.environ['MASTER_ADDR']
        rank = int(os.environ['RANK'])
    master_port = int(os.environ.get('MASTER_PORT', 8889))

    input_file_path = os.environ['INPUT_FILE_PATH']
    output_path = os.environ['OUTPUT_PATH']

    if rank == 0:
        populate_partitions(os.environ['INPUT_FILE_PATH'], int(os.environ.get('MAX_CHUNK_SIZE', 1024*1024*2)))
        print("Partitions:", len(partitions))
        server = ServerThread(app)
        server.start()

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


def all_partitions_completed():
    # check if all partitions are processed
    for partition in partitions.values():
        if partition['status'] != STATUS_PROCESSED:
            return False
    return True


def process(world_size, rank, master_addr, master_port, input_file_path, output_path):

    if rank == 0:
        while not all_partitions_completed():
            time.sleep(1)
        print("All partitions processed")
        return

    # get the next partition to process
    partition_id, (start_pos, chunk_size) = ask_for_partition(master_addr, master_port)

    attempts = 0
    while partition_id is not None:
        # if the partition is None, it means the work is done
        if partition_id is None:
            attempts += 1
            if attempts > 5:
                print("No more partitions to process")
                break
            time.sleep(1)
            continue

        output_file_path = os.path.join(output_path, f"{partition_id}.json")

        try:
            # Open the file and seek to the start position of the chunk
            with open(input_file_path, 'r', encoding='utf8') as f:
                f.seek(start_pos)
                head_offset = 0
                tail_offset = 0
                completed = False
                while not completed:
                    f.seek(start_pos+head_offset)
                    try:
                        data = f.read(chunk_size - tail_offset)
                        chunk_data = trim_chunk(data)
                        if len(chunk_data) == 0:
                            raise Exception("Chunk is empty")
                        print("Processing chunk", partition_id, head_offset, chunk_size, len(chunk_data))
                        process_chunk(chunk_data, output_file_path)
                        report_partition_completion(master_addr, master_port, partition_id)
                        completed = True
                    except:
                        print("Failed to process chunk, retrying", partition_id, head_offset, tail_offset, chunk_size)
                        if head_offset == tail_offset:
                            head_offset += 1
                        else:
                            tail_offset += 1
        except Exception as e:
            print("Failed to process chunk", partition_id, e)
            report_partition_completion(master_addr, master_port, partition_id)

        partition_id, (start_pos, chunk_size) = ask_for_partition(master_addr, master_port)


def process_chunk(chunk_data, output_file_path):
    # tokenize the chunk
    enc = tiktoken.encoding_for_model("gpt-4")
    tokens = enc.encode(chunk_data)

    # write the tokens as JSON to the output file
    with open(output_file_path, 'w') as f:
        json.dump(tokens, f)


if __name__ == "__main__":
    process(*setup())

    print("Done")

    if server is not None:
        server.shutdown()
        server.join()
