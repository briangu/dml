import os
import threading
from collections import Counter

import requests
from flask import Flask, jsonify, request
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer


app = Flask(__name__)

lock = threading.Lock()
global_counter = Counter()
update_count = 0
on_complete = None


def train_tokenizer(iter, output_path):
    # Create a BPE tokenizer instance
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

    # Create and configure the trainer
    trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])

    # Train the tokenizer
    tokenizer.train_from_iterator(iterator=iter, trainer=trainer)

    # Save the tokenizer
    tokenizer.save(output_path=output_path, pretty=True)


@app.route('/merge', methods=['POST'])
def merge_counts():
    global update_count, on_complete

    if request.method == 'POST':
        data = request.json
        # enumerate over the data and update the global counter
        with lock:
            global_counter.update(data)
            update_count += 1
        print("Update count", update_count, "World size", world_size, "Global counter", len(global_counter))
        if update_count == world_size:
            print("All counts merged")
            on_complete()
        return jsonify({"message": "Counts merged successfully"}), 200


def start_server():
    app.run(host='0.0.0.0', port=5000)


def report_to_master(final_counts, master_addr):
    url = f"http://{master_addr}:5000/report"
    headers = {'Content-Type': 'application/json'}
    response = requests.post(url, json=final_counts, headers=headers)
    if response.status_code != 200:
        print("Failed to report to master:", response.status_code, response.reason)
    else:
        print("Reported to master successfully")


def register_with_discovery():
    """ register with the dml-discover service and get the master address and worker rank """
    url = "http://dml-discovery-service:5000/register"
    headers = {'Content-Type': 'application/json'}

    job_id = os.environ.get('JOB_UUID', 'dml-tokens')
    print(f"Current Job ID: {job_id}")

    request = {"job_id": job_id}
    response = requests.post(url, json=request, headers=headers)
    if response.status_code != 200:
        print(response.status_code, response.reason, response.text)
        raise Exception("Failed to register with dml-discover")
    return response.json()["master_addr"], response.json()["rank"]


def setup():
    input_file_path = os.environ['INPUT_FILE_PATH']
    output_file_path = os.environ['OUTPUT_FILE_PATH']
    world_size = int(os.environ['WORLD_SIZE'])
    master_addr, rank = register_with_discovery()
    master_port = int(os.environ.get('MASTER_PORT', 8889))

    print("Master Address", master_addr, "Master Port", master_port, "Rank", rank, "World Size", world_size)

    return world_size, rank, master_addr, master_port, input_file_path, output_file_path


def process(world_size, rank, master_addr, master_port, input_file_path):
    file_size = os.path.getsize(input_file_path)

    # Calculate the size of each chunk
    chunk_size = file_size // world_size

    # Calculate the starting position of the chunk for this worker
    start_pos = rank * chunk_size

    # Open the file and seek to the start position of the chunk
    with open(input_file_path, 'r', encoding='utf8') as f:
        if rank != 0:  # If not the first worker, seek to start_pos
            f.seek(start_pos)

        chunk_data = f.read(chunk_size)

        # skip to teh first space to ensure we have a complete word
        i = 0
        while chunk_data[i] != ' ':
            i += 1
        chunk_data = chunk_data[i:]

        # skip the reverse to the first space to ensure we have a complete word
        i = -1
        while chunk_data[i] != ' ':
            i -= 1
        chunk_data = chunk_data[:i]

        process_chunk(chunk_data)

        if rank != 0:
            report_to_master(global_counter, master_addr, master_port)


def process_chunk(chunk_data):
    # process the incoming text and update the global counter
    global_counter.update(chunk_data.split())


if __name__ == "__main__":
    # Start Flask server in a new thread
    server_thread = threading.Thread(target=start_server)
    server_thread.start()

    world_size, rank, master_addr, master_port, input_file_path, output_file_path = setup()

    def _on_complete():
        print("Training tokenizer")
        train_tokenizer(global_counter.items(), output_file_path)
    on_complete = _on_complete

    process(world_size, rank, master_addr, master_port, input_file_path)
