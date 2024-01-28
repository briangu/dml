import os
import threading

from flask import Flask, jsonify, request

app = Flask(__name__)

# Use a lock for thread-safe operations
lock = threading.Lock()

# Store information about the pods, grouped by job ID
pods_info = {}

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"}), 200

@app.route('/register', methods=['POST'])
def register_pod():
    pod_ip = request.remote_addr
    job_id = request.json.get('job_id', 'default_job')  # Default job ID if not provided

    with lock:
        if job_id not in pods_info:
            pods_info[job_id] = {
                "pods": [],
                "master_addr": pod_ip,  # First registered node of the job becomes the master
            }

        # Assign the next rank based on the number of already registered pods for this job
        rank = len(pods_info[job_id]["pods"])
        pods_info[job_id]["pods"].append({"ip": pod_ip, "rank": rank})

    data = {
        "rank": rank,
        "master_addr": pods_info[job_id]["master_addr"],
    }

    # log the data to the flask logging
    app.logger.info(f"Job ID: {job_id}, Data: {data}")

    return jsonify(data)

@app.route('/pods', methods=['GET'])
def get_pods():
    job_id = request.args.get('job_id', 'default_job')  # Default job ID if not provided
    return jsonify(pods_info.get(job_id, {}))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
