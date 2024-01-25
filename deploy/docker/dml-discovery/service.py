import os
import threading

from flask import Flask, jsonify, request

app = Flask(__name__)

# Use a lock for thread-safe operations
lock = threading.Lock()

# Store information about the pods
pods_info = {
    "pods": [],
    "master_addr": None,
    "master_port": None,
}

@app.route('/health', methods=['GET'])
def health_check():
    # Perform any necessary checks here (e.g., database connection, external service availability, etc.)
    # For a simple check, just return a 'success' message
    return jsonify({"status": "healthy"}), 200

@app.route('/register', methods=['POST'])
def register_pod():
    pod_ip = request.remote_addr
    with lock:
        if not pods_info["pods"]:
            # First registered node becomes the master
            pods_info["master_addr"] = pod_ip
        # Assign the next rank based on the number of already registered pods
        rank = len(pods_info["pods"])
        pods_info["pods"].append({"ip": pod_ip, "rank": rank})

    return jsonify({
        "rank": rank,
        "master_addr": pods_info["master_addr"],
        "master_port": pods_info["master_port"]
    })

@app.route('/pods', methods=['GET'])
def get_pods():
    return jsonify(pods_info)

if __name__ == '__main__':
    pods_info['master_port'] = int(os.environ.get('MASTER_PORT',8888))
    app.run(host='0.0.0.0', port=5000, debug=True)
