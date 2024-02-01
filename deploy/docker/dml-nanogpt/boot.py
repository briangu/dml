import os

import requests


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
    # config_port = int(os.environ.get('CONFIG_PORT', 8891))

    output_path = os.environ['OUTPUT_PATH']
    input_path = os.environ['INPUT_PATH']

    # if rank == 0:
    #     print("START: creating partitions")
    #     # populate_partitions(os.environ['INPUT_PATH'])
    #     print("STOP: created partitions")
    #     print("starting config server")
    #     # server = ServerThread(app)
    #     # server.start()

    # set the environment variables for the process group
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = str(pytorch_port)
    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)

    os.environ['LOCAL_RANK'] = "0" # k8s allocates all container GPUs at 0
    os.environ['INPUT_PATH'] = input_path
    os.environ['OUTPUT_PATH'] = output_path

