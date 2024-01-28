import uuid
import yaml
import subprocess
import tempfile
import sys

def generate_uuid():
    return str(uuid.uuid4())

def update_job_manifest(file_path, job_uuid):
    with open(file_path, 'r') as file:
        job_manifest = yaml.safe_load(file)

    # Update the UUID in the job manifest
    for container in job_manifest['spec']['template']['spec']['containers']:
        if container['name'] == 'dml-minst-container':
            for env_var in container['env']:
                if env_var['name'] == 'JOB_UUID':
                    env_var['value'] = job_uuid
                    break

    return job_manifest

def apply_job_manifest(job_manifest):
    # Create a temporary file and write the updated job manifest to it
    with tempfile.NamedTemporaryFile(mode='w') as file:
        print(job_manifest)
        yaml.dump(job_manifest, file)
        print(file.name)
        subprocess.run(["kubectl", "apply", "-f", file.name], check=True)

if __name__ == "__main__":
    job_uuid = generate_uuid()
    file_path = sys.argv[1]
    updated_manifest = update_job_manifest(file_path, job_uuid)
    apply_job_manifest(updated_manifest)
    print(f"Job with UUID {job_uuid} launched.")
