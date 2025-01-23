import sys
import os
import json
import pyterrier as pt


def load_config(filename="/mnt/ceph/storage/data-tmp/current/ho62zoq/thesis-fhofer/code/src/config.json"):
    with open(filename, "r") as f:
        config = json.load(f)
    return config


# Get the configuration settings
config = load_config()
DOCUMENT_DATASET_SOURCE_NAME = config['DOCUMENT_DATASET_SOURCE_NAME']
DOCUMENT_DATASET_SOURCE_NAME_PYTERRIER = config['DOCUMENT_DATASET_SOURCE_NAME_PYTERRIER']
SOURCE_PATH = os.path.join(config['DATA_PATH'], DOCUMENT_DATASET_SOURCE_NAME)
DATA_PATH = config['DATA_PATH']

JOB_ID = int(sys.argv[1])
NUM_JOBS = int(sys.argv[2])

# Create file in data path
with open(os.path.join(DATA_PATH, "ray", f"output_{JOB_ID}.txt"), "w") as f:
    f.write(f"Hello from process {JOB_ID} of {NUM_JOBS}!\n")
