import json
import os


# Load the configuration settings
def load_config(filename="../config.json"):
    with open(filename, "r") as f:
        config = json.load(f)
    return config


# Get the configuration settings
config = load_config()

DOCUMENT_DATASET_NAME = config['DOCUMENT_DATASET_NAME']
DOCUMENT_DATASET_NAME_PYTERRIER = config['DOCUMENT_DATASET_NAME_PYTERRIER']
DOCUMENT_DATASET_NAME_PYTHON_API = config['DOCUMENT_DATASET_NAME_PYTHON_API']

DATA_PATH = os.path.join(config['DATA_PATH'], DOCUMENT_DATASET_NAME)
DOCUMENT_DATASET_INDEX_PATH = os.path.join(DATA_PATH, config['DOCUMENT_DATASET_INDEX_PATH'])

PASSAGE_DATASET_PATH = os.path.join(DATA_PATH, config['PASSAGE_DATASET_PATH'])
PASSAGE_DATASET_SCORE_PATH = os.path.join(DATA_PATH, config['PASSAGE_DATASET_SCORE_PATH'])
PASSAGE_ID_SEPARATOR = config['PASSAGE_ID_SEPARATOR']

PASSAGES_TO_DOCUMENT_CORRELATION_SCORE_PATH = os.path.join(
    DATA_PATH, config['PASSAGES_TO_DOCUMENT_CORRELATION_SCORE_PATH'])


def setup_folder_structure():

    paths_to_create = [
        DATA_PATH,
        DOCUMENT_DATASET_INDEX_PATH,
        PASSAGE_DATASET_PATH,
        PASSAGE_DATASET_SCORE_PATH,
        PASSAGES_TO_DOCUMENT_CORRELATION_SCORE_PATH
    ]

    for path in paths_to_create:
        # Get only the directory part of the path, ignoring the file if present
        directory = os.path.dirname(path)
        if not os.path.exists(directory):
            print(f"Directory does not exist, creating: {directory}")
            os.makedirs(directory, exist_ok=True)
        else:
            print(f"Directory exists: {directory}")


setup_folder_structure()
