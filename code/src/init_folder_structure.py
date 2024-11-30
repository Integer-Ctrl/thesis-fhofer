import json
import os


# Load the configuration settings
def load_config(filename="./config.json"):
    with open(filename, "r") as f:
        config = json.load(f)
    return config


# Get the configuration settings
config = load_config()

# Build absolute paths from the configuration


def build_paths(config):

    DATA_PATH = config["DATA_PATH"]
    OLD_PATH = os.path.join(DATA_PATH, config["DOCUMENT_DATASET_OLD_NAME"])
    NEW_PATH = os.path.join(DATA_PATH, config["DOCUMENT_DATASET_NEW_NAME"])

    paths = {
        # Dataset used to transfer from
        "DOCUMENT_DATASET_OLD_INDEX_PATH": os.path.join(OLD_PATH, config["DOCUMENT_DATASET_OLD_INDEX_PATH"]),
        "PASSAGE_DATASET_OLD_PATH": os.path.join(OLD_PATH, config["PASSAGE_DATASET_OLD_PATH"]),
        "PASSAGE_DATASET_OLD_INDEX_PATH": os.path.join(OLD_PATH, config["PASSAGE_DATASET_OLD_INDEX_PATH"]),

        "PASSAGE_DATASET_OLD_SCORE_REL_PATH": os.path.join(OLD_PATH, config["PASSAGE_DATASET_OLD_SCORE_REL_PATH"]),
        "PASSAGE_DATASET_OLD_SCORE_AQ_PATH": os.path.join(OLD_PATH, config["PASSAGE_DATASET_OLD_SCORE_AQ_PATH"]),

        "PASSAGES_TO_DOCUMENT_CORRELATION_SCORE_PATH": os.path.join(
            OLD_PATH, config["PASSAGES_TO_DOCUMENT_CORRELATION_SCORE_PATH"]),
        "PASSAGES_TO_DOCUMENT_CORRELATION_SCORE_AQ_PATH": os.path.join(
            OLD_PATH, config["PASSAGES_TO_DOCUMENT_CORRELATION_SCORE_AQ_PATH"]),
        "PASSAGES_TO_DOCUMENT_CORRELATION_SCORE_PQ_PATH": os.path.join(
            OLD_PATH, config["PASSAGES_TO_DOCUMENT_CORRELATION_SCORE_PQ_PATH"]),
        "PASSAGES_TO_DOCUMENT_CORRELATION_SCORE_PQ_AQ_PATH": os.path.join(
            OLD_PATH, config["PASSAGES_TO_DOCUMENT_CORRELATION_SCORE_PQ_AQ_PATH"]),

        # Dataset used to transfer to
        "DOCUMENT_DATASET_NEW_INDEX_PATH": os.path.join(NEW_PATH, config["DOCUMENT_DATASET_NEW_INDEX_PATH"]),
        "PASSAGE_DATASET_NEW_PATH": os.path.join(NEW_PATH, config["PASSAGE_DATASET_NEW_PATH"]),
        "PASSAGE_DATASET_NEW_INDEX_PATH": os.path.join(NEW_PATH, config["PASSAGE_DATASET_NEW_INDEX_PATH"]),

        "CANIDATES_PATH": os.path.join(NEW_PATH, config["CANIDATES_PATH"]),
        "DUOT5_QID_DOC_DOC_SYSTEM_SCORES_PATH": os.path.join(NEW_PATH, config["DUOT5_QID_DOC_DOC_SYSTEM_SCORES_PATH"]),
    }
    return paths


# Set up the folder structure
def setup_folder_structure(paths):
    for key, path in paths.items():
        directory = os.path.dirname(path)
        if not os.path.exists(directory):
            print(f"Directory does not exist, creating: {directory}")
            os.makedirs(directory, exist_ok=True)
        else:
            print(f"Directory exists: {directory}")


# Build paths and set up the structure
paths = build_paths(config)
setup_folder_structure(paths)
