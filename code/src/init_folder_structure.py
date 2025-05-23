import json
import os


# Load the configuration settings
pwd = os.path.dirname(os.path.abspath(__file__))


def load_config(filename= os.path.join(pwd, "config.json")):
    with open(filename, "r") as f:
        config = json.load(f)
    return config


# Get the configuration settings
config = load_config()

# Build absolute paths from the configuration


def build_paths(config):

    DATA_PATH = config["DATA_PATH"]
    SOURCE_PATH = os.path.join(DATA_PATH, config["DOCUMENT_DATASET_SOURCE_NAME"])
    TARGET_PATH = os.path.join(SOURCE_PATH, config["DOCUMENT_DATASET_TARGET_NAME"])
    DUOPROMPT_PATH = os.path.join(TARGET_PATH, config["DUOPROMPT_PATH"])
    MONOPROMPT_PATH = os.path.join(TARGET_PATH, config["MONOPROMPT_PATH"])

    paths = {
        # Dataset used to transfer from
        "DOCUMENT_DATASET_SOURCE_INDEX_PATH": os.path.join(SOURCE_PATH, config["DOCUMENT_DATASET_SOURCE_INDEX_PATH"]),
        "PASSAGE_DATASET_SOURCE_PATH": os.path.join(SOURCE_PATH, config["PASSAGE_DATASET_SOURCE_PATH"]),
        "PASSAGE_DATASET_SOURCE_INDEX_PATH": os.path.join(SOURCE_PATH, config["PASSAGE_DATASET_SOURCE_INDEX_PATH"]),

        "PASSAGE_DATASET_SOURCE_SCORE_REL_PATH": os.path.join(
            SOURCE_PATH, config["PASSAGE_DATASET_SOURCE_SCORE_REL_PATH"]),
        "PASSAGE_DATASET_SOURCE_SCORE_AQ_PATH": os.path.join(
            SOURCE_PATH, config["PASSAGE_DATASET_SOURCE_SCORE_AQ_PATH"]),

        "RANK_CORRELATION_SCORE_PQ_AQ_PATH": os.path.join(
            SOURCE_PATH, config["RANK_CORRELATION_SCORE_PQ_AQ_PATH"]),

        # Dataset used to transfer to
        "DOCUMENT_DATASET_TARGET_INDEX_PATH": os.path.join(TARGET_PATH, config["DOCUMENT_DATASET_TARGET_INDEX_PATH"]),
        "PASSAGE_DATASET_TARGET_PATH": os.path.join(TARGET_PATH, config["PASSAGE_DATASET_TARGET_PATH"]),
        "PASSAGE_DATASET_TARGET_INDEX_PATH": os.path.join(TARGET_PATH, config["PASSAGE_DATASET_TARGET_INDEX_PATH"]),

        "CANDIDATES_LOCAL_PATH": os.path.join(TARGET_PATH, config["CANDIDATES_LOCAL_PATH"]),
        "CANDIDATE_CHATNOIR_PATH": os.path.join(TARGET_PATH, config["CANDIDATE_CHATNOIR_PATH"]),

        "DUOPROMPT_PATH": DUOPROMPT_PATH,
        "MONOPROMPT_PATH": MONOPROMPT_PATH,

        "DUOPROMPT_LABEL_RANK_CORRELATION_SCORE_PQ_AQ_PATH": os.path.join(
            DUOPROMPT_PATH, config["LABEL_RANK_CORRELATION_SCORE_PQ_AQ_PATH"]),
        "MONOPROMPT_LABEL_RANK_CORRELATION_SCORE_PQ_AQ_PATH": os.path.join(
            MONOPROMPT_PATH, config["LABEL_RANK_CORRELATION_SCORE_PQ_AQ_PATH"]),

        "DUOPROMPT_LABEL_CROSS_VALIDATION_PATH": os.path.join(DUOPROMPT_PATH, config["LABEL_CROSS_VALIDATION_PATH"]),
        "MONOPROMPT_LABEL_CROSS_VALIDATION_PATH": os.path.join(MONOPROMPT_PATH, config["LABEL_CROSS_VALIDATION_PATH"]),

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
