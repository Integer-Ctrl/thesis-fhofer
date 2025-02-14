import gzip
import json
import pyterrier as pt
import os
from glob import glob


# Load the configuration settings
pwd = os.path.dirname(os.path.abspath(__file__))


def load_config(filename=pwd + "/../../config.json"):
    with open(filename, "r") as f:
        config = json.load(f)
    return config


# Get the configuration settings
config = load_config()

DOCUMENT_DATASET_TARGET_NAME = config['DOCUMENT_DATASET_TARGET_NAME']
DOCUMENT_DATASET_TARGET_NAME_PYTERRIER = config['DOCUMENT_DATASET_TARGET_NAME_PYTERRIER']

SOURCE_PATH = os.path.join(config['DATA_PATH'], config["DOCUMENT_DATASET_SOURCE_NAME"])
TARGET_PATH = os.path.join(SOURCE_PATH, config["DOCUMENT_DATASET_TARGET_NAME"])

EVALUATION_METHODS = config['EVALUATION_METHODS']
NUMBER_OF_CROSS_VALIDATION_FOLDS = config['NUMBER_OF_CROSS_VALIDATION_FOLDS']
KEY_SEPARATOR = config['KEY_SEPARATOR']

BACKBONES = config['BACKBONES']  # all backbones
DUOPROMPT_PATH = config['DUOPROMPT_PATH']


# 1. Load correlation scores per evaluation
#    KEY1: eval_method---aggr_passage---tra_passage---aggr_doc---tra_doc
#    SAVING: {KEY1: {qid: score}}}
def get_key(list):
    return KEY_SEPARATOR.join(list)


def run_cross_validation(candidate_path):
    correlation_scores = {}
    files = 0
    entries_in_files = 0

    # Extract the file name
    dir_name = os.path.basename(candidate_path)
    write_path = os.path.join(LABEL_CORRELATION_AVR_PER_QUERY_PATH, f"{dir_name}.jsonl.gz")

    print(f"Processing candidates: {dir_name}")

    FILE_PATTERN = os.path.join(candidate_path, "job_*.jsonl.gz")
    for file_path in glob(FILE_PATTERN):
        with gzip.open(file_path, 'rt', encoding='UTF-8') as file:
            files += 1
            for line in file:
                entries_in_files += 1
                line = json.loads(line)

                aggr_passage = line['aggregation_method_passage']
                tra_passage = line['transformation_method_passage']

                aggr_doc = line['aggregation_method_document']
                tra_doc = line['transformation_method_document']

                eva_met = line['evaluation_method']
                correlation_per_query = line['correlation_scores']

                key1 = get_key([eva_met, aggr_passage, tra_passage, aggr_doc, tra_doc])

                # Correlation scores for each evaluation
                if key1 not in correlation_scores:
                    correlation_scores[key1] = {}
                correlation_scores[key1] = correlation_per_query

    print(f"Files: {files}, Entries: {entries_in_files}")
    print(f"Correlation scores: {len(correlation_scores)}")

    # 3. Avarage correlation scores per query
    cross_validation_scores = {}

    for key1, scores in correlation_scores.items():
        total_score = 0
        for qid, score in scores.items():
            total_score += score
        cross_validation_scores[key1] = total_score / len(scores)

    # Sort and save the cross validation scores
    cross_validation_scores = dict(sorted(cross_validation_scores.items(), key=lambda item: item[1], reverse=True))

    with gzip.open(write_path, 'wt', encoding='UTF-8') as file:
        for key, score in cross_validation_scores.items():
            file.write(json.dumps({'evaluation_method': key, 'score': score}) + '\n')


if __name__ == '__main__':
    for backbone in BACKBONES:

        # to compute on duoprompt pairwise preferences
        LABEL_RANK_CORRELATION_SCORE_PQ_AQ_PATH = os.path.join(
            TARGET_PATH, config['DUOPROMPT_PATH'], backbone, config['LABEL_RANK_CORRELATION_SCORE_PQ_AQ_PATH'])
        CANDIDATE_PATTERN = os.path.join(LABEL_RANK_CORRELATION_SCORE_PQ_AQ_PATH, '*')  # naive, union, ...

        LABEL_CORRELATION_AVR_PER_QUERY_PATH = os.path.join(
            TARGET_PATH, config['DUOPROMPT_PATH'], backbone, config['LABEL_CORRELATION_AVR_PER_QUERY_PATH'])
        if not os.path.exists(LABEL_CORRELATION_AVR_PER_QUERY_PATH):
            os.makedirs(LABEL_CORRELATION_AVR_PER_QUERY_PATH)

        for candidate_path in glob(CANDIDATE_PATTERN):
            run_cross_validation(candidate_path)
