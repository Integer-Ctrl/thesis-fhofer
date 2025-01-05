import glob
import gzip
import json
import pyterrier as pt
import os


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

TARGET_PATH = os.path.join(config['DATA_PATH'], DOCUMENT_DATASET_TARGET_NAME)

EVALUATION_METHODS = config['EVALUATION_METHODS']
NUMBER_OF_CROSS_VALIDATION_FOLDS = config['NUMBER_OF_CROSS_VALIDATION_FOLDS']
KEY_SEPARATOR = config['KEY_SEPARATOR']

LABEL_RANK_CORRELATION_SCORE_PQ_AQ_PATH = os.path.join(
    TARGET_PATH, config['LABEL_RANK_CORRELATION_SCORE_PQ_AQ_PATH'])
FILE_PATTERN = os.path.join(LABEL_RANK_CORRELATION_SCORE_PQ_AQ_PATH, "job_*.jsonl.gz")

LABEL_CROSS_VALIDATION_PATH = os.path.join(TARGET_PATH, config['LABEL_CROSS_VALIDATION_PATH'])


# 1. Load correlation scores per evaluation
#    KEY1: eval_method
#    KEY2: aggr_passage---tra_passage---aggr_doc---tra_doc
#    SAVING: {KEY1: {KEY2: {qid: score}}}
def get_key(list):
    return KEY_SEPARATOR.join(list)


correlation_scores = {}
files = 0
entries_in_files = 0
for file_path in glob.glob(FILE_PATTERN):
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

            key1 = get_key([eva_met])
            key2 = get_key([aggr_passage, tra_passage, aggr_doc, tra_doc])

            # Correlation scores for each evaluation
            if key1 not in correlation_scores:
                correlation_scores[key1] = {}
            if key2 in correlation_scores[key1]:
                print('Dublicate:', key1, key2)
                exit()
            correlation_scores[key1][key2] = correlation_per_query

print(f"Files: {files}, Entries: {entries_in_files}")
print(f"Correlation scores: {len(correlation_scores)}")

# 2. get all qids
qids = []
dataset = pt.get_dataset(DOCUMENT_DATASET_TARGET_NAME_PYTERRIER)
for query in dataset.irds_ref().queries_iter():
    qids.append(query.query_id)

# 3. N-fold cross validation for each retrieval method and evaluation method pair
cross_validation_scores = {}  # key should be evaluation

fold_size = len(qids) // NUMBER_OF_CROSS_VALIDATION_FOLDS

for eva_method in EVALUATION_METHODS:

    key1 = get_key([eva_method])
    total_score = 0

    for i in range(NUMBER_OF_CROSS_VALIDATION_FOLDS):

        max_score = -float('inf')
        max_key2 = None

        for key2, scores in correlation_scores[key1].items():
            test_qids = qids[i * fold_size:(i + 1) * fold_size]
            train_qids = qids[:i * fold_size] + qids[(i + 1) * fold_size:]

            # Get best avarage score for each key in train set
            score = 0
            for qid in train_qids:
                score += scores[qid]
            if (score / len(train_qids)) > max_score:
                max_score = score
                max_key2 = key2

        # Get avarage score for best key in test set
        score = 0
        for qid in test_qids:
            score += correlation_scores[key1][max_key2][qid]

        score = score / len(test_qids)

        total_score += score

    total_score = total_score / NUMBER_OF_CROSS_VALIDATION_FOLDS
    cross_validation_scores[key1] = total_score

# Sort and save the cross validation scores
cross_validation_scores = dict(sorted(cross_validation_scores.items(), key=lambda item: item[1], reverse=True))

with gzip.open(LABEL_CROSS_VALIDATION_PATH, 'wt', encoding='UTF-8') as file:
    for key, score in cross_validation_scores.items():
        file.write(json.dumps({'evaluation_method': key, 'score': score}) + '\n')
