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

DOCUMENT_DATASET_SOURCE_NAME = config['DOCUMENT_DATASET_SOURCE_NAME']
DOCUMENT_DATASET_SOURCE_NAME_PYTERRIER = config['DOCUMENT_DATASET_SOURCE_NAME_PYTERRIER']

SOURCE_PATH = os.path.join(config['DATA_PATH'], DOCUMENT_DATASET_SOURCE_NAME)

NUMBER_OF_CROSS_VALIDATION_FOLDS = config['NUMBER_OF_CROSS_VALIDATION_FOLDS']
KEY_SEPARATOR = config['KEY_SEPARATOR']

RANK_CORRELATION_SCORE_PQ_AQ_PATH = os.path.join(
    SOURCE_PATH, config['RANK_CORRELATION_SCORE_PQ_AQ_PATH'])
FILE_PATTERN = os.path.join(RANK_CORRELATION_SCORE_PQ_AQ_PATH, "job_*.jsonl.gz")

CROSS_VALIDATION_PATH = os.path.join(SOURCE_PATH, config['CROSS_VALIDATION_PATH'])

CHATNOIR_RETRIEVAL = config['CHATNOIR_RETRIEVAL']
PT_RETRIEVERS = config['PT_RETRIEVERS']

if CHATNOIR_RETRIEVAL:
    PT_RETRIEVERS = ['BM25_chatnoir']

EVALUATION_METHODS = config['EVALUATION_METHODS']


# 1. Load correlation scores per evaluation method and retriever
#    KEY1: eval_method___retriever
#    KEY2: aggregation_method___transformation_method___metric
#    SAVING: {KEY1: {KEY2: {qid: score}}}
def get_key(list):
    return KEY_SEPARATOR.join(list)


correlation_scores_eva_ret = {}
files = 0
entries_in_files = 0
for file_path in glob.glob(FILE_PATTERN):
    with gzip.open(file_path, 'rt', encoding='UTF-8') as file:
        files += 1
        for line in file:
            entries_in_files += 1
            line = json.loads(line)
            agr_met = line['aggregation_method']
            tra_met = line['transformation_method']
            eva_met = line['evaluation_method']
            for pt_retriever in PT_RETRIEVERS:
                if pt_retriever in line['metric']:  # eg p10_BM25
                    retriever = pt_retriever
                    metric = line['metric'].replace('_' + pt_retriever, '')
            correlation_per_query = line['correlation_per_query']

            key1 = get_key([eva_met, retriever])
            key2 = get_key([agr_met, tra_met, metric])

            # Correlation scores for each evaluation method and retriever
            if key1 not in correlation_scores_eva_ret:
                correlation_scores_eva_ret[key1] = {}
            if key2 in correlation_scores_eva_ret[key1]:
                print('Dublicate:', key1, key2)
                exit()
            correlation_scores_eva_ret[key1][key2] = correlation_per_query

print(f"Files: {files}, Entries: {entries_in_files}")
print(f"Correlation scores: {len(correlation_scores_eva_ret)}")

# 2. get all qids
qids = []
dataset = pt.get_dataset(DOCUMENT_DATASET_SOURCE_NAME_PYTERRIER)
for qrel in dataset.irds_ref().qrels_iter():
    qids.append(qrel.query_id)
qids = list(set(qids))

# 3. N-fold cross validation for each retrieval method and evaluation method pair
cross_validation_scores = {}  # key should be evaluation method and retriever

fold_size = len(qids) // NUMBER_OF_CROSS_VALIDATION_FOLDS

for pt_retriever in PT_RETRIEVERS:
    for eva_method in EVALUATION_METHODS:

        key1 = get_key([eva_method, pt_retriever])
        total_score = 0

        for i in range(NUMBER_OF_CROSS_VALIDATION_FOLDS):

            max_score = -float('inf')
            max_key2 = None

            for key2, scores in correlation_scores_eva_ret[key1].items():
                test_qids = qids[i * fold_size:(i + 1) * fold_size]
                train_qids = qids[:i * fold_size] + qids[(i + 1) * fold_size:]

                # Get best avarage score for each key in train set
                score = 0
                for qid in train_qids:
                    score += scores[qid]

                score = score / len(train_qids)
                if score > max_score:
                    max_score = score
                    max_key2 = key2

            # Get avarage score for best key in test set
            score = 0
            for qid in test_qids:
                score += correlation_scores_eva_ret[key1][max_key2][qid]

            score = score / len(test_qids)
            total_score += score

        total_score = total_score / NUMBER_OF_CROSS_VALIDATION_FOLDS
        cross_validation_scores[key1] = total_score

# Sort and save the cross validation scores
cross_validation_scores = dict(sorted(cross_validation_scores.items(), key=lambda item: item[1], reverse=True))

with gzip.open(CROSS_VALIDATION_PATH, 'wt', encoding='UTF-8') as file:
    for key, score in cross_validation_scores.items():
        file.write(json.dumps({'eval_method___retriever': key, 'score': score}) + '\n')
