import gzip
import json
import pyterrier as pt
import os


# Load the configuration settings
pwd = os.path.dirname(os.path.abspath(__file__))


def load_config(filename=pwd + "/../config.json"):
    with open(filename, "r") as f:
        config = json.load(f)
    return config


# Get the configuration settings
config = load_config()

DOCUMENT_DATASET_SOURCE_NAME = config['DOCUMENT_DATASET_SOURCE_NAME']
DOCUMENT_DATASET_SOURCE_NAME_PYTERRIER = config['DOCUMENT_DATASET_SOURCE_NAME_PYTERRIER']

OLD_PATH = os.path.join(config['DATA_PATH'], DOCUMENT_DATASET_SOURCE_NAME)

PT_RETRIEVERS = config['PT_RETRIEVERS']
EVALUATION_METHODS = config['EVALUATION_METHODS']

NUMBER_OF_CROSS_VALIDATION_FOLDS = config['NUMBER_OF_CROSS_VALIDATION_FOLDS']
KEY_SEPARATOR = config['KEY_SEPARATOR']

RANK_CORRELATION_SCORE_PATH = os.path.join(
    OLD_PATH, config['RANK_CORRELATION_SCORE_PQ_AQ_PATH'])
CROSS_VALIDATION_PATH = os.path.join(OLD_PATH, config['CROSS_VALIDATION_PATH'])


# 1. Load correlation scores per evaluation method and retriever
#    KEY1: eval_method___retriever
#    KEY2: aggregation_method___transformation_method___metric
#    SAVING: {KEY1: {KEY2: {qid: score}}}
def get_key(list):
    return KEY_SEPARATOR.join(list)


correlation_scores_eva_ret = {}
with gzip.open(RANK_CORRELATION_SCORE_PATH, 'rt', encoding='UTF-8') as file:
    for line in file:
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
        correlation_scores_eva_ret[key1][key2] = correlation_per_query


# 2. get all qids
qids = []
dataset = pt.get_dataset(DOCUMENT_DATASET_SOURCE_NAME_PYTERRIER)
for query in dataset.irds_ref().queries_iter():
    qids.append(query.query_id)

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
                if (score / len(train_qids)) > max_score:
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
