import re
import gzip
from tqdm import tqdm
import json
import numpy as np
import pyterrier as pt
import os
import copy
from greedy_series import GreedySeries
import time
from glob import glob
import sys


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
NUMBER_OF_CROSS_VALIDATION_FOLDS = config['NUMBER_OF_CROSS_VALIDATION_FOLDS']

SOURCE_PATH = os.path.join(config['DATA_PATH'], DOCUMENT_DATASET_SOURCE_NAME)

# Pattern to match the files
PASSAGE_DATASET_SOURCE_SCORE_AQ_PATH = os.path.join(SOURCE_PATH, config['PASSAGE_DATASET_SOURCE_SCORE_AQ_PATH'])
FILE_PATTERN = os.path.join(PASSAGE_DATASET_SOURCE_SCORE_AQ_PATH, "qid_*.jsonl.gz")
# Regular expression to extract the number
NUMBER_PATTERN = re.compile(r"qid_(\d+)\.jsonl\.gz")

RANK_CORRELATION_SCORE_PQ_AQ_PATH = os.path.join(
    SOURCE_PATH, config['RANK_CORRELATION_SCORE_PQ_AQ_PATH'])

PASSAGE_ID_SEPARATOR = config['PASSAGE_ID_SEPARATOR']

AGGREGATION_METHODS = config['AGGREGATION_METHODS']
TRANSFORMATION_METHODS = config['TRANSFORMATION_METHODS']
EVALUATION_METHODS = config['EVALUATION_METHODS']

METRICS = []
for metric in config['METRICS']:
    for retriever in config['PT_RETRIEVERS']:
        METRICS.append(metric + '_' + retriever)

# Script should only compute passage scores for none existing qids
if len(sys.argv) < 3:
    print("Please provide a job ID and the number of jobs as an argument.")
    sys.exit(1)

JOB_ID = int(sys.argv[1])
NUM_JOBS = int(sys.argv[2])

# Read passsage scores and cache them
qid_docno_passages_scores_cache = {}
for file_path in glob(FILE_PATTERN):
    # Extract the file name
    file_name = os.path.basename(file_path)
    # Extract the query ID from the file path
    qid = int(NUMBER_PATTERN.search(file_name).group(1))

    with gzip.open(file_path, 'rt', encoding='UTF-8') as file:
        for line in file:
            line = json.loads(line)
            docno, passageno = line['docno'].split(PASSAGE_ID_SEPARATOR)
            qid = line['qid']
            if qid not in qid_docno_passages_scores_cache:
                qid_docno_passages_scores_cache[qid] = {}
            if docno not in qid_docno_passages_scores_cache[qid]:
                qid_docno_passages_scores_cache[qid][docno] = []
            qid_docno_passages_scores_cache[qid][docno] += [line]


# Function to get dictonary of aggregated score for a document using passage scores
# Return a list of dictionaries with aggregated scores for each document {docno, qid, metric: score}
def get_qid_docno_aggregated_scores(qid_docno_passages_scores, aggregation_method):
    # All metrics that are available in the passage scores

    aggregated_scores = []

    for qid, docno_passages_scores in qid_docno_passages_scores.items():
        for docno, passages_scores in docno_passages_scores.items():

            aggregated_doc_scores = {'docno': docno, 'qid': qid, 'label': passages_scores[0]['label']}
            for metric in METRICS:
                scores = [passage[metric] for passage in passages_scores]

                if aggregation_method == 'mean':
                    aggregated_doc_scores[metric] = float(np.mean(scores))
                elif aggregation_method == 'max':
                    aggregated_doc_scores[metric] = float(np.max(scores))
                elif aggregation_method == 'min':
                    aggregated_doc_scores[metric] = float(np.min(scores))
                elif aggregation_method == 'sum':
                    aggregated_doc_scores[metric] = float(np.sum(scores))

            aggregated_scores.append(aggregated_doc_scores)

    return aggregated_scores


# Function to get transformed scores
def get_qid_docno_transformed_scores(qid_docno_aggregated_scores, transformation_method, bins=[0.3, 0.7]):

    qid_docno_aggregated_scores_transformed = copy.deepcopy(qid_docno_aggregated_scores)
    for entry in qid_docno_aggregated_scores_transformed:
        for metric in METRICS:
            if transformation_method == 'id':
                pass
            elif transformation_method == 'log' and entry[metric] > 0:
                entry[metric] = float(np.log(entry[metric]))
            elif transformation_method == 'exp':
                entry[metric] = float(np.exp(entry[metric]))
            elif transformation_method == 'sqrt' and entry[metric] > 0:
                entry[metric] = float(np.sqrt(entry[metric]))
            # elif transformation_method == 'binned':
            #     entry[metric] = float(np.digitize(entry[metric], bins))

    return qid_docno_aggregated_scores_transformed


# Function to get evaluated score based on the specified metric and evaluation method (pearson, spearman, kendall)
def get_evaluated_score(qid_docno_transformed_scores, qid,
                        metric='ndcg10_bm25', evaluation_method='pearson'):

    # Lists to store the matched scores for correlation calculation
    transformed_scores = []
    relevance_labels = []

    # Filter scores for the given query ID (qid)
    filtered_scores = [entry for entry in qid_docno_transformed_scores if entry['qid'] == qid]

    transformed_scores = [entry[metric] for entry in filtered_scores]
    relevance_labels = [entry['label'] for entry in filtered_scores]

    # Ensure we have pairs to evaluate correlation
    if len(transformed_scores) > 1:
        # Convert lists to pandas Series
        transformed_series = GreedySeries(transformed_scores)
        relevance_series = GreedySeries(relevance_labels)

        # Calculate correlation based on the specified method
        if len(set(relevance_labels)) == 1:
            print('Correlation is NaN. All relevance_scores have the same value')
            correlation = 0
        elif len(set(transformed_scores)) == 1:
            print('Correlation is NaN. All transformed_scores have the same value')
            correlation = 0
        elif evaluation_method == 'pearson':
            correlation = transformed_series.corr(relevance_series, method='pearson')
        elif evaluation_method == 'kendall':
            correlation = transformed_series.corr(relevance_series, method='kendall')
        elif evaluation_method == 'spearman':
            correlation = transformed_series.corr(relevance_series, method='spearman')
        elif evaluation_method == 'pearson-greedy':
            correlation = transformed_series.corr(relevance_series, method='pearson-greedy')
        elif evaluation_method == 'kendall-greedy':
            correlation = transformed_series.corr(relevance_series, method='kendall-greedy')
        elif evaluation_method == 'spearman-greedy':
            correlation = transformed_series.corr(relevance_series, method='spearman-greedy')

        return correlation


if __name__ == '__main__':

    start_time = time.time()

    combinations = []
    for aggregation_method in AGGREGATION_METHODS:
        for transformation_method in TRANSFORMATION_METHODS:
            for evaluation_method in EVALUATION_METHODS:
                combinations += [(aggregation_method, transformation_method, evaluation_method)]

    # Determine the range of combinations for this job
    total_combinations = len(combinations)
    combinations_per_job = (total_combinations + NUM_JOBS - 1) // NUM_JOBS
    start_index = (JOB_ID - 1) * combinations_per_job
    end_index = min(start_index + combinations_per_job, total_combinations)

    COMBINATIONS = combinations[start_index:end_index]

    correlation_scores = []
    for combination in COMBINATIONS:
        aggregation_method, transformation_method, evaluation_method = combination
        print(f"Job {JOB_ID} processing {aggregation_method}, {transformation_method}, {evaluation_method}")

        qid_docno_aggregated_scores = get_qid_docno_aggregated_scores(
            qid_docno_passages_scores_cache, aggregation_method)

        qid_docno_transformed_scores = get_qid_docno_transformed_scores(
            qid_docno_aggregated_scores, transformation_method)

        for metric in METRICS:
            # Iterate over all unique QIDs
            all_qids = set(entry['qid'] for entry in qid_docno_transformed_scores)
            query_correlations = {}

            for qid in all_qids:
                correlation = get_evaluated_score(qid_docno_transformed_scores,
                                                  qid, metric, evaluation_method)
                query_correlations[qid] = correlation

            # Save correlation scores for the current settings for each query
            if query_correlations:
                correlation_scores.append({'aggregation_method': aggregation_method,
                                           'transformation_method': transformation_method,
                                           'evaluation_method': evaluation_method,
                                           'metric': metric,
                                           'correlation_per_query': query_correlations})

    # Save the correlation scores to an indexed file
    rank_correlation_job_path = os.path.join(RANK_CORRELATION_SCORE_PQ_AQ_PATH, f'job_{JOB_ID}.jsonl.gz')
    with gzip.open(rank_correlation_job_path, 'wt', encoding='UTF-8') as file:
        for evaluation_entry in correlation_scores:
            file.write(json.dumps(evaluation_entry) + '\n')

    end_time = time.time()
    print(f"Job {JOB_ID} finished rank correlation per query in {(end_time - start_time) / 60} minutes.")
