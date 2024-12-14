import pandas as pd
import gzip
from tqdm import tqdm
import json
import numpy as np
import pyterrier as pt
import os
import copy
from greedy_series import GreedySeries
import time


# Load the configuration settings
pwd = os.path.dirname(os.path.abspath(__file__))


def load_config(filename=pwd + "/../config.json"):
    with open(filename, "r") as f:
        config = json.load(f)
    return config


# Get the configuration settings
config = load_config()

ALL_QRELS = config['ALL_QRELS']
DOCUMENT_DATASET_OLD_NAME = config['DOCUMENT_DATASET_OLD_NAME']
DOCUMENT_DATASET_OLD_NAME_PYTERRIER = config['DOCUMENT_DATASET_OLD_NAME_PYTERRIER']
NUMBER_OF_CROSS_VALIDATION_FOLDS = config['NUMBER_OF_CROSS_VALIDATION_FOLDS']

OLD_PATH = os.path.join(config['DATA_PATH'], DOCUMENT_DATASET_OLD_NAME)

PASSAGE_DATASET_SCORE_PATH = os.path.join(OLD_PATH, config['PASSAGE_DATASET_OLD_SCORE_AQ_PATH'])
# PASSAGE_DATASET_SCORE_PATH = os.path.join(OLD_PATH, 'retrieval-scores-aq-test.jsonl.gz')
PASSAGES_TO_DOCUMENT_CORRELATION_SCORE_PATH = os.path.join(
    OLD_PATH, config['PASSAGES_TO_DOCUMENT_CORRELATION_SCORE_PQ_AQ_PATH'])

PASSAGE_ID_SEPARATOR = config['PASSAGE_ID_SEPARATOR']

AGGREGATION_METHODS = config['AGGREGATION_METHODS']
TRANSFORMATION_METHODS = config['TRANSFORMATION_METHODS']
EVALUATION_METHODS = config['EVALUATION_METHODS']

METRICS = []
for metric in config['METRICS']:
    for retriever in config['PT_RETRIEVERS']:
        METRICS.append(metric + '_' + retriever)

# Read qrels and cache relevant qrels
dataset = pt.get_dataset(DOCUMENT_DATASET_OLD_NAME_PYTERRIER)
qrels = dataset.get_qrels(variant='relevance')
qrels_cache = {}
for index, row in tqdm(qrels.iterrows(), desc='Caching qrels', unit='qrel'):
    if row['qid'] not in qrels_cache:
        qrels_cache[row['qid']] = qrels.loc[
            (qrels['qid'] == row['qid'])
        ]


# Read passsage scores and cache them
docno_qid_passages_scores_cache = {}
with gzip.open(PASSAGE_DATASET_SCORE_PATH, 'rt', encoding='UTF-8') as file:
    for line in tqdm(file, desc='Caching passage scores', unit='passage'):
        line = json.loads(line)
        docno, passageno = line['docno'].split(PASSAGE_ID_SEPARATOR)
        qid = line['qid']
        if docno not in docno_qid_passages_scores_cache:
            docno_qid_passages_scores_cache[docno] = {}
        if qid not in docno_qid_passages_scores_cache[docno]:
            docno_qid_passages_scores_cache[docno][qid] = []
        docno_qid_passages_scores_cache[docno][qid] += [line]


# Function to get dictonary of aggregated score for a document using passage scores
# Return a list of dictionaries with aggregated scores for each document {docno, qid, metric: score}
def get_docno_qid_aggregated_scores(docno_qid_passages_scores_cache, aggregation_method='mean'):
    # All metrics that are available in the passage scores

    aggregated_scores = []

    for docno, qid_passages_scores in docno_qid_passages_scores_cache.items():
        for qid, passages_scores in qid_passages_scores.items():

            aggregated_doc_scores = {'docno': docno, 'qid': qid}
            for metric in METRICS:
                scores = [passage[metric] for passage in passages_scores]

                if aggregation_method == 'mean':
                    aggregated_doc_scores[metric] = float(np.mean(scores))
                elif aggregation_method == 'max':
                    aggregated_doc_scores[metric] = float(np.max(scores))
                elif aggregation_method == 'min':
                    aggregated_doc_scores[metric] = float(np.min(scores))

            aggregated_scores.append(aggregated_doc_scores)

    return aggregated_scores


# Function to get transformed scores
def get_docno_qid_transformed_scores(docno_qid_aggregated_scores, transformation_method='id', bins=[0.3, 0.7]):

    docno_qid_aggregated_scores_transformed = copy.deepcopy(docno_qid_aggregated_scores)
    for entry in docno_qid_aggregated_scores_transformed:
        for metric in METRICS:
            if transformation_method == 'id':
                pass
            elif transformation_method == 'log' and entry[metric] != 0:
                entry[metric] = float(np.log(entry[metric]))
            elif transformation_method == 'binned':
                entry[metric] = float(np.digitize(entry[metric], bins))

    return docno_qid_aggregated_scores_transformed


# Function to get evaluated score based on the specified metric and evaluation method (pearson, spearman, kendall)
def get_evaluated_score(docno_qid_transformed_scores, qrels_cache, qid,
                        metric='ndcg10_bm25', evaluation_method='pearson'):

    # Lists to store the matched scores for correlation calculation
    transformed_scores = []
    relevance_labels = []

    # Filter scores for the given query ID (qid)
    filtered_scores = [entry for entry in docno_qid_transformed_scores if entry['qid'] == qid]

    # Check if the qid is in qrels_cache
    if qid in qrels_cache:
        qrels_doc = qrels_cache[qid]

        # Match scores with relevance labels
        for entry in filtered_scores:
            docno = entry['docno']
            # Find the matching row in qrels for this docno
            qrels_match = qrels_doc[qrels_doc['docno'] == docno]

            # If there is a match, append scores to lists
            if not qrels_match.empty:
                relevance_score = qrels_match['label'].values[0]
                transformed_scores.append(entry[metric])
                relevance_labels.append(float(relevance_score))
    else:
        print('QID not in qrels_cache:', qid)

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


def check_scores_smaller_zero(scores, location=''):
    for entry in scores:
        for metric in METRICS:
            if entry[metric] < 0:
                print(location, metric, entry[metric])


if __name__ == '__main__':
    start_time = time.time()

    correlation_scores = []
    for aggregation_method in AGGREGATION_METHODS:
        docno_qid_aggregated_scores = get_docno_qid_aggregated_scores(
            docno_qid_passages_scores_cache, aggregation_method)

        for transformation_method in TRANSFORMATION_METHODS:
            docno_qid_transformed_scores = get_docno_qid_transformed_scores(
                docno_qid_aggregated_scores, transformation_method)

            for evaluation_method in EVALUATION_METHODS:
                for metric in METRICS:
                    # Iterate over all unique QIDs
                    all_qids = set(entry['qid'] for entry in docno_qid_transformed_scores)
                    query_correlations = {}

                    for qid in all_qids:
                        correlation = get_evaluated_score(docno_qid_transformed_scores,
                                                          qrels_cache, qid, metric, evaluation_method)
                        query_correlations[qid] = correlation

                    # Save correlation scores for the current settings for each query
                    if query_correlations:
                        correlation_scores.append({'aggregation_method': aggregation_method,
                                                   'transformation_method': transformation_method,
                                                   'evaluation_method': evaluation_method,
                                                   'metric': metric,
                                                   'correlation_per_query': query_correlations})

    with gzip.open(PASSAGES_TO_DOCUMENT_CORRELATION_SCORE_PATH, 'wt', encoding='UTF-8') as file:
        for evaluation_entry in correlation_scores:
            file.write(json.dumps(evaluation_entry) + '\n')

    end_time = time.time()
    print('Time taken:', end_time - start_time)
