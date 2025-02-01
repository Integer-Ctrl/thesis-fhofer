import re
import gzip
from tqdm import tqdm
import json
import numpy as np
import pyterrier as pt
import os
import copy
from greedy_series import GreedySeries
from glob import glob
import sys
import ray

# Initialize Ray
ray.init()


def load_config(filename='/mnt/ceph/storage/data-tmp/current/ho62zoq/thesis-fhofer/code/src/config.json'):
    with open(filename, "r") as f:
        config = json.load(f)
    return config


# Get the configuration settings
config = load_config()
SOURCE_PATH = os.path.join(config['DATA_PATH'], config['DOCUMENT_DATASET_SOURCE_NAME'])
TARGET_PATH = os.path.join(SOURCE_PATH, config['DOCUMENT_DATASET_TARGET_NAME'])
POINTWISE_PREFERENCES_PATTERN = os.path.join(TARGET_PATH, config['MONOPROMPT_PATH'], '*.jsonl.gz')  # glob pattern
LABEL_RANK_CORRELATION_SCORE_PQ_AQ_PATH = os.path.join(
    TARGET_PATH, config['MONOPROMPT_PATH'], config['LABEL_RANK_CORRELATION_SCORE_PQ_AQ_PATH'])


@ray.remote
def ray_wrapper(JOB_ID, NUM_JOBS):
    def load_config(filename='/mnt/ceph/storage/data-tmp/current/ho62zoq/thesis-fhofer/code/src/config.json'):
        with open(filename, "r") as f:
            config = json.load(f)
        return config

    # Get the configuration settings
    config = load_config()

    SOURCE_PATH = os.path.join(config['DATA_PATH'], config['DOCUMENT_DATASET_SOURCE_NAME'])
    TARGET_PATH = os.path.join(SOURCE_PATH, config['DOCUMENT_DATASET_TARGET_NAME'])

    DOCUMENT_DATASET_TARGET_NAME_PYTERRIER = config['DOCUMENT_DATASET_TARGET_NAME_PYTERRIER']

    POINTWISE_PREFERENCES_PATTERN = os.path.join(TARGET_PATH, config['MONOPROMPT_PATH'], '*.jsonl.gz')  # glob pattern
    # to compute on monoprompt pointwise preferences (not on duoprompt)
    LABEL_RANK_CORRELATION_SCORE_PQ_AQ_PATH = os.path.join(
        TARGET_PATH, config['MONOPROMPT_PATH'], config['LABEL_RANK_CORRELATION_SCORE_PQ_AQ_PATH'])

    PASSAGE_ID_SEPARATOR = config['PASSAGE_ID_SEPARATOR']
    AGGREGATION_METHODS = config['AGGREGATION_METHODS']
    TRANSFORMATION_METHODS = config['TRANSFORMATION_METHODS']
    EVALUATION_METHODS = config['EVALUATION_METHODS']

    # Read qrels and cache relevant qrels
    dataset = pt.get_dataset(DOCUMENT_DATASET_TARGET_NAME_PYTERRIER)
    qrels = dataset.get_qrels(variant='relevance')
    qrels_cache = {}
    for index, row in tqdm(qrels.iterrows(), desc='Caching qrels', unit='qrel'):
        if row['qid'] not in qrels_cache:
            qrels_cache[row['qid']] = qrels.loc[
                (qrels['qid'] == row['qid'])
            ]

    # Read pointwise preference scores {qid: docno: passage_id: scores[]}
    # Which known relevant passages are used for pointwise preferences is not important here

    def read_pointwise_preferences(path):
        qid_docno_passage_scores = {}

        with gzip.open(path, 'rt', encoding='UTF-8') as file:
            for line in file:
                line = json.loads(line)

                qid = line['qid']
                docno, passageno = line['passage_to_judge_id'].split(PASSAGE_ID_SEPARATOR)

                if qid not in qid_docno_passage_scores:
                    qid_docno_passage_scores[qid] = {}
                if docno not in qid_docno_passage_scores[qid]:
                    qid_docno_passage_scores[qid][docno] = {}
                if passageno not in qid_docno_passage_scores[qid][docno]:
                    qid_docno_passage_scores[qid][docno][passageno] = []

                # Known relevant passage vs. known non relevant passage
                qid_docno_passage_scores[qid][docno][passageno] += [line['score']]

        return qid_docno_passage_scores

    # Function to get dictonary of aggregated score for a passage or document
    # Return a list of dictionaries with aggregated scores for each document {docno, qid, metric: score}

    def get_aggregated_scores_passages(qid_docno_passage_scores, aggregation_method):

        # Agregate scores for each passage
        qid_docno_passage_score = {}

        for qid, docno_passage_scores in qid_docno_passage_scores.items():
            for docno, passage_scores in docno_passage_scores.items():
                for passage, scores in passage_scores.items():

                    if qid not in qid_docno_passage_score:
                        qid_docno_passage_score[qid] = {}
                    if docno not in qid_docno_passage_score[qid]:
                        qid_docno_passage_score[qid][docno] = {}

                    if aggregation_method == 'mean':
                        qid_docno_passage_score[qid][docno][passage] = float(np.mean(scores))
                    elif aggregation_method == 'max':
                        qid_docno_passage_score[qid][docno][passage] = float(np.max(scores))
                    elif aggregation_method == 'min':
                        qid_docno_passage_score[qid][docno][passage] = float(np.min(scores))
                    elif aggregation_method == 'sum':
                        qid_docno_passage_score[qid][docno][passage] = float(np.sum(scores))

        return qid_docno_passage_score

    def get_transformed_scores_passages(qid_docno_passage_score, transformation_method, bins=[0.3, 0.7]):

        qid_docno_passage_score_transformed = copy.deepcopy(qid_docno_passage_score)
        for qid, docno_passage_scores in qid_docno_passage_score_transformed.items():
            for docno, passage_scores in docno_passage_scores.items():
                for passage, score in passage_scores.items():
                    if transformation_method == 'id':
                        pass
                    elif transformation_method == 'log' and score > 0:
                        qid_docno_passage_score_transformed[qid][docno][passage] = float(np.log(score))
                    elif transformation_method == 'exp':
                        qid_docno_passage_score_transformed[qid][docno][passage] = float(np.exp(score))
                    elif transformation_method == 'sqrt' and score > 0:
                        qid_docno_passage_score_transformed[qid][docno][passage] = float(np.sqrt(score))
                    # elif transformation_method == 'binned':
                    #     qid_docno_passage_score_transformed[qid][docno][passage] = float(np.digitize(score, bins))

        return qid_docno_passage_score_transformed

    def get_aggregated_scores_documents(qid_docno_passage_score, aggregation_method):

        qid_docno_score = {}

        for qid, docno_passage_scores in qid_docno_passage_score.items():
            for docno, passage_scores in docno_passage_scores.items():

                if qid not in qid_docno_score:
                    qid_docno_score[qid] = {}

                if aggregation_method == 'mean':
                    qid_docno_score[qid][docno] = float(np.mean(list(passage_scores.values())))
                elif aggregation_method == 'max':
                    qid_docno_score[qid][docno] = float(np.max(list(passage_scores.values())))
                elif aggregation_method == 'min':
                    qid_docno_score[qid][docno] = float(np.min(list(passage_scores.values())))
                elif aggregation_method == 'sum':
                    qid_docno_score[qid][docno] = float(np.sum(list(passage_scores.values())))

        return qid_docno_score

    def get_transformed_scores_documents(qid_docno_score, transformation_method, bins=[0.3, 0.7]):

        qid_docno_score_transformed = copy.deepcopy(qid_docno_score)
        for qid, docno_score in qid_docno_score_transformed.items():
            for docno, score in docno_score.items():
                if transformation_method == 'id':
                    pass
                elif transformation_method == 'log' and score > 0:
                    qid_docno_score_transformed[qid][docno] = float(np.log(score))
                elif transformation_method == 'exp':
                    qid_docno_score_transformed[qid][docno] = float(np.exp(score))
                elif transformation_method == 'sqrt' and score > 0:
                    qid_docno_score_transformed[qid][docno] = float(np.sqrt(score))
                # elif transformation_method == 'binned':
                #     qid_docno_score_transformed[qid][docno] = float(np.digitize(score, bins))

        return qid_docno_score_transformed

    # Function to get evaluated score based on the specified metric and evaluation method (pearson, spearman, kendall)

    def get_evaluated_score(qid_docno_score, qrels_cache, evaluation_method='pearson'):

        correlations_per_query = {}

        for qid in qrels_cache.keys():

            # Lists to store the matched scores for correlation calculation
            transformed_scores = []
            relevance_labels = []

            # Add the scores and relevance labels to the lists in the correct order
            for index, qrel in qrels_cache[qid].iterrows():

                docno = qrel['docno']
                if docno not in qid_docno_score[qid]:
                    # Only evaluate on judged documents
                    continue

                transformed_scores.append(qid_docno_score[qid][docno])
                relevance_labels.append(qrel['label'])

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

                correlations_per_query[qid] = correlation

        return correlations_per_query

    ############################
    #           MAIN           #
    ############################
    combinations = []
    for aggregation_method_passage in AGGREGATION_METHODS:
        for transformation_method_passage in TRANSFORMATION_METHODS:
            for aggregation_method_document in AGGREGATION_METHODS:
                for transformation_method_document in TRANSFORMATION_METHODS:
                    for evaluation_method in EVALUATION_METHODS:
                        combinations += [(aggregation_method_passage,
                                         transformation_method_passage,
                                         aggregation_method_document,
                                         transformation_method_document,
                                         evaluation_method)]

    # Determine the range of combinations for this job
    total_combinations = len(combinations)
    combinations_per_job = (total_combinations + NUM_JOBS - 1) // NUM_JOBS
    start_index = (JOB_ID - 1) * combinations_per_job
    end_index = min(start_index + combinations_per_job, total_combinations)

    COMBINATIONS = combinations[start_index:end_index]

    for pointwise_preferences_path in glob(POINTWISE_PREFERENCES_PATTERN):
        # Extract the file name
        file_name = os.path.basename(pointwise_preferences_path)
        base_name = file_name.split('.')[0]
        write_path = os.path.join(LABEL_RANK_CORRELATION_SCORE_PQ_AQ_PATH, base_name)

        print(f"Processing candidates: {file_name}")
        qid_docno_passage_scores = read_pointwise_preferences(pointwise_preferences_path)

        correlation_scores = []
        for combination in COMBINATIONS:
            agr_met_passage, tra_met_passage, agr_met_doc, tra_met_doc, eval_met = combination
            print(f"Job {JOB_ID} processing {agr_met_passage}-{tra_met_passage}-{agr_met_doc}-{tra_met_doc}-{eval_met}")

            # Get the aggregated and transformed scores for passages
            aggregated_scores_passages = get_aggregated_scores_passages(qid_docno_passage_scores, agr_met_passage)
            transformed_scores_passages = get_transformed_scores_passages(aggregated_scores_passages, tra_met_passage)

            # Get the aggregated and transformed scores for documents
            aggregated_scores_documents = get_aggregated_scores_documents(transformed_scores_passages, agr_met_doc)
            transformed_scores_documents = get_transformed_scores_documents(aggregated_scores_documents, tra_met_doc)

            # Get the correlation scores
            correlations_per_query = get_evaluated_score(transformed_scores_documents, qrels_cache, eval_met)
            correlation_scores += [{'aggregation_method_passage': agr_met_passage,
                                    'transformation_method_passage': tra_met_passage,
                                    'aggregation_method_document': agr_met_doc,
                                    'transformation_method_document': tra_met_doc,
                                    'evaluation_method': eval_met,
                                    'correlation_scores': correlations_per_query}]

        # Save the correlation scores to an indexed file
        rank_correlation_job_path = os.path.join(write_path, f'job_{JOB_ID}.jsonl.gz')
        with gzip.open(rank_correlation_job_path, 'wt', encoding='UTF-8') as file:
            for evaluation_entry in correlation_scores:
                file.write(json.dumps(evaluation_entry) + '\n')


if __name__ == '__main__':

    for pointwise_preferences_path in glob(POINTWISE_PREFERENCES_PATTERN):
        # Extract the file name
        file_name = os.path.basename(pointwise_preferences_path)
        base_name = file_name.split('.')[0]
        write_path = os.path.join(LABEL_RANK_CORRELATION_SCORE_PQ_AQ_PATH, base_name)
        if not os.path.exists(write_path):
            os.makedirs(write_path)

    NUM_WORKERS = 50

    futures = []
    for i in range(1, NUM_WORKERS + 1):
        futures.append(ray_wrapper.remote(i, NUM_WORKERS))

    # Wait for all workers to finish
    ray.get(futures)
