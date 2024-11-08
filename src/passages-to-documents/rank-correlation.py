import pandas as pd
import gzip
from tqdm import tqdm
import json
import numpy as np
import pyterrier as pt
from scipy.stats import pearsonr, kendalltau, spearmanr

DATASET_NAME = 'irds:argsme/2020-04-01/touche-2021-task-1'  # PyTerrier dataset name
PASSAGE_PATH = '../data/' + DATASET_NAME.replace('irds:', '') + '/passage-dataset/passages.jsonl.gz'
PASSAGE_SCORES_PATH = '../data/' + DATASET_NAME.replace('irds:', '') + '/passage-dataset/passage-scores.jsonl.gz'
PASSAGE_TO_DOCUMENT_SCORES_PATH = '../data/' + \
    DATASET_NAME.replace('irds:', '') + '/document-dataset/passages-to-document/correlation-scores.jsonl.gz'

AGGREGATION_METHODS = ['mean', 'max', 'min']
TRANSFORMATION_METHODS = ['id', 'log', 'binned']
EVALUATION_METHODS = ['pearson_pd', 'pearson_scipy', 'kendall_pd', 'kendall_scipy', 'spearman_pd', 'spearman_scipy']
METRICS = ['p10_bm25', 'p10_bm25_wod', 'p10_tfidf', 'p10_tfidf_wod',
           'ndcg10_bm25', 'ndcg10_bm25_wod', 'ndcg10_tfidf', 'ndcg10_tfidf_wod']

# Read qrels and cache relevant qrels
dataset = pt.get_dataset(DATASET_NAME)
qrels = dataset.get_qrels()
qrels_cache = {}
for index, row in tqdm(qrels.iterrows(), desc='Caching qrels', unit='qrel'):
    # Only relevant qrels
    if row['label'] > 0:
        if row['qid'] not in qrels_cache:
            qrels_cache[row['qid']] = qrels.loc[
                (qrels['qid'] == row['qid']) & (qrels['label'] > 0)  # All relevant entries for the query ID
            ].rename(columns={'qid': 'query', 'docno': 'docid', 'label': 'rel'})  # Rename columns
            qrels_cache[row['qid']]['query'] = 0  # Dummy value to enable merge of run and qrels (TrecEval)


# Read passsage scores and cache them
docno_qid_passages_scores_cache = {}
with gzip.open(PASSAGE_SCORES_PATH, 'rt', encoding='UTF-8') as file:
    for line in tqdm(file, desc='Caching passage scores', unit='passage'):
        line = json.loads(line)
        docno, passageno = line['docno'].split('___')
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
        # if docno == 'S6555987d-Ad4128081':
        #     f = open("aggregated_scores.txt", "w")
        #     f.write(str(aggregated_scores))
        #     f.close()
        #     exit()

    return aggregated_scores


# Function to get transformed scores
def get_docno_qid_transformed_scores(docno_qid_aggregated_scores, transformation_method='id', bins=[0.3, 0.7]):

    for entry in docno_qid_aggregated_scores:
        for metric in METRICS:
            if transformation_method == 'id':
                pass
            elif transformation_method == 'log':
                if entry[metric] == 0:
                    entry[metric] = 0
                else:
                    entry[metric] = float(np.log(entry[metric]))
            elif transformation_method == 'binned':
                entry[metric] = float(np.digitize(entry[metric], bins))

    return docno_qid_aggregated_scores


# Function to get evaluated score based on the specified metric and evaluation method (pearson, spearman, kendall)
def get_evaluated_score(docno_qid_transformed_scores, qrels_cache,
                        metric='ndcg10_bm25', evaluation_method='pearson_pd'):
    # Lists to store the matched scores for correlation calculation
    transformed_scores = []
    relevance_scores = []

    # Iterate over transformed scores and find matching qrels scores
    for entry in docno_qid_transformed_scores:
        docno = entry['docno']
        qid = entry['qid']

        # Check if the qid is in qrels_cache
        if qid in qrels_cache:
            qrels_doc = qrels_cache[qid]

            # Find the matching row in qrels for this docno
            qrels_match = qrels_doc[qrels_doc['docid'] == docno]

            # If there is a match, append scores to lists
            if not qrels_match.empty:
                relevance_score = qrels_match['rel'].values[0]
                transformed_scores.append(entry[metric])
                relevance_scores.append(relevance_score)

    # Ensure we have pairs to evaluate correlation
    if len(transformed_scores) > 1:
        # Convert lists to pandas Series
        transformed_series = pd.Series(transformed_scores)
        relevance_series = pd.Series(relevance_scores)

        # Calculate correlation based on the specified method
        if evaluation_method == 'pearson_pd':
            correlation = transformed_series.corr(relevance_series, method='pearson')
        elif evaluation_method == 'kendall_pd':
            correlation = transformed_series.corr(relevance_series, method='kendall')
        elif evaluation_method == 'spearman_pd':
            correlation = transformed_series.corr(relevance_series, method='spearman')
        elif evaluation_method == 'pearson_scipy':
            correlation, _ = pearsonr(transformed_scores, relevance_scores)
        elif evaluation_method == 'kendall_scipy':
            correlation, _ = kendalltau(transformed_scores, relevance_scores)
        elif evaluation_method == 'spearman_scipy':
            correlation, _ = spearmanr(transformed_scores, relevance_scores)

        return correlation


correlation_scores = []
for aggregation_method in AGGREGATION_METHODS:
    docno_qid_aggregated_scores = get_docno_qid_aggregated_scores(
        docno_qid_passages_scores_cache, aggregation_method)

    for transformation_method in TRANSFORMATION_METHODS:
        docno_qid_transformed_scores = get_docno_qid_transformed_scores(
            docno_qid_aggregated_scores, transformation_method)

        for evaluation_method in EVALUATION_METHODS:
            for metric in METRICS:
                # correlation_pandas = get_evaluated_score_pandas(docno_qid_transformed_scores,
                #                                                 qrels_cache, metric, evaluation_method)
                # correlation_scores.append({'aggregation_method': aggregation_method,
                #                            'transformation_method': transformation_method,
                #                            'evaluation_method': evaluation_method,
                #                            'metric': metric,
                #                            'lib': 'pandas',
                #                            'correlation': correlation_pandas})
                correlation = get_evaluated_score(docno_qid_transformed_scores,
                                                  qrels_cache, metric, evaluation_method)
                correlation_scores.append({'aggregation_method': aggregation_method,
                                           'transformation_method': transformation_method,
                                           'evaluation_method': evaluation_method,
                                           'metric': metric,
                                           'correlation': correlation})

correlation_scores = sorted(correlation_scores, key=lambda x: x['correlation'], reverse=True)
with gzip.open(PASSAGE_TO_DOCUMENT_SCORES_PATH, 'wt', encoding='UTF-8') as file:
    for evaluation_entry in correlation_scores:
        file.write(json.dumps(evaluation_entry) + '\n')
