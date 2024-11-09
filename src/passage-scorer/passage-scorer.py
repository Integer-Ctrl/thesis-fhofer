import json
import os
import pyterrier as pt
from tqdm import tqdm
import gzip
from trectools import TrecQrel, TrecRun, TrecEval

# DATASET_NAME = 'irds:argsme/2020-04-01/touche-2020-task-1'  # PyTerrier dataset name
DATASET_NAME = 'irds:argsme/2020-04-01/touche-2021-task-1'  # PyTerrier dataset name
INDEX_PATH = '../data/' + DATASET_NAME.replace('irds:', '') + '/document-dataset/indices/'
PASSAGE_PATH = '../data/' + DATASET_NAME.replace('irds:', '') + '/passage-dataset/passages.jsonl.gz'
PASSAGE_SCORES_PATH = '../data/' + DATASET_NAME.replace('irds:', '') + '/passage-dataset/passage-scores.jsonl.gz'

# Initialize PyTerrier and Tokenizer
if not pt.java.started():
    pt.java.init()
tokeniser = pt.java.autoclass('org.terrier.indexing.tokenisation.Tokeniser').getTokeniser()


# Tokenize text
def pt_tokenize(text):
    return ' '.join(tokeniser.getTokens(text))


# Document yield function for indexing without duplicates
def yield_docs(dataset):
    known_docnos = set()
    for i in tqdm(dataset.irds_ref().docs_iter()):
        if i.doc_id not in known_docnos:
            known_docnos.add(i.doc_id)
            yield {'docno': i.doc_id, 'text': i.default_text()}


# Index dataset
dataset = pt.get_dataset(DATASET_NAME)
if not os.path.exists(INDEX_PATH + '/data.properties'):
    indexer = pt.IterDictIndexer(INDEX_PATH)
    index_ref = indexer.index(yield_docs(dataset),
                              meta={'docno': 50, 'text': 20000})
else:
    index_ref = pt.IndexRef.of(INDEX_PATH + '/data.properties')

dataset_index = pt.IndexFactory.of(index_ref)

# Read passages and cache them
passages_cache = {}
with gzip.open(PASSAGE_PATH, 'rt', encoding='UTF-8') as file:
    for line in tqdm(file, desc='Caching passages', unit='passage'):
        line = json.loads(line)
        docno, passageno = line['docno'].split('___')
        if docno not in passages_cache:
            passages_cache[docno] = []
        passages_cache[docno] += [line]

# Read qrels and cache relevant qrels
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


# retrieval models
bm25 = pt.terrier.Retriever(dataset_index, wmodel='BM25')
tfidf = pt.terrier.Retriever(dataset_index, wmodel='TF_IDF')


# Get reciprocal rank of the original document in a run
def get_reciprocal_rank_of_docno(run_data, docno):
    for index, row in run_data.iterrows():
        if row['docid'] == docno:
            return 1 / (index + 1)
    return 0


# Infer run from passage text and retrieve top 10 passages
def get_infered_run(retriever, passage_text, system_name, docno):
    run = TrecRun()
    run_wod = TrecRun()

    # Retrieve the top 11 entries
    run.run_data = retriever.search(pt_tokenize(passage_text)).loc[
        :, ['qid', 'docno', 'rank', 'score']].rename(
        columns={'qid': 'query', 'docno': 'docid', 'score': 'score'}).head(11)

    run.run_data['query'] = 0  # Dummy value to enable merge of run and qrels (TrecEval)
    run.run_data['q0'] = 'Q0'  # Dummy value to get ndcg score (TrecEval)
    run.run_data['system'] = system_name  # Dummy value to get ndcg score (TrecEval)

    # Drop the last row to keep top 10, can contain orginal document
    run.run_data = run.run_data.iloc[:-1]

    reciprocal_rank_docno = get_reciprocal_rank_of_docno(run.run_data, docno)

    # If docno is in top 10, remove it; otherwise, remove the last entry
    if docno in run.run_data['docid'].values:
        run_wod.run_data = run.run_data[run.run_data['docid'] != docno]
    else:
        run_wod.run_data = run.run_data

    return run, run_wod, reciprocal_rank_docno


# Get all qrels for a query and remove original document if specified
def get_qrels_for_query(qid, include_original_document):
    qrels_for_query = TrecQrel()
    qrels_for_query.qrels_data = qrels_cache[qid]
    # Remove original document if specified
    if not include_original_document:
        qrels_for_query.qrels_data = qrels_for_query.qrels_data[qrels_for_query.qrels_data['docid'] != qid]
    return qrels_for_query


# Evaluate run using TrecEval
def evaluate_run(run, qrels_for_query):
    te = TrecEval(run, qrels_for_query)
    p10_score = float(te.get_precision(depth=10, removeUnjudged=True))
    ndcg10_score = float(te.get_ndcg(depth=10, removeUnjudged=True))
    return p10_score, ndcg10_score


# Write passage scores to file
with gzip.open(PASSAGE_SCORES_PATH, 'wt', encoding='UTF-8') as file:
    for qid, docnos in tqdm(qrels_cache.items(), desc='Scoring and saving passages', unit='qid'):
        for docno in docnos['docid']:
            for passage in passages_cache[docno]:
                # wod = without original document
                qrels_for_query = get_qrels_for_query(qid, include_original_document=True)
                qrels_for_query_wod = get_qrels_for_query(qid, include_original_document=False)

                run_bm25, run_bm25_wod, reciprocal_rank_docno_bm25 = get_infered_run(
                    bm25, passage['text'], 'bm25', docno)
                run_tfidf, run_tfidf_wod, reciprocal_rank_docno_tfidf = get_infered_run(
                    tfidf, passage['text'], 'tfidf', docno)

                # Evaluate passage scores
                p10_bm25, ndcg10_bm25 = evaluate_run(run_bm25, qrels_for_query)
                p10_bm25_wod, ndcg10_bm25_wod = evaluate_run(run_bm25_wod, qrels_for_query_wod)

                p10_tfidf, ndcg10_tfidf = evaluate_run(run_tfidf, qrels_for_query)
                print(p10_tfidf, ndcg10_tfidf)
                p10_tfidf_wod, ndcg10_tfidf_wod = evaluate_run(run_tfidf_wod, qrels_for_query_wod)

                file.write((json.dumps({'qid': qid,
                                        'docno': passage['docno'],
                                        'p10_bm25': p10_bm25,
                                        'p10_bm25_wod': p10_bm25_wod,
                                        'p10_tfidf': p10_tfidf,
                                        'p10_tfidf_wod': p10_tfidf_wod,
                                        'ndcg10_bm25': ndcg10_bm25,
                                        'ndcg10_bm25_wod': ndcg10_bm25_wod,
                                        'ndcg10_tfidf': ndcg10_tfidf,
                                        'ndcg10_tfidf_wod': ndcg10_tfidf_wod,
                                        'reciprocal_rank_docno_bm25': reciprocal_rank_docno_bm25,
                                        'reciprocal_rank_docno_tfidf': reciprocal_rank_docno_tfidf}) + '\n'))
