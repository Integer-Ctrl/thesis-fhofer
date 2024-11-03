import json
import os
import pyterrier as pt
from tqdm import tqdm
import gzip
from trectools import TrecQrel, TrecRun, TrecEval

INDEX_PATH = '../data/argsme/document-dataset/indices/'
PASSAGE_PATH = '../data/argsme/passage-dataset/passages.jsonl.gz'
PASSAGE_SCORES_PATH = '../data/argsme/passage-dataset/bm25-scores.jsonl.gz'

if not pt.java.started():
    pt.java.init()

tokeniser = pt.java.autoclass('org.terrier.indexing.tokenisation.Tokeniser').getTokeniser()


def yield_docs(dataset):
    known_docnos = set()
    for i in tqdm(dataset.irds_ref().docs_iter()):
        if i.doc_id not in known_docnos:
            known_docnos.add(i.doc_id)
            yield {'docno': i.doc_id, 'text': i.default_text()}


def pt_tokenize(text):
    return ' '.join(tokeniser.getTokens(text))


dataset = pt.get_dataset('irds:argsme/2020-04-01/touche-2020-task-1')
# Index argsme/2020-04-01
if not os.path.exists(INDEX_PATH + '/data.properties'):
    indexer = pt.IterDictIndexer(INDEX_PATH)
    index_ref = indexer.index(yield_docs(dataset),
                              meta={'docno': 50, 'text': 20000})
else:
    index_ref = pt.IndexRef.of(INDEX_PATH + '/data.properties')

index = pt.IndexFactory.of(index_ref)

# BM25 retrieval
bm25 = pt.terrier.Retriever(index, wmodel='BM25')

# Read passages - chunked documents
passages = []
with gzip.open(PASSAGE_PATH, 'rt', encoding='UTF-8') as file:
    for line in file:
        passages.append(json.loads(line))

# Read qrels
qrels = dataset.get_qrels()
qrels_cache = {}

# Write passage scores to file
with gzip.open(PASSAGE_SCORES_PATH, 'at', encoding='UTF-8') as f_out:
    for index, row in tqdm(qrels.iterrows(), desc='Scoring and saving passages', unit='qrel'):
        if row['label'] > 0:  # only relevant docs

            # Get all relevant documents for query
            # Check if the query ID is already cached
            if row['qid'] not in qrels_cache:
                # Cache the relevant entries for the query ID
                qrels_cache[row['qid']] = qrels.loc[
                    (qrels['qid'] == row['qid']) & (qrels['label'] > 0)
                    ].rename(columns={'qid': 'query', 'docno': 'docid', 'label': 'rel'})

            # Access the cached results
            qrels_for_query = TrecQrel()
            qrels_for_query.qrels_data = qrels_cache[row['qid']]
            qrels_for_query.qrels_data['query'] = 0

            # Get passages for relevant doc
            for passage in passages:
                if passage['docno'].startswith(row['docno']):
                    run = TrecRun()
                    run.run_data = bm25.search(pt_tokenize(passage['text'])).loc[
                        :, ['qid', 'docno', 'rank', 'score']].rename(
                        columns={'qid': 'query', 'docno': 'docid', 'score': 'score'}).head(10)
                    run.run_data['query'] = 0  # dummy value to fix merge of run and qrels
                    run.run_data['q0'] = 'Q0'  # dummy value to get ndcg score
                    run.run_data['system'] = 'bm25'  # dummy value to get ndcg score
                    te = TrecEval(run, qrels_for_query)
                    p10_score = te.get_precision(depth=10, removeUnjudged=True)
                    ndcg10_score = te.get_ndcg(depth=10, removeUnjudged=True)
                    f_out.write((json.dumps({'qid': row['qid'],
                                             'docno': passage['docno'],
                                             'p10': p10_score,
                                             'ndcg10': ndcg10_score}) + '\n'))
