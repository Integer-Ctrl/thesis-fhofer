import json
import os
import pyterrier as pt
import pandas as pd
from tqdm import tqdm
import gzip

INDEX_PATH = '../data/argsme/document-dataset/indices/'
PASSAGE_PATH = '../data/argsme/passage-dataset/passages.jsonl.gz'
PASSAGE_SCORES_PATH = '../data/argsme/passage-dataset/experiment-scores'

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

# retriever
bm25 = pt.terrier.Retriever(index, wmodel='BM25')
tfidf = pt.terrier.Retriever(index, wmodel='TF_IDF')

# Read passages - chunked documents
passages = []
with gzip.open(PASSAGE_PATH, 'rt', encoding='UTF-8') as file:
    for line in file:
        line = json.loads(line)
        line['text'] = pt_tokenize(line['text'])
        passages.append(line)

passages = pd.DataFrame(passages)
passages = passages.rename(columns={'docno': 'qid', 'text': 'query'})  # rename columns to match Experiment API
print(passages.keys())
print(dataset.get_qrels().keys())

pt.Experiment(
    [bm25, tfidf],
    passages,
    dataset.get_qrels(),
    eval_metrics=['P_10', 'ndcg_cut_10'],
    names=['bm25', 'tfidf'],
    filter_by_qrels=False,
    filter_by_topics=False,
    perquery=True,
    save_dir=PASSAGE_SCORES_PATH,
    save_mode='overwrite',
    verbose=True
)
