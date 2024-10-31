import json
import os
import pyterrier as pt
import re
from tqdm import tqdm

INDEX_PATH = '../data/argsme/document-dataset/indices/'
PASSAGE_PATH = '../data/argsme/passage-dataset/passages.jsonl.gz'

if not pt.java.started():
    pt.java.init()


def yield_docs(dataset):
    known_docnos = set()
    for i in tqdm(dataset.irds_ref().docs_iter()):
        if i.doc_id not in known_docnos:
            known_docnos.add(i.doc_id)
            yield {'docno': i.doc_id, 'text': i.default_text()}


dataset = pt.get_dataset('irds:argsme/2020-04-01/touche-2020-task-1')
# Index argsme/2020-04-01
if not os.path.exists(INDEX_PATH + '/data.properties'):
    indexer = pt.IterDictIndexer(INDEX_PATH)
    index_ref = indexer.index(yield_docs(dataset),
                              meta={'docno': 50, 'text': 20000})
else:
    index_ref = pt.IndexRef.of(INDEX_PATH + '/data.properties')

index = pt.IndexFactory.of(index_ref)


exit()

print('\n\n')

# Read the chunked documents
with open(PASSAGE_PATH) as f:
    data = json.load(f)

# BM25 retrieval
bm25 = pt.terrier.Retriever(index, wmodel='BM25')
# passages
# --> save to new file
# iter over qrels exclude all not relevant docs
# for relevant doc get all passages from json
# use passage as query against dataset - tokenize query before submitting to bm25
# ranking with bm25 retriev
# see dc chat - 1. qrels 2. run (result of bm25, see comment before) 3. eval (precision, ndcg, ...)
# --> save to new file
# use score for autoqrels to infer new rel for passages of "new" dataset (json)
# --> save to new file
for doc in data['documents'][:10]:
    for content in doc['contents']:
        sub = re.sub(r'[^a-zA-Z0-9\s.,]', '', content['body'])  # wrong see above
        res = bm25.search(sub).head(11)

        for _, row in res.iterrows():
            if doc['docno'].split('-')[0] in row['docno']:
                print(f"Match found: {row['docno']}")
            print(doc['docno'], row['docno'], '\n')


# print("\n\n")
# print(index.getDocumentIndex().getDocumentEntry(0), '\n')
# print(index.getDocumentIndex().getDocumentLength(0), '\n')
# print(index.getDocumentIndex().getNumberOfDocuments(), '\n')

# print(index.getMetaIndex().getKeys(), '\n')
# print(index.getLexicon()["chemic"].getDocumentFrequency(), '\n')

# bm25 = pt.terrier.Retriever(index, wmodel='BM25')
# print(bm25.transform("Actually again lol NO."))
