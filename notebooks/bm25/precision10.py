import json
import os
import pyterrier as pt
import re

INDEX_PATH = './indices/argsme_2020-04-01'
PASSAGE_PATH = '../data/chunked-docs-small.json'

if not pt.java.started():
    pt.java.init()

dataset = pt.get_dataset('irds:argsme/2020-04-01')
# Index argsme/2020-04-01
if not os.path.exists(INDEX_PATH + '/data.properties'):
    indexer = pt.IterDictIndexer(INDEX_PATH)
    index_ref = indexer.index(dataset.get_corpus_iter(),
                              fields=('conclusion', 'source_id', 'source_url', 'source_title', 'topic'),
                              meta=('docno',))
else:
    index_ref = pt.IndexRef.of(INDEX_PATH + '/data.properties')

index = pt.IndexFactory.of(index_ref)

print('\n\n')

# Read the chunked documents
with open(PASSAGE_PATH) as f:
    data = json.load(f)

# BM25 retrieval
bm25 = pt.terrier.Retriever(index, wmodel='BM25')

for doc in data['documents'][:10]:
    for content in doc['contents']:
        sub = re.sub(r'[^a-zA-Z0-9\s.,]', '', content['body'])
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
