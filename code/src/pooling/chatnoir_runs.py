#!/usr/bin/env python3
import pandas as pd
from chatnoir_pyterrier import ChatNoirRetrieve
import pyterrier as pt
from pathlib import Path
import json

CORPUS_DIRECTORY = '/mnt/ceph/storage/data-in-progress/data-teaching/theses/thesis-fhofer/data/clueweb22-transfer'
topics = pd.read_json(f'{CORPUS_DIRECTORY}/queries.jsonl', lines=True, dtype={"qid": str, "query": str})


for r in ['bm25', 'default']:
    if not Path(f'chatnoir-{r}.run.txt.gz').exists():
        if not pt.started():
            pt.init()

        print('Prepare Retrieval')
        chatnoir = ChatNoirRetrieve(index="clueweb22/b", search_method=r, num_results=1000, features=[])

        print('do retrieval')
        run = chatnoir(topics)

        print('persist results')
        pt.io.write_results(run, 'chatnoir-bm25.run.txt.gz')

with open('/mnt/ceph/storage/data-in-progress/data-teaching/theses/thesis-fhofer/data/maik_cw22_doc_ids.jsonl', 'w') as f:
    result = {}
    for r in ['bm25', 'default']:
        res = pt.io.read_results(f'chatnoir-{r}.run.txt.gz')
        for _, i in res.iterrows():
            if i['qid'] not in result:
                result[i['qid']] = {'doc_ids': []}
            result[i['qid']]['doc_ids'].append(i['docno'])

    for k, v in result.items():
        f.write(json.dumps(v) + '\n')
