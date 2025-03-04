#!/usr/bin/env python3
from tira.rest_api_client import Client
from tqdm import tqdm
from trectools import TrecPoolMaker
import json

tira = Client(allow_local_execution=True)
DATA_DIR = '/mnt/ceph/storage/data-in-progress/data-teaching/theses/thesis-fhofer/data'

RETRIEVAL_SYSTEMS = [
    'ir-benchmarks/tira-ir-starter/BM25 (tira-ir-starter-pyterrier)',
    'ir-benchmarks/tira-ir-starter/DFIC (tira-ir-starter-pyterrier)',
    'ir-benchmarks/tira-ir-starter/DFIZ (tira-ir-starter-pyterrier)',
    'ir-benchmarks/tira-ir-starter/DirichletLM (tira-ir-starter-pyterrier)',
    'ir-benchmarks/tira-ir-starter/DFRee (tira-ir-starter-pyterrier)',
    'ir-benchmarks/tira-ir-starter/DLH (tira-ir-starter-pyterrier)',
    'ir-benchmarks/tira-ir-starter/DPH (tira-ir-starter-pyterrier)',
    'ir-benchmarks/tira-ir-starter/Hiemstra_LM (tira-ir-starter-pyterrier)',
    'ir-benchmarks/tira-ir-starter/InB2 (tira-ir-starter-pyterrier)',
    'ir-benchmarks/tira-ir-starter/LGD (tira-ir-starter-pyterrier)',
    'ir-benchmarks/tira-ir-starter/Js_KLs (tira-ir-starter-pyterrier)',
    'ir-benchmarks/tira-ir-starter/PL2 (tira-ir-starter-pyterrier)',
]

RE_RANKER_SYSTEMS = [
    'ir-benchmarks/tira-ir-starter/ANCE Base Cosine (tira-ir-starter-beir)',
    'ir-benchmarks/tira-ir-starter/MonoT5 Base (tira-ir-starter-gygaggle)',
    'ir-benchmarks/tira-ir-starter/MonoBERT Base (tira-ir-starter-gygaggle)',
]

runs = []

for DATASET in ['clueweb22-transfer-english-only', 'clueweb22-transfer']:
    CORPUS_DIRECTORY = f'{DATA_DIR}/{DATASET}'
    for system in tqdm(RETRIEVAL_SYSTEMS):
        runs.append(tira.trectools.from_submission(system, CORPUS_DIRECTORY))

    bm25_base = tira.get_run_output(RETRIEVAL_SYSTEMS[0], CORPUS_DIRECTORY)
    for system in tqdm(RE_RANKER_SYSTEMS):
        runs.append(tira.trectools.from_submission(system, CORPUS_DIRECTORY, file_to_re_rank=bm25_base))

pool = TrecPoolMaker().make_pool(runs, strategy='topX', topX=50).pool
pool = {k: list(v) for k, v in pool.items()}

pool_size = []
for k, v in pool.items():
    pool_size.append(len(v))

from statistics import mean
print(mean(pool_size))

with open(f'{DATA_DIR}/clueweb22-transfer/judgment-pool.json', 'w') as f:
    f.write(json.dumps(pool))

