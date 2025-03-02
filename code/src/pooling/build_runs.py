#!/usr/bin/env python3
from tira.rest_api_client import Client
import pandas as pd
from tqdm import tqdm

tira = Client(allow_local_execution=True)
CORPUS_DIRECTORY = 'subsampled-ms-marco-deep-learning-20241201-training'

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
]

topics = pd.read_json(f'{CORPUS_DIRECTORY}/queries.jsonl', lines=True, dtype={"qid": str, "query": str})

#for system in tqdm(RETRIEVAL_SYSTEMS):
#    run = tira.pt.from_submission(system, CORPUS_DIRECTORY)(topics)

for system in tqdm(RE_RANKER_SYSTEMS):
    run = tira.pt.from_submission(system, CORPUS_DIRECTORY)(topics)

