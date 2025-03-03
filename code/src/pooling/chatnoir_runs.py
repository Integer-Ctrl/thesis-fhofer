#!/usr/bin/env python3
import pandas as pd
from chatnoir_pyterrier import ChatNoirRetrieve

CORPUS_DIRECTORY = 'clueweb22-transfer'
topics = pd.read_json(f'{CORPUS_DIRECTORY}/queries.jsonl', lines=True, dtype={"qid": str, "query": str})


chatnoir = ChatNoirRetrieve(index="clueweb22/b", search_method="bm25", num_results=1000, features=[])

run = chatnoir(topics)


