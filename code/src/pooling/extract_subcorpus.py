#!/usr/bin/env python3
import copy
import gzip
import json
import os
from tqdm import tqdm
from base64 import b64encode
from pathlib import Path
from typing import Any, Iterable, Optional
import ir_datasets

# export IR_DATASETS_HOME=/mnt/ceph/tira/state/ir_datasets/

SELECTED_QUERIES = {
    'msmarco-passage/trec-dl-2019/judged': ('1037798', '1129237'),
    'msmarco-passage/trec-dl-2020/judged': ('997622', '1051399', '1127540'),
    'argsme/2020-04-01/touche-2020-task-1': ('49', '34'),
    'disks45/nocr/trec-robust-2004': ('681', '448'),
    'disks45/nocr/trec7': ('354', '358'),
    'disks45/nocr/trec8': ('441', '422')
}

def selected_queries():
    ret = {}
    for dataset_id, qids in SELECTED_QUERIES.items():
        dataset = ir_datasets.load(dataset_id)
        for q in dataset.queries_iter():
            if str(q.query_id) in qids:
                assert q.query_id not in ret

                ret[q.query_id] = q.default_text()
    return ret

def write_lines_to_file(lines: Iterable[str], path: Path) -> None:
    if os.path.abspath(path).endswith(".gz"):
        with gzip.open(os.path.abspath(path), "ab") as file:
            for line in lines:
                if not line:
                    continue
                file.write((line + "\n").encode("utf-8"))
                file.flush()
    else:
        with path.open("wt") as file:
            file.writelines("%s\n" % line for line in lines if line)

def yield_docs(dataset, include_original, skip_duplicate_ids, allowed_ids):
    already_covered_ids = set()
    docs_store = dataset.docs_store()

    for doc_id in tqdm(allowed_ids, "Load Documents"):
        doc = docs_store.get(doc_id)
        if skip_duplicate_ids and doc.doc_id in already_covered_ids:
            continue

        yield map_doc(doc, include_original)
        if skip_duplicate_ids:
            already_covered_ids.add(doc.doc_id)

def map_doc(doc: tuple, include_original=True) -> str:
    """Maps a document of any dataset (loaded through ir_datasets) to a standarized format
    stores full document data too, if flag 'include_original' is set

    @param doc: the document as a namedtuple
    @param include_original: flag which signals if the original document data should be stored too
    :return ret: the mapped document
    """
    ret = {"docno": doc.doc_id, "text": doc.default_text()}
    if include_original:
        ret["original_document"] = self.make_serializable(doc._asdict())
    return json.dumps(ret)


if __name__ == '__main__':
    from ir_datasets import load
    from ir_datasets_clueweb22 import register
    register()
    DATA_DIR = '/mnt/ceph/storage/data-in-progress/data-teaching/theses/thesis-fhofer/data'
    CORPUS_DIR = f'{DATA_DIR}/clueweb22-transfer-english-only'

    PATHS = [
        #'argsme/2020-04-01/touche-2020-task-1/clueweb22/b/candidates-chatnoir/doc_ids.jsonl',
        #'disks45/nocr/trec-robust-2004/clueweb22/b/candidates-chatnoir/doc_ids.jsonl',
        #'disks45/nocr/trec7/clueweb22/b/candidates-chatnoir/doc_ids.jsonl',
        #'disks45/nocr/trec8/clueweb22/b/candidates-chatnoir/doc_ids.jsonl',
        #'msmarco-passage/trec-dl-2019/judged/clueweb22/b/candidates-chatnoir/doc_ids.jsonl',
        #'msmarco-passage/trec-dl-2020/judged/clueweb22/b/candidates-chatnoir/doc_ids.jsonl'
        'maik_cw22_doc_ids.jsonl',
    ]

    OUTPUT_FILE = f'{CORPUS_DIR}/documents.jsonl.gz'
    doc_ids = set()
    dataset = load("clueweb22/b")
    already_covered_ids = set()

    with gzip.open(OUTPUT_FILE, 'rt') as f:
        for l in f:
            try:
                l = json.loads(l)
                already_covered_ids.add(l['docno'])
            except:
                pass

    print(f'skip {len(already_covered_ids)}')
    for json_file in PATHS:
        with open(f'{DATA_DIR}/{json_file}', 'r') as f:
            for l in f:
                l = json.loads(l)
                for doc_id in l['doc_ids']:
                    if doc_id not in already_covered_ids:
                        doc_ids.add(doc_id)
    print(len(doc_ids))

    docs = yield_docs(dataset, False, True, doc_ids)
    write_lines_to_file(docs, OUTPUT_FILE)

    with open(f'{CORPUS_DIR}/queries.jsonl', 'w') as f:
        for qid, query in selected_queries().items():
            f.write(json.dumps({"qid": qid, "query": query, "original_query": {}}) + '\n')

    with open(f'{CORPUS_DIR}/queries.xml', 'w') as f:
        f.write("<topics>\n")
        for qid, query in selected_queries().items():
            f.write(f'  <topic number="{qid}">\n    <query>{query}</query>\n  </topic>\n')

        f.write("</topics>")
