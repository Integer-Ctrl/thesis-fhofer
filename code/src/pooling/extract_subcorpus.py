#!/usr/bin/env python3
import copy
import gzip
import json
import os
from tqdm import tqdm
from base64 import b64encode
from pathlib import Path
from typing import Any, Iterable, Optional


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
    PATHS = [
        'argsme/2020-04-01/touche-2020-task-1/clueweb22/b/candidates-chatnoir/doc_ids.jsonl',
        'disks45/nocr/trec-robust-2004/clueweb22/b/candidates-chatnoir/doc_ids.jsonl',
        'disks45/nocr/trec7/clueweb22/b/candidates-chatnoir/doc_ids.jsonl',
        'disks45/nocr/trec8/clueweb22/b/candidates-chatnoir/doc_ids.jsonl',
        'msmarco-passage/trec-dl-2019/judged/clueweb22/b/candidates-chatnoir/doc_ids.jsonl',
        'msmarco-passage/trec-dl-2020/judged/clueweb22/b/candidates-chatnoir/doc_ids.jsonl'
    ]

    OUTPUT_FILE = f'{DATA_DIR}/clueweb22-transfer/documents.jsonl.gz'
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
