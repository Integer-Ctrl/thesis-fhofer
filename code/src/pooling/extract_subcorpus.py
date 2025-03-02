#!/usr/bin/env python3
import copy
import gzip
import json
import os
from base64 import b64encode
from pathlib import Path
from typing import Any, Iterable, Optional


def write_lines_to_file(self, lines: Iterable[str], path: Path) -> None:
    if path.exists():
        raise RuntimeError(f"File already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)

    if os.path.abspath(path).endswith(".gz"):
        with gzip.open(os.path.abspath(path), "wb") as file:
            for line in lines:
                if not line:
                    continue
                file.write((line + "\n").encode("utf-8"))
    else:
        with path.open("wt") as file:
            file.writelines("%s\n" % line for line in lines if line)

def yield_docs(self, dataset, include_original, skip_duplicate_ids, allowlist_path_ids):
    already_covered_ids = set()
    allowed_ids = set()
    if allowlist_path_ids:
        with open(allowlist_path_ids, "r") as inp_file:
            for i in inp_file:
                allowed_ids.add(i.strip())
        print("I use a allow list of size ", len(allowed_ids))

    docs_store = dataset.docs_store()

    for doc_id in tqdm(allowed_ids, "Load Documents"):
        doc = docs_store.get(doc_id)
        if skip_duplicate_ids and doc.doc_id in already_covered_ids:
            continue
        if allowlist_path_ids and str(doc.doc_id) not in allowed_ids:
            continue

        yield self.map_doc(doc, include_original)
        if skip_duplicate_ids:
            already_covered_ids.add(doc.doc_id)

def map_doc(self, doc: tuple, include_original=True) -> str:
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
    print('foo')
