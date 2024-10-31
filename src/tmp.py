import json

import os
import pyterrier as pt

import ir_datasets


def read_json():
    with open('data/chunked-docs.json') as f:
        data = json.load(f)

    for doc in data['documents']:
        if doc['docno'] == 'S2db48a61-A430a7cb1':
            print('ID:', doc['docno'])
            print(doc['contents'], '\n')


def docs_pyterrier():
    dataset = pt.get_dataset('irds:argsme/2020-04-01')
    for doc in dataset.get_corpus_iter():
        print(doc.keys(), '\n')
        print(doc['docno'], '\n')
        print(doc['conclusion'], '\n')
        print(doc['topic'], '\n')
        print(doc['premises'], '\n')
        break


def docs_ir_dataset():
    dataset = ir_datasets.load('argsme/2020-04-01')
    for doc in dataset.docs_iter()[:1]:
        print(doc, '\n')
        break


def ir_read_qrels():
    dataset = ir_datasets.load("argsme/2020-04-01/touche-2020-task-1")
    for qrel in dataset.qrels_iter():
        print(qrel, '\n')


def pt_read_qrels():
    dataset = pt.get_dataset("irds:argsme/2020-04-01/touche-2020-task-1")
    count = 0
    for index, row in dataset.get_qrels().iterrows():
        count += 1
    print(count)


# docs_pyterrier()
# docs_ir_dataset()
# read_json()
# ir_read_qrels()
pt_read_qrels()
