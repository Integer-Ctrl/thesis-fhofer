import json

import os
import pyterrier as pt

import ir_datasets


def read_json():
    with open('data/chunked-docs.json') as f:
        data = json.load(f)

    for doc in data['documents']:
        if doc['id'] == 'S16db688b-A11fae2f9' or doc['id'] == 'S16db688b' or doc['id'] == 'A11fae2f9':
            print('ID:', doc['id'])
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
    for doc in dataset.docs_iter()[:7]:
        print(doc, '\n')
        break


# docs_pyterrier()
docs_ir_dataset()
