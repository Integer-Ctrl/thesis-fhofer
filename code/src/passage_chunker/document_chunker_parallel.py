import ir_datasets
from spacy_passage_chunker import SpacyPassageChunker
import json
import os
import gzip
from multiprocessing import Pool
import time
import tracemalloc
from tqdm import tqdm

# Load the configuration settings
pwd = os.path.dirname(os.path.abspath(__file__))


def load_config(filename=pwd + "/../config.json"):
    with open(filename, "r") as f:
        config = json.load(f)
    return config


# Get the configuration settings
config = load_config()
PASSAGE_ID_SEPARATOR = config['PASSAGE_ID_SEPARATOR']

DOCUMENT_DATASET_OLD_NAME = config['DOCUMENT_DATASET_OLD_NAME']
DOCUMENT_DATASET_OLD_NAME_PYTHON_API = config['DOCUMENT_DATASET_OLD_NAME_PYTHON_API']

DOCUMENT_DATASET_NEW_NAME = config['DOCUMENT_DATASET_NEW_NAME']
DOCUMENT_DATASET_NEW_NAME_PYTHON_API = config['DOCUMENT_DATASET_NEW_NAME_PYTHON_API']

OLD_PATH = os.path.join(config['DATA_PATH'], DOCUMENT_DATASET_OLD_NAME)
NEW_PATH = os.path.join(config['DATA_PATH'], DOCUMENT_DATASET_NEW_NAME)

PASSAGE_DATASET_OLD_PATH = os.path.join(OLD_PATH, config['PASSAGE_DATASET_OLD_PATH'])
PASSAGE_DATASET_NEW_PATH = os.path.join(NEW_PATH, config['PASSAGE_DATASET_NEW_PATH'])

TYPE_OLD = config['TYPE_OLD']  # do not chunk if the dataset is already chunked
TYPE_NEW = config['TYPE_NEW']  # do not chunk if the dataset is already chunked


def process_documents(args):
    pid = os.getpid()
    print(f"Processing documents in process with PID: {pid}")
    chunk, batch_size, passage_id_separator = args
    results = []

    # Initialize the passage chunker
    chunker = SpacyPassageChunker()

    batch = []
    BATCH_SIZE = batch_size
    doc_count = 0
    known_doc_ids = set()

    for doc in chunk:
        # Skip documents that have already been processed
        if doc.doc_id in known_doc_ids:
            continue
        known_doc_ids.add(doc.doc_id)

        # Format the document
        formatted_doc = {
            'docno': doc.doc_id,
            'contents': doc.default_text()[:2000000]  # spacy character limit
        }

        # Add the document to the current batch
        batch.append(formatted_doc)

        # If the batch reaches the specified batch size, process and save it
        if len(batch) >= BATCH_SIZE:
            # Chunk the batch of documents
            chunked_batch = chunker.process_batch(batch)

            for chunked_doc in chunked_batch:
                # Write each passage as a JSON object
                for passage in chunked_doc['contents']:
                    passage_id = chunked_doc['docno'] + passage_id_separator + str(passage['id'])
                    results.append({"docno": passage_id, "text": passage['body']})

            # Reset the batch after saving
            doc_count += len(batch)
            if doc_count % 100000 == 0:
                print(f"Processed so far {doc_count} documents in process with PID: {pid}")
            batch = []

    # Process and save any remaining documents in the batch
    if batch:
        chunked_batch = chunker.process_batch(batch)

        for chunked_doc in chunked_batch:
            # Write each passage as a JSON object
            for passage in chunked_doc['contents']:
                passage_id = chunked_doc['docno'] + passage_id_separator + str(passage['id'])
                results.append({"docno": passage_id, "text": passage['body']})

        doc_count += len(batch)

    print(f"Processed {doc_count} documents in process with PID: {pid}")
    return results


def parallel_process_documents(dataset, num_workers, batch_size=1000):

    # Only chunk judged documents
    judged_doc_ids = set()
    for qrel in dataset.qrels_iter():
        judged_doc_ids.add(qrel.doc_id)

    total_chunked_docs = 0
    total_judged_docs = len(judged_doc_ids)
    chunk_size = total_judged_docs // num_workers

    # Divide dataset into slices for each worker
    known_doc_ids = set()
    slices = []

    chunk = list()
    chunk_docs_count = 0

    for document in dataset.docs_iter():
        # Skip documents that are not judged
        if document.doc_id not in judged_doc_ids:
            continue

        # Skip documents that have already been processed
        if document.doc_id in known_doc_ids:
            continue
        known_doc_ids.add(document.doc_id)

        chunk.append(document)
        chunk_docs_count += 1

        # Create a chunk for the current worker
        if chunk_docs_count >= chunk_size:
            slices.append(chunk)
            chunk = list()
            total_chunked_docs += chunk_docs_count
            chunk_docs_count = 0

    # Add the remaining documents to the last worker chunk
    if chunk:
        slices[-1].extend(chunk)

    total_chunked_docs += chunk_docs_count
    print(f"Total judged documents: {total_judged_docs}, Total chunked documents: {total_chunked_docs}")

    # Prepare arguments for each process
    process_args = [(chunk, batch_size, PASSAGE_ID_SEPARATOR) for chunk in slices]

    with Pool(num_workers) as pool:
        results = list(tqdm(pool.imap(process_documents, process_args), total=num_workers))

    # Flatten the results
    return [result for sublist in results for result in sublist]


# If ir_dataset already chunked as passages, save to file
def save_passages_local(dataset, num_workers, batch_size=1000):

    with gzip.open(PASSAGE_DATASET_NEW_PATH, 'wt', encoding='UTF-8') as file:
        for doc in dataset.docs_iter():
            file.write(json.dumps({"docno": doc.doc_id, "text": doc.default_text()}) + '\n')


BATCH_SIZE = 1000
NUM_WORKERS = 16

time_start = time.time()

if TYPE_OLD == 'document':
    print(f"Chunking {DOCUMENT_DATASET_OLD_NAME} dataset")
    dataset = ir_datasets.load(DOCUMENT_DATASET_OLD_NAME_PYTHON_API)
    results = parallel_process_documents(dataset, NUM_WORKERS, batch_size=BATCH_SIZE)

    with gzip.open(PASSAGE_DATASET_OLD_PATH, 'wt', encoding='UTF-8') as file:
        for result in results:
            file.write((json.dumps(result) + '\n'))

if TYPE_OLD == 'passage':
    print(f"Dataset {DOCUMENT_DATASET_OLD_NAME} is already chunked and now saved")
    dataset = ir_datasets.load(DOCUMENT_DATASET_OLD_NAME_PYTHON_API)
    save_passages_local(dataset, NUM_WORKERS, BATCH_SIZE)

# Chunk new dataset and save to file
if DOCUMENT_DATASET_OLD_NAME not in DOCUMENT_DATASET_NEW_NAME and TYPE_NEW == 'document':
    print(f"Chunking {DOCUMENT_DATASET_NEW_NAME} dataset")
    dataset = ir_datasets.load(DOCUMENT_DATASET_NEW_NAME_PYTHON_API)
    results = parallel_process_documents(dataset, NUM_WORKERS, batch_size=BATCH_SIZE)

    with gzip.open(PASSAGE_DATASET_NEW_PATH, 'wt', encoding='UTF-8') as file:
        for result in results:
            file.write((json.dumps(result) + '\n'))

if TYPE_NEW == 'passage':
    print(f"Dataset {DOCUMENT_DATASET_NEW_NAME} is already chunked and now saved")
    dataset = ir_datasets.load(DOCUMENT_DATASET_NEW_NAME_PYTHON_API)
    save_passages_local(dataset, NUM_WORKERS, BATCH_SIZE)

time_end = time.time()
print(f"Processed and saved documents in {(time_end - time_start) / 60} minutes")
