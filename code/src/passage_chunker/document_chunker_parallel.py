import ir_datasets
from spacy_passage_chunker import SpacyPassageChunker
import json
import os
import gzip
from multiprocessing import Pool
import time
import tracemalloc
from tqdm import tqdm

tracemalloc.start()
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


def parallel_process(dataset, num_workers, batch_size=1000):
    total_docs = dataset.docs_count()  # Assume the dataset has a length attribute or equivalent.
    chunk_size = total_docs // num_workers

    # Divide dataset into slices for each worker
    known_doc_ids = set()
    slices = []
    for i in range(num_workers):
        if i < num_workers - 1:
            chunk = list(dataset.docs_iter()[i * chunk_size:(i + 1) * chunk_size])
        else:
            chunk = list(dataset.docs_iter()[i * chunk_size:])

        # Remove duplicates within the chunk
        unique_chunk = []
        for doc in chunk:
            if doc.doc_id not in known_doc_ids:
                known_doc_ids.add(doc.doc_id)
                unique_chunk.append(doc)

        slices.append(unique_chunk)

    # Prepare arguments for each process
    process_args = [(chunk, batch_size, PASSAGE_ID_SEPARATOR) for chunk in slices]

    with Pool(num_workers) as pool:
        results = list(tqdm(pool.imap(process_documents, process_args), total=num_workers))

    # Flatten the results
    return [result for sublist in results for result in sublist]


BATCH_SIZE = 1000
NUM_WORKERS = 16

# Chunk old dataset and save to file
time_start = time.time()
print(f"Current memory usage: {tracemalloc.get_traced_memory()[0] / 1024 ** 3:.2f} GB, \
      Peak memory usage: {tracemalloc.get_traced_memory()[1] / 1024 ** 3:.2f} GB")

if TYPE_OLD == 'document':
    dataset = ir_datasets.load(DOCUMENT_DATASET_OLD_NAME_PYTHON_API)
    results = parallel_process(dataset, NUM_WORKERS, batch_size=BATCH_SIZE)

    with gzip.open(PASSAGE_DATASET_OLD_PATH, 'wt', encoding='UTF-8') as file:
        for result in results:
            file.write((json.dumps(result) + '\n'))
    print(f"Current memory usage: {tracemalloc.get_traced_memory()[0] / 1024 ** 3:.2f} GB, \
          Peak memory usage: {tracemalloc.get_traced_memory()[1] / 1024 ** 3:.2f} GB")

# Chunk new dataset and save to file
if DOCUMENT_DATASET_OLD_NAME not in DOCUMENT_DATASET_NEW_NAME and TYPE_NEW == 'document':
    dataset = ir_datasets.load(DOCUMENT_DATASET_NEW_NAME_PYTHON_API)
    results = parallel_process(dataset, NUM_WORKERS, batch_size=BATCH_SIZE)

    with gzip.open(PASSAGE_DATASET_NEW_PATH, 'wt', encoding='UTF-8') as file:
        for result in results:
            file.write((json.dumps(result) + '\n'))
    print(f"Current memory usage: {tracemalloc.get_traced_memory()[0] / 1024 ** 3:.2f} GB, \
          Peak memory usage: {tracemalloc.get_traced_memory()[1] / 1024 ** 3:.2f} GB")

time_end = time.time()
print(f"Processed and saved documents in {(time_end - time_start) / 60} minutes")
print(f"Current memory usage: {tracemalloc.get_traced_memory()[0] / 1024 ** 3:.2f} GB, \
      Peak memory usage: {tracemalloc.get_traced_memory()[1] / 1024 ** 3:.2f} GB")
tracemalloc.stop()
