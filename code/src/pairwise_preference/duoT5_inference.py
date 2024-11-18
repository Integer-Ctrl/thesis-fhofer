from typing import List
from transformers import T5ForConditionalGeneration, AutoTokenizer
import autoqrels
import more_itertools
import torch
import ir_datasets
import json
import os


# Load the configuration settings
def load_config(filename="../config.json"):
    with open(filename, "r") as f:
        config = json.load(f)
    return config


config = load_config()
logger = ir_datasets.log.easy()

DOCUMENT_DATASET_NAME = config['DOCUMENT_DATASET_NAME']
DATA_PATH = os.path.join(config['DATA_PATH'], DOCUMENT_DATASET_NAME)
DUOT5_QID_DOC_DOC_SCORES_PATH = os.path.join(DATA_PATH, config['DUOT5_QID_DOC_DOC_SCORES_PATH'])

PASSAGE_ID_SEPARATOR = config['PASSAGE_ID_SEPARATOR']


class RelevanceInference(autoqrels.oneshot.OneShotLabeler):
    def __init__(self, model, tokenizer, queries_cache, passages_text_cache, batch_size=8):
        """
        :param model: Pre-trained DuoT5 model
        :param tokenizer: Corresponding tokenizer for the model
        :param queries_cache: A dictionary mapping query_id to query text
        :param passages_text_cache: A nested dictionary mapping query_id -> doc_id -> document text
        """
        self.model = model
        self.tokenizer = tokenizer
        self.queries_cache = queries_cache
        self.passages_text_cache = passages_text_cache
        self.batch_size = batch_size
        self.REL = self.tokenizer.encode('true')[0]
        self.NREL = self.tokenizer.encode('false')[0]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cache = {}  # TODO: Implement caching from disk

    def _infer_oneshot(self, query_id: str, unk_doc_id: str, rel_doc_ids: List[str]) -> List[float]:
        # Extract `unk_docno` from `unk_doc_id`
        unk_docno, _ = unk_doc_id.split(PASSAGE_ID_SEPARATOR)

        # Extract `rel_docnos` from `rel_doc_ids`
        rel_docnos = [rel_doc_id.split(PASSAGE_ID_SEPARATOR)[0] for rel_doc_id in rel_doc_ids]

        # Retrieve the text for the unknown document
        unk_doc_text = next(
            item['text'] for item in self.passages_text_cache[unk_docno] if item['docno'] == unk_doc_id
        )

        # Retrieve the text for the relevant documents
        rel_doc_texts = [
            next(item['text'] for item in self.passages_text_cache[rel_docno] if item['docno'] == rel_doc_id)
            for rel_docno, rel_doc_id in zip(rel_docnos, rel_doc_ids)
        ]
        return self.infer_oneshot_text(
            self.queries_cache[query_id],
            unk_doc_text,
            rel_doc_texts)

    def infer_oneshot_text(self, query_text: str, unk_doc_text: str, rel_doc_texts: List[str]) -> List[float]:
        it = ((query_text, unk_doc_text, d) for d in rel_doc_texts)
        result = []
        with torch.no_grad():
            for chunk in more_itertools.chunked(it, self.batch_size):
                batch = self.tokenizer.batch_encode_plus(
                    [f'Query: {q} Document0: {t2} Document1: {t1} Relevant:' for q, t1, t2 in chunk],
                    return_tensors='pt',
                    padding=True,
                    truncation=True)
                batch['decoder_input_ids'] = torch.full(
                    (batch['input_ids'].shape[0], 1),
                    self.model.config.decoder_start_token_id,
                    dtype=torch.long
                )
                res = self.model(**{k: v.to(self.device) for k, v in batch.items()})
                scores = res['logits'][:, 0, [self.REL, self.NREL]]
                scores = scores.softmax(dim=1)[:, 0].cpu().tolist()
                result.extend(scores)
        return result

# def _infer_oneshot(self, query_id: str, unk_doc_id: str, rel_doc_ids: List[str]) -> List[float]:
#         """
#         Compute scores for a query, an unknown document, and a list of relevant documents.
#         :param query_id: ID of the query
#         :param unk_doc_id: ID of the document to be labeled
#         :param rel_doc_ids: List of document IDs known to be relevant for the query
#         :return: List of relevance scores
#         """
#         # Retrieve query text
#         query_text = self.queries_cache[query_id]

#         # Retrieve document texts
#         unk_doc_text = self.passages_text_cache[query_id][unk_doc_id]
#         rel_doc_texts = [self.passages_text_cache[query_id][doc_id] for doc_id in rel_doc_ids]

#         # Prepare inputs for the DuoT5 model
#         scores = []
#         # Score for unknown document
#         unk_input = f"query: {query_text} document: {unk_doc_text}"
#         unk_tokens = self.tokenizer(unk_input, return_tensors="pt", truncation=True, padding=True)
#         unk_score = self.model(**unk_tokens).logits.item()  # Assume the model outputs logits as relevance scores
#         scores.append(unk_score)

#         # Scores for relevant documents
#         for rel_doc_text in rel_doc_texts:
#             rel_input = f"query: {query_text} document: {rel_doc_text}"
#             rel_tokens = self.tokenizer(rel_input, return_tensors="pt", truncation=True, padding=True)
#             rel_score = self.model(**rel_tokens).logits.item()
#             scores.append(rel_score)

#         return scores
