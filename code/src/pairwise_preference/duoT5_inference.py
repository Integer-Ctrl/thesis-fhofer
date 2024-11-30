from typing import List
from transformers import T5ForConditionalGeneration, AutoTokenizer
import autoqrels
import more_itertools
import torch
import gzip
import json
import os


# Load the configuration settings
# def load_config(filename="../config.json"): does not work with debug
def load_config(filename="/mnt/ceph/storage/data-tmp/current/ho62zoq/thesis-fhofer/code/src/config.json"):
    with open(filename, "r") as f:
        config = json.load(f)
    return config


# Get the configuration settings
config = load_config()

DOCUMENT_DATASET_OLD_NAME = config['DOCUMENT_DATASET_OLD_NAME']
DATA_PATH = os.path.join(config['DATA_PATH'], DOCUMENT_DATASET_OLD_NAME)
DUOT5_QID_DOC_DOC_SYSTEM_SCORES_PATH = os.path.join(DATA_PATH, config['DUOT5_QID_DOC_DOC_SYSTEM_SCORES_PATH'])

KEY_SEP = config['KEY_SEPARATOR']
PASSAGE_ID_SEPARATOR = config['PASSAGE_ID_SEPARATOR']


def get_key(query_id: str, rel_doc_id: str, unk_doc_id: str, system: str) -> str:
    return f"{query_id}{KEY_SEP}{rel_doc_id}{KEY_SEP}{unk_doc_id}{KEY_SEP}{system}"


class RelevanceInference(autoqrels.oneshot.OneShotLabeler):
    def __init__(self, model, model_name, tokenizer, queries_cache, passages_text_cache, pairwise_cache, batch_size=8):
        """
        :param model: Pre-trained DuoT5 model
        :param tokenizer: Corresponding tokenizer for the model
        :param queries_cache: A dictionary mapping query_id to query text
        :param passages_text_cache: A nested dictionary mapping query_id -> doc_id -> document text
        :param batch_size: The batch size for inference
        :param pairwise_cache: A dictionary to cache pairwise scores
        """
        self.model = model
        self.model_name = model_name
        self.tokenizer = tokenizer
        self.queries_cache = queries_cache
        self.passages_text_cache = passages_text_cache
        self.batch_size = batch_size
        self.REL = self.tokenizer.encode('true')[0]
        self.NREL = self.tokenizer.encode('false')[0]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # cache for qid, docno1, docno2, system/model -> score
        self.cache = pairwise_cache

    def _infer_oneshot(self, query_id: str, unk_doc_id: str, rel_doc_ids: List[str]) -> List[float]:

        cached_scores = []
        uncached_doc_ids = []

        # check cache
        for rel_doc_id in rel_doc_ids:
            key = get_key(query_id, unk_doc_id, rel_doc_id, self.model_name)
            if key in self.cache:
                cached_scores.append(self.cache[key])
            else:
                uncached_doc_ids.append(rel_doc_id)

        # Extract unk_docno from unk_doc_id
        unk_docno, _ = unk_doc_id.split(PASSAGE_ID_SEPARATOR)

        # Extract rel_docnos from rel_doc_ids
        rel_docnos = [rel_doc_id.split(PASSAGE_ID_SEPARATOR)[0] for rel_doc_id in uncached_doc_ids]

        # Retrieve the text for the unknown document
        unk_doc_text = next(
            item['text'] for item in self.passages_text_cache[unk_docno] if item['docno'] == unk_doc_id
        )

        # Retrieve the text for the relevant documents
        rel_doc_texts = [
            next(item['text'] for item in self.passages_text_cache[rel_docno] if item['docno'] == rel_doc_id)
            for rel_docno, rel_doc_id in zip(rel_docnos, uncached_doc_ids)
        ]

        # infer uncached scores
        computed_scores = []
        if uncached_doc_ids:
            computed_scores = self.infer_oneshot_text(
                self.queries_cache[query_id],
                unk_doc_text,
                rel_doc_texts
            )

            # update cache
            for rel_doc_id, score in zip(uncached_doc_ids, computed_scores):
                key = get_key(query_id, unk_doc_id, rel_doc_id, self.model_name)
                self.cache[key] = score

        # combine cached and computed scores
        infered_scores = []
        uncached_idx = 0  # to respect the order
        for rel_doc_id in rel_doc_ids:
            key = get_key(query_id, unk_doc_id, rel_doc_id, self.model_name)
            if key in self.cache:
                infered_scores.append(self.cache[key])
            else:
                infered_scores.append(computed_scores[uncached_idx])
                uncached_idx += 1

        return infered_scores

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
