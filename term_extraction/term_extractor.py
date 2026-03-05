from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import itertools
from collections import Counter
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer
import torch.nn.functional as F
from collections import defaultdict
from candidate_extractor import CandidateExtractor
from util import l2_normalize
from multiprocessing import Pool, cpu_count

# TODO add type hints...
from typing import Dict, List, Tuple, Union

ResultType = Tuple[str, Union[np.ndarray, List[np.ndarray]]]

# from qdrant_client import QdrantClient
# from qdrant_client.http import models
# from qdrant_client.http.models import CollectionStatus


class TermExtractor:
    def __init__(
        self,
        corpus_path,
        stop_words,
        model_name="sentence-transformers/all-mpnet-base-v2",
        max_seq_length=384,
    ):
        self.corpus_path = corpus_path
        self.stop_words = stop_words  # stop word list
        self.max_seq_length = max_seq_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()

    def extract_candidates(self):
        # TODO maybe count occurances per sentence for early stopping
        # REVIEW include document IDs? --> maybe come back and do this once I figure out the top2vec integration
        """
        Input: List[Tuple(Span, Span)]], List[Tuple(Span, Span)]]
        --> Tuple(uni or ngram candidate, sentence they came from)
        """
        candidate_extractor = CandidateExtractor(path=self.corpus_path)
        unigram_candidates, ngram_candidates = candidate_extractor.process_corpus()

        unigram_grouped = self.group_candidates(unigram_candidates)
        ngram_grouped = self.group_candidates(ngram_candidates)

        return unigram_grouped, ngram_grouped

    def group_candidates(self, candidate_tuples):
        # restructuring the candidate tuples now that POS info isn't needed
        # TODO this should be done during candidate extraction
        grouped = defaultdict(list)

        for span, sent in candidate_tuples:
            text = span.text
            key = text.lower() if not text.islower() else text
            grouped.setdefault(key, []).append(sent.text)
        return grouped

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    def encode(self, candidates):
        encoded_candidates = {}

        all_sentences = []
        candidate_map = []

        # str, list of str
        for candidate, sentences in candidates.items():
            for s in sentences:
                all_sentences.append(s)
                candidate_map.append(candidate)

        # much more efficient to encode everything all at once
        encoded_input = self.tokenizer(
            all_sentences,
            padding=True,
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors="pt",
        )

        with torch.no_grad():
            model_output = self.model(**encoded_input)

        token_embeddings = model_output.last_hidden_state
        token_embeddings = token_embeddings.cpu().numpy()
        sentence_embeddings = self.mean_pooling(
            model_output, encoded_input["attention_mask"]
        )
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        sentence_embeddings = sentence_embeddings.cpu().numpy()

        # regroup
        for i, candidate in enumerate(candidate_map):
            if candidate not in encoded_candidates:  # defaultdict?
                encoded_candidates[candidate] = [[], [], []]

            encoded_candidates[candidate][0].append(
                encoded_input["input_ids"][i].tolist()
            )
            encoded_candidates[candidate][1].append(token_embeddings[i])
            encoded_candidates[candidate][2].append(sentence_embeddings[i])
        # List[tokenized_sentences, token_embeddings, sentence_embeddings]
        return encoded_candidates

    def token_to_word(self, args):
        candidate, info, tokenizer, mode = args
        tokenized_sentences, token_embeds, _ = info  # just to be more clear
        token_embeds = np.array(
            token_embeds
        )  # shape: (num_sentences, seq_len, hidden_dim)
        word_embeddings = []

        for sent_idx, tokens in enumerate(tokenized_sentences):
            embeddings = token_embeds[sent_idx]
            tokens = self.tokenizer.convert_ids_to_tokens(tokens)

            word_frags = []
            frag_embeds = []

            for t_idx, token in enumerate(tokens):
                # skip special tokens
                if token == "[PAD]" or token == "[CLS]" or token == "[SEP]":
                    continue

                if token.startswith("##"):
                    # add to current word
                    word_frags.append(token[2:])
                    frag_embeds.append(embeddings[t_idx])

                else:
                    # current word is done
                    if word_frags:
                        word = "".join(word_frags)
                        if word.lower() == candidate:
                            word_embeddings.append(
                                l2_normalize(np.mean(frag_embeds, axis=0))
                            )
                        word_frags.clear()
                        frag_embeds.clear()

                word_frags.append(token)
                frag_embeds = [embeddings[t_idx]]

        # flush final word
        if word_frags:
            word = "".join(word_frags)
            if word.lower() == candidate:
                word_embeddings.append(l2_normalize(np.mean(frag_embeds, axis=0)))

        if len(word_embeddings) > 0:
            if mode == "mean":
                final = np.mean(word_embeddings, axis=0)
                final = l2_normalize(np.mean(word_embeddings, axis=0))
                return candidate, final
            else:
                return candidate, word_embeddings
        else:
            raise ValueError(f"No embeddings found for candidate: {candidate}")

    def create_word_embeddings(
        self, encoded_candidates, tokenizer, mode="mean", n_processes=None
    ):
        if n_processes is None:
            n_processes = max(cpu_count() - 1, 1)

        args = [
            (candidate, info, tokenizer, mode)
            for candidate, info in encoded_candidates.items()
        ]

        # create pool and map
        with Pool(processes=n_processes) as pool:
            results: List[ResultType] = pool.map(self.token_to_word, args)

        candidate_embeddings = dict(results)
        return candidate_embeddings
