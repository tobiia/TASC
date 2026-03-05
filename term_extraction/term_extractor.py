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
from scipy.spatial.distance import cdist
import random
import os
import re
import pickle
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import nltk

nltk.download("stopwords")
# nltk.download('vader_lexicon')
from nltk.corpus import stopwords
from collections import defaultdict
from candidate_extractor import CandidateExtractor
from util import l2_normalize

# from qdrant_client import QdrantClient
# from qdrant_client.http import models
# from qdrant_client.http.models import CollectionStatus


class TermExtractor:
    def __init__(
        self,
        corpus,
        additional_text,
        stop_words,
        model_name="'sentence-transformers/all-mpnet-base-v2",
        max_seq_length=384,
    ):
        self.corpus = corpus
        self.additional_text = additional_text  # if there is additional text, it is used to calculate frequencies, terms are NOT extracted from it
        self.stop_words = stop_words  # stop word list
        self.max_seq_length = max_seq_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def extract_candidates(self, corpus):
        # TODO maybe count occurances per sentence for early stopping
        """
        Output: Dict[str, List[str]] --> dict of candidates and all their occurances in the corpus
        """
        # REVIEW include document IDs?
        candidates = CandidateExtractor(corpus)
        return candidates

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

        for candidate, sentences in tqdm(candidates.items()):
            # [ "blah blah word blah", "blah blah word", "word blah"]
            encoded_input = self.tokenizer(
                sentences,
                padding="max_length",
                truncation=True,
                max_length=self.max_seq_length,
                return_tensors="pt",
            )
            tokenized_sentences = encoded_input["input_ids"].tolist()

            with torch.no_grad():
                model_output = self.model(
                    encoded_input["input_ids"],
                    encoded_input["attention_mask"],
                )
                token_embeddings = model_output.last_hidden_state.numpy()

            sentence_embeddings = self.mean_pooling(
                model_output, encoded_input["attention_mask"]
            )
            sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
            sentence_embeddings = sentence_embeddings.numpy()

            encoded_candidates[candidate] = [
                tokenized_sentences,
                token_embeddings,
                sentence_embeddings,
            ]

        return encoded_candidates

    def token_to_word(self, encoded_candidates):
        candidate_embeddings = {}
        for candidate, info in tqdm(encoded_candidates.items()):
            word_embeddings = []
            word_frags = []
            frag_embeds = []
            for sentence_idx, tokenized_sentence in enumerate(info[0]):
                token_idx = 0
                for input_id in tokenized_sentence:
                    token = self.tokenizer.decode(input_id)
                    if token.startswith("#"):
                        # add to current word
                        word_frags.append(token[2:])
                        frag_embeds.append(info[1][sentence_idx][token_idx])
                    elif len(word_frags) > 0:
                        # current word is done
                        word = "".join(word_frags)
                        if word.lower() == candidate:
                            word_embeddings.append(
                                l2_normalize(np.mean(frag_embeds, axis=0))
                            )
                        word_frags.clear()
                        frag_embeds.clear()
                    else:
                        # start of loop
                        word_frags.append(token)
                        frag_embeds.append(info[1][sentence_idx][token_idx])
                    token_idx += 1
            # get final word if it exists
            if len(word_frags) > 0:
                word = "".join(word_frags)
                if word.lower() == candidate:
                    word_embeddings.append(l2_normalize(np.mean(frag_embeds, axis=0)))
            candidate_embeddings[candidate] = word_embeddings

        return candidate_embeddings

    def term_extraction(self):
        candidates = self.extract_candidates(self.corpus)

        encoded_candidates = self.encode(candidates)

        candidate_embeddings = self.token_to_word(encoded_candidates)
