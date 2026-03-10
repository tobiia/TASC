import random

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
        topic_score_thres=0.4,
    ):
        self.corpus_path = corpus_path
        self.stop_words = stop_words  # stop word list
        self.max_seq_length = max_seq_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        self.topic_score_thres = topic_score_thres

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
        # Dict: [str, List[tokenized_sentences, token_embeddings, sentence_embeddings]]
        return encoded_candidates

    def encode_whole(self, candidates):
        indiv_candidates = candidates.keys()
        encoded_candidates = {}

        all_sentences = []
        candidate_map = []

        # str, list of str
        for candidate, sentences in candidates.items():
            for s in sentences:
                all_sentences.append(s)
                candidate_map.append(candidate)

        encoded_input_sent = self.tokenizer(
            all_sentences,
            padding=True,
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors="pt",
        )

        encoded_input_cand = self.tokenizer(
            indiv_candidates,
            padding=True,
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors="pt",
        )

        with torch.no_grad():
            model_output_sent = self.model(**encoded_input_sent)
            model_output_cand = self.model(**encoded_input_cand)

        sentence_embeddings = self.mean_pooling(
            model_output_sent, encoded_input_sent["attention_mask"]
        )
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        sentence_embeddings = sentence_embeddings.cpu().numpy()

        candidate_embeddings = self.mean_pooling(
            model_output_cand, encoded_input_cand["attention_mask"]
        )
        candidate_embeddings = F.normalize(candidate_embeddings, p=2, dim=1)
        candidate_embeddings = candidate_embeddings.cpu().numpy()

        # regroup
        for i in range(len(indiv_candidates)):
            if indiv_candidates[i] not in encoded_candidates:  # defaultdict?
                encoded_candidates[candidate] = [[], []]
            # add candidate embedding for each
            encoded_candidates[indiv_candidates[i]][0] = candidate_embeddings[i]

        for i, candidate in enumerate(candidate_map):
            encoded_candidates[candidate][1].append(sentence_embeddings[i])

        # Dict: [str, Tuple[word embedding, List[sentence_embedding(s)]]
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
            # REVIEW i misread Xiao 2026 so i probs don't need this
            if mode == "contextual":
                final = np.mean(word_embeddings, axis=0)
                final = l2_normalize(np.mean(word_embeddings, axis=0))
                return candidate, final
            else:
                return (
                    candidate,
                    word_embeddings,
                )  # REVIEW i don't think there's a scenario where i use this
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

    # NOTE cosine distance = 1-cosim

    # TODO rewrite as indiv so parallelize

    def compute_self_similarity_cast(self, word_embeddings, max_sample_size=5000):
        ss_score = {}

        for word, embeds in tqdm(
            word_embeddings.items(), desc="Calculating self-similarity scores..."
        ):
            if len(embeds) < 2:
                continue

            if len(embeds) > max_sample_size:
                embeds = random.sample(embeds, max_sample_size)

            # create matrix, (N, D)
            X = np.vstack(embeds)

            # each embed is divided by its norm -> A/||A||
            X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)

            # cosim = A dot B / ||A||||B||
            # = A * B^T, @ = matrix mult, see above
            sim_matrix = X_norm @ X_norm.T

            N = sim_matrix.shape[0]

            # remember: now we are calculating the avg of all scores (each entry in the matrix)
            # sum all entries incl diagonal
            # subtract diagonal entries (which would be N 1s/ones = N)
            # divide by (N entries * N-1 non-diagonal pairs)
            ss = (np.sum(sim_matrix) - N) / (N * (N - 1))

            ss_score[word] = float(round(ss, 3))

        return ss_score

    # NOTE input aka candidate_embeddings must = Dict[str, contextual_embedding]
    def contextualized_vs_general(self, candidate_embeddings, model, tokenizer):
        diff_scores = {}
        all_candidates = candidate_embeddings.keys

        encoded_input = tokenizer(
            all_candidates,
            padding="max_length",
            truncation=True,
            max_length=32,
            return_tensors="pt",
        )

        with torch.no_grad():
            model_output = model(**encoded_input)

        general_embeddings = self.mean_pooling(
            model_output, encoded_input["attention_mask"]
        )
        general_embeddings = F.normalize(general_embeddings, p=2, dim=1)
        general_embeddings = general_embeddings.cpu().numpy()

        # FIXME this can be faster --> pytorch?
        for idx, word in enumerate(candidate_embeddings.keys()):
            context_e = candidate_embeddings[idx]
            general_e = general_embeddings[idx]

            # normalize
            context_e_norm = context_e / (np.linalg.norm(context_e) + 1e-9)
            general_e_norm = general_e / (np.linalg.norm(general_e) + 1e-9)

            # cosine similarity = dot product
            cos_sim = np.dot(context_e_norm, general_e_norm)

            # difference
            diff_scores[word] = 1 - cos_sim

        return diff_scores

    # FIXME assignment on parameters means actually changing the reference so make sure i don't do that unless its intentional

    # REVIEW either use contextual or general, and then either avg or just 1 > threshold

    def topic_score(
        self,
        candidate_tuples,
        method="max",
    ):
        # dict [str, tuple[context embedding, List[sentence embeddings]]]

        topic_scores = {}
        filtered_candidates = []
        cand_idx = 0

        for word, info in candidate_tuples.items():
            cand_embedding = info[0]
            sentence_embeddings = info[1]

            sent_embeds = np.vstack(sentence_embeddings)

            cand_norm = cand_embedding / (np.linalg.norm(cand_embedding) + 1e-9)
            sent_norms = sent_embeds / (
                np.linalg.norm(sent_embeds, axis=1, keepdims=True) + 1e-9
            )

            cos_sims = np.dot(sent_norms, cand_norm)  # shape: (num_occurrences,)

            if method == "max":
                score = np.max(cos_sims)
            elif method == "avg":
                score = np.mean(cos_sims)
            else:
                raise ValueError(f"Unknown method {method}")

            topic_scores[word] = float(score)

            if score <= self.topic_score_thres:
                filtered_candidates.append(candidate_tuples[cand_idx])

            cand_idx += 1

        return filtered_candidates, topic_scores
