import random
import numpy as np
from tqdm import tqdm
import torch
from transformers import AutoModel, AutoTokenizer
from collections import defaultdict
from candidate_extractor import CandidateExtractor
from util import l2_normalize
from multiprocessing import Pool, cpu_count
from typing import Tuple
from models import CandidateSummary, TermEmbeddings, TermSummary

# from qdrant_client import QdrantClient
# from qdrant_client.http import models
# from qdrant_client.http.models import CollectionStatus

# NOTE --> can try fasttext?


class TermExtractor:
    def __init__(
        self,
        corpus_path,
        # stop_words,
        model_name="sentence-transformers/all-mpnet-base-v2",
        max_seq_length=384,
        topic_score_thres=0.4,
    ):
        self.corpus_path = corpus_path
        # self.stop_words = stop_words  # stop word list
        self.max_seq_length = max_seq_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        self.topic_score_thres = topic_score_thres

    # Dict: [cand, list[sentences]]
    def extract_candidates(self) -> Tuple[dict[str, list[str]], dict[str, list[str]]]:
        # REVIEW include document IDs? --> maybe come back and do this once I figure out the top2vec integration
        candidate_extractor = CandidateExtractor(path=self.corpus_path)
        unigram_candidates, ngram_candidates = candidate_extractor.process_corpus()

        return unigram_candidates, ngram_candidates

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    # input: Dict: [cand, list[sentences]]
    # output: Dict: [str, CandidateSummary(tokenized sentences, token embeddings, sentence embeddings)]
    def encode(self, candidates: dict[str, list[str]]) -> dict[str, CandidateSummary]:

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

        token_embeddings = model_output.last_hidden_state.cpu().numpy()

        sentence_embeddings = self.mean_pooling(
            model_output, encoded_input["attention_mask"]
        )
        sentence_embeddings = sentence_embeddings.cpu().numpy()
        sentence_embeddings = l2_normalize(sentence_embeddings)

        # regroup
        tokens_list = [
            self.tokenizer.convert_ids_to_tokens(ids)
            for ids in encoded_input["input_ids"]
        ]

        # Build index groups per candidate
        groups = defaultdict(list)
        for i, cand in enumerate(candidate_map):
            groups[cand].append(i)

        # Regroup embeddings safely into CandidateSummary
        encoded_candidates = {}
        for cand, indices in groups.items():
            indices = np.array(indices)
            encoded_candidates[cand] = CandidateSummary(
                tok_sents=[tokens_list[i] for i in indices],
                tok_embeds=token_embeddings[
                    indices
                ],  # (num_sentences, seq_len, hidden_dim)
                sent_embeds=sentence_embeddings[indices],  # (num_sentences, hidden_dim)
            )

        return encoded_candidates

    # input: Dict: [cand, list[sentences]]
    # output: Dict: [str, TermEmbeddings(word embedding, slist[sentence embedding(s)])]
    def encode_general(
        self, candidates: dict[str, list[str]]
    ) -> dict[str, TermEmbeddings]:
        all_sentences = []
        candidate_map = []
        indiv_candidates = list(candidates.keys())

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
        sentence_embeddings = sentence_embeddings.cpu().numpy()
        sentence_embeddings = l2_normalize(sentence_embeddings)

        candidate_embeddings = self.mean_pooling(
            model_output_cand, encoded_input_cand["attention_mask"]
        )
        candidate_embeddings = candidate_embeddings.cpu().numpy()
        candidate_embeddings = l2_normalize(candidate_embeddings)

        # Build index groups per candidate (handles unordered dict)
        groups = defaultdict(list)
        for i, cand in enumerate(candidate_map):
            groups[cand].append(i)

        encoded_candidates = {}
        for i, cand in enumerate(indiv_candidates):
            indices = np.array(groups[cand])
            encoded_candidates[cand] = TermEmbeddings(
                word_embed=candidate_embeddings[i],  # (hidden_dim,)
                sent_embeds=sentence_embeddings[indices],  # (num_sentences, hidden_dim)
            )

        return encoded_candidates

    # input: Tuple[str, CandidateSummary]
    # output: Tuple[str, TermEmbeddings]
    def reconstruct_words(self, tokens, embeddings):
        words = []
        word_embeds = []
        frags, frags_emb = [], []

        for t, e in zip(tokens, embeddings):
            if t in ["[PAD]", "[CLS]", "[SEP]"]:
                continue
            if t.startswith("##"):
                frags.append(t[2:])
                frags_emb.append(e)
            else:
                if frags:
                    words.append("".join(frags))
                    word_embeds.append(np.mean(frags_emb, axis=0))
                    frags, frags_emb = [], []
                frags = [t]
                frags_emb = [e]
        if frags:
            words.append("".join(frags))
            word_embeds.append(np.mean(frags_emb, axis=0))
        return words, np.vstack(word_embeds)

    def token_to_word(self, args: Tuple[str, CandidateSummary, str]):
        candidate, info, mode = args
        all_embeds = []

        for tokens, token_embeds in zip(info.tok_sents, info.tok_embeds):
            words, word_embeds = self.reconstruct_words(tokens, token_embeds)
            # check if any of the constructed words are the candidate
            mask = np.array([w.lower() == candidate for w in words])
            if mask.any():
                # add only those rows to total list of contextual word embeddings for the candidate
                all_embeds.append(word_embeds[mask])

        if not all_embeds:
            raise ValueError(f"No embeddings found for candidate: {candidate}")

        all_embeds = np.vstack(all_embeds)
        all_embeds = l2_normalize(all_embeds)

        if mode == "mean":
            return candidate, TermEmbeddings(
                word_embed=l2_normalize(np.mean(all_embeds, axis=0)),
                sent_embeds=info.sent_embeds,
            )
        else:
            return candidate, TermSummary(
                word_embeds=all_embeds, sent_embeds=info.sent_embeds
            )

    # input: Dict: [str, CandidateSummary]
    # output: Dict: [str, TermEmbeddings(contextualized embedding, sentence embed(s))]
    def create_word_embeddings(
        self,
        encoded_candidates: dict[str, CandidateSummary],
        n_processes: int | None = None,
        mode: str = "mean",
    ) -> dict[str, TermEmbeddings | TermSummary]:

        if n_processes is None:
            n_processes = max(cpu_count() - 1, 1)

        args = [
            (candidate, info, mode) for candidate, info in encoded_candidates.items()
        ]

        # create pool and map
        with Pool(processes=n_processes) as pool:
            results = pool.map(self.token_to_word, args)

        candidate_embeddings = dict(results)
        return candidate_embeddings

    # NOTE cosine distance = 1-cosim

    # TODO rewrite as indiv so parallelize

    def self_similarity(
        self, word_embeddings: dict[str, TermSummary], max_sample_size=5000
    ):
        ss_score = {}

        for word, info in tqdm(
            word_embeddings.items(), desc="Calculating self-similarity scores..."
        ):
            # NOTE check the dimensions for all these ndarrays, not sure if this is ok...
            X = info.word_embeds
            N = X.shape[0]

            if N < 2:
                continue

            # subsample if too many
            if N > max_sample_size:
                idx = np.random.choice(N, max_sample_size, replace=False)
                X = X[idx]
                N = X.shape[0]

            # each embed is divided by its norm -> A/||A||
            # cosim = A dot B / ||A||||B||
            # = A * B^T, @ = matrix mult, see above
            sim_matrix = X @ X.T

            # remember: now we are calculating the avg of all scores (each entry in the matrix)
            # sum all entries incl diagonal
            # subtract diagonal entries (which would be N 1s/ones = N)
            # divide by (N entries * N-1 non-diagonal pairs)
            ss = (np.sum(sim_matrix) - N) / (N * (N - 1))
            ss_score[word] = float(round(ss, 3))

        return ss_score

    def contextualized_vs_general(
        self, candidate_embeddings: dict[str, TermEmbeddings], model, tokenizer
    ):
        diff_scores = {}
        all_candidates = list(candidate_embeddings.keys())

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
        general_embeddings = general_embeddings.cpu().numpy()
        general_embeddings = l2_normalize(general_embeddings)

        context_embeddings = np.vstack(
            [candidate_embeddings[c].word_embed for c in all_candidates]
        )

        cos_sims = np.sum(context_embeddings * general_embeddings, axis=1)

        diff_scores = dict(zip(all_candidates, 1 - cos_sims))

        return diff_scores

    # input: dict [str, tuple[word embedding, List[sentence embeddings]]]
    def topic_score(
        self,
        candidate_tuples: dict[str, TermEmbeddings],
        method="max",
    ):
        topic_scores = {}
        filtered_candidates = {}

        all_candidates = list(candidate_tuples.keys())

        # cand_embeds = np.vstack([candidate_tuples[c].word_embed for c in all_candidates])

        for word, info in candidate_tuples.items():

            W = info.word_embed  # (D,)
            S = info.sent_embeds  # (N, D)

            cand_embedding = info.word_embed

            cos_sims = np.dot(
                info.sent_embeds, cand_embedding
            )  # shape: (num_occurrences,)

            if method == "max":
                score = np.max(cos_sims)
            elif method == "avg":
                score = np.mean(cos_sims)
            else:
                raise ValueError(f"Unknown method {method}")

            topic_scores[word] = float(score)

            if score <= self.topic_score_thres:
                filtered_candidates[word] = info

            cand_idx += 1

        return filtered_candidates, topic_scores

    def self_similarity_change(
        self,
        fine_tuned_embeddings: dict[str, TermSummary],
        vanilla_embeddings: dict[str, TermSummary],
        max_sample_size=5000,
    ):

        ssf = self.self_similarity(
            fine_tuned_embeddings, max_sample_size=max_sample_size
        )
        ssv = self.self_similarity(vanilla_embeddings, max_sample_size=max_sample_size)

        ssc_scores = {}
        for word in ssf.keys():
            if word in ssv:
                ssc_scores[word] = round(ssf[word] - ssv[word], 3)

        return ssc_scores

    def extract_terms(self) -> list[str]:
        return ["BOO!"]
