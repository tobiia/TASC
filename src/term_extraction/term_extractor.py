import os
import numpy as np
from tqdm import tqdm
import torch
from transformers import AutoModel, AutoTokenizer
from collections import defaultdict
from typing import Tuple
from models import CandidateSummary, TermEmbeddings, TermSummary

from candidate_extractor import EnglishPhraseExtractor

# from qdrant_client import QdrantClient
# from qdrant_client.http import models
# from qdrant_client.http.models import CollectionStatus

# NOTE --> can try fasttext?

"""This is a pipeline for terminology extraction.

Leave one blank line.  The rest of this docstring should contain an
overall description of the module or program.  Optionally, it may also
contain a brief description of exported classes and functions and/or usage
examples.

Typical usage example:

  extractor = TermExtractor()
  terms = extractor.extract_terms(corpus_path)
"""


class TermExtractor:
    def __init__(
        self,
        corpus_path: str,
        stop_words_path: str | None,
        max_seq_length=384,
        batch_size=64,
        model_name="sentence-transformers/all-mpnet-base-v2",
        topic_threshold=0.4,
        self_sim_threshold=0.5,
        context_diff_threshold=0.3,
    ):
        self.corpus_path = corpus_path
        self.stop_words_path = stop_words_path
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()

        self.topic_threshold = topic_threshold
        self.self_sim_threshold = self_sim_threshold
        self.context_diff_threshold = context_diff_threshold

        special_tokens = set(self.tokenizer.special_tokens_map.values())
        special_tokens.update(self.tokenizer.added_tokens_encoder.keys())
        self.special_tokens = list(special_tokens)

        # maps sentences to their token ids and token embeddings
        self.sentence_cache: dict[str, tuple[list[str], np.ndarray]] = {}

    def extract_candidates(self) -> Tuple[dict[str, list[str]], dict[str, list[str]]]:
        """Extracts candidate terms from the corpus.

        Retrieves candidate terms from the corpus using EnglishPhraseExtractor.

        Returns:
            2 dicts mapping candidates to their corresponding list of
            sentences. The first are the unigrams, second are the ngrams. For example:
            {"corporation": ["I hate corporations", "I love corporations"]}
        """

        # FIXME how do i call stop words??
        candidate_extractor = EnglishPhraseExtractor(path=self.corpus_path)
        unigram_candidates, ngram_candidates = candidate_extractor.extract_candidates()

        return unigram_candidates, ngram_candidates

    def l2_normalize(self, x):
        return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-9)

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]  # input ids
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    def create_embeddings(
        self,
        texts: list[str],
        tokenizer=None,
        model=None,
        store_cache: bool = False,
    ) -> np.ndarray:
        """Creates mean-pooled sentence embeddings.

        Creates mean-pooled sentence embeddings of the given strings. Will use the class
        tokenizer and sentence embedding model unless another is given.

        Args:
            store_cache:
                If True then, for each text, its token IDs and token embeddings will be
                used to construct word embeddings for each word within. These are then
                stored in a cache dict with the sentences as keys and a Tuple of
                (list[words], ndarray[word embeddings]). If contextualized word
                embeddings need to be constructed, this decreases complexity.

        Returns:
            Numpy array of all the embeddings stacked and L2 normalized
        """

        tokenizer = tokenizer or self.tokenizer
        model = model or self.model

        all_embeddings = []

        for batch_start in range(0, len(texts), self.batch_size):
            batch = texts[batch_start : batch_start + self.batch_size]
            encoded_input = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_seq_length,
                return_tensors="pt",
            )

            with torch.no_grad():
                model_output = model(**encoded_input)

            # pool token embeddings to create sentence embeddings
            sent_embeds = self.mean_pooling(
                model_output, encoded_input["attention_mask"]
            )
            all_embeddings.append(sent_embeds.cpu().numpy())

            # reconstruct word embeddings from token embeddings and store
            # sentences are the keys to the cache dict
            if store_cache:
                batch_tok_embeds = model_output.last_hidden_state.cpu().numpy()
                for i, sent in enumerate(batch):
                    tokens = tokenizer.convert_ids_to_tokens(
                        encoded_input["input_ids"][i]
                    )
                    token_embeds = batch_tok_embeds[i]
                    symbols, symbol_embeds = self.reconstruct_words(
                        tokens, token_embeds
                    )
                    words, word_embeds = self.merge_hyphenated(
                        [s.lower() for s in symbols], symbol_embeds
                    )
                    self.sentence_cache[sent] = (words, word_embeds)

        return self.l2_normalize(np.concatenate(all_embeddings, axis=0))

    def encode(
        self, candidates: dict[str, list[str]], model_name: str | None = None
    ) -> dict[str, CandidateSummary]:
        """Encodes all candidate term occurrences and sentences

        Creates embeddings for each occurrences of each candidate term and the corresponding sentences. Used to create contextualized word embeddings.

        Returns:
            Dict with candidates as keys and CandidateSummary as values. CandidateSummary are Tuples containing a list of sentences and ndarray of sentence embeddings for each candidate.
        """

        tokenizer = self.tokenizer
        model = self.model

        # if given a different model name, program will use this instead
        if model_name:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name)
            model.eval()

        all_sentences = [s for sents in candidates.values() for s in sents]
        unique_sentences = list(dict.fromkeys(all_sentences))
        # save where each sentence is within unique_sentences
        sentence_to_idx = {s: i for i, s in enumerate(unique_sentences)}

        # all embeddings stacked, axis 0 indices match unique_sentences
        # + sentence_to_idx
        sentence_embeddings = self.create_embeddings(
            unique_sentences, tokenizer=tokenizer, model=model, store_cache=True
        )

        # for each candidate, create an array of the indices of its occurrence
        # sentences in sentence_embeddings using sentence_to_idx
        candidate_sent_indices = {
            cand: np.array([sentence_to_idx[s] for s in sents], dtype=int)
            for cand, sents in candidates.items()
        }

        # create CandidateSummary dataclass for holding info
        encoded_candidates = {}
        # indices is an ndarray
        for cand, indices in candidate_sent_indices.items():
            # fancy indexing
            # gets the embeddings/rows at each index
            sent_embeds = sentence_embeddings[indices]
            encoded_candidates[cand] = CandidateSummary(
                sentence_list=candidates[cand],
                sent_embeds=sent_embeds,
            )
        return encoded_candidates

    def encode_general(
        self, candidates: dict[str, list[str]], model_name: str | None = None
    ) -> dict[str, TermEmbeddings]:
        """Encodes all candidate terms and sentences

        Creates mean-pooled embeddings for all candidate terms and their corresponding sentences. Same as encode() but this creates only one word embedding for the candidates. Use encode() for contextualized word embeddings.

        Returns:
            Dict with candidates as keys and TermEmbeddings as values. TermEmbeddings are Tuples containing the word embedding and ndarray of sentence embeddings for each candidate.
        """

        # look at encode() if you want to understand how this works

        tokenizer = self.tokenizer
        model = self.model

        if model_name:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name)
            model.eval()

        all_sentences = [s for sents in candidates.values() for s in sents]
        unique_sentences = list(dict.fromkeys(all_sentences))
        sentence_to_idx = {s: i for i, s in enumerate(unique_sentences)}

        sentence_embeddings = self.create_embeddings(
            unique_sentences, tokenizer=tokenizer, model=model
        )

        indiv_candidates = list(candidates.keys())
        candidate_embeddings = self.create_embeddings(
            indiv_candidates, tokenizer=tokenizer, model=model
        )

        candidate_sent_indices = {
            cand: np.array([sentence_to_idx[s] for s in sents], dtype=int)
            for cand, sents in candidates.items()
        }

        encoded_candidates = {}
        for i, cand in enumerate(indiv_candidates):
            idx = candidate_sent_indices[cand]
            sent_embeds = sentence_embeddings[idx]
            encoded_candidates[cand] = TermEmbeddings(
                word_embed=candidate_embeddings[i],
                sent_embeds=sent_embeds,
            )
        return encoded_candidates

    def create_word_embeddings(
        self,
        encoded_candidates: dict[str, CandidateSummary],
        mode: str = "mean",
    ) -> dict[str, TermEmbeddings | TermSummary]:
        """Create word embeddings for all candidates

        Word embeddings are the mean average of the token embeddings
        comprising them.

        Args:
            NOTE should probably be changed to a flag
            mode: whether a single contextualized or multiple word
                embeddings for each word are created

        Returns:
            Dict with candidates as keys and TermEmbeddings OR TermSummarys
            as values. TermEmbedding = single contextualized word embedding,
            TermSummary = word embeddings for every occurrence of the word
        """

        candidate_embeddings = {}
        for candidate, info in encoded_candidates.items():
            _, emb = self.token_to_word(candidate, info, mode)
            if emb is not None:
                candidate_embeddings[candidate] = emb
        return candidate_embeddings

    def token_to_word(self, candidate: str, info: CandidateSummary, mode: str):
        """Collects and returns all constructed word embeddings for a candidate

        Matches word embeddings with the given single candidate.

        Args:
            mode: whether a single contextualized or multiple word
                embeddings for each word are created

        Returns:
            Dict with candidates as keys and TermEmbeddings OR TermSummarys
            as values. TermEmbedding = single contextualized word embedding,
            TermSummary = word embeddings for every occurrence of the word
        """
        all_embeds = []
        # split multi-word candidates into separate words to ease
        # reconstruction
        candidate_subwords = candidate.split()
        # for the sliding window
        k = len(candidate_subwords)

        for sent in info.sentence_list:
            # use cached tokens & embeddings
            words, word_embeds = self.sentence_cache[sent]
            for i in range(len(words) - k + 1):
                if words[i : i + k] == candidate_subwords:
                    # add to list of all word embeddings for candidate
                    # if multi-word, the word embeddings of each are
                    # averaged to combine them
                    all_embeds.append(
                        np.mean(word_embeds[i : i + k], axis=0, keepdims=True)
                    )

        if not all_embeds:
            return candidate, None

        # stack the word embeddings of each occurrence of the candidate together
        all_embeds = np.vstack(all_embeds)
        all_embeds = self.l2_normalize(all_embeds)

        if mode == "mean":
            return candidate, TermEmbeddings(
                word_embed=self.l2_normalize(np.mean(all_embeds, axis=0)),
                sent_embeds=info.sent_embeds,
            )
        else:
            return candidate, TermSummary(
                word_embeds=all_embeds, sent_embeds=info.sent_embeds
            )

    def reconstruct_words(self, tokens, embeddings):
        """Constructs word embeddings of a sentence from token embeddings

        Word embeddings are the mean average of the token embeddings
        comprising them. Input is one sentence

        Returns:
            all the reconstructed words in the sentence +
            corresponding embeddings, the indices will match
        """
        words = []
        word_embeds = []
        frags, frags_emb = [], []

        for t, e in zip(tokens, embeddings):
            # ignore the special tokens of the model, ex. <cls>
            if t in self.special_tokens:
                continue
            # NOTE need to check if I can expand to other model types
            # this was specifically written for mpnet
            if t.startswith("##"):
                frags.append(t[2:])
                frags_emb.append(e)
            else:
                if frags:
                    words.append("".join(frags))
                    # avg all token embeddings to get word embedding
                    word_embeds.append(np.mean(frags_emb, axis=0))
                    frags, frags_emb = [], []
                frags = [t.lstrip("#")]
                frags_emb = [e]
        # get final word if there is one
        if frags:
            words.append("".join(frags))
            word_embeds.append(np.mean(frags_emb, axis=0))
        if not word_embeds:
            return [], np.empty((0, embeddings.shape[-1]))

        # output: all the reconstructed words in the sentence +
        # corresponding embeddings, the indices will match
        return words, np.vstack(word_embeds)

    def merge_hyphenated(self, words, word_embeds):
        """Re-join words that BERT split at hyphens/apostrophes into their original form.

        Re-join words that BERT split at hyphens/apostrophes into their original form.
        ex. candidate -> ["it-developers"]
        tokenizer -> ["it", "-", "developers"]
        reconstruct_words -> ["it", "-", "developers"] with embedding for each part.
        merge_hyphenated -> ["it-developers"] with averaged embedding for whole word.

        Returns:
            the merged word list and corresponding embeddings.
        """

        if not words:
            return words, word_embeds
        merged_words = []
        merged_embeds = []
        i = 0
        while i < len(words):
            word = words[i]
            emb_parts = [word_embeds[i]]
            while i + 2 < len(words) and words[i + 1] in ("-", "'"):
                word = word + words[i + 1] + words[i + 2]
                emb_parts.append(word_embeds[i + 1])
                emb_parts.append(word_embeds[i + 2])
                i += 2
            merged_words.append(word)
            merged_embeds.append(np.mean(emb_parts, axis=0))
            i += 1
        return merged_words, np.array(merged_embeds)

    def self_similarity(
        self, word_embeddings: dict[str, TermSummary], max_sample_size=5000
    ):
        ss_score = {}

        for word, info in tqdm(
            word_embeddings.items(), desc="Calculating self-similarity scores..."
        ):
            X = info.word_embeds
            # N = number of candidate occurrences and thus word embeddings
            N = X.shape[0]

            # can't calculate cosim on just 1
            if N < 2:
                continue

            # subsample if too many
            if N > max_sample_size:
                idx = np.random.choice(N, max_sample_size, replace=False)
                X = X[idx]
                N = X.shape[0]

            # each embed is divided by its norm -> A/||A||
            # should've alr been normalized before getting to this func
            # cosim = A dot B / ||A||||B||
            # = A * B^T, @ = matrix mult
            sim_matrix = X @ X.T

            # remember: now we are calculating the avg of all scores (each entry in the matrix)
            # sum all entries incl diagonal
            # subtract diagonal entries (which would be N 1s/ones = N)
            # divide by (N entries * N-1 non-diagonal pairs)
            ss = (np.sum(sim_matrix) - N) / (N * (N - 1))
            ss_score[word] = float(round(ss, 3))

        return ss_score

    def contextualized_vs_general(
        self, candidate_embeddings: dict[str, TermEmbeddings]
    ):
        all_candidates = list(candidate_embeddings.keys())

        general_embeddings = self.create_embeddings(all_candidates)

        context_embeddings = np.vstack(
            [candidate_embeddings[c].word_embed for c in all_candidates]
        )
        # calculate cosine similarity between the general word embeddings
        # and all their corresponding contextualized word embedding
        cos_sims = np.sum(context_embeddings * general_embeddings, axis=1)
        diff_scores = dict(zip(all_candidates, 1 - cos_sims))
        return diff_scores

    def topic_score(
        self,
        candidate_tuples: dict[str, TermEmbeddings],
        method="max",
    ):
        topic_scores = {}

        for word, info in candidate_tuples.items():

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

        return topic_scores

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

    def extract_terms(
        self,
        use_ngrams: bool = True,
        mode: str = "mean",  # mean or all
        compute_topic: bool = True,
        compute_self_sim: bool = False,
        compute_context_diff: bool = False,
        compute_ssc: bool = False,
    ):
        unigram_candidates, ngram_candidates = self.extract_candidates()
        candidates = ngram_candidates if use_ngrams else unigram_candidates

        if not candidates:
            return []

        encoded = self.encode(candidates)

        # either TermEmbeddings aka "contextualized" single embedding
        # or TermSummary aka embeddings for each context
        term_candidates = self.create_word_embeddings(encoded, mode=mode)

        # FIXME will not be keeping all these functions, but it's like this for testing. must remember to remove pylance commands later

        # requires TermEmbeddings
        if compute_topic:
            filtered_candidates = {}
            topic_scores = self.topic_score(term_candidates)  # type: ignore
            for word, info in term_candidates.items():
                if topic_scores[word] >= self.topic_threshold:
                    filtered_candidates[word] = info
            return filtered_candidates

        # requires TermSummary
        if compute_self_sim:
            filtered_candidates = {}
            ss_scores = self.self_similarity(term_candidates)  # type: ignore
            for word, info in term_candidates.items():
                score = ss_scores.get(word)
                # meaningful = high self sim
                # CAST says thres = 0.3
                if score is None or score >= self.self_sim_threshold:
                    filtered_candidates[word] = info
            return filtered_candidates

        # requires TermEmbeddings
        if compute_context_diff:
            filtered_candidates = {}
            diff_scores = self.contextualized_vs_general(term_candidates)  # type: ignore
            for word, info in term_candidates.items():
                if diff_scores[word] >= self.context_diff_threshold:
                    filtered_candidates[word] = info
            return filtered_candidates

        # requires TermSummary
        if compute_ssc:
            vanilla_encoded = self.encode(candidates, model_name="microsoft/mpnet-base")

            vanilla_candidates = self.create_word_embeddings(vanilla_encoded, mode=mode)

            filtered_candidates = {}
            ssc_scores = self.self_similarity_change(
                term_candidates, vanilla_candidates  # type: ignore
            )
            for word, info in term_candidates.items():
                score = ssc_scores.get(word)
                # thres >= 0, positive
                if score is None or score >= self.self_sim_threshold:
                    filtered_candidates[word] = info
            return filtered_candidates

        return term_candidates
