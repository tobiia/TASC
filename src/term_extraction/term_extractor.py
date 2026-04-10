import csv
import os
import numpy as np
from tqdm import tqdm
import torch
from transformers import AutoModel, AutoTokenizer
from term_extraction.models import CandidateSummary, TermSummary
from term_extraction.split_candidate_extractor import CandidateExtractor
from term_extraction.term_cache import EmbeddingCache
from config import TERM_PKG

# FIXME - remove later
from transformers.utils import logging as hf_logging

hf_logging.set_verbosity_error()

# from qdrant_client import QdrantClient
# from qdrant_client.http import models
# from qdrant_client.http.models import CollectionStatus

# REVIEW --> can try fasttext?

"""This is a pipeline for terminology extraction.

Leave one blank line.  The rest of this docstring should contain an
overall description of the module or program.  Optionally, it may also
contain a brief description of exported classes and functions and/or usage
examples.

Typical usage example:

  extractor = TermExtractor()
  terms = extractor.extract_terms(corpus_path)
"""


def save_set_to_csv(data_set, file_path):
    with open(file_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["term"])
        for item in sorted(data_set):
            writer.writerow([item])


class TermExtractor:
    def __init__(
        self,
        corpus_path: str,
        stop_words_path: str = "stop_words_en.txt",
        max_seq_length=256,
        batch_size=64,
        model_name="sentence-transformers/all-mpnet-base-v2",
        vanilla_model_name="microsoft/mpnet-base",
        # randomly gen via secrets.randbits(128)
        rng_seed=267135941556543938173580506427407010431,
    ):
        self.corpus_path = corpus_path
        self.stop_words_path = stop_words_path

        self.max_seq_length = max_seq_length
        self.batch_size = batch_size

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        self.vanilla_model_name = vanilla_model_name

        self.rng_seed = rng_seed
        self.rng = np.random.default_rng(rng_seed)

        # programmatically adding special tokens in case other models used
        special_tokens = set(self.tokenizer.special_tokens_map.values())
        special_tokens.update(self.tokenizer.added_tokens_encoder.keys())
        self.special_tokens = list(special_tokens)

        # maps sentence index (int) to token ids as text and token embeddings
        # can reconstruct sentences using dict[int][0]
        self.sentence_cache: dict[int, tuple[list[str], np.ndarray]] = {}
        self.error_terms = set()  # set that collects any terms dropped b/c of errors

    def get_unigram_cands(self) -> dict[str, list[str]]:
        """Extracts candidate terms from the corpus.

        Retrieves candidate terms from the corpus using CandidateExtractor.

        Returns:
            2 dicts mapping candidates to their corresponding list of
            sentences. The first are the unigrams, second are the ngrams. For example:
            {"corporation": ["I hate corporations", "I love corporations"]}
        """
        candidate_extractor = CandidateExtractor(
            path=self.corpus_path,
            stop_words_path=self.stop_words_path,
            cohesion_filter=False,
        )

        return candidate_extractor.unigram_candidates()

    def _l2_normalize(self, x):
        return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-9)

    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    def compute_anisotropy(self, embeddings, n_samples=1000):
        """Compute anisoptropy baseline
        Calculate anisotropy baseline as mean off-diagonal cosine similarity
        over randomly sampled embeddings.

        Returns:
            float
                Anisotropy baseline (mean cosine similarity of random pairs)
        """

        embeddings = np.asarray(embeddings)

        N = embeddings.shape[0]
        n_samples = min(n_samples, N)

        if n_samples < 2:
            self.anisotropy_baseline = 0

        # random sample of indices
        idx = self.rng.choice(N, size=n_samples, replace=False)
        sample = embeddings[idx]

        # cosine similarity matrix
        sim = sample @ sample.T

        # remove diagonal (self-similarity)
        n = sim.shape[0]
        self.anisotropy_baseline = (sim.sum() - np.trace(sim)) / (n * (n - 1))

    def _create_embeddings(
        self,
        texts: list[str],
        tokenizer=None,
        model=None,
        sentence_cache: bool = False,
        sentence_to_idx: dict[str, int] | None = None,
    ) -> np.ndarray:
        """Creates mean-pooled sentence embeddings.

        Creates mean-pooled sentence embeddings of the given strings. Will use the class
        tokenizer and sentence embedding model unless another is given.

        Args:
            sentence_cache:
                If True then, for each text, its token IDs and token embeddings will be
                used to construct word embeddings for each word within. These are then
                stored in self.sentence_cache keyed by integer index (via sentence_to_idx)
                as a Tuple of (list[words], ndarray[word embeddings]). Requires
                sentence_to_idx to be provided. If contextualized word embeddings need
                to be constructed, this decreases complexity.

        Returns:
            Numpy array of all the embeddings stacked and L2 normalized
        """

        tokenizer = tokenizer or self.tokenizer
        model = model or self.model

        all_embeddings = []

        batch_iter = tqdm(
            range(0, len(texts), self.batch_size),
            desc="******************************************** embedding batches...",
        )
        for batch_start in batch_iter:
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
            sent_embeds = self._mean_pooling(
                model_output, encoded_input["attention_mask"]
            )
            all_embeddings.append(sent_embeds.cpu().numpy())

            # reconstruct word embeddings from token embeddings and store
            # integer indices (via sentence_to_idx) are the keys to the cache
            if sentence_cache and sentence_to_idx is not None:
                batch_tok_embeds = model_output.last_hidden_state.cpu().numpy()
                for i, sent in enumerate(batch):
                    tokens = tokenizer.convert_ids_to_tokens(
                        encoded_input["input_ids"][i]
                    )
                    token_embeds = batch_tok_embeds[i]
                    symbols, symbol_embeds = self._reconstruct_words(
                        tokens, token_embeds
                    )
                    words, word_embeds = self._merge_hyphenated(
                        [s.lower() for s in symbols], symbol_embeds
                    )
                    self.sentence_cache[sentence_to_idx[sent]] = (
                        words,
                        word_embeds,
                    )

        return self._l2_normalize(np.concatenate(all_embeddings, axis=0))

    def _encode(
        self,
        candidates: dict[str, list[str]],
        model_name: str | None = None,
        sentence_cache: bool = False,
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
        sentence_embeddings = self._create_embeddings(
            unique_sentences,
            tokenizer=tokenizer,
            model=model,
            sentence_cache=sentence_cache,
            sentence_to_idx=sentence_to_idx,
        )

        # for each candidate, create an array of the indices of its occurrence
        # sentences in sentence_embeddings using sentence_to_idx
        candidate_sent_indices = {}
        for cand, sents in candidates.items():
            try:
                candidate_sent_indices[cand] = np.array(
                    [sentence_to_idx[s] for s in sents], dtype=int
                )
            except KeyError:
                self.error_terms.add(cand)

        # create CandidateSummary dataclass for holding info
        encoded_candidates = {}
        # indices is an ndarray
        for cand, indices in candidate_sent_indices.items():
            if len(indices) == 0:
                self.error_terms.add(cand)
                continue
            # fancy indexing
            # gets the embeddings/rows at each index
            sent_embeds = sentence_embeddings[indices]
            encoded_candidates[cand] = CandidateSummary(
                sentence_indices=indices.tolist(),
                sent_embeds=sent_embeds,
            )
        return encoded_candidates

    def _create_word_embeddings(
        self,
        encoded_candidates: dict[str, CandidateSummary],
    ) -> dict[str, TermSummary]:
        """Create word embeddings for all candidates

        Word embeddings are the mean average of the token embeddings
        comprising them.

        Returns:
            Dict with candidates as keys and TermSummarys as values.
            TermSummary = word embeddings for every occurrence of the word
        """

        candidate_embeddings = {}
        for candidate, info in tqdm(
            encoded_candidates.items(),
            desc="******************************************** building word embeddings...",
        ):
            try:
                _, emb = self._token_to_word(candidate, info)
                if emb is not None:
                    candidate_embeddings[candidate] = emb
                else:
                    self.error_terms.add(candidate)
            except Exception:
                self.error_terms.add(candidate)
        return candidate_embeddings

    def _token_to_word(
        self,
        candidate: str,
        info: CandidateSummary,
    ):
        """Collects and returns all constructed word embeddings for a candidate

        Matches word embeddings with the given single candidate.

        Args:
            info: TermSummary

        Returns:
            Dict with candidates as keys and TermSummarys as values.
            TermSummary = word embeddings for every occurrence of the word
        """
        all_embeds = []
        matched_positions = []  # positions in sentence_indices that produced a match
        # split multi-word candidates into separate words to ease reconstruction
        candidate_subwords = candidate.split()
        # for the sliding window
        k = len(candidate_subwords)

        # each index maps to the tokens and token embeddings of a sentence
        # so we're looping over all sentences containing the candidate

        for pos, idx in enumerate(info.sentence_indices):
            words, word_embeds = self.sentence_cache[idx]
            for i in range(len(words) - k + 1):
                if words[i : i + k] == candidate_subwords:
                    # add to list of all word embeddings for candidate
                    # if multi-word, the word embeddings of each are
                    # averaged to combine them
                    all_embeds.append(
                        np.mean(word_embeds[i : i + k], axis=0, keepdims=True)
                    )
                    matched_positions.append(
                        pos
                    )  # i.e position = idx in the sentences list/indices
                    break  # limit to 1 occurrence per sentence

        if not all_embeds:
            return candidate, None

        # stack the word embeddings of each occurrence of the candidate together
        all_embeds = np.vstack(all_embeds)
        # only keep sent_embeds for sentences that actually matched to ensure word+sent embeds align
        matched_sent_embeds = info.sent_embeds[matched_positions]

        # we only normalize the word embeddings at the VERY end to ensure each part of the word's magnitude contributes to the embedding for semantics
        all_embeds = self._l2_normalize(all_embeds)
        return candidate, TermSummary(
            word_embeds=all_embeds, sent_embeds=matched_sent_embeds
        )

    def _reconstruct_words(self, tokens, embeddings):
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
            # REVIEW - need to check if I can expand to other model types
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
                frags = [t]
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

    def _merge_hyphenated(self, words, word_embeds):
        """Re-join words that BERT split at hyphens/apostrophes into their original form.

        Re-join words that BERT split at hyphens/apostrophes into their original form.
        ex. candidate -> ["it-developers"]
        tokenizer -> ["it", "-", "developers"]
        _reconstruct_words -> ["it", "-", "developers"] with embedding for each part.
        _merge_hyphenated -> ["it-developers"] with averaged embedding for whole word.

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

    def contextualized_vs_general(
        self,
        candidate_embeddings: dict[str, TermSummary],
        general_embeddings,
        max_sample_size=50,
    ):
        all_candidates = list(candidate_embeddings.keys())

        cvg_scores = {}
        for candidate, general_emb in tqdm(
            zip(all_candidates, general_embeddings),
            desc="******************************************** calculating context vs general scores...",
        ):
            context_embs = candidate_embeddings[
                candidate
            ].word_embeds  # (n_contexts, H)

            # subsample if too many
            N = context_embs.shape[0]
            if N > max_sample_size:
                idx = self.rng.choice(N, max_sample_size, replace=False)
                context_embs = context_embs[idx]

            cos_sims = context_embs @ general_emb  # (n_contexts,)
            cvg_scores[candidate] = np.mean(1 - cos_sims)

        return cvg_scores

    def topic_score(
        self,
        candidate_embeddings: dict[str, TermSummary],
        # REVIEW topic score aggregate type
        method="max",
        max_sample_size=500,
        adjust_anisotropy=True,
    ):
        topic_scores = {}

        for word, info in tqdm(
            candidate_embeddings.items(),
            desc="******************************************** calculating topic scores...",
        ):
            X = info.sent_embeds
            Y = info.word_embeds
            N = X.shape[0]

            # subsample if too many
            if N > max_sample_size:
                idx = self.rng.choice(N, max_sample_size, replace=False)
                X = X[idx]
                Y = Y[idx]
                N = X.shape[0]

            # cos sim of each sent-word pair
            cos_sim = np.sum(X * Y, axis=1)
            if adjust_anisotropy and hasattr(self, "anisotropy_baseline"):
                cos_sim = cos_sim - self.anisotropy_baseline

            if method == "max":
                # t-extractor method
                score = cos_sim.max()
            elif method == "mean":
                score = cos_sim.mean()
            else:
                raise ValueError(f"Unknown method {method}")

            topic_scores[word] = float(score)

        return topic_scores

    def context_dispersion(
        self,
        word_embeddings: dict[str, TermSummary],
        min_count: int = 5,
        max_sample_size: int = 500,
        adjust_anisotropy: bool = True,
    ) -> dict[str, float]:
        """essentially sentence variation --> domain = low dispersion"""
        scores = {}

        for word, info in tqdm(
            word_embeddings.items(),
            desc="******************************************** calculating context dispersion...",
        ):
            sent_embeds = info.sent_embeds  # (N, H)
            N = sent_embeds.shape[0]

            if N < min_count:
                continue

            if N > max_sample_size:
                idx = self.rng.choice(N, max_sample_size, replace=False)
                sent_embeds = sent_embeds[idx]
                N = sent_embeds.shape[0]

            # pairwise cosine similarities between sentence embeddings
            sim_matrix = sent_embeds @ sent_embeds.T  # (N, N)

            # mean off-diagonal similarity = how similar the contexts are to each other
            mean_sim = (sim_matrix.sum() - N) / (N * (N - 1))

            if adjust_anisotropy and hasattr(self, "anisotropy_baseline"):
                mean_sim -= self.anisotropy_baseline

            # distance = 1 - similarity, so low score = tight contexts
            scores[word] = float(round(1 - mean_sim, 3))

        return scores

    # only for high frequency terms
    def self_similarity(
        self,
        word_embeddings: dict[str, TermSummary],
        adjust_anisotropy=True,
        anisotropy_value: float | None = None,
        max_sample_size=500,
        min_count=2,
    ):
        ss_score = {}

        for word, info in tqdm(
            word_embeddings.items(),
            desc="******************************************** calculating self-similarity scores...",
        ):
            X = info.word_embeds
            # N = number of candidate occurrences and thus word embeddings
            N = X.shape[0]

            if N < min_count:
                continue

            # subsample if too many
            if N > max_sample_size:
                idx = self.rng.choice(N, max_sample_size, replace=False)
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
            if adjust_anisotropy:
                if anisotropy_value is not None:
                    ss -= anisotropy_value
                elif hasattr(self, "anisotropy_baseline"):
                    ss -= self.anisotropy_baseline
            ss_score[word] = float(round(ss, 3))

        return ss_score

    def self_similarity_change(
        self,
        fine_tuned_embeddings: dict[str, TermSummary],
        vanilla_embeddings: dict[str, TermSummary],
        max_sample_size=5000,
        adjust_anisotropy=True,
    ):

        shared_keys = fine_tuned_embeddings.keys() & vanilla_embeddings.keys()
        shared_fine_tuned = {k: fine_tuned_embeddings[k] for k in shared_keys}
        shared_vanilla = {k: vanilla_embeddings[k] for k in shared_keys}

        # compute anisotropy baseline from the vanilla model's embeddings
        vanilla_anisotropy = None
        if adjust_anisotropy:
            vanilla_embeds_list = []
            for info in shared_vanilla.values():
                vanilla_embeds_list.append(info.word_embeds)
            if vanilla_embeds_list:
                all_vanilla_embeds = np.vstack(vanilla_embeds_list)
                N = all_vanilla_embeds.shape[0]
                n_samples = min(1000, N)
                idx = self.rng.choice(N, size=n_samples, replace=False)
                sample = all_vanilla_embeds[idx]
                sim = sample @ sample.T
                n = sim.shape[0]
                vanilla_anisotropy = float((sim.sum() - np.trace(sim)) / (n * (n - 1)))
                print(f"##### vanilla anisotropy baseline: {vanilla_anisotropy:.4f}")

        ssf = self.self_similarity(
            shared_fine_tuned,
            adjust_anisotropy=adjust_anisotropy,
            max_sample_size=max_sample_size,
        )
        ssv = self.self_similarity(
            shared_vanilla,
            adjust_anisotropy=adjust_anisotropy,
            anisotropy_value=vanilla_anisotropy,
            max_sample_size=max_sample_size,
        )

        ssc_scores = {}
        for word in tqdm(
            ssf.keys() & ssv.keys(),
            desc="******************************************** calculating self similarity changes...",
        ):
            ssc_scores[word] = round(ssf[word] - ssv[word], 3)

        return ssc_scores

    def domain_vs_general(
        self,
        fine_tuned_embeddings: dict[str, TermSummary],
        vanilla_embeddings: dict[str, TermSummary],
    ):

        shared_keys = fine_tuned_embeddings.keys() & vanilla_embeddings.keys()
        scores = {}

        for word in tqdm(
            shared_keys,
            desc="******************************************** calculating domain vs general shift...",
        ):
            ft_info = fine_tuned_embeddings[word]
            van_info = vanilla_embeddings[word]

            ft_embeds = ft_info.word_embeds  # (N_ft, H)
            van_embeds = van_info.word_embeds  # (N_van, H)

            # use the smaller count to keep pairs aligned by occurrence order
            N = min(ft_embeds.shape[0], van_embeds.shape[0])
            if N == 0:
                continue

            ft_embeds = ft_embeds[:N]
            van_embeds = van_embeds[:N]

            # cosine similarity per matched occurrence pair, then distance
            cos_sims = np.sum(ft_embeds * van_embeds, axis=1)  # (N,)
            scores[word] = float(round(np.mean(1 - cos_sims), 3))

        return scores

    def variance(
        self, word_embeddings: dict[str, TermSummary], max_sample_size=500, min_count=3
    ):
        variances_scores = {}

        for word, info in tqdm(
            word_embeddings.items(),
            desc="******************************************** calculating variances...",
        ):
            embeddings = info.word_embeds
            N = embeddings.shape[0]

            if N < min_count:
                continue

            # subsample if too many
            if N > max_sample_size:
                idx = self.rng.choice(N, max_sample_size, replace=False)
                embeddings = embeddings[idx]
                N = embeddings.shape[0]

            centroid = self._l2_normalize(embeddings.mean(axis=0))

            # squared distances to centroid
            sq_dists = np.sum((embeddings - centroid) ** 2, axis=1)

            # mean squared distance
            variance = np.mean(sq_dists)
            variances_scores[word] = float(round(variance, 3))

        return variances_scores

    def cross_context_stability(
        self,
        word_embeddings: dict[str, TermSummary],
        min_count: int = 2,
    ) -> dict[str, float]:
        """Score semantic stability of a term across sentence contexts

        For each pair of occurrences, computes the ratio of term embedding
        similarity to sentence embedding similarity.

        CORRELATION
        """
        scores = {}

        for word, info in tqdm(
            word_embeddings.items(),
            desc="******************************************** calculating cross-context stability...",
        ):
            N = info.word_embeds.shape[0]
            if N < max(min_count, 2):
                continue

            word_embeds = info.word_embeds
            sent_embeds = info.sent_embeds

            # pairwise cosine similarities via matrix multiply
            word_sim_matrix = word_embeds @ word_embeds.T  # (N, N)
            sent_sim_matrix = sent_embeds @ sent_embeds.T  # (N, N)

            # upper triangle only — avoid double-counting pairs and self-similarity on diagonal
            triu_idx = np.triu_indices(N, k=1)
            word_sims = word_sim_matrix[triu_idx]  # (num_pairs,)
            sent_sims = sent_sim_matrix[triu_idx]  # (num_pairs,)

            # only keep pairs where the sentence contexts are actually different
            # if sent_sim is near 1.0, the two sentences are nearly identical and
            # the pair tells us nothing about stability across varied contexts
            varied_mask = sent_sims < 0.95
            if not np.any(varied_mask):
                continue

            word_sims = word_sims[varied_mask]
            sent_sims = sent_sims[varied_mask]

            # handle edge case where all word_sims or sent_sims are identical (zero variance)
            # corrcoef returns nan in that case
            if len(word_sims) < 2:
                continue
            if np.std(word_sims) < 1e-6 or np.std(sent_sims) < 1e-6:
                continue

            correlation = np.corrcoef(word_sims, sent_sims)[0, 1]
            if np.isnan(correlation):
                continue
            scores[word] = round(float(1 - correlation), 3)

        return scores

    def ratioed_ccs(
        self,
        word_embeddings: dict[str, TermSummary],
        min_count: int = 2,
        adjust_anisotropy=True,
    ) -> dict[str, float]:
        """Score semantic stability of a term across sentence contexts

        For each pair of occurrences, computes the ratio of term embedding
        similarity to sentence embedding similarity.

        when sent_sim is low (very different contexts) and word_sim is high,
        the ratio is high
        """
        scores = {}

        for word, info in tqdm(
            word_embeddings.items(),
            desc="******************************************** calculating cross-context stability...",
        ):
            N = info.word_embeds.shape[0]
            if N < max(min_count, 2):
                continue

            word_embeds = info.word_embeds
            sent_embeds = info.sent_embeds

            # pairwise cosine similarities via matrix multiply
            word_sim_matrix = word_embeds @ word_embeds.T  # (N, N)
            sent_sim_matrix = sent_embeds @ sent_embeds.T  # (N, N)

            # upper triangle only — avoid double-counting pairs and self-similarity on diagonal
            triu_idx = np.triu_indices(N, k=1)
            word_sims = word_sim_matrix[triu_idx]  # (num_pairs,)
            sent_sims = sent_sim_matrix[triu_idx]  # (num_pairs,)
            if adjust_anisotropy and hasattr(self, "anisotropy_baseline"):
                word_sims = word_sims - self.anisotropy_baseline
                sent_sims = sent_sims - self.anisotropy_baseline

            # only keep pairs where the sentence contexts are actually different
            # if sent_sim is near 1.0, the two sentences are nearly identical and
            # the pair tells us nothing about stability across varied contexts
            varied_mask = sent_sims < 0.95
            if not np.any(varied_mask):
                continue

            word_sims = word_sims[varied_mask]
            sent_sims = sent_sims[varied_mask]

            sent_sims_clipped = np.clip(sent_sims, 0.05, None)
            ratios = np.clip(word_sims, 0, None) / sent_sims_clipped

            scores[word] = round(float(np.mean(ratios)), 3)

        return scores

    # ANCHOR - start of filters
    def stop_word_sim(
        self, candidate_embedding: dict[str, TermSummary], sw_embeddings: np.ndarray
    ):
        # NOTE - setup for this function --> currently not used but placed here just in case
        """if os.path.exists(self.stop_words_path):
            with open(self.stop_words_path, encoding="utf-8") as f:
                stop_words = set(f.read().split(","))
        else:
            raise FileNotFoundError("!!!!!!!!!! stop word file could not be opened.")

        sw_embeddings = self._create_embeddings(list(stop_words))

        filtered_candidates = self.score_function(
            self.stopword_distance_score,
            threshold,
            term_candidates,
            sw_embeddings,
        )"""

        stop_scores = {}
        for word, info in tqdm(
            candidate_embedding.items(),
            desc="******************************************** calculating stop word similarity...",
        ):
            word_embeds = info.word_embeds

            # calculate cosine similarity between the stop word embeddings
            # and all the contextualized word embedding
            cos_sims = word_embeds @ sw_embeddings.T
            stop_scores[word] = float(round(cos_sims.mean(), 3))

        return stop_scores

    def embed_setup(
        self,
        domain="corp",
        gram_type="uni",  # "uni" or "n"
        use_cache=True,
        model_name: str | None = None,
        candidates_dict: dict | None = None,
        update_sent_cache=False,
    ):
        print(
            f"******************************************** setting up {gram_type} embedding data..."
        )
        # ex. cache_path="cache_corp_uni.npz",
        cache = EmbeddingCache(cache_domain=domain, gram=gram_type)

        if use_cache and os.path.exists(cache.path):
            print("******************************************** loading from cache...")
            term_candidates, sentence_cache, orig_candidates = cache.load_cache()
            if update_sent_cache:
                self.sentence_cache = sentence_cache
            print(f"##### number of candidates: {len(orig_candidates)}")
            print(
                f"##### number of candidates after word embeddings: {len(term_candidates)}"
            )
        else:
            print("ERROR: no cache found! getting data...")
            if not candidates_dict:
                orig_candidates = self.get_unigram_cands()
            else:
                orig_candidates = candidates_dict

            if not orig_candidates:
                raise ValueError("ERROR: error with candidate extraction.")

            print(f"##### number candidates: {len(orig_candidates)}")

            # _encode will update self.sentence_cache if true
            encoded = self._encode(orig_candidates, model_name, update_sent_cache)
            print(f"##### number after encoding: {len(encoded)}")

            term_candidates = self._create_word_embeddings(encoded)
            print(f"##### number after word embeddings: {len(term_candidates)}")

            print("******************************************** caching...")
            if use_cache:
                cache.save_cache(term_candidates, self.sentence_cache, orig_candidates)

        return term_candidates, orig_candidates

    def extract_unigrams(
        self,
        domain="corp",
        gram_type="uni",
        use_cache=True,
    ):
        # FIXME - COMPLTETE ONCE FILTER COMBINATION IS DECIDED
        term_candidates, orig_candidates = self.embed_setup(
            domain=domain,
            gram_type=gram_type,
            use_cache=use_cache,
        )
        save_set_to_csv(self.error_terms, "error_terms.csv")

        pass
