import csv
import os
import numpy as np
from tqdm import tqdm
import torch
from transformers import AutoModel, AutoTokenizer
from typing import Tuple
from models import CandidateSummary, TermEmbeddings, TermSummary

from split_candidate_extractor import CandidateExtractor
from term_cache import EmbeddingCache

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
        max_seq_length=384,
        batch_size=64,
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        topic_threshold=0.4,  # NOTE from paper
        context_diff_threshold=0.3,
        self_sim_threshold=0.3,  # NOTE from paper
        ssc_threshold=0,  # NOTE from paper, pos v neg
        var_threshold=0.5,
        ccs_threshold=1.0,
        stop_threshold=1.0,
    ):
        self.corpus_path = corpus_path
        self.stop_words_path = stop_words_path
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()

        self.topic_threshold = topic_threshold
        self.context_diff_threshold = context_diff_threshold
        self.self_sim_threshold = self_sim_threshold
        self.ssc_threshold = ssc_threshold
        self.var_threshold = var_threshold
        self.ccs_threshold = ccs_threshold
        self.stop_threshold = stop_threshold

        special_tokens = set(self.tokenizer.special_tokens_map.values())
        special_tokens.update(self.tokenizer.added_tokens_encoder.keys())
        self.special_tokens = list(special_tokens)

        # maps sentence index (int) to token ids and token embeddings
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
        token_embeddings = model_output[0]  # input ids
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

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
        sentence_embeddings = self._create_embeddings(
            unique_sentences,
            tokenizer=tokenizer,
            model=model,
            sentence_cache=True,
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
        contextualized_mode: str = "all",
    ) -> dict[str, TermEmbeddings | TermSummary]:
        """Create word embeddings for all candidates

        Word embeddings are the mean average of the token embeddings
        comprising them.

        Args:
            contextualized_mode: whether a single contextualized or multiple word
                embeddings for each word are created

        Returns:
            Dict with candidates as keys and TermEmbeddings OR TermSummarys
            as values. TermEmbedding = single contextualized word embedding,
            TermSummary = word embeddings for every occurrence of the word
        """

        candidate_embeddings = {}
        for candidate, info in tqdm(
            encoded_candidates.items(),
            desc="******************************************** building word embeddings...",
        ):
            try:
                _, emb = self._token_to_word(candidate, info, contextualized_mode)
                if emb is not None:
                    candidate_embeddings[candidate] = emb
                else:
                    self.error_terms.add(candidate)
            except Exception:
                self.error_terms.add(candidate)
        return candidate_embeddings

    def _token_to_word(
        self, candidate: str, info: CandidateSummary, contextualized_mode: str
    ):
        """Collects and returns all constructed word embeddings for a candidate

        Matches word embeddings with the given single candidate.

        Args:
            contextualized_mode: If True, function creates a single contextualized
            word embedding. If False, returns multiple word embeddings for each
            word occurrence

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

        # each index maps to the tokens and token embeddings of a sentence
        # so we're looping over all sentences containing the candidate

        for idx in info.sentence_indices:
            words, word_embeds = self.sentence_cache[idx]
            for i in range(len(words) - k + 1):
                if words[i : i + k] == candidate_subwords:
                    # add to list of all word embeddings for candidate
                    # if multi-word, the word embeddings of each are
                    # averaged to combine them
                    all_embeds.append(
                        np.mean(word_embeds[i : i + k], axis=0, keepdims=True)
                    )
                    break  # limit to 1 occurrence per sentence

        if not all_embeds:
            return candidate, None

        # stack the word embeddings of each occurrence of the candidate together
        all_embeds = np.vstack(all_embeds)

        # we only normalize the word embeddings at the VERY end to ensure each part of the word's magnitude contributes to the embedding for semantics
        if contextualized_mode == "mean":
            return candidate, TermEmbeddings(
                word_embed=self._l2_normalize(np.mean(all_embeds, axis=0)),
                sent_embeds=info.sent_embeds,
            )
        else:
            all_embeds = self._l2_normalize(all_embeds)
            return candidate, TermSummary(
                word_embeds=all_embeds, sent_embeds=info.sent_embeds
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
        max_sample_size=50,
    ):
        all_candidates = list(candidate_embeddings.keys())
        general_embeddings = self._create_embeddings(
            all_candidates
        )  # (n_candidates, H)

        cvg_scores = {}
        for i, candidate in tqdm(
            enumerate(candidate_embeddings),
            desc="******************************************** calculating context vs general scores...",
        ):

            general_emb = general_embeddings[i]
            context_embs = candidate_embeddings[
                candidate
            ].word_embeds  # (n_contexts, H)

            # subsample if too many
            N = context_embs.shape[0]
            if N > max_sample_size:
                idx = np.random.choice(N, max_sample_size, replace=False)
                context_embs = context_embs[idx]

            cos_sims = context_embs @ general_emb  # (n_contexts,)
            cvg_scores[candidate] = np.mean(1 - cos_sims)

        return cvg_scores

    def topic_score(
        self,
        candidate_embeddings: dict[str, TermSummary],
        # REVIEW change while testing here
        method="max",
        max_sample_size=50,
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
                idx = np.random.choice(N, max_sample_size, replace=False)
                X = X[idx]
                Y = Y[idx]
                N = X.shape[0]

            # cos sim of each sent-word pair
            cos_sim = np.sum(X * Y, axis=1)

            if method == "max":
                # t-extractor method
                score = cos_sim.max()
            elif method == "mean":
                score = cos_sim.mean()
            else:
                raise ValueError(f"Unknown method {method}")

            topic_scores[word] = float(score)

        return topic_scores

    # only for high frequency terms
    def self_similarity(
        self, word_embeddings: dict[str, TermSummary], max_sample_size=500, min_count=50
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

    def self_similarity_change(
        self,
        fine_tuned_embeddings: dict[str, TermSummary],
        vanilla_embeddings: dict[str, TermSummary],
        max_sample_size=5000,
    ):

        shared_candidates = {
            k: fine_tuned_embeddings[k]
            for k in fine_tuned_embeddings
            if k in vanilla_embeddings
        }

        ssf = self.self_similarity(shared_candidates, max_sample_size=max_sample_size)
        ssv = self.self_similarity(vanilla_embeddings, max_sample_size=max_sample_size)

        ssc_scores = {}
        for word in tqdm(
            ssf.keys(),
            desc="******************************************** calculating self similarity changes...",
        ):
            if word in ssv:
                ssc_scores[word] = round(ssf[word] - ssv[word], 3)

        return ssc_scores

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
                idx = np.random.choice(N, max_sample_size, replace=False)
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
        scores = {}

        for word, info in tqdm(
            word_embeddings.items(),
            desc="******************************************** calculating cross-context stability...",
        ):
            N = info.word_embeds.shape[0]
            if N < max(min_count, 2):
                continue

            word_embeds = info.word_embeds  # (N, D), already L2 normalized
            sent_embeds = info.sent_embeds  # (N, D), already L2 normalized

            # pairwise cosine similarities via matrix multiply (both already normalized)
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

            # ratio: how stable is the term relative to how different the contexts are
            # sent_sim is clipped away from zero to prevent division instability
            # when sent_sim is low (very different contexts) and word_sim is high,
            # the ratio is high — exactly the signal we want
            correlation = np.corrcoef(word_sims, sent_sims)[0, 1]

            # handle edge case where all word_sims or sent_sims are identical (zero variance)
            # corrcoef returns nan in that case
            if np.isnan(correlation):
                continue

            scores[word] = round(float(1 - correlation), 3)

        return scores

    def stopword_distance_score(
        self, candidate_embedding: dict[str, TermSummary], sw_embeddings: np.ndarray
    ):
        stop_scores = {}
        for word, info in tqdm(
            candidate_embedding.items(),
            desc="******************************************** calculating stop word distance...",
        ):
            word_embeds = info.word_embeds

            # calculate cosine similarity between the general word embeddings
            # and all their corresponding contextualized word embedding
            cos_sims = word_embeds @ sw_embeddings.T
            stop_scores[word] = float(round(cos_sims.mean(), 3))

        return stop_scores

    def embed_setup(
        self,
        domain="corp",
        contextualized_mode: str = "all",  # mean or all
        gram_type="uni",  # uni or n
        use_cache=True,
    ):
        print(
            f"******************************************** setting up {gram_type}gram embedding data..."
        )
        # ex. cache_path="cache_corp_all_uni.npz",
        cache = EmbeddingCache(
            cache_domain=domain, cache_context=contextualized_mode, gram=gram_type
        )

        if use_cache and os.path.exists(cache.path):
            print("******************************************** loading from cache...")
            term_candidates, self.sentence_cache, orig_candidates = cache.load_cache()
        else:
            print("!!!!!!!!!!!! no cache found! getting data...")
            if gram_type == "uni":
                orig_candidates = self.get_unigram_cands()
            else:
                raise ValueError("I haven't implemented this yet lol")

            print(f"##### number candidates: {len(orig_candidates)}")

            if not orig_candidates:
                raise ValueError(
                    "!!!!!!!!!!!!!! something went wrong with candidate extraction"
                )

            encoded = self._encode(orig_candidates)
            print(f"##### number after encoding: {len(encoded)}")

            term_candidates = self._create_word_embeddings(
                encoded, contextualized_mode=contextualized_mode
            )
            print(f"##### number after word embeddings: {len(term_candidates)}")

            print("******************************************** caching...")
            if use_cache:
                cache.save_cache(term_candidates, self.sentence_cache, orig_candidates)

        return term_candidates, orig_candidates

    # topic, context, selfsim, var, ccs
    def score_function(self, func, threshold, term_candidates, *args):
        filtered_candidates = {}
        scores = func(term_candidates, *args)  # type: ignore
        for word, info in term_candidates.items():
            score = scores.get(word)
            if score is None or scores[word] >= threshold:
                filtered_candidates[word] = info
        print(f"##### number after filter: {len(filtered_candidates)}")
        return filtered_candidates

    def uni_test_filter(
        self,
        func,
        thresholds,
        domain="corp",
        contextualized_mode="all",
        gram_type="uni",
        use_cache=True,
        compute_ssc=False,
        compute_stop=False,
    ):
        term_candidates, orig_candidates = self.embed_setup(
            domain=domain,
            contextualized_mode=contextualized_mode,
            gram_type=gram_type,
            use_cache=use_cache,
        )
        save_set_to_csv(self.error_terms, "error_terms.csv")

        for threshold in thresholds:

            if compute_ssc:
                vanilla_encoded = self._encode(
                    orig_candidates, model_name="nreimers/MiniLM-L6-H384-uncased"
                )

                vanilla_candidates = self._create_word_embeddings(
                    vanilla_encoded, contextualized_mode=contextualized_mode
                )

                filtered_candidates = self.score_function(
                    self.self_similarity_change,
                    threshold,
                    term_candidates,
                    vanilla_candidates,
                )

            elif compute_stop:
                if os.path.exists(self.stop_words_path):
                    with open(self.stop_words_path, encoding="utf-8") as f:
                        stop_words = set(f.read().split(","))
                else:
                    raise FileNotFoundError(
                        "!!!!!!!!!! stop word file could not be opened."
                    )

                sw_embeddings = self._create_embeddings(list(stop_words))

                filtered_candidates = self.score_function(
                    self.stopword_distance_score,
                    threshold,
                    term_candidates,
                    sw_embeddings,
                )

            else:
                filtered_candidates = self.score_function(
                    func, threshold, term_candidates
                )

        return filtered_candidates

    def extract_unigrams(
        self,
        domain="corp",
        contextualized_mode="all",
        gram_type="uni",
        use_cache=True,
    ):
        term_candidates, orig_candidates = self.embed_setup(
            domain=domain,
            contextualized_mode=contextualized_mode,
            gram_type=gram_type,
            use_cache=use_cache,
        )
        save_set_to_csv(self.error_terms, "error_terms.csv")

        pass
