import csv
import os
import numpy as np
from tqdm import tqdm
import torch
from transformers import AutoModel, AutoTokenizer
from representation.models import CandidateSummary, TermSummary
from representation.embed_cache import EmbeddingCache

# TODO - remove later
from transformers.utils import logging as hf_logging

hf_logging.set_verbosity_error()

"""Pipeline for creating token-level contextual word embeddings

Implements the word occurrence representation step of the
lexical shift change workflow. Word embeddings are
reconstructed from token embeddings.

Typical usage example:

  embedding_creator = EmbeddingCreator()
  embeddings = embedding_creator.encode(word_dict)
"""


def save_set_to_csv(data_set, file_path):
    with open(file_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["term"])
        for item in sorted(data_set):
            writer.writerow([item])


class EmbeddingCreator:
    def __init__(
        self,
        corpus: dict,
        stop_words_path: str = "stop_words_en.txt",
        max_seq_length: int = 256,
        batch_size: int = 64,
        subword_prefix: str = "##",
        model_name="sentence-transformers/all-mpnet-base-v2",
        # randomly gen via secrets.randbits(128)
        rng_seed=267135941556543938173580506427407010431,
    ):
        self.corpus = corpus
        self.stop_words_path = stop_words_path

        self.max_seq_length = max_seq_length
        self.batch_size = batch_size

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        self.subword_prefix = subword_prefix

        self.base_anisotropy = 0.0

        self.rng_seed = rng_seed
        self.rng = np.random.default_rng(rng_seed)

        # programmatically adding special tokens in case other models used
        special_tokens = set(self.tokenizer.special_tokens_map.values())
        special_tokens.update(self.tokenizer.added_tokens_encoder.keys())
        self.special_tokens = list(special_tokens)
        # REVIEW there is a return_offsets_mapping on HG tokenizers
        # can use to make word embed creation model-agnostic but it'll require
        # rewriting

        self.error_terms = set()  # set that collects any terms dropped b/c of errors

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
            return 0.0

        # random sample of indices
        idx = self.rng.choice(N, size=n_samples, replace=False)
        sample = embeddings[idx]

        # cosine similarity matrix
        sim = sample @ sample.T

        # remove diagonal (self-similarity)
        n = sim.shape[0]
        return (sim.sum() - np.trace(sim)) / (n * (n - 1))

    def _generate_embeddings(
        self,
        texts: list[str],
        update_sentence_cache: bool,
        sentence_to_idx: dict[str, int] | None = None,
        sentence_cache: dict | None = None,
    ) -> np.ndarray:
        """Creates mean-pooled sentence embeddings.

        Creates mean-pooled sentence embeddings of the given strings. Will use the class
        tokenizer and sentence embedding model unless another is given.

        Args:
            update_sentence_cache:
                If True then, for each text, its token IDs and token embeddings will be
                used to construct word embeddings for each word within. These are then
                stored in provided sentence_cache keyed by integer index (via sentence_to_idx)
                as a Tuple of (list[words], ndarray[word embeddings]). Requires
                sentence_to_idx to be provided. If contextualized word embeddings need
                to be constructed, this decreases complexity.

        Returns:
            Numpy array of all the embeddings stacked and L2 normalized
        """

        all_embeddings = []

        batch_iter = tqdm(
            range(0, len(texts), self.batch_size),
            desc="******************************************** embedding batches...",
        )
        for batch_start in batch_iter:
            batch = texts[batch_start : batch_start + self.batch_size]
            encoded_input = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_seq_length,
                return_tensors="pt",
            )

            with torch.no_grad():
                model_output = self.model(**encoded_input)

            # pool token embeddings to create sentence embeddings
            sent_embeds = self._mean_pooling(
                model_output, encoded_input["attention_mask"]
            )
            all_embeddings.append(sent_embeds.cpu().numpy())

            # reconstruct word embeddings from token embeddings and store
            # integer indices (via sentence_to_idx) are the keys to the cache
            if (
                update_sentence_cache
                and sentence_to_idx is not None
                and sentence_cache is not None
            ):
                batch_tok_embeds = model_output.last_hidden_state.cpu().numpy()
                for i, sent in enumerate(batch):
                    tokens = self.tokenizer.convert_ids_to_tokens(
                        encoded_input["input_ids"][i]
                    )
                    token_embeds = batch_tok_embeds[i]
                    symbols, symbol_embeds = self._reconstruct_words(
                        tokens, token_embeds
                    )
                    words, word_embeds = self._merge_hyphenated(
                        [s.lower() for s in symbols], symbol_embeds
                    )
                    sentence_cache[sentence_to_idx[sent]] = (
                        words,
                        word_embeds,
                    )

        return self._l2_normalize(np.concatenate(all_embeddings, axis=0))

    def _encode(
        self,
        candidates: dict[str, list[str]],
        sentence_cache: dict,
    ) -> dict[str, CandidateSummary]:
        """Encodes all candidate term occurrences and sentences

        Creates embeddings for each occurrences of each candidate term and the corresponding sentences. Used to create contextualized word embeddings.

        Returns:
            Dict with candidates as keys and CandidateSummary as values. CandidateSummary are Tuples containing a list of sentences and ndarray of sentence embeddings for each candidate.
        """

        tokenizer = self.tokenizer
        model = self.model

        all_sentences = [s for sents in candidates.values() for s in sents]
        unique_sentences = list(dict.fromkeys(all_sentences))
        # save where each sentence is within unique_sentences
        sentence_to_idx = {s: i for i, s in enumerate(unique_sentences)}

        # all embeddings stacked, axis 0 indices match unique_sentences
        # + sentence_to_idx
        sentence_embeddings = self._generate_embeddings(
            unique_sentences,
            update_sentence_cache=True,
            sentence_to_idx=sentence_to_idx,
            sentence_cache=sentence_cache,
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
        sentence_cache: dict,
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
                _, emb = self._token_to_word(
                    candidate, info, sentence_cache=sentence_cache
                )
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
        sentence_cache: dict,
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
            words, word_embeds = sentence_cache[idx]
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
            if t.startswith(self.subword_prefix):
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

    def create_embeddings(self):
        orig_candidates = self.corpus

        if not orig_candidates:
            raise ValueError("ERROR: error with candidate extraction.")

        print(
            f"EMBEDDING CREATOR: number of initial candidates: {len(orig_candidates)}"
        )

        sentence_cache = {}

        encoded = self._encode(orig_candidates, sentence_cache)
        print(f"EMBEDDING CREATOR: number of candidates after encoding: {len(encoded)}")

        term_candidates = self._create_word_embeddings(
            encoded,
            sentence_cache,
        )
        print(
            f"EMBEDDING CREATOR: number of candidates after word embeddings: {len(term_candidates)}"
        )

        return term_candidates, sentence_cache, orig_candidates
