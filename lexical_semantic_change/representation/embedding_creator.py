import csv
import numpy as np
import spacy
from tqdm import tqdm
import torch
from transformers import AutoModel, AutoTokenizer
from .models import TermSummary

from transformers.utils import logging as hf_logging

hf_logging.set_verbosity_error()

"""Pipeline for creating token-level contextual word embeddings

Implements the word occurrence representation step of the
lexical shift change workflow. Word embeddings are reconstructed
from token embeddings.

Surface forms (e.g. "running", "ran") are input in their original
forms, but results are grouped under the lemma so embeddings 
across all forms are combined under one canonical term.

Typical usage example:

  embedding_creator = EmbeddingCreator(corpus)
  embeddings = embedding_creator.create_embeddings()
"""


def save_set_to_csv(data_set, file_path):
    with open(file_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["term"])
        for item in sorted(data_set):
            writer.writerow([item])


MODELS_NEEDING_PREFIX_SPACE = {
    "roberta",
    "xl-lexeme",
}


class EmbeddingCreator:
    def __init__(
        self,
        corpus: dict,
        model_name: str = "sentence-transformers/all-mpnet-base-v2",
        token_embedding_layer: int | None = None,
        max_seq_length: int = 256,
        batch_size: int = 64,
        stop_words_path: str = "stop_words_en.txt",
        rng_seed: int = 267135941556543938173580506427407010431,
    ):
        self.corpus = corpus

        self.token_embedding_layer = token_embedding_layer
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size
        self.stop_words_path = stop_words_path

        # to ensure alignment for certain models
        needs_prefix = any(m in model_name.lower() for m in MODELS_NEEDING_PREFIX_SPACE)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, add_prefix_space=needs_prefix
        )
        self.model = AutoModel.from_pretrained(
            model_name, output_hidden_states=token_embedding_layer is not None
        )
        self.model.eval()

        self.model_nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])

        self.rng = np.random.default_rng(rng_seed)
        self.error_terms = set()  # set that collects any terms dropped b/c of errors

    def _lemmatize_term(self, term: str):
        return " ".join([token.lemma_ for token in self.model_nlp(term)])

    def _l2_normalize(self, x):
        return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-9)

    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * mask, 1) / torch.clamp(
            mask.sum(1), min=1e-9
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

        n_samples = min(n_samples, embeddings.shape[0])
        if n_samples < 2:
            return 0.0

        # random sample of indices
        sample = embeddings[
            self.rng.choice(embeddings.shape[0], size=n_samples, replace=False)
        ]
        sample = sample / (np.linalg.norm(sample, axis=-1, keepdims=True) + 1e-9)

        # cosine similarity matrix
        sim = sample @ sample.T

        # remove diagonal (self-similarity)
        return (sim.sum() - np.trace(sim)) / (sim.shape[0] * (sim.shape[0] - 1))

    def _word_embeddings_from_tokens(self, encoding, token_embeds: np.ndarray):
        word_ids = encoding.word_ids()
        groups: dict[int, list] = {}
        for tok_idx, word_idx in enumerate(word_ids):
            if word_idx is not None:
                groups.setdefault(word_idx, []).append(tok_idx)

        words, word_embeds = [], []
        for word_idx in sorted(groups):
            tok_indices = groups[word_idx]
            token_list = self.tokenizer.convert_ids_to_tokens(
                [encoding["input_ids"][0][t].item() for t in tok_indices]
            )
            words.append(
                self.tokenizer.convert_tokens_to_string(token_list).strip().lower()
            )
            word_embeds.append(np.mean(token_embeds[tok_indices], axis=0))

        return words, (
            np.vstack(word_embeds)
            if word_embeds
            else np.empty((0, token_embeds.shape[-1]))
        )

    def _encode_sentences(self, sentences: list[str]) -> tuple[np.ndarray, dict]:
        """Encode all sentences, returning sentence embeddings and a word
        embedding cache.

        Batches model forward passes for efficiency. Re-tokenizes each sentence
        individually to get reliable word_ids() for the cache because padding
        offsets the original ids. Sentence cache allows translation between
        original word-sentence dict and word + sentence ndarrays.

        Returns:
            sentence_embeddings: L2-normalized array of shape (N, hidden_size)
            sentence_cache: maps sentence index -> (words, word_embeds)
        """
        all_sent_embeds, sentence_cache = [], {}

        for batch_start in tqdm(
            range(0, len(sentences), self.batch_size),
            desc="****************** embedding batches...",
        ):
            batch = sentences[batch_start : batch_start + self.batch_size]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_seq_length,
                return_tensors="pt",
            )
            with torch.no_grad():
                output = self.model(**encoded)

            all_sent_embeds.append(
                self._mean_pooling(output, encoded["attention_mask"]).cpu().numpy()
            )

            tok_embeds = (
                (
                    output.hidden_states[self.token_embedding_layer]
                    if self.token_embedding_layer is not None
                    else output.last_hidden_state
                )
                .cpu()
                .numpy()
            )

            for i, sent in enumerate(batch):
                single = self.tokenizer(
                    sent,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.max_seq_length,
                )
                words, word_embeds = self._word_embeddings_from_tokens(
                    single, tok_embeds[i]
                )
                sentence_cache[batch_start + i] = (words, word_embeds)

        if not all_sent_embeds:
            return np.empty((0,)), sentence_cache
        return (
            self._l2_normalize(np.concatenate(all_sent_embeds, axis=0)),
            sentence_cache,
        )

    def _build_term_embeddings(
        self,
        candidates: dict,
        sentence_embeddings: np.ndarray,
        sentence_to_idx: dict,
        sentence_cache: dict,
    ) -> tuple[dict[str, TermSummary], dict[str, list[str]]]:
        """Collect contextualized word embeddings for each candidate term.

        Slides a window of width k over each sentence's word list to find
        the candidate SURFACE form. Results are grouped under the lemma,
        so "running", "ran", "runs" -> "run".

        Multi-word candidates are handled by averaging across the matched span.
        One match per sentence.

        Returns:
            results: {lemma: TermSummary} with word and sentence embeddings
            lemma_sentences: {lemma: [original sentence strings]} for the API
        """
        # accumulator keyed by lemma to merge surface forms
        accumulators = {}

        for surface_form, sents in tqdm(
            candidates.items(), desc="***************** building word embeddings..."
        ):
            lemma = self._lemmatize_term(surface_form)
            candidate_words = surface_form.split()
            k = len(candidate_words)

            if lemma not in accumulators:
                accumulators[lemma] = {
                    "word_embeds": [],
                    "sent_embeds": [],
                    "sentences": [],
                }

            for sent in sents:
                idx = sentence_to_idx.get(sent)
                if idx is None:
                    continue
                words, word_embeds = sentence_cache[idx]
                for i in range(len(words) - k + 1):
                    if words[i : i + k] == candidate_words:
                        accumulators[lemma]["word_embeds"].append(
                            np.mean(word_embeds[i : i + k], axis=0)
                        )
                        accumulators[lemma]["sent_embeds"].append(
                            sentence_embeddings[idx]
                        )
                        accumulators[lemma]["sentences"].append(sent)
                        break

        # build final TermSummary objects, dropping any lemma with no matches
        results = {}
        lemma_sentences = {}
        for lemma, acc in accumulators.items():
            if not acc["word_embeds"]:
                self.error_terms.add(lemma)
                continue
            results[lemma] = TermSummary(
                word_embeds=self._l2_normalize(np.vstack(acc["word_embeds"])),
                sent_embeds=np.vstack(acc["sent_embeds"]),
            )
            lemma_sentences[lemma] = acc["sentences"]

        return results, lemma_sentences

    def create_sent_embeddings(self):
        if not self.corpus:
            raise ValueError("ERROR: error with candidate extraction.")

        print(f"EMBEDDING CREATOR: initial number of words: {len(self.corpus)}")

        unique_sentences = list(
            dict.fromkeys(s for sents in self.corpus.values() for s in sents)
        )
        sentence_to_idx = {s: i for i, s in enumerate(unique_sentences)}

        sentence_embeddings, sentence_cache = self._encode_sentences(unique_sentences)

        # grouping by lemma
        accumulators: dict[str, dict] = {}
        for surface_form, sents in self.corpus.items():
            lemma = self._lemmatize_term(surface_form)
            if lemma not in accumulators:
                accumulators[lemma] = {"embeds": [], "sentences": []}
            for s in sents:
                if s in sentence_to_idx:
                    accumulators[lemma]["embeds"].append(
                        sentence_embeddings[sentence_to_idx[s]]
                    )
                    accumulators[lemma]["sentences"].append(s)

        results = {}
        lemma_sentences = {}
        for lemma, acc in accumulators.items():
            if not acc["embeds"]:
                self.error_terms.add(lemma)
                continue
            results[lemma] = TermSummary(sent_embeds=np.vstack(acc["embeds"]))
            lemma_sentences[lemma] = acc["sentences"]

        print(
            f"EMBEDDING CREATOR: candidates after sentence embeddings: {len(results)}"
        )

        return results, sentence_cache, lemma_sentences

    def create_embeddings(self):
        if not self.corpus:
            raise ValueError("ERROR: error with candidate extraction.")

        print(f"EMBEDDING CREATOR: initial candidates: {len(self.corpus)}")

        unique_sentences = list(
            dict.fromkeys(s for sents in self.corpus.values() for s in sents)
        )
        sentence_to_idx = {s: i for i, s in enumerate(unique_sentences)}

        sentence_embeddings, sentence_cache = self._encode_sentences(unique_sentences)

        term_candidates, lemma_sentences = self._build_term_embeddings(
            self.corpus, sentence_embeddings, sentence_to_idx, sentence_cache
        )
        print(
            f"EMBEDDING CREATOR: candidates after word embeddings: {len(term_candidates)}"
        )

        return term_candidates, sentence_cache, lemma_sentences
