import logging
import numpy as np
import spacy
from tqdm import tqdm
import torch
from transformers import AutoModel, AutoTokenizer
from .models import TermSummary

from transformers.utils import logging as hf_logging

hf_logging.set_verbosity_error()

try:
    from WordTransformer import WordTransformer, InputExample

    _WORD_TRANSFORMER_AVAILABLE = True
except ImportError:
    _WORD_TRANSFORMER_AVAILABLE = False

logger = logging.getLogger(__name__)

"""Pipeline for creating token-level contextual word embeddings

Implements the word occurrence representation step of the
lexical shift change workflow. Word embeddings are reconstructed
from token embeddings.

Surface forms (e.g. "running", "ran") are input in their original
forms, but results are grouped under the lemma so embeddings
across all forms are combined under one canonical term.

Typical usage example:

  embedding_creator = create_embedding_creator(corpus, model_name)
  embeddings = embedding_creator.create_embeddings()
"""


# SentencePiece-based models need a prefix space so that the first word
# is tokenized identically to mid-sentence words, ensuring word_ids()
# alignment is consistent regardless of position.
MODELS_NEEDING_PREFIX_SPACE = {
    "roberta",
    "FacebookAI/xlm-roberta-base",
    "xlm-roberta-base",
}

# Model name substrings that indicate XL-Lexeme, which uses the
# WordTransformer library and a different inference path entirely.
XL_LEXEME_NAMES = {
    "xl-lexeme",
    "pierluigic/xl-lexeme",
}


class BaseEmbeddingCreator:
    """Shared base for all embedding creator variants.

    Provides common initialisation (corpus validation, spaCy, rng),
    math helpers, and the full create_embeddings() orchestration loop.
    Subclasses implement _encode_sentences() and _build_term_embeddings().
    """

    def __init__(
        self,
        corpus: dict,
        model_name: str,
        token_embedding_layer: int | None = None,
        max_seq_length: int = 256,
        batch_size: int = 32,
        rng_seed: int = 267135941556543938173580506427407010431,
    ):
        """Initialise shared state.

        Args:
            corpus: dict mapping word -> list of sentences
            model_name: HuggingFace model name (used by subclasses)
            token_embedding_layer: Which hidden layer to extract (None = last layer)
            max_seq_length: Max tokens per sentence
            batch_size: Batch size for inference
            rng_seed: Random seed for reproducibility
        """
        if not corpus:
            raise ValueError("corpus cannot be empty")

        self.corpus = corpus
        self.model_name = model_name
        self.token_embedding_layer = token_embedding_layer
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size
        self.error_terms = set()  # collects any terms dropped because of errors

        logger.info(f"Initializing with {len(corpus)} words")

        try:
            self.model_nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
        except Exception as e:
            logger.error(f"Failed to load spaCy model: {e}")
            raise

        self.rng = np.random.default_rng(rng_seed)

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _lemmatize_term(self, term: str) -> str:
        return " ".join([token.lemma_ for token in self.model_nlp(term)])

    def _l2_normalize(self, x: np.ndarray) -> np.ndarray:
        return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-9)

    def _mean_pooling(self, model_output, attention_mask) -> torch.Tensor:
        token_embeddings = model_output[0]
        mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * mask, 1) / torch.clamp(
            mask.sum(1), min=1e-9
        )

    def compute_anisotropy(self, embeddings, n_samples: int = 1000) -> float:
        """Compute anisotropy baseline.

        Calculate anisotropy baseline as mean off-diagonal cosine similarity
        over randomly sampled embeddings.

        Returns:
            float: Anisotropy baseline (mean cosine similarity of random pairs)
        """
        embeddings = np.asarray(embeddings)
        n_samples = min(n_samples, embeddings.shape[0])
        if n_samples < 2:
            return 0.0

        sample = embeddings[
            self.rng.choice(embeddings.shape[0], size=n_samples, replace=False)
        ]
        sample = sample / (np.linalg.norm(sample, axis=-1, keepdims=True) + 1e-9)
        sim = sample @ sample.T
        return (sim.sum() - np.trace(sim)) / (sim.shape[0] * (sim.shape[0] - 1))

    # ------------------------------------------------------------------
    # Interface for subclasses
    # ------------------------------------------------------------------

    def _encode_sentences(self, sentences: list[str]) -> tuple[np.ndarray, dict]:
        raise NotImplementedError

    def _build_term_embeddings(
        self,
        candidates: dict,
        sentence_embeddings: np.ndarray,
        sentence_to_idx: dict,
        sentence_cache: dict,
    ) -> tuple[dict[str, TermSummary], dict[str, list[str]]]:
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Shared orchestration
    # ------------------------------------------------------------------

    def create_embeddings(
        self,
    ) -> tuple[dict[str, TermSummary], dict, dict[str, list[str]]]:
        """Compute embeddings for all words in corpus.

        Returns:
            (term_candidates, sentence_cache, lemma_sentences):
            - term_candidates: dict mapping lemma -> TermSummary
            - sentence_cache: dict mapping sentence_idx -> (words, embeddings),
              empty for XLLexemeEmbeddingCreator
            - lemma_sentences: dict mapping lemma -> list of sentence strings
        """
        if not self.corpus:
            raise ValueError("Corpus is empty")

        logger.info(f"Initial candidates: {len(self.corpus)}")

        unique_sentences = list(
            dict.fromkeys(s for sents in self.corpus.values() for s in sents)
        )
        logger.info(f"Unique sentences: {len(unique_sentences)}")

        sentence_to_idx = {s: i for i, s in enumerate(unique_sentences)}

        logger.info("Encoding sentences...")
        sentence_embeddings, sentence_cache = self._encode_sentences(unique_sentences)

        logger.info("Building word embeddings...")
        term_candidates, lemma_sentences = self._build_term_embeddings(
            self.corpus, sentence_embeddings, sentence_to_idx, sentence_cache
        )
        logger.info(f"Terms after word embedding creation: {len(term_candidates)}")

        if self.error_terms:
            logger.warning(
                f"{len(self.error_terms)} terms had errors and weren't embedded"
            )

        return term_candidates, sentence_cache, lemma_sentences


class StandardEmbeddingCreator(BaseEmbeddingCreator):
    """Embedding creator for standard HuggingFace transformer models.

    Handles XLM-RoBERTa, all-mpnet-base-v2, all-MiniLM, and any other
    AutoModel-compatible model. Word embeddings are reconstructed from
    token embeddings via word_ids() mean pooling.
    """

    def __init__(self, corpus: dict, model_name: str, **kwargs):
        super().__init__(corpus, model_name, **kwargs)

        try:
            logger.info(f"Loading tokenizer: {model_name}")
            # SentencePiece models (e.g. XLM-RoBERTa) need add_prefix_space=True
            # so the first token of a sentence is tokenized consistently with
            # mid-sentence positions, keeping word_ids() alignment correct.
            needs_prefix = any(
                m in model_name.lower() for m in MODELS_NEEDING_PREFIX_SPACE
            )
            tokenizer_kwargs = {"add_prefix_space": True} if needs_prefix else {}
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, **tokenizer_kwargs
            )
            logger.info(f"Loading model: {model_name}")
            self.model = AutoModel.from_pretrained(
                model_name,
                output_hidden_states=self.token_embedding_layer is not None,
            )
            self.model.eval()
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise

    def _word_embeddings_from_tokens(
        self, encoding, token_embeds: np.ndarray
    ) -> tuple[list[str], np.ndarray]:
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
            desc="*** embedding batches...",
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

            # sentence embeddings always use last_hidden_state regardless of
            # token_embedding_layer
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
                # re-tokenize individually so word_ids() indices are not
                # offset by batch padding
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
        accumulators = {}

        for surface_form, sents in tqdm(
            candidates.items(), desc="*** building word embeddings..."
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

        results, lemma_sentences = {}, {}
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


class XLLexemeEmbeddingCreator(BaseEmbeddingCreator):
    """Embedding creator for XL-Lexeme via the WordTransformer library.

    XL-Lexeme is fine-tuned for Word-in-Context tasks and produces a single
    contextualised word embedding per (sentence, character span) pair directly,
    without exposing token-level hidden states. This means:

    - token_embedding_layer has no effect and a warning is logged if set
    - sentence_cache is always empty (no reusable token cache is built)
    - sent_embeds in TermSummary reuse the word embedding, keeping the
      output structure consistent with StandardEmbeddingCreator
    """

    def __init__(self, corpus: dict, model_name: str, **kwargs):
        if kwargs.get("token_embedding_layer") is not None:
            logger.warning(
                "token_embedding_layer is not supported for XL-Lexeme"
                "(WordTransformer does not expose hidden states)."
                "The model's default output will be used."
            )
        super().__init__(corpus, model_name, **kwargs)

        if not _WORD_TRANSFORMER_AVAILABLE:
            raise ImportError(
                "WordTransformer is required for XL-Lexeme."
                "Install it at: https://github.com/pierluigic/xl-lexeme"
            )
        try:
            logger.info(f"Loading XL-Lexeme model via WordTransformer: {model_name}")
            self.model = WordTransformer(model_name)
            logger.info("XL-Lexeme model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load XL-Lexeme model {model_name}: {e}")
            raise

    def _encode_sentences(self, sentences: list[str]) -> tuple[np.ndarray, dict]:
        """XL-Lexeme does not support standalone sentence encoding.

        Word embeddings are computed per (sentence, span) pair in
        _build_term_embeddings. Returns empty placeholders so that
        create_embeddings() can call both methods uniformly.
        """
        return np.empty((0,)), {}

    def _build_term_embeddings(
        self,
        candidates: dict,
        sentence_embeddings: np.ndarray,
        sentence_to_idx: dict,
        sentence_cache: dict,
    ) -> tuple[dict[str, TermSummary], dict[str, list[str]]]:
        """Collect contextualized word embeddings for each candidate term.

        Locates each surface form in the sentence via character offset and
        calls WordTransformer.encode() with the span. The word embedding is
        reused as sent_embeds to keep TermSummary structurally consistent
        with StandardEmbeddingCreator.

        Returns:
            results: {lemma: TermSummary} with word and sentence embeddings
            lemma_sentences: {lemma: [original sentence strings]} for the API
        """
        # Collect all InputExamples across every term so encode() can be
        # called in batches rather than one example at a time.
        examples: list = []
        meta: list[tuple[str, str]] = []  # (lemma, sent) parallel to examples

        for surface_form, sents in candidates.items():
            lemma = self._lemmatize_term(surface_form)
            surface_lower = surface_form.lower()
            for sent in sents:
                start = sent.find(surface_lower)
                if start == -1:
                    continue
                span = (start, start + len(surface_lower))
                examples.append(InputExample(texts=sent, positions=span))  # type: ignore[arg-type]
                meta.append((lemma, sent))

        accumulators: dict = {}

        embeds = self.model.encode(
            examples, batch_size=self.batch_size, show_progress_bar=True
        )
        for i, word_embed in enumerate(embeds):
            lemma, sent = meta[i]
            if lemma not in accumulators:
                accumulators[lemma] = {
                    "word_embeds": [],
                    "sent_embeds": [],
                    "sentences": [],
                }
            accumulators[lemma]["word_embeds"].append(word_embed)
            # XL-Lexeme has no independent sentence embedding;
            # reuse the word embedding so TermSummary stays consistent.
            accumulators[lemma]["sent_embeds"].append(word_embed)
            accumulators[lemma]["sentences"].append(sent)

        results, lemma_sentences = {}, {}
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


def create_embedding_creator(
    corpus: dict,
    model_name: str = "sentence-transformers/all-mpnet-base-v2",
    token_embedding_layer: int | None = None,
    max_seq_length: int = 256,
    batch_size: int = 64,
    rng_seed: int = 267135941556543938173580506427407010431,
) -> BaseEmbeddingCreator:
    """Factory — returns the correct EmbeddingCreator subclass for model_name.

    Detects XL-Lexeme by model name and returns XLLexemeEmbeddingCreator;
    all other models return StandardEmbeddingCreator. The returned object
    exposes the same create_embeddings() interface regardless of model.

    Args:
        corpus: dict mapping word -> list of sentences
        model_name: HuggingFace model name or path
        token_embedding_layer: Which hidden layer to extract (None = last layer).
            Has no effect for XL-Lexeme; a warning will be logged if set.
        max_seq_length: Max tokens per sentence
        batch_size: Batch size for inference
        rng_seed: Random seed for reproducibility

    Returns:
        StandardEmbeddingCreator or XLLexemeEmbeddingCreator
    """
    kwargs = dict(
        token_embedding_layer=token_embedding_layer,
        max_seq_length=max_seq_length,
        batch_size=batch_size,
        rng_seed=rng_seed,
    )
    if any(n in model_name.lower() for n in XL_LEXEME_NAMES):
        return XLLexemeEmbeddingCreator(corpus, model_name, **kwargs)
    return StandardEmbeddingCreator(corpus, model_name, **kwargs)
