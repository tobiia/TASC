import logging
import numpy as np
from lsc.representation.models import TermSummary
from lsc.representation.embedding_creator import create_embedding_creator
from lsc.config import PROJECT_ROOT, CACHE_DIR
from lsc.utils import save_set_to_csv

logger = logging.getLogger(__name__)


class EmbeddingCache:
    """Initialize WordCache.

    Args:
        corpora_label: Name prefix for cache files
        model_name: Huggingface name for the desired model
        layer: Hidden layer to extract token embeddings (or None)
    """

    def __init__(self, corpora_label, model_name, layer=None):
        layer_str = f"_L{layer}" if layer is not None else ""
        # remove "/" from file name to avoid errors
        safe_name = model_name.replace("/", "_")
        self.path = CACHE_DIR / f"{corpora_label}_{safe_name}{layer_str}.npz"

    def save_cache(self, term_candidates, sentence_cache, lemma_sentences):

        cand_keys = list(term_candidates.keys())

        word_embeds_flat, word_offsets = self._flatten_word_embeds_all(
            term_candidates, cand_keys
        )

        sent_embeds_flat, sent_offsets = self._flatten_sent_embeds(
            term_candidates, cand_keys
        )

        cache_words, cache_offsets, cache_embeds_flat = self._flatten_sentence_cache(
            sentence_cache
        )

        ls_keys, ls_values_flat, ls_offsets = self._flatten_lemma_sentences(
            lemma_sentences
        )

        self._save_common(
            self.path,
            cand_keys,
            word_embeds_flat=word_embeds_flat,
            word_offsets=word_offsets,
            sent_embeds_flat=sent_embeds_flat,
            sent_offsets=sent_offsets,
            cache_words=cache_words,
            cache_offsets=cache_offsets,
            cache_embeds_flat=cache_embeds_flat,
            ls_keys=ls_keys,
            ls_values_flat=ls_values_flat,
            ls_offsets=ls_offsets,
        )

    def load_cache(self):

        data = self._load_common(self.path)

        term_candidates = {}

        for i, cand in enumerate(data["candidates"]):
            w_start = data["word_offsets"][i]
            w_end = data["word_offsets"][i + 1]

            s_start = data["sent_offsets"][i]
            s_end = data["sent_offsets"][i + 1]

            term_candidates[cand] = TermSummary(
                word_embeds=data["word_embeds_flat"][w_start:w_end],
                sent_embeds=data["sent_embeds_flat"][s_start:s_end],
            )

        sentence_cache = self._reconstruct_sentence_cache(
            data["cache_words"],
            data["cache_offsets"],
            data["cache_embeds_flat"],
        )

        lemma_sentences = self._reconstruct_lemma_sentences(
            data["ls_keys"],
            data["ls_values_flat"],
            data["ls_offsets"],
        )

        return term_candidates, sentence_cache, lemma_sentences

    def _save_common(self, path, candidates, **arrays):
        np.savez_compressed(
            path,
            candidate_list=np.array(candidates, dtype=object),
            **arrays,
        )

    def _load_common(self, path):
        data = np.load(path, allow_pickle=True)
        result = {k: data[k] for k in data.files}
        result["candidates"] = result.pop("candidate_list")
        return result

    def _flatten_sent_embeds(self, term_candidates, candidates):
        # list of ndarrays = embeddings for each sentence stacked
        sent_embeds_flat = []
        sent_offsets = [0]

        for c in candidates:
            # ndarray of the sentence embeddings
            se = term_candidates[c].sent_embeds
            sent_embeds_flat.append(se)
            # ex. word_offsets = [0, 5, 8]
            # A (5, 768) -> start = 0, end = 5 -> sent_embeds_flat[0:5]
            # B (3, 768) -> start = 5, end = 8 -> sent_embeds_flat[5:8]
            sent_offsets.append(sent_offsets[-1] + len(se))
        # sent_embeds_flat (8, 768) after vstack
        return (
            np.vstack(sent_embeds_flat) if sent_embeds_flat else np.empty((0, 0))
        ), np.array(sent_offsets)

    def _flatten_word_embeds_all(self, term_candidates, candidates):
        word_embeds_flat = []
        word_offsets = [0]

        for c in candidates:
            # the different attribute name is the only reason this exists
            # otherwise works the same as above
            we = term_candidates[c].word_embeds
            word_embeds_flat.append(we)
            word_offsets.append(word_offsets[-1] + len(we))

        return (
            np.vstack(word_embeds_flat) if word_embeds_flat else np.empty((0, 0))
        ), np.array(word_offsets)

    # storing the embeddings for each unique sentence
    def _flatten_sentence_cache(self, sentence_cache):
        cache_words = []
        cache_offsets = [0]
        cache_embeds_flat = []

        # cache keys are integers
        for idx in sorted(sentence_cache.keys(), key=int):
            words, word_embeds = sentence_cache[idx]

            cache_words.extend(words)
            cache_embeds_flat.append(word_embeds)
            cache_offsets.append(cache_offsets[-1] + len(words))

        return (
            np.array(cache_words, dtype=object),
            np.array(cache_offsets),
            np.vstack(cache_embeds_flat) if cache_embeds_flat else np.empty((0, 0)),
        )

    def _flatten_lemma_sentences(self, candidates):
        keys = list(candidates.keys())
        values_flat = []
        offsets = [0]

        for k in keys:
            sents = candidates[k]
            values_flat.extend(sents)
            offsets.append(offsets[-1] + len(sents))

        return (
            np.array(keys, dtype=object),
            np.array(values_flat, dtype=object),
            np.array(offsets),
        )

    def _reconstruct_lemma_sentences(self, keys, values_flat, offsets):
        # short form version of the same to reconstruct word/sent embeds
        return {
            keys[i]: list(values_flat[offsets[i] : offsets[i + 1]])
            for i in range(len(keys))
        }

    def _reconstruct_sentence_cache(self, words, offsets, word_embeds_flat):
        sentence_cache = {}

        for i in range(len(offsets) - 1):
            start = offsets[i]
            end = offsets[i + 1]

            # i = idx of sentence
            sentence_cache[i] = (
                # corresponding word and word embeddings
                list(words[start:end]),
                word_embeds_flat[start:end],
            )

        return sentence_cache


def _load_or_compute(
    cache: "EmbeddingCache", corpus: dict, model_name: str, layer, corpora_label: str
):
    """Load cached embeddings or compute if not available.

    Args:
        cache: EmbeddingCache instance
        corpus: dict mapping word -> list of sentences
        model_name: Huggingface name for the desired model
        layer: Hidden layer to extract token embeddings (or None)
        corpora_label: Name prefix for cache files

    Returns:
        (term_candidates, sentence_cache, lemma_sentences)
    """
    if cache.path.exists():
        logger.info(f"Loading {corpora_label} embeddings from cache...")
        term_candidates, sentence_cache, lemma_sentences = cache.load_cache()
        logger.info(
            f"({corpora_label}): {len(lemma_sentences)} initial, {len(term_candidates)} after embeddings"
        )
    else:
        logger.info(f"Computing embeddings for {corpora_label}...")
        embedding_creator = create_embedding_creator(
            corpus, model_name=model_name, token_embedding_layer=layer
        )
        term_candidates, sentence_cache, lemma_sentences = (
            embedding_creator.create_embeddings()
        )
        if embedding_creator.error_terms:
            err_path = PROJECT_ROOT / f"error_terms_{corpora_label}.csv"
            logger.warning(
                f"Saving {len(embedding_creator.error_terms)} error terms to {err_path}"
            )
            save_set_to_csv(embedding_creator.error_terms, err_path)
        logger.info(f"Caching {corpora_label} embeddings...")
        cache.save_cache(term_candidates, sentence_cache, lemma_sentences)
    return term_candidates, sentence_cache, lemma_sentences


def run_cache(
    x_corpus: dict,
    y_corpus: dict,
    corpora_label: str,
    model_name: str,
    layer: int | None = None,
):
    """Load or compute embeddings for two corpora.

    Args:
        x_corpus: dict mapping word -> list of sentences (corpus 1)
        y_corpus: dict mapping word -> list of sentences (corpus 2)
        corpora_label: Name prefix for cache files
        model_name: Huggingface name for the desired model
        layer: Hidden layer to extract token embeddings (or None)

    Returns:
        ((x_embeds, x_cache, x_sentences), (y_embeds, y_cache, y_sentences)):
        - x_embeds, y_embeds: dicts mapping lemma -> TermSummary
        - x_cache, y_cache: sentence cache dicts
        - x_sentences, y_sentences: lemma -> sentences dicts
    """
    logger.info(f"Processing embeddings: {corpora_label}")

    cache_x = EmbeddingCache(f"{corpora_label}_c1", model_name, layer)
    cache_y = EmbeddingCache(f"{corpora_label}_c2", model_name, layer)

    x_result = _load_or_compute(
        cache_x, x_corpus, model_name, layer, f"{corpora_label}_c1"
    )
    y_result = _load_or_compute(
        cache_y, y_corpus, model_name, layer, f"{corpora_label}_c2"
    )

    return x_result, y_result
