import logging
import numpy as np

from .word_extractor import WordExtractor
from ..config import CACHE_DIR

logger = logging.getLogger(__name__)


class WordCache:
    def __init__(self, cache_domain):
        """Initialize WordCache.

        Args:
            cache_domain: Name prefix for cache files so
                the cache knows which to save to and load
        """
        self.path = CACHE_DIR / f"{cache_domain}_words.npz"

    def exists(self) -> bool:
        return self.path.exists()

    def save(self, word_map: dict[str, list[str]]):
        keys = list(word_map.keys())
        values_flat = []
        offsets = [0]

        for k in keys:
            sents = word_map[k]
            values_flat.extend(sents)
            offsets.append(offsets[-1] + len(sents))

        np.savez_compressed(
            self.path,
            keys=np.array(keys, dtype=object),
            values_flat=np.array(values_flat, dtype=object),
            offsets=np.array(offsets),
        )

    def load(self) -> dict[str, list[str]]:
        data = np.load(self.path, allow_pickle=True)
        keys = data["keys"]
        values_flat = data["values_flat"]
        offsets = data["offsets"]

        return {
            keys[i]: list(values_flat[offsets[i] : offsets[i + 1]])
            for i in range(len(keys))
        }


def _load_or_compute(
    corpus_path: str,
    cache: "WordCache",
    terms: list[str] | None = None,
):
    """Load cached embeddings or compute if not available.

    Args:
        corpus_path: Path to corpus
        cache: WordCache instance
        terms: Optional list of terms to bypass full corpora extraction

    Returns:
        corpus: dict mapping word -> list of sentences
    """
    if cache.exists():
        logger.info("Loading cached words for corpus...")
        corpus = cache.load()
    else:
        extractor = WordExtractor(corpus_path)
        if terms is not None:
            logger.info(f"Targeted extraction of {len(terms)} terms from corpus...")
            corpus = extractor.targeted_extraction(terms)
        else:
            logger.info("Extracting words from corpus...")
            corpus = extractor.unigram_extraction()
        if not corpus:
            logger.warning("No words extracted from corpus")
        logger.info("Caching corpus words...")
        cache.save(corpus)

    return corpus


def run_cache(
    corpus1_path,
    corpus2_path,
    cache_domain,
    terms: list[str] | None = None,
):
    """Extract and cache words from two corpora, returning shared words.

    Args:
        corpus1_path: Path to first corpus directory
        corpus2_path: Path to second corpus directory
        cache_domain: Name prefix for cache files
        terms: Optional list of terms to bypass full corpora extraction
            and search for using fast string matching. Cache is suffixed
            with '_targeted') to avoid confusion

    Returns:
        (shared_corpus1, shared_corpus2): dicts mapping word -> list of sentences
    """
    if terms is not None:
        c1_key = f"{cache_domain}_c1_targeted"
        c2_key = f"{cache_domain}_c2_targeted"
    else:
        c1_key = f"{cache_domain}_c1"
        c2_key = f"{cache_domain}_c2"

    cache1 = WordCache(c1_key)
    cache2 = WordCache(c2_key)

    logger.info(
        f"Processing corpora: {cache_domain}" + (" (targeted)" if terms else "")
    )

    logger.info("Searching for corpus1 cached words...")
    corpus1 = _load_or_compute(corpus1_path, cache1, terms)
    logger.info("Searching for corpus2 cached words...")
    corpus2 = _load_or_compute(corpus2_path, cache2, terms)

    shared_words = corpus1.keys() & corpus2.keys()
    logger.info(f"Found {len(shared_words)} shared words between corpora")

    shared_corpus1 = {k: corpus1[k] for k in shared_words}
    shared_corpus2 = {k: corpus2[k] for k in shared_words}

    return shared_corpus1, shared_corpus2
