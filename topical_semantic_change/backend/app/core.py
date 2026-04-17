from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from lexical_semantic_change.extraction.word_cache import run_cache as run_word_cache
from lexical_semantic_change.representation.embed_cache import run_cache
from .config import DATA_DIR, TERMS_FILE

CORPUS1 = str(DATA_DIR / "sample" / "corpus1")
CORPUS2 = str(DATA_DIR / "sample" / "corpus2")
CORPUS1_NAME = Path(CORPUS1).stem  # "corpus1"
CORPUS2_NAME = Path(CORPUS2).stem  # "corpus2"
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"


def _load_terms(path: Path):
    """Load terms from a CSV (looks for 'lemma' column, falls back to first column) or TXT file."""
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
        col = "lemma" if "lemma" in df.columns else df.columns[0]
        return df[col].dropna().str.lower().tolist()
    else:
        return [
            line.strip().lower()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]


def _validate_corpus_paths():
    """Validate that corpus directories exist."""
    for path, name in [(CORPUS1, "CORPUS1"), (CORPUS2, "CORPUS2")]:
        if not Path(path).exists():
            raise FileNotFoundError(
                f"{name} path does not exist: {path}\n"
                f"Expected corpus files at: {DATA_DIR}/sample/corpus1/*.txt and corpus2/*.txt"
            )


def load_data():
    """Extract shared words, compute embeddings, return data needed by the API.

    Returns:
        words: sorted list of words present in both corpora
        word_means: {word: (mean_embed_c1, mean_embed_c2)}
        word_occurrences: {word: [{"text": str, "date": corpus_name}, ...]}
        all_sentences: deduplicated list of all sentences (for topic modelling)
    """
    _validate_corpus_paths()

    terms: list[str] | None = None
    if TERMS_FILE is not None:
        terms_path = Path(TERMS_FILE)
        if not terms_path.exists():
            raise FileNotFoundError(f"TERMS_FILE not found: {TERMS_FILE}")
        terms = _load_terms(terms_path)

    try:
        x_words, y_words = run_word_cache(
            CORPUS1, CORPUS2, f"{CORPUS1_NAME}_{CORPUS2_NAME}", terms=terms
        )
    except Exception as e:
        raise RuntimeError(f"Failed to extract words from corpora: {e}") from e

    if not x_words or not y_words:
        raise ValueError(
            "Word extraction produced empty results. "
            "Check that corpus files contain valid text."
        )

    try:
        (x_embeds, _, x_lemma_sentences), (y_embeds, _, y_lemma_sentences) = run_cache(
            x_words, y_words, f"{CORPUS1_NAME}_{CORPUS2_NAME}", MODEL_NAME, layer=5
        )
    except Exception as e:
        raise RuntimeError(f"Failed to compute embeddings: {e}") from e

    words = sorted(set(x_embeds) & set(y_embeds))

    if not words:
        raise ValueError(
            "No shared words found between corpora after embedding. "
            "Verify that both corpora contain the same vocabulary."
        )

    word_means = {
        w: (
            x_embeds[w].word_embeds.mean(axis=0),
            y_embeds[w].word_embeds.mean(axis=0),
        )
        for w in words
    }

    word_occurrences = {
        w: [{"text": s, "date": CORPUS1_NAME} for s in x_lemma_sentences.get(w, [])]
        + [{"text": s, "date": CORPUS2_NAME} for s in y_lemma_sentences.get(w, [])]
        for w in words
    }

    all_sentences = list(
        {
            s
            for w in words
            for s in x_lemma_sentences.get(w, []) + y_lemma_sentences.get(w, [])
        }
    )

    if not all_sentences:
        raise ValueError(
            "No sentences extracted from corpora. "
            "This may indicate a problem with word extraction."
        )

    return words, word_means, word_occurrences, all_sentences


def fit_pca(word_means, extra_vecs=None):
    """Fit 3-component PCA on word mean embeddings, optionally including extra vectors
    (e.g. Top2Vec topic vectors) so words and topics share one coordinate system.

    Args:
        word_means: dict of word -> (embed_c1, embed_c2) tuples
        extra_vecs: optional array of additional vectors to include in PCA fit

    Returns:
        pca: Fitted PCA object with n_components=3
    """
    if not word_means:
        raise ValueError("word_means is empty, cannot fit PCA")

    vecs = [v for x_mean, y_mean in word_means.values() for v in (x_mean, y_mean)]

    if extra_vecs is not None and len(extra_vecs):
        vecs.extend(list(extra_vecs))

    if not vecs:
        raise ValueError("No vectors available for PCA fitting")

    try:
        pca = PCA(n_components=3)
        pca.fit(np.vstack(vecs))
        return pca
    except Exception as e:
        raise RuntimeError(f"PCA fitting failed: {e}") from e


def get_word_trajectory(word, word_means, pca):
    """Return [{period, x, y, z}] for the two corpus time-points.

    Args:
        word: word string
        word_means: dict of word -> (embed_c1, embed_c2)
        pca: fitted PCA object

    Returns:
        List of dicts with keys: period, x, y, z (3D coordinates)
        Returns empty list if word not found in word_means
    """
    if word not in word_means:
        return []

    try:
        x_mean, y_mean = word_means[word]
        coords = pca.transform(np.vstack([x_mean, y_mean]))
        return [
            {
                "period": CORPUS1_NAME,
                "x": float(coords[0][0]),
                "y": float(coords[0][1]),
                "z": float(coords[0][2]),
            },
            {
                "period": CORPUS2_NAME,
                "x": float(coords[1][0]),
                "y": float(coords[1][1]),
                "z": float(coords[1][2]),
            },
        ]
    except Exception as e:
        raise RuntimeError(
            f"Failed to compute trajectory for word '{word}': {e}"
        ) from e
