from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from lsc import run_word_cache, run_embed_cache
from src.config import TERMS_FILE, CORPUS1, CORPUS2, MODEL_NAME, LAYER


def _load_terms(path: Path):
    """Load terms from a CSV (looks for 'lemma' column, falls back to first column) or TXT file."""
    if path.suffix.lower() == ".csv":
        first_line = path.read_text(encoding="utf-8").splitlines()[0]
        sep = "\t" if "\t" in first_line else ","
        df = pd.read_csv(path, sep=sep)
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
        if not path.exists():
            raise FileNotFoundError(
                f"{name} path does not exist: {path}\n"
                f"Expected corpus files at: {CORPUS1}/*.txt and {CORPUS2}/*.txt"
            )


def load_data():
    """Extract shared words, compute embeddings, return data needed by the API.

    Returns:
        words: sorted list of words present in both corpora
        word_means: {word: (mean_embed_c1, mean_embed_c2)}
        word_occurrences: {word: [{"text": str, "date": corpus_name}, ...]}
        all_sentences: deduplicated list of all sentences (for topic modelling)
        x_embeds: raw embed objects for corpus 1, keyed by word
        y_embeds: raw embed objects for corpus 2, keyed by word
    """
    _validate_corpus_paths()

    # REVIEW - should eventually be optional
    terms: list[str] | None = None
    if TERMS_FILE is not None:
        if not TERMS_FILE.exists():
            raise FileNotFoundError(f"TERMS_FILE not found: {TERMS_FILE}")
        terms = _load_terms(TERMS_FILE)

    # FIXME what if the user only has cache files? need to refactor this
    corpora_label = Path(CORPUS1).parent.name
    corpus1_name = Path(CORPUS1).stem
    corpus2_name = Path(CORPUS2).stem

    try:
        x_words, y_words = run_word_cache(
            str(CORPUS1), str(CORPUS2), corpora_label, terms=terms
        )
    except Exception as e:
        raise RuntimeError(f"Failed to extract words from corpora: {e}") from e

    if not x_words or not y_words:
        raise ValueError(
            "Word extraction produced empty results. "
            "Check that corpus files contain valid text."
        )

    try:
        (x_embeds, _, x_lemma_sentences), (y_embeds, _, y_lemma_sentences) = (
            run_embed_cache(x_words, y_words, corpora_label, MODEL_NAME, layer=LAYER)
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
        w: [{"text": s, "date": corpus1_name} for s in x_lemma_sentences.get(w, [])]
        + [{"text": s, "date": corpus2_name} for s in y_lemma_sentences.get(w, [])]
        for w in words
    }

    all_sentences = list(
        dict.fromkeys(
            s
            for w in words
            for s in x_lemma_sentences.get(w, []) + y_lemma_sentences.get(w, [])
        )
    )

    if not all_sentences:
        raise ValueError(
            "No sentences extracted from corpora. "
            "This may indicate a problem with word extraction."
        )

    return words, word_means, word_occurrences, all_sentences, x_embeds, y_embeds


# FIXME can probably remove any references to entropy/variance calc


def compute_word_entropy(word, x_embeds, y_embeds):
    """Compute per-corpus contextual entropy for a word.

    Both corpora are compared against the shared cross-corpus prototype,
    so the two entropy values are directly comparable — a word that was
    stable in c1 but variable in c2 will show low then high Z in the plot,
    making the diachronic change in variability visible as vertical movement.

    High entropy = semantically dispersed / variable usages relative to the
                   shared prototype.
    Low entropy  = stable, predictable usages.

    Args:
        word: word string
        x_embeds: raw embed objects for corpus 1
        y_embeds: raw embed objects for corpus 2

    Returns:
        (entropy_c1, entropy_c2): floats in [0, 1]
    """
    x_vecs = x_embeds[word].word_embeds  # (n_c1, hidden_dim)
    y_vecs = y_embeds[word].word_embeds  # (n_c2, hidden_dim)
    all_vecs = np.vstack([x_vecs, y_vecs])  # (n_total, hidden_dim)

    # shared cross-corpus prototype (L2-normalised mean)
    prototype = all_vecs.mean(axis=0)
    prototype_norm = np.linalg.norm(prototype)
    if prototype_norm > 0:
        prototype = prototype / prototype_norm

    def _mean_cosine_distance(vecs):
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        normalised = vecs / np.maximum(norms, 1e-9)
        sims = normalised @ prototype
        return float(np.clip(1.0 - sims.mean(), 0.0, 1.0))

    return _mean_cosine_distance(x_vecs), _mean_cosine_distance(y_vecs)


def compute_all_entropies(words, x_embeds, y_embeds):
    """Compute per-corpus contextual entropy for every word.

    Args:
        words: list of word strings
        x_embeds: raw embed objects for corpus 1
        y_embeds: raw embed objects for corpus 2

    Returns:
        dict {word: (entropy_c1, entropy_c2)}
    """
    entropies = {}
    for w in words:
        try:
            entropies[w] = compute_word_entropy(w, x_embeds, y_embeds)
        except Exception:
            entropies[w] = (0.0, 0.0)
    return entropies


def fit_pca(word_means, extra_vecs=None) -> PCA:
    """Fit a 3-component PCA.

    All three spatial axes are PCA components so that words, documents,
    and topic centroids share a fully coherent semantic space.
    """
    vecs = [(x_mean + y_mean) / 2 for x_mean, y_mean in word_means.values()]

    if extra_vecs is not None and len(extra_vecs):
        vecs.extend(list(extra_vecs))

    if not vecs:
        raise ValueError("No vectors available for PCA fitting")
    try:
        reducer = PCA(n_components=3)
        reducer.fit(np.vstack(vecs))
        return reducer
    except Exception as e:
        raise RuntimeError(f"PCA fitting failed: {e}") from e


def get_word_trajectory(word, word_means, pca):
    """Return [{period, x, y, z}] for the two corpus time-points.

    Axes:
        x — PC 1  }
        y — PC 2  } all three from PCA — fully coherent semantic space
        z — PC 3  }

    Args:
        word: word string
        word_means: dict of word -> (embed_c1, embed_c2)
        pca: fitted 3-component PCA object

    Returns:
        List of dicts with keys: period, x, y, z
        Returns empty list if word not found in word_means.
    """
    if word not in word_means:
        return []

    try:
        x_mean, y_mean = word_means[word]
        coords = pca.transform(np.vstack([x_mean, y_mean]))  # (2, 3)

        corpus1_name = Path(CORPUS1).stem
        corpus2_name = Path(CORPUS2).stem

        return [
            {
                "period": corpus1_name,
                "x": float(coords[0][0]),
                "y": float(coords[0][1]),
                "z": float(coords[0][2]),
            },
            {
                "period": corpus2_name,
                "x": float(coords[1][0]),
                "y": float(coords[1][1]),
                "z": float(coords[1][2]),
            },
        ]
    except Exception as e:
        raise RuntimeError(
            f"Failed to compute trajectory for word '{word}': {e}"
        ) from e


def get_nearest_topics(word_embed, topic_vectors, topic_ids, n=2):
    """Find the n nearest topic centroids to a word embedding by cosine similarity.

    Args:
        word_embed: np.ndarray (hidden_dim,) — the word's mean embedding for one period
        topic_vectors: np.ndarray (n_topics, hidden_dim) — raw topic vectors from Top2Vec
        topic_ids: list of int — topic ids corresponding to rows of topic_vectors
        n: number of nearest topics to return (default 2)

    Returns:
        List of dicts: [{id, distance}] sorted closest first.
        distance is cosine distance (1 - cosine_similarity), in [0, 1].
    """
    norm_word = word_embed / np.maximum(np.linalg.norm(word_embed), 1e-9)
    norms = np.linalg.norm(topic_vectors, axis=1, keepdims=True)
    norm_topics = topic_vectors / np.maximum(norms, 1e-9)
    sims = norm_topics @ norm_word  # (n_topics,)
    distances = 1.0 - sims

    top_n = np.argsort(distances)[:n]
    return [{"id": int(topic_ids[i]), "distance": float(distances[i])} for i in top_n]
