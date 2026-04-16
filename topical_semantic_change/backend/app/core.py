from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA

from lexical_semantic_change.extraction.word_extractor import extract_common_words
from lexical_semantic_change.representation.embed_cache import run_cache
from config import DATA_DIR

CORPUS1 = str(DATA_DIR / "sample" / "corpus1")
CORPUS2 = str(DATA_DIR / "sample" / "corpus2")
CORPUS1_NAME = Path(CORPUS1).stem  # "corpus1"
CORPUS2_NAME = Path(CORPUS2).stem  # "corpus2"
MODEL_NAME = "pierluigic/xl-lexeme"


def load_data():
    """Extract shared words, compute embeddings, return data needed by the API.

    Returns:
        words: sorted list of words present in both corpora
        word_means: {word: (mean_embed_c1, mean_embed_c2)}
        word_occurrences: {word: [{"text": str, "date": corpus_name}, ...]}
        all_sentences: deduplicated list of all sentences (for topic modelling)
    """
    x_words, y_words = extract_common_words(CORPUS1, CORPUS2)

    x_embeds, _, x_candidates = run_cache(x_words, CORPUS1_NAME, MODEL_NAME, layer=5)
    y_embeds, _, y_candidates = run_cache(y_words, CORPUS2_NAME, MODEL_NAME, layer=5)

    words = sorted(set(x_embeds) & set(y_embeds))

    word_means = {
        w: (
            x_embeds[w].word_embeds.mean(axis=0),
            y_embeds[w].word_embeds.mean(axis=0),
        )
        for w in words
    }

    word_occurrences = {
        w: [{"text": s, "date": CORPUS1_NAME} for s in x_candidates.get(w, [])]
        + [{"text": s, "date": CORPUS2_NAME} for s in y_candidates.get(w, [])]
        for w in words
    }

    all_sentences = list(
        {s for w in words for s in x_candidates.get(w, []) + y_candidates.get(w, [])}
    )

    return words, word_means, word_occurrences, all_sentences


def fit_pca(word_means, extra_vecs=None):
    """Fit 3-component PCA on word mean embeddings, optionally including extra vectors
    (e.g. Top2Vec topic vectors) so words and topics share one coordinate system.
    """
    vecs = [v for x_mean, y_mean in word_means.values() for v in (x_mean, y_mean)]
    if extra_vecs is not None and len(extra_vecs):
        vecs.extend(list(extra_vecs))
    pca = PCA(n_components=3)
    pca.fit(np.vstack(vecs))
    return pca


def get_word_trajectory(word, word_means, pca):
    """Return [{period, x, y, z}] for the two corpus time-points."""
    if word not in word_means:
        return []
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
