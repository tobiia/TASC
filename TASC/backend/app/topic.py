import logging
import numpy as np
from ....config import (
    CACHE_DIR,
    TOP2VEC_MODEL,
    LAYER,
    TOP2VEC_WORKERS,
    TOP2VEC_NUM_TOPICS,
)
from top2vec import Top2Vec

logger = logging.getLogger(__name__)


def assign_sentence_topics(model, sentences):
    """Assign topics and extract embeddings for sentence-level Top2Vec.

    Works for both standard and contextual Top2Vec. With standard mode
    (used when precomputed_embeddings are passed), topic assignments come
    from model.doc_top — the per-document nearest topic. With contextual
    mode, doc_top_tokens is used as a fallback.

    Args:
        model: trained Top2Vec model
        sentences: list of sentence strings in the same order as documents

    Returns:
        sentence_topic: {sent_idx: topic_id}  (-1 if unresolvable)
        sentence_embeddings: {sent_idx: np.ndarray (hidden_dim,), L2-normalised}
    """
    sentence_topic = {}
    sentence_embeddings = {}

    # Standard Top2Vec stores per-document topic in doc_top
    has_doc_top = hasattr(model, "doc_top") and model.doc_top is not None
    has_doc_top_tokens = hasattr(model, "doc_top_tokens") and model.doc_top_tokens

    for doc_idx in range(len(sentences)):
        emb = model.document_vectors[doc_idx]
        norm = np.linalg.norm(emb)
        sentence_embeddings[doc_idx] = emb / norm if norm > 0 else emb

        topic_id = -1
        if has_doc_top:
            # Standard mode — direct per-document assignment
            topic_id = int(model.doc_top[doc_idx])
        elif has_doc_top_tokens and doc_idx in model.doc_top_tokens:
            # Contextual mode fallback — majority vote over token assignments
            tokens = model.doc_top_tokens[doc_idx]
            if tokens:
                topic_id = int(max(tokens, key=lambda t: len(tokens[t])))

        sentence_topic[doc_idx] = topic_id

    return sentence_topic, sentence_embeddings


def train_top2vec(
    documents,
    corpora_label: str = "corpora",
    precomputed_embeddings: np.ndarray | None = None,
):
    """Train a Top2Vec model on documents, with disk caching.

    If a cached model exists for this corpora_label it is loaded directly,
    skipping training entirely. Otherwise the model is trained and saved.

    When precomputed_embeddings is provided, Top2Vec skips its internal
    embedding step and uses these vectors directly — significantly faster
    since the corpus has already been embedded for word LSC purposes.
    The embeddings must be L2-normalised and in the same order as documents.

    Args:
        documents: list of sentence strings
        corpora_label: identifies the corpus name, e.g. "semeval_2020"
        precomputed_embeddings: optional (n_docs, hidden_dim) L2-normalised
            ndarray. When provided, Top2Vec skips internal embedding.

    Returns:
        Trained (or loaded) Top2Vec model
    """
    if not documents:
        raise ValueError("Cannot train Top2Vec on empty document list")

    cache_path = CACHE_DIR / f"top2vec_{corpora_label}.pkl"

    # ── Load from cache if available ──────────────────────────────────────
    if cache_path.exists():
        try:
            logger.info(f"Loading cached Top2Vec model from {cache_path}")
            model = Top2Vec.load(str(cache_path))
            cached_doc_count = model.document_vectors.shape[0]
            logger.info(
                f"Loaded model: {model.get_num_topics()} topics, "
                f"{cached_doc_count} documents"
            )
            if cached_doc_count != len(documents):
                logger.warning(
                    f"Cached model has {cached_doc_count} documents but corpus has "
                    f"{len(documents)}. Delete {cache_path} to retrain."
                )

            if TOP2VEC_NUM_TOPICS and model.get_num_topics() > TOP2VEC_NUM_TOPICS:
                try:
                    model.hierarchical_topic_reduction(num_topics=TOP2VEC_NUM_TOPICS)
                    logger.info(f"Reduced to {model.get_num_topics()} topics")
                except Exception as e:
                    logger.warning(f"Topic reduction failed: {e}")

            return model
        except Exception as e:
            logger.warning(f"Failed to load cached model ({e}), retraining...")

    # ── Train fresh ────────────────────────────────────────────────────────
    try:
        logger.info(f"Training Top2Vec with {len(documents)} documents...")
        init_kwargs = dict(
            contextual_top2vec=False,  # standard mode — we supply embeddings
            embedding_model=TOP2VEC_MODEL,
            embedding_layer=LAYER,
            workers=TOP2VEC_WORKERS,
        )
        if precomputed_embeddings is not None:
            init_kwargs["precomputed_embeddings"] = precomputed_embeddings  # type: ignore
            logger.info(
                "Using precomputed sentence embeddings — skipping internal embedding step"
            )

        model = Top2Vec(documents, **init_kwargs)  # type: ignore
        logger.info("Top2Vec trained successfully")
        logger.info(f"Top2Vec embedding model: {model.embedding_model}")
        logger.info(f"Top2Vec document vector shape: {model.document_vectors.shape}")

        # ── Save to cache ──────────────────────────────────────────────────
        try:
            model.save(str(cache_path))
            logger.info(f"Saved Top2Vec model to {cache_path}")
        except Exception as e:
            logger.warning(f"Failed to save Top2Vec model: {e}")

        return model
    except Exception as e:
        raise RuntimeError(f"Top2Vec training failed: {e}") from e


def get_topics(model):
    """Extract topic information from trained Top2Vec model.

    Returns:
        List of dicts with keys: id, words, size
    """
    try:
        n = model.get_num_topics()
        if n == 0:
            logger.warning("Top2Vec model has no topics")
            return []

        topic_words, _, topic_nums = model.get_topics(num_topics=n)
        topic_sizes_arr, topic_size_nums = model.get_topic_sizes()
        size_by_id = dict(zip(topic_size_nums.tolist(), topic_sizes_arr.tolist()))

        topics = []
        for i, num in enumerate(topic_nums):
            try:
                topics.append(
                    {
                        "id": int(num),
                        "words": topic_words[i][:10].tolist(),
                        "size": int(size_by_id.get(int(num), 0)),
                    }
                )
            except (IndexError, AttributeError, TypeError) as e:
                logger.error(f"Failed to extract topic {num}: {e}")
                continue

        return topics

    except Exception as e:
        raise RuntimeError(f"Failed to get topics from model: {e}") from e
