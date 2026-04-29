import logging
import numpy as np
from transformers import AutoTokenizer
from top2vec import Top2Vec
from .config import CACHE_DIR, TOP2VEC_MODEL, TOP2VEC_WORKERS, TOP2VEC_NUM_TOPICS

logger = logging.getLogger(__name__)


def group_sentences(sentences, hf_model_name, max_tokens=450):
    """Pack sentences sequentially into mega-documents within the token budget.

    The tokenizer's 512-token limit means we cannot pass all sentences as one
    document. Each mega-document is a contiguous group of sentences joined by
    spaces, staying under max_tokens (leaving headroom for [CLS] and [SEP]).

    Returns:
        groups: list of {'text': str, 'sentence_info': [(sent_idx, tok_start, tok_end)]}
            tok_start/tok_end are 0-indexed offsets into the mega-document's
            content tokens (not counting [CLS]). The actual positions in the
            tokenized document are tok_start+1 .. tok_end (exclusive of [SEP]).
        mega_doc_texts: list of str (the text field of each group, for Top2Vec)
    """
    tokenizer = AutoTokenizer.from_pretrained(hf_model_name)

    token_counts = [
        len(tokenizer.encode(s, add_special_tokens=False)) for s in sentences
    ]

    groups = []
    current_sents = []
    current_infos = []
    current_count = 0

    for sent_idx, (sentence, n_tok) in enumerate(zip(sentences, token_counts)):
        if n_tok == 0:
            continue
        if current_count + n_tok > max_tokens and current_sents:
            groups.append(
                {
                    "text": " ".join(current_sents),
                    "sentence_info": current_infos[:],
                }
            )
            current_sents, current_infos, current_count = [], [], 0

        tok_start = current_count
        current_sents.append(sentence)
        current_infos.append((sent_idx, tok_start, tok_start + n_tok))
        current_count += n_tok

    if current_sents:
        groups.append(
            {
                "text": " ".join(current_sents),
                "sentence_info": current_infos[:],
            }
        )

    return groups, [g["text"] for g in groups]


def assign_sentence_topics(model, groups):
    """Map contextual Top2Vec token-level topics back to original sentences.

    Uses doc_top_tokens (set by compute_topics for contextual top2vec), which
    stores per-document token-position → topic assignments. Sentence boundaries
    are the tok_start/tok_end offsets recorded during grouping, shifted by +1
    to account for the [CLS] token at position 0.

    Returns:
        sentence_topic: {sent_idx: topic_id}  (-1 if unresolvable)
        sentence_embeddings: {sent_idx: np.ndarray (768,), L2-normalised}
    """
    sentence_topic = {}
    sentence_embeddings = {}

    for doc_idx, group in enumerate(groups):
        doc_embs = model.document_token_embeddings[doc_idx]  # (n_tokens, 768)
        n_tokens = doc_embs.shape[0]

        # token_topic_arr[i] = topic assigned to token i in this document
        token_topic_arr = np.full(n_tokens, -1, dtype=np.int32)
        for topic_id, token_inds in model.doc_top_tokens.get(doc_idx, {}).items():
            valid = token_inds[token_inds < n_tokens]
            token_topic_arr[valid] = int(topic_id)

        for sent_idx, tok_start, tok_end in group["sentence_info"]:
            start = tok_start + 1  # skip [CLS] at position 0
            end = min(tok_end + 1, n_tokens - 1)  # stop before [SEP]
            if start >= end:
                continue

            token_topics = token_topic_arr[start:end]
            assigned = token_topics[token_topics >= 0]
            sentence_topic[sent_idx] = (
                int(np.bincount(assigned).argmax()) if len(assigned) else -1
            )

            emb = doc_embs[start:end].mean(axis=0)
            norm = np.linalg.norm(emb)
            sentence_embeddings[sent_idx] = emb / norm if norm > 0 else emb

    return sentence_topic, sentence_embeddings


def assign_sentence_topics_direct(model, sentences):
    """Assign topics and extract embeddings when each document is one sentence.

    When Top2Vec is trained on individual sentences rather than mega-documents,
    each document_vectors[i] is already the sentence-level embedding and each
    document has exactly one topic assignment. No token-offset arithmetic needed.

    Args:
        model: trained Top2Vec model
        sentences: list of sentence strings in the same order passed to Top2Vec

    Returns:
        sentence_topic: {sent_idx: topic_id}  (-1 if unresolvable)
        sentence_embeddings: {sent_idx: np.ndarray (hidden_dim,), L2-normalised}
    """
    sentence_topic = {}
    sentence_embeddings = {}

    for doc_idx in range(len(sentences)):
        # document_vectors[doc_idx] is the sentence-level embedding from Top2Vec
        emb = model.document_vectors[doc_idx]
        norm = np.linalg.norm(emb)
        sentence_embeddings[doc_idx] = emb / norm if norm > 0 else emb

        # topic assignment — use doc_top_tokens majority vote if available,
        # otherwise fall back to nearest centroid via document_vectors
        topic_id = -1
        if hasattr(model, "doc_top_tokens") and doc_idx in model.doc_top_tokens:
            tokens = model.doc_top_tokens[doc_idx]
            if tokens:
                # take the topic with the most assigned tokens
                topic_id = int(max(tokens, key=lambda t: len(tokens[t])))
        sentence_topic[doc_idx] = topic_id

    return sentence_topic, sentence_embeddings


def train_top2vec(documents, cache_domain: str = "corpus"):
    """Train a Top2Vec model on documents, with disk caching.

    If a cached model exists for this cache_domain + document count it is
    loaded directly, skipping training entirely. Otherwise the model is
    trained and saved for future runs.

    The cache filename encodes cache_domain and document count so that a model
    trained on a different corpus or different subsample is never silently
    reused.

    Args:
        documents: list of document strings (all sentences)
        cache_domain: identifies the corpus pair, e.g. "1860s_1950s"

    Returns:
        Trained (or loaded) Top2Vec model
    """
    if not documents:
        raise ValueError("Cannot train Top2Vec on empty document list")

    cache_path = CACHE_DIR / f"top2vec_{cache_domain}.pkl"

    # ── Load from cache if available ──────────────────────────────────────
    if cache_path.exists():
        try:
            logger.info(f"Loading cached Top2Vec model from {cache_path}")
            model = Top2Vec.load(str(cache_path))
            logger.info(
                f"Loaded model: {model.get_num_topics()} topics, "
                f"{model.document_vectors.shape[0]} documents"
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
        model = Top2Vec(
            documents,
            contextual_top2vec=True,
            embedding_model=TOP2VEC_MODEL,
            workers=TOP2VEC_WORKERS,
        )
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
