import logging
from top2vec import Top2Vec

logger = logging.getLogger(__name__)


def train_top2vec(documents, model_name):
    """Train a Top2Vec model on documents.

    Args:
        documents: list of document strings
        model_name: name of transformer model to use

    Returns:
        Trained Top2Vec model
    """
    if not documents:
        raise ValueError("Cannot train Top2Vec on empty document list")

    try:
        logger.info(f"Training Top2Vec with {len(documents)} documents...")
        model = Top2Vec(documents, embedding_model=model_name, speed="learn")
        logger.info(f"Top2Vec trained successfully")
        return model
    except Exception as e:
        raise RuntimeError(f"Top2Vec training failed: {e}") from e


def get_topics(model):
    """Extract topic information from trained Top2Vec model.

    Args:
        model: Trained Top2Vec model

    Returns:
        List of dicts with keys: id, words, centroid, size
        Each dict contains:
        - id (int): Topic ID
        - words (list): Top 10 words for the topic
        - centroid (list): Topic embedding vector
        - size (int): Number of documents in topic
    """
    try:
        n = model.get_num_topics()
        if n == 0:
            logger.warning("Top2Vec model has no topics")
            return []

        topic_words, _, topic_nums = model.get_topics(num_topics=n)
        topic_sizes, _ = model.get_topic_sizes()

        if len(topic_nums) != len(topic_sizes):
            raise ValueError(
                f"Topic count mismatch: {len(topic_nums)} topics but {len(topic_sizes)} sizes"
            )

        topics = []
        for i, num in enumerate(topic_nums):
            try:
                topic_dict = {
                    "id": int(num),
                    "words": topic_words[i][:10].tolist(),
                    "centroid": model.topic_vectors[num].tolist(),
                    "size": int(topic_sizes[i]),
                }
                topics.append(topic_dict)
            except (IndexError, AttributeError, TypeError) as e:
                logger.error(f"Failed to extract topic {num}: {e}")
                continue

        return topics

    except Exception as e:
        raise RuntimeError(f"Failed to get topics from model: {e}") from e
