import json
import logging
from pathlib import Path

import numpy as np

from src.backend.app.core import get_word_trajectory, get_nearest_topics
from src.backend.app.data.load import load_app_data
from src.backend.app.topic import get_topics
from src.config import PROJECT_ROOT, RANDOM_SEED

logger = logging.getLogger(__name__)

DEMO_WORDS = ["graft", "plane", "prop", "record", "chairman"]
MAX_SAMPLE_OCCURRENCES_PER_PERIOD = 15

OUTPUT_PATH = PROJECT_ROOT / "src" / "frontend" / "public" / "demo_data.json"


def build_demo_data(app_data=None) -> dict:
    if app_data is None:
        app_data = load_app_data()

    rng = np.random.default_rng(seed=RANDOM_SEED)

    topic_vecs = app_data.top2vec_model.topic_vectors
    topic_ids_arr = list(range(app_data.top2vec_model.get_num_topics()))
    topics_by_id = {t["id"]: t for t in get_topics(app_data.top2vec_model)}

    # maps a sentence's text back to its index into all_sentences/sentence_embeddings/
    # sentence_topic — needed to find which sampled occurrences can double as document points
    sent_to_global = {s: i for i, s in enumerate(app_data.all_sentences)}

    def sample(pool, n):
        if len(pool) <= n:
            return pool
        keep = sorted(rng.choice(len(pool), size=n, replace=False))
        return [pool[i] for i in keep]

    words_payload = []
    referenced_topic_ids = set()
    document_indices = set()

    for word in DEMO_WORDS:
        trajectory = get_word_trajectory(word, app_data.word_means, app_data.pca)
        if len(trajectory) != 2:
            continue

        x_mean, y_mean = app_data.word_means[word]
        nearest_by_period = {
            trajectory[0]["period"]: get_nearest_topics(
                x_mean, topic_vecs, topic_ids_arr
            ),
            trajectory[1]["period"]: get_nearest_topics(
                y_mean, topic_vecs, topic_ids_arr
            ),
        }

        for topics in nearest_by_period.values():
            for t in topics:
                referenced_topic_ids.add(t["id"])

        occurrences = app_data.word_occurrences.get(word, [])
        sampled_occurrences = []
        for period in (trajectory[0]["period"], trajectory[1]["period"]):
            period_occs = [o for o in occurrences if o["date"] == period]

            # only sentences Top2Vec actually saw have an embedding + topic —
            # prefer those so occurrences double as points in the document cloud
            has_topic = [
                sent_to_global.get(o["text"]) in app_data.sentence_embeddings
                for o in period_occs
            ]
            with_topic = [o for o, ok in zip(period_occs, has_topic) if ok]
            without_topic = [o for o, ok in zip(period_occs, has_topic) if not ok]

            chosen = sample(with_topic, MAX_SAMPLE_OCCURRENCES_PER_PERIOD)
            if len(chosen) < MAX_SAMPLE_OCCURRENCES_PER_PERIOD:
                chosen += sample(
                    without_topic, MAX_SAMPLE_OCCURRENCES_PER_PERIOD - len(chosen)
                )

            sampled_occurrences.extend(chosen)
            document_indices.update(
                sent_to_global[o["text"]]
                for o in chosen
                if sent_to_global.get(o["text"]) in app_data.sentence_embeddings
            )

        words_payload.append(
            {
                "word": word,
                "trajectory": trajectory,
                "nearest_topics": nearest_by_period,
                "occurrences": sampled_occurrences,
            }
        )

    # project only the sentences backing the sampled occurrences — ties the
    # demo's document cloud directly to the 5 words and their nearest topics,
    # rather than a disconnected random subsample of the whole corpus
    # FIXME - repetitive, should use doc payload from load instead
    documents_payload = []
    if document_indices:
        ordered_indices = sorted(document_indices)
        vecs = app_data.pca.transform(
            np.array([app_data.sentence_embeddings[i] for i in ordered_indices])
        )
        for j, idx in enumerate(ordered_indices):
            sentence = app_data.all_sentences[idx]
            documents_payload.append(
                {
                    "x": float(vecs[j][0]),
                    "y": float(vecs[j][1]),
                    "z": float(vecs[j][2]),
                    "topic": app_data.sentence_topic.get(idx, -1),
                    "period": app_data.sentence_period.get(sentence, ""),
                    "text": sentence,
                }
            )

    topic_centroids_payload = [
        t for t in app_data.topic_centroids_payload if t["id"] in referenced_topic_ids
    ]

    topics_payload = [
        topics_by_id[i] for i in sorted(referenced_topic_ids) if i in topics_by_id
    ]

    return {
        "words": words_payload,
        "topics": topics_payload,
        "topic_centroids": topic_centroids_payload,
        "documents": documents_payload,
    }


def export_demo_data(output_path: Path = OUTPUT_PATH) -> Path:
    data = build_demo_data()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    logger.info(f"Exported demo data ({len(data['words'])} words) to {output_path}")
    return output_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    export_demo_data()
