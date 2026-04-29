from contextlib import asynccontextmanager
import logging
import os
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sklearn.decomposition import PCA
from top2vec import Top2Vec

from .core import (
    load_data,
    fit_pca,
    get_word_trajectory,
    compute_all_entropies,
    get_nearest_topics,
    MODEL_NAME,
)
from .topic import train_top2vec, get_topics, group_sentences, assign_sentence_topics

from .config import MAX_TOPIC_SENTENCES, MAX_RENDER_SENTENCES, RANDOM_SEED
from pathlib import Path
from .config import CORPUS1, CORPUS2

logger = logging.getLogger(__name__)

# populated during lifespan startup
word_list = []
word_means = {}
word_occurrences = {}
word_entropies = {}
sentence_period: dict = {}
all_sentences: list = []
sentence_topic: dict = {}
sentence_embeddings: dict = {}
documents_payload: list = []
topic_centroids_payload: list = []
pca: Optional[PCA] = None
top2vec_model: Optional[Top2Vec] = None
startup_error = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global word_list, word_means, word_occurrences, word_entropies, sentence_period
    global all_sentences, sentence_topic, sentence_embeddings, documents_payload
    global topic_centroids_payload, pca, top2vec_model, startup_error

    try:
        logger.info("Loading data...")
        word_list, word_means, word_occurrences, all_sentences, x_embeds, y_embeds = (
            load_data()
        )
        logger.info(f"Loaded {len(word_list)} words, {len(all_sentences)} sentences")

        logger.info("Computing word entropies...")
        word_entropies = compute_all_entropies(word_list, x_embeds, y_embeds)
        logger.info(f"Computed entropy for {len(word_entropies)} words")

        logger.info("Grouping sentences into mega-documents...")
        # Randomly subsample for Top2Vec if corpus exceeds limit.
        # Seed is fixed so the same corpus always picks the same sentences
        # on a fresh run (before the cache exists). Word embeddings and the
        # occurrence bar always use the full all_sentences list unchanged.
        rng = np.random.default_rng(seed=RANDOM_SEED)
        if len(all_sentences) > MAX_TOPIC_SENTENCES:
            topic_indices = np.sort(
                rng.choice(len(all_sentences), size=MAX_TOPIC_SENTENCES, replace=False)
            )
            topic_sentences = [all_sentences[i] for i in topic_indices]
            logger.info(
                f"Sampled {MAX_TOPIC_SENTENCES} / {len(all_sentences)} "
                f"sentences for Top2Vec"
            )
        else:
            topic_indices = np.arange(len(all_sentences))
            topic_sentences = all_sentences

        groups, mega_doc_texts = group_sentences(topic_sentences, MODEL_NAME)
        logger.info(
            f"Grouped {len(topic_sentences)} sentences into "
            f"{len(mega_doc_texts)} documents"
        )

        logger.info("Training Top2Vec model...")
        cache_domain = f"{Path(CORPUS1).stem}_{Path(CORPUS2).stem}"
        top2vec_model = train_top2vec(mega_doc_texts, cache_domain=cache_domain)
        logger.info(
            f"Trained Top2Vec model with {top2vec_model.get_num_topics()} topics"
        )

        logger.info("Assigning sentence topics...")
        # assign_sentence_topics returns indices into topic_sentences (0..N-1).
        # Remap to global all_sentences indices using topic_indices.
        raw_topic, raw_embeddings = assign_sentence_topics(top2vec_model, groups)
        sentence_topic = {
            int(topic_indices[local_idx]): topic_id
            for local_idx, topic_id in raw_topic.items()
        }
        sentence_embeddings = {
            int(topic_indices[local_idx]): emb
            for local_idx, emb in raw_embeddings.items()
        }
        logger.info(f"Assigned topics to {len(sentence_topic)} sentences")

        for occ_list in word_occurrences.values():
            for occ in occ_list:
                sentence_period.setdefault(occ["text"], occ["date"])
        logger.info(f"Built sentence→period map for {len(sentence_period)} sentences")

        logger.info("Fitting PCA...")
        sent_vecs = np.array(
            [
                sentence_embeddings[i]
                for i in range(len(all_sentences))
                if i in sentence_embeddings
            ]
        )
        pca = fit_pca(word_means, extra_vecs=sent_vecs)
        logger.info("PCA fitting complete")

        logger.info("Projecting topic centroids...")
        n_topics = top2vec_model.get_num_topics()
        topic_vecs = top2vec_model.topic_vectors  # (n_topics, hidden_dim)
        topic_ids_list = list(range(n_topics))
        topic_coords = pca.transform(topic_vecs)  # (n_topics, 3)
        topics_meta = get_topics(top2vec_model)
        words_by_id = {t["id"]: t["words"] for t in topics_meta}
        for i, tid in enumerate(topic_ids_list):
            topic_centroids_payload.append(
                {
                    "id": tid,
                    "x": float(topic_coords[i][0]),
                    "y": float(topic_coords[i][1]),
                    "z": float(topic_coords[i][2]),
                    "words": words_by_id.get(tid, []),
                }
            )
        logger.info(f"Projected {len(topic_centroids_payload)} topic centroids")

        logger.info("Pre-computing document 3D projections...")
        valid_indices = [
            i for i in range(len(all_sentences)) if i in sentence_embeddings
        ]

        # Randomly subsample for rendering — uses the same rng (seed=42) so the
        # rendered subset is stable across restarts. Occurrence bar still uses
        # all sentences unchanged.
        if len(valid_indices) > MAX_RENDER_SENTENCES:
            render_indices = sorted(
                rng.choice(
                    valid_indices, size=MAX_RENDER_SENTENCES, replace=False
                ).tolist()
            )
            logger.info(
                f"Subsampled {MAX_RENDER_SENTENCES} / {len(valid_indices)} "
                f"sentences for 3D rendering"
            )
        else:
            render_indices = valid_indices

        # Project only the sentences we're actually going to render
        vecs_3d = pca.transform(
            np.array([sentence_embeddings[i] for i in render_indices])
        )

        # Pre-compute topic centroid norms once for efficiency
        n_topics = top2vec_model.get_num_topics()
        topic_centroid_norms = {}
        for tid in range(n_topics):
            try:
                centroid = top2vec_model.topic_vectors[tid]
                norm = np.linalg.norm(centroid)
                topic_centroid_norms[tid] = centroid / norm if norm > 0 else centroid
            except (IndexError, AttributeError):
                pass

        for j, sent_idx in enumerate(render_indices):
            sentence = all_sentences[sent_idx]
            topic_id = sentence_topic.get(sent_idx, -1)

            # Document entropy: cosine distance from assigned topic centroid
            # display-only — not used as a spatial axis
            doc_entropy = 0.0
            if topic_id >= 0 and topic_id in topic_centroid_norms:
                sent_norm = sentence_embeddings[sent_idx]  # already L2-normalised
                doc_entropy = float(
                    np.clip(
                        1.0 - float(sent_norm @ topic_centroid_norms[topic_id]),
                        0.0,
                        1.0,
                    )
                )

            documents_payload.append(
                {
                    "x": float(vecs_3d[j][0]),
                    "y": float(vecs_3d[j][1]),
                    "z": float(vecs_3d[j][2]),
                    "entropy": doc_entropy,
                    "topic": topic_id,
                    "period": sentence_period.get(sentence, ""),
                    "text": sentence,
                }
            )
        logger.info(f"Pre-computed {len(documents_payload)} document points")

    except Exception as e:
        logger.error(f"Startup failed: {e}", exc_info=True)
        startup_error = e

    yield

    logger.info("Shutting down...")


app = FastAPI(lifespan=lifespan)

_cors_env = os.getenv("CORS_ORIGINS", "http://localhost:5173")
origins = [o.strip() for o in _cors_env.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _ensure_ready():
    if startup_error:
        raise HTTPException(
            status_code=503,
            detail=f"Service initialization failed: {str(startup_error)}",
        )
    if not word_list or pca is None or top2vec_model is None:
        raise HTTPException(
            status_code=503,
            detail="Service not yet initialized. Please retry.",
        )


@app.get("/api/health")
def health_check():
    if startup_error:
        return {"status": "error", "error": str(startup_error)}
    if not word_list or pca is None or top2vec_model is None:
        return {"status": "initializing"}
    return {"status": "ready", "words": len(word_list)}


@app.get("/api/words")
def list_words():
    _ensure_ready()
    return {"words": word_list}


@app.get("/api/word/{word}")
def get_word_data(word):
    _ensure_ready()
    try:
        if word not in word_means:
            raise HTTPException(status_code=404, detail=f"Word '{word}' not found")
        trajectory = get_word_trajectory(word, word_means, pca, word_entropies)
        occurrences = word_occurrences.get(word, [])

        # nearest 2 topics per time period
        topic_vecs = top2vec_model.topic_vectors
        topic_ids_arr = list(range(top2vec_model.get_num_topics()))
        x_mean, y_mean = word_means[word]
        nearest = (
            {
                trajectory[0]["period"]: get_nearest_topics(
                    x_mean, topic_vecs, topic_ids_arr
                ),
                trajectory[1]["period"]: get_nearest_topics(
                    y_mean, topic_vecs, topic_ids_arr
                ),
            }
            if len(trajectory) == 2
            else {}
        )

        return {
            "trajectory": trajectory,
            "occurrences": occurrences,
            "nearest_topics": nearest,
        }
    except Exception as e:
        logger.error(f"Error getting word data for '{word}': {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve word data: {str(e)}",
        ) from e


@app.get("/api/topics")
def get_topics_endpoint():
    _ensure_ready()
    try:
        topics_data = get_topics(top2vec_model)
        if not topics_data:
            logger.warning("No topics returned from Top2Vec model")
            return {"topics": []}

        result = []
        for t in topics_data:
            try:
                result.append(
                    {
                        "id": t["id"],
                        "words": t["words"],
                        "size": t["size"],
                    }
                )
            except Exception as e:
                logger.error(f"Failed to process topic {t.get('id')}: {e}")
                continue

        return {"topics": result}

    except Exception as e:
        logger.error(f"Error getting topics: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve topics: {str(e)}",
        ) from e


@app.get("/api/topic-centroids")
def get_topic_centroids_endpoint():
    """Return PCA-projected 3D coordinates for every topic centroid."""
    _ensure_ready()
    return {"topic_centroids": topic_centroids_payload}


@app.get("/api/documents")
def get_documents_endpoint():
    """Return one point per original sentence with its topic and corpus-period label."""
    _ensure_ready()
    return {"documents": documents_payload}
