from contextlib import asynccontextmanager
import logging
import os

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .core import (
    load_data,
    fit_pca,
    get_word_trajectory,
    compute_all_entropies,
    MODEL_NAME,
)
from .topic import train_top2vec, get_topics, group_sentences, assign_sentence_topics

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
pca = None
top2vec_model = None
startup_error = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global word_list, word_means, word_occurrences, word_entropies, sentence_period
    global all_sentences, sentence_topic, sentence_embeddings, documents_payload
    global pca, top2vec_model, startup_error

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
        groups, mega_doc_texts = group_sentences(all_sentences, MODEL_NAME)
        logger.info(
            f"Grouped {len(all_sentences)} sentences into {len(mega_doc_texts)} documents"
        )

        logger.info("Training Top2Vec model...")
        top2vec_model = train_top2vec(mega_doc_texts)
        logger.info(
            f"Trained Top2Vec model with {top2vec_model.get_num_topics()} topics"
        )

        logger.info("Assigning sentence topics...")
        sentence_topic, sentence_embeddings = assign_sentence_topics(
            top2vec_model, groups
        )
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

        logger.info("Pre-computing document 3D projections...")
        valid_indices = [
            i for i in range(len(all_sentences)) if i in sentence_embeddings
        ]
        # PCA is 3-component: transform returns (n, 3)
        vecs_3d = pca.transform(
            np.array([sentence_embeddings[i] for i in valid_indices])
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

        for j, sent_idx in enumerate(valid_indices):
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
                    "x": float(vecs_3d[j][0]),  # PC 1
                    "y": float(vecs_3d[j][1]),  # PC 2
                    "z": float(vecs_3d[j][2]),  # PC 3
                    "entropy": doc_entropy,  # display only
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
        return {"trajectory": trajectory, "occurrences": occurrences}
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


@app.get("/api/documents")
def get_documents_endpoint():
    """Return one point per original sentence with its topic and corpus-period label."""
    _ensure_ready()
    return {"documents": documents_payload}
