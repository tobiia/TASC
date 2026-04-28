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
    MODEL_NAME,
)
from .topic import train_top2vec, get_topics, group_sentences, assign_sentence_topics

logger = logging.getLogger(__name__)

# populated during lifespan startup
word_list = []
word_means = {}
word_occurrences = {}
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
    global word_list, word_means, word_occurrences, sentence_period
    global all_sentences, sentence_topic, sentence_embeddings, documents_payload
    global pca, top2vec_model, startup_error

    try:
        logger.info("Loading data...")
        word_list, word_means, word_occurrences, all_sentences = load_data()
        logger.info(f"Loaded {len(word_list)} words, {len(all_sentences)} sentences")

        logger.info("Grouping sentences into mega-documents...")
        groups, mega_doc_texts = group_sentences(all_sentences, MODEL_NAME)
        logger.info(f"Grouped {len(all_sentences)} sentences into {len(mega_doc_texts)} documents")

        logger.info("Training Top2Vec model...")
        top2vec_model = train_top2vec(mega_doc_texts)
        logger.info(f"Trained Top2Vec model with {top2vec_model.get_num_topics()} topics")

        logger.info("Assigning sentence topics...")
        sentence_topic, sentence_embeddings = assign_sentence_topics(top2vec_model, groups)
        logger.info(f"Assigned topics to {len(sentence_topic)} sentences")

        for occ_list in word_occurrences.values():
            for occ in occ_list:
                sentence_period.setdefault(occ["text"], occ["date"])
        logger.info(f"Built sentence→period map for {len(sentence_period)} sentences")

        logger.info("Fitting PCA...")
        sent_vecs = np.array([sentence_embeddings[i] for i in range(len(all_sentences))
                              if i in sentence_embeddings])
        pca = fit_pca(word_means, extra_vecs=sent_vecs)
        logger.info("PCA fitting complete")

        logger.info("Pre-computing document 3D projections...")
        valid_indices = [i for i in range(len(all_sentences)) if i in sentence_embeddings]
        vecs_3d = pca.transform(np.array([sentence_embeddings[i] for i in valid_indices]))
        for j, sent_idx in enumerate(valid_indices):
            sentence = all_sentences[sent_idx]
            documents_payload.append({
                "x": float(vecs_3d[j][0]),
                "y": float(vecs_3d[j][1]),
                "z": float(vecs_3d[j][2]),
                "topic": sentence_topic.get(sent_idx, -1),
                "period": sentence_period.get(sentence, ""),
                "text": sentence,
            })
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
        trajectory = get_word_trajectory(word, word_means, pca)
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
                result.append({
                    "id": t["id"],
                    "words": t["words"],
                    "size": t["size"],
                })
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
