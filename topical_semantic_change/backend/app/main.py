from contextlib import asynccontextmanager
import logging

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .core import (
    load_data,
    fit_pca,
    get_word_trajectory,
    MODEL_NAME,
)
from .topic import train_top2vec, get_topics

logger = logging.getLogger(__name__)

# populated during lifespan startup
word_list = []
word_means = {}
word_occurrences = {}
pca = None
top2vec_model = None
startup_error = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global word_list, word_means, word_occurrences, pca, top2vec_model, startup_error

    try:
        logger.info("Loading data...")
        word_list, word_means, word_occurrences, all_sentences = load_data()
        logger.info(f"Loaded {len(word_list)} words, {len(all_sentences)} sentences")

        logger.info("Training Top2Vec model...")
        top2vec_model = train_top2vec(all_sentences, MODEL_NAME)
        logger.info(
            f"Trained Top2Vec model with {top2vec_model.get_num_topics()} topics"
        )

        logger.info("Fitting PCA...")
        pca = fit_pca(word_means, extra_vecs=top2vec_model.topic_vectors)
        logger.info("PCA fitting complete")

    except Exception as e:
        logger.error(f"Startup failed: {e}", exc_info=True)
        startup_error = e

    yield  # app runs here

    # currently gets stuck if this fails even with KeyboardIterrupt
    logger.info("Shutting down...")


app = FastAPI(lifespan=lifespan)

origins = ["http://localhost:5173", "localhost:5173"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _ensure_ready():
    """Raise error if app failed to initialize"""
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
    """Check if service is ready"""
    if startup_error:
        return {
            "status": "error",
            "error": str(startup_error),
        }
    if not word_list:
        return {"status": "initializing"}
    return {"status": "ready", "words": len(word_list)}


@app.get("/api/words")
def list_words():
    """List all available words"""
    _ensure_ready()
    return {"words": word_list}


@app.get("/api/word/{word}")
def get_word_data(word):
    """Get trajectory and occurrences for a word"""
    _ensure_ready()

    try:
        if word not in word_means:
            return {"trajectory": [], "occurrences": []}

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
    """Get topics with 3D coordinates and metadata"""
    _ensure_ready()

    try:
        topics_data = get_topics(top2vec_model)

        if not topics_data:
            logger.warning("No topics returned from Top2Vec model")
            return {"topics": []}

        result = []
        for t in topics_data:
            try:
                # IGNORING PYLANCE B/C I LITERALLY CATCH EVERY POSSIBLE ERROR
                centroid_3d = pca.transform(np.array([t["centroid"]]))[0]  # type: ignore

                # Get documents assigned to this topic
                if not hasattr(top2vec_model, "doc_top"):
                    logger.warning("Top2Vec model missing 'doc_top' attribute")
                    radius = 0.1
                else:
                    doc_mask = top2vec_model.doc_top == t["id"]  # type: ignore
                    doc_vecs = top2vec_model.document_vectors[doc_mask]  # type: ignore

                    doc_vecs_3d = pca.transform(doc_vecs) if len(doc_vecs) > 0 else None  # type: ignore
                    radius = (
                        float(
                            np.mean(
                                np.linalg.norm(
                                    doc_vecs_3d - doc_vecs_3d.mean(axis=0), axis=1
                                )
                            )
                        )
                        if doc_vecs_3d is not None and len(doc_vecs_3d) > 0
                        else 0.1
                    )

                result.append(
                    {
                        "id": t["id"],
                        "words": t["words"],
                        "x": float(centroid_3d[0]),
                        "y": float(centroid_3d[1]),
                        "z": float(centroid_3d[2]),
                        "radius": radius,
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
