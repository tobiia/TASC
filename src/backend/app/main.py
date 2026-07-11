from contextlib import asynccontextmanager
import logging
import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from src.backend.app.core import (
    get_word_trajectory,
    get_nearest_topics,
)
from src.backend.app.topic import get_topics
from src.backend.app.data.load import load_app_data
from src.config import DIST

logging.basicConfig(
    level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s\n"
)
logger = logging.getLogger(__name__)
from transformers.utils import logging as hf_logging

hf_logging.set_verbosity_error()
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("top2vec").setLevel(logging.WARNING)
logging.getLogger("top2vec").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global app_data, startup_error

    # prevent uvicorn logs
    logging.getLogger("uvicorn").propagate = False
    logging.getLogger("uvicorn.error").propagate = False

    try:
        app_data = load_app_data()

    except Exception as e:
        logger.error(f"Startup failed: {e}", exc_info=True)
        startup_error = e

    yield

    logger.info("Shutting down...")


app = FastAPI(lifespan=lifespan)

IS_DEV = os.environ.get("TASC_DEV") == "1"

if IS_DEV:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173"],
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
    if not app_data.word_list or app_data.pca is None or app_data.top2vec_model is None:
        raise HTTPException(
            status_code=503,
            detail="Service not yet initialized. Please retry.",
        )


@app.get("/api/health")
def health_check():
    if startup_error:
        return {"status": "error", "error": str(startup_error)}
    if not app_data.word_list or app_data.pca is None or app_data.top2vec_model is None:
        return {"status": "initializing"}
    return {"status": "ready", "words": len(app_data.word_list)}


@app.get("/api/words")
def list_words():
    _ensure_ready()
    return {"words": app_data.word_list}


@app.get("/api/word/{word}")
def get_word_data(word):
    _ensure_ready()
    try:
        if word not in app_data.word_means:
            raise HTTPException(status_code=404, detail=f"Word '{word}' not found")
        trajectory = get_word_trajectory(word, app_data.word_means, app_data.pca)
        occurrences = app_data.word_occurrences.get(word, [])

        # nearest 2 topics per time period
        topic_vecs = app_data.top2vec_model.topic_vectors
        topic_ids_arr = list(range(app_data.top2vec_model.get_num_topics()))
        x_mean, y_mean = app_data.word_means[word]
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
        topics_data = get_topics(app_data.top2vec_model)
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
    return {"topic_centroids": app_data.topic_centroids_payload}


@app.get("/api/documents")
def get_documents_endpoint():
    """Return one point per original sentence with its topic and corpus-period label."""
    _ensure_ready()
    return {"documents": app_data.documents_payload}


if not IS_DEV:
    app.mount(
        "/",
        StaticFiles(directory=DIST, html=True),
        name="frontend",
    )
