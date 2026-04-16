from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .core import (
    load_data,
    fit_pca,
    get_word_trajectory,
    MODEL_NAME,
)
from .topic import train_top2vec, get_topics

# Populated during lifespan startup
word_list = []
word_means = {}
word_occurrences = {}
pca = None
top2vec_model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global word_list, word_means, word_occurrences, pca, top2vec_model

    word_list, word_means, word_occurrences, all_sentences = load_data()
    top2vec_model = train_top2vec(all_sentences, MODEL_NAME)
    # Include Top2Vec topic vectors when fitting PCA so words and topics
    # share the same 2D coordinate system in the plot.
    pca = fit_pca(word_means, extra_vecs=top2vec_model.topic_vectors)

    yield  # app runs here


app = FastAPI(lifespan=lifespan)

origins = ["http://localhost:5173", "localhost:5173"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/words")
def list_words():
    return {"words": word_list}


@app.get("/api/word/{word}")
def get_word_data(word: str):
    if word not in word_means:
        return {"trajectory": [], "occurrences": []}
    trajectory = get_word_trajectory(word, word_means, pca)
    return {"trajectory": trajectory, "occurrences": word_occurrences.get(word, [])}


@app.get("/api/topics")
def get_topics_endpoint():
    if top2vec_model is None or pca is None:
        return {"topics": []}
    topics = get_topics(top2vec_model)
    result = []
    for t in topics:
        centroid_3d = pca.transform(np.array([t["centroid"]]))[0]
        doc_mask = top2vec_model.doc_top == t["id"]
        doc_vecs = top2vec_model.document_vectors[doc_mask]
        doc_vecs_3d = pca.transform(doc_vecs) if len(doc_vecs) > 0 else None
        radius = (
            float(
                np.mean(np.linalg.norm(doc_vecs_3d - doc_vecs_3d.mean(axis=0), axis=1))
            )
            if doc_vecs_3d is not None
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
    return {"topics": result}
