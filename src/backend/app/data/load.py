from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from top2vec import Top2Vec

from src.backend.app.core import (
    load_data,
    fit_pca,
)
from src.backend.app.topic import (
    train_top2vec,
    get_topics,
    assign_sentence_topics,
)
from src.backend.config import (
    CORPUS1,
    CORPUS2,
    MAX_TOPIC_SENTENCES,
    MAX_RENDER_SENTENCES,
    RANDOM_SEED,
)

import logging

logger = logging.getLogger(__name__)


@dataclass
class AppData:
    word_list: list
    word_means: dict
    word_occurrences: dict

    sentence_period: dict

    all_sentences: list
    sentence_topic: dict
    sentence_embeddings: dict

    documents_payload: list
    topic_centroids_payload: list

    pca: PCA
    top2vec_model: Top2Vec


# FIXME need to ask user for corpora labels/dates
def load_app_data() -> AppData:

    logger.info("Computing or loading data...")

    (
        word_list,
        word_means,
        word_occurrences,
        all_sentences,
        x_embeds,
        y_embeds,
    ) = load_data()

    logger.info(f"Loaded {len(word_list)} words, {len(all_sentences)} sentences")

    rng = np.random.default_rng(seed=RANDOM_SEED)

    if len(all_sentences) > MAX_TOPIC_SENTENCES:
        topic_sentences = [
            all_sentences[i]
            for i in np.sort(
                rng.choice(
                    len(all_sentences),
                    size=MAX_TOPIC_SENTENCES,
                    replace=False,
                )
            )
        ]
    else:
        topic_sentences = all_sentences

    logger.info("Building sentence embedding lookup...")

    sent_embed_lookup = {}

    corpus1_name = Path(CORPUS1).stem
    corpus2_name = Path(CORPUS2).stem
    corpora_label = Path(CORPUS1).parent.name

    for w in word_list:
        for corpus_embeds, corpus_name in [
            (x_embeds, corpus1_name),
            (y_embeds, corpus2_name),
        ]:

            if w not in corpus_embeds:
                continue

            term = corpus_embeds[w]

            occ_sents = [
                o["text"] for o in word_occurrences[w] if o["date"] == corpus_name
            ]

            for sent, emb in zip(occ_sents, term.sent_embeds):
                sent_embed_lookup.setdefault(sent, emb)

    # build ordered embedding matrix matching topic_sentences order
    # Sentences without a cached embedding are omitted — Top2Vec requires
    # precomputed_embeddings to be exactly len(documents) rows, so we
    # filter topic_sentences to only those we have embeddings for.

    covered = [
        (s, sent_embed_lookup[s]) for s in topic_sentences if s in sent_embed_lookup
    ]

    topic_sentences = [s for s, _ in covered]

    sent_embed_matrix = np.vstack([e for _, e in covered])

    # remap topic_indices to match the filtered sentence list
    sent_to_global = {s: i for i, s in enumerate(all_sentences)}

    topic_indices = np.array([sent_to_global[s] for s in topic_sentences])

    logger.info("Training Top2Vec...")

    top2vec_model = train_top2vec(
        topic_sentences,
        corpora_label=corpora_label,
        precomputed_embeddings=sent_embed_matrix,
    )

    logger.info("Assigning sentence topics...")

    raw_topic, raw_embeddings = assign_sentence_topics(
        top2vec_model,
        topic_sentences,
    )

    sentence_topic = {int(topic_indices[i]): t for i, t in raw_topic.items()}

    sentence_embeddings = {int(topic_indices[i]): e for i, e in raw_embeddings.items()}

    sentence_period = {}

    for occ_list in word_occurrences.values():
        for occ in occ_list:
            sentence_period.setdefault(
                occ["text"],
                occ["date"],
            )

    logger.info("Fitting PCA...")
    sent_vecs = np.array(
        [
            sentence_embeddings[i]
            for i in range(len(all_sentences))
            if i in sentence_embeddings
        ]
    )

    pca = fit_pca(
        word_means,
        extra_vecs=sent_vecs,
    )

    logger.info("Projecting topic centroids...")

    topic_centroids_payload = []

    topic_vecs = top2vec_model.topic_vectors  # (n_topics, hidden_dim)

    topic_coords = pca.transform(topic_vecs)  # (n_topics, 3)

    topics = get_topics(top2vec_model)

    words_by_id = {t["id"]: t["words"] for t in topics}

    for i in range(top2vec_model.get_num_topics()):

        topic_centroids_payload.append(
            {
                "id": i,
                "x": float(topic_coords[i][0]),
                "y": float(topic_coords[i][1]),
                "z": float(topic_coords[i][2]),
                "words": words_by_id.get(i, []),
            }
        )

    logger.info("Pre-computing document 3D projections...")
    valid_indices = [i for i in range(len(all_sentences)) if i in sentence_embeddings]

    # random subsample to ease rendering
    if len(valid_indices) > MAX_RENDER_SENTENCES:

        render_indices = sorted(
            rng.choice(
                valid_indices,
                size=MAX_RENDER_SENTENCES,
                replace=False,
            )
        )
    else:
        render_indices = valid_indices

    # project only the sentences we're actually going to render
    vecs = pca.transform(np.array([sentence_embeddings[i] for i in render_indices]))

    documents_payload = []

    for j, idx in enumerate(render_indices):
        # idx = sentence indices

        documents_payload.append(
            {
                "x": float(vecs[j][0]),
                "y": float(vecs[j][1]),
                "z": float(vecs[j][2]),
                "topic": sentence_topic.get(idx, -1),
                "period": sentence_period.get(
                    all_sentences[idx],
                    "",
                ),
                "text": all_sentences[idx],
            }
        )

    return AppData(
        word_list=word_list,
        word_means=word_means,
        word_occurrences=word_occurrences,
        sentence_period=sentence_period,
        all_sentences=all_sentences,
        sentence_topic=sentence_topic,
        sentence_embeddings=sentence_embeddings,
        documents_payload=documents_payload,
        topic_centroids_payload=topic_centroids_payload,
        pca=pca,
        top2vec_model=top2vec_model,
    )
