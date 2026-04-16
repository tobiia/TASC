from top2vec import Top2Vec


def train_top2vec(documents, model_name):
    return Top2Vec(documents, embedding_model=model_name, speed="learn")


def get_topics(model):
    # returns list[dict[id, words, centroid, size]] for each topic
    n = model.get_num_topics()
    topic_words, _, topic_nums = model.get_topics(num_topics=n)
    topic_sizes, _ = model.get_topic_sizes()

    topics = []
    for i, num in enumerate(topic_nums):
        topics.append(
            {
                "id": int(num),
                "words": topic_words[i][:10].tolist(),
                "centroid": model.topic_vectors[num].tolist(),
                "size": int(topic_sizes[i]),
            }
        )
    return topics
