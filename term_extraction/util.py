from sklearn.preprocessing import normalize


def l2_normalize(vectors):
    if vectors.ndim == 2:
        return normalize(vectors)
    else:
        return normalize(vectors.reshape(1, -1))[0]
