import csv
import numpy as np
import pandas as pd
import logging

# FIXME - remove scipy, too big
from scipy.spatial.distance import cdist, cosine
from sklearn.metrics import pairwise_distances
from scipy.stats import spearmanr

logger = logging.getLogger(__name__)

# utility functions


def save_set_to_csv(data_set, file_path):
    with open(file_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["term"])
        for item in sorted(data_set):
            writer.writerow([item])


def load_gold_set(path):
    with open(path, encoding="utf-8") as f:
        first_line = f.readline()
    sep = "\t" if "\t" in first_line else ","
    df = pd.read_csv(path, sep=sep, usecols=["lemma", "change_graded"])
    return df


def l2_normalize(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-9)


def dist(a, b, metric):
    d = cdist(a, b, metric)
    if np.isnan(d).any():
        return pairwise_distances(a, b, metric)
    return d


def apd(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    D = dist(x, y, metric="cosine")  # shape: (|φ1|, |φ2|)

    return float(D.mean())  # == sum / (|φ1||φ2|)


def prt(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mu1 = x.mean(axis=0) if x.ndim > 1 else x
    mu2 = y.mean(axis=0) if y.ndim > 1 else y

    cos_dist = cosine(mu1, mu2)

    return 1.0 / max(cos_dist, 1e-9)


def spearmans(x: pd.Series, y: pd.Series):
    return spearmanr(x, y).statistic


""" 
def cosine_sim(x, y, eps: float = 1e-9):
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)

    if x_arr.shape != y_arr.shape:
        raise ValueError("ERROR - COSIM: input shapes must match")

    if x_arr.ndim == 0:
        x_arr = x_arr.reshape(1)
        y_arr = y_arr.reshape(1)

    dot = np.sum(x_arr * y_arr, axis=-1)
    norm_x = np.linalg.norm(x_arr, axis=-1)
    norm_y = np.linalg.norm(y_arr, axis=-1)
    sim = dot / (norm_x * norm_y + eps)

    if x_arr.ndim == 1:
        return float(sim)
    return sim 
    
def prt(x, y):
    sim = cosine_sim(x, y)
    if sim == 0.0:
        logger.warning("PRT: cosine similarity is exactly 0, clamping to avoid inf")
    return 1.0 / max(sim, 1e-9)


def apd(x, y):
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)

    if x_arr.ndim == 1 and y_arr.ndim == 1:
        return float(np.linalg.norm(x_arr - y_arr))

    # mean of all pairwise distances between rows of x and y
    diff = x_arr[:, None, :] - y_arr[None, :, :]
    return float(np.mean(np.linalg.norm(diff, axis=-1)))


def spearmans(x: pd.Series, y: pd.Series) -> float:
    return x.corr(y, method="spearman") """
