import os
import csv
import numpy as np
import pandas as pd

"""Utility functions"""


def save_set_to_csv(data_set, file_path):
    with open(file_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["term"])
        for item in sorted(data_set):
            writer.writerow([item])


def append_metrics(file_path, threshold, label, precision, recall, f1_score):
    file_exists = os.path.exists(file_path) and os.path.getsize(file_path) > 0
    with open(file_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["threshold", "label", "precision", "recall", "f1_score"])
        writer.writerow(
            [
                threshold,
                label,
                round(precision, 4),
                round(recall, 4),
                round(f1_score, 4),
            ]
        )


def load_gold_set(path):

    df = pd.read_table(path, sep="\t", header=None, names=["word", "gold_score"])

    # common_terms = df.loc[df["label"] == "Common_Term", "term"].to_list()

    return df


def l2_normalize(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-9)


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
    return 1.0 / cosine_sim(x, y)


def adp(x, y):
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)

    if x_arr.ndim == 1 and y_arr.ndim == 1:
        return float(np.linalg.norm(x_arr - y_arr))

    # mean of all pairwise distances between rows of x and y
    diff = x_arr[:, None, :] - y_arr[None, :, :]
    return float(np.mean(np.linalg.norm(diff, axis=-1)))


def spearmans(x: pd.Series, y: pd.Series) -> float:
    return x.corr(y, method="spearman")
