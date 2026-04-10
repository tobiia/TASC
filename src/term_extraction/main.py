"""Combination filter evaluation for unigram terminology extraction.

Precomputes all score dicts once, then grid-searches over threshold
combinations to find the highest F1. Results are ranked and saved to CSV.

Outputs:
    combo_results.csv: all evaluated combinations, sorted by F1 desc
"""

import csv
import itertools
import os
import numpy as np
import pandas as pd
from term_extraction.term_extractor import TermExtractor
from config import PROJECT_ROOT

# ---------------------------------------------------------------------------
# ANCHOR - config

DOMAIN = "corp"
USE_CACHE = True
CORPUS_PATH = (
    PROJECT_ROOT / "corpus" / "ACTER" / "en" / DOMAIN / "annotated" / "texts_tokenised"
)

GOLD_SET_PATH = (
    PROJECT_ROOT
    / "corpus"
    / "ACTER"
    / "en"
    / DOMAIN
    / "annotated"
    / "annotations"
    / "unique_annotation_lists"
    / f"{DOMAIN}_en_tokenised_terms.tsv"
)

# Thresholds to sweep per filter. Adjust granularity as needed.
THRESHOLD_RANGE = np.round(np.arange(0.0, 1.05, 0.1), 2).tolist()

# filters to combine -- >= 4 currently too slow
MAX_COMBO_SIZE = 3

OUTPUT_CSV = "combo_results.csv"


# ---------------------------------------------------------------------------


def main():
    print("********************************************* setting up tests...")
    extractor = TermExtractor(corpus_path=str(CORPUS_PATH))
    # REVIEW - change gold set if need
    gold_set = load_gold_set(str(GOLD_SET_PATH), "uni")

    # load all embeddings from cache
    term_candidates, orig_candidates = extractor.embed_setup(
        domain=DOMAIN,
        gram_type="uni",
        use_cache=USE_CACHE,
        update_sent_cache=True,
    )

    # compute anisotropy baseline from randomly sampled word embeddings
    embeds_list = []
    for info in term_candidates.values():
        embeds_list.append(info.word_embeds)
    if embeds_list:
        all_word_embeds = np.vstack(embeds_list)
        extractor.compute_anisotropy(all_word_embeds)
        print(
            f"##### fine-tuned anisotropy baseline: {extractor.anisotropy_baseline:.4f}"
        )

    # creates non-fine tuned sentence embeds
    vanilla_candidates, _ = extractor.embed_setup(
        domain=DOMAIN,
        gram_type="uni_vanilla",
        use_cache=USE_CACHE,
        model_name=extractor.vanilla_model_name,
        candidates_dict=orig_candidates,
        update_sent_cache=False,
    )

    # creates 1 word embeddings needed for contextualized_vs_general
    general_embeddings = extractor._create_embeddings(list(orig_candidates.keys()))

    # ----------------------------------------------------------------------

    # precompute all score dicts
    print(
        f"******************************************** precomputing candidate scores..."
    )

    all_filters: dict[str, tuple[dict[str, float], bool]] = {}
    # struct: { filter_name: (scores_dict, keep_above) }
    # keep_above=True -> keep terms with score >= threshold
    # keep_above=False -> keep terms with score <= threshold

    # self_similarity: keep >= threshold
    print("--- self_similarity")
    all_filters["self_similarity"] = (
        extractor.self_similarity(term_candidates),
        True,
    )

    # variance: keep <= threshold
    print("--- variance")
    all_filters["variance"] = (
        extractor.variance(term_candidates),
        False,
    )

    # contextualized_vs_general: keep <= threshold
    if general_embeddings is not None:
        print("--- contextualized_vs_general")
        all_filters["contextualized_vs_general"] = (
            extractor.contextualized_vs_general(term_candidates, general_embeddings),
            False,
        )

    if vanilla_candidates is not None:
        # self_similarity_change: keep >= threshold
        print("--- self_similarity_change")
        all_filters["self_similarity_change"] = (
            extractor.self_similarity_change(term_candidates, vanilla_candidates),
            True,
        )

        # domain_vs_general: keep <= threshold
        print("--- domain_vs_general")
        all_filters["domain_vs_general"] = (
            extractor.domain_vs_general(term_candidates, vanilla_candidates),
            False,
        )

    # establishing the baseline
    print("********************************************* dataset statistics...")
    full_candidate_set = set(term_candidates.keys())
    print(f"\ntotal candidates: {len(full_candidate_set)}")
    print(f"gold set size:    {len(gold_set)}")

    base_p, base_r, base_f1 = precision_recall_f1(full_candidate_set, gold_set)
    print(f"baseline (no filter): P={base_p}, R={base_r}, F1={base_f1}\n")

    # ------------------------------------------------------------------
    # grid search over filter combinations + thresholds

    print("********************************************* running grid search...")
    filter_names = list(all_filters.keys())
    results = []

    for combo_size in range(1, MAX_COMBO_SIZE + 1):
        for filter_combo in itertools.combinations(filter_names, combo_size):
            # build all threshold combos for this filter combination
            threshold_grid = itertools.product(*[THRESHOLD_RANGE for _ in filter_combo])
            for thresholds in threshold_grid:
                # intersect: a term must pass ALL filters
                passing = full_candidate_set.copy()
                for fname, threshold in zip(filter_combo, thresholds):
                    scores, keep_above = all_filters[fname]
                    passing &= apply_threshold(scores, threshold, keep_above)

                if not passing:
                    continue

                p, r, f1 = precision_recall_f1(passing, gold_set)

                # gonna save dict to file
                results.append(
                    {
                        "filters": " & ".join(
                            f"{n}>={t}" if all_filters[n][1] else f"{n}<={t}"
                            for n, t in zip(filter_combo, thresholds)
                        ),
                        "n_filters": combo_size,
                        "n_predicted": len(passing),
                        "precision": p,
                        "recall": r,
                        "f1": f1,
                    }
                )

    # ------------------------------------------------------------------
    # saving to file

    results.sort(key=lambda x: x["f1"], reverse=True)

    print(f"\n=== Top 10 combinations by F1 ===")
    for r in results[:10]:
        print(
            f"F1={r['f1']:.4f}  P={r['precision']:.4f}  R={r['recall']:.4f}"
            f"  n={r['n_predicted']}  | {r['filters']}"
        )

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "filters",
                "n_filters",
                "n_predicted",
                "precision",
                "recall",
                "f1",
            ],
        )
        writer.writeheader()
        writer.writerows(results)

    print(f"{len(results)} combinations ran; saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()

# ---------------------------------------------------------------------------
# ANCHOR - helpers


def calculate_metrics(true_terms, extracted_terms, print_results=False):
    true_positives = len(true_terms.intersection(extracted_terms))
    false_positives = len(extracted_terms.difference(true_terms))
    false_negatives = len(true_terms.difference(extracted_terms))

    precision = (
        true_positives / (true_positives + false_positives)
        if (true_positives + false_positives) != 0
        else 0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if (true_positives + false_negatives) != 0
        else 0
    )
    f1_score = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) != 0
        else 0
    )

    if print_results:
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1 Score:", f1_score)

    return precision, recall, f1_score


def compare_sets(true_terms, extracted_terms):
    true_positives = true_terms.intersection(extracted_terms)
    false_positives = extracted_terms.difference(true_terms)
    false_negatives = true_terms.difference(extracted_terms)

    return true_positives, false_positives, false_negatives


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


def load_gold_set(path, wanted_set):

    df = pd.read_table(path, sep="\t", header=None, names=["term", "label"])

    true_terms = set(
        w.lower().replace("  ", " ").replace("- ", "-") for w in df["term"].tolist()
    )

    ood = set(df.loc[df["label"] == "OOD_Term", "term"].str.lower())
    ne = set(df.loc[df["label"] == "Named_Entity", "term"].str.lower())
    """ specific_terms = df.loc[df["label"] == "Specific_Term", "term"].to_list()
    common_terms = df.loc[df["label"] == "Common_Term", "term"].to_list() """

    if wanted_set == "all":
        return true_terms - ood - ne

    true_terms_mwe = set(
        [w for w in true_terms if len(w.split(" ")) > 1]
        + [w for w in true_terms if len(w.split("-")) > 1]
    )
    if wanted_set == "n":
        return true_terms_mwe - ood - ne

    if wanted_set == "uni":
        true_terms_uni = set(w for w in true_terms if w not in true_terms_mwe)
        return true_terms_uni - ood - ne

    return true_terms - ood - ne


def precision_recall_f1(predicted, gold):
    if not predicted:
        return 0.0, 0.0, 0.0
    tp = len(predicted & gold)
    precision = tp / len(predicted)
    recall = tp / len(gold) if gold else 0.0
    f1 = (
        (2 * precision * recall / (precision + recall))
        if (precision + recall) > 0
        else 0.0
    )
    return round(precision, 4), round(recall, 4), round(f1, 4)


def apply_threshold(scores, threshold, keep_above):
    if keep_above:
        return {w for w, s in scores.items() if s >= threshold}
    else:
        return {w for w, s in scores.items() if s <= threshold}
