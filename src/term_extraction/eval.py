import csv
import os
import pandas as pd
import csv

from candidate_extractor import EnglishPhraseExtractor

# from term_extractor import TermExtractor


def calculate_metrics(true_terms, extracted_terms):
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

    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1_score)

    return precision, recall, f1_score


def compare_sets(true_terms, extracted_terms, print_results=False):
    true_positives = true_terms.intersection(extracted_terms)
    false_positives = extracted_terms.difference(true_terms)
    false_negatives = true_terms.difference(extracted_terms)

    if print_results:
        # print("True Positives:", true_positives)
        # print("False Positives:", false_positives)
        print("False Negatives:", false_negatives)

    return true_positives, false_positives, false_negatives


# specifically for the ACTER dataset
def evaluation(domain):

    current_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.abspath(os.path.join(current_dir, "..", ".."))

    path = (
        src_dir + "/ACTER/en/" + domain + "/annotated/texts_tokenised"
    )  # unannotated_texts       annotated/texts_tokenised

    # Extract terms as list
    # NOTE CHANGE IF NEEDED
    true_terms = []
    ann_path = (
        src_dir
        + "/ACTER/en/"
        + domain
        + "/annotated/annotations/unique_annotation_lists/"
        + domain
        + "_en_terms.tsv"
    )
    with open(ann_path, "r", newline="") as tsv_file:
        reader = csv.reader(tsv_file, delimiter="\t")
        for row in reader:
            true_terms.append(row[0].lower())

    true_terms_all = set(
        [w.lower().replace("  ", " ").replace("- ", "-") for w in true_terms]
    )
    true_terms_mwe = set(
        [w for w in true_terms_all if (len(w.split(" ")) > 1)]
        + [w for w in true_terms_all if len(w.split("-")) > 1]
    )  # True Phrase Terms
    true_terms_uni = set(
        [w for w in true_terms_all if w not in true_terms_mwe]
    )  # True Unigrams Terms

    print("True terms all: ", len(true_terms_all))
    print("True terms uni: ", len(true_terms_uni))
    print("True terms mwe: ", len(true_terms_mwe))

    """df = pd.read_table(term_path, sep="\t", header=None)
    df.columns = ["term", "label"]
    true_terms = df["term"].to_list()
    # specific_terms = df.loc[df["label"] == "Specific_Term", "term"].to_list()
    # common_terms = df.loc[df["label"] == "Common_Term", "term"].to_list()
    # ood_terms = df.loc[df["label"] == "OOD_Term", "term"].to_list()
    # named_entities = df.loc[df["label"] == "Named_Entity", "term"].to_list()"""

    base_extractor = EnglishPhraseExtractor(path)
    unigrams_dict, ngrams_dict = base_extractor.extract_candidates()
    extracted_terms = list(unigrams_dict.keys()) + list(ngrams_dict.keys())

    E = set(extracted_terms)
    G = set(true_terms)

    # UNSUPERVISED METHODS

    # SUPERVISED METHODS

    precision, recall, f1_score = calculate_metrics(G, E)
    true_positives, false_positives, false_negatives = compare_sets(
        G, E, print_results=False
    )

    # save_set_to_csv(true_positives, "true_positives.csv")
    # save_set_to_csv(false_positives, "false_positives.csv")
    # save_set_to_csv(false_negatives, "false_negatives.csv")


def save_set_to_csv(data_set, file_path):
    with open(file_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["term"])
        for item in sorted(data_set):
            writer.writerow([item])


evaluation("corp")
