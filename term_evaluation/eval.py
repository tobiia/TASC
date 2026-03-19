import os
import pandas as pd
from term_extraction.term_extractor import TermExtractor


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
        print("True Positives:", true_positives)
        print("False Positives:", false_positives)
        print("False Negatives:", false_negatives)

    return true_positives, false_positives, false_negatives


def evaluation(domain):
    # ACTER corpus
    corpus_path = "/ACTER/en/" + domain + "/annotated/texts_tokenised"

    # true terms from ACTER
    term_path = (
        "/ACTER/en"
        + domain
        + "/annotated/annotations/unique_annotation_lists/corp_en_tokenised_terms.tsv"
    )

    true_terms = []

    df = pd.read_table(term_path, sep="\t", header=None)
    df.columns = ["term", "label"]
    true_terms = df["term"].to_list()
    specific_terms = df.loc[df["label"] == "Specific_Term", "term"].to_list()
    common_terms = df.loc[df["label"] == "Common_Term", "term"].to_list()
    ood_terms = df.loc[df["label"] == "OOD_Term", "term"].to_list()
    named_entities = df.loc[df["label"] == "Named_Entity", "term"].to_list()

    # MY METHOD
    term_extractor = TermExtractor(corpus_path)
    extracted_terms = term_extractor.extract_terms()

    # UNSUPERVISED METHODS

    # SUPERVISED METHODS

    precision, recall, f1_score = calculate_metrics(true_terms, extracted_terms)
    true_positives, false_positives, false_negatives = compare_sets(
        true_terms, extracted_terms, print_results=True
    )
