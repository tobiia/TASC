import csv
import os
import csv
from collections import Counter
import pandas as pd

# from english_extractor import EnglishPhraseExtractor as eee
from split_candidate_extractor import CandidateExtractor
from term_extractor import TermExtractor


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
def uni_candidates(domain):

    current_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.abspath(os.path.join(current_dir, "..", ".."))

    path = (
        src_dir + "/ACTER/en/" + domain + "/annotated/texts_tokenised"
    )  # unannotated_texts       annotated/texts_tokenised

    # Extract terms as list
    # NOTE CHANGE IF NEEDED
    ann_path = (
        src_dir
        + "/ACTER/en/"
        + domain
        + "/annotated/annotations/unique_annotation_lists/"
        + domain
        + "_en_terms.tsv"
    )

    df = pd.read_table(ann_path, sep="\t", header=None, names=["term", "label"])
    true_terms = df["term"].to_list()
    specific_terms = df.loc[df["label"] == "Specific_Term", "term"].to_list()
    common_terms = df.loc[df["label"] == "Common_Term", "term"].to_list()
    ood_terms = df.loc[df["label"] == "OOD_Term", "term"].to_list()
    named_entities = df.loc[df["label"] == "Named_Entity", "term"].to_list()

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

    print("********************************************* dataset statistics...")
    print("#### True terms all: ", len(true_terms_all))
    print("####True terms uni: ", len(true_terms_uni))
    print("####True terms mwe: ", len(true_terms_mwe))

    lengths = Counter(len(t.split()) for t in true_terms_mwe)
    print(f"#### lengths of mwe true terms: {lengths}")
    print()

    G = set(true_terms)
    G_uni = set(true_terms_uni)
    S = set(specific_terms)
    C = set(common_terms)
    D = set(ood_terms)
    N = set(named_entities)
    G = G - D - N
    G_uni = G_uni - D - N

    print(
        "********************************************* unigram candidates extracted..."
    )
    base_extractor = CandidateExtractor(path, cohesion_filter=True)
    unigrams_dict = base_extractor.unigram_candidates()
    print(f"#### total number of candidates extracted: {len(unigrams_dict)}")
    E = set(unigrams_dict)
    precision, recall, f1_score = calculate_metrics(G_uni, E)
    print("********************************************* stats for specific terms")
    precision, recall, f1_score = calculate_metrics(G_uni - C, E - C)
    print("********************************************* stats for common terms")
    precision, recall, f1_score = calculate_metrics(G_uni - S, E - S)

    # MWE LENGTHS
    """ lengths = Counter(len(c.split()) for c in ngrams_dict.keys())
    print(f"#### lengths of mwe candidates: {lengths}")
    print() """

    """ print("***************** candidates with filter ON...")
    base_extractor = CandidateExtractor(path)
    unigrams_dict, ngrams_dict = base_extractor.extract_candidates()
    extracted_terms = list(unigrams_dict.keys()) + list(ngrams_dict.keys())
    print(f"############# total number of candidates extracted: {len(extracted_terms)}")
    E = set(extracted_terms)
    precision, recall, f1_score = calculate_metrics(G, E)
    print("********************************************* stats for specific terms")
    precision, recall, f1_score = calculate_metrics(S, E - C)
    print("********************************************* stats for common terms")
    precision, recall, f1_score = calculate_metrics(C, E - S)
    lengths = Counter(len(c.split()) for c in ngrams_dict.keys())
    print(f"#### lengths of mwe candidates: {lengths}")
    print() """

    """ true_positives, false_positives, false_negatives = compare_sets(
        G, E, print_results=False
    ) """

    """ save_set_to_csv(true_positives, "true_positives.csv")
    save_set_to_csv(false_positives, "false_positives.csv")
    save_set_to_csv(false_negatives, "false_negatives.csv") """


def uni_evaluation(domain, contextualized):  # mean or all

    current_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.abspath(os.path.join(current_dir, "..", ".."))

    path = (
        src_dir + "/ACTER/en/" + domain + "/annotated/texts_tokenised"
    )  # unannotated_texts       annotated/texts_tokenised

    # Extract terms as list
    # NOTE CHANGE IF NEEDED
    ann_path = (
        src_dir
        + "/ACTER/en/"
        + domain
        + "/annotated/annotations/unique_annotation_lists/"
        + domain
        + "_en_terms.tsv"
    )

    df = pd.read_table(ann_path, sep="\t", header=None, names=["term", "label"])
    true_terms = df["term"].to_list()
    specific_terms = df.loc[df["label"] == "Specific_Term", "term"].to_list()
    common_terms = df.loc[df["label"] == "Common_Term", "term"].to_list()
    ood_terms = df.loc[df["label"] == "OOD_Term", "term"].to_list()
    named_entities = df.loc[df["label"] == "Named_Entity", "term"].to_list()

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

    print("********************************************* dataset statistics...")
    print("#### True terms all: ", len(true_terms_all))
    print("####True terms uni: ", len(true_terms_uni))
    print("####True terms mwe: ", len(true_terms_mwe))

    lengths = Counter(len(t.split()) for t in true_terms_mwe)
    print(f"#### lengths of mwe true terms: {lengths}")
    print()

    G = set(true_terms)
    G_uni = set(true_terms_uni)
    S = set(specific_terms)
    C = set(common_terms)
    D = set(ood_terms)
    N = set(named_entities)
    G = G - D - N
    G_uni = G_uni - D - N

    # cache_path = f"cache_{contextualized}_{domain}.npz"

    term_extractor = TermExtractor(
        path,
        topic_threshold=0.4,
        context_diff_threshold=0.3,
        self_sim_threshold=0.6,
        ssc_threshold=0,
    )

    # REVIEW - change thresholds here
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

    extracted_uni = term_extractor.uni_test_filter(
        term_extractor.self_similarity,
        thresholds=thresholds,
        domain=domain,
        contextualized_mode=contextualized,
        gram_type="uni",
        use_cache=True,
        compute_ssc=False,
        compute_stop=False,
    )

    print(f"#### total number of candidates extracted: {len(extracted_uni)}")
    E = set(extracted_uni)
    precision, recall, f1_score = calculate_metrics(G_uni, E)
    print("********************************************* stats for specific terms")
    precision, recall, f1_score = calculate_metrics(G_uni - C, E - C)
    print("********************************************* stats for common terms")
    precision, recall, f1_score = calculate_metrics(G_uni - S, E - S)

    true_positives, false_positives, false_negatives = compare_sets(
        G_uni, E, print_results=False
    )

    save_set_to_csv(true_positives, "true_positives.csv")
    save_set_to_csv(false_positives, "false_positives.csv")
    save_set_to_csv(false_negatives, "false_negatives.csv")


def save_set_to_csv(data_set, file_path):
    with open(file_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["term"])
        for item in sorted(data_set):
            writer.writerow([item])


def uni_extraction(domain, contextualized):  # mean or all

    current_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.abspath(os.path.join(current_dir, "..", ".."))

    path = (
        src_dir + "/ACTER/en/" + domain + "/annotated/texts_tokenised"
    )  # unannotated_texts       annotated/texts_tokenised

    # cache_path = f"cache_{contextualized}_{domain}.npz"

    term_extractor = TermExtractor(
        path,
        topic_threshold=0.4,
        context_diff_threshold=0.3,
        self_sim_threshold=0.6,
        ssc_threshold=0,
    )

    # REVIEW - change thresholds here
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

    extracted_uni = term_extractor.uni_test_filter(
        term_extractor.self_similarity,
        thresholds=thresholds,
        domain=domain,
        contextualized_mode=contextualized,
        gram_type="uni",
        use_cache=True,
        compute_ssc=False,
        compute_stop=False,
    )


# uni_candidates("corp")
uni_extraction("corp", "all")
