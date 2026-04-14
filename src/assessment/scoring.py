"""Assessing performance of different models, scores, and layers."""

import argparse
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from utils import l2_normalize, load_gold_set, adp, prt, spearmans
from representation.embed_cache import run_cache
from extraction.word_extractor import extract_common_words

from transformers.utils import logging as hf_logging

hf_logging.set_verbosity_error()

# -----------------------------------------------------------
# ANCHOR - config

parser = argparse.ArgumentParser(
    prog="evaluate configurations for LSC",
    description="compute Spearman correlation across model/layer/scoring-function combinations",
)
parser.add_argument("corpus1", help="path to first corpus")
parser.add_argument("corpus2", help="path to second corpus")
parser.add_argument("-g", "--gold", help="path to gold standard file", required=True)
parser.add_argument("-m", "--models", nargs="+", help="model names", required=True)
parser.add_argument("-l", "--label", help="optional label for output file")
args = parser.parse_args()

OUTPUT_CSV = "combo_results.csv"
LAYERS = range(3, 13)

# ----------------------------------------------------------------
# ANCHOR - main()


def main():
    print("***************** setting up...")
    x_words, y_words = extract_common_words(args.corpus1, args.corpus2)
    x_name = Path(args.corpus1).stem
    y_name = Path(args.corpus2).stem

    gold_df = load_gold_set(args.gold)
    gold_series = gold_df.set_index("lemma")["change_graded"]
    words = gold_series.index.tolist()

    results = []

    for model_name, layer in product(args.models, LAYERS):
        print(f"***************** model={model_name};    layer={layer}")
        x_embeds, _, _ = run_cache(x_words, x_name, model_name, layer=layer)
        y_embeds, _, _ = run_cache(y_words, y_name, model_name, layer=layer)

        for fn_name in ("adp", "prt"):
            word_scores = {}
            for word in words:
                if word not in x_embeds or word not in y_embeds:
                    print(
                        f"ERROR - SCORING: {word} was missing from one of the corpora"
                    )
                    continue
                x_ts = x_embeds[word]
                y_ts = y_embeds[word]
                if fn_name == "adp":
                    word_scores[word] = adp(x_ts.word_embeds, y_ts.word_embeds)
                else:
                    x_vec = l2_normalize(
                        np.mean(l2_normalize(x_ts.word_embeds), axis=0)
                    )
                    y_vec = l2_normalize(
                        np.mean(l2_normalize(y_ts.word_embeds), axis=0)
                    )
                    word_scores[word] = prt(x_vec, y_vec)

            if not word_scores:
                print(
                    f"ERROR - SCORING: no scores computed for fn={fn_name}, model={model_name}, layer={layer}"
                )
                continue

            pred = pd.Series(word_scores)
            gold = gold_series.reindex(pred.index)
            pred = pred.reindex(gold.index)

            rho = spearmans(pred, gold)
            print(f"    fn={fn_name}    rho={rho:.4f}")
            results.append(
                {"model": model_name, "layer": layer, "fn": fn_name, "rho": rho}
            )

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_CSV, index=False)
    print("\n", results_df.to_string(index=False))


if __name__ == "__main__":
    main()
