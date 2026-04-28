import argparse
import logging
from itertools import product
from pathlib import Path
import numpy as np
import pandas as pd

from ..utils import l2_normalize, load_gold_set, adp, prt, spearmans
from ..representation.embed_cache import run_cache
from ..extraction.word_cache import run_cache as run_word_cache
from ..config import PROJECT_ROOT

from transformers.utils import logging as hf_logging

hf_logging.set_verbosity_error()

from huggingface_hub.utils.tqdm import disable_progress_bars

disable_progress_bars("huggingface_hub")

logger = logging.getLogger(__name__)
# silence httpx 404s and HTTP request logs
logging.getLogger("httpx").setLevel(logging.WARNING)
# silence sentence_transformers
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

# -----------------------------------------------------------
# ANCHOR - config

LAYERS = [None]

# ----------------------------------------------------------------
# ANCHOR - main()


def main():
    parser = argparse.ArgumentParser(
        prog="evaluate configurations for LSC",
        description="compute Spearman correlation across model/layer/scoring-function combinations",
    )
    parser.add_argument("corpus1", help="path to first corpus")
    parser.add_argument("corpus2", help="path to second corpus")
    parser.add_argument(
        "-w", "--words", help="path to chosen/input words", required=True
    )
    parser.add_argument("-m", "--models", nargs="+", help="model names", required=True)
    parser.add_argument("-l", "--label", help="optional label for output file")
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
    logger.info(f"Starting LSC evaluation with label: {args.label or 'default'}")

    # Validate paths
    if not Path(args.corpus1).exists():
        raise FileNotFoundError(f"corpus1 not found: {args.corpus1}")
    if not Path(args.corpus2).exists():
        raise FileNotFoundError(f"corpus2 not found: {args.corpus2}")
    if not Path(args.words).exists():
        raise FileNotFoundError(f"input words not found: {args.words}")

    output_path = (
        PROJECT_ROOT / f"eval_results{'_' + args.label if args.label else ''}.csv"
    )

    logger.info("Setting up...")
    corpora_direct = Path(args.corpus1)
    cache_domain = corpora_direct.parent.name

    try:
        gold_df = load_gold_set(args.words)
    except Exception as e:
        logger.error(f"Failed to load gold standard: {e}", exc_info=True)
        raise

    terms = gold_df["lemma"].tolist()
    logger.info(f"Using {len(terms)} gold standard terms for targeted extraction")

    try:
        x_words, y_words = run_word_cache(
            args.corpus1, args.corpus2, cache_domain, terms=terms
        )
    except Exception as e:
        logger.error(f"Word cache failed: {e}", exc_info=True)
        raise

    if not x_words or not y_words:
        raise ValueError("Word extraction produced empty results")

    gold_series = gold_df.set_index("lemma")["change_graded"]
    words = gold_series.index.tolist()
    logger.info(f"Loaded {len(words)} gold standard words")

    results = []

    for model_name, layer in product(args.models, LAYERS):
        logger.info(f"Model={model_name}; Layer={layer}")
        try:
            (x_embeds, _, _), (y_embeds, _, _) = run_cache(
                x_words, y_words, cache_domain, model_name, layer=layer
            )
        except Exception as e:
            logger.error(
                f"Embedding cache failed for {model_name} layer {layer}: {e}",
                exc_info=True,
            )
            continue

        for fn_name in ("adp", "prt"):
            word_scores = {}
            for word in words:
                if word not in x_embeds or word not in y_embeds:
                    logger.debug(f"Word {word} missing from one corpus")
                    continue
                try:
                    x_ts = x_embeds[word]
                    y_ts = y_embeds[word]
                    if fn_name == "adp":
                        word_scores[word] = adp(x_ts.word_embeds, y_ts.word_embeds)
                    elif fn_name == "prt":
                        x_vec = l2_normalize(
                            np.mean(l2_normalize(x_ts.word_embeds), axis=0)
                        )
                        y_vec = l2_normalize(
                            np.mean(l2_normalize(y_ts.word_embeds), axis=0)
                        )
                        word_scores[word] = prt(x_vec, y_vec)
                except Exception as e:
                    logger.warning(f"Failed to score word {word}: {e}")
                    continue

            if not word_scores:
                logger.warning(
                    f"No scores computed for fn={fn_name}, model={model_name}, layer={layer}"
                )
                continue

            try:
                pred = pd.Series(word_scores)
                gold = gold_series.reindex(pred.index)
                rho = spearmans(pred, gold)
                logger.info(f"  fn={fn_name}    rho={rho:.4f}")
                results.append(
                    {"model": model_name, "layer": layer, "fn": fn_name, "rho": rho}
                )
            except Exception as e:
                logger.error(f"Failed to compute correlation: {e}", exc_info=True)
                continue

    if not results:
        logger.error("No results computed")
        return

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False, mode="a")
    logger.info(f"Results saved to {output_path}")
    print("\n", results_df.to_string(index=False))


if __name__ == "__main__":
    main()
