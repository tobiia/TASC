import argparse
import logging
from pathlib import Path
import pandas as pd

from .utils import adp
from .representation.embed_cache import run_cache
from .extraction.word_cache import run_cache as run_word_cache
from .config import PROJECT_ROOT

from transformers.utils import logging as hf_logging

hf_logging.set_verbosity_error()
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s\n"
)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------
# ANCHOR - config

LAYER = None

# ----------------------------------------------------------------
# ANCHOR - main()


def main():
    parser = argparse.ArgumentParser(
        prog="Measure lexical semantic change over time",
        description="Compute the lexical semantic change of a given set of terms from 2 diachronic corpora",
    )
    parser.add_argument("corpus1", help="path to first corpus")
    parser.add_argument("corpus2", help="path to second corpus")
    parser.add_argument(
        "-w",
        "--words",
        help="path to terms file (CSV with 'lemma' column, or TXT one term per line)",
        required=True,
    )
    parser.add_argument(
        "-m",
        "--model",
        help="HuggingFace model name",
        default="sentence-transformers/all-mpnet-base-v2",
    )
    parser.add_argument("-l", "--label", help="optional label for output file")
    args = parser.parse_args()

    logger.info("Beginning measurement of LEXICAL SEMANTIC CHANGE")

    # Validate paths
    if not Path(args.corpus1).exists():
        raise FileNotFoundError(f"corpus1 not found: {args.corpus1}")
    if not Path(args.corpus2).exists():
        raise FileNotFoundError(f"corpus2 not found: {args.corpus2}")
    if not Path(args.words).exists():
        raise FileNotFoundError(f"words file not found: {args.words}")

    output_path = PROJECT_ROOT / f"results{'_' + args.label if args.label else ''}.csv"

    logger.info("Setting up...")
    corpora_direct = Path(args.corpus1)
    cache_domain = f"{corpora_direct.parent.name}"

    # load terms
    words_path = Path(args.words)
    if words_path.suffix.lower() == ".csv":
        first_line = words_path.read_text(encoding="utf-8").splitlines()[0]
        sep = "\t" if "\t" in first_line else ","
        terms_df = pd.read_csv(words_path, sep=sep)
        terms = terms_df["lemma"].dropna().str.lower().tolist()
    else:
        terms = [
            ln.strip().lower()
            for ln in words_path.read_text(encoding="utf-8").splitlines()
            if ln.strip()
        ]

    logger.info(f"Using {len(terms)} terms for targeted extraction")

    try:
        x_words, y_words = run_word_cache(
            args.corpus1, args.corpus2, cache_domain, terms=terms
        )
    except Exception as e:
        logger.error(f"Word cache failed: {e}", exc_info=True)
        raise

    if not x_words or not y_words:
        raise ValueError("Word extraction produced empty results")

    try:
        (x_embeds, _, _), (y_embeds, _, _) = run_cache(
            x_words, y_words, cache_domain, args.model, layer=LAYER
        )
    except Exception as e:
        logger.error(f"Embedding cache failed: {e}", exc_info=True)
        raise

    word_scores = {}
    for word in terms:
        if word not in x_embeds or word not in y_embeds:
            logger.debug(f"Word '{word}' missing from one corpus")
            continue
        try:
            word_scores[word] = adp(
                x_embeds[word].word_embeds, y_embeds[word].word_embeds
            )
        except Exception as e:
            logger.warning(f"Failed to score word '{word}': {e}")

    if not word_scores:
        logger.warning("No scores computed.")
        return

    results_df = (
        pd.Series(word_scores, name="change_score").rename_axis("lemma").reset_index()
    )
    results_df.to_csv(output_path, index=False)
    logger.info(f"Results saved to {output_path}")
    print("\n", results_df.to_string(index=False))


if __name__ == "__main__":
    main()
