from pathlib import Path
import os

RANDOM_SEED = 42

PORT_BACKEND = 8000

PORT_FRONTEND = 5173

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SRC_DIR = PROJECT_ROOT / "topical_semantic_change"

CACHE_DIR = PROJECT_ROOT / "cache"

CORPUS_DIR = PROJECT_ROOT / "corpus"

BACKEND = SRC_DIR / "backend"

FRONTEND = SRC_DIR / "frontend"

# ANCHOR - where to set paths for app data

# Optional: path to a CSV or TXT file of terms to search for.
# CSV: must have a "lemma" column (or terms are read from the first column).
# TXT: one term per line.
# When set, extraction is done via fast string matching instead of full NLP.
# None = extract all words (takes an extremely long time for long corpora)
TERMS_FILE = str(CORPUS_DIR / "sample" / "truth.csv")
CORPUS1 = str(CORPUS_DIR / "sample" / "1800s")
CORPUS2 = str(CORPUS_DIR / "sample" / "1900s")
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"


TOP2VEC_MODEL = "all-mpnet-base-v2"
# Maximum number of sentences fed to Top2Vec for topic modelling.
MAX_TOPIC_SENTENCES = 100_000
# Maximum number of sentence points rendered in the 3D plot.
# Sentences beyond this limit are still available in the occurrence bar.
MAX_RENDER_SENTENCES = 10_000
TOP2VEC_NUM_TOPICS = None
TOP2VEC_WORKERS = os.cpu_count() or 4
