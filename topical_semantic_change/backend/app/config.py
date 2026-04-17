from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]

SRC_DIR = PROJECT_ROOT / "topical_semantic_change"

BACKEND = SRC_DIR / "backend"

FRONTEND = SRC_DIR / "frontend"

DATA_DIR = BACKEND / "data"

# Optional: path to a CSV or TXT file of terms to search for.
# CSV: must have a "lemma" column (or terms are read from the first column).
# TXT: one term per line.
# When set, extraction is done via fast string matching instead of full NLP.
# None = extract all words (takes an extremely long time for long corpora)
TERMS_FILE: Path | None = None
