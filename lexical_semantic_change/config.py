from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SRC_DIR = PROJECT_ROOT / "lexical_semantic_change"

MODELS_DIR = PROJECT_ROOT / "models"

CORPUS_DIR = PROJECT_ROOT / "corpus"

CACHE_DIR = PROJECT_ROOT / "cache"

EXTRACT_DIR = SRC_DIR / "extraction"

REP_DIR = SRC_DIR / "representation"

ASSESS_DIR = SRC_DIR / "assessment"
