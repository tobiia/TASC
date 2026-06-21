from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]

PACKAGE_ROOT = Path(__file__).resolve().parent

MODELS_DIR = PROJECT_ROOT / "models"

CORPORA_DIR = PROJECT_ROOT / "corpora"

CACHE_DIR = PROJECT_ROOT / "cache"

RESULTS_DIR = PROJECT_ROOT / "results"

EXTRACT_DIR = PACKAGE_ROOT / "extraction"

REP_DIR = PACKAGE_ROOT / "representation"

ASSESS_DIR = PACKAGE_ROOT / "assessment"
