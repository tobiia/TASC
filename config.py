from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[0]

SRC_DIR = PROJECT_ROOT / "src"

TERM_PKG = SRC_DIR / "term_extraction"

SIDD_PKG = SRC_DIR / "SIDD"

MODELS = PROJECT_ROOT / "models"

CORPUS = PROJECT_ROOT / "corpus"
