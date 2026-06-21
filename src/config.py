from pathlib import Path
import os
import tomllib

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PACKAGE_ROOT = Path(__file__).resolve().parent

# project paths

CACHE_DIR = PROJECT_ROOT / "cache"
CORPORA_DIR = PROJECT_ROOT / "corpora"

BACKEND = PROJECT_ROOT / "src" / "backend"
FRONTEND = PROJECT_ROOT / "src" / "frontend"
DIST = PROJECT_ROOT / "src" / "frontend_dist"


CONFIG_FILE = PROJECT_ROOT / "config.toml"

with open(CONFIG_FILE, "rb") as f:
    _cfg = tomllib.load(f)

RANDOM_SEED = _cfg["constants"]["random_seed"]

# server

PORT_BACKEND = _cfg["server"]["port_backend"]
PORT_FRONTEND = _cfg["server"]["port_frontend"]

# data

CORPUS1 = PROJECT_ROOT / _cfg["data"]["corpus1"]
CORPUS2 = PROJECT_ROOT / _cfg["data"]["corpus2"]

_terms_file = _cfg["data"]["terms_file"]
TERMS_FILE = None if _terms_file in ("", "None") else PROJECT_ROOT / _terms_file


# model

MODEL_NAME = _cfg["model"]["name"]
LAYER = _cfg["model"]["layer"]


# top2vec

TOP2VEC_MODEL = _cfg["top2vec"]["top2vec_model"]
MAX_TOPIC_SENTENCES = _cfg["top2vec"]["max_topic_sentences"]

_num_topics = _cfg["top2vec"]["top2vec_num_topics"]
TOP2VEC_NUM_TOPICS = None if _num_topics in ("", "None") else int(_num_topics)

_workers = _cfg["top2vec"]["top2vec_num_topics"]
TOP2VEC_WORKERS = os.cpu_count() or 4 if _workers in ("", "None") else int(_workers)

MAX_RENDER_SENTENCES = _cfg["top2vec"]["max_render_sentences"]
