# TASC: Topical-Aware Semantic Change

An interactive 3D visualization for analyzing and characterizing lexical semantic change across corpora using topic modeling.

## Features

- **Lexical Semantic Change Detection**: Quantify how word meanings shift between two corpora using contextual embeddings
- **Interactive Visualization**: 3D visualization of word trajectories and topic clusters in PCA space over time
- **Topic Modeling**: Automatically discover topic clusters in document collections using Top2Vec
- **Multi-Model Support**: Evaluate semantic shift across different transformer models (XL-Lexeme, all-mpnet-base-v2, XLM-RoBERTa, etc.)
- **Layer-wise Analysis**: Compute and compare change metrics across different transformer layers
- **Caching System**: Efficient embedding caching to avoid recomputation across runs
- **Scoring Functions**: Multiple metrics for quantifying semantic change (APD, PRT)

## Installation

### Requirements
- Python 3.10+ (for type hints: `int | None`)
- Node.js 16+ (for frontend only)

### 1. Backend Setup

```bash
# clone repository
git clone <repo url>
cd topic-lsc

# create virtual environment
python -m venv .venv
source .venv/bin/activate
# or .venv\Scripts\activate on Windows

# install dependencies and package
pip install -e ./Top2Vec
# if above fails, try: 
cd Top2Vec
pip3 install .

pip install -e .
python -m spacy download en_core_web_sm

# setup frontend
cd TASC/frontend
npm install
```

### 2. Data Setup

TASC generates embeddings from the source corpora and stores them in a cache for efficient reuse. During the initial run, embeddings are computed from the corpus files and saved to the cache directory. Subsequent runs will load the cached embeddings automatically, significantly reducing processing time. Embeddings will only be regenerated if the corresponding cache files are removed or renamed.

#### Using the Sample Data (SemEval-2020 Task 1)

The sample dataset is based on the SemEval-2020 Task 1: Unsupervised Lexical Semantic Change Detection benchmark. It includes a pair of lemmatized English corpora and 37 target lemmas annotated for lexical semantic change between the two time periods.

To use the sample dataset:

1. Download the sample cache files [HERE](https://zenodo.org/records/20636728)
2. Create a directory named `cache` in the project root
3. Place all downloaded cache files into the `cache` directory

#### Using Your Own Data

1. Prepare the following files: 
  - A text file containing the first corpus
  - A text file containing the second corpus
  - *(Optional)* A CSV file named `truth.csv` containing at least one column named `lemma`. When provided, TASC will only create embeddings and visualize the listed lemmas. Useful when analyzing a predefined set of target terms or you want to increase the speed of the program.

2. TASC expects a directory structure within the `corpus` directory similar to the following:

  ```text
  tasc_data/
  ├── corpus1/
  ├── corpus2/
  └── ...
  ```

  Each corpus should reside in its own dedicated subdirectory within `tasc_data`

1. Create the required directory structure manually or run the provided `create_directory.py` script

2. Place your corpus files in the appropriate locations

3. Update the `CORPUS1`, `CORPUS2`, and `TERMS_FILE` (if applicable) variables in `TASC/config.py` to point to the corresponding file paths

```py
# if files are in a directory called "my_data"
# change variables to ->
CORPUS1 = str(CORPUS_DIR / "my_data" / "corpus1")
CORPUS2 = str(CORPUS_DIR / "my_data" / "corpus2")
TERMS_FILE = str(CORPUS_DIR / "my_data" / "truth.csv")
   ```

Multiple diachronic datasets may be stored within the `corpus` directory for convenient organization and reuse. Ensure that each corpora directory has a unique name as cache identifiers are derived from these names. When switching to a different dataset, follow the same setup procedure and be sure to update the configuration paths in config.py (as described above).

## Usage

### 1. Topical Semantic Change (Interactive Visualization)

The visualization tool analyzes a pair of corpora and displays word trajectories + topic clusters in 3D space.

Run:

```bash
pip install -e .
tasc
```

**Visit `http://localhost:5173` to explore**:
- **Left panel**: Select words to visualize
- **Center**: 3D plot showing word trajectories (corpus1 → corpus2) and topic clusters
- **Right panel**: Topic information and top words per cluster
- **Bottom**: Example sentences for selected word

### 2. Semantic Change Analysis (LSC Framework)

Evaluate how word meanings change between two corpora using different models:

```bash
# after pip install -e .
scoring corpus/semeval2020_ulscd_eng/corpus1 \
  corpus/semeval2020_ulscd_eng/corpus2 \
  --gold corpus/semeval2020_ulscd_eng/truth.csv \
  --models sentence-transformers/all-mpnet-base-v2 pierluigic/xl-lexeme
```

**Output**: `eval_results.csv` containing Spearman correlations for each model/layer combination

**Supported Models**:
- `sentence-transformers/all-mpnet-base-v2` (Multilingual, 768-dim)
- `pierluigic/xl-lexeme` (English-specific, 512-dim)
- `FacebookAI/roberta-base` (RoBERTa base)
- Any HuggingFace transformer model

## Data Format

**Corpus Format**: Plain text files, one sentence per line
```
The quick brown fox jumps over the lazy dog.
Machine learning is transforming technology.
...
```

**Gold Standard Format**: Tab-separated CSV with columns `lemma` and `change_graded`
```
lemma	change_graded
word1	0.85
word2	0.12
...
```

## Caching

Embeddings are cached as compressed NPZ files to avoid recomputation:

```
cache_corpus1_corpus2_sentence-transformers_all-mpnet-base-v2_L8.npz
cache_corpus1_corpus2_pierluigic_xl-lexeme_L5.npz
```

Delete cache files to force recomputation.

## Performance Notes

- **Embedding computation**: ~10-50 min per corpus (depends on model & corpus size)
- **First run** includes transformation & PCA fitting; use cached results on subsequent runs
- **Memory**: Typical usage 4-8GB for mid-size corpora; increase `batch_size` parameter if memory-constrained
- **GPU**: Significantly faster; ensure CUDA is available via `torch.cuda.is_available()`

## Project Structure

```
.
├── lexical_semantic_change/          # LSC analysis framework (academic library)
│   ├── extraction/                   # Word candidate extraction & corpus processing
│   │   ├── word_extractor.py         # Extract common words, filter by POS
│   │   └── word_cache.py             # NPZ caching for extracted words
│   ├── representation/               # Embedding generation & caching
│   │   ├── embedding_creator.py      # Contextual word embeddings from transformers
│   │   ├── embed_cache.py            # Efficient NPZ-based caching
│   │   └── models.py                 # Data structures (TermSummary)
│   ├── assessment/                   # Semantic change scoring
│   │   └── scoring.py                # Evaluate model/layer combinations
│   ├── utils.py                      # Shared utilities (metrics, normalization)
│   └── config.py                     # Configuration & paths
│
├── topical_semantic_change/          # Interactive visualization tool
│   ├── backend/                      # FastAPI server
│   │   ├── app/
│   │   │   ├── main.py               # API endpoints & lifespan
│   │   │   ├── core.py               # Data loading & PCA fitting
│   │   │   ├── topic.py              # Top2Vec topic extraction
│   │   │   └── config.py             # Backend configuration
│   │   └── data/sample/              # Sample corpus data (required for backend)
│   │       ├── corpus1/*.txt         # First corpus files
│   │       └── corpus2/*.txt         # Second corpus files
│   └── frontend/                     # React visualization
│       ├── src/
│       │   ├── App.jsx               # Root component
│       │   ├── api.js                # Backend API client
│       │   ├── Plotly.jsx            # Plotly integration
│       │   └── components/
│       │       ├── PlotCanvas.jsx    # 3D scatter plot
│       │       ├── WordList.jsx      # Word selection
│       │       ├── TopicList.jsx     # Topic display
│       │       └── OccurrenceBar.jsx # Sentence examples
│       └── package.json
│
├── corpus/                           # Benchmark datasets (for LSC evaluation only)
│   └── semeval2020_ulscd_eng/        # SemEval 2020 English LSC data
│
└── pyproject.toml                    # Project metadata & dependencies
```

## Scoring Functions

The framework supports multiple metrics for measuring semantic change:

- **ADP (Average Pairwise Distance)**: Mean pairwise distance between embeddings across corpora
- **PRT (Inverted Similarity over Prototype Distance)**: Inverse of cosine similarity between mean embeddings

## License

MIT License (see LICENCE.md)

## Contributing

Contributions welcome! Please ensure:
- Running `scoring.py` with sample corpora completes without errors
- Visualization renders correctly for test data
- Testing with edge cases (empty corpora, single-word terms, etc.)

