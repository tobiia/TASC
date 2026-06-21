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

## Requirements
- Python 3.10 <= version <=3.12.10
  - Direct download links for Python 3.12.10: [Windows](https://www.python.org/ftp/python/3.12.10/python-3.12.10-amd64.exe), [Mac](https://www.python.org/ftp/python/3.12.10/python-3.12.10-macos11.pkg)
  - For Python 3.12.10 on linux, run
```bash
sudo apt update
sudo apt-get install python3.12 python3.12-venv
```
  - can use Python 3.13+ if you download [Microsoft Visual C++ 14.0 or greater](https://visualstudio.microsoft.com/visual-cpp-build-tools/)

## Usage

### 1. Topical Semantic Change (Interactive Visualization)

The visualization tool analyzes a pair of corpora and displays word trajectories + topic clusters in 3D space.

#### Quick Installation with Sample Data

- Download the latest release
- Unzip the folder
- If on Windows:
  - Double-click **Install.bat** and allow it to finish installation
  - Double-click **Run.bat**
- If on Linux:
```bash
  chmod +x install.sh run.sh
./install.sh
./run.sh
```
- If on MacOS:
  - Double-click **tasc_install.app** and allow it to finish installation
  - Double-click **tasc_run.app**

You only need to install once. After, you can simply double-click on your Run executable.

**Visit `http://localhost:9000` to explore**:
- **Left panel**: Select words to visualize
- **Center**: 3D plot showing word trajectories (corpus1 → corpus2) and topic clusters
- **Right panel**: Topic information and top words per cluster
- **Bottom**: Example sentences for selected 

The release comes with a sample dataset, the SemEval-2020 Task 1: Unsupervised Lexical Semantic Change Detection benchmark.

If you would like to analyze your own datasets/diachronic corpora in the tool, see [Using Your Own Data](#using-your-own-data).

### 2. Lexical Semantic Change Analysis Framework

Evaluate how word meanings change between two corpora using different models:

```bash

score [first corpus path] \
[second corpus path] \
--gold [optional, graded scores for comparison] \
--models [models to use, can add 1 or multiple]

# example
score corpora/semeval2020_ulscd_eng/corpus1 \
  corpora/semeval2020_ulscd_eng/corpus2 \
  --gold corpora/semeval2020_ulscd_eng/truth.csv \
  --models pierluigic/xl-lexeme sentence-transformers/all-mpnet-base-v2
```

**Output**: `eval_results.csv` containing Spearman correlations for each model/layer combination

**Supported Models**:
- `sentence-transformers/all-mpnet-base-v2` (Multilingual, 768-dim)
- `pierluigic/xl-lexeme` (English-specific, 512-dim)
- `FacebookAI/roberta-base` (RoBERTa base)
- Any HuggingFace transformer model

## Data

TASC generates embeddings from the source corpora and stores them in a cache for efficient reuse. During the initial run, embeddings are computed from the corpus files and saved to the cache directory. Subsequent runs will load the cached embeddings automatically, significantly reducing processing time. Embeddings will only be regenerated if the corresponding cache files are removed or renamed.

### Data Format

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

### Using Your Own Data in TASC

1. Prepare the following files: 
  - Text files(s) composing the first corpus
  - Text files(s) composing the second corpus
  - *(Optional)* A CSV file named `truth.csv` containing at least one column named `lemma`. When provided, TASC will only create embeddings and visualize the listed lemmas. Useful when analyzing a predefined set of target terms or you want to increase the speed of the program.

2. TASC expects a directory structure within the `corpora` directory similar to the following:

  ```text
  corpora/
    datasets1/
    ├── corpus1/
    ├── corpus2/
    └── ...
    datasets2/
    ├── corpus1/
    ├── corpus2/
    ├── truth.csv # optional!
    └── ...
    ...
  ```

  Each corpus should reside in its own dedicated subdirectory within `datasets` directories

1. Create the required directory structure if it does not already exist (the release comes with a sample dataset with the correct structure for your reference)

2. Place your corpus files in the appropriate locations

3. Update the `CORPUS1`, `CORPUS2`, and `TERMS_FILE` (if applicable) variables in `tasc/config.py` to point to the corresponding file paths

```py
# if files are in a directory called "datasets2"
# change variables in config.py to ->
CORPUS1 = str(CORPORA_DIR / "datasets2" / "corpus1")
CORPUS2 = str(CORPORA_DIR / "datasets2" / "corpus2")
TERMS_FILE = str(CORPORA_DIR / "datasets2" / "truth.csv")
   ```

Multiple diachronic datasets may be stored within the `corpora` directory for convenient organization and reuse. Ensure that each corpora directory has a unique name as cache identifiers are derived from these names. When switching to a different dataset, repeat the above setup procedure and be sure to update the configuration paths in config.py (as described above).

## Caching

Embeddings are cached as compressed NPZ files to avoid recomputation:

```
datasets2_c1_sentence-transformers_all-mpnet-base-v2_L8.npz
datasets2_c1_sentence-transformers_all-mpnet-base-v2_L8.npz
top2vec_datasets2.pkl
```

Delete ALL cache files with the same label as your dataset (ex. `datasets2`) to force recomputation.

## Performance Notes

- **Embedding computation**: ~10-50 min per corpus (depending on model & corpus size)
- First run includes transformation & PCA fitting; uses cached results on subsequent runs
- **Memory**: Typical usage 4-8GB for mid-size corpora; increase `batch_size` parameter if memory-constrained
- **GPU**: Significantly faster; ensure CUDA is available via `torch.cuda.is_available()`

## Project Structure

```
.
├── LSC/                              # LSC analysis library
│   ├── pyproject.toml                # Package metadata & dependencies
│   └── lsc/                          # Installable package (import as 'lsc')
│       ├── extraction/               # Word candidate extraction & corpus processing
│       │   ├── word_extractor.py     # Extract common words, filter by POS
│       │   └── word_cache.py         # NPZ caching for extracted words
│       ├── representation/           # Embedding generation & caching
│       │   ├── embedding_creator.py  # Contextual word embeddings from transformers
│       │   ├── embed_cache.py        # Efficient NPZ-based caching
│       │   └── models.py             # Data structures (TermSummary)
│       ├── assessment/               # Semantic change scoring
│       │   └── scoring.py            # Evaluate model/layer combinations
│       ├── utils.py                  # Shared utilities (metrics, normalization)
│       └── config.py                 # Configuration & paths
│
├── Top2Vec/                          # Modified Top2Vec fork (installable)
│   ├── setup.py
│   └── top2vec/
│       └── Top2Vec.py
│
├── tasc/                             # Interactive visualization app
│   ├── cli.py                        # Entry point (tasc run / tasc dev)
│   ├── backend/                      # FastAPI server
│   │   ├── app/
│   │   │   ├── main.py               # API endpoints & lifespan
│   │   │   ├── core.py               # Data loading & PCA fitting
│   │   │   ├── topic.py              # Top2Vec topic extraction
│   │   │   └── config.py             # Backend configuration
│   │   ├── run.py                    # Production server runner
│   │   ├── dev.py                    # Development server runner
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
├── corpora/                          # Datasets
│   └── semeval2020_ulscd_eng/        # Sample data: SemEval2020 Task 1 English
│           ├── corpus1/*.txt         # First corpus files
│           └── corpus2/*.txt         # Second corpus files
│           └── truth.csv             # Graded scores
│
├── pyproject.toml                    # Project metadata, dependencies & CLI entry points
├── install.py                        # Automated installer
├── Install.bat                       # Windows: run to install
└── Run.bat                           # Windows: run to start
```

## Scoring Functions

The framework supports multiple metrics for measuring semantic change:

- **APD (Average Pairwise Distance)**: Mean pairwise distance between embeddings across corpora
- **PRT (Inverted Similarity over Prototype Distance)**: Inverse of cosine similarity between mean embeddings

## License

MIT License (see LICENCE.md)

## Contributing

### Requirements
- Python 3.10 <= version <=3.12.10
- [Git](https://git-scm.com/install/windows)
- [Node.js](https://nodejs.org/en/download/current)

### Contributor Installation

#### Copy from Git

```bash
# clone repository
git clone https://github.com/tobiia/TASC.git
cd TASC

# create virtual environment
py -3.12 -m venv .venv
.venv\Scripts\activate # Windows
# source .venv/bin/activate

# install dependencies and package
python -m pip install --upgrade pip
pip install lib/LSC
pip install lib/Top2Vec
pip install -e .
python -m spacy download en_core_web_sm

# start dev server
tasc dev
```

If you copy the repo using git, you will need to download the sample dataset  as it is too large to upload to Github.

To use the sample dataset:

1. Download the sample cache files [HERE](https://zenodo.org/records/20636728)
2. Create a directory named `cache` in the project root
3. Place all downloaded cache files into the `cache` directory

#### Use the release version

```bash
# download release
# open scripts/
# click dev_install.bat --> creates venv + installs all dependencies

# start dev server
tasc dev
```