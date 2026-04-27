# Topical Semantic Change (TSC) Analysis & Visualization

A comprehensive framework for analyzing lexical semantic change across corpora with integrated topic modeling and interactive 3D visualization.

## Features

- **Lexical Semantic Change Detection**: Quantify how word meanings shift between two corpora using contextual embeddings
- **Interactive Visualization**: 3D visualization of word trajectories and topic clusters in PCA space over time
- **Topic Modeling**: Automatically discover topic clusters in document collections using Top2Vec
- **Multi-Model Support**: Evaluate semantic shift across different transformer models (BERT, RoBERTa, XL-Lexeme, etc.)
- **Layer-wise Analysis**: Compute change metrics across different transformer layers to understand semantic representation
- **Caching System**: Efficient embedding caching to avoid recomputation across runs
- **Scoring Functions**: Multiple metrics for quantifying semantic change (Adaptational Distance, Proximity Ratio)

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

## Installation

### Prerequisites
- Python 3.10+ (for type hints: `int | None`)
- CUDA 11+ (recommended for GPU acceleration)
- Node.js 16+ (for frontend only)

### Backend Setup

```bash
# Clone repository
git clone <repo>
cd topic-lsc

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install dependencies and package
pip install -e .
python -m spacy download en_core_web_sm
```

### Frontend Setup (Optional, for visualization)

```bash
cd topical_semantic_change/frontend
npm install
```

## Usage

### 1. Semantic Change Analysis (LSC Framework)

Evaluate how word meanings change between two corpora using different models:

```bash
# Using the command-line entry point (after pip install -e .)
scoring corpus/semeval2020_ulscd_eng/corpus1 \
  corpus/semeval2020_ulscd_eng/corpus2 \
  --gold corpus/semeval2020_ulscd_eng/truth.csv \
  --models sentence-transformers/all-mpnet-base-v2 pierluigic/xl-lexeme

# Or run as Python module
python -m lexical_semantic_change.assessment.scoring \
  corpus/semeval2020_ulscd_eng/corpus1 \
  corpus/semeval2020_ulscd_eng/corpus2 \
  --gold corpus/semeval2020_ulscd_eng/truth.csv \
  --models sentence-transformers/all-mpnet-base-v2 pierluigic/xl-lexeme
```

**Output**: `eval_results.csv` containing Spearman correlations for each model/layer combination

**Supported Models**:
- `sentence-transformers/all-mpnet-base-v2` (Multilingual, 768-dim)
- `sentence-transformers/all-MiniLM-L6-v2` (Lightweight, 384-dim)
- `pierluigic/xl-lexeme` (English-specific, 512-dim)
- `FacebookAI/roberta-base` (RoBERTa base)
- Any HuggingFace transformer model

### 2. Topical Semantic Change (Interactive Visualization)

The visualization tool analyzes a pair of corpora and displays word trajectories + topic clusters in 3D space.

**Setup**: Create sample corpus data at the expected path:

```bash
# Create directories
mkdir -p topical_semantic_change/backend/data/sample/corpus1
mkdir -p topical_semantic_change/backend/data/sample/corpus2

# Copy or create corpus files (one sentence per line)
cp your_corpus1.txt topical_semantic_change/backend/data/sample/corpus1/
cp your_corpus2.txt topical_semantic_change/backend/data/sample/corpus2/
```

**Start the backend server**:

```bash
cd topical_semantic_change/backend
# Check health before startup (optional)
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**In another terminal, start the frontend dev server**:

```bash
cd topical_semantic_change/frontend
npm run dev
```

**Visit `http://localhost:5173` to explore**:
- **Left panel**: Select words to visualize
- **Center**: 3D plot showing word trajectories (corpus1 → corpus2) and topic clusters
- **Right panel**: Topic information and top words per cluster
- **Bottom**: Example sentences for selected word

**API Endpoints**:
- `GET /api/health` — Server status and error details
- `GET /api/words` — List all available words
- `GET /api/word/{word}` — Word trajectory and sentence occurrences
- `GET /api/topics` — Topic centroids with 3D coordinates and metadata

### 3. Data Format

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

## Scoring Functions

The framework supports multiple metrics for measuring semantic change:

- **ADP (Adaptational Distance)**: Mean pairwise distance between embeddings across corpora
- **PRT (Proximity Ratio)**: Inverse of cosine similarity between mean embeddings
- **Spearman Correlation**: Ranked correlation with gold-standard change scores

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

## License

MIT License (see LICENCE.md)

## Contributing

Contributions welcome! Please ensure:
- Running `scoring.py` with sample corpora completes without errors
- Visualization renders correctly for test data
- Testing with edge cases (empty corpora, single-word terms, etc.)

## Troubleshooting

**Stop words not loading**
- Verify `lexical_semantic_change/extraction/stop_words_en.txt` exists

**Backend startup fails: "corpus path does not exist"**
- Create sample corpus directories: `topical_semantic_change/backend/data/sample/{corpus1,corpus2}/`
- Add plain text files (one sentence per line) to each directory
- Check `GET /api/health` endpoint for detailed error message

**Frontend blank/errors**
- Verify backend is running: `curl http://localhost:8000/api/health`
- Check browser console for errors
- Ensure sample corpus data exists at `topical_semantic_change/backend/data/sample/`

**Out of memory**
- Reduce `batch_size` in `EmbeddingCreator` (default 64 → try 32 or 16)
- Process corpora in chunks
- Use a lighter model (e.g., DistilBERT instead of MPNet)

**Slow embedding computation**
- Verify GPU is in use: `torch.cuda.is_available()` should return `True`
- Enable mixed precision in EmbeddingCreator
- Use a smaller model or reduce `max_seq_length` parameter
