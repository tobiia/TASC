# Lexical Semantic Change (LSC) Analysis & Visualization

A comprehensive framework for analyzing lexical semantic change across corpora with integrated topic modeling and interactive 3D visualization.

## Features

- **Semantic Change Detection**: Quantify how word meanings shift between two corpora using contextual embeddings
- **Interactive Visualization**: 3D visualization of word trajectories and topic clusters in PCA space over time
  - **Topic Modeling**: Automatically discover topic clusters in document collections using Top2Vec
- **Multi-Model Support**: Evaluate semantic shift across different transformer models (BERT, RoBERTa, XL-Lexeme, etc.)
- **Layer-wise Analysis**: Compute change metrics across different transformer layers to understand semantic representation
- **Caching System**: Efficient embedding caching to avoid recomputation across runs
- **Scoring Functions**: Multiple metrics for quantifying semantic change (Adaptational Distance, Proximity Ratio)

## Project Structure

```
.
├── lexical_semantic_change/          # LSC analysis framework
│   ├── extraction/                   # Word candidate extraction & corpus processing
│   │   ├── word_extractor.py         # Extract common words, filter by POS
│   ├── representation/               # Embedding generation & caching
│   │   ├── embedding_creator.py      # Contextual word embeddings from transformers
│   │   ├── embed_cache.py            # Efficient NPZ-based caching
│   │   └── models.py                 # Data structures (TermSummary)
│   ├── assessment/                   # Semantic change scoring
│   │   └── scoring.py                # Evaluate model/layer combinations
│   └── config.py                     # Configuration & paths
│
├── topical_semantic_change/          # Visualization tool
│   ├── backend/                      # FastAPI server
│   │   ├── app/
│   │   │   ├── main.py               # API endpoints & lifespan
│   │   │   ├── core.py               # Data loading & PCA fitting
│   │   │   ├── topic.py              # Top2Vec topic extraction
│   │   │   └── config.py             # Backend configuration
│   │   └── requirements.txt
│   ├── frontend/                     # React visualization
│   │   ├── src/
│   │   │   ├── App.jsx               # Root component
│   │   │   ├── api.js                # Backend API client
│   │   │   ├── Plotly.jsx            # Plotly integration
│   │   │   └── components/
│   │   │       ├── PlotCanvas.jsx    # 3D scatter plot
│   │   │       ├── WordList.jsx      # Word selection
│   │   │       ├── TopicList.jsx     # Topic display
│   │   │       └── OccurrenceBar.jsx # Sentence examples
│   │   └── package.json
│
├── corpus/                           # Sample/benchmark datasets
│   └── semeval2020_ulscd_eng/        # SemEval 2020 English LSC data
│
├── utils.py                          # Shared utilities (cosine similarity, metrics)
└── requirements.txt                       # Python dependencies
```

## Installation

### Prerequisites
- Python 3.9+
- CUDA 11+ (recommended for GPU acceleration)
- Node.js 16+ (for frontend)

### Backend Setup

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### Frontend Setup

```bash
cd topical_semantic_change/frontend
npm install
```

## Usage

### 1. Semantic Change Analysis (LSC Framework)

Evaluate how word meanings change between two corpora using different models:

```bash
python lexical_semantic_change/assessment/scoring.py \
  corpus/path/to/corpus1 \
  corpus/path/to/corpus2 \
  --gold path/to/gold_standard.csv \
  --models sentence-transformers/all-mpnet-base-v2 pierluigic/xl-lexeme \
  --label my_experiment
```

**Output**: `combo_results.csv` containing Spearman correlations for each model/layer combination

### 2. Interactive Visualization

Start the backend server:

```bash
cd topical_semantic_change/backend
uvicorn app.main:app --reload
```

In another terminal, start the frontend dev server:

```bash
cd topical_semantic_change/frontend
npm run dev
```

Visit `http://localhost:5173` to explore:
- **Left panel**: Select words to visualize
- **Center**: 3D plot showing word trajectories (corpus1 → corpus2) and topic clusters
- **Right panel**: Topic information and top words per cluster
- **Bottom**: Example sentences for selected word

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

## Model Support

Pre-trained transformer models supported:
- `sentence-transformers/all-mpnet-base-v2` (Multilingual, 768-dim)
- `pierluigic/xl-lexeme` (English-specific, 512-dim)
- `roberta-base`, `bert-base-uncased`, etc.

Extract embeddings from any HuggingFace model via `EmbeddingCreator`:

```python
from lexical_semantic_change.representation.embedding_creator import EmbeddingCreator

creator = EmbeddingCreator(
    corpus={"word_form": ["sentence1", "sentence2", ...]},
    model_name="sentence-transformers/all-mpnet-base-v2",
    token_embedding_layer=8
)
embeddings = creator.create_embeddings()
```

## Caching

Embeddings are cached as compressed NPZ files to avoid recomputation:

```
cache_corpus1_sentence-transformers_all-mpnet-base-v2_L8.npz
cache_corpus2_pierluigic_xl-lexeme_L5.npz
```

Delete cache files to force recomputation.

## Performance Notes

- **Embedding computation**: ~10-50 min per corpus (depends on model & corpus size)
- **First run** includes transformation & PCA fitting; use cached results on subsequent runs
- **Memory**: Typical usage 4-8GB for mid-size corpora; increase `batch_size` parameter if memory-constrained
- **GPU**: Significantly faster; ensure CUDA is available via `torch.cuda.is_available()`

## License

[Add your license here]

## Contributing

Contributions welcome! Please ensure all logic bugs are caught by:
- Running `scoring.py` with sample corpora
- Checking visualization renders correctly for test data
- Testing with edge cases (empty corpora, single-word terms, etc.)

## Troubleshooting

**Stop words not loading**
- Verify `lexical_semantic_change/extraction/stop_words_en.txt` exists or replace with your own file

**Frontend blank/errors**
- Ensure backend is running: `curl http://localhost:8000/api/words`
- Check browser console for errors
- Frontend expects corpus data at `DATA_DIR / "sample" / "corpus{1,2}"`

**Out of memory**
- Reduce `batch_size` in `EmbeddingCreator` (default 64 → try 32 or 16)
- Process corpora in chunks
- Use a lighter model (e.g., DistilBERT instead of MPNet)

**Slow embedding computation**
- Verify GPU is in use: `torch.cuda.is_available()` should return `True`
- Check for CPU-bound bottlenecks with `torch.cuda.synchronize()`
- Enable mixed precision: set `torch.cuda.amp.autocast()`
