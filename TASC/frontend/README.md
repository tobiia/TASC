# Lexical Semantic Change – Frontend

Interactive 3D visualization of word embeddings and topic clusters over time, built with React + Vite + Plotly.

## Setup

```bash
npm install
```

## Development

```bash
npm run dev
```

Starts dev server at `http://localhost:5173`. Requires backend running at `http://localhost:8000`.

## Architecture

### Components

- **App.jsx** – Root component; manages active words, topics, and state
- **PlotCanvas.jsx** – 3D scatter plot showing word trajectories and topic spheres
- **WordList.jsx** – Searchable word selector (left panel)
- **TopicList.jsx** – Topic cluster display with top words (right panel)
- **OccurrenceBar.jsx** – Example sentences for selected word (bottom)
- **Plotly.jsx** – Factory wrapper for react-plotly.js with minified Plotly

### API Integration

**api.js** – Axios client with three endpoints:

- `fetchWords()` → list of all tracked words
- `fetchWord(word)` → trajectory + sentence examples
- `fetchTopics()` → topic clusters with 3D coordinates & radius

### Data Flow

```
App state (activeWords, topics)
    ↓
    ├→ PlotCanvas (renders 3D plot)
    ├→ WordList (user selects words)
    ├→ TopicList (displays clusters)
    └→ OccurrenceBar (shows examples)
```

## Key Features

- **Real-time word selection** – Add/remove words to visualize
- **Interactive 3D plot** – Zoom, pan, rotate; click topics to highlight
- **Sentence examples** – View actual corpus sentences for context
- **Topic exploration** – Visualize document clusters alongside word shifts

## Environment

The frontend expects the backend API at `/api` (proxied in dev, configured in vite.config.js for production).

Customize `BASE` in `api.js` if your backend is on a different host/port.
