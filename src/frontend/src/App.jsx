// App.jsx
// root component — owns all state, fetches data, passes props down

import { useState, useEffect, useRef, useMemo, Component } from 'react';
import { fetchWords, fetchWord, fetchTopics, fetchDocuments, fetchHealth, fetchTopicCentroids } from './api';
import WordList from './components/WordList';
import TopicList from './components/TopicList';
import PlotCanvas from './components/PlotCanvas';
import OccurrenceBar from './components/OccurrenceBar';

const WORD_COLORS = ["#6450a8", "#638123", "#573bed", "#914c0f", "#1d7bc3", "#ce1365", "#2a8476", "#db3c23", "#134424", "#d518bd", "#9d5b2e", "#060369", "#807477", "#620932", "#b45a78", "#7e1616"];

class ErrorBoundary extends Component {
  constructor(props) {
    super(props);
    this.state = { error: null };
  }
  static getDerivedStateFromError(error) {
    return { error };
  }
  render() {
    if (this.state.error) {
      return (
        <div className="plot-area">
          <div className="plot-placeholder">
            <div className="plot-placeholder-big">Something went wrong</div>
            <div className="plot-placeholder-small">{String(this.state.error)}</div>
          </div>
        </div>
      );
    }
    return this.props.children;
  }
}

export default function App() {
  // words from backend
  const [allWords, setAllWords] = useState([]);

  // topics
  const [topics, setTopics] = useState([]);

  // topic centroid 3D positions
  const [topicCentroids, setTopicCentroids] = useState([]);

  // document vectors for 3D scatter
  const [documents, setDocuments] = useState([]);

  // { word, color, trajectory, occurrences }
  const [activeWords, setActiveWords] = useState([]);

  // word whose occurrences are shown in the bottom bar
  const [selectedWord, setSelectedWord] = useState(null);

  // word currently focused via plot click
  const [focusedWord, setFocusedWord] = useState(null);

  // whether to show document embeddings or only topic centroids
  const [showDocuments, setShowDocuments] = useState(false);

  // topics derived from active words —> cannot be manually deselected
  const autoTopics = useMemo(() => {
    const ids = new Set();
    activeWords.forEach(({ nearest_topics }) => {
      if (!nearest_topics) return;
      Object.values(nearest_topics).forEach(list =>
        list.forEach(({ id }) => ids.add(id))
      );
    });
    return ids;
  }, [activeWords]);

  // topics the user has manually toggled on — independent of words
  const [manualTopics, setManualTopics] = useState(new Set());

  // final set = union of auto + manual
  const activeTopics = useMemo(() =>
    new Set([...autoTopics, ...manualTopics]),
    [autoTopics, manualTopics]
  );
  const [phase, setPhase] = useState('polling');
  const [statusMessage, setStatusMessage] = useState('Waiting for backend…');
  const [loadError, setLoadError] = useState(null);
  const pollTimer = useRef(null);

  useEffect(() => {
    let cancelled = false;

    async function loadData() {
      try {
        await Promise.all([
          fetchWords().then(words => { if (!cancelled) setAllWords(words); }),
          fetchTopics().then(data => { if (!cancelled) setTopics(data); }),
          fetchDocuments().then(data => { if (!cancelled) setDocuments(data); }),
          fetchTopicCentroids().then(data => { if (!cancelled) setTopicCentroids(data); }),
        ]);
        if (!cancelled) setPhase('ready');
      } catch (err) {
        if (!cancelled) {
          setLoadError(err?.message ?? String(err));
          setPhase('error');
        }
      }
    }

    async function pollHealth() {
      if (cancelled) return;
      const health = await fetchHealth();

      if (cancelled) return;

      if (health.status === 'ready') {
        setPhase('loading');
        setStatusMessage('Loading data…');
        loadData();
      } else if (health.status === 'error') {
        setLoadError(health.error ?? 'Backend failed to initialize');
        setPhase('error');
      } else {
        // "init" or "unavail" = keep polling
        setStatusMessage(
          health.status === 'unavailable'
            ? 'Waiting for backend…'
            : 'Backend is initializing (this may take a few minutes)…'
        );
        pollTimer.current = setTimeout(pollHealth, 3000);
      }
    }

    pollHealth();

    return () => {
      cancelled = true;
      clearTimeout(pollTimer.current);
    };
  }, []);

  // user clicks on word in WORD LIST
  async function handleWordToggle(word) {
    const alreadyActive = activeWords.find(w => w.word === word);

    if (alreadyActive) {
      const remaining = activeWords.filter(w => w.word !== word);
      setActiveWords(remaining);
      if (selectedWord === word)
        setSelectedWord(remaining.length > 0 ? remaining[remaining.length - 1].word : null);
      if (focusedWord === word) setFocusedWord(null);
      return;
    }

    const usedColors = activeWords.map(w => w.color);
    const color = WORD_COLORS.find(c => !usedColors.includes(c))
      ?? WORD_COLORS[activeWords.length % WORD_COLORS.length];

    setActiveWords(prev => [...prev, { word, color, trajectory: [], occurrences: [] }]);
    setSelectedWord(word);

    try {
      const data = await fetchWord(word);
      setActiveWords(prev =>
        prev.map(w =>
          w.word === word
            ? { ...w, trajectory: data.trajectory, occurrences: data.occurrences, nearest_topics: data.nearest_topics ?? {} }
            : w
        )
      );
    } catch (err) {
      console.error(`Failed to load data for "${word}":`, err);
      setActiveWords(prev => prev.filter(w => w.word !== word));
      setSelectedWord(prev => prev === word ? null : prev);
    }
  }

  // toggle a topic manually — only affects manualTopics
  function handleTopicClick(id) {
    if (autoTopics.has(id)) return; // word-driven topics are not manually clickable
    setManualTopics(prev => {
      const next = new Set(prev);
      next.has(id) ? next.delete(id) : next.add(id);
      return next;
    });
  }

  // get occurences for selected word
  const selectedOccurrences = activeWords.find(w => w.word === selectedWord)?.occurrences ?? [];

  // color lookup for WordList, { word: color }
  const activeColorMap = Object.fromEntries(activeWords.map(w => [w.word, w.color]));

  return (
    <div className="app">
      <header className="topbar">
        <h1>Lexical Semantic Change</h1>
        <span>3D PCA visualization of word embeddings over time</span>
      </header>

      {(phase === 'polling' || phase === 'loading') && (
        <div className="loading-overlay">
          <div className="loading-spinner" />
          <span>{statusMessage}</span>
        </div>
      )}

      {phase === 'error' && (
        <div className="loading-overlay">
          <div className="plot-placeholder-big">Failed to load data</div>
          <div className="plot-placeholder-small">{loadError}</div>
        </div>
      )}

      <div className="main">

        {/* left panel = list of all words to choose from */}
        <WordList
          words={allWords}
          activeColorMap={activeColorMap}
          onToggle={handleWordToggle}
          focusedWord={focusedWord}
        />

        {/* centre = the plot on top, occurrences bar on the bottom */}
        <div className="centre">
          <ErrorBoundary>
            <PlotCanvas
              activeWords={activeWords}
              topics={topics}
              topicCentroids={topicCentroids}
              documents={documents}
              activeTopics={activeTopics}
              showDocuments={showDocuments}
              focusedWord={focusedWord}
              onWordSelect={(word) => setFocusedWord(prev => prev === word ? null : word)}
            />
          </ErrorBoundary>
          <OccurrenceBar
            word={selectedWord}
            occurrences={selectedOccurrences}
          />
        </div>

        {/* right panel = topic list */}
        <TopicList
          topics={topics}
          activeTopics={activeTopics}
          autoTopics={autoTopics}
          onSelect={handleTopicClick}
          showDocuments={showDocuments}
          onToggleDocuments={() => setShowDocuments(prev => !prev)}
          focusedWordNearestTopics={
            focusedWord
              ? activeWords.find(w => w.word === focusedWord)?.nearest_topics ?? null
              : null
          }
          focusedWordColor={
            focusedWord ? activeColorMap[focusedWord] ?? null : null
          }
        />

      </div>
    </div>
  );
}