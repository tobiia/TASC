// App.jsx
// Root component — owns all state, fetches data, passes props down.
// Layout: left panel (words) | centre (plot + occurrences) | right panel (topics)

import { useState, useEffect, useRef, Component } from 'react';
import { fetchWords, fetchWord, fetchTopics, fetchDocuments, fetchHealth } from './api';
import WordList from './components/WordList';
import TopicList from './components/TopicList';
import PlotCanvas from './components/PlotCanvas';
import OccurrenceBar from './components/OccurrenceBar';

// Colors assigned to words in order as they're added
const WORD_COLORS = [
  '#946bd6', '#d92970', '#2da5e6', '#621369', '#9ce194', '#BA7517',
  '#E63946', '#2EC4B6', '#F77F00', '#4CC9F0', '#06A77D', '#E76F51',
];

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

  // document vectors for 3D scatter
  const [documents, setDocuments] = useState([]);

  // { word, color, trajectory, occurrences }
  const [activeWords, setActiveWords] = useState([]);

  // word whose occurrences are shown in the bottom bar
  const [selectedWord, setSelectedWord] = useState(null);

  // set of highlighted topic IDs
  const [activeTopics, setActiveTopics] = useState(new Set());

  // 'polling' while waiting for backend | 'loading' while fetching data | 'ready' | 'error'
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
        // 'initializing' or 'unavailable' — keep polling
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
      // removed from plot
      const remaining = activeWords.filter(w => w.word !== word);
      setActiveWords(remaining);

      // if selected in the occurrence bar, switch to another word or clear
      if (selectedWord === word) {
        setSelectedWord(remaining.length > 0 ? remaining[remaining.length - 1].word : null);
      }
      return;
    }
    // else, this is a new word selected

    // pick next color
    const usedColors = activeWords.map(w => w.color);
    const color = WORD_COLORS.find(c => !usedColors.includes(c))
      ?? WORD_COLORS[activeWords.length % WORD_COLORS.length];

    // add a placeholder entry straight away so the word appears selected
    setActiveWords(prev => [...prev, { word, color, trajectory: [], occurrences: [] }]);
    setSelectedWord(word);

    // fetching data to fill in placeholder
    try {
      const data = await fetchWord(word);
      setActiveWords(prev =>
        prev.map(w =>
          w.word === word
            ? { ...w, trajectory: data.trajectory, occurrences: data.occurrences }
            : w
        )
      );
    } catch (err) {
      console.error(`Failed to load data for "${word}":`, err);
      // remove the placeholder so the word doesn't appear stuck as selected
      setActiveWords(prev => prev.filter(w => w.word !== word));
      setSelectedWord(prev => prev === word ? null : prev);
    }
  }

  // toggle a topic highlight on/off
  function handleTopicClick(id) {
    setActiveTopics(prev => {
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
        <h1>TASC - Topic-Aware Semantic Change</h1>
        <span>Word embeddings over 2 time periods · Top2Vec topic clusters · Contextual Entropy as 3rd Dimension</span>
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

        {/* Left panel — list of all words to choose from */}
        <WordList
          words={allWords}
          activeColorMap={activeColorMap}
          onToggle={handleWordToggle}
        />

        {/* Centre — the plot on top, occurrences bar on the bottom */}
        <div className="centre">
          <ErrorBoundary>
            <PlotCanvas
              activeWords={activeWords}
              topics={topics}
              documents={documents}
              activeTopics={activeTopics}
              onTopicClick={handleTopicClick}
            />
          </ErrorBoundary>
          <OccurrenceBar
            word={selectedWord}
            occurrences={selectedOccurrences}
          />
        </div>

        {/* Right panel — topic list */}
        <TopicList
          topics={topics}
          activeTopics={activeTopics}
          onSelect={handleTopicClick}
        />

      </div>
    </div>
  );
}