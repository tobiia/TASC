// App.jsx
// Root component — owns all state, fetches data, passes props down.
// Layout: left panel (words) | centre (plot + occurrences) | right panel (topics)

import { useState, useEffect } from 'react';
import { fetchWords, fetchWord, fetchTopics } from './api';
import WordList from './components/WordList';
import TopicList from './components/TopicList';
import PlotCanvas from './components/PlotCanvas';
import OccurrenceBar from './components/OccurrenceBar';

// Colours assigned to words in order as they're added
const WORD_COLOURS = ['#946bd6', '#d92970', '#2da5e6', '#621369', '#9ce194', '#BA7517'];

export default function App() {
  // words from backend
  const [allWords, setAllWords] = useState([]);

  // topics
  const [topics, setTopics] = useState([]);

  // { word, colour, trajectory, occurrences }
  const [activeWords, setActiveWords] = useState([]);

  // word whose occurrences are shown in the bottom bar
  const [selectedWord, setSelectedWord] = useState(null);

  // topic currently highlighted or none/null
  const [activeTopic, setActiveTopic] = useState(null);

  // load words + topics
  useEffect(() => {
    fetchWords()
      .then(words => setAllWords(words))
      .catch(err => console.error('Failed to load words:', err));

    fetchTopics()
      .then(data => setTopics(data.topics))
      .catch(err => console.error('Failed to load topics:', err));
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

    // pick next colour
    const usedColours = activeWords.map(w => w.colour);
    const colour = WORD_COLOURS.find(c => !usedColours.includes(c))
      ?? WORD_COLOURS[activeWords.length % WORD_COLOURS.length];

    // add a placeholder entry straight away so the word appears selected
    setActiveWords(prev => [...prev, { word, colour, trajectory: [], occurrences: [] }]);
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
    }
  }

  // toggle a topic highlight on/off
  function handleTopicClick(id) {
    setActiveTopic(prev => prev === id ? null : id);
  }

  // get occurences for selected word
  const selectedOccurrences = activeWords.find(w => w.word === selectedWord)?.occurrences ?? [];

  // colour lookup for WordList, { word: colour }
  const activeColourMap = Object.fromEntries(activeWords.map(w => [w.word, w.colour]));

  return (
    <div className="app">
      <header className="topbar">
        <h1>Lexical Semantic Change</h1>
        <span>word embeddings over time · Top2Vec topic clusters</span>
      </header>

      <div className="main">

        {/* Left panel — list of all words to choose from */}
        <WordList
          words={allWords}
          activeColourMap={activeColourMap}
          onToggle={handleWordToggle}
        />

        {/* Centre — the plot on top, occurrences bar on the bottom */}
        <div className="centre">
          <PlotCanvas
            activeWords={activeWords}
            topics={topics}
            activeTopic={activeTopic}
            onTopicClick={handleTopicClick}
          />
          <OccurrenceBar
            word={selectedWord}
            occurrences={selectedOccurrences}
          />
        </div>

        {/* Right panel — topic list */}
        <TopicList
          topics={topics}
          activeTopic={activeTopic}
          onSelect={handleTopicClick}
        />

      </div>
    </div>
  );
}