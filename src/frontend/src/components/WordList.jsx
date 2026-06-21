// left panel
// shows all available words + toggle
//
// HALO BEHAVIOUR
// When a word point is clicked in the plot, App.jsx sets `focusedWord`.
// The corresponding word item gets a coloured ring (halo) so the user
// can see which word's topic connections are currently highlighted.

export default function WordList({ words, activeColorMap, onToggle, focusedWord }) {
  const sortedWords = [...words].sort((a, b) => {
    const aActive = a in activeColorMap ? 0 : 1;
    const bActive = b in activeColorMap ? 0 : 1;
    return aActive - bActive;
  });

  return (
    <div className="panel">
      <div className="panel-header">
        <div className="panel-title">Words</div>
        <p>Click to add to graph</p>
      </div>

      <div className="panel-body">
        {words.length === 0 && (
          <div className="empty-state">Loading words...</div>
        )}

        {sortedWords.map((word) => {
          const isActive = word in activeColorMap;
          const isFocused = word === focusedWord;
          const color = activeColorMap[word];

          return (
            <div
              key={word}
              className={`word-item ${isActive ? 'active' : ''}`}
              style={isFocused && color ? {
                outline: `2px solid ${color}`,
                outlineOffset: '2px',
              } : {}}
              onClick={() => onToggle(word)}
            >
              <span>{word}</span>
              <div
                className="dot"
                style={{
                  background: isActive ? color : 'transparent',
                  border: `1.5px solid ${isActive ? color : 'var(--color-border-secondary, #ccc)'}`,
                }}
              />
            </div>
          );
        })}
      </div>
    </div>
  );
}