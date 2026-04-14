// left panel
// shows all available words + toggle

export default function WordList({ words, activeColourMap, onToggle }) {
  // activeColourMap = obj, { bank: "###", ... }
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

        {words.map((word) => {
          const isActive = word in activeColourMap;
          const colour = activeColourMap[word];

          return (
            <div
              key={word}
              className={`word-item ${isActive ? 'active' : ''}`}
              onClick={() => onToggle(word)}
            >
              <span>{word}</span>
              <div
                className="dot"
                style={{
                  background: isActive ? colour : 'transparent',
                  border: `1.5px solid ${isActive ? colour : 'var(--colour-border-secondary, #ccc)'}`,
                }}
              />
            </div>
          );
        })}
      </div>
    </div>
  );
}