// left panel
// shows all available words + toggle

export default function WordList({ words, activeColorMap, onToggle }) {
  // activeColorMap = obj, { bank: "###", ... }
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
          const isActive = word in activeColorMap;
          const color = activeColorMap[word];

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