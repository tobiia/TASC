// bottom bar
// all sentence occurrences for the currently selected word

export default function OccurrenceBar({ word, occurrences }) {
  return (
    <div className="info-bar">
      <div className="info-bar-header">
        <span>Occurrences</span>
        {word && (
          <span className="selected-word-label">{word}</span>
        )}
      </div>

      <div className="occurrence-list">
        {!word && (
          <div className="empty-state">Select a word to see its occurrences</div>
        )}

        {word && occurrences.length === 0 && (
          <div className="empty-state">No occurrences found for "{word}"</div>
        )}

        {occurrences.map((occ, i) => {
          // occ = { text: "xxxxx", date: "2026" }
          const year = occ.date ? occ.date.slice(0, 4) : '';

          return (
            <div key={i} className="occ-chip" title={occ.text}>
              {year && <span className="occ-date">{year}</span>}
              {occ.text}
            </div>
          );
        })}
      </div>
    </div>
  );
}
