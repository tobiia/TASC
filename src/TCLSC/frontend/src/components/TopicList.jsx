// right panel
// all Top2Vec topics, lets user highlight one on plot

const TOPIC_COLOURS = ['#1D9E75', '#7F77DD', '#D85A30', '#378ADD', '#BA7517', '#993556'];

export default function TopicList({ topics, activeTopic, onSelect }) {
  return (
    <div className="panel panel-right">
      <div className="panel-header">
        <div className="panel-title">Topics</div>
        <p>Top2Vec clusters</p>
      </div>

      <div className="panel-body">
        {topics.length === 0 && (
          <div className="empty-state">Loading topics...</div>
        )}

        {topics.map((topic, i) => {
          const isActive = activeTopic === topic.id;
          const colour = TOPIC_COLOURS[i % TOPIC_COLOURS.length];

          return (
            <div
              key={topic.id}
              className={`topic-item ${isActive ? 'active' : ''}`}
              onClick={() => onSelect(isActive ? null : topic.id)}
            >
            </div>
          );
        })}
      </div>
    </div>
  );
}

export { TOPIC_COLOURS };
