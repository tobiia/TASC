// right panel
// all Top2Vec topics, lets user highlight one on plot

const TOPIC_COLORS = [
  '#1D9E75', '#7F77DD', '#D85A30', '#378ADD', '#BA7517', '#993556',
  '#2EC4B6', '#E76F51', '#8338EC', '#06A77D', '#F4A261', '#457B9D',
  '#E63946', '#A8DADC', '#6A0572', '#F77F00', '#4CC9F0', '#B5179E',
];

export default function TopicList({ topics, activeTopics, onSelect }) {
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
          const isActive = activeTopics.has(topic.id);
          const color = TOPIC_COLORS[i % TOPIC_COLORS.length];

          return (
            <div
              key={topic.id}
              className={`topic-item ${isActive ? 'active' : ''}`}
              style={isActive ? { borderColor: color, background: `${color}18` } : {}}
              onClick={() => onSelect(topic.id)}
            >
              <div className="topic-num" style={{ color }}>
                topic {topic.id}
              </div>
              <div className="topic-words">
                {topic.words.slice(0, 6).join(' · ')}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

export { TOPIC_COLORS };
