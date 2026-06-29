import { useMemo } from 'react';

// right panel
// all Top2Vec topics, lets user highlight one on plot
//
// HALO BEHAVIOUR
// When a word point is clicked in the plot, App.jsx sets "focusedWord"
// TopicList receives "focusedWordNearestTopics" = the nearest_topics entry
// for that word and renders a coloured ring (halo) around the 1st and 2nd
// nearest topic items to visually connect the word to its topic neighbourhood

const TOPIC_COLORS = ["#6bb8ef", "#c61723", "#2e349b", "#9d08d3", "#1be826", "#e29c18", "#b2632e", "#eac327", "#4a9cd3", "#19a5c4", "#11938f", "#233cdd", "#e01186", "#c9dd33", "#f4b369", "#739f5e", "#c91a60", "#70cdff", "#f47aec", "#d67733", "#041f5e", "#26e28a", "#cc395190", "#ce0ecb", "#46127c", "#af3f23", "#a494fd", "#f890c8", "#683312", "#395e1066",];

export default function TopicList({
  topics,
  activeTopics,
  autoTopics,
  onSelect,
  focusedWordNearestTopics,
  focusedWordColor,
  showDocuments,
  onToggleDocuments,
}) {
  // flatten nearest topic ids across both periods, ranked 1st/2nd
  // nearest = { topicId: minRank } where rank 0 = closest
  const nearestRankMap = {};
  if (focusedWordNearestTopics) {
    Object.values(focusedWordNearestTopics).forEach(periodList => {
      periodList.forEach(({ id }, rank) => {
        if (!(id in nearestRankMap) || rank < nearestRankMap[id]) {
          nearestRankMap[id] = rank;
        }
      });
    });
  }

  // building colour map keyed by topic id so colours don't shift on sort
  const topicColorById = useMemo(() =>
    Object.fromEntries(topics.map((t, i) => [t.id, TOPIC_COLORS[i % TOPIC_COLORS.length]])),
    [topics]
  );

  const sortedTopics = useMemo(() => {
    return [...topics].sort((a, b) => {
      const aActive = activeTopics.has(a.id) ? 0 : 1;
      const bActive = activeTopics.has(b.id) ? 0 : 1;
      return aActive - bActive;
    });
  }, [topics, activeTopics]);

  return (
    <div className="panel panel-right">
      <div className="panel-header">
        <div className="panel-title">Topics</div>
        <p>Click to toggle</p>
        <button
          onClick={onToggleDocuments}
          style={{
            marginTop: '6px',
            padding: '3px 8px',
            fontSize: '11px',
            cursor: 'pointer',
            background: showDocuments ? '#e8f0fe' : '#f1f3f4',
            border: '1px solid #ccc',
            borderRadius: '4px',
            color: '#333',
            width: '100%',
            textAlign: 'left',
          }}
        >
          {showDocuments ? '● Full topic clusters' : '○ Only topic centroids'}
        </button>
      </div>

      <div className="panel-body">
        {topics.length === 0 && (
          <div className="empty-state">Loading topics...</div>
        )}

        {sortedTopics.map((topic) => {
          const isActive = activeTopics.has(topic.id);
          const isAuto = autoTopics?.has(topic.id) ?? false;
          const color = topicColorById[topic.id] ?? '#888';
          const nearestRank = nearestRankMap[topic.id];
          const isNearest = nearestRank === 0;
          const isSecond = nearestRank === 1;

          // halo is solid ring for 1st nearest, thinner dashed for 2nd
          const haloStyle = (isNearest || isSecond) && focusedWordColor
            ? {
              outline: `${isNearest ? 2.5 : 1.5}px ${isNearest ? 'solid' : 'dashed'} ${focusedWordColor}`,
              outlineOffset: '2px',
            }
            : {};

          return (
            <div
              key={topic.id}
              className={`topic-item ${isActive ? 'active' : ''}`}
              style={{
                ...(isActive ? { borderColor: color, background: `${color}18` } : {}),
                ...haloStyle,
                cursor: isAuto ? 'default' : 'pointer',
              }}
              onClick={() => !isAuto && onSelect(topic.id)}
            >
              <div className="topic-num" style={{ color }}>
                topic {topic.id}
                {isAuto && (
                  <span style={{ fontSize: '10px', marginLeft: '6px', color: '#999', fontWeight: 400 }}>
                    ⬡ word-linked
                  </span>
                )}
                {isNearest && focusedWordColor && (
                  <span style={{ fontSize: '10px', marginLeft: '6px', color: focusedWordColor }}>
                    ● nearest
                  </span>
                )}
                {isSecond && focusedWordColor && (
                  <span style={{ fontSize: '10px', marginLeft: '6px', color: focusedWordColor }}>
                    ○ 2nd
                  </span>
                )}
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