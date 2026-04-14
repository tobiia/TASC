import { useMemo } from 'react';
import { TOPIC_COLOURS } from './TopicList';

export default function PlotCanvas({ activeWords, topics, activeTopic, onTopicClick }) {
  // activeWords: array of { word, color, trajectory: [{ x, y, period }] }
  // topics:      array of { id, words, x, y, radius }

  // only recalculates when activeWords, topics, or activeTopic change
  const traces = useMemo(() => {
    const result = [];

    // topic hazes
    if (topics.length > 0) {
      result.push({
        type: 'scatter',
        mode: 'markers+text',
        name: 'topics',
        x: topics.map(t => t.x),
        y: topics.map(t => t.y),
        text: topics.map(t => `topic ${t.id}`),
        textposition: 'bottom center',
        textfont: { size: 11 },
        marker: {
          size: topics.map(t => Math.max(40, t.radius * 120)),
          color: topics.map((_, i) => TOPIC_COLOURS[i % TOPIC_COLOURS.length]),
          line: {
            color: topics.map((_, i) => TOPIC_COLOURS[i % TOPIC_COLOURS.length]),
          },
        },
      });
    }

    activeWords.forEach(({ word, color, trajectory }) => {

      const total = trajectory.length;

      result.push({
        type: 'scatter',
        mode: 'lines+markers+text',
        name: word,
        x: trajectory.map(p => p.x),
        y: trajectory.map(p => p.y),
        // show the year on every dot + the word name on the last dot
        text: trajectory.map((p, i) => i === total - 1 ? word : p.period),
        textposition: trajectory.map((_, i) => i === total - 1 ? 'middle right' : 'top center'),
        textfont: {
          // last label is bigger and uses the word colour
          size: trajectory.map((_, i) => i === total - 1 ? 13 : 10),
          // colour each dot indiv w marker.color array
          color: trajectory.map((_, i) => i === total - 1 ? color : '#999'),
        },
        line: { color, width: 1.5 },
      });
    });

    return result;
  }, [activeWords, topics, activeTopic]);

  return (
    <div className="plot-area">
      <Plot
        data={traces}
        layout={{
          autosize: true,
          showlegend: true,
          legend: {
            x: 0.01,
            y: 0.99,
            bgcolor: 'rgba(255,255,255,0.85)',
            bordercolor: 'rgba(0,0,0,0.1)',
            borderwidth: 1,
            font: { size: 12 },
          },
          margin: { l: 40, r: 20, t: 20, b: 40 },
          xaxis: { showgrid: false, zeroline: false, showticklabels: false },
          yaxis: { showgrid: false, zeroline: false, showticklabels: false },
          plot_bgcolor: 'transparent',
          paper_bgcolor: 'transparent',
          hovermode: 'closest',
        }}
        config={{
          displayModeBar: false,
          responsive: true,
        }}
        useResizeHandler
        style={{ width: '100%', height: '100%' }}
        onClick={(e) => {
          if (!e.points.length) return;
          const pt = e.points[0];
          if (pt.customdata?.type === 'topic') {
            onTopicClick(pt.customdata.id);
          }
        }}
      />
    </div>
  );
}