// MUST USE "COLOR" NOT COLOURS!!!

import { useMemo } from 'react';
import Plot from '../plotly';
import { TOPIC_COLOURS } from './TopicList';

export default function PlotCanvas({ activeWords, topics, activeTopic }) {
  // activeWords: array of { word, color, trajectory: [{ x, y, period }] }
  // topics:      array of { id, words, x, y, radius }

  // only recalculates when activeWords, topics, or activeTopic change
  const traces = useMemo(() => {
    const result = [];

    // topic hazes per centroid
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
          opacity: 0.15,
          line: {
            width: topics.map(t => t.id === activeTopic ? 2 : 0),
            color: topics.map((_, i) => TOPIC_COLOURS[i % TOPIC_COLOURS.length]),
          },
        },
        customdata: topics.map(t => ({ type: 'topic', id: t.id })),
        hovertemplate: '<b>Topic %{customdata.id}</b><extra></extra>',
      });
    }

    // WORD TRAJECTORIES
    activeWords.forEach(({ word, color, trajectory }) => {
      if (!trajectory || trajectory.length === 0) return;

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
        marker: {
          // a value of 0 = blue (early), 1 = red (late)
          color: trajectory.map((_, i) => total <= 1 ? 0 : i / (total - 1)),
          colorscale: [[0, '#378ADD'], [1, '#D85A30']],
          size: trajectory.map((_, i) => i === total - 1 ? 8 : 6),
          line: { width: 1, color: 'white' },
        },
      });
    });

    return result;
  }, [activeWords, topics, activeTopic]);

  const hasData = activeWords.length > 0 || topics.length > 0;

  if (!hasData) {
    return (
      <div className="plot-area">
        <div className="plot-placeholder">
          <div className="plot-placeholder-big">Select words to visualise</div>
          <div className="plot-placeholder-small">
            Embeddings will appear here, coloured by time period
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="plot-area">
      <Plot
        data={traces}
        layout={{
          autosize: true,
          showlegend: true,
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
        }}
      />
    </div>
  );
}