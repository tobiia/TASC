// 3D visualization of word embeddings and topics over time

import { useMemo } from 'react';
import Plot from '../plotly';
import { TOPIC_COLORS } from './TopicList';

export default function PlotCanvas({ activeWords, topics, activeTopic, onTopicClick }) {
  // activeWords: array of { word, color, trajectory: [{ x, y, z, period }] }
  // topics:      array of { id, words, x, y, z, radius }

  const traces = useMemo(() => {
    const result = [];

    // topic spheres (markers) at centroids
    if (topics.length > 0) {
      result.push({
        type: 'scatter3d',
        mode: 'markers+text',
        name: 'topics',
        x: topics.map(t => t.x),
        y: topics.map(t => t.y),
        z: topics.map(t => t.z),
        text: topics.map(t => `topic ${t.id}`),
        textposition: 'top center',
        textfont: { size: 10 },
        marker: {
          size: topics.map(t => Math.max(5, t.radius * 15)),
          color: topics.map((_, i) => TOPIC_COLORS[i % TOPIC_COLORS.length]),
          opacity: 0.6,
          line: {
            width: topics.map(t => t.id === activeTopic ? 3 : 0),
            color: topics.map((_, i) => TOPIC_COLORS[i % TOPIC_COLORS.length]),
          },
        },
        customdata: topics.map(t => ({ type: 'topic', id: t.id })),
        hovertemplate: '<b>Topic %{customdata.id}</b><extra></extra>',
      });
    }

    // WORD TRAJECTORIES (lines + markers through time)
    activeWords.forEach(({ word, color, trajectory }) => {
      if (!trajectory || trajectory.length === 0) return;

      const total = trajectory.length;

      result.push({
        type: 'scatter3d',
        mode: 'lines+markers+text',
        name: word,
        x: trajectory.map(p => p.x),
        y: trajectory.map(p => p.y),
        z: trajectory.map(p => p.z),
        text: trajectory.map((p, i) => i === total - 1 ? word : p.period),
        textposition: 'top center',
        textfont: {
          size: trajectory.map((_, i) => i === total - 1 ? 12 : 10),
          color: trajectory.map((_, i) => i === total - 1 ? color : '#999'),
        },
        line: { color, width: 2 },
        marker: {
          color: trajectory.map((_, i) => total <= 1 ? 0 : i / (total - 1)),
          colorscale: [[0, '#378ADD'], [1, '#D85A30']],
          size: trajectory.map((_, i) => i === total - 1 ? 8 : 6),
          line: { width: 1, color: 'white' },
        },
        customdata: trajectory.map(p => ({ type: 'word', word, period: p.period })),
        hovertemplate: `<b>${word}</b><br>%{customdata.period}<extra></extra>`,
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
            Embeddings will appear here, colored by time period
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
          legend: {
            x: 0.01,
            y: 0.99,
            bgcolor: 'rgba(255,255,255,0.85)',
            bordercolor: 'rgba(0,0,0,0.1)',
            borderwidth: 1,
            font: { size: 12 },
          },
          scene: {
            xaxis: { showgrid: false, zeroline: false },
            yaxis: { showgrid: false, zeroline: false },
            zaxis: { showgrid: false, zeroline: false },
            bgcolor: 'rgba(240,240,240,0.1)',
          },
          margin: { l: 0, r: 0, t: 0, b: 0 },
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