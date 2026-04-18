// 3D visualization of word embeddings and topics over time

import { useMemo } from 'react';
import Plot from '../plotly';
import { TOPIC_COLORS } from './TopicList';

const EARLY_COLOR = '#378ADD';  // corpus1 / older period
const LATE_COLOR = '#D85A30';  // corpus2 / newer period

export default function PlotCanvas({ activeWords, topics, activeTopic, onTopicClick }) {
  // activeWords: array of { word, color, trajectory: [{ x, y, z, period }] }
  // topics:      array of { id, words, x, y, z, radius }

  // Derive period names from the first active word that has data
  const [earlyPeriod, latePeriod] = useMemo(() => {
    const traj = activeWords.find(w => w.trajectory?.length >= 2)?.trajectory;
    if (!traj) return [null, null];
    return [traj[0].period, traj[traj.length - 1].period];
  }, [activeWords]);

  const traces = useMemo(() => {
    const result = [];

    // Time-period legend markers (invisible geometry, show in legend only)
    if (earlyPeriod && latePeriod) {
      result.push({
        type: 'scatter3d', mode: 'markers',
        name: earlyPeriod,
        x: [null], y: [null], z: [null],
        marker: { color: EARLY_COLOR, size: 9, symbol: 'circle' },
        showlegend: true,
        hoverinfo: 'skip',
      });
      result.push({
        type: 'scatter3d', mode: 'markers',
        name: latePeriod,
        x: [null], y: [null], z: [null],
        marker: { color: LATE_COLOR, size: 9, symbol: 'circle' },
        showlegend: true,
        hoverinfo: 'skip',
      });
    }

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
        textfont: { size: 12, color: '#333' },
        marker: {
          size: topics.map(t => Math.max(6, t.radius * 15)),
          color: topics.map((_, i) => TOPIC_COLORS[i % TOPIC_COLORS.length]),
          opacity: 0.75,
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

      // Color each label: first point = early (blue), last = word color, middle = grey
      const labelColors = trajectory.map((_, i) => {
        if (i === 0) return EARLY_COLOR;
        if (i === total - 1) return color;
        return '#888';
      });

      // Label: first point = period name, last = "word\nperiod", middle = period name
      const labels = trajectory.map((p, i) => {
        if (i === total - 1) return `${word}  (${p.period})`;
        return p.period;
      });

      result.push({
        type: 'scatter3d',
        mode: 'lines+markers+text',
        name: word,
        x: trajectory.map(p => p.x),
        y: trajectory.map(p => p.y),
        z: trajectory.map(p => p.z),
        text: labels,
        textposition: 'top center',
        textfont: {
          size: trajectory.map((_, i) => i === total - 1 ? 13 : 11),
          color: labelColors,
        },
        line: { color, width: 3 },
        marker: {
          color: trajectory.map((_, i) => total <= 1 ? 0 : i / (total - 1)),
          colorscale: [[0, EARLY_COLOR], [1, LATE_COLOR]],
          size: trajectory.map((_, i) => i === total - 1 ? 10 : 8),
          line: { width: 1.5, color: 'white' },
        },
        customdata: trajectory.map(p => ({ type: 'word', word, period: p.period })),
        hovertemplate: `<b>${word}</b><br>%{customdata.period}<extra></extra>`,
      });
    });

    return result;
  }, [activeWords, topics, activeTopic, earlyPeriod, latePeriod]);

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

  const axisStyle = {
    showgrid: true,
    gridcolor: '#888',
    gridwidth: 2,
    zeroline: true,
    zerolinecolor: '#444',
    zerolinewidth: 2,
    showline: true,
    linecolor: '#333',
    linewidth: 3,
    tickfont: { size: 10, color: '#333' },
    titlefont: { size: 11, color: '#222' },
    backgroundcolor: 'rgba(225,223,218,0.6)',
    showbackground: true,
  };

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
            bgcolor: 'rgba(255,255,255,0.92)',
            bordercolor: 'rgba(0,0,0,0.15)',
            borderwidth: 1,
            font: { size: 12, color: '#111' },
          },
          scene: {
            xaxis: { ...axisStyle, title: 'PC 1' },
            yaxis: { ...axisStyle, title: 'PC 2' },
            zaxis: { ...axisStyle, title: 'PC 3' },
            bgcolor: 'rgba(235,233,228,0.3)',
            camera: {
              eye: { x: 1.5, y: 0.1, z: 0.1 },   // looking mostly along Y/Z, X spread is horizontal
              up: { x: 0, y: 0, z: 1 },
              center: { x: 0, y: 0, z: 0 },
            },
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
