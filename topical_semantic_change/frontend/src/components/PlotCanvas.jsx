// 3D visualization of word embeddings and topics over time
// Axes: X = PC 1, Y = PC 2, Z = Contextual Entropy (semantic variability)

import { useMemo } from 'react';
import Plot from '../plotly';
import { TOPIC_COLORS } from './TopicList';

const EARLY_COLOR = '#378ADD';  // corpus1 / older period
const LATE_COLOR = '#D85A30';   // corpus2 / newer period

export default function PlotCanvas({ activeWords, topics, documents, activeTopics, onTopicClick }) {
  // activeWords: array of { word, color, trajectory: [{ x, y, z, period }] }
  // documents:   array of { x, y, z, topic, period, text }
  //
  // x = PC 1, y = PC 2, z = contextual entropy

  // Derive period names from active word trajectories, falling back to documents
  const [earlyPeriod, latePeriod] = useMemo(() => {
    const traj = activeWords.find(w => w.trajectory?.length >= 2)?.trajectory;
    if (traj) return [traj[0].period, traj[traj.length - 1].period];
    if (documents?.length > 0) {
      const seen = [...new Set(documents.map(d => d.period).filter(Boolean))].sort();
      if (seen.length >= 2) return [seen[0], seen[1]];
      if (seen.length === 1) return [seen[0], seen[0]];
    }
    return [null, null];
  }, [activeWords, documents]);

  const topicColorMap = useMemo(() =>
    Object.fromEntries(topics.map((t, i) => [t.id, TOPIC_COLORS[i % TOPIC_COLORS.length]])),
    [topics]
  );

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

    // Document scatter — points colored by corpus period, dimmed when a different topic is active
    // Z = contextual entropy (distance from topic centroid)
    if (documents?.length > 0) {
      const docTrace = (docs, opacity, size, showlegend) => ({
        type: 'scatter3d',
        mode: 'markers',
        name: 'documents',
        x: docs.map(d => d.x),
        y: docs.map(d => d.y),
        z: docs.map(d => d.z),
        marker: {
          color: docs.map(d =>
            d.period === earlyPeriod ? EARLY_COLOR :
              d.period === latePeriod ? LATE_COLOR : '#888888'
          ),
          size,
          opacity,
          line: { width: 0 },
        },
        customdata: docs.map(d => ({ type: 'topic', id: d.topic, entropy: d.entropy })),
        hovertemplate: 'Topic %{customdata.id}<br>variability: %{customdata.entropy:.3f}<extra></extra>',
        showlegend,
      });

      if (activeTopics.size > 0) {
        const rest = documents.filter(d => !activeTopics.has(d.topic));
        if (rest.length > 0) result.push(docTrace(rest, 0.1, 3, false));
        for (const topicId of activeTopics) {
          const color = topicColorMap[topicId] ?? '#888';
          const topicDocs = documents.filter(d => d.topic === topicId);
          if (topicDocs.length === 0) continue;
          result.push({
            type: 'scatter3d', mode: 'markers',
            name: `topic ${topicId}`,
            x: topicDocs.map(d => d.x),
            y: topicDocs.map(d => d.y),
            z: topicDocs.map(d => d.z),
            marker: { color, size: 5, opacity: 0.85, line: { width: 0 } },
            customdata: topicDocs.map(d => ({ type: 'topic', id: d.topic, entropy: d.entropy })),
            hovertemplate: 'Topic %{customdata.id}<br>variability: %{customdata.entropy:.3f}<extra></extra>',
            showlegend: false,
          });
        }
      } else {
        result.push(docTrace(documents, 0.4, 4, false));
      }
    }

    // Word trajectories (lines + markers through time)
    // Both time-points share the same Z (entropy is cross-corpus),
    // so the arrow is vertical in the XY semantic plane at a fixed height.
    activeWords.forEach(({ word, color, trajectory }) => {
      if (!trajectory || trajectory.length === 0) return;

      const total = trajectory.length;

      const labelColors = trajectory.map((_, i) => {
        if (i === 0) return EARLY_COLOR;
        if (i === total - 1) return color;
        return '#888';
      });

      const labels = trajectory.map((p, i) => {
        if (i === total - 1) return `${word}  (${p.period})`;
        return p.period;
      });

      // entropy differs per time point now — read from each trajectory point's z
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
        customdata: trajectory.map(p => ({
          type: 'word',
          word,
          period: p.period,
          entropy: p.z,
        })),
        hovertemplate: `<b>${word}</b><br>%{customdata.period}<br>entropy: %{customdata.entropy:.3f}<extra></extra>`,
      });
    });

    return result;
  }, [activeWords, documents, activeTopics, topicColorMap, earlyPeriod, latePeriod]);

  const hasData = activeWords.length > 0 || documents?.length > 0;

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
              eye: { x: 1.4, y: 1.4, z: 0.8 },
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
          if (pt.customdata?.type === 'topic' && pt.customdata.id !== -1) {
            onTopicClick(pt.customdata.id);
          }
        }}
      />
    </div>
  );
}