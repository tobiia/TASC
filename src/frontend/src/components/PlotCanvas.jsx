// 3D visualization of word embeddings and topics over time
// Axes: X = PC 1, Y = PC 2, Z = PC 3
//
// NEAREST TOPIC LINES
// when a word is active, two lines are drawn from each of its time-period
// points to its 2 nearest topic centroids
// Both appear in the legend under the word they belong to.
//
// WORD SELECTION / HALO
// clicking a word point in the plot calls onWordSelect(word), which App.jsx
// uses to set "focusedWord". TopicList and WordList both receive focusedWord
// and apply a halo ring to the two nearest topics / the word itself.

import { useMemo } from 'react';
import Plot from '../plotly';
import { TOPIC_COLORS } from './TopicList';

/** lighten/move colour toward white by "amount" (0–1). */
function lightenColor(hex, amount = 0.5) {
  const n = parseInt(hex.replace('#', ''), 16);
  const r = Math.round(((n >> 16) & 0xff) + (255 - ((n >> 16) & 0xff)) * amount);
  const g = Math.round(((n >> 8) & 0xff) + (255 - ((n >> 8) & 0xff)) * amount);
  const b = Math.round((n & 0xff) + (255 - (n & 0xff)) * amount);
  return `#${[r, g, b].map(v => v.toString(16).padStart(2, '0')).join('')}`;
}


export default function PlotCanvas({
  activeWords,
  topics,
  topicCentroids,
  documents,
  activeTopics,
  showDocuments,
  focusedWord,     // word string currently focused via plot click
  onWordSelect,
}) {
  const topicColorMap = useMemo(() =>
    Object.fromEntries(topics.map((t, i) => [t.id, TOPIC_COLORS[i % TOPIC_COLORS.length]])),
    [topics]
  );

  // lookup: topicId -> {x, y, z} for drawing nearest-topic lines
  const centroidPosMap = useMemo(() =>
    Object.fromEntries((topicCentroids ?? []).map(c => [c.id, c])),
    [topicCentroids]
  );

  const traces = useMemo(() => {
    const result = [];

    // compute nearest topic ids for the focused word inline so the linter
    // can see the variable is used within the same useMemo scope.
    const nearestTopicIds = (() => {
      if (!focusedWord) return new Set();
      const wordData = activeWords.find(w => w.word === focusedWord);
      if (!wordData?.nearest_topics) return new Set();
      const ids = new Set();
      Object.values(wordData.nearest_topics).forEach(list =>
        list.forEach(({ id }) => ids.add(id))
      );
      return ids;
    })();

    // DOC CLOUD -> only visible when a topic is selected and not hidden ──
    if (showDocuments && documents?.length > 0 && activeTopics.size > 0) {
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
          marker: { color, size: 2, opacity: 1, line: { width: 0 } },
          hovertemplate: `Topic ${topicId}<extra></extra>`,
          showlegend: false,
        });
      }
    }

    // TOPIC CENTROIDS -> visible when:
    //   no words on plot AND no topic clicked, all centroids show
    //   a topic is highlighted via TopicList = that topic's centroid shows
    //   a word is focused via plot click = its 2 nearest centroids show
    if (topicCentroids?.length > 0) {
      const nothingSelected =
        activeTopics.size === 0 &&
        nearestTopicIds.size === 0 &&
        activeWords.length === 0;
      const visibleCentroids = nothingSelected
        ? topicCentroids
        : topicCentroids.filter(c =>
          activeTopics.has(c.id) || nearestTopicIds.has(c.id)
        );

      if (visibleCentroids.length > 0) {
        result.push({
          type: 'scatter3d', mode: 'markers',
          name: 'topic centroids',
          x: visibleCentroids.map(c => c.x),
          y: visibleCentroids.map(c => c.y),
          z: visibleCentroids.map(c => c.z),
          marker: {
            color: visibleCentroids.map(c => lightenColor(topicColorMap[c.id] ?? '#888', 0.55)),
            size: 5,
            opacity: 0.70,
            symbol: 'circle',
            line: {
              width: 1.5,
              color: visibleCentroids.map(c => topicColorMap[c.id] ?? '#888'),
            },
          },
          customdata: visibleCentroids.map(c => ({
            id: c.id,
            words: c.words?.slice(0, 5).join(', ') ?? '',
          })),
          hovertemplate: 'Topic %{customdata.id}<br>%{customdata.words}<extra></extra>',
          showlegend: false,
        });
      }
    }

    // WORD TRAJECTORIES
    activeWords.forEach(({ word, color, trajectory, nearest_topics }) => {
      if (!trajectory || trajectory.length === 0) return;

      const total = trajectory.length;

      const labels = trajectory.map((p, i) =>
        i === total - 1 ? `${word}  (${p.period})` : p.period
      );

      // nearest topic lines + legend
      if (nearest_topics) {
        let addedLegendFirst = false;
        let addedLegendSecond = false;

        trajectory.forEach((pt) => {
          const periodNearest = nearest_topics[pt.period] ?? [];
          periodNearest.forEach(({ id }, rank) => {
            const centroid = centroidPosMap[id];
            if (!centroid) return;
            const topicColor = topicColorMap[id] ?? '#e017c5';
            const isFirst = rank === 0;

            // one legend entry per rank, per word
            const showThisInLegend = isFirst ? !addedLegendFirst : !addedLegendSecond;
            if (isFirst) addedLegendFirst = true;
            else addedLegendSecond = true;

            result.push({
              type: 'scatter3d',
              mode: 'lines',
              name: isFirst
                ? `${word} → nearest sense`
                : `${word} → 2nd sense`,
              x: [pt.x, centroid.x],
              y: [pt.y, centroid.y],
              z: [pt.z, centroid.z],
              line: {
                color: topicColor,
                width: isFirst ? 2.5 : 3.5,
                dash: isFirst ? 'solid' : 'dash',
              },
              opacity: isFirst ? 0.5 : 0.6,
              showlegend: showThisInLegend,
              hoverinfo: 'skip',
            });
          });
        });
      }

      // main word trajectory
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
          size: trajectory.map((_, i) => i === total - 1 ? 14 : 12),
          color: '#111111',
        },
        line: { color, width: 4 },
        marker: {
          color,
          size: trajectory.map((_, i) => i === total - 1 ? 14 : 12),
          line: { width: 5, color: 'black' },
        },
        customdata: trajectory.map(p => {
          const nearestList = nearest_topics?.[p.period] ?? [];
          const t1 = nearestList[0];
          const t2 = nearestList[1];
          return {
            word,
            period: p.period,
            t1_id: t1?.id ?? '—',
            t1_dist: t1 ? t1.distance.toFixed(3) : '—',
            t2_id: t2?.id ?? '—',
            t2_dist: t2 ? t2.distance.toFixed(3) : '—',
          };
        }),
        hovertemplate: [
          `<b>%{customdata.word}</b>  (%{customdata.period})`,
          `nearest sense: %{customdata.t1_id}  (dist %{customdata.t1_dist})`,
          `2nd sense: %{customdata.t2_id}  (dist %{customdata.t2_dist})`,
          `<extra></extra>`,
        ].join('<br>'),
      });
    });

    return result;
  }, [activeWords, documents, topicCentroids, activeTopics, showDocuments,
    topicColorMap, centroidPosMap, focusedWord]);

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

  // Plotly
  const axisStyle = {
    showgrid: true,
    gridcolor: 'white',
    gridwidth: 1,
    zeroline: false,
    showline: false,
    tickfont: { size: 10, color: '#444' },
    titlefont: { size: 11, color: '#333' },
    backgroundcolor: 'rgb(229,236,246)',
    showbackground: true,
    showspikes: false,   // disabling the confusing spike/box lines on hover
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
            bgcolor: 'rgb(229,236,246)',
            camera: {
              eye: { x: 2.0, y: 2.0, z: 1.2 },
              up: { x: 0, y: 0, z: 1 },
              center: { x: 0, y: 0, z: 0 },
              projection: { type: 'perspective' },
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
          if (!e.points?.length) return;
          const pt = e.points[0];
          // only fire for word trajectory points aka customdata.type == word
          if (pt.customdata?.word && onWordSelect) {
            onWordSelect(pt.customdata.word);
          }
        }}
      />
    </div>
  );
}