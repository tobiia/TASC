// same function signatures as httpApi.js — reads from the static demo_data.json
// bundled by `npm run build:demo` (see src/backend/app/data/export.py)

let demoDataPromise = null;
function loadDemoData() {
  if (!demoDataPromise) {
    demoDataPromise = fetch(`${import.meta.env.BASE_URL}demo_data.json`).then(res => {
      if (!res.ok) throw new Error(`Failed to fetch demo_data.json: ${res.status}`);
      return res.json();
    });
  }
  return demoDataPromise;
}

// returns { status: "ready", "init", "error" }
export async function fetchHealth() {
  try {
    const data = await loadDemoData();
    return { status: 'ready', words: data.words.length };
  } catch (err) {
    return { status: 'unavailable', error: err?.message ?? String(err) };
  }
}

export async function fetchWords() {
  const data = await loadDemoData();
  return data.words.map(w => w.word);
}

export async function fetchWord(word) {
  const data = await loadDemoData();
  const entry = data.words.find(w => w.word === word);
  if (!entry) throw new Error(`Word "${word}" not found in demo data`);
  return {
    trajectory: entry.trajectory,
    occurrences: entry.occurrences,
    nearest_topics: entry.nearest_topics,
  };
}

export async function fetchTopics() {
  const data = await loadDemoData();
  return data.topics;
}

export async function fetchTopicCentroids() {
  const data = await loadDemoData();
  return data.topic_centroids;
}

export async function fetchDocuments() {
  const data = await loadDemoData();
  return data.documents;
}
