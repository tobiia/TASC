import axios from 'axios';

const BASE = '/api';

function apiError(label, err) {
  const detail = err?.response?.data?.detail ?? err?.message ?? String(err);
  return new Error(`${label}: ${detail}`);
}

// Returns { status: 'ready' | 'initializing' | 'error', error?, words? }
export async function fetchHealth() {
  try {
    const res = await axios.get(`${BASE}/health`);
    return res.data;
  } catch (err) {
    // Network error or backend not running — treat as not-yet-ready
    return { status: 'unavailable', error: err?.message ?? String(err) };
  }
}

export async function fetchWords() {
  try {
    const res = await axios.get(`${BASE}/words`);
    return res.data.words;
  } catch (err) {
    throw apiError('Failed to load words', err);
  }
}

export async function fetchWord(word) {
  try {
    const res = await axios.get(`${BASE}/word/${encodeURIComponent(word)}`);
    return res.data;
  } catch (err) {
    throw apiError(`Failed to load word "${word}"`, err);
  }
}

export async function fetchTopics() {
  try {
    const res = await axios.get(`${BASE}/topics`);
    return res.data.topics;
  } catch (err) {
    throw apiError('Failed to load topics', err);
  }
}

export async function fetchTopicCentroids() {
  try {
    const res = await axios.get(`${BASE}/topic-centroids`);
    return res.data.topic_centroids;
  } catch (err) {
    throw apiError('Failed to load topic centroids', err);
  }
}

export async function fetchDocuments() {
  try {
    const res = await axios.get(`${BASE}/documents`);
    return res.data.documents;
  } catch (err) {
    throw apiError('Failed to load documents', err);
  }
}