import axios from 'axios';

const BASE = '/api';

// Fetch the list of all tracked words
export async function fetchWords() {
  const res = await axios.get(`${BASE}/words`);
  return res.data.words; // string[]
}

// Fetch trajectory + occurrences for a single word
// Returns:
//   trajectory: [{ period: '2018', x, y, z }]
//   occurrences: [{ text: '...', date: '2018-06-01' }]
export async function fetchWord(word) {
  const res = await axios.get(`${BASE}/word/${encodeURIComponent(word)}`);
  return res.data;
}

// Fetch all topic clusters
// Returns:
//   topics: [{ id, words: string[], x, y, z, radius }]
export async function fetchTopics() {
  const res = await axios.get(`${BASE}/topics`);
  return res.data.topics;
}
