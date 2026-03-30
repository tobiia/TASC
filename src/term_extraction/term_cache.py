import numpy as np

from models import TermEmbeddings, TermSummary


class EmbeddingCache:
    def __init__(self, cache_domain, cache_context):
        self.cache_context = cache_context
        self.path = f"cache_{cache_domain}_{cache_context}.npz"
        self.base_path = f"cache_{cache_domain}"

    def save_cache(self, term_candidates, sentence_cache):
        if self.cache_context == "mean":
            self._save_mean_cache(term_candidates, sentence_cache)
        else:
            self._save_all_cache(term_candidates, sentence_cache)

    def load_cache(self):
        if self.cache_context == "mean":
            return self._load_mean_cache()
        else:
            return self._load_all_cache()

    def _save_mean_cache(self, term_candidates, sentence_cache):
        path = self.base_path + "_mean.npz"

        candidates = list(term_candidates.keys())

        word_embeds = np.vstack([term_candidates[c].word_embed for c in candidates])

        sent_embeds_flat, sent_offsets = self._flatten_sent_embeds(
            term_candidates, candidates
        )

        cache_words, cache_offsets, cache_embeds_flat = self._flatten_sentence_cache(
            sentence_cache
        )

        self._save_common(
            path,
            candidates,
            word_embeds=word_embeds,
            sent_embeds_flat=sent_embeds_flat,
            sent_offsets=sent_offsets,
            cache_words=cache_words,
            cache_offsets=cache_offsets,
            cache_embeds_flat=cache_embeds_flat,
        )

    def _load_mean_cache(self):
        path = self.base_path + "_mean.npz"

        data = self._load_common(path)

        term_candidates = {}

        for i, cand in enumerate(data["candidates"]):
            s_start = data["sent_offsets"][i]
            s_end = data["sent_offsets"][i + 1]

            term_candidates[cand] = TermEmbeddings(
                word_embed=data["word_embeds"][i],
                sent_embeds=data["sent_embeds_flat"][s_start:s_end],
            )

        sentence_cache = self._reconstruct_sentence_cache(
            data["cache_words"],
            data["cache_offsets"],
            data["cache_embeds_flat"],
        )

        return term_candidates, sentence_cache

    def _save_all_cache(self, term_candidates, sentence_cache):
        path = self.base_path + "_all.npz"

        candidates = list(term_candidates.keys())

        word_embeds_flat, word_offsets = self._flatten_word_embeds_all(
            term_candidates, candidates
        )

        sent_embeds_flat, sent_offsets = self._flatten_sent_embeds(
            term_candidates, candidates
        )

        cache_words, cache_offsets, cache_embeds_flat = self._flatten_sentence_cache(
            sentence_cache
        )

        self._save_common(
            path,
            candidates,
            word_embeds_flat=word_embeds_flat,
            word_offsets=word_offsets,
            sent_embeds_flat=sent_embeds_flat,
            sent_offsets=sent_offsets,
            cache_words=cache_words,
            cache_offsets=cache_offsets,
            cache_embeds_flat=cache_embeds_flat,
        )

    def _load_all_cache(self):
        path = self.base_path + "_all.npz"

        data = self._load_common(path)

        term_candidates = {}

        for i, cand in enumerate(data["candidates"]):
            w_start = data["word_offsets"][i]
            w_end = data["word_offsets"][i + 1]

            s_start = data["sent_offsets"][i]
            s_end = data["sent_offsets"][i + 1]

            term_candidates[cand] = TermSummary(
                word_embeds=data["word_embeds_flat"][w_start:w_end],
                sent_embeds=data["sent_embeds_flat"][s_start:s_end],
            )

        sentence_cache = self._reconstruct_sentence_cache(
            data["cache_words"],
            data["cache_offsets"],
            data["cache_embeds_flat"],
        )

        return term_candidates, sentence_cache

    def _save_common(self, path, candidates, **arrays):
        np.savez_compressed(
            path,
            candidate_list=np.array(candidates, dtype=object),
            **arrays,
        )

    def _load_common(self, path):
        data = np.load(path, allow_pickle=True)

        return {
            "candidates": data["candidate_list"],
            **data,
        }

    def _flatten_sent_embeds(self, term_candidates, candidates):
        # list of ndarrays = embeddings for each sentence stacked
        sent_embeds_flat = []
        sent_offsets = [0]

        for c in candidates:
            # ndarray of the sentence embeddings
            se = term_candidates[c].sent_embeds
            sent_embeds_flat.append(se)
            # ex. word_offsets = [0, 5, 8]
            # A (5, 768) -> start = 0, end = 5 -> sent_embeds_flat[0:5]
            # B (3, 768) -> start = 5, end = 8 -> sent_embeds_flat[5:8]
            sent_offsets.append(sent_offsets[-1] + len(se))
        # sent_embeds_flat (8, 768) after vstack
        return np.vstack(sent_embeds_flat), np.array(sent_offsets)

    def _flatten_word_embeds_all(self, term_candidates, candidates):
        word_embeds_flat = []
        word_offsets = [0]

        for c in candidates:
            # the different attribute name is the only reason this exists
            # otherwise works the same as above
            we = term_candidates[c].word_embeds
            word_embeds_flat.append(we)
            word_offsets.append(word_offsets[-1] + len(we))

        return np.vstack(word_embeds_flat), np.array(word_offsets)

    # storing the embeddings for each unique sentence
    def _flatten_sentence_cache(self, sentence_cache):
        cache_words = []
        cache_offsets = [0]
        cache_embeds_flat = []

        # cache keys are integers
        for idx in sorted(sentence_cache.keys(), key=int):
            words, word_embeds = sentence_cache[idx]

            cache_words.extend(words)
            cache_embeds_flat.append(word_embeds)
            cache_offsets.append(cache_offsets[-1] + len(words))

        return (
            np.array(cache_words, dtype=object),
            np.array(cache_offsets),
            np.vstack(cache_embeds_flat),
        )

    def _reconstruct_sentence_cache(self, words, offsets, embeds_flat):
        sentence_cache = {}

        for i in range(len(offsets) - 1):
            start = offsets[i]
            end = offsets[i + 1]

            sentence_cache[i] = (
                list(words[start:end]),
                embeds_flat[start:end],
            )

        return sentence_cache
