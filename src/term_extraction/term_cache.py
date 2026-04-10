import numpy as np

from term_extraction.models import TermSummary


class EmbeddingCache:
    # corp, equi, htfl, wind
    # uni, uni_vanilla, ngram, ngram_vanilla
    def __init__(self, cache_domain, gram):
        self.path = f"cache_{cache_domain}_{gram}.npz"

    def save_cache(self, term_candidates, sentence_cache, candidates):
        self._save_all_cache(term_candidates, sentence_cache, candidates)

    def load_cache(self):
        return self._load_all_cache()

    def _save_all_cache(self, term_candidates, sentence_cache, candidates):

        cand_keys = list(term_candidates.keys())

        word_embeds_flat, word_offsets = self._flatten_word_embeds_all(
            term_candidates, cand_keys
        )

        sent_embeds_flat, sent_offsets = self._flatten_sent_embeds(
            term_candidates, cand_keys
        )

        cache_words, cache_offsets, cache_embeds_flat = self._flatten_sentence_cache(
            sentence_cache
        )

        cand_dict_keys, cand_dict_values_flat, cand_dict_offsets = (
            self._flatten_candidates(candidates)
        )

        self._save_common(
            self.path,
            cand_keys,
            word_embeds_flat=word_embeds_flat,
            word_offsets=word_offsets,
            sent_embeds_flat=sent_embeds_flat,
            sent_offsets=sent_offsets,
            cache_words=cache_words,
            cache_offsets=cache_offsets,
            cache_embeds_flat=cache_embeds_flat,
            cand_dict_keys=cand_dict_keys,
            cand_dict_values_flat=cand_dict_values_flat,
            cand_dict_offsets=cand_dict_offsets,
        )

    def _load_all_cache(self):

        data = self._load_common(self.path)

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

        candidates = self._reconstruct_candidates(
            data["cand_dict_keys"],
            data["cand_dict_values_flat"],
            data["cand_dict_offsets"],
        )

        return term_candidates, sentence_cache, candidates

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

    def _flatten_candidates(self, candidates):
        keys = list(candidates.keys())
        values_flat = []
        offsets = [0]

        for k in keys:
            sents = candidates[k]
            values_flat.extend(sents)
            offsets.append(offsets[-1] + len(sents))

        return (
            np.array(keys, dtype=object),
            np.array(values_flat, dtype=object),
            np.array(offsets),
        )

    def _reconstruct_candidates(self, keys, values_flat, offsets):
        # short form version of the same to reconstruct word/sent embeds
        return {
            keys[i]: list(values_flat[offsets[i] : offsets[i + 1]])
            for i in range(len(keys))
        }

    def _reconstruct_sentence_cache(self, words, offsets, word_embeds_flat):
        sentence_cache = {}

        for i in range(len(offsets) - 1):
            start = offsets[i]
            end = offsets[i + 1]

            # i = idx of sentence
            sentence_cache[i] = (
                # corresponding word and word embeddings
                list(words[start:end]),
                word_embeds_flat[start:end],
            )

        return sentence_cache
