import os
import numpy as np
from tqdm import tqdm
import torch
from transformers import AutoModel, AutoTokenizer
from collections import defaultdict
from typing import Tuple
from models import CandidateSummary, TermEmbeddings, TermSummary

from candidate_extractor import EnglishPhraseExtractor

# from qdrant_client import QdrantClient
# from qdrant_client.http import models
# from qdrant_client.http.models import CollectionStatus

# NOTE --> can try fasttext?

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, ".."))
domain = "corp"
path = (
    src_dir + "/ACTER/en/" + domain + "/annotated/texts_tokenised"
)  # unannotated_texts       annotated/texts_tokenised


def _l2_normalize(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-9)


def merge_hyphenated(words, word_embeds):
    """Re-join words that BERT split at hyphens/apostrophes into their original form.
    e.g. ["it", "-", "developers"] -> ["it-developers"] with averaged embedding.
    Returns the merged word list and corresponding embeddings."""
    if not words:
        return words, word_embeds
    merged_words = []
    merged_embeds = []
    i = 0
    while i < len(words):
        word = words[i]
        emb_parts = [word_embeds[i]]
        while i + 2 < len(words) and words[i + 1] in ("-", "'"):
            word = word + words[i + 1] + words[i + 2]
            emb_parts.append(word_embeds[i + 1])
            emb_parts.append(word_embeds[i + 2])
            i += 2
        merged_words.append(word)
        merged_embeds.append(np.mean(emb_parts, axis=0))
        i += 1
    return merged_words, np.array(merged_embeds)


class TermExtractor:
    def __init__(
        self,
        corpus_path: str,
        # stop_words,
        model_name="sentence-transformers/all-mpnet-base-v2",
        special_tokens=("<s>", "</s>", "<pad>", "[cls]", "[sep]", "[pad]"),
        max_seq_length=384,
        topic_threshold=0.4,
        self_sim_threshold=0.5,
        context_diff_threshold=0.3,
    ):
        self.corpus_path = corpus_path
        # self.stop_words = stop_words  # stop word list
        self.max_seq_length = max_seq_length
        self.special_tokens = special_tokens

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()

        self.topic_threshold = topic_threshold
        self.self_sim_threshold = self_sim_threshold
        self.context_diff_threshold = context_diff_threshold

    # Dict: [cand, list[sentences]]
    def extract_candidates(self) -> Tuple[dict[str, list[str]], dict[str, list[str]]]:
        candidate_extractor = EnglishPhraseExtractor(path=self.corpus_path)
        unigram_candidates, ngram_candidates = candidate_extractor.extract_candidates()

        return unigram_candidates, ngram_candidates

    def l2_normalize(self, x):
        return _l2_normalize(x)

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    # input: Dict: [cand, list[sentences]]
    # output: Dict: [str, CandidateSummary(tokenized sentences, token embeddings, sentence embeddings)]
    def encode(
        self, candidates: dict[str, list[str]], model_name: str | None = None
    ) -> dict[str, CandidateSummary]:

        if model_name:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name)
            model.eval()
        else:
            tokenizer = self.tokenizer
            model = self.model

        all_sentences = []
        candidate_map = []

        for candidate, sentences in candidates.items():
            for s in sentences:
                all_sentences.append(s)
                # [A, A, A, B, B, C, ...]
                candidate_map.append(candidate)

        if not all_sentences:
            return {}

        # Tokenize without padding to get lengths for sorting, then sort so
        # sentences of similar length land in the same mini-batch, minimising padding waste.
        lengths = [
            len(ids)
            for ids in tokenizer(
                all_sentences, truncation=True, max_length=self.max_seq_length
            )["input_ids"]
        ]
        sorted_order = np.argsort(lengths).tolist()
        sentences_sorted = [all_sentences[i] for i in sorted_order]
        cand_map_sorted = [candidate_map[i] for i in sorted_order]

        batch_size = 64
        tok_embeds_flat = (
            []
        )  # token embeds for each sentence, indices = sentence, list of (seq_len, H = hidden dim) arrays
        all_sentence_embeddings = []
        all_tokens_list = []

        for batch_start in range(0, len(sentences_sorted), batch_size):
            batch_sents = sentences_sorted[batch_start : batch_start + batch_size]
            encoded_input = tokenizer(
                batch_sents,
                padding=True,
                truncation=True,
                max_length=self.max_seq_length,
                return_tensors="pt",
            )
            with torch.no_grad():
                model_output = model(**encoded_input)
            # token embeddings, 1 for each token that makes up a sentence
            batch_tok = model_output.last_hidden_state.cpu().numpy()  # (B, seq_len, H)
            tok_embeds_flat.extend(batch_tok[i] for i in range(batch_tok.shape[0]))

            sent_emb = self.mean_pooling(model_output, encoded_input["attention_mask"])
            all_sentence_embeddings.append(sent_emb.cpu().numpy())
            # input_ids = list[list of tokens(num) for each sentence]
            # ids = 1 row of tokens = 1 sentence
            # extends adds each element as its generated
            # all_tokens_list = list[list of tokens(text) for each sentence]
            all_tokens_list.extend(
                tokenizer.convert_ids_to_tokens(ids)
                for ids in encoded_input["input_ids"]
            )

        sentence_embeddings = self.l2_normalize(
            np.concatenate(all_sentence_embeddings, axis=0)
        )

        # Group by candidate — indices into sorted arrays; order within a candidate doesn't matter.
        groups = defaultdict(list)
        for j, cand in enumerate(cand_map_sorted):
            groups[cand].append(j)

        encoded_candidates = {}
        # indices = list of indices
        for cand, indices in groups.items():
            idx = np.array(indices)
            encoded_candidates[cand] = CandidateSummary(
                tok_sents=[all_tokens_list[j] for j in indices],
                tok_embeds=[
                    tok_embeds_flat[j] for j in indices
                ],  # list of (seq_len, H)
                # fancy indexing -> [[1,2,...]]
                sent_embeds=sentence_embeddings[idx],  # (num_sentences, H)
            )

        return encoded_candidates

    # input: Dict: [cand, list[sentences]]
    # output: Dict: [str, TermEmbeddings(word embedding, slist[sentence embedding(s)])]
    def encode_general(
        self, candidates: dict[str, list[str]]
    ) -> dict[str, TermEmbeddings]:
        all_sentences = []
        candidate_map = []
        indiv_candidates = list(candidates.keys())

        # str, list of str
        for candidate, sentences in candidates.items():
            for s in sentences:
                all_sentences.append(s)
                candidate_map.append(candidate)

        encoded_input_sent = self.tokenizer(
            all_sentences,
            padding=True,
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors="pt",
        )

        encoded_input_cand = self.tokenizer(
            indiv_candidates,
            padding=True,
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors="pt",
        )

        with torch.no_grad():
            model_output_sent = self.model(**encoded_input_sent)
            model_output_cand = self.model(**encoded_input_cand)

        sentence_embeddings = self.mean_pooling(
            model_output_sent, encoded_input_sent["attention_mask"]
        )
        sentence_embeddings = sentence_embeddings.cpu().numpy()
        sentence_embeddings = self.l2_normalize(sentence_embeddings)

        candidate_embeddings = self.mean_pooling(
            model_output_cand, encoded_input_cand["attention_mask"]
        )
        candidate_embeddings = candidate_embeddings.cpu().numpy()
        candidate_embeddings = self.l2_normalize(candidate_embeddings)

        # Build index groups per candidate (handles unordered dict)
        groups = defaultdict(list)
        for i, cand in enumerate(candidate_map):
            groups[cand].append(i)

        encoded_candidates = {}
        for i, cand in enumerate(indiv_candidates):
            indices = np.array(groups[cand])
            encoded_candidates[cand] = TermEmbeddings(
                word_embed=candidate_embeddings[i],  # (hidden_dim,)
                sent_embeds=sentence_embeddings[indices],  # (num_sentences, hidden_dim)
            )

        return encoded_candidates

    # input: Dict: [str, CandidateSummary]
    # output: Dict: [str, TermEmbeddings(contextualized embedding, sentence embed(s))]
    def create_word_embeddings(
        self,
        encoded_candidates: dict[str, CandidateSummary],
        mode: str = "mean",
    ) -> dict[str, TermEmbeddings | TermSummary]:

        candidate_embeddings = {}
        for candidate, info in encoded_candidates.items():
            _, emb = self.token_to_word(candidate, info, mode)
            if emb is not None:
                candidate_embeddings[candidate] = emb
        return candidate_embeddings

    # input: a single candidate, ngram or unigram
    def token_to_word(self, candidate: str, info: CandidateSummary, mode: str):
        try:
            all_embeds = []
            # needed to deal with ngrams -> must work on each sub-word
            # ex. corp-greed -> [corp-greed]
            # ex. criminal activity -> [criminal, activity]
            candidate_subwords = candidate.split()
            k = len(candidate_subwords)

            # loop over each sentence of the candiddate
            for tokens, token_embeds in zip(info.tok_sents, info.tok_embeds):
                # returns embeds for all words in the sentence, like [i,hate,corp,-,greed,and,criminal,activity]
                symbols, symbol_embeds = self.reconstruct_words(tokens, token_embeds)
                # re-join tokens that BERT split at hyphens/apostrophes
                # [corp,-,greed] -> [corp-greed]
                words, word_embeds = merge_hyphenated(
                    [s.lower() for s in symbols], symbol_embeds
                )
                for i in range(len(words) - k + 1):
                    if words[i : i + k] == candidate_subwords:
                        all_embeds.append(
                            np.mean(word_embeds[i : i + k], axis=0, keepdims=True)
                        )

            if not all_embeds:
                raise ValueError(f"No embeddings found for candidate: {candidate}")

            all_embeds = np.vstack(all_embeds)
            all_embeds = _l2_normalize(all_embeds)

            if mode == "mean":
                return candidate, TermEmbeddings(
                    word_embed=_l2_normalize(np.mean(all_embeds, axis=0)),
                    sent_embeds=info.sent_embeds,
                )
            else:
                return candidate, TermSummary(
                    word_embeds=all_embeds, sent_embeds=info.sent_embeds
                )
        except Exception as e:
            print(f"Warning: skipping candidate '{candidate}': {e}")
            return candidate, None

    # input: sentences in the form of tokens + embeddings for each token
    # REVIEW EARLY STOP? check if word is the candidate?
    def reconstruct_words(self, tokens, embeddings):
        words = []
        word_embeds = []
        frags, frags_emb = [], []

        for t, e in zip(tokens, embeddings):
            if t in self.special_tokens:
                continue
            if t.startswith("##"):
                frags.append(t[2:])
                frags_emb.append(e)
            else:
                if frags:
                    words.append("".join(frags))
                    # avg all the embeddings to get the word
                    word_embeds.append(np.mean(frags_emb, axis=0))
                    frags, frags_emb = [], []
                frags = [t.lstrip("#")]
                frags_emb = [e]
        if frags:
            words.append("".join(frags))
            word_embeds.append(np.mean(frags_emb, axis=0))
        # output: all the words in the sentence + corresponding embeddings, the indices will match
        if not word_embeds:
            return [], np.empty((0, embeddings.shape[-1]))
        return words, np.vstack(word_embeds)

    # NOTE cosine distance = 1-cosim

    def self_similarity(
        self, word_embeddings: dict[str, TermSummary], max_sample_size=5000
    ):
        ss_score = {}

        for word, info in tqdm(
            word_embeddings.items(), desc="Calculating self-similarity scores..."
        ):
            X = info.word_embeds
            N = X.shape[0]

            if N < 2:
                continue

            # subsample if too many
            if N > max_sample_size:
                idx = np.random.choice(N, max_sample_size, replace=False)
                X = X[idx]
                N = X.shape[0]

            # each embed is divided by its norm -> A/||A||
            # cosim = A dot B / ||A||||B||
            # = A * B^T, @ = matrix mult, see above
            sim_matrix = X @ X.T

            # remember: now we are calculating the avg of all scores (each entry in the matrix)
            # sum all entries incl diagonal
            # subtract diagonal entries (which would be N 1s/ones = N)
            # divide by (N entries * N-1 non-diagonal pairs)
            ss = (np.sum(sim_matrix) - N) / (N * (N - 1))
            ss_score[word] = float(round(ss, 3))

        return ss_score

    def contextualized_vs_general(
        self,
        candidate_embeddings: dict[str, TermEmbeddings],
    ):
        all_candidates = list(candidate_embeddings.keys())

        encoded_input = self.tokenizer(
            all_candidates,
            padding="max_length",
            truncation=True,
            max_length=32,
            return_tensors="pt",
        )

        with torch.no_grad():
            model_output = self.model(**encoded_input)

        general_embeddings = self.mean_pooling(
            model_output, encoded_input["attention_mask"]
        )
        general_embeddings = general_embeddings.cpu().numpy()
        general_embeddings = self.l2_normalize(general_embeddings)

        context_embeddings = np.vstack(
            [candidate_embeddings[c].word_embed for c in all_candidates]
        )

        cos_sims = np.sum(context_embeddings * general_embeddings, axis=1)
        diff_scores = dict(zip(all_candidates, 1 - cos_sims))

        return diff_scores

    # input: dict [str, tuple[word embedding, List[sentence embeddings]]]
    def topic_score(
        self,
        candidate_tuples: dict[str, TermEmbeddings],
        method="max",
    ):
        topic_scores = {}

        # cand_embeds = np.vstack([candidate_tuples[c].word_embed for c in all_candidates])

        for word, info in candidate_tuples.items():

            cand_embedding = info.word_embed

            cos_sims = np.dot(
                info.sent_embeds, cand_embedding
            )  # shape: (num_occurrences,)

            if method == "max":
                score = np.max(cos_sims)
            elif method == "avg":
                score = np.mean(cos_sims)
            else:
                raise ValueError(f"Unknown method {method}")

            topic_scores[word] = float(score)

        return topic_scores

    def self_similarity_change(
        self,
        fine_tuned_embeddings: dict[str, TermSummary],
        vanilla_embeddings: dict[str, TermSummary],
        max_sample_size=5000,
    ):

        ssf = self.self_similarity(
            fine_tuned_embeddings, max_sample_size=max_sample_size
        )
        ssv = self.self_similarity(vanilla_embeddings, max_sample_size=max_sample_size)

        ssc_scores = {}
        for word in ssf.keys():
            if word in ssv:
                ssc_scores[word] = round(ssf[word] - ssv[word], 3)

        return ssc_scores

    def extract_terms(
        self,
        use_ngrams: bool = True,
        mode: str = "mean",  # mean or all
        compute_topic: bool = True,
        compute_self_sim: bool = False,
        compute_context_diff: bool = False,
        compute_ssc: bool = False,
    ):
        unigram_candidates, ngram_candidates = self.extract_candidates()
        candidates = ngram_candidates if use_ngrams else unigram_candidates

        if not candidates:
            return []

        # dict[str, CandidateSummary]
        encoded = self.encode(candidates)

        # either TermEmbeddings aka "contextualized" single embedding
        # or TermSummary aka embeddings for each context
        term_candidates = self.create_word_embeddings(encoded, mode=mode)

        # I KNOW THAT DIFFERENT FUNCTIONS REQUIRE DIFFERENT TYPES. IT IS ON PURPOSE. I WILL NOT BE KEEPING ALL OF THE FUNCTIONS. JUST ENSURE THAT THEY WORK CORRECTLY

        # requires TermEmbeddings
        if compute_topic:
            filtered_candidates = {}
            topic_scores = self.topic_score(term_candidates)
            for word, info in term_candidates.items():
                if topic_scores[word] >= self.topic_threshold:
                    filtered_candidates[word] = info
            return filtered_candidates

        # requires TermSummary
        if compute_self_sim:
            filtered_candidates = {}
            ss_scores = self.self_similarity(term_candidates)
            for word, info in term_candidates.items():
                score = ss_scores.get(word)
                # meaningful = high self sim
                # CAST says thres = 0.3
                if score is None or score >= self.self_sim_threshold:
                    filtered_candidates[word] = info
            return filtered_candidates

        # requires TermEmbeddings
        if compute_context_diff:
            filtered_candidates = {}
            diff_scores = self.contextualized_vs_general(term_candidates)
            for word, info in term_candidates.items():
                if diff_scores[word] >= self.context_diff_threshold:
                    filtered_candidates[word] = info
            return filtered_candidates

        # requires TermSummary
        if compute_ssc:
            vanilla_encoded = self.encode(candidates, model_name="microsoft/mpnet-base")

            vanilla_candidates = self.create_word_embeddings(vanilla_encoded, mode=mode)

            filtered_candidates = {}
            ssc_scores = self.self_similarity_change(
                term_candidates, vanilla_candidates
            )
            for word, info in term_candidates.items():
                score = ssc_scores.get(word)
                # thres >= 0, positive
                if score is None or score >= self.self_sim_threshold:
                    filtered_candidates[word] = info
            return filtered_candidates

        return term_candidates
