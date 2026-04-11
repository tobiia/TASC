from collections import Counter, defaultdict
import math
import string
import re
import spacy
import os
import pandas as pd

from spacy.lang.char_classes import ALPHA, ALPHA_LOWER, ALPHA_UPPER
from spacy.lang.char_classes import CONCAT_QUOTES, LIST_ELLIPSES, LIST_ICONS
from spacy.util import compile_infix_regex

# from spacy.tokens import Doc, Token

from config import TERM_PKG

# Parts of speech templates
pos_tag_patterns = [
    "PROPN",
    "NOUN",
    "ADJ",
    "VERB",
    "ADV",
    [["PROPN", "NOUN"], "*"],
    ["ADJ", "*", ["PROPN", "NOUN"], "*"],
    ["ADJ", "*"],
    ["VERB", "ADJ", ["PROPN", "NOUN"], "*"],
    [["PROPN", "NOUN"], "*", "ADJ", "*", ["PROPN", "NOUN"], "*"],
    ["ADJ", "VERB", ["PROPN", "NOUN"], "*"],
    ["VERB", "*", ["PROPN", "NOUN"], "*"],
    ["ADV", "*", "ADJ", "*"],
    [["PROPN", "NOUN"], "ADP", ["PROPN", "NOUN"]],
    [["ADJ", "PROPN", "NOUN"], "*", "PART", ["PROPN", "NOUN"], "*"],
    [["VERB", "ADV", "X"]],
    [["ADJ", "PROPN", "NOUN"], "*", "ADP", ["PROPN", "NOUN"], "*"],
]

# Setting up punctuation lists to check, punc_without does not contain hyphens and apostrophes as they can be part of phrases. punc_all is needed to check if there is a hyphen at the beginning or end of a phrase
punc_without = set(string.punctuation)
punc_without.update(["»", "«"])
punc_all = punc_without.copy()
punc_without.remove("-")
punc_without.remove("'")
num_set = set("1234567890")
CLAUSE_BREAKS = {",", ";", ":", "(", ")", "[", "]", "—", "–", "-", "/"}
NOISE_TOKENS = set(string.punctuation).union(("''", "``", "..."))


def remove_punc_spaces(text):
    """Remove spaces before punctuation and around apostrophes."""
    # Remove spaces before common punctuation marks
    text = re.sub(r"\s+([.,;:!?)\]}–—])", r"\1", text)
    # Remove spaces around apostrophes
    text = re.sub(r"\s+[''´`]\s*", "'", text)
    return text


# Text tokenizer, input text with original case NOT in lower case
# output a set of tokens marked by sentences, an element in the list is a sentence that contains tokens with information about them
#  [ [(token1, pos, index),(token2, pos, index),(token3, pos, index)],
#    [(token1, pos, index),(token2, pos, index),(token3, pos, index)]]
def tokinizer(doc):
    sent_tokens = []
    index = 0
    for sent in doc.sents:
        list_tok = []
        for i in sent:  # i = Token
            list_tok.append(
                (i, i.pos_, index)
            )  # creating a list of tokens with content, the actual unigram in lower case, its part of speech, position number in the text
            index += 1
        sent_tokens.append(list_tok)
    return sent_tokens


# function for combining phrase tokens into a single string
# input is a list of spacy Tokens [token1, token2, token3] OR list of str
# output is "word1 word2 word3"
# no space is put between the hyphen and the apostrophe
def concatenate_ngrams(candidate):
    temp = ""
    if type(candidate) != type(str()):
        for w in candidate:  # w = Token
            if isinstance(w, str):
                w_text = w
            else:
                w_text = w.lower_
            if (
                (w_text not in punc_without)
                and (len(temp) > 0)
                and (
                    (temp[-1] == "'")
                    or (
                        (w_text[0] not in punc_without)
                        and (temp[-1] not in punc_without)
                    )
                )
            ):
                temp = temp + " " + w_text
            else:
                temp = temp + w_text
    else:
        temp = candidate
    temp = temp.lower()
    return str(temp)


ngram_cache = {}


# input: Sequence of either spacy Tokens or strings representing a mwe
def concat_cache(ngram):
    first = ngram[0]
    if isinstance(first, str):
        key = tuple(token.lower() for token in ngram)
    else:
        key = tuple(Token.lower_ for Token in ngram)  # spaCy Token
    if key not in ngram_cache:
        ngram_cache[key] = concatenate_ngrams(ngram)
    return ngram_cache[key]


# Cleaning a Single phrase from stop words
# if the phrase starts or ends with an adposition, phrase is deleted.
# if a word in a phrase is a stop word, the phrase is deleted.
# all words in the phrase are checked except prepositions and PROPN
# at the input:
# mwe_list = [    [[list of words/Tokens in the phrase], template by which the candidate was extracted, sequence of parts of speech of the candidate, indexes of word positions, number of words, number of characters in the candidate], [6 elems], [6], ...,    ]
# a list of lists for each phrase coming from a sentence
# at the output is a filtered list
def filter_stop_words(mwe_list, stop_words):
    filtred_ngrams = []
    for mwe in mwe_list:
        checker3 = True
        # if first or last term is adposition
        if mwe[2][0] == "ADP" or mwe[2][-1] == "ADP":
            checker3 = False

        for w in mwe[0]:  # iterating over each Token in the phrase
            if w.pos_ not in ["PROPN", "ADP"]:
                if w.lower_ in stop_words:
                    checker3 = False

        if checker3 == True:
            filtred_ngrams.append(mwe)
    return filtred_ngrams


# extracting candidates based on part-of-speech templates
# input: list of tokens in one sentence [(token1, pos, index),(token2, pos, index),(token3, pos, index)] and part-of-speech template
# output: list of lists of extracted candidates + information about them:
# [    [[list of words/Tokens in the phrase], template by which the candidate was extracted, sequence of parts of speech of the candidate, indexes of word positions, number of words, number of characters in the candidate], [6 elems], [6], ...,    ]
def filter_ngrams_by_pos_tag(sentence, pos_sequences):
    filtered_ngrams = []
    seen = set()
    for seq in pos_sequences:
        # i = tuple representing 1 Token (word) in the sentence
        for i in range(len(sentence)):
            temp = []  # list of Tokens
            temp_index = []
            temp_pos = []
            checker = True

            # sentence[i] = (token1, pos, index), a tuple
            # sentence[i][0].lower_ = Token.text.lower()
            if ((sentence[i][1] in seq[0]) or (sentence[i][1] == seq[0])) and (
                sentence[i][0].lower_ not in punc_without
            ):
                seq_index = 0
                sent_index = 0

                while (
                    seq_index < len(seq)
                    and i + sent_index < len(sentence)
                    and checker == True
                ):
                    if (sentence[i + sent_index][1] in seq[seq_index]) or (
                        sentence[i + sent_index][0].lower_ in seq[seq_index]
                    ):
                        temp.append(sentence[i + sent_index][0])
                        temp_pos.append(sentence[i + sent_index][1])
                        temp_index.append(sentence[i + sent_index][2])
                        seq_index += 1
                        sent_index += 1

                    elif seq[seq_index] == "*" and (
                        sentence[i + sent_index][1] in seq[seq_index - 1]
                    ):
                        if seq_index < len(seq) - 1:
                            temp.append(sentence[i + sent_index][0])
                            temp_pos.append(sentence[i + sent_index][1])
                            temp_index.append(sentence[i + sent_index][2])
                            sent_index += 1

                        elif seq_index == len(seq) - 1:
                            temp_text = [t.lower_ for t in temp]
                            if (len(temp) > 1 or "-" in "".join(temp_text)) and (
                                len(set("".join(temp_text)).intersection(punc_without))
                                == 0
                            ):
                                temp_2 = temp.copy()
                                temp_pos2 = temp_pos.copy()
                                temp_index2 = temp_index.copy()
                                _key = (
                                    tuple(w.lower_ for w in temp_2),
                                    str(seq),
                                    tuple(temp_index2),
                                )
                                if (
                                    _key not in seen
                                    and temp_text[-1] not in punc_without
                                ):
                                    seen.add(_key)
                                    temp_2 = [word for word in temp_2]
                                    filtered_ngrams.append(
                                        [
                                            temp_2,
                                            seq,
                                            temp_pos2,
                                            temp_index2,
                                            len(temp_2),
                                            len(concat_cache(temp_2)),
                                        ]
                                    )

                            if (i + sent_index) < len(sentence):
                                temp.append(sentence[i + sent_index][0])
                                temp_pos.append(sentence[i + sent_index][1])
                                temp_index.append(sentence[i + sent_index][2])
                                sent_index += 1

                    elif seq[seq_index] == "*" and (
                        sentence[i + sent_index][1] not in seq[seq_index - 1]
                    ):
                        seq_index += 1

                    else:
                        checker = False

                temp_text = [t.lower_ for t in temp]
                if (
                    seq_index == len(seq)
                    and (len(temp) > 1 or "-" in "".join(temp_text))
                    and len(set("".join(temp_text)).intersection(punc_without)) == 0
                ):
                    _key = (tuple(w.lower_ for w in temp), str(seq), tuple(temp_index))
                    if _key not in seen and temp_text[-1] not in punc_without:
                        seen.add(_key)
                        temp = [word for word in temp]
                        filtered_ngrams.append(
                            [
                                temp,
                                seq,
                                temp_pos,
                                temp_index,
                                len(temp),
                                len(concat_cache(temp)),
                            ]
                        )
    return filtered_ngrams


# calculation of the rectified frequency - the number of phrases in the text, not in phrases longer than
# input: list of words of the phrase (one candidate), all texts in the form of a single string, list f_raw_req_list which contains the frequencies of phrases longer than/or it is empty, since it is filled in during the function call
# output: rectified frequency of the phrase
# the function is called as many times as candidates. to calculate the rectified frequency, the calculation is made from the longest to the shortest phrase, since for its calculation it is necessary to know the frequency of phrases longer than the target
def f_req_calc(mwe, all_txt, f_raw_req_list):
    temp = all_txt
    mwe_c = concat_cache(mwe)
    for i in f_raw_req_list:
        i_c = concat_cache(i[0])
        if mwe_c in i_c and mwe_c != i_c and len(mwe) != len(i[0]):
            temp = temp.replace(i_c, " ")
    f = temp.count(mwe_c)
    return f


def _build_phrase_positions(token_list, possible_mwe):
    """Map each phrase tuple to the list of its start positions in token_list."""
    first_token_index = defaultdict(list)
    for phrase_tuple in possible_mwe:
        if phrase_tuple:
            first_token_index[phrase_tuple[0]].append(phrase_tuple)

    phrase_positions = {pt: [] for pt in possible_mwe}
    for i, token in enumerate(token_list):
        for phrase_tuple in first_token_index.get(token, []):
            n = len(phrase_tuple)
            if (
                i + n <= len(token_list)
                and tuple(token_list[i : i + n]) == phrase_tuple
            ):
                phrase_positions[phrase_tuple].append(i)
    return phrase_positions


# input: list of tuples, dict of tuples and their indices in sentences
def _compute_rectified_frequencies(possible_mwe, phrase_positions):
    """
    Compute raw and rectified frequencies for each phrase.
    Uses fast token position matching with old-style substring blocking.
    possible_mwe must be sorted longest-first.
    Returns list of [phrase_tuple, f_raw, f_req] in the same order.
    """
    result = []

    for phrase_tuple in possible_mwe:
        positions = phrase_positions.get(phrase_tuple, [])
        f_raw = len(positions)
        f_req = f_raw

        # Old logic: subtract occurrences that appear within longer phrases
        phrase_str = " ".join(phrase_tuple)

        for longer_phrase_tuple in result:
            longer_str = " ".join(longer_phrase_tuple[0])
            # If this phrase is a substring of a longer phrase (and not identical)
            if phrase_str in longer_str and phrase_str != longer_str:
                # Reduce count by occurrences within the longer phrase
                longer_positions = phrase_positions.get(longer_phrase_tuple[0], [])
                f_req -= len(longer_positions)

        result.append([phrase_tuple, f_raw, max(0, f_req)])

    return result


# To unite candidates for common positions into groups
class UnionFind:
    def __init__(self, size):
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, u):
        if self.parent[u] != u:
            self.parent[u] = self.find(self.parent[u])
        return self.parent[u]

    def union(self, u, v):
        root_u = self.find(u)
        root_v = self.find(v)
        if root_u != root_v:
            if self.rank[root_u] > self.rank[root_v]:
                self.parent[root_v] = root_u
            elif self.rank[root_u] < self.rank[root_v]:
                self.parent[root_u] = root_v
            else:
                self.parent[root_v] = root_u
                self.rank[root_u] += 1


def group_items(lst):
    n = len(lst)
    uf = UnionFind(n)

    pos_to_items = defaultdict(list)
    for i, item in enumerate(lst):
        for pos in item[3]:
            pos_to_items[pos].append(i)

    for items_at_pos in pos_to_items.values():
        for j in range(1, len(items_at_pos)):
            uf.union(items_at_pos[0], items_at_pos[j])

    # Assign group numbers
    group_map = {}
    group_number = 1
    for i in range(n):
        root = uf.find(i)
        if root not in group_map:
            group_map[root] = group_number
            group_number += 1
        lst[i].append(group_map[root])

    return lst


def rejoin_hyphen_apostrophe(tokens):
    if not tokens:
        return tokens

    result = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]

        # Case 1: next token starts with apostrophe → contraction
        # e.g. ["do", "n't"] or ["it", "'s"] or ["i", "'m"]
        if (
            i + 1 < len(tokens)
            and tokens[i + 1].startswith("'")
            and tokens[i + 1] not in {"'", "''"}  # lone quote is not a contraction
        ):
            result.append(tok + tokens[i + 1])
            i += 2
            continue

        # Case 2: current token is a bare hyphen between two word tokens
        # e.g. ["anti", "-", "corruption"]
        # We merge only if the hyphen is surrounded by non-punctuation tokens
        # (i.e. it's a tokenization artifact, not a clause-break dash)
        if (
            tok == "-"
            and result  # there is a preceding token already merged
            and i + 1 < len(tokens)
            and tokens[i + 1] not in CLAUSE_BREAKS
            and tokens[i + 1] not in NOISE_TOKENS
            and tokens[i + 1].strip()
        ):
            # Attach the hyphen and the next token to the previous result token
            result[-1] = result[-1] + "-" + tokens[i + 1]
            i += 2
            continue

        result.append(tok)
        i += 1

    return result


# main body of phrase extraction
# input text and various parameters
# output list of phrases: ["phrase 1", "phrase 2", "phrase 3"]
class CandidateExtractor:
    def __init__(
        self,
        corpus_path: str,
        stop_words_path: str = "stop_words_en.txt",
        list_seq=pos_tag_patterns,
        dependency_filter=True,
        ov_filter=True,
        cohesion_filter=False,
        additional_text="1",
        f_raw_sc=9,
        f_req_sc=3,
    ):
        self.corpus_path = corpus_path  # FULL PATH

        self.cohesion_filter = cohesion_filter  #  Enable or disable the cohesive filter
        self.dependency_filter = dependency_filter
        self.ov_filter = ov_filter
        self.additional_text = additional_text  # if there is additional text, it is used to calculate frequencies, terms are NOT extracted from it
        self.f_req_sc = f_req_sc  # rectified frequency threshold
        self.f_raw_sc = f_raw_sc  # raw frequency threshold
        self.list_seq = list_seq  # list of part of speech patterns
        self.model_nlp = spacy.load(
            "en_core_web_sm", disable=["ner", "parser"]
        )  # Spacy model
        self.model_nlp.add_pipe("sentencizer")
        # prevent spacy from stopping on long docs
        self.model_nlp.max_length = 2_000_000

        try:
            stop_path = TERM_PKG / stop_words_path
            with open(stop_path, encoding="utf-8") as f:
                self.stop_words = set(f.read().split(","))
        except FileNotFoundError:
            print(f"ERROR: stop words file not found at: {stop_path}")
            self.stop_words = set()
        except Exception as e:
            print(f"ERROR: stop words file could not be opened: {e}")
            self.stop_words = set()

    def get_corpus(self, corpus_path: str) -> str:
        file_list = os.listdir(corpus_path)
        texts = []

        for filename in file_list:
            if filename.endswith(".txt"):
                file_path = os.path.join(corpus_path, filename)
                with open(file_path, "r") as file:
                    text = file.read()
                    texts.append(
                        text.replace("  ", " ").replace(" -", "-").replace(" - ", "-")
                    )

        all_texts = " .".join(texts)
        return all_texts

    def ngram_candidates(self):

        # Change the tokenizer so that it does not separate words with hyphens.
        # IT-developers create innovative solutions. --> ["IT-developers","create","innovative","solutions","."]
        # Instead of ["IT","-","developers"] tokenized as a whole token "IT-developers", this helps to avoid extracting unigrams that are part of the word, which reduces noise

        nlp = self.model_nlp

        infixes = (
            LIST_ELLIPSES
            + LIST_ICONS
            + [
                r"(?<=[0-9])[+\\-\\*^](?=[0-9-])",
                r"(?<=[{al}{q}])\\.(?=[{au}{q}])".format(
                    al=ALPHA_LOWER, au=ALPHA_UPPER, q=CONCAT_QUOTES
                ),
                r"(?<=[{a}]),(?=[{a}])".format(a=ALPHA),
                r"(?<=[{a}0-9])[:<>=/](?=[{a}])".format(a=ALPHA),
            ]
        )
        infix_re = compile_infix_regex(infixes)
        nlp.tokenizer.infix_finditer = infix_re.finditer

        corpus = self.get_corpus(self.corpus_path)

        # removing extra spaces
        corpus = (
            corpus.replace(" -", "-")
            .replace("- ", "-")
            .replace(" '", "'")
            .replace("  ", " ")
        )

        # text tokenization, parts of speech and position index
        doc = nlp(corpus)
        #  [ [(token1, pos, index),(token2, pos, index),(token3, pos, index)], [sentence],...]
        text_sent_tokens = tokinizer(doc)
        mwe_list = []

        # extracting candidates from each sentence separately
        for sent in text_sent_tokens:
            # sent = [(token1, pos, index),(token2, pos, index),(token3, pos, index)]

            # returns list of lists of extracted candidates + information about them:
            # [    [[list of words/Tokens in the phrase], template by which the candidate was extracted, sequence of parts of speech of the candidate, indexes of word positions, number of words, number of characters in the candidate], [6 elems], [6], ...,    ]

            temp_mwe_list = filter_ngrams_by_pos_tag(
                sent, self.list_seq
            )  # Part-of-speech extraction

            temp_mwe_list = filter_stop_words(
                temp_mwe_list, self.stop_words
            )  # stop word filtering

            sent_text = " ".join([s[0].lower_ for s in sent])
            # now list of 7 element list, each inner list = 1 mwe
            temp_mwe_list = [mwe + [sent_text] for mwe in temp_mwe_list]
            mwe_list += temp_mwe_list

        # sentence lookup for all extracted phrases
        phrase_sents = defaultdict(list)
        # mwe_list = list of ALL 7 elem lists for the candidate mwes
        # mwe[-1] = the full sentence a mwe is from
        for mwe in mwe_list:
            phrase_sents[concat_cache(mwe[0])].append(mwe[-1].strip())

        # creating a list of candidates containing only the words of the candidates: [("phrase","one"),("mwe","next"),("phrase","other")]
        mwe_list_n = [tuple(i[0]) for i in mwe_list]
        candidates = []

        # combination of words of a phrase: ["phrase one", "mwe next","phrase other"]
        for i in set(mwe_list_n):
            candidates.append(concat_cache(i))

        # cleaning phrases from punctuation and if it consists entirely of punctuation and numbers
        candidates = [
            i
            for i in candidates
            if ((i[-1] not in punc_without) and (i[-1] not in string.punctuation))
        ]
        candidates = [i for i in candidates if len(num_set.intersection(set(i))) == 0]

        candidate_map = {term: phrase_sents.get(term, []) for term in set(candidates)}

        print(
            f"CANDIDATE EXTRACTOR: before filtering: {len(candidate_map)} mwe candidates"
        )

        # ANCHOR - DEPENDENCY FILTER

        if self.dependency_filter == True:
            # True = rejected or penalized
            # phrase = list of spacy Tokens/words in the phrase
            # description of spacy pos and tags here:
            # https://ashutoshtripathi.com/2020/04/13/parts-of-speech-tagging-and-dependency-parsing-using-spacy-nlp/
            # https://universaldependencies.org/u/pos/

            def filter_verb_initial(phrase) -> bool:
                first = phrase[0]
                if first.pos_ == "VERB":
                    # allow gerunds/nominalisations tagged as noun by fine-grained tagger
                    if first.tag_ in ("NN", "NNS"):
                        return False  # don't filter — it's a nominalisation
                    return True
                if first.tag_ in ("VBZ", "VBP", "VBD"):  # conjugated verbs only
                    return True
                # VBG (gerund) — only filter if it's truly verbal, not nominal
                if first.tag_ == "VBG":
                    # if the next token is a determiner or preposition, it's nominal
                    if len(phrase) > 1 and phrase[1].pos_ in (
                        "DET",
                        "ADP",
                        "NOUN",
                        "PROPN",
                    ):
                        return False
                    return True
                return False

            DEICTIC_DEPS = {"det", "predet"}
            DEICTIC_TAGS = {"DT", "PDT"}

            def filter_deictic_initial(phrase):
                first = phrase[0]
                # demonstratives and discourse-anchored adjectives
                if first.lemma_.lower() in {
                    "this",
                    "that",
                    "these",
                    "those",
                    "such",
                    "said",
                    "aforementioned",
                    "same",
                }:
                    return True

                # catches "some", "any", "each", "every"
                if first.tag_ in DEICTIC_TAGS:
                    return True

                if first.pos_ in DEICTIC_DEPS:
                    return True
                return False

            # REVIEW covered by filter_stop_words but that may be too harsh?
            def filter_stop_word_initial(phrase):
                first = phrase[0]
                # first term is a stop word
                if first.lemma_.lower() in self.stop_words:
                    return True
                return False

            def filter_quantifier_initial(phrase):
                first = phrase[0]
                if first.pos_ == "NUM":
                    return True
                # spaCy tags "several", "many", etc as ADJ
                if first.pos_ == "ADJ" and first.lemma_.lower() in {
                    "several",
                    "many",
                    "few",
                    "various",
                    "certain",
                    "enough",
                    "plenty",
                }:
                    return True
                return False

            SCALAR_ADJ = {
                # Evaluative / stance
                "effective",
                "appropriate",
                "adequate",
                "proper",
                "sound",
                "thorough",
                "coherent",
                "comparable",
                "exceptional",
                "sufficient",
                # Intensifiers / degree
                "comprehensive",
                "enhanced",
                "firm",
                "strong",
                "dissuasive",
                "systematic",
                "improved",
                "increased",
                "broader",
                "clearer",
                "deeper",
                "further",
                "greater",
                "higher",
                "stronger",
                "wider",
                # Deictic-adjacent
                "current",
                "overall",
                "remaining",
                "potential",
                "particular",
                "special",
                "full",
                "complete",
                # Scalar quantity
                "major",
                "main",
                "key",
                "important",
                "necessary",
                "unnecessary",  # might need to remove negations too!
                "limited",
                "central",
                "local",
            }

            def filter_scalar_adj_initial(phrase):
                first = phrase[0]
                if first.pos_ == "ADJ" and first.lemma_.lower() in SCALAR_ADJ:
                    return True
                return False

            dep_filters = [
                filter_verb_initial,
                filter_deictic_initial,
                filter_quantifier_initial,
                filter_scalar_adj_initial,
            ]

            filtered_dep = {}
            # mwe_list is a list of 7 elem lists for each phrase in Tokens + info
            # first elem = tuple of spacy tokens, last = full sentence
            for mwe in mwe_list:
                # A candidate is kept only if ALL filter functions return False
                # (i.e. none of the "bad pattern" checks fire)
                if not any(f(mwe[0]) for f in dep_filters):
                    filtered_dep[concat_cache(mwe[0])] = mwe[-1]

            common_keys = filtered_dep.keys() & candidate_map.keys()
            candidate_map = {k: candidate_map[k] for k in common_keys}
            # candidate_map = filtered_dep
            print(
                f"CANDIDATE EXTRACTOR: after dependency filter: {len(candidate_map)} mwe candidates"
            )

        # ANCHOR - COHESION FILTER

        # if the cohesive filter is on
        if self.cohesion_filter == True:

            token_list = [tok.text.lower() for tok in doc]
            if (
                len(self.additional_text) > 10
            ):  # if there is text (number 10 is random, the main thing is that it is not empty) then we combine it with the text from which we extracted candidates
                text_ref = (
                    self.additional_text.replace(" -", "-")
                    .replace("- ", "-")
                    .replace(" '", "'")
                    .replace("  ", " ")
                    .lower()
                )
                ref_doc = nlp(text_ref)
                token_list = token_list + [tok.text.lower() for tok in ref_doc]

            # i = mwe = 7 elem list of a phrase, i[0] = tuple of spaCy Tokens/words in the phrase
            # we don't need Tokens anymore so convert all to text
            for item in mwe_list:
                item[0] = [token.lower_ for token in item[0]]

            all_cand_r = [tuple(i[0]) for i in mwe_list]
            possible_mwe = sorted(
                set(all_cand_r), key=len, reverse=True
            )  # sort phrases from longest to shortest

            phrase_positions = _build_phrase_positions(token_list, possible_mwe)
            f_raw_req_list = _compute_rectified_frequencies(
                possible_mwe, phrase_positions
            )

            mwe_f = []
            f_raw_req_dict = {concat_cache(i[0]): i for i in f_raw_req_list}

            # Merging Phrase Frequency and Phrase Info Lists
            for mwe in mwe_list:
                entry = f_raw_req_dict[concat_cache(mwe[0])]
                mwe_f.append(
                    [
                        mwe[0],
                        entry[1],
                        entry[2],
                        mwe[3],
                        mwe[-1],
                    ]
                )

            # grouping phrases by common word position
            # input: [candidate, raw frequency, rectified frequency, word position indices, sentence in which it is located])
            # output: phrases grouped by positions (at the end, the number of the group to which the phrase belongs is indicated):
            # [candidate, raw frequency, rectified frequency, word position indices, sentence in which it is located, group number])

            grouped_data = group_items(mwe_f)

            candidates = []
            candid_q = []
            remover = []
            df = pd.DataFrame(
                grouped_data,
                columns=["mwe", "f raw", "f req", "index", "sent", "group"],
            )

            # selecting a candidate from the group with the highest rectified or raw frequency
            for i in range(1, len(set(df["group"])) + 1):
                df_temp = df[df["group"] == i]
                while len(df_temp) > 0:
                    max = df_temp["f req"].max()
                    cand = df_temp[df_temp["f req"] == max].values.tolist()[0]
                    if max > 1:
                        candidates.append(cand)
                    elif df_temp["f raw"].max() > 1:
                        candid_q.append(cand)

                    index = df_temp["index"][df_temp["f req"] == max].values.tolist()[0]
                    drop = df_temp.index[
                        df_temp["index"].apply(lambda x: any(i in index for i in x))
                    ].tolist()
                    dd = df_temp.loc[drop].values.tolist()
                    df_temp = df_temp.drop(drop)
                    remover += dd

            # or phrases are accepted if they have a rectified or raw frequency above the specified threshold
            data1 = df[df["f req"] >= self.f_req_sc].values.tolist()
            data2 = df[df["f raw"] >= self.f_raw_sc].values.tolist()

            # combine words of a phrase and create a single list of extracted phrases: ["phrase 1","phrase 2","phrase3"]
            cand_mwe = [concat_cache(i[0]) for i in candidates + candid_q]
            cand_mwe1 = [concat_cache(i[0]) for i in data1]
            cand_mwe2 = [concat_cache(i[0]) for i in data2]
            cand = cand_mwe + cand_mwe1 + cand_mwe2
            candidates = [
                i
                for i in cand
                if ((i[-1] not in punc_without) and (i[-1] not in string.punctuation))
            ]

            candidate_map = {
                term: phrase_sents.get(term, []) for term in set(candidates)
            }

            print(
                f"CANDIDATE EXTRACTOR: after cohesion (original) filter: {len(candidate_map)} mwe candidates"
            )

        # ANCHOR - OV FILTER
        if self.ov_filter == True:

            def build_av_scores(token_list):
                # find all positions of each n-gram
                positions = defaultdict(list)
                n = len(token_list)

                for length in range(1, 8):  # up to 7-grams
                    for i in range(n - length + 1):
                        ngram = tuple(token_list[i : i + length])
                        positions[ngram].append(i)

                av_scores = {}
                for ngram, pos_list in positions.items():
                    left_neighbors = set()
                    right_neighbors = set()
                    for p in pos_list:
                        if p > 0:
                            left_neighbors.add(token_list[p - 1])
                        if p + len(ngram) < n:
                            right_neighbors.add(token_list[p + len(ngram)])
                    lav = len(left_neighbors)
                    rav = len(right_neighbors)
                    av = min(lav, rav)
                    av_scores[ngram] = math.log(
                        av + 1
                    )  # log-transform, +1 for smoothing

                return av_scores, positions

            def overlap_variety(candidate_tokens, av_scores, token_list, positions):
                s = tuple(candidate_tokens)
                n = len(s)

                if n < 2:
                    return 0.0

                cand_av = av_scores.get(s, 0.0)

                # collect all overlapping strings at each overlap level
                # overlap level i means i characters/tokens overlap
                total_weighted = 0.0
                total_weight = 0.0

                for overlap_level in range(1, n):
                    weight = 1.0 / (n - overlap_level)  # wi = 1/|s - overlap|

                    # preceding overlapping strings: start overlap_level positions before candidate
                    # e.g. for "criminal law enforcement" at overlap_level=1,
                    # preceding overlap is the trigram ending at "law" = ("X", "criminal", "law")

                    overlapping = []

                    # get all positions of candidate
                    cand_positions = positions.get(s, [])

                    for pos in cand_positions:
                        # preceding overlap: string of length n starting overlap_level before
                        pre_start = pos - overlap_level
                        if pre_start >= 0:
                            pre_string = tuple(token_list[pre_start : pre_start + n])
                            overlapping.append(pre_string)

                        # following overlap: string of length n starting overlap_level into candidate
                        post_start = pos + overlap_level
                        if post_start + n <= len(token_list):
                            post_string = tuple(token_list[post_start : post_start + n])
                            overlapping.append(post_string)

                    if not overlapping:
                        continue

                    # fraction of overlapping strings with lower AV than candidate
                    better = sum(
                        1 for o in overlapping if av_scores.get(o, 0.0) < cand_av
                    )
                    ov_i = better / len(overlapping)

                    total_weighted += weight * ov_i
                    total_weight += weight

                return total_weighted / total_weight if total_weight > 0 else 0.0

            token_list_ov = [tok.lower_ for tok in doc]

            # include additional text if needed
            if len(self.additional_text) > 10:
                text_ref_ov = (
                    self.additional_text.replace(" -", "-")
                    .replace("- ", "-")
                    .replace(" '", "'")
                    .replace("  ", " ")
                    .lower()
                )
                ref_doc_ov = nlp(text_ref_ov)
                token_list_ov = token_list_ov + [tok.lower_ for tok in ref_doc_ov]

            av_scores_ov, positions_ov = build_av_scores(token_list_ov)

            # OV score 0.0  = the candidate's AV is NEVER better than its overlapping strings -- very likely a fragment / incoherent span
            # score 1.0  = the candidate's AV is ALWAYS better than its overlapping strings, likely a genuine tight unit

            OV_THRESHOLD = 0.0

            # True = skip OV check for bigrams
            # b/c OV is weak on bigrams, not enough overlaps
            OV_PASSTHROUGH_BIGRAMS = True

            filtered_ov = {}

            for term, sents in candidate_map.items():
                tokens = tuple(term.split())

                # unigrams: no internal boundaries, pass through with score = 1.0
                if len(tokens) < 2:
                    filtered_ov[term] = sents
                    continue

                # bigrams bypass OV filter entirely or not
                if OV_PASSTHROUGH_BIGRAMS and len(tokens) == 2:
                    filtered_ov[term] = sents
                    continue

                score = overlap_variety(
                    tokens,
                    av_scores_ov,
                    token_list_ov,
                    positions_ov,
                )

                if score > OV_THRESHOLD:
                    filtered_ov[term] = sents

            candidate_map = filtered_ov
            print(
                f"CANDIDATE EXTRACTOR: after OV filter: {len(candidate_map)} candidates "
            )

        print(
            f"CANDIDATE EXTRACTOR: TOTAL number of ngrams extracted: {len(candidate_map)}"
        )

        candidate_map = {
            term: [remove_punc_spaces(sent) for sent in sents]
            for term, sents in candidate_map.items()
        }

        return candidate_map

    def unigram_candidates(self) -> dict[str, list[str]]:
        """Extract only unigram candidates (single tokens, no hyphens) from the corpus"""
        nlp = self.model_nlp

        # reconfigure tokenizer to keep hyphenated words as single tokens
        infixes = (
            LIST_ELLIPSES
            + LIST_ICONS
            + [
                r"(?<=[0-9])[+\\-\\*^](?=[0-9-])",
                r"(?<=[{al}{q}])\\.(?=[{au}{q}])".format(
                    al=ALPHA_LOWER, au=ALPHA_UPPER, q=CONCAT_QUOTES
                ),
                r"(?<=[{a}]),(?=[{a}])".format(a=ALPHA),
                r"(?<=[{a}0-9])[:<>=/](?=[{a}])".format(a=ALPHA),
            ]
        )
        infix_re = compile_infix_regex(infixes)
        nlp.tokenizer.infix_finditer = infix_re.finditer

        corpus = self.get_corpus(self.corpus_path)
        corpus = (
            corpus.replace(" -", "-")
            .replace("- ", "-")
            .replace(" '", "'")
            .replace("  ", " ")
        )

        doc = nlp(corpus)

        # collect the POS tags that qualify as unigrams (the string-only patterns)
        uni_pos_set = set(
            pattern for pattern in self.list_seq if isinstance(pattern, str)
        )
        digit_punc_set = num_set | punc_all

        unigram_map = defaultdict(list)
        for token in doc:
            if token.pos_ not in uni_pos_set:
                continue
            w = token.text.lower()
            if (
                w in self.stop_words
                or w[0] in punc_all
                or w[-1] in punc_all
                or not set(w).isdisjoint(punc_without)  # skip if contains punctuation
                or not set(w).isdisjoint(string.digits)  # skip if contains digits
                or set(w)
                <= digit_punc_set  # skip if only digits/punctuation -- maybe redundant now
            ):
                continue
            unigram_map[w].append(token.sent.text.lower())

        print(
            f"CANDIDATE EXTRACTOR: TOTAL number of unigrams extracted: {len(unigram_map)}"
        )

        unigram_map = {
            term: [remove_punc_spaces(sent) for sent in sents]
            for term, sents in unigram_map.items()
        }

        return dict(unigram_map)
