from collections import defaultdict
import string
import spacy
import os
import pandas as pd

from spacy.lang.char_classes import ALPHA, ALPHA_LOWER, ALPHA_UPPER
from spacy.lang.char_classes import CONCAT_QUOTES, LIST_ELLIPSES, LIST_ICONS
from spacy.util import compile_infix_regex

from config import TERM_PKG

# Parts of speech templates
pos_tag_patterns = ["PROPN", "NOUN", "ADJ", "VERB"]

# Setting up punctuation lists to check, punc_without does not contain hyphens and apostrophes as they can be part of phrases. punc_all is needed to check if there is a hyphen at the beginning or end of a phrase
punc_without = set(string.punctuation)
punc_without.update(["»", "«"])
punc_all = punc_without.copy()
punc_without.remove("-")
punc_without.remove("'")
num_set = set("1234567890")


# Text tokenizer, input text with original case NOT in lower case
# output a set of tokens marked by sentences, an element in the list is a sentence that contains tokens with information about them
#  [ [("token1", pos, index),("token2", pos, index),("token3", pos, index)],
#    [("token1", pos, index),("token2", pos, index),("token3", pos, index)]]
def tokinizer(doc):
    sent_tokens = []
    index = 0
    for sent in doc.sents:
        list_tok = []
        for i in sent:
            list_tok.append(
                (i.text.lower(), i.pos_, index)
            )  # creating a list of tokens with content, the actual unigram in lower case, its part of speech, position number in the text
            index += 1
        sent_tokens.append(list_tok)
    return sent_tokens


# function for combining phrase tokens into a single string
# input is a list of words ["word1","word2", "word3"]
# output is "word1 word2 word3"
# no space is put between the hyphen and the apostrophe
def concatenate_ngrams(candidate):
    cand_temp = []
    temp = ""
    if type(candidate) != type(str()):
        for w in candidate:
            if (
                (w not in punc_without)
                and (len(temp) > 0)
                and (
                    (temp[-1] == "'")
                    or ((w[0] not in punc_without) and (temp[-1] not in punc_without))
                )
            ):
                temp = temp + " " + str(w)
            else:
                temp = temp + str(w)
    else:
        temp = candidate
    temp = temp.lower()
    return str(temp)


ngram_cache = {}


# REVIEW helps?
def concat_cache(ngram):
    key = tuple(ngram)
    if key not in ngram_cache:
        ngram_cache[key] = concatenate_ngrams(ngram)
    return ngram_cache[key]


# filter based on changing part of speech
# input list of candidates
# output filtered list
# if during repeated marking of phrases that end in NOUN or PROPN the part of speech changed to something other than "NOUN", "PROPN", "VERB", then such phrase is not complete, and it is deleted
# when marking Spacy even a hyphen can be PROPN or NOUN if it is part of a whole word, for example IT-developers
def filter_propn_noun(mwe_list, nlp):
    filtred_ngrams = []
    for i in mwe_list:
        checker2 = True
        if concat_cache(i[0])[-1] in punc_all and concat_cache(i[0])[0] in punc_all:
            checker2 = False

        if len(i[2]) > 1:
            if (
                ("NOUN" in i[1][-1])
                or ("PROPN" in i[1][-1])
                or ("NOUN" in i[1][-2])
                or ("PROPN" in i[1][-2])
            ) and (("ADJ" not in i[1][-1]) and ("ADJ" not in i[1][-2])):
                temp_seq = str(concat_cache(i[0]))
                temp_token = nlp(temp_seq)
                if (temp_token[-1]).pos_ not in ["NOUN", "PROPN", "VERB"]:
                    checker2 = False

        if checker2 == True:
            filtred_ngrams.append(i)
    return filtred_ngrams


# Cleaning phrases from stop words,
# if a word in a phrase is a stop word, the phrase is deleted. all words in the phrase are checked except prepositions and PROPN
# at the input is a list of phrases and a list of stop words
# at the output is a filtered list
def filter_stop_words(mwe_list, stop_words):
    filtred_ngrams = []
    for mwe in mwe_list:
        checker3 = True
        temp = mwe
        if mwe[2][0] == "ADP" or mwe[2][-1] == "ADP":
            checker3 = False

        for i, w in enumerate(mwe[0]):
            if mwe[2][i] not in ["PROPN", "ADP"]:
                if w in stop_words:
                    checker3 = False

        if checker3 == True:
            filtred_ngrams.append(mwe)
    return filtred_ngrams


# extracting candidates based on part-of-speech templates
# input: list of tokens in one sentence [("token1", pos, index),("token2", pos, index),("token3", pos, index)] and part-of-speech template
# output: list of extracted candidates with information about them:
# [[list of words in the phrase], template by which the candidate was extracted, sequence of parts of speech of the candidate, indexes of word positions, number of words, number of characters in the candidate]
def filter_ngrams_by_pos_tag(sentence, sequense):
    filtered_ngrams = []
    seen = set()
    for seq in sequense:
        for i in range(len(sentence)):
            temp = []
            temp_index = []
            temp_pos = []
            checker = True

            if ((sentence[i][1] in seq[0]) or (sentence[i][1] == seq[0])) and (
                sentence[i][0] not in punc_without
            ):
                seq_index = 0
                sent_index = 0

                while (
                    seq_index < len(seq)
                    and i + sent_index < len(sentence)
                    and checker == True
                ):
                    if (sentence[i + sent_index][1] in seq[seq_index]) or (
                        sentence[i + sent_index][0] in seq[seq_index]
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
                            if (len(temp) > 1 or "-" in "".join(temp)) and (
                                len(set("".join(temp)).intersection(punc_without)) == 0
                            ):
                                temp_2 = temp.copy()
                                temp_pos2 = temp_pos.copy()
                                temp_index2 = temp_index.copy()
                                _key = (
                                    tuple(w.lower() for w in temp_2),
                                    str(seq),
                                    tuple(temp_index2),
                                )
                                if _key not in seen and temp[-1] not in punc_without:
                                    seen.add(_key)
                                    temp_2 = [word.lower() for word in temp_2]
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

                if (
                    seq_index == len(seq)
                    and (len(temp) > 1 or "-" in "".join(temp))
                    and len(set("".join(temp)).intersection(punc_without)) == 0
                ):
                    _key = (tuple(w.lower() for w in temp), str(seq), tuple(temp_index))
                    if _key not in seen and temp[-1] not in punc_without:
                        seen.add(_key)
                        temp = [word.lower() for word in temp]
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


def _compute_rectified_frequencies(possible_mwe, phrase_positions):
    """
    Compute raw and rectified frequencies for each phrase.
    possible_mwe must be sorted longest-first.
    Returns list of [phrase_tuple, f_raw, f_req] in the same order.
    """
    phrase_pos_sets = {pt: set(pos) for pt, pos in phrase_positions.items()}
    blocked = defaultdict(set)

    result = []
    for phrase_tuple in possible_mwe:
        n = len(phrase_tuple)
        positions = phrase_pos_sets.get(phrase_tuple, set())
        f_raw = len(positions)
        f_req = sum(1 for s in positions if s not in blocked[phrase_tuple])
        result.append([phrase_tuple, f_raw, f_req])

        # Block sub-span start positions for all shorter sub-phrases covered by this phrase
        for start in positions:
            for sub_len in range(1, n):
                for offset in range(n - sub_len + 1):
                    sub_tuple = phrase_tuple[offset : offset + sub_len]
                    if sub_tuple in phrase_pos_sets:
                        blocked[sub_tuple].add(start + offset)

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


# main body of phrase extraction
# input text and various parameters
# output list of phrases: ["phrase 1", "phrase 2", "phrase 3"]
class CandidateExtractor:
    def __init__(
        self,
        path,
        stop_words_path="stop_words_en.txt",
        list_seq=pos_tag_patterns,
        cohesion_filter=True,
        ngrams_filtered=3,
        additional_text="1",
        f_raw_sc=9,
        f_req_sc=3,
    ):
        self.path = path  # text in original case

        self.cohesion_filter = cohesion_filter  #  Enable or disable the cohesive filter
        # the minimum n for ngrams that will go through the filter
        self.ngrams_filtered = ngrams_filtered
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
            print(f"!!!!!!!!!!! stop words file not found at: {stop_path}")
            self.stop_words = set()
        except Exception as e:
            print(f"!!!!!!!!!!! stop words file could not be opened: {e}")
            self.stop_words = set()

    def get_corpus(self, path: str) -> str:
        file_list = os.listdir(path)
        texts = []

        for filename in file_list:
            if filename.endswith(".txt"):
                file_path = os.path.join(path, filename)
                with open(file_path, "r") as file:
                    text = file.read()
                    texts.append(
                        text.replace("  ", " ").replace(" -", "-").replace(" - ", "-")
                    )

        all_texts = " .".join(texts)
        return all_texts

    def all_candidates(self):

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

        corpus = self.get_corpus(self.path)

        # removing extra spaces
        corpus = (
            corpus.replace(" -", "-")
            .replace("- ", "-")
            .replace(" '", "'")
            .replace("  ", " ")
        )

        #   text tokenization, parts of speech and position index
        doc = nlp(corpus)
        text_sent_tokens = tokinizer(doc)
        mwe_list = []

        # extracting candidates from each sentence separately
        for sent in text_sent_tokens:
            temp_mwe_list = filter_ngrams_by_pos_tag(
                sent, self.list_seq
            )  # Part-of-speech extraction
            # REVIEW COMMENTED OUT --> this is so extremely slow b/c nlp is called on
            # THOUSANDS of tokens with no benefit
            """ temp_mwe_list = filter_propn_noun(
                temp_mwe_list, nlp
            )  #  filtering from changing parts of speech """
            temp_mwe_list = filter_stop_words(
                temp_mwe_list, self.stop_words
            )  # stop word filtering

            # temp_mwe_list contains lists of candidates with additional information about them:
            # [[list of words of the phrase], the template by which the candidate was extracted, the sequence of parts of speech of the candidate, the indices of the positions of words, the number of words, the number of characters in the candidate]

            sent_text = " ".join([s[0] for s in sent])
            temp_mwe_list = [mwe + [sent_text] for mwe in temp_mwe_list]
            mwe_list += temp_mwe_list

        # sentence lookup for all extracted phrases
        phrase_sents = defaultdict(list)
        for mwe in mwe_list:
            phrase_sents[concat_cache(mwe[0])].append(mwe[-1])

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

        # if the cohesive filter is on
        if self.cohesion_filter == True:
            short_mwes = {
                k: v
                for k, v in candidate_map.items()
                if len(k.split()) <= self.ngrams_filtered
            }
            long_mwes = {
                k: v
                for k, v in candidate_map.items()
                if len(k.split()) > self.ngrams_filtered
            }

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

        candidate_map = {term: phrase_sents.get(term, []) for term in set(candidates)}

        print(f"#### number of mwe extracted: {len(candidate_map)}")

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

        corpus = self.get_corpus(self.path)
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

        print(f"#### number of unigrams extracted: {len(unigram_map)}")
        return dict(unigram_map)
