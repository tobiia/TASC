from collections import defaultdict
import string
import re
import spacy
import os

from spacy.lang.char_classes import ALPHA, ALPHA_LOWER, ALPHA_UPPER
from spacy.lang.char_classes import CONCAT_QUOTES, LIST_ELLIPSES, LIST_ICONS
from spacy.util import compile_infix_regex

# from spacy.tokens import Doc, Token

from ..config import EXTRACT_DIR

# Parts of speech templates
pos_tag_patterns = ["PROPN", "NOUN", "ADJ", "VERB", "ADV"]

# Setting up punctuation lists to check, punc_without does not contain hyphens and apostrophes as they can be part of phrases. punc_all is needed to check if there is a hyphen at the beginning or end of a phrase
punc_without = set(string.punctuation)
punc_without.update(["»", "«"])
punc_all = punc_without.copy()
punc_without.remove("-")
punc_without.remove("'")
CLAUSE_BREAKS = {",", ";", ":", "(", ")", "[", "]", "—", "–", "-", "/"}
NOISE_TOKENS = set(string.punctuation).union(("''", "``", "..."))


def remove_punc_spaces(text):
    # Remove spaces before common punctuation marks
    text = re.sub(r"\s+([.,;:!?)\]}–—])", r"\1", text)
    # Remove spaces around apostrophes
    text = re.sub(r"\s+[''´`]\s*", "'", text)
    return text


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


class WordExtractor:
    def __init__(
        self,
        corpus_path: str,
        stop_words_path: str = "stop_words_en.txt",
        list_seq=pos_tag_patterns,
    ):
        self.corpus_path = corpus_path  # FULL PATH

        self.list_seq = list_seq  # list of part of speech patterns

        self.model_nlp = spacy.load(
            "en_core_web_sm", disable=["ner", "parser"]
        )  # Spacy model
        self.model_nlp.add_pipe("sentencizer")
        # prevent spacy from stopping on long docs
        self.model_nlp.max_length = 2_000_000

        try:
            stop_path = EXTRACT_DIR / stop_words_path
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
                with open(file_path, "r", encoding="utf-8") as file:
                    text = file.read()
                    texts.append(
                        text.replace("  ", " ").replace(" -", "-").replace(" - ", "-")
                    )

        all_texts = " .".join(texts)
        return all_texts

    def unigram_extraction(self) -> dict[str, list[str]]:
        """Extract only unigram candidates (single tokens) from the corpus"""
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
            ):
                continue
            unigram_map[token.lower_].append(token.sent.text.lower())

        print(f"WORD EXTRACTOR: TOTAL number of unigrams extracted: {len(unigram_map)}")

        unigram_map = {
            term: [remove_punc_spaces(sent) for sent in sents]
            for term, sents in unigram_map.items()
        }

        return dict(unigram_map)

    # TODO extract sentences + definitions for sense disambiguation


def extract_common_words(corpus1_path, corpus2_path):
    extractor = WordExtractor(corpus1_path)
    corpus1 = extractor.unigram_extraction()
    extractor.corpus_path = corpus2_path
    corpus2 = extractor.unigram_extraction()
    shared_words = corpus1.keys() & corpus2.keys()
    shared_corpus1 = {k: corpus1[k] for k in shared_words}
    shared_corpus2 = {k: corpus2[k] for k in shared_words}
    return shared_corpus1, shared_corpus2
