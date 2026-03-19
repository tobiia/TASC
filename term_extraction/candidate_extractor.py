from os import listdir
from os.path import join
import spacy
from spacy.matcher import Matcher
from transformers import BertTokenizer
import requests
from tqdm import tqdm
from collections import defaultdict
from spacy.tokens import Doc, Span
from typing import Tuple

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
stopword_url = "https://raw.githubusercontent.com/term-extraction-project/stop_words/main/stop_words_en.txt"
stop_words = set(requests.get(stopword_url).text.split(","))
nlp = spacy.load("en_core_web_sm", disable=["ner", "parser", "lemmatizer"])
# prevent spacy from stopping on long docs
nlp.max_length = 2_000_000

from spacy.lang.char_classes import ALPHA, ALPHA_LOWER, ALPHA_UPPER
from spacy.lang.char_classes import CONCAT_QUOTES, LIST_ELLIPSES, LIST_ICONS
from spacy.util import compile_infix_regex
from operator import itemgetter

# TODO ensure hyphens are extracted as unigrams
# TODO error-checking

N = ["PROPN", "NOUN"]
K = ["ADJ", "PROPN", "NOUN"]
M = ["VERB", "ADV", "X"]

pos_patterns = [
    [
        {"POS": {"IN": N}, "OP": "+"},
    ],
    [
        {"POS": "ADJ", "OP": "+"},
        {"POS": {"IN": N}, "OP": "+"},
    ],
    [
        {"POS": "ADJ", "OP": "+"},
    ],
    [
        {"POS": "VERB"},
        {"POS": "ADJ"},
        {"POS": {"IN": N}, "OP": "+"},
    ],
    [
        {"POS": {"IN": N}, "OP": "+"},
        {"POS": "ADJ", "OP": "+"},
        {"POS": {"IN": N}, "OP": "+"},
    ],
    [
        {"POS": "ADJ"},
        {"POS": "VERB"},
        {"POS": {"IN": N}, "OP": "+"},
    ],
    [
        {"POS": "VERB", "OP": "+"},
        {"POS": {"IN": N}, "OP": "+"},
    ],
    [
        {"POS": "ADV", "OP": "+"},
        {"POS": "ADJ", "OP": "+"},
    ],
    [
        {"POS": {"IN": N}},
        {
            "POS": "ADP",
        },
        {"POS": {"IN": N}},
    ],
    [
        {"POS": {"IN": K}, "OP": "+"},
        {"POS": "ADP"},
        {"POS": {"IN": N}, "OP": "+"},
    ],
    [
        {"POS": {"IN": M}, "TEXT": {"REGEX": ".*-.*"}},
    ],
]


class CandidateExtractor:

    def __init__(
        self,
        path=None,
        text=None,
        stop_words=stop_words,
        ngram_max_length=4,
        uni_subtoken_threshold=4,
    ):
        self.path = path
        self.text = text
        self.stop_words = stop_words or set()
        self.ngram_max_length = ngram_max_length
        self.uni_subtoken_threshold = uni_subtoken_threshold

        self.matcher = Matcher(nlp.vocab)
        self.matcher.add("POS_PATTERNS", pos_patterns)

        self.uni_matcher = Matcher(nlp.vocab)
        self.uni_matcher.add("UNIGRAM_PATTERN", [[{"POS": {"IN": list(K)}}]])

        # infix pattern taken from: https://github.com/term-extraction-project/multi_word_expressions/blob/main/extractors/english.py
        # changes the tokenizer so that it does not separate words with hyphens
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

    def stream_corpus(self, path: str):
        for filename in listdir(path):
            full_path = join(path, filename)
            with open(full_path) as f:
                yield f.read()

    def clean_text(self, text: str) -> str:
        normal = ["-", '"', "'"]
        dashes = ["-", "−", "‐"]
        double_quotes = ['"', "“", "”", "„", "„", "„"]
        single_quotes = ["'", "`", "´", "’", "‘", "’"]

        i = -1
        for char_list in [dashes, double_quotes, single_quotes]:
            i += 1
            for j in range(len(char_list)):
                text = text.replace(char_list[j], normal[i])

        text = text.strip()
        text = text.replace("  ", " ")
        text = text.replace(" -", "-").replace(" - ", "-")
        return text

    def process_corpus(self) -> Tuple[dict[str, list[str]], dict[str, list[str]]]:
        all_ngrams = []
        all_unigrams = []

        if self.path:
            docs = tqdm(
                nlp.pipe(
                    (self.clean_text(t) for t in self.stream_corpus(self.path)),
                    batch_size=32,
                    n_process=4,
                ),
                desc="Processing documents...",
            )
        elif self.text:
            docs = [nlp(self.text)]
        else:
            raise FileExistsError(
                "You have not provided text to extract candidates from."
            )

        for doc in tqdm(docs, desc="Extracting candidates..."):

            ngrams = self.extract_multiwords(doc)
            unigrams = self.extract_unigrams(doc)

            ngrams = self.filter_stopwords(ngrams)

            unigrams = self.filter_tokenizer(unigrams)

            all_unigrams.extend(unigrams)
            all_ngrams.extend(ngrams)

        unigram_grouped = self.group_candidates(all_unigrams)
        ngram_grouped = self.group_candidates(all_ngrams)

        return unigram_grouped, ngram_grouped

    def extract_multiwords(self, doc: Doc) -> list[Tuple[Span, Span]]:
        candidate_tuples = []
        matches = self.matcher(doc, as_spans=True)

        for span in matches:
            if len(span) <= self.ngram_max_length:
                candidate_tuples.append((span, span.sent))

        return candidate_tuples

    def extract_unigrams(self, doc: Doc) -> list[Tuple[Span, Span]]:
        candidate_tuples = []
        matches = self.uni_matcher(doc, as_spans=True)

        for span in matches:
            if span.text.lower() not in self.stop_words:
                candidate_tuples.append((span, span.sent))

        return candidate_tuples

    def filter_stopwords(
        self, ngram_tuples: list[Tuple[Span, Span]]
    ) -> list[Tuple[Span, Span]]:
        filtered = []
        for span, sent in ngram_tuples:
            tokens = [
                token.text.lower()
                for token in span
                # ignore prepositions + proper nouns
                if token.pos_ not in {"ADP", "PROPN"}
            ]

            if not any(token in self.stop_words for token in tokens):
                filtered.append((span, sent))
        return filtered

    def filter_tokenizer(
        self, candidate_tuples: list[Tuple[Span, Span]]
    ) -> list[Tuple[Span, Span]]:
        filtered = []
        for span, sent in candidate_tuples:
            subtokens = tokenizer.tokenize(span.text)
            if len(subtokens) <= self.uni_subtoken_threshold:
                filtered.append((span, sent))
        return filtered

    def group_candidates(
        self, candidate_tuples: list[Tuple[Span, Span]]
    ) -> dict[str, list[str]]:
        # restructuring the candidate tuples now that POS info isn't needed
        grouped = defaultdict(list)

        for span, sent in candidate_tuples:
            text = span.text
            key = text.lower() if not text.islower() else text
            grouped.setdefault(key, []).append(sent.text)
        return grouped
