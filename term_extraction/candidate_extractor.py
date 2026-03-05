from os import listdir
from os.path import join
import spacy
from spacy.matcher import Matcher
from transformers import BertTokenizer
import requests
from tqdm import tqdm

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
stopword_url = "https://raw.githubusercontent.com/term-extraction-project/stop_words/main/stop_words_en.txt"
stop_words = set(requests.get(stopword_url).text.split(","))
nlp = spacy.load("en_core_web_sm", disable=["ner", "parser", "lemmatizer"])
# prevent spacy from stopping on long docs
nlp.max_length = 2_000_000

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
        stop_words=None,
        ngram_max_length=4,
        uni_subtoken_threshold=3,
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

    def stream_corpus(self, path):
        for filename in listdir(path):
            full_path = join(path, filename)
            with open(full_path) as f:
                yield f.read()

    def clean_text(self, text):
        text = text.strip()
        text = text.replace("  ", " ")
        text = text.replace(" -", "-").replace(" - ", "-")
        return text

    def process_corpus(self):
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
            ngrams = self.filter_tokenizer(ngrams, self.ngram_max_length)

            unigrams = self.filter_tokenizer(unigrams, self.uni_subtoken_threshold)

            all_ngrams.extend(ngrams)
            all_unigrams.extend(unigrams)

        return all_unigrams, all_ngrams

    def extract_multiwords(self, doc):
        candidate_tuples = []
        matches = self.matcher(doc, as_spans=True)

        for span in matches:
            if len(span) <= self.ngram_max_length:
                candidate_tuples.append((span, span.sent))

        return candidate_tuples

    def extract_unigrams(self, doc):
        candidate_tuples = []
        matches = self.uni_matcher(doc, as_spans=True)

        for span in matches:
            if span.text.lower() not in self.stop_words:
                candidate_tuples.append((span, span.sent))

        return candidate_tuples

    def filter_stopwords(self, ngram_tuples):
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

    def filter_tokenizer(self, candidate_tuples, subtoken_threshold):
        filtered = []
        for span, sent in candidate_tuples:
            subtokens = tokenizer.tokenize(span.text)
            if len(subtokens) <= subtoken_threshold:
                filtered.append((span, sent))
        return filtered
