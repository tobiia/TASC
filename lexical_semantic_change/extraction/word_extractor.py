from collections import defaultdict
from pathlib import Path
import string
import re
import logging
import spacy

from spacy.lang.char_classes import ALPHA, ALPHA_LOWER, ALPHA_UPPER
from spacy.lang.char_classes import CONCAT_QUOTES, LIST_ELLIPSES, LIST_ICONS
from spacy.util import compile_infix_regex
from tqdm import tqdm

from ..config import EXTRACT_DIR

logger = logging.getLogger(__name__)

# parts of speech templates
pos_tag_patterns = ["PROPN", "NOUN", "ADJ", "VERB"]

# setting up punctuation lists to check, punc_without does not contain hyphens and apostrophes as they can be part of phrases. punc_all is needed to check if there is a hyphen at the beginning or end of a phrase
punc_without = set(string.punctuation)
punc_without.update(["»", "«"])
punc_all = punc_without.copy()
punc_without.remove("-")
punc_without.remove("'")


def remove_punc_spaces(text):
    # remove spaces before common punctuation marks
    text = re.sub(r"\s+([.,;:!?)\]}–—])", r"\1", text)
    # remove spaces around apostrophes
    text = re.sub(r"\s+[''´`]\s*", "'", text)
    return text


class WordExtractor:
    def __init__(
        self,
        corpus_path: str,
        stop_words_path: str = "stop_words_en.txt",
        list_seq=pos_tag_patterns,
    ):
        """Initialize WordExtractor.

        Args:
            corpus_path: Path to directory containing .txt files
            stop_words_path: Filename of stop words file, must be placed
                in the same directory, EXTRACTION
            list_seq: List of POS tags to extract
        """
        corpus_path_obj = Path(corpus_path)
        if not corpus_path_obj.exists():
            raise FileNotFoundError(f"Corpus path does not exist: {corpus_path}")

        self.corpus_path = corpus_path
        self.list_seq = list_seq

        try:
            self.model_nlp = spacy.load(
                "en_core_web_sm",
                disable=[
                    "ner",
                    "parser",
                ],
            )
            self.model_nlp.add_pipe("sentencizer")
            self.model_nlp.max_length = 50_000_000
        except Exception as e:
            logger.error(f"Failed to load spaCy model: {e}")
            raise

        try:
            stop_path = EXTRACT_DIR / stop_words_path
            with open(stop_path, encoding="utf-8") as f:
                self.stop_words = set(f.read().split(","))
            logger.info(f"Loaded {len(self.stop_words)} stop words")
        except FileNotFoundError:
            logger.warning(
                f"Stop words file not found at: {stop_path}. Proceeding without filtering."
            )
            self.stop_words = set()
        except Exception as e:
            logger.warning(
                f"Failed to load stop words: {e}. Proceeding without filtering."
            )
            self.stop_words = set()

    def _file_chunks(self, filepath: Path, max_chars: int = 50_000):
        """Yield cleaned paragraph-level chunks without loading the whole file.

        Accumulates lines until a blank line is hit, then yields the cleaned
        chunk. Only one paragraph is held in memory at a time, so large files
        are streamed rather than read whole. Paragraphs longer than max_chars
        are split at the nearest space boundary to avoid OOM in the NLP pipeline.
        """

        def _clean(parts: list[str]) -> str:
            return (
                " ".join("".join(parts).split())
                .replace(" -", "-")
                .replace("- ", "-")
                .replace(" '", "'")
            )

        def _emit(chunk: str):
            while len(chunk) > max_chars:
                split_at = chunk.rfind(" ", 0, max_chars)
                if split_at == -1:
                    split_at = max_chars
                yield chunk[:split_at]
                chunk = chunk[split_at:].lstrip()
            if chunk:
                yield chunk

        current: list[str] = []
        with open(filepath, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    current.append(line)
                elif current:
                    chunk = _clean(current)
                    if chunk:
                        yield from _emit(chunk)
                    current = []
        if current:
            chunk = _clean(current)
            if chunk:
                yield from _emit(chunk)

    def unigram_extraction(self, max_sents_per_word=200) -> dict[str, list[str]]:
        """Extract unigram candidates (single tokens) from corpus.

        Args:
            max_sents_per_word: Maximum sentences to store per word

        Returns:
            dict mapping word -> list of sentences
        """
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

        uni_pos_set = set(p for p in self.list_seq if isinstance(p, str))
        unigram_map = defaultdict(list)

        file_list = list(Path(self.corpus_path).glob("*.txt"))
        if not file_list:
            raise ValueError(f"No .txt files found in {self.corpus_path}")

        logger.info(f"Extracting from {len(file_list)} corpus files")

        for filepath in file_list:
            logger.debug(f"Processing {filepath.name}...")
            for doc in tqdm(
                nlp.pipe(self._file_chunks(filepath), batch_size=32),
                desc="*** extracting words...",
            ):
                for token in doc:
                    if token.pos_ not in uni_pos_set:
                        continue
                    w = token.lower_
                    if not w:
                        continue
                    if (
                        w in self.stop_words
                        or w[0] in punc_all
                        or w[-1] in punc_all
                        or any(c in punc_without for c in w)
                        or any(c.isdigit() for c in w)
                    ):
                        continue
                    sents = unigram_map[w]
                    if len(sents) < max_sents_per_word:
                        sents.append(remove_punc_spaces(token.sent.text.lower()))

        logger.info(f"Total number of unigrams extracted: {len(unigram_map)}")
        return dict(unigram_map)

    def targeted_extraction(
        self, terms: list[str], max_sents_per_word: int = 200
    ) -> dict[str, list[str]]:
        """Extract sentences containing specific terms using fast regex string matching.

        Args:
            terms: List of terms to search for (matched case-insensitively at word boundaries)
            max_sents_per_word: Maximum sentences to store per term

        Returns:
            dict mapping term -> list of sentences (only terms that appear in the corpus)
        """
        if not terms:
            return {}

        term_set = {t.lower() for t in terms}
        # longest terms first so overlapping patterns match greedily
        pattern = re.compile(
            r"\b("
            + "|".join(re.escape(t) for t in sorted(term_set, key=len, reverse=True))
            + r")\b"
        )

        result: dict[str, list[str]] = defaultdict(list)

        file_list = list(Path(self.corpus_path).glob("*.txt"))
        if not file_list:
            raise ValueError(f"No .txt files found in {self.corpus_path}")

        logger.info(
            f"Targeted extraction of {len(term_set)} terms from {len(file_list)} files"
        )

        for filepath in tqdm(file_list, desc="*** targeted extraction..."):
            for doc in self.model_nlp.pipe(
                self._file_chunks(filepath),
                batch_size=64,
                disable=["tok2vec", "tagger"],
            ):
                for spacy_sent in doc.sents:
                    sent_text = remove_punc_spaces(spacy_sent.text.strip().lower())
                    if not sent_text:
                        continue
                    found = {m.group(1) for m in pattern.finditer(sent_text)}
                    for term in found:
                        if len(result[term]) < max_sents_per_word:
                            result[term].append(sent_text)

        logger.info(f"Found {len(result)}/{len(term_set)} requested terms")
        return dict(result)

    # TODO extract sentences + definitions for sense disambiguation
