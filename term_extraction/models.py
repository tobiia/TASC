# dataclasses for better organization
# tuples were driving me insane

from dataclasses import dataclass, field
import numpy as np
from spacy.tokens import Span


@dataclass
class CandidateSummary:
    tok_sents: list = field(default_factory=list)
    tok_embeds: list = field(default_factory=list)  # list of ndarrays
    sent_embeds: np.ndarray = field(default_factory=lambda: np.array([]))


@dataclass
class TermSummary:
    word_embeds: np.ndarray = field(default_factory=lambda: np.array([]))
    sent_embeds: np.ndarray = field(default_factory=lambda: np.array([]))


@dataclass
class TermEmbeddings:
    word_embed: np.ndarray = field(default_factory=lambda: np.array([]))
    sent_embeds: np.ndarray = field(default_factory=lambda: np.array([]))
