# dataclasses for better organization
# tuples were driving me insane

from dataclasses import dataclass, field
import numpy as np
from spacy.tokens import Span


@dataclass
class CandidateSummary:
    tok_sents: list = field(default_factory=list)
    tok_embeds: list = field(default_factory=list)
    sent_embeds: list = field(default_factory=list)


@dataclass
class TermSummary:
    word_embeds: list = field(default_factory=list)
    sent_embeds: list = field(default_factory=list)


@dataclass
class TermEmbeddings:
    word_embed: np.ndarray = field(default_factory=lambda: np.array([]))
    sent_embeds: list = field(default_factory=list)
