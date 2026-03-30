# dataclasses for better organization
# tuples were driving me insane

from dataclasses import dataclass, field
import numpy as np


@dataclass
class CandidateSummary:
    sentence_indices: list[int] = field(default_factory=list)
    sent_embeds: np.ndarray = field(default_factory=lambda: np.array([]))


@dataclass
class TermSummary:
    word_embeds: np.ndarray = field(default_factory=lambda: np.array([]))
    sent_embeds: np.ndarray = field(default_factory=lambda: np.array([]))


@dataclass
class TermEmbeddings:
    word_embed: np.ndarray = field(default_factory=lambda: np.array([]))
    sent_embeds: np.ndarray = field(default_factory=lambda: np.array([]))
