# dataclasses for better organization
# tuples were driving me insane

from dataclasses import dataclass, field
import numpy as np


@dataclass
class TermSummary:
    word_embeds: np.ndarray = field(default_factory=lambda: np.array([]))
    sent_embeds: np.ndarray = field(default_factory=lambda: np.array([]))


# need to change so i keep the sentences tbh
