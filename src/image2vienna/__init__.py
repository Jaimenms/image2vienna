"""image2vienna: trade mark image -> Vienna Classification codes.

A vision-language model describes the figurative elements of the image; the
description is embedded and scored against the embedded classification entries with
hierarchy-aware heuristics, the strategy of text2ipc applied to images.
"""

from .classifier import ViennaClassifier, classify
from .search import Match

__version__ = "0.1.0"

__all__ = ["Match", "ViennaClassifier", "classify", "__version__"]
