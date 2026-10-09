"""The simple interface: edition + level + image -> ranked Vienna codes.

Two stages. A vision-language model describes the figurative elements of the image
in prose (``describe``); an embedder maps that prose to a vector that is scored
against the vectors of the classification entries with the hierarchy heuristics of
``search.scorer`` (``classify_text``). Both stages can be used alone: a cached
description can be classified again without the vision model, and a text written by
hand is a valid query.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

from .config import LEVELS, default_describer, default_model, describer_spec, home
from .describe import DEFAULT_PROMPT, Describer, get_describer
from .embeddings import Embedder, get_embedder, model_slug
from .embeddings.base import normalize as normalize_vectors
from .index import ViennaIndex, available_indexes, scheme_table_path
from .search import Beam, Match, SearchParams, Weights, search
from .textnorm import normalize_query, split_sentences

CHUNKINGS = ("whole", "mean", "max")


class ViennaClassifier:
    """Loads one built index and answers queries against it.

    Construction is cheap; the index, the embedder and the vision model load on
    first use.
    """

    def __init__(
        self,
        edition: str = "latest",
        *,
        lang: str = "EN",
        model: str | Embedder | None = None,
        describer: str | Describer | None = None,
        notes: bool = False,
        root: Path | None = None,
    ):
        """``describer`` is ``"heavy"`` (Qwen2.5-VL 7B through Ollama, the default),
        ``"light"`` (SmolVLM-256M through transformers, no Ollama) or a spec."""
        self.root = root or home()
        self.lang = lang.upper()
        self.notes = notes
        self._model_spec = model or default_model()
        spec = describer or default_describer()
        self._describer_spec = describer_spec(spec) if isinstance(spec, str) else spec
        self.edition = resolve_built_edition(edition, self.lang, self._model_name, notes, self.root)
        self._index: ViennaIndex | None = None
        self._embedder: Embedder | None = None
        self._describer: Describer | None = None
        #: The description the last ``classify`` call scored.
        self.last_description: str | None = None

    @property
    def _model_name(self) -> str:
        return self._model_spec if isinstance(self._model_spec, str) else self._model_spec.name

    @property
    def index(self) -> ViennaIndex:
        if self._index is None:
            path = _index_for(self.edition, self.lang, self._model_name, self.notes, self.root)
            self._index = ViennaIndex.read(
                path, scheme_table_path(self.edition, self.lang, self.root)
            )
        return self._index

    @property
    def embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = get_embedder(self._model_spec)
            if self._embedder.name != self.index.meta.model:
                raise ValueError(
                    f"Index built with {self.index.meta.model}, embedder is {self._embedder.name}"
                )
        return self._embedder

    @property
    def describer(self) -> Describer:
        if self._describer is None:
            self._describer = get_describer(self._describer_spec)
        return self._describer

    def describe(self, image: Path | bytes, *, prompt: str = DEFAULT_PROMPT) -> str:
        """Stage one alone: the vision model's inventory of the figurative elements."""
        return self.describer.describe(image, prompt=prompt)

    def classify(
        self,
        image: Path | bytes,
        level: str = "section",
        top_k: int = 10,
        *,
        prompt: str = DEFAULT_PROMPT,
        **params,
    ) -> list[Match]:
        """Describe the image, then rank the classification entries for the description.

        Keyword arguments go to ``classify_text`` (``gap``, ``weights``, ``beam``,
        ``auto_margin``, ``principal_only``).
        """
        self.last_description = self.describe(image, prompt=prompt)
        return self.classify_text(self.last_description, level=level, top_k=top_k, **params)

    def classify_text(
        self,
        text: str,
        level: str = "section",
        top_k: int = 10,
        *,
        gap: float | None = None,
        weights: Weights | None = None,
        beam: Beam | None = None,
        auto_margin: float = 0.02,
        principal_only: bool = False,
        exclude_codes: tuple[str, ...] = (),
        chunking: str = "whole",
        normalize: bool = True,
    ) -> list[Match]:
        """Stage two alone: rank entries for a description (or any text).

        ``chunking`` says how the sentences of the description are embedded:
        ``"whole"`` embeds the text as one query; ``"mean"`` embeds each sentence and
        searches with their unit-length mean; ``"max"`` embeds each sentence and
        scores an entry by its best sentence, the reading of general note (e): one
        code per distinct element, each usually its own sentence.
        """
        if normalize:
            text = normalize_query(text)
        params = SearchParams(
            level=level,
            top_k=top_k,
            gap=gap,
            weights=weights or Weights(),
            beam=beam or Beam(),
            auto_margin=auto_margin,
            principal_only=principal_only,
            exclude_codes=tuple(exclude_codes),
        )
        return search(self.index, self.embed_text(text, chunking=chunking), params)

    def embed_text(self, text: str, *, chunking: str = "whole") -> np.ndarray:
        """Query vector for a text; a ``(sentences, dim)`` stack for ``"max"``."""
        if chunking not in CHUNKINGS:
            raise ValueError(f"chunking must be one of {CHUNKINGS}")
        if chunking == "whole":
            return self.embedder.embed_query(text)
        sentences = split_sentences(text)
        if len(sentences) == 1:
            return self.embedder.embed_query(text)
        vectors = self.embedder.embed_queries(sentences)
        if chunking == "max":
            return vectors
        return normalize_vectors(vectors.mean(axis=0))[0]


def classify(
    image: Path | bytes,
    edition: str = "latest",
    level: str = "section",
    top_k: int = 10,
    *,
    lang: str = "EN",
    model: str | None = None,
    describer: str | None = None,
    notes: bool = False,
    root: Path | None = None,
    **kwargs,
) -> list[Match]:
    """One-call form. Classifiers are cached per (edition, lang, model, describer)."""
    if level not in (*LEVELS, "auto"):
        raise ValueError(f"level must be one of {LEVELS} or 'auto'")
    clf = _cached(
        edition,
        lang.upper(),
        model or default_model(),
        describer or default_describer(),
        notes,
        str(root) if root else None,
    )
    return clf.classify(image, level=level, top_k=top_k, **kwargs)


@lru_cache(maxsize=8)
def _cached(
    edition: str, lang: str, model: str, describer: str, notes: bool, root: str | None
) -> ViennaClassifier:
    return ViennaClassifier(
        edition,
        lang=lang,
        model=model,
        describer=describer,
        notes=notes,
        root=Path(root) if root else None,
    )


def resolve_built_edition(spec: str, lang: str, model: str, notes: bool, root: Path) -> str:
    """Resolve ``latest`` against indexes already built for lang, model and text style."""
    if spec.isdigit():
        return spec
    slug = model_slug(model)
    editions = sorted(
        (
            r.edition
            for r in available_indexes(root)
            if r.lang == lang and r.model_slug == slug and r.notes == notes
        ),
        key=int,
    )
    if not editions:
        raise FileNotFoundError(
            f"No index built for lang={lang} model={model} notes={notes} under {root}. "
            "Run: i2vienna build --edition 10"
        )
    return editions[-1]


def _index_for(edition: str, lang: str, model: str, notes: bool, root: Path) -> Path:
    slug = model_slug(model)
    for r in available_indexes(root):
        if r.edition == edition and r.lang == lang and r.model_slug == slug and r.notes == notes:
            return r.path
    raise FileNotFoundError(
        f"No index for edition={edition} lang={lang} model={model} notes={notes} under {root}. "
        f"Run: i2vienna build --edition {edition} --lang {lang} --model {model}"
        + (" --notes" if notes else "")
    )
