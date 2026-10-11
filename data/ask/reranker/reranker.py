#!/usr/bin/env python3

"""
SoakinGarri Reranker Program

PURPOSE
-------
Re-scores a short list of retrieval candidates with a cross-encoder, so the
single best 5-8 passages can be kept from a wider pool of 30-50. A vector
search (or BM25, or a hybrid merge of the two) scores a query and a passage
as two separately-embedded vectors compared by cosine similarity; a
cross-encoder instead reads the query and the passage together, as one input,
and predicts a relevance score directly. That extra context is usually the
single biggest quality gain available after retrieval, at the cost of being
too slow to run over an entire vector base — hence reranking only a short
list that a cheaper first-pass retriever has already narrowed down.

INPUT
-----
Any list of objects with a `.text` attribute: `vb.SearchResult` instances
from data/ask/vector_database/vector_base.py, BM25 hits, a hybrid merge of
the two, or anything else shaped the same way. This module does not import
vector_base.py, so it has no hard dependency on that type — see `rerank`'s
PARAMETERS section for exactly what is required of a candidate.

OUTPUT
------
A new list of `RerankedResult` objects: same `text`/`record` as the input
candidates, re-ordered and re-scored by the cross-encoder, truncated to
`top_k`.

MODEL
-----
BAAI/bge-reranker-v2-m3, loaded through sentence-transformers' CrossEncoder.

INSTALL
-------
python3 -m pip install sentence-transformers

USAGE
-----
This module is a library, meant to be imported by retrieval.py (a later task
wires that in) rather than run as its own pipeline stage over a data folder.
A minimal CLI is included for manual testing:

RUN
---
python3 reranker.py --query "how is momentum defined" \\
    --candidate-text "Momentum is the product of mass and velocity." \\
    --candidate-text "Refraction bends light as it crosses a boundary."
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Any, Protocol

DEFAULT_MODEL_NAME = "BAAI/bge-reranker-v2-m3"

DEFAULT_TOP_K = 8

DEPENDENCY_MESSAGE = (
    "Reranking requires sentence-transformers: "
    "python3 -m pip install sentence-transformers"
)


class CrossEncoderModel(Protocol):
    """
    PURPOSE
    -------
    Define the interface any cross-encoder backend must implement to be used
    by :func:`rerank`. Any object exposing `.predict` works, which is what
    lets the tests and doctests run against a tiny stub instead of
    downloading BAAI/bge-reranker-v2-m3.
    """

    def predict(self, pairs: list[tuple[str, str]]) -> Any:
        """
        PURPOSE
        -------
        Score each (query, passage) pair.

        PARAMETERS
        ----------
        pairs   A list of (query, passage text) tuples.

        RETURNS
        -------
        A sequence of scores, one per pair, in the same order; higher means
        more relevant. A real CrossEncoder returns a numpy array; a plain
        list of floats works equally well.
        """
        ...


@dataclass(slots=True)
class RerankedResult:
    """
    PURPOSE
    -------
    One ranked hit returned from :func:`rerank`. Deliberately not the same
    class as `vb.SearchResult` (this module does not import vector_base.py),
    but shaped the same way so a caller can treat either type the same: both
    carry `rank`, `score`, `text`, and `record`.

    ATTRIBUTES
    ----------
    rank            1-based position of this result after reranking.
    score           The cross-encoder's relevance score for (query, text).
    text            Passage text for this result, copied from the candidate.
    record          Full record for this result, copied from the candidate.
    original_rank   This candidate's `rank` before reranking (or its 1-based
                     position in the input list, when the candidate had no
                     `rank` attribute). Lets a caller compare before/after.
    """

    rank: int
    score: float
    text: str
    record: dict[str, Any] = field(repr=False)
    original_rank: int | None = None


def load_reranker(model_name: str = DEFAULT_MODEL_NAME) -> CrossEncoderModel:
    r"""
    PURPOSE
    -------
    Load the sentence-transformers cross-encoder used to rerank candidates.
    Tests pass their own stub model instead of calling this.

    PARAMETERS
    ----------
    model_name   The sentence-transformers cross-encoder identifier to load.

    RETURNS
    -------
    The loaded CrossEncoder.

    RAISES
    ------
    RuntimeError   When sentence-transformers is not installed; the message
                   includes the pip install command.
    """
    try:
        from sentence_transformers import CrossEncoder
    except ImportError as error:
        raise RuntimeError(DEPENDENCY_MESSAGE) from error

    return CrossEncoder(model_name)


def rerank(
    query: str,
    candidates: list[Any],
    model: CrossEncoderModel | None = None,
    top_k: int = DEFAULT_TOP_K,
) -> list[RerankedResult]:
    r"""
    PURPOSE
    -------
    Re-score `candidates` against `query` with a cross-encoder and return the
    best `top_k`, best first.

    PARAMETERS
    ----------
    query        The question text the candidates are being matched against.
    candidates   A list of candidate objects. Each one needs only a `.text`
                 attribute (the passage text to score) and a `.record`
                 attribute (an arbitrary dict carried through unchanged);
                 this is intentionally duck-typed rather than importing
                 `vb.SearchResult`, so it also accepts BM25 hits or a hybrid
                 merge of the two. When a candidate also has a `.rank`
                 attribute, it is copied onto the result as `original_rank`;
                 otherwise the candidate's 1-based position in `candidates`
                 is used instead.
    model        A loaded cross-encoder exposing `.predict(pairs) -> scores`
                 (see `CrossEncoderModel`). When None, `load_reranker()` is
                 called to load BAAI/bge-reranker-v2-m3.
    top_k        Maximum number of results to return. When `top_k` is at
                 least as large as `len(candidates)`, every candidate is
                 returned, reordered.

    RETURNS
    -------
    Up to `top_k` RerankedResult objects, highest cross-encoder score first.
    `[]` when `candidates` is empty or `top_k` is not positive.

    EXAMPLES
    --------
    >>> class FakeModel:
    ...     def predict(self, pairs):
    ...         return [1.0 if "heat" in text.casefold() else 0.0 for _, text in pairs]
    >>> class Candidate:
    ...     def __init__(self, rank, text):
    ...         self.rank = rank
    ...         self.text = text
    ...         self.record = {"text": text}
    >>> candidates = [
    ...     Candidate(1, "Light travels in straight lines."),
    ...     Candidate(2, "Heat flows from hot to cold."),
    ... ]
    >>> results = rerank("heat transfer", candidates, model=FakeModel(), top_k=2)
    >>> [(r.rank, r.original_rank, r.text) for r in results]
    [(1, 2, 'Heat flows from hot to cold.'), (2, 1, 'Light travels in straight lines.')]
    >>> rerank("heat transfer", candidates, model=FakeModel(), top_k=1)[0].text
    'Heat flows from hot to cold.'
    >>> rerank("heat transfer", [], model=FakeModel())
    []
    """
    if not candidates or top_k <= 0:
        return []

    if model is None:
        model = load_reranker()

    pairs = [(query, candidate.text) for candidate in candidates]
    scores = model.predict(pairs)

    scored = [
        (
            candidate,
            float(score),
            getattr(candidate, "rank", position),
        )
        for position, (candidate, score) in enumerate(
            zip(candidates, scores), start=1
        )
    ]

    scored.sort(key=lambda item: item[1], reverse=True)

    return [
        RerankedResult(
            rank=rank,
            score=score,
            text=candidate.text,
            record=candidate.record,
            original_rank=original_rank,
        )
        for rank, (candidate, score, original_rank) in enumerate(
            scored[:top_k], start=1
        )
    ]


def parse_arguments(argument_list: list[str] | None = None) -> argparse.Namespace:
    r"""
    PURPOSE
    -------
    Parse the command line for the manual-testing CLI.

    PARAMETERS
    ----------
    argument_list   The arguments to parse, or None to read from sys.argv.

    RETURNS
    -------
    The parsed argparse.Namespace.

    EXAMPLES
    --------
    >>> options = parse_arguments(["--query", "heat transfer",
    ...                             "--candidate-text", "Heat flows.",
    ...                             "--candidate-text", "Light bends."])
    >>> options.query, options.candidate_text, options.top_k
    ('heat transfer', ['Heat flows.', 'Light bends.'], 8)
    """
    parser = argparse.ArgumentParser(
        description="Rerank candidate passages against a query with a cross-encoder."
    )

    parser.add_argument(
        "--query",
        required=True,
        help="the question to match candidates against",
    )

    parser.add_argument(
        "--candidate-text",
        dest="candidate_text",
        action="append",
        required=True,
        help="candidate passage text; repeat for more candidates",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="how many reranked results to keep",
    )

    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_NAME,
        help="sentence-transformers cross-encoder to load",
    )

    return parser.parse_args(argument_list)


def main() -> None:
    """
    PURPOSE
    -------
    Rerank a handful of candidate texts typed on the command line, using the
    options given there. For manual testing only: real pipeline use goes
    through `rerank()` called from retrieval.py once that wiring exists.
    """
    options = parse_arguments()

    class TextCandidate:
        def __init__(self, rank: int, text: str):
            self.rank = rank
            self.text = text
            self.record: dict[str, Any] = {"text": text}

    candidates = [
        TextCandidate(rank, text)
        for rank, text in enumerate(options.candidate_text, start=1)
    ]

    model = load_reranker(options.model)
    results = rerank(options.query, candidates, model=model, top_k=options.top_k)

    for result in results:
        print(
            f"{result.rank}. [{result.score:.4f}] "
            f"(was #{result.original_rank}) {result.text}"
        )


if __name__ == "__main__":
    main()
