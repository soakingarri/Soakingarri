#!/usr/bin/env python3

"""
SoakinGarri Retrieval Program

PURPOSE
-------
Answers questions against the vector base: embeds a query with the same model
that built the base, finds the closest passages, and returns them with their
citations. This is the retrieval half of retrieval-augmented generation; it
finds and formats context, it does not generate answers.

MODEL CONSISTENCY
-----------------
A query must be embedded with the model that produced the base, or the distances
mean nothing. The base manifest records which model that was, and this program
refuses to run against a different one unless --allow-model-mismatch is given.

INPUT FOLDER
------------
<repo>/data/ask/vector_base

INSTALL
-------
python3 -m pip install sentence-transformers numpy

RUN
---
python3 retrieval.py --query "how is momentum defined"
python3 retrieval.py --query "stress and strain" --top-k 8 --dataset rag
python3 retrieval.py --query "thermodynamics" --filter topic="Chapter 4"
python3 retrieval.py --query "entropy" --context
python3 retrieval.py --query "entropy" --json
python3 retrieval.py --interactive

HYBRID SEARCH
-------------
--hybrid merges vector hits with BM25 hits from data/ask/keyword_index
(build it with keyword_index.py) using Reciprocal Rank Fusion (k=60).
--rerank also re-scores the top --candidate-pool fused hits with
BAAI/bge-reranker-v2-m3 and keeps --top-k.

python3 retrieval.py --query "agbara" --hybrid --language yor_Latn
python3 retrieval.py --query "entropy" --hybrid --source-type book --date-from 2020-01-01
python3 retrieval.py --query "entropy" --rerank --top-k 8
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

sys.path.insert(
    0, str(Path(__file__).resolve().parent.parent / "vector_database")
)
sys.path.insert(
    0, str(Path(__file__).resolve().parent.parent.parent.parent / "source")
)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "keyword_index"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "reranker"))

from vector_base import (
    DEFAULT_TOP_K,
    SOURCE_LABEL,
    SearchResult,
    VectorBase,
    normalize_rows,
)
from common import DATA_ROOT  # noqa: E402

VECTOR_BASE_FOLDER = DATA_ROOT / "vector_base"
KEYWORD_INDEX_FOLDER = DATA_ROOT / "keyword_index"

DEFAULT_CANDIDATE_POOL = 50
DEFAULT_RRF_K = 60

DEFAULT_MODEL_NAME = "BAAI/bge-m3"

DEFAULT_RELEVANCE_WEIGHT = 0.7
DEFAULT_MINIMUM_SCORE = 0.15

MAXIMUM_CONTEXT_CHARACTERS = 1200

QueryEncoder = Callable[[str], np.ndarray]


def load_query_encoder(
    model_name: str = DEFAULT_MODEL_NAME,
) -> tuple[QueryEncoder, str]:
    """
    PURPOSE
    -------
    Return an `encode(text) -> vector` callable and the model's name. Tests
    pass their own encoder instead of calling this.

    PARAMETERS
    ----------
    model_name   Name of the sentence-transformers model to load.

    RETURNS
    -------
    A tuple of the encode callable and the model name.

    RAISES
    ------
    RuntimeError   When sentence-transformers is not installed; the message
                   includes the pip install command.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as error:
        raise RuntimeError(
            "Retrieval requires sentence-transformers: "
            "python3 -m pip install sentence-transformers numpy"
        ) from error

    model = SentenceTransformer(model_name)

    def encode(text: str) -> np.ndarray:
        return np.asarray(
            model.encode(
                [text],
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )[0],
            dtype="float32",
        )

    return encode, model_name


def check_model_match(
    base: VectorBase,
    model_name: str,
    allow_mismatch: bool = False,
) -> str | None:
    r"""
    PURPOSE
    -------
    Compare the query model with the one recorded in the base. A query must
    be embedded with the model that produced the base, or the resulting
    distances mean nothing, so this refuses to run against a different model
    unless the caller explicitly allows the mismatch.

    PARAMETERS
    ----------
    base             The loaded vector base to check against.
    model_name       Name of the model about to embed the query.
    allow_mismatch   When True, downgrade a mismatch to a warning instead of
                      raising.

    RETURNS
    -------
    A warning string when the base does not record a model, or when a
    mismatch was explicitly allowed; ``None`` when the models agree.

    RAISES
    ------
    ValueError   When the recorded model and the query model disagree and
                 allow_mismatch is False.

    EXAMPLES
    --------
    >>> base = VectorBase(np.zeros((0, 0), dtype="float32"), [],
    ...                   {"embedding_model": "model-a"})
    >>> check_model_match(base, "model-a") is None
    True
    >>> check_model_match(base, "model-b")
    Traceback (most recent call last):
        ...
    ValueError: The vector base was built with 'model-a' but the query uses 'model-b'. Rebuild the base, or pass --allow-model-mismatch.
    >>> check_model_match(base, "model-b", allow_mismatch=True)
    "Model mismatch ignored: base 'model-a', query 'model-b'."
    >>> check_model_match(VectorBase(np.zeros((0, 0), dtype="float32"), [], {}), "model-a")
    'The vector base does not record which embedding model built it.'
    """
    recorded = base.embedding_model

    if not recorded:
        return "The vector base does not record which embedding model built it."

    if recorded == model_name:
        return None

    if allow_mismatch:
        return (
            f"Model mismatch ignored: base '{recorded}', query '{model_name}'."
        )

    raise ValueError(
        f"The vector base was built with '{recorded}' but the query uses "
        f"'{model_name}'. Rebuild the base, or pass --allow-model-mismatch."
    )


def parse_filter_arguments(values: Iterable[str] | None) -> dict[str, Any]:
    r"""
    PURPOSE
    -------
    Turn `field=value` command-line pairs into a filter dictionary. Repeating
    a field collects its values, which the base treats as "any of".

    PARAMETERS
    ----------
    values   The raw `field=value` strings from the command line, or None.

    RETURNS
    -------
    A dictionary mapping field name to a single value, or to a list of
    values when the field was repeated.

    RAISES
    ------
    ValueError   When an item does not contain an `=`.

    EXAMPLES
    --------
    >>> parse_filter_arguments(None)
    {}
    >>> parse_filter_arguments(["topic=Mechanics"])
    {'topic': 'Mechanics'}
    >>> parse_filter_arguments(["topic=Mechanics", "topic=Optics"])
    {'topic': ['Mechanics', 'Optics']}
    >>> parse_filter_arguments(["source_file=physics.pdf", "dataset=rag"])
    {'source_file': 'physics.pdf', 'dataset': 'rag'}
    >>> parse_filter_arguments(["broken"])
    Traceback (most recent call last):
        ...
    ValueError: Filters must look like field=value, not 'broken'.
    """
    filters: dict[str, Any] = {}

    for item in values or []:
        if "=" not in item:
            raise ValueError(
                f"Filters must look like field=value, not '{item}'."
            )

        field_name, _, value = item.partition("=")
        field_name = field_name.strip()
        value = value.strip()

        if field_name in filters:
            existing = filters[field_name]
            filters[field_name] = (
                existing + [value]
                if isinstance(existing, list)
                else [existing, value]
            )
        else:
            filters[field_name] = value

    return filters


def retrieve(
    base: VectorBase,
    query: str,
    encode: QueryEncoder,
    top_k: int = DEFAULT_TOP_K,
    filters: dict[str, Any] | None = None,
    relevance_weight: float = DEFAULT_RELEVANCE_WEIGHT,
    minimum_score: float = DEFAULT_MINIMUM_SCORE,
) -> list[SearchResult]:
    r"""
    PURPOSE
    -------
    Embed a query and return the closest records in the base.

    PARAMETERS
    ----------
    base               The loaded vector base to search.
    query              The question text to search for; must not be empty.
    encode             Callable turning query text into an embedding vector.
    top_k              Maximum number of results to return.
    filters            Optional field filters, as produced by
                        parse_filter_arguments.
    relevance_weight   How strongly to favour pure relevance over diversity;
                        1.0 is pure relevance.
    minimum_score      Drop passages scoring below this cosine similarity.

    RETURNS
    -------
    The list of matching SearchResult objects, closest first.

    RAISES
    ------
    ValueError   When the query is empty or whitespace-only.

    EXAMPLES
    --------
    >>> vectors = normalize_rows(np.array([[1.0, 0.0], [0.0, 1.0], [0.9, 0.1]]))
    >>> records = [
    ...     {"row": 0, "dataset": "rag", "topic": "Heat", "text": "Heat flows."},
    ...     {"row": 1, "dataset": "rag", "topic": "Light", "text": "Light travels."},
    ...     {"row": 2, "dataset": "llm", "topic": "Heat", "text": "Heat again."},
    ... ]
    >>> base = VectorBase(vectors, records, {"embedding_model": "stub"})
    >>> heat_query = lambda text: np.array([1.0, 0.0], dtype="float32")
    >>> [r.record["topic"] for r in retrieve(base, "heat", heat_query, top_k=2)]
    ['Heat', 'Heat']
    >>> [r.record["dataset"] for r in retrieve(base, "heat", heat_query, top_k=2,
    ...                                        filters={"dataset": "llm"})]
    ['llm']
    >>> retrieve(base, "heat", lambda text: np.array([0.0, 1.0], dtype="float32"),
    ...          top_k=3, minimum_score=0.9)[0].record["topic"]
    'Light'
    >>> retrieve(base, "   ", heat_query)
    Traceback (most recent call last):
        ...
    ValueError: The query is empty.
    """
    if not query or not query.strip():
        raise ValueError("The query is empty.")

    query_vector = normalize_rows(
        np.asarray(encode(query.strip()), dtype="float32").reshape(1, -1)
    )[0]

    return base.search(
        query_vector,
        top_k=top_k,
        filters=filters,
        relevance_weight=relevance_weight,
        minimum_score=minimum_score,
    )


def result_key(result: SearchResult) -> tuple[str, str]:
    r"""
    PURPOSE
    -------
    Identify a hit so the same passage found by vector search and by BM25
    is recognised as one passage. The LLM dataset holds several records per
    chunk, so record_id is preferred over chunk_id; the row number is the
    last resort.

    PARAMETERS
    ----------
    result   A SearchResult from either index.

    RETURNS
    -------
    A (dataset, identifier) tuple.

    EXAMPLES
    --------
    >>> result_key(SearchResult(1, 0.5, 3, "t", {"dataset": "llm", "record_id": "R1", "chunk_id": "C1"}))
    ('llm', 'R1')
    >>> result_key(SearchResult(1, 0.5, 3, "t", {"dataset": "rag", "record_id": None, "chunk_id": "C1"}))
    ('rag', 'C1')
    >>> result_key(SearchResult(1, 0.5, 3, "t", {}))
    ('', 'row:3')
    """
    record = result.record
    identifier = record.get("record_id") or record.get("chunk_id") or f"row:{result.row}"

    return str(record.get("dataset") or ""), str(identifier)


def within_date_range(
    record: dict[str, Any],
    date_from: str | None = None,
    date_to: str | None = None,
) -> bool:
    r"""
    PURPOSE
    -------
    Test a record's "date" field against an inclusive ISO date range. Only
    the first ten characters (YYYY-MM-DD) are compared, so full timestamps
    and plain dates mix freely; a bare year such as "1998" also works as a
    bound. A record without a date fails any range.

    PARAMETERS
    ----------
    record      The record to test.
    date_from   Earliest allowed date, or None for no lower bound.
    date_to     Latest allowed date, or None for no upper bound.

    RETURNS
    -------
    True when the record falls inside the range (or no range is given).

    EXAMPLES
    --------
    >>> within_date_range({"date": "2024-05-01T10:00:00+00:00"}, "2024-01-01", "2024-12-31")
    True
    >>> within_date_range({"date": "2023-12-31"}, date_from="2024-01-01")
    False
    >>> within_date_range({"date": "1998"}, date_to="2000")
    True
    >>> within_date_range({}, date_from="2024-01-01")
    False
    >>> within_date_range({})
    True
    """
    if not date_from and not date_to:
        return True

    value = str(record.get("date") or "")[:10]

    if not value:
        return False

    if date_from and value < date_from[:10]:
        return False

    if date_to and value[: len(date_to[:10])] > date_to[:10]:
        return False

    return True


def reciprocal_rank_fusion(
    result_lists: list[list[SearchResult]],
    k: int = DEFAULT_RRF_K,
    top_k: int | None = None,
) -> list[SearchResult]:
    r"""
    PURPOSE
    -------
    Merge ranked lists with Reciprocal Rank Fusion: each passage scores
    sum(1 / (k + rank)) over the lists it appears in. Only ranks are used,
    never raw scores, which is what makes cosine similarities and BM25
    scores (different scales, BM25 can be negative) safe to combine.

    PARAMETERS
    ----------
    result_lists   Ranked SearchResult lists, best first in each.
    k              RRF damping constant; 60 is the standard value.
    top_k          Keep at most this many fused results, or None for all.

    RETURNS
    -------
    New SearchResult objects, highest fused score first, re-ranked from 1.
    Ties keep the order in which passages were first seen.

    EXAMPLES
    --------
    >>> def hit(rank, chunk_id):
    ...     return SearchResult(rank, 0.0, rank - 1, chunk_id, {"dataset": "rag", "chunk_id": chunk_id})
    >>> vector_hits = [hit(1, "A"), hit(2, "B"), hit(3, "C")]
    >>> keyword_hits = [hit(1, "B"), hit(2, "D")]
    >>> fused = reciprocal_rank_fusion([vector_hits, keyword_hits])
    >>> [(r.rank, r.text) for r in fused]
    [(1, 'B'), (2, 'A'), (3, 'D'), (4, 'C')]
    >>> round(fused[0].score, 5) == round(1 / 62 + 1 / 61, 5)
    True
    >>> [r.text for r in reciprocal_rank_fusion([vector_hits, keyword_hits], top_k=2)]
    ['B', 'A']
    >>> reciprocal_rank_fusion([[], []])
    []
    """
    fused_scores: dict[tuple[str, str], float] = {}
    first_seen: dict[tuple[str, str], SearchResult] = {}

    for results in result_lists:
        for position, result in enumerate(results, start=1):
            key = result_key(result)
            fused_scores[key] = fused_scores.get(key, 0.0) + 1.0 / (k + position)
            first_seen.setdefault(key, result)

    ordered = sorted(fused_scores, key=lambda key: -fused_scores[key])

    if top_k is not None:
        ordered = ordered[:top_k]

    return [
        SearchResult(
            rank=rank,
            score=fused_scores[key],
            row=first_seen[key].row,
            text=first_seen[key].text,
            record=first_seen[key].record,
        )
        for rank, key in enumerate(ordered, start=1)
    ]


def hybrid_retrieve(
    base: VectorBase,
    keyword_index: Any,
    query: str,
    encode: QueryEncoder,
    top_k: int = DEFAULT_TOP_K,
    filters: dict[str, Any] | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    candidate_pool: int = DEFAULT_CANDIDATE_POOL,
    rrf_k: int = DEFAULT_RRF_K,
    minimum_score: float = DEFAULT_MINIMUM_SCORE,
    reranker_model: Any = None,
) -> list[SearchResult]:
    r"""
    PURPOSE
    -------
    Hybrid search: take up to `candidate_pool` hits from the vector base
    and from the BM25 keyword index, apply field and date filters, merge
    them with Reciprocal Rank Fusion, and optionally re-score the merged
    pool with a cross-encoder before keeping the best `top_k`.

    PARAMETERS
    ----------
    base             The loaded vector base.
    keyword_index    A keyword_index.KeywordIndex (anything with
                      `.search(query, top_k, filters)` works).
    query            The question text; must not be empty.
    encode           Callable turning query text into an embedding vector.
    top_k            Number of results to return.
    filters          Field filters such as {"language": "yor_Latn",
                      "source_type": "book", "tool": "ask"}; a list value
                      means "any of".
    date_from        Earliest allowed record "date" (ISO), or None.
    date_to          Latest allowed record "date" (ISO), or None.
    candidate_pool   Hits taken from each index before fusion (30-50 is
                      the useful range for reranking).
    rrf_k            RRF damping constant.
    minimum_score    Drop vector hits scoring below this cosine similarity.
    reranker_model   A cross-encoder with `.predict(pairs)` (see
                      reranker.load_reranker), or None to skip reranking.

    RETURNS
    -------
    Up to top_k SearchResult objects, best first. Scores are RRF scores, or
    cross-encoder scores when a reranker model was given.

    RAISES
    ------
    ValueError   When the query is empty or whitespace-only.

    EXAMPLES
    --------
    >>> import keyword_index as ki
    >>> vectors = normalize_rows(np.array([[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]]))
    >>> records = [
    ...     {"row": 0, "dataset": "rag", "chunk_id": "C0", "language": "eng_Latn",
    ...      "date": "2020-01-01", "text": "Momentum is mass times velocity."},
    ...     {"row": 1, "dataset": "rag", "chunk_id": "C1", "language": "yor_Latn",
    ...      "date": "2024-06-01", "text": "Agbara ni ise ti a le se."},
    ...     {"row": 2, "dataset": "rag", "chunk_id": "C2", "language": "eng_Latn",
    ...      "date": "2024-03-01", "text": "Kinetic energy depends on velocity squared."},
    ... ]
    >>> base = VectorBase(vectors, records, {"embedding_model": "stub"})
    >>> index = ki.KeywordIndex(records, [ki.tokenize(r["text"]) for r in records])
    >>> encode = lambda text: np.array([1.0, 0.0], dtype="float32")
    >>> [r.record["chunk_id"] for r in hybrid_retrieve(base, index, "velocity", encode)]
    ['C0', 'C2']

    The vector side alone never finds C1 (cosine 0), but BM25 matches the
    exact Yoruba word, so fusion brings it in:

    >>> [r.record["chunk_id"] for r in retrieve(base, "agbara", encode, top_k=3)]
    ['C0', 'C2']
    >>> [r.record["chunk_id"] for r in hybrid_retrieve(base, index, "agbara", encode)]
    ['C0', 'C1', 'C2']
    >>> [r.record["chunk_id"] for r in hybrid_retrieve(base, index, "agbara", encode,
    ...                                                filters={"language": "yor_Latn"})]
    ['C1']
    >>> [r.record["chunk_id"] for r in hybrid_retrieve(base, index, "velocity", encode,
    ...                                                date_from="2024-01-01")]
    ['C2']
    >>> class FakeReranker:
    ...     def predict(self, pairs):
    ...         return [1.0 if "Kinetic" in text else 0.0 for _, text in pairs]
    >>> hybrid_retrieve(base, index, "velocity", encode, top_k=1,
    ...                 reranker_model=FakeReranker())[0].record["chunk_id"]
    'C2'
    >>> hybrid_retrieve(base, index, " ", encode)
    Traceback (most recent call last):
        ...
    ValueError: The query is empty.
    """
    if not query or not query.strip():
        raise ValueError("The query is empty.")

    query = query.strip()

    vector_hits = retrieve(
        base=base,
        query=query,
        encode=encode,
        top_k=candidate_pool,
        filters=filters,
        relevance_weight=1.0,
        minimum_score=minimum_score,
    )
    # A zero BM25 score means the passage shares no word with the query;
    # it must not earn RRF credit just for appearing in the list.
    keyword_hits = [
        hit
        for hit in keyword_index.search(query, top_k=candidate_pool, filters=filters)
        if hit.score > 0
    ]

    if date_from or date_to:
        vector_hits = [
            hit for hit in vector_hits
            if within_date_range(hit.record, date_from, date_to)
        ]
        keyword_hits = [
            hit for hit in keyword_hits
            if within_date_range(hit.record, date_from, date_to)
        ]

    fused = reciprocal_rank_fusion(
        [vector_hits, keyword_hits], k=rrf_k, top_k=candidate_pool
    )

    if reranker_model is None:
        return fused[:top_k]

    from reranker import rerank

    reranked = rerank(query, fused, model=reranker_model, top_k=top_k)

    return [
        SearchResult(
            rank=result.rank,
            score=result.score,
            row=fused[result.original_rank - 1].row,
            text=result.text,
            record=result.record,
        )
        for result in reranked
    ]


def describe_source(record: dict[str, Any]) -> str:
    r"""
    PURPOSE
    -------
    Build a citation line for a record.

    PARAMETERS
    ----------
    record   The matched record, containing source metadata fields.

    RETURNS
    -------
    A human-readable citation string, or 'unknown source' when no source
    fields are present.

    EXAMPLES
    --------
    >>> describe_source({"source_file": "physics.pdf", "chapter_title": "Ch 4",
    ...                  "source_page_start": 61, "source_page_end": 78})
    'physics.pdf | Ch 4 | pp. 61-78'
    >>> describe_source({"source_file": "notes.pdf", "source_page_start": 12})
    'notes.pdf | p. 12'
    >>> describe_source({})
    'unknown source'
    """
    parts: list[str] = []

    if record.get("source_file"):
        parts.append(str(record["source_file"]))

    if record.get("chapter_title"):
        parts.append(str(record["chapter_title"]))

    start = record.get("source_page_start")
    end = record.get("source_page_end")

    if start is not None and end is not None and end != start:
        parts.append(f"pp. {start}-{end}")
    elif start is not None:
        parts.append(f"p. {start}")

    return " | ".join(parts) if parts else "unknown source"


def shorten(text: str, limit: int = MAXIMUM_CONTEXT_CHARACTERS) -> str:
    r"""
    PURPOSE
    -------
    Trim text to `limit` characters at a word boundary.

    PARAMETERS
    ----------
    text    The text to trim; internal whitespace is collapsed first.
    limit   Maximum number of characters to keep.

    RETURNS
    -------
    The trimmed text, with an ellipsis appended when it was cut.

    EXAMPLES
    --------
    >>> shorten("short", 40)
    'short'
    >>> shorten("one two three four five", 12)
    'one two …'
    """
    text = " ".join(str(text).split())

    if len(text) <= limit:
        return text

    return f"{text[:limit].rsplit(' ', 1)[0]} …"


def format_results(
    query: str,
    results: list[SearchResult],
    show_text: bool = True,
) -> str:
    r"""
    PURPOSE
    -------
    Render search results for reading in a terminal.

    PARAMETERS
    ----------
    query        The original query text, echoed at the top of the output.
    results      The SearchResult objects to render.
    show_text    Whether to include a shortened excerpt of each passage.

    RETURNS
    -------
    The formatted, human-readable results block as a single string.

    EXAMPLES
    --------
    >>> vectors = normalize_rows(np.array([[1.0, 0.0]]))
    >>> records = [{"row": 0, "dataset": "rag", "topic": "Heat",
    ...             "source_file": "physics.pdf", "source_page_start": 9,
    ...             "text": "Heat flows from hot to cold."}]
    >>> base = VectorBase(vectors, records, {"embedding_model": "stub"})
    >>> results = base.search(np.array([1.0, 0.0]), top_k=1)
    >>> print(format_results("how does heat move", results))
    Query: how does heat move
    Results: 1
    <BLANKLINE>
    1. [1.000] Heat
       physics.pdf | p. 9
       Heat flows from hot to cold.
    <BLANKLINE>
    >>> print(format_results("nothing", []))
    Query: nothing
    No matching passages were found. Try a shorter query, a lower --minimum-score, or fewer filters.
    """
    lines = [f"Query: {query}"]

    if not results:
        lines.append(
            "No matching passages were found. Try a shorter query, "
            "a lower --minimum-score, or fewer filters."
        )
        return "\n".join(lines)

    lines.append(f"Results: {len(results)}")
    lines.append("")

    for result in results:
        heading = str(
            result.record.get("topic")
            or result.record.get("concept")
            or result.record.get("chapter_title")
            or result.record.get("dataset")
            or "untitled"
        )

        lines.append(f"{result.rank}. [{result.score:.3f}] {heading}")
        lines.append(f"   {describe_source(result.record)}")

        if show_text:
            text = result.text or "(no passage text joined into this base)"
            lines.append(f"   {shorten(text, 300)}")

        lines.append("")

    return "\n".join(lines)


def build_context(
    results: list[SearchResult],
    character_budget: int = 4000,
) -> str:
    r"""
    PURPOSE
    -------
    Concatenate retrieved passages into a citation-labelled context block.
    Passages are added in rank order until the budget is spent, so the block
    can be pasted straight into a prompt.

    PARAMETERS
    ----------
    results            The SearchResult objects to include, in rank order.
    character_budget   Maximum total characters to include across passages.

    RETURNS
    -------
    The assembled context block as a single string, or '' when there are no
    passages with text.

    EXAMPLES
    --------
    >>> vectors = normalize_rows(np.array([[1.0, 0.0], [0.9, 0.1]]))
    >>> records = [
    ...     {"row": 0, "source_file": "a.pdf", "source_page_start": 1, "text": "First passage."},
    ...     {"row": 1, "source_file": "b.pdf", "source_page_start": 2, "text": "Second passage."},
    ... ]
    >>> base = VectorBase(vectors, records, {"embedding_model": "stub"})
    >>> print(build_context(base.search(np.array([1.0, 0.0]), top_k=2)))
    [1] a.pdf | p. 1
    First passage.
    <BLANKLINE>
    [2] b.pdf | p. 2
    Second passage.
    >>> print(build_context(base.search(np.array([1.0, 0.0]), top_k=2), character_budget=30))
    [1] a.pdf | p. 1
    First passage.
    >>> build_context([])
    ''
    """
    blocks: list[str] = []
    used = 0

    for result in results:
        if not result.text:
            continue

        block = f"[{result.rank}] {describe_source(result.record)}\n{result.text}"

        if blocks and used + len(block) > character_budget:
            break

        blocks.append(block)
        used += len(block)

    return "\n\n".join(blocks)


def results_as_json(query: str, results: list[SearchResult]) -> str:
    r"""
    PURPOSE
    -------
    Serialise results as JSON for another program to consume.

    PARAMETERS
    ----------
    query     The original query text, included in the payload.
    results   The SearchResult objects to serialise.

    RETURNS
    -------
    A JSON string with the query, result count, source label, and each
    result's rank, score, row, text, citation, and record.

    EXAMPLES
    --------
    >>> vectors = normalize_rows(np.array([[1.0, 0.0]]))
    >>> records = [{"row": 0, "dataset": "rag", "topic": "Heat", "text": "Heat flows."}]
    >>> base = VectorBase(vectors, records, {"embedding_model": "stub"})
    >>> payload = json.loads(results_as_json("heat", base.search(np.array([1.0, 0.0]))))
    >>> payload["query"], payload["result_count"], payload["source"] == SOURCE_LABEL
    ('heat', 1, True)
    >>> payload["results"][0]["rank"], round(payload["results"][0]["score"], 3)
    (1, 1.0)
    >>> payload["results"][0]["record"]["topic"]
    'Heat'
    """
    return json.dumps(
        {
            "query": query,
            "result_count": len(results),
            "source": SOURCE_LABEL,
            "results": [
                {
                    "rank": result.rank,
                    "score": round(result.score, 6),
                    "row": result.row,
                    "text": result.text,
                    "citation": describe_source(result.record),
                    "record": result.record,
                }
                for result in results
            ],
        },
        ensure_ascii=False,
        indent=2,
    )


def run_query(
    base: VectorBase,
    query: str,
    encode: QueryEncoder,
    top_k: int = DEFAULT_TOP_K,
    filters: dict[str, Any] | None = None,
    relevance_weight: float = DEFAULT_RELEVANCE_WEIGHT,
    minimum_score: float = DEFAULT_MINIMUM_SCORE,
    output_format: str = "text",
    keyword_index: Any = None,
    date_from: str | None = None,
    date_to: str | None = None,
    candidate_pool: int = DEFAULT_CANDIDATE_POOL,
    reranker_model: Any = None,
) -> str:
    r"""
    PURPOSE
    -------
    Retrieve for one query and render it in the requested format. With a
    keyword index this runs hybrid_retrieve; otherwise plain vector search.

    PARAMETERS
    ----------
    base               The loaded vector base to search.
    query              The question text to search for.
    encode             Callable turning query text into an embedding vector.
    top_k              Maximum number of results to return.
    filters            Optional field filters, as produced by
                        parse_filter_arguments.
    relevance_weight   How strongly to favour pure relevance over diversity
                        (vector-only search).
    minimum_score      Drop passages scoring below this cosine similarity.
    output_format      One of 'text', 'json', or 'context'.
    keyword_index      A loaded KeywordIndex to enable hybrid search, or None.
    date_from          Earliest record date (hybrid search only), or None.
    date_to            Latest record date (hybrid search only), or None.
    candidate_pool     Hits per index before fusion (hybrid search only).
    reranker_model     Cross-encoder for reranking (hybrid search only), or
                        None.

    RETURNS
    -------
    The rendered results as a string, in the requested format.

    EXAMPLES
    --------
    >>> vectors = normalize_rows(np.array([[1.0, 0.0]]))
    >>> records = [{"row": 0, "dataset": "rag", "topic": "Heat",
    ...             "source_file": "p.pdf", "text": "Heat flows."}]
    >>> base = VectorBase(vectors, records, {"embedding_model": "stub"})
    >>> encode = lambda text: np.array([1.0, 0.0], dtype="float32")
    >>> print(run_query(base, "heat", encode, output_format="context"))
    [1] p.pdf
    Heat flows.
    >>> json.loads(run_query(base, "heat", encode, output_format="json"))["result_count"]
    1
    >>> "Results: 1" in run_query(base, "heat", encode)
    True
    >>> import keyword_index as ki
    >>> index = ki.KeywordIndex(records, [ki.tokenize(r["text"]) for r in records])
    >>> json.loads(run_query(base, "heat", encode, output_format="json",
    ...                      keyword_index=index))["result_count"]
    1
    """
    if keyword_index is not None:
        results = hybrid_retrieve(
            base=base,
            keyword_index=keyword_index,
            query=query,
            encode=encode,
            top_k=top_k,
            filters=filters,
            date_from=date_from,
            date_to=date_to,
            candidate_pool=candidate_pool,
            minimum_score=minimum_score,
            reranker_model=reranker_model,
        )
    else:
        results = retrieve(
            base=base,
            query=query,
            encode=encode,
            top_k=top_k,
            filters=filters,
            relevance_weight=relevance_weight,
            minimum_score=minimum_score,
        )

    if output_format == "json":
        return results_as_json(query, results)

    if output_format == "context":
        return build_context(results)

    return format_results(query, results)


def interactive_loop(
    base: VectorBase,
    encode: QueryEncoder,
    top_k: int = DEFAULT_TOP_K,
    filters: dict[str, Any] | None = None,
    input_function: Callable[[str], str] = input,
    output_function: Callable[[str], None] = print,
    search_options: dict[str, Any] | None = None,
) -> int:
    r"""
    PURPOSE
    -------
    Read queries until the user stops, printing results for each one. A
    blank line is skipped, and 'quit', 'exit', or ':q' (case-insensitive)
    ends the loop.

    PARAMETERS
    ----------
    base              The loaded vector base to search.
    encode            Callable turning query text into an embedding vector.
    top_k             Maximum number of results to return per query.
    filters           Optional field filters, as produced by
                       parse_filter_arguments.
    input_function    Callable used to prompt for and read each query.
    output_function   Callable used to print each query's rendered results.
    search_options    Extra keyword arguments for run_query, such as
                       keyword_index, date_from or reranker_model.

    RETURNS
    -------
    The number of queries that were answered.

    EXAMPLES
    --------
    >>> vectors = normalize_rows(np.array([[1.0, 0.0]]))
    >>> records = [{"row": 0, "dataset": "rag", "topic": "Heat", "text": "Heat flows."}]
    >>> base = VectorBase(vectors, records, {"embedding_model": "stub"})
    >>> encode = lambda text: np.array([1.0, 0.0], dtype="float32")
    >>> typed = iter(["heat", "", "quit"])
    >>> printed = []
    >>> interactive_loop(base, encode, input_function=lambda prompt: next(typed),
    ...                  output_function=printed.append)
    1
    >>> "Results: 1" in printed[0]
    True
    """
    answered = 0

    while True:
        try:
            query = input_function("query> ")
        except (EOFError, KeyboardInterrupt, StopIteration):
            return answered

        query = query.strip()

        if not query:
            continue

        if query.casefold() in {"quit", "exit", ":q"}:
            return answered

        output_function(
            run_query(
                base,
                query,
                encode,
                top_k=top_k,
                filters=filters,
                **(search_options or {}),
            )
        )

        answered += 1


def parse_arguments(argument_list: list[str] | None = None) -> argparse.Namespace:
    r"""
    PURPOSE
    -------
    Parse the command line.

    PARAMETERS
    ----------
    argument_list   The arguments to parse, or None to read from sys.argv.

    RETURNS
    -------
    The parsed argparse.Namespace.

    EXAMPLES
    --------
    >>> options = parse_arguments(["--query", "how is momentum defined"])
    >>> options.query, options.top_k, options.format
    ('how is momentum defined', 5, 'text')
    >>> options = parse_arguments(["--query", "x", "--top-k", "8", "--json",
    ...                            "--filter", "topic=Heat", "--dataset", "rag"])
    >>> options.top_k, options.format, options.filter, options.dataset
    (8, 'json', ['topic=Heat'], 'rag')
    >>> parse_arguments(["--interactive"]).interactive
    True
    >>> parse_arguments(["--query", "x", "--context"]).format
    'context'
    >>> options = parse_arguments(["--query", "x", "--hybrid", "--rerank",
    ...                            "--language", "yor_Latn", "--date-from", "2024-01-01"])
    >>> options.hybrid, options.rerank, options.language, options.date_from
    (True, True, 'yor_Latn', '2024-01-01')
    >>> parse_arguments(["--query", "x"]).hybrid
    False
    """
    parser = argparse.ArgumentParser(
        description="Search the vector base and return cited passages."
    )

    parser.add_argument("--query", help="the question to search for")

    parser.add_argument(
        "--interactive",
        action="store_true",
        help="ask for queries repeatedly instead of taking one",
    )

    parser.add_argument(
        "--base",
        type=Path,
        default=VECTOR_BASE_FOLDER,
        help="vector base folder",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="how many passages to return",
    )

    parser.add_argument(
        "--dataset",
        help="restrict to one dataset, for example rag or llm",
    )

    parser.add_argument(
        "--filter",
        action="append",
        help="field=value filter; repeat for more (repeating a field means any of)",
    )

    parser.add_argument(
        "--relevance-weight",
        type=float,
        default=DEFAULT_RELEVANCE_WEIGHT,
        help="1.0 is pure relevance; lower values return more varied passages",
    )

    parser.add_argument(
        "--minimum-score",
        type=float,
        default=DEFAULT_MINIMUM_SCORE,
        help="drop passages scoring below this cosine similarity",
    )

    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_NAME,
        help="model used to embed the query",
    )

    parser.add_argument(
        "--allow-model-mismatch",
        action="store_true",
        help="search even when the base was built with a different model",
    )

    parser.add_argument(
        "--hybrid",
        action="store_true",
        help="combine vector search with the BM25 keyword index (RRF)",
    )

    parser.add_argument(
        "--keyword-index",
        type=Path,
        default=KEYWORD_INDEX_FOLDER,
        help="keyword index folder (used with --hybrid)",
    )

    parser.add_argument(
        "--rerank",
        action="store_true",
        help="re-score the hybrid candidates with bge-reranker-v2-m3 (implies --hybrid)",
    )

    parser.add_argument(
        "--candidate-pool",
        type=int,
        default=DEFAULT_CANDIDATE_POOL,
        help="hits taken from each index before fusion and reranking",
    )

    parser.add_argument("--language", help="filter by language code, e.g. yor_Latn")
    parser.add_argument("--source-type", help="filter by source type, e.g. book or web")
    parser.add_argument("--tool", help="filter by tool, e.g. ask or examflow")
    parser.add_argument("--date-from", help="earliest record date, YYYY-MM-DD (hybrid)")
    parser.add_argument("--date-to", help="latest record date, YYYY-MM-DD (hybrid)")

    output_group = parser.add_mutually_exclusive_group()

    output_group.add_argument(
        "--json",
        dest="format",
        action="store_const",
        const="json",
        help="print results as JSON",
    )

    output_group.add_argument(
        "--context",
        dest="format",
        action="store_const",
        const="context",
        help="print a citation-labelled context block for a prompt",
    )

    parser.set_defaults(format="text")

    return parser.parse_args(argument_list)


def main() -> None:
    """
    PURPOSE
    -------
    Search the vector base using the options given on the command line.
    """
    options = parse_arguments()

    if not options.query and not options.interactive:
        raise SystemExit("Provide --query \"...\" or --interactive.")

    base = VectorBase.load(options.base)

    if len(base) == 0:
        raise SystemExit(
            f"The vector base in {options.base} is empty. Run vector_base.py first."
        )

    encode, model_name = load_query_encoder(options.model)

    warning = check_model_match(
        base, model_name, allow_mismatch=options.allow_model_mismatch
    )

    if warning:
        print(f"Warning: {warning}", file=sys.stderr)

    filters = parse_filter_arguments(options.filter)

    if options.dataset:
        filters["dataset"] = options.dataset

    for field_name in ("language", "source_type", "tool"):
        if getattr(options, field_name):
            filters[field_name] = getattr(options, field_name)

    search_options: dict[str, Any] = {}

    if options.hybrid or options.rerank:
        from keyword_index import KeywordIndex

        search_options = {
            "keyword_index": KeywordIndex.load(options.keyword_index),
            "date_from": options.date_from,
            "date_to": options.date_to,
            "candidate_pool": options.candidate_pool,
        }

        if options.rerank:
            from reranker import load_reranker

            search_options["reranker_model"] = load_reranker()
    elif options.date_from or options.date_to:
        raise SystemExit("--date-from/--date-to need --hybrid.")

    if options.interactive:
        print(
            f"{len(base)} passages loaded. Type a question, or 'quit' to stop."
        )
        interactive_loop(
            base,
            encode,
            top_k=options.top_k,
            filters=filters,
            search_options=search_options,
        )
        return

    print(
        run_query(
            base=base,
            query=options.query,
            encode=encode,
            top_k=options.top_k,
            filters=filters,
            relevance_weight=options.relevance_weight,
            minimum_score=options.minimum_score,
            output_format=options.format,
            **search_options,
        )
    )


if __name__ == "__main__":
    main()
