#!/usr/bin/env python3

"""
SoakinGarri Vector Base Program

PURPOSE
-------
Assembles the outputs of the embedding program into one searchable vector base:
a single matrix of unit-length vectors, one aligned record per row, and a
manifest describing how the base was built.

The base is a plain folder of ordinary files. Nothing here needs a database
server, and the folder can be copied, versioned or shipped as-is.

TEXT JOIN
---------
The embedding program stores identifiers and metadata, not passages. A vector
base that cannot return the passage it matched is not much use, so this program
joins the text back in from the chunked data folder, keyed on chunk_id and
record_id. When that folder is unavailable the base is still built, with empty
text fields and a warning recorded in the manifest.

INPUT FOLDERS
-------------
<repo>/data/ask/embedding_data      vectors and metadata
<repo>/data/ask/chunked_data        passages, for the text join

OUTPUT FOLDER
-------------
<repo>/data/ask/vector_base

OUTPUT FILES
------------
vectors.npy                 float32 matrix, one unit-length row per record
records.jsonl               one record per row, in row order
vector_base_manifest.json

INSTALL
-------
python3 -m pip install numpy

RUN
---
python3 vector_base.py
python3 vector_base.py --datasets rag
python3 vector_base.py --no-deduplicate
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import numpy as np

sys.path.insert(
    0, str(Path(__file__).resolve().parent.parent.parent.parent / "source")
)

from common import DATA_ROOT  # noqa: E402

EMBEDDING_FOLDER = DATA_ROOT / "embedding_data"

CHUNKED_FOLDER = DATA_ROOT / "chunked_data"

OUTPUT_FOLDER = DATA_ROOT / "vector_base"

VECTOR_FILE_NAME = "vectors.npy"
RECORD_FILE_NAME = "records.jsonl"
MANIFEST_FILE_NAME = "vector_base_manifest.json"

SOURCE_LABEL = "Borrowed content and open source"

VECTOR_DATA_TYPE = np.float32

DATASET_FILES = {
    "llm": ("llm_embeddings.npy", "llm_embedding_metadata.jsonl"),
    "rag": ("rag_embeddings.npy", "rag_embedding_metadata.jsonl"),
}

DATASET_NAMES = tuple(DATASET_FILES)

TEXT_SOURCE_FILES = (
    "rag_data.jsonl",
    "llm_training_data.jsonl",
)

SEARCH_BLOCK_SIZE = 50_000

DEFAULT_TOP_K = 5
DEFAULT_MMR_LAMBDA = 1.0


@dataclass(slots=True)
class SearchResult:
    """
    PURPOSE
    -------
    One ranked hit returned from ``VectorBase.search``.

    ATTRIBUTES
    ----------
    rank     1-based position of this result among the returned results.
    score    Cosine similarity of this result's row to the query.
    row      Row number of this result in the vector base.
    text     Passage text for this result.
    record   Full vector-base record for this result.
    """

    rank: int
    score: float
    row: int
    text: str
    record: dict[str, Any] = field(repr=False)


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    r"""
    PURPOSE
    -------
    Yield each JSON object in a JSONL file, skipping blank lines.

    PARAMETERS
    ----------
    path   Path to the JSONL file to read.

    RETURNS
    -------
    An iterator of the parsed JSON objects, in file order.

    RAISES
    ------
    ValueError   A non-blank line is not valid JSON.

    EXAMPLES
    --------
    >>> import tempfile
    >>> with tempfile.TemporaryDirectory() as folder:
    ...     path = Path(folder) / "a.jsonl"
    ...     _ = path.write_text('{"a": 1}\n\n{"b": 2}\n', encoding="utf-8")
    ...     list(read_jsonl(path))
    [{'a': 1}, {'b': 2}]
    >>> with tempfile.TemporaryDirectory() as folder:
    ...     path = Path(folder) / "bad.jsonl"
    ...     _ = path.write_text('{"a": 1}\noops\n', encoding="utf-8")
    ...     list(read_jsonl(path))
    Traceback (most recent call last):
        ...
    ValueError: Invalid JSON in bad.jsonl on line 2: Expecting value: line 1 column 1 (char 0)
    """
    with path.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON in {path.name} on line "
                    f"{line_number}: {error}"
                ) from error

            if isinstance(record, dict):
                yield record


def atomic_write_text(destination: Path, text: str) -> None:
    r"""
    PURPOSE
    -------
    Write text through a temporary file, then rename it over the target, so
    a reader never sees a partially written file.

    PARAMETERS
    ----------
    destination   Path the text should end up at.
    text          Content to write, encoded as UTF-8.

    EXAMPLES
    --------
    >>> import tempfile
    >>> with tempfile.TemporaryDirectory() as folder:
    ...     path = Path(folder) / "out" / "file.txt"
    ...     atomic_write_text(path, "café\n")
    ...     (path.read_text("utf-8"), [p.name for p in path.parent.iterdir()])
    ('café\n', ['file.txt'])
    """
    destination.parent.mkdir(parents=True, exist_ok=True)

    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )

    temporary_path = Path(temporary_name)

    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as output_file:
            output_file.write(text)
            output_file.flush()
            os.fsync(output_file.fileno())

        temporary_path.replace(destination)

    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def atomic_write_numpy(destination: Path, array: np.ndarray) -> None:
    r"""
    PURPOSE
    -------
    Write a .npy file through a temporary file, then rename it over the
    target, so a reader never sees a partially written file.

    PARAMETERS
    ----------
    destination   Path the array should end up at.
    array         Array to save in NumPy's .npy format.

    EXAMPLES
    --------
    >>> import tempfile
    >>> with tempfile.TemporaryDirectory() as folder:
    ...     path = Path(folder) / "v.npy"
    ...     atomic_write_numpy(path, np.eye(2, dtype="float32"))
    ...     (np.load(path).shape, [p.name for p in path.parent.iterdir()])
    ((2, 2), ['v.npy'])
    """
    destination.parent.mkdir(parents=True, exist_ok=True)

    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".npy",
    )

    os.close(file_descriptor)
    temporary_path = Path(temporary_name)

    try:
        np.save(temporary_path, array, allow_pickle=False)
        temporary_path.replace(destination)

    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    r"""
    PURPOSE
    -------
    Scale every row to unit length, leaving all-zero rows alone. Cosine
    similarity between unit vectors is a plain dot product, so normalizing
    once at build time makes every later search cheaper.

    PARAMETERS
    ----------
    matrix   Array of row vectors, 1-D or 2-D.

    RETURNS
    -------
    A float32 matrix of the same shape, with each nonzero row scaled to
    unit length.

    EXAMPLES
    --------
    >>> normalize_rows(np.array([[3.0, 4.0], [0.0, 0.0]], dtype="float32"))
    array([[0.6, 0.8],
           [0. , 0. ]], dtype=float32)
    >>> float(np.linalg.norm(normalize_rows(np.array([[1.0, 1.0, 1.0, 1.0]]))[0]))
    1.0
    """
    matrix = np.asarray(matrix, dtype=VECTOR_DATA_TYPE)

    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)

    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0

    return (matrix / norms).astype(VECTOR_DATA_TYPE)


JOIN_FIELDS = (
    "topic",
    "concept",
    "formula",
    "chapter_title",
    "source_file",
    "source_page_start",
    "source_page_end",
    "language",
    "source_type",
    "tool",
    "date",
)


def build_join_lookup(chunked_folder: Path) -> dict[str, dict[str, Any]]:
    r"""
    PURPOSE
    -------
    Map chunk and record identifiers to their passage text and citation
    fields. The embedding metadata keeps identifiers but drops the chapter
    and page for LLM records, so those are recovered here — a result you
    cannot cite is not much of a result.

    PARAMETERS
    ----------
    chunked_folder   Folder holding the chunked passage files.

    RETURNS
    -------
    A dict mapping each chunk/record/document id to a dict of its text and
    citation fields. Empty if the folder does not exist.

    EXAMPLES
    --------
    >>> import tempfile
    >>> with tempfile.TemporaryDirectory() as folder:
    ...     root = Path(folder)
    ...     _ = (root / "rag_data.jsonl").write_text(
    ...         '{"chunk_id": "C1", "text": "A chunk of prose.",'
    ...         ' "source_metadata": {"chapter_title": "Ch 1", "source_page_start": 4}}\n',
    ...         encoding="utf-8")
    ...     _ = (root / "llm_training_data.jsonl").write_text(
    ...         '{"record_id": "CE-1", "type": "concept_explanation",'
    ...         ' "concept": "Heat", "explanation": "Energy in transit.",'
    ...         ' "chapter_title": "Ch 2", "source_file": "physics.pdf"}\n',
    ...         encoding="utf-8")
    ...     lookup = build_join_lookup(root)
    >>> lookup["C1"]["text"]
    'A chunk of prose.'
    >>> lookup["C1"]["chapter_title"], lookup["C1"]["source_page_start"]
    ('Ch 1', 4)
    >>> lookup["CE-1"]["text"]
    'Concept: Heat\nExplanation: Energy in transit.'
    >>> lookup["CE-1"]["chapter_title"], lookup["CE-1"]["source_file"]
    ('Ch 2', 'physics.pdf')
    >>> build_join_lookup(Path("/no/such/folder"))
    {}
    """
    lookup: dict[str, dict[str, Any]] = {}

    if not chunked_folder.is_dir():
        return lookup

    for file_name in TEXT_SOURCE_FILES:
        path = chunked_folder / file_name

        if not path.exists():
            continue

        for record in read_jsonl(path):
            text = record_text(record)

            if not text:
                continue

            source_metadata = record.get("source_metadata")

            if not isinstance(source_metadata, dict):
                source_metadata = {}

            entry: dict[str, Any] = {"text": text}

            for field_name in JOIN_FIELDS:
                value = record.get(field_name)

                if value is None:
                    value = source_metadata.get(field_name)

                if value is not None:
                    entry[field_name] = value

            for key_name in ("chunk_id", "record_id", "document_id"):
                key = record.get(key_name)

                if key and str(key) not in lookup:
                    lookup[str(key)] = entry

    return lookup


def record_text(record: dict[str, Any]) -> str:
    r"""
    PURPOSE
    -------
    Reconstruct the readable text of a chunked record.

    PARAMETERS
    ----------
    record   A chunked record, either with a plain "text" field or with
             typed fields (question/reasoning/answer, concept/explanation,
             formula/derivation/example).

    RETURNS
    -------
    The reconstructed passage text, or "" if none could be assembled.

    EXAMPLES
    --------
    >>> record_text({"text": "The chunk body."})
    'The chunk body.'
    >>> record_text({"type": "question_reasoning_answer", "question": "What is X?",
    ...              "reasoning": "Because.", "answer": "X is Y."})
    'Question: What is X?\nReasoning: Because.\nAnswer: X is Y.'
    >>> record_text({"type": "formula_derivation_example", "formula": "F = ma",
    ...              "derivation": "From Newton.", "example": "A cart."})
    'Formula: F = ma\nDerivation: From Newton.\nExample: A cart.'
    >>> record_text({})
    ''
    """
    text = str(record.get("text") or "").strip()

    if text:
        return text

    record_type = str(
        record.get("type") or record.get("training_type") or ""
    ).strip()

    layouts = {
        "question_reasoning_answer": (
            ("Question", "question"),
            ("Reasoning", "reasoning"),
            ("Answer", "answer"),
        ),
        "concept_explanation": (
            ("Concept", "concept"),
            ("Explanation", "explanation"),
        ),
        "formula_derivation_example": (
            ("Formula", "formula"),
            ("Derivation", "derivation"),
            ("Example", "example"),
        ),
    }

    layout = layouts.get(record_type)

    if layout:
        lines = [
            f"{label}: {str(record.get(key) or '').strip()}"
            for label, key in layout
            if str(record.get(key) or "").strip()
        ]

        if lines:
            return "\n".join(lines)

    return str(
        record.get("content")
        or record.get("explanation")
        or record.get("answer")
        or ""
    ).strip()


def load_dataset(
    embedding_folder: Path,
    dataset_name: str,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    r"""
    PURPOSE
    -------
    Load one dataset's vectors and metadata, checking that they line up.

    PARAMETERS
    ----------
    embedding_folder   Folder holding the .npy vectors and metadata files.
    dataset_name       Which dataset to load ("llm" or "rag").

    RETURNS
    -------
    A tuple of (vectors, metadata records). Both are empty if the dataset's
    files are missing.

    RAISES
    ------
    ValueError   dataset_name is not a known dataset, the vector and
                 metadata counts disagree, or vector_index values are not
                 0..n-1 in order.

    EXAMPLES
    --------
    >>> import tempfile
    >>> with tempfile.TemporaryDirectory() as folder:
    ...     root = Path(folder)
    ...     np.save(root / "rag_embeddings.npy", np.eye(2, dtype="float32"))
    ...     _ = (root / "rag_embedding_metadata.jsonl").write_text(
    ...         '{"vector_index": 0, "chunk_id": "C1"}\n'
    ...         '{"vector_index": 1, "chunk_id": "C2"}\n', encoding="utf-8")
    ...     vectors, records = load_dataset(root, "rag")
    >>> vectors.shape, [r["chunk_id"] for r in records]
    ((2, 2), ['C1', 'C2'])
    >>> with tempfile.TemporaryDirectory() as folder:
    ...     root = Path(folder)
    ...     np.save(root / "rag_embeddings.npy", np.eye(3, dtype="float32"))
    ...     _ = (root / "rag_embedding_metadata.jsonl").write_text(
    ...         '{"vector_index": 0}\n', encoding="utf-8")
    ...     load_dataset(root, "rag")
    Traceback (most recent call last):
        ...
    ValueError: rag: 3 vectors but 1 metadata record(s).
    >>> load_dataset(Path("/tmp"), "nonsense")
    Traceback (most recent call last):
        ...
    ValueError: Unknown dataset: nonsense
    """
    if dataset_name not in DATASET_FILES:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    vector_name, metadata_name = DATASET_FILES[dataset_name]
    vector_path = embedding_folder / vector_name
    metadata_path = embedding_folder / metadata_name

    if not vector_path.exists() or not metadata_path.exists():
        return np.zeros((0, 0), dtype=VECTOR_DATA_TYPE), []

    vectors = np.load(vector_path, mmap_mode="r", allow_pickle=False)
    records = list(read_jsonl(metadata_path))

    if len(vectors) != len(records):
        raise ValueError(
            f"{dataset_name}: {len(vectors)} vectors but "
            f"{len(records)} metadata record(s)."
        )

    indexes = [record.get("vector_index") for record in records]

    if indexes != list(range(len(records))):
        raise ValueError(
            f"{dataset_name}: metadata vector_index values are not "
            "0..n-1 in order."
        )

    return np.asarray(vectors, dtype=VECTOR_DATA_TYPE), records


def build_record(
    metadata: dict[str, Any],
    dataset_name: str,
    row: int,
    join_lookup: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    r"""
    PURPOSE
    -------
    Turn one metadata line into a vector-base record, joining text and
    citation fields back in. Values in the embedding metadata win; the join
    only fills in what is missing.

    PARAMETERS
    ----------
    metadata       One embedding metadata record.
    dataset_name   Name of the dataset the record came from.
    row            Row number the record will occupy in the vector base.
    join_lookup    Lookup built by build_join_lookup, for text and citation.

    RETURNS
    -------
    The assembled vector-base record.

    EXAMPLES
    --------
    >>> metadata = {"vector_index": 0, "chunk_id": "C1", "document_id": "D1",
    ...             "topic": "Heat", "source_metadata": {"chapter_title": "Ch 1"}}
    >>> record = build_record(metadata, "rag", 0,
    ...                       {"C1": {"text": "A chunk of prose.", "source_page_start": 9}})
    >>> record["row"], record["dataset"], record["text"], record["topic"]
    (0, 'rag', 'A chunk of prose.', 'Heat')
    >>> record["chapter_title"], record["source_page_start"], record["has_text"]
    ('Ch 1', 9, True)
    >>> joined = build_record({"record_id": "CE-1"}, "llm", 1,
    ...                       {"CE-1": {"text": "Body.", "chapter_title": "Ch 2",
    ...                                 "topic": "Optics"}})
    >>> joined["chapter_title"], joined["topic"], joined["has_text"]
    ('Ch 2', 'Optics', True)
    >>> build_record({"chunk_id": "missing"}, "rag", 1, {})["has_text"]
    False
    >>> tagged = build_record({"chunk_id": "C1", "language": "yor_Latn",
    ...                        "source_metadata": {"source_type": "book"}}, "rag", 0, {})
    >>> tagged["language"], tagged["source_type"], tagged["tool"]
    ('yor_Latn', 'book', None)
    """
    source_metadata = metadata.get("source_metadata")

    if not isinstance(source_metadata, dict):
        source_metadata = {}

    joined: dict[str, Any] = {}

    for key_name in ("chunk_id", "record_id", "document_id"):
        key = metadata.get(key_name)

        if key and str(key) in join_lookup:
            joined = join_lookup[str(key)]
            break

    text = str(joined.get("text") or "")

    def pick(field_name: str) -> Any:
        value = metadata.get(field_name)

        if value is None:
            value = source_metadata.get(field_name)

        if value is None:
            value = joined.get(field_name)

        return value

    record = {
        "row": row,
        "dataset": dataset_name,
        "record_id": metadata.get("record_id"),
        "chunk_id": metadata.get("chunk_id"),
        "document_id": metadata.get("document_id"),
        "training_type": metadata.get("training_type"),
        "topic": pick("topic"),
        "concept": pick("concept"),
        "formula": pick("formula"),
        "chapter_title": pick("chapter_title"),
        "source_file": source_metadata.get("source_file")
        or joined.get("source_file")
        or metadata.get("source_file"),
        "source_page_start": pick("source_page_start"),
        "source_page_end": pick("source_page_end"),
        # Filter fields for hybrid retrieval; None until an upstream stage
        # (language_detection.py, source_registry.py, web_ingestion.py)
        # supplies them.
        "language": pick("language"),
        "source_type": pick("source_type"),
        "tool": pick("tool"),
        "date": pick("date"),
        "text_sha256": metadata.get("text_sha256"),
        "text": text,
        "has_text": bool(text),
        "source": SOURCE_LABEL,
    }

    return record


def deduplicate(
    vectors: np.ndarray,
    records: list[dict[str, Any]],
) -> tuple[np.ndarray, list[dict[str, Any]], int]:
    r"""
    PURPOSE
    -------
    Drop rows whose ``text_sha256`` was already seen, keeping the first.
    The same passage can reach the base from both datasets; keeping one
    copy stops a single passage from filling the whole result list.

    PARAMETERS
    ----------
    vectors   Row-aligned matrix of vectors.
    records   Row-aligned list of records, one per vector.

    RETURNS
    -------
    A tuple of (kept vectors, kept records with "row" renumbered, count of
    rows removed).

    EXAMPLES
    --------
    >>> vectors = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype="float32")
    >>> records = [{"text_sha256": "a"}, {"text_sha256": "b"}, {"text_sha256": "a"}]
    >>> kept_vectors, kept_records, removed = deduplicate(vectors, records)
    >>> kept_vectors.shape, [r["text_sha256"] for r in kept_records], removed
    ((2, 2), ['a', 'b'], 1)
    >>> [r["row"] for r in kept_records]
    [0, 1]
    >>> _, _, removed = deduplicate(vectors, [{}, {}, {}])   # no hashes: nothing dropped
    >>> removed
    0
    """
    keep: list[int] = []
    seen: set[str] = set()

    for index, record in enumerate(records):
        digest = record.get("text_sha256")

        if digest:
            if digest in seen:
                continue

            seen.add(digest)

        keep.append(index)

    kept_records = []

    for new_row, old_row in enumerate(keep):
        record = dict(records[old_row])
        record["row"] = new_row
        kept_records.append(record)

    return vectors[keep], kept_records, len(records) - len(keep)


def matches_filters(
    record: dict[str, Any],
    filters: dict[str, Any] | None,
) -> bool:
    r"""
    PURPOSE
    -------
    Test one record against a field filter. A list value means "any of
    these". Comparison is case-insensitive for strings, so
    ``{"topic": "heat"}`` finds ``"Heat"``.

    PARAMETERS
    ----------
    record    Record to test.
    filters   Dict of field name to wanted value (or list of values), or
              None for no filtering.

    RETURNS
    -------
    True if the record matches every filter field (or filters is empty),
    False otherwise.

    EXAMPLES
    --------
    >>> record = {"dataset": "rag", "topic": "Heat", "source_file": "physics.pdf"}
    >>> matches_filters(record, None), matches_filters(record, {})
    (True, True)
    >>> matches_filters(record, {"dataset": "rag"})
    True
    >>> matches_filters(record, {"dataset": "llm"})
    False
    >>> matches_filters(record, {"topic": "heat"})
    True
    >>> matches_filters(record, {"dataset": ["llm", "rag"]})
    True
    >>> matches_filters(record, {"topic": "Heat", "source_file": "biology.pdf"})
    False
    >>> matches_filters(record, {"unknown_field": "x"})
    False
    """
    if not filters:
        return True

    for field_name, wanted in filters.items():
        actual = record.get(field_name)

        if actual is None:
            return False

        options = wanted if isinstance(wanted, (list, tuple, set)) else [wanted]

        if not any(
            str(actual).casefold() == str(option).casefold()
            for option in options
        ):
            return False

    return True


def score_against(
    vectors: np.ndarray,
    query_vector: np.ndarray,
) -> np.ndarray:
    r"""
    PURPOSE
    -------
    Compute the cosine similarity of every row against a query vector.

    PARAMETERS
    ----------
    vectors        Matrix of unit-length row vectors.
    query_vector   Unit-length query vector.

    RETURNS
    -------
    An array of similarity scores, one per row of vectors.

    EXAMPLES
    --------
    >>> vectors = normalize_rows(np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]))
    >>> query = normalize_rows(np.array([1.0, 0.0]))[0]
    >>> [round(float(value), 3) for value in score_against(vectors, query)]
    [1.0, 0.0, 0.707]
    >>> score_against(np.zeros((0, 2), dtype="float32"), query).shape
    (0,)
    """
    if len(vectors) == 0:
        return np.zeros((0,), dtype=VECTOR_DATA_TYPE)

    query_vector = np.asarray(query_vector, dtype=VECTOR_DATA_TYPE).reshape(-1)
    scores = np.empty(len(vectors), dtype=VECTOR_DATA_TYPE)

    for start in range(0, len(vectors), SEARCH_BLOCK_SIZE):
        block = np.asarray(
            vectors[start:start + SEARCH_BLOCK_SIZE],
            dtype=VECTOR_DATA_TYPE,
        )
        scores[start:start + len(block)] = block @ query_vector

    return scores


def maximal_marginal_relevance(
    vectors: np.ndarray,
    scores: np.ndarray,
    candidates: list[int],
    top_k: int,
    relevance_weight: float,
) -> list[int]:
    r"""
    PURPOSE
    -------
    Re-rank candidates to trade relevance against variety. A
    ``relevance_weight`` of 1.0 is pure relevance; lower values push the
    selection away from rows that repeat what is already chosen.

    PARAMETERS
    ----------
    vectors            Matrix of unit-length row vectors.
    scores             Similarity score for each row, aligned by index.
    candidates         Row numbers to choose from.
    top_k              Maximum number of rows to select.
    relevance_weight   Trade-off between relevance and diversity, in [0, 1].

    RETURNS
    -------
    The selected row numbers, in selection order.

    EXAMPLES
    --------
    >>> vectors = normalize_rows(np.array([
    ...     [1.0, 0.0], [0.99, 0.01], [0.0, 1.0]]))
    >>> query = normalize_rows(np.array([1.0, 0.0]))[0]
    >>> scores = score_against(vectors, query)
    >>> maximal_marginal_relevance(vectors, scores, [0, 1, 2], 2, 1.0)
    [0, 1]
    >>> maximal_marginal_relevance(vectors, scores, [0, 1, 2], 2, 0.3)
    [0, 2]
    """
    if relevance_weight >= 1.0 or len(candidates) <= 1:
        return sorted(candidates, key=lambda row: -scores[row])[:top_k]

    remaining = sorted(candidates, key=lambda row: -scores[row])
    selected: list[int] = [remaining.pop(0)]

    while remaining and len(selected) < top_k:
        best_row = None
        best_value = -np.inf

        for row in remaining:
            similarity = max(
                float(np.dot(vectors[row], vectors[chosen]))
                for chosen in selected
            )
            value = (
                relevance_weight * float(scores[row])
                - (1.0 - relevance_weight) * similarity
            )

            if value > best_value:
                best_value = value
                best_row = row

        selected.append(best_row)
        remaining.remove(best_row)

    return selected


class VectorBase:
    r"""
    PURPOSE
    -------
    An in-memory view of a vector base folder: a row-aligned matrix of
    unit-length vectors, the record for each row, and the manifest
    describing how the base was built. This is the object that both builds
    a base (via ``build_vector_base``) and is reloaded to search one (via
    ``VectorBase.load``), so it is the one place that knows how to turn a
    query vector into ranked, filtered, cited results.

    ATTRIBUTES
    ----------
    vectors    Matrix of unit-length row vectors, one row per record.
    records    List of record dicts, aligned by row with vectors.
    manifest   Dict describing how the base was built.

    EXAMPLES
    --------
    >>> vectors = normalize_rows(np.array([[1.0, 0.0], [0.0, 1.0]]))
    >>> records = [
    ...     {"row": 0, "dataset": "rag", "topic": "Heat", "text": "About heat."},
    ...     {"row": 1, "dataset": "llm", "topic": "Light", "text": "About light."},
    ... ]
    >>> base = VectorBase(vectors, records, {"embedding_model": "stub"})
    >>> len(base), base.dimension
    (2, 2)
    >>> results = base.search(np.array([1.0, 0.0]), top_k=1)
    >>> results[0].rank, round(results[0].score, 3), results[0].text
    (1, 1.0, 'About heat.')
    >>> [r.record["topic"] for r in base.search(np.array([1.0, 0.0]), top_k=2,
    ...                                         filters={"dataset": "llm"})]
    ['Light']
    >>> base.search(np.array([1.0, 0.0]), top_k=2, filters={"topic": "nothing"})
    []
    >>> base.topics()
    ['Heat', 'Light']
    """

    def __init__(
        self,
        vectors: np.ndarray,
        records: list[dict[str, Any]],
        manifest: dict[str, Any] | None = None,
    ):
        if len(vectors) != len(records):
            raise ValueError(
                f"{len(vectors)} vectors but {len(records)} record(s)."
            )

        self.vectors = vectors
        self.records = records
        self.manifest = manifest or {}

    def __len__(self) -> int:
        """
        PURPOSE
        -------
        Number of records in the base.
        """
        return len(self.records)

    @property
    def dimension(self) -> int:
        """
        PURPOSE
        -------
        Vector width, or 0 if the base is empty.
        """
        return int(self.vectors.shape[1]) if len(self.vectors) else 0

    @property
    def embedding_model(self) -> str | None:
        """
        PURPOSE
        -------
        Name of the embedding model recorded in the manifest, if any.
        """
        return self.manifest.get("embedding_model")

    def topics(self) -> list[str]:
        """
        PURPOSE
        -------
        List the distinct topics in the base.

        RETURNS
        -------
        Sorted list of the distinct topic values.
        """
        return sorted(
            {
                str(record["topic"])
                for record in self.records
                if record.get("topic")
            }
        )

    def datasets(self) -> list[str]:
        """
        PURPOSE
        -------
        List the dataset names present in the base.

        RETURNS
        -------
        Sorted list of the distinct dataset names.
        """
        return sorted(
            {
                str(record["dataset"])
                for record in self.records
                if record.get("dataset")
            }
        )

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = DEFAULT_TOP_K,
        filters: dict[str, Any] | None = None,
        relevance_weight: float = DEFAULT_MMR_LAMBDA,
        minimum_score: float = -1.0,
    ) -> list[SearchResult]:
        """
        PURPOSE
        -------
        Return the ``top_k`` closest records, optionally filtered and
        diversified.

        PARAMETERS
        ----------
        query_vector       Vector to search for, of the base's dimension.
        top_k              Maximum number of results to return.
        filters            Dict of field name to wanted value (or list of
                            values), or None for no filtering.
        relevance_weight   Trade-off between relevance and diversity, in
                            [0, 1]; 1.0 is pure relevance.
        minimum_score      Lowest cosine similarity a row may have to be
                            considered.

        RETURNS
        -------
        Up to top_k SearchResult objects, best match first.

        RAISES
        ------
        ValueError   query_vector's dimension does not match the base's.
        """
        if len(self) == 0 or top_k <= 0:
            return []

        query_vector = normalize_rows(
            np.asarray(query_vector, dtype=VECTOR_DATA_TYPE).reshape(1, -1)
        )[0]

        if query_vector.shape[0] != self.dimension:
            raise ValueError(
                f"Query has {query_vector.shape[0]} dimensions but the base "
                f"has {self.dimension}."
            )

        scores = score_against(self.vectors, query_vector)

        candidates = [
            row
            for row in range(len(self))
            if scores[row] >= minimum_score
            and matches_filters(self.records[row], filters)
        ]

        if not candidates:
            return []

        # consider a wider pool than top_k so re-ranking has room to work
        pool_size = top_k if relevance_weight >= 1.0 else min(
            len(candidates), max(top_k * 4, 20)
        )
        pool = sorted(candidates, key=lambda row: -scores[row])[:pool_size]

        chosen = maximal_marginal_relevance(
            self.vectors, scores, pool, top_k, relevance_weight
        )

        return [
            SearchResult(
                rank=rank,
                score=float(scores[row]),
                row=row,
                text=str(self.records[row].get("text") or ""),
                record=self.records[row],
            )
            for rank, row in enumerate(chosen, start=1)
        ]

    def save(self, folder: Path) -> dict[str, Any]:
        """
        PURPOSE
        -------
        Write the base to ``folder`` as vectors.npy, records.jsonl and a
        manifest.

        PARAMETERS
        ----------
        folder   Folder to write the base's files into.

        RETURNS
        -------
        The manifest that was written, with build statistics merged in.
        """
        folder.mkdir(parents=True, exist_ok=True)

        atomic_write_numpy(folder / VECTOR_FILE_NAME, self.vectors)
        atomic_write_text(
            folder / RECORD_FILE_NAME,
            "".join(
                json.dumps(record, ensure_ascii=False) + "\n"
                for record in self.records
            ),
        )

        manifest = dict(self.manifest)
        manifest.update(
            {
                "vector_count": len(self),
                "dimension": self.dimension,
                "vector_dtype": str(np.dtype(VECTOR_DATA_TYPE)),
                "vector_file": str(folder / VECTOR_FILE_NAME),
                "record_file": str(folder / RECORD_FILE_NAME),
            }
        )

        atomic_write_text(
            folder / MANIFEST_FILE_NAME,
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        )

        self.manifest = manifest

        return manifest

    @classmethod
    def load(cls, folder: Path) -> VectorBase:
        """
        PURPOSE
        -------
        Open a vector base folder, memory-mapping the vectors.

        PARAMETERS
        ----------
        folder   Folder previously written by ``save``.

        RETURNS
        -------
        A VectorBase over the folder's vectors, records and manifest.

        RAISES
        ------
        FileNotFoundError   The folder has no vector base files in it.
        """
        vector_path = folder / VECTOR_FILE_NAME
        record_path = folder / RECORD_FILE_NAME
        manifest_path = folder / MANIFEST_FILE_NAME

        if not vector_path.exists() or not record_path.exists():
            raise FileNotFoundError(
                f"No vector base found in {folder}. Run vector_base.py first."
            )

        vectors = np.load(vector_path, mmap_mode="r", allow_pickle=False)
        records = list(read_jsonl(record_path))

        manifest: dict[str, Any] = {}

        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        return cls(vectors, records, manifest)


def build_vector_base(
    embedding_folder: Path = EMBEDDING_FOLDER,
    chunked_folder: Path = CHUNKED_FOLDER,
    output_folder: Path = OUTPUT_FOLDER,
    datasets: tuple[str, ...] = DATASET_NAMES,
    deduplicate_rows: bool = True,
    verbose: bool = True,
) -> dict[str, Any]:
    r"""
    PURPOSE
    -------
    Assemble the embedding outputs into a vector base folder and return its
    manifest.

    PARAMETERS
    ----------
    embedding_folder   Folder holding the .npy vectors and metadata.
    chunked_folder     Folder holding the passages, for the text join.
    output_folder      Folder to write the vector base into.
    datasets           Which embedded datasets to include.
    deduplicate_rows   Whether to collapse rows with duplicate text.
    verbose            Whether to print progress as the base is built.

    RETURNS
    -------
    The manifest describing the built vector base.

    RAISES
    ------
    FileNotFoundError   embedding_folder does not exist.
    ValueError           No vectors were found across the requested
                          datasets, or the datasets have mismatched vector
                          dimensions.

    EXAMPLES
    --------
    >>> import tempfile
    >>> with tempfile.TemporaryDirectory() as folder:
    ...     root = Path(folder)
    ...     embedded, chunked, base = root / "emb", root / "chunk", root / "base"
    ...     embedded.mkdir(); chunked.mkdir()
    ...     np.save(embedded / "rag_embeddings.npy",
    ...             np.array([[1.0, 0.0], [0.0, 1.0]], dtype="float32"))
    ...     _ = (embedded / "rag_embedding_metadata.jsonl").write_text(
    ...         '{"vector_index": 0, "chunk_id": "C1", "topic": "Heat",'
    ...         ' "text_sha256": "aa"}\n'
    ...         '{"vector_index": 1, "chunk_id": "C2", "topic": "Light",'
    ...         ' "text_sha256": "bb"}\n', encoding="utf-8")
    ...     _ = (chunked / "rag_data.jsonl").write_text(
    ...         '{"chunk_id": "C1", "text": "Heat flows from hot to cold."}\n'
    ...         '{"chunk_id": "C2", "text": "Light travels in straight lines."}\n',
    ...         encoding="utf-8")
    ...     manifest = build_vector_base(embedded, chunked, base,
    ...                                  datasets=("rag",), verbose=False)
    ...     written = sorted(p.name for p in base.iterdir())
    ...     loaded = VectorBase.load(base)
    >>> manifest["vector_count"], manifest["dimension"], manifest["records_with_text"]
    (2, 2, 2)
    >>> written
    ['records.jsonl', 'vector_base_manifest.json', 'vectors.npy']
    >>> loaded.search(np.array([1.0, 0.0]), top_k=1)[0].text
    'Heat flows from hot to cold.'
    >>> build_vector_base(Path("/definitely/not/here"), Path("/tmp"), Path("/tmp/x"))
    Traceback (most recent call last):
        ...
    FileNotFoundError: Embedding folder does not exist: /definitely/not/here
    """
    if not embedding_folder.is_dir():
        raise FileNotFoundError(
            f"Embedding folder does not exist: {embedding_folder}"
        )

    def announce(message: str) -> None:
        if verbose:
            print(message)

    join_lookup = build_join_lookup(chunked_folder)

    if not join_lookup:
        announce(
            f"Warning: no passages found in {chunked_folder}. "
            "Records will carry identifiers but no text."
        )

    all_vectors: list[np.ndarray] = []
    all_records: list[dict[str, Any]] = []
    dataset_results: list[dict[str, Any]] = []
    embedding_models: set[str] = set()

    for dataset_name in datasets:
        vectors, metadata_records = load_dataset(embedding_folder, dataset_name)

        if len(vectors) == 0:
            dataset_results.append(
                {
                    "dataset": dataset_name,
                    "status": "skipped",
                    "reason": "No vectors found for this dataset.",
                    "vector_count": 0,
                }
            )
            announce(f"{dataset_name}: skipped (no vectors)")
            continue

        for metadata in metadata_records:
            model = metadata.get("embedding_model")

            if model:
                embedding_models.add(str(model))

        start_row = len(all_records)

        all_vectors.append(normalize_rows(vectors))
        all_records.extend(
            build_record(metadata, dataset_name, start_row + offset, join_lookup)
            for offset, metadata in enumerate(metadata_records)
        )

        dataset_results.append(
            {
                "dataset": dataset_name,
                "status": "included",
                "vector_count": len(vectors),
                "dimension": int(vectors.shape[1]),
            }
        )

        announce(f"{dataset_name}: {len(vectors)} vector(s)")

    if not all_records:
        raise ValueError(
            f"No vectors were found in {embedding_folder}. "
            "Run embedding.py first."
        )

    dimensions = {int(block.shape[1]) for block in all_vectors}

    if len(dimensions) > 1:
        raise ValueError(
            f"Datasets have different vector dimensions: {sorted(dimensions)}. "
            "They must come from the same embedding model."
        )

    vectors = np.vstack(all_vectors).astype(VECTOR_DATA_TYPE)
    removed = 0

    if deduplicate_rows:
        vectors, all_records, removed = deduplicate(vectors, all_records)
        announce(f"Deduplicated: {removed} row(s) removed")

    manifest = {
        "program": "SoakinGarri Vector Base Program",
        "embedding_folder": str(embedding_folder),
        "chunked_folder": str(chunked_folder),
        "output_folder": str(output_folder),
        "source_label": SOURCE_LABEL,
        "embedding_model": (
            sorted(embedding_models)[0] if len(embedding_models) == 1 else None
        ),
        "embedding_models_seen": sorted(embedding_models),
        "datasets": dataset_results,
        "deduplicated": deduplicate_rows,
        "duplicates_removed": removed,
        "records_with_text": sum(
            1 for record in all_records if record.get("has_text")
        ),
        "records_without_text": sum(
            1 for record in all_records if not record.get("has_text")
        ),
    }

    base = VectorBase(vectors, all_records, manifest)
    manifest = base.save(output_folder)

    if verbose:
        print(
            f"\nVector base built.\n"
            f"Vectors: {manifest['vector_count']} x {manifest['dimension']}\n"
            f"With text: {manifest['records_with_text']}\n"
            f"Folder: {output_folder}"
        )

    return manifest


def parse_arguments(argument_list: list[str] | None = None) -> argparse.Namespace:
    r"""
    PURPOSE
    -------
    Parse the command line.

    PARAMETERS
    ----------
    argument_list   Arguments to parse, or None to use sys.argv.

    RETURNS
    -------
    The parsed options.

    EXAMPLES
    --------
    >>> parse_arguments([]).datasets
    ['llm', 'rag']
    >>> parse_arguments(["--datasets", "rag"]).datasets
    ['rag']
    >>> parse_arguments(["--no-deduplicate"]).deduplicate
    False
    """
    parser = argparse.ArgumentParser(
        description="Assemble embedding outputs into a searchable vector base."
    )

    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=list(DATASET_NAMES),
        default=list(DATASET_NAMES),
        help="which embedded datasets to include",
    )

    parser.add_argument(
        "--embeddings",
        type=Path,
        default=EMBEDDING_FOLDER,
        help="folder holding the .npy vectors and metadata",
    )

    parser.add_argument(
        "--chunked",
        type=Path,
        default=CHUNKED_FOLDER,
        help="folder holding the passages, for the text join",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_FOLDER,
        help="folder to write the vector base into",
    )

    parser.add_argument(
        "--no-deduplicate",
        dest="deduplicate",
        action="store_false",
        help="keep repeated passages instead of collapsing them",
    )

    return parser.parse_args(argument_list)


def main() -> None:
    """
    PURPOSE
    -------
    Build the vector base using the options given on the command line.
    """
    options = parse_arguments()

    build_vector_base(
        embedding_folder=options.embeddings,
        chunked_folder=options.chunked,
        output_folder=options.output,
        datasets=tuple(options.datasets),
        deduplicate_rows=options.deduplicate,
    )


if __name__ == "__main__":
    main()
