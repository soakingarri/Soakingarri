#!/usr/bin/env python3

"""
SoakinGarri FAISS Vector Base Program

PURPOSE
-------
A FAISS-backed alternative to vector_base.py. Same job — assemble the
embedding program's outputs into one searchable vector base, with the same
text join, deduplication, filtering and diversity re-ranking — but search
runs through a FAISS index instead of a plain numpy matrix.

This module deliberately reuses vector_base.py's dataset loading, text join,
deduplication and filtering logic rather than re-implementing it, so the two
programs cannot drift apart on anything but the search engine itself.

WHAT FAISS BUYS YOU HERE
-------------------------
IndexFlatIP is an *exact* index: it still scans every vector per query, same
as the numpy version, so this swap does not change the algorithmic scaling by
itself — it mainly gives faster constant-factor search (FAISS's C++/SIMD
brute force) and a real on-disk index format. The payoff for scale comes
later, when a large corpus justifies swapping IndexFlatIP for an approximate
index (IndexIVFFlat, IndexHNSWFlat): that change happens in build_index()
below without touching search(), build_vector_base_faiss(), or callers —
but approximate indexes trade a little recall for speed, and combining them
with the field filters here needs care (a small candidate pool from an
approximate search can be filtered down to nothing). Cross that bridge when
the corpus size actually calls for it.

INPUT FOLDERS
-------------
<repo>/data/ask/embedding_data      vectors and metadata
<repo>/data/ask/chunked_data        passages, for the text join

OUTPUT FOLDER
-------------
<repo>/data/ask/vector_base_faiss

OUTPUT FILES
------------
vectors.faiss                a FAISS IndexFlatIP, written with faiss.write_index
records.jsonl                one record per row, in row order
vector_base_manifest.json

INSTALL
-------
python3 -m pip install numpy faiss-cpu

RUN
---
python3 vector_base_faiss.py
python3 vector_base_faiss.py --datasets rag
python3 vector_base_faiss.py --no-deduplicate
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import faiss
import numpy as np

import vector_base as vb

sys.path.insert(
    0, str(Path(__file__).resolve().parent.parent.parent.parent / "source")
)

from common import DATA_ROOT  # noqa: E402

EMBEDDING_FOLDER = vb.EMBEDDING_FOLDER
CHUNKED_FOLDER = vb.CHUNKED_FOLDER

OUTPUT_FOLDER = DATA_ROOT / "vector_base_faiss"

FAISS_INDEX_FILE_NAME = "vectors.faiss"
RECORD_FILE_NAME = vb.RECORD_FILE_NAME
MANIFEST_FILE_NAME = vb.MANIFEST_FILE_NAME

DATASET_NAMES = vb.DATASET_NAMES
DEFAULT_TOP_K = vb.DEFAULT_TOP_K
DEFAULT_MMR_LAMBDA = vb.DEFAULT_MMR_LAMBDA


def atomic_write_faiss_index(destination: Path, index: faiss.Index) -> None:
    r"""
    PURPOSE
    -------
    Write a FAISS index through a temporary file, then rename it over the
    target.

    PARAMETERS
    ----------
    destination   Path the index should end up at.
    index         The FAISS index to write.

    EXAMPLES
    --------
    >>> import tempfile
    >>> with tempfile.TemporaryDirectory() as folder:
    ...     path = Path(folder) / "v.faiss"
    ...     idx = faiss.IndexFlatIP(2)
    ...     idx.add(np.eye(2, dtype="float32"))
    ...     atomic_write_faiss_index(path, idx)
    ...     loaded = faiss.read_index(str(path))
    ...     (loaded.ntotal, loaded.d, [p.name for p in path.parent.iterdir()])
    (2, 2, ['v.faiss'])
    """
    destination.parent.mkdir(parents=True, exist_ok=True)

    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".faiss",
    )

    os.close(file_descriptor)
    temporary_path = Path(temporary_name)

    try:
        faiss.write_index(index, str(temporary_path))
        temporary_path.replace(destination)

    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


class FaissVectorBase:
    """
    PURPOSE
    -------
    A FAISS-backed view of a vector base folder. It mirrors
    vector_base.VectorBase's search contract exactly (field filters,
    minimum_score, maximal-marginal-relevance diversity), but the vectors
    live in a FAISS IndexFlatIP instead of a numpy matrix held in this
    process.

    ATTRIBUTES
    ----------
    index      The FAISS index backing the vectors.
    records    One metadata dict per row, in row order.
    manifest   The base's manifest dict (build/save metadata).
    """

    def __init__(
        self,
        index: faiss.Index,
        records: list[dict[str, Any]],
        manifest: dict[str, Any] | None = None,
    ):
        """
        PURPOSE
        -------
        Build a FaissVectorBase around an already-populated FAISS index and
        its matching records.

        PARAMETERS
        ----------
        index      A FAISS index already populated with row vectors.
        records    One metadata dict per row, in row order.
        manifest   Optional manifest dict; defaults to an empty dict.

        RAISES
        ------
        ValueError   If the index's row count does not match len(records).
        """
        if index.ntotal != len(records):
            raise ValueError(
                f"{index.ntotal} vectors but {len(records)} record(s)."
            )

        self.index = index
        self.records = records
        self.manifest = manifest or {}

    def __len__(self) -> int:
        """PURPOSE -- Number of records in the base."""
        return len(self.records)

    @property
    def dimension(self) -> int:
        """PURPOSE -- Vector dimension of the base, or 0 if empty."""
        return int(self.index.d) if len(self.records) else 0

    @property
    def embedding_model(self) -> str | None:
        """PURPOSE -- The embedding model name recorded in the manifest, if any."""
        return self.manifest.get("embedding_model")

    def topics(self) -> list[str]:
        """PURPOSE -- Sorted list of the distinct topics in the base."""
        return sorted(
            {
                str(record["topic"])
                for record in self.records
                if record.get("topic")
            }
        )

    def datasets(self) -> list[str]:
        """PURPOSE -- Sorted list of the dataset names present in the base."""
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
    ) -> list[vb.SearchResult]:
        """
        PURPOSE
        -------
        Return the top_k closest records, optionally filtered and
        diversified. Because IndexFlatIP is an exact index, every row's
        score is requested from FAISS so that arbitrary field filters can
        be applied afterward without missing a match that a smaller
        top-k search would have discarded first; only the small
        candidate pool that survives filtering and scoring is then
        reconstructed to numpy vectors for MMR re-ranking, which keeps
        memory flat as the base grows.

        PARAMETERS
        ----------
        query_vector       The query embedding.
        top_k              Maximum number of results to return.
        filters            Optional field filters records must match.
        relevance_weight   MMR relevance/diversity trade-off (lambda).
        minimum_score      Minimum similarity score a record must reach.

        RETURNS
        -------
        Up to top_k SearchResult objects, ranked and diversified.

        RAISES
        ------
        ValueError   If query_vector's dimension does not match the base's.
        """
        total = len(self)

        if total == 0 or top_k <= 0:
            return []

        query_vector = vb.normalize_rows(
            np.asarray(query_vector, dtype=vb.VECTOR_DATA_TYPE).reshape(1, -1)
        )[0]

        if query_vector.shape[0] != self.dimension:
            raise ValueError(
                f"Query has {query_vector.shape[0]} dimensions but the base "
                f"has {self.dimension}."
            )

        # An exact index, so asking for every row's score is the correct way
        # to support arbitrary field filters without missing a match that a
        # smaller top-k search would have discarded before filtering.
        raw_scores, raw_rows = self.index.search(
            query_vector.reshape(1, -1).astype(np.float32), total
        )

        scores = np.full(total, -np.inf, dtype=np.float32)

        for score, row in zip(raw_scores[0], raw_rows[0]):
            if row >= 0:
                scores[row] = score

        candidates = [
            row
            for row in range(total)
            if scores[row] >= minimum_score
            and vb.matches_filters(self.records[row], filters)
        ]

        if not candidates:
            return []

        # consider a wider pool than top_k so re-ranking has room to work
        pool_size = top_k if relevance_weight >= 1.0 else min(
            len(candidates), max(top_k * 4, 20)
        )
        pool = sorted(candidates, key=lambda row: -scores[row])[:pool_size]

        # only reconstruct vectors for the small candidate pool, never the
        # whole base — this is the part that keeps memory flat as the base grows
        pool_vectors = {
            row: self.index.reconstruct(int(row)) for row in pool
        }

        chosen = vb.maximal_marginal_relevance(
            pool_vectors, scores, pool, top_k, relevance_weight
        )

        return [
            vb.SearchResult(
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
        Write the base to folder as vectors.faiss, records.jsonl and a
        manifest.

        PARAMETERS
        ----------
        folder   Destination folder to write the base into.

        RETURNS
        -------
        The manifest dict that was written, with vector_count, dimension
        and file locations merged in.
        """
        folder.mkdir(parents=True, exist_ok=True)

        atomic_write_faiss_index(folder / FAISS_INDEX_FILE_NAME, self.index)
        vb.atomic_write_text(
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
                "index_type": type(self.index).__name__,
                "index_file": str(folder / FAISS_INDEX_FILE_NAME),
                "record_file": str(folder / RECORD_FILE_NAME),
            }
        )

        vb.atomic_write_text(
            folder / MANIFEST_FILE_NAME,
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        )

        self.manifest = manifest

        return manifest

    @classmethod
    def load(cls, folder: Path) -> FaissVectorBase:
        """
        PURPOSE
        -------
        Open a FAISS vector base folder.

        PARAMETERS
        ----------
        folder   Folder holding vectors.faiss and records.jsonl.

        RETURNS
        -------
        The loaded FaissVectorBase.

        RAISES
        ------
        FileNotFoundError   If the index or records file is missing from folder.
        """
        index_path = folder / FAISS_INDEX_FILE_NAME
        record_path = folder / RECORD_FILE_NAME
        manifest_path = folder / MANIFEST_FILE_NAME

        if not index_path.exists() or not record_path.exists():
            raise FileNotFoundError(
                f"No FAISS vector base found in {folder}. "
                "Run vector_base_faiss.py first."
            )

        index = faiss.read_index(str(index_path))
        records = list(vb.read_jsonl(record_path))

        manifest: dict[str, Any] = {}

        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        return cls(index, records, manifest)

    def upsert(
        self,
        new_vectors: np.ndarray,
        new_records: list[dict[str, Any]],
    ) -> FaissVectorBase:
        r"""
        PURPOSE
        -------
        Fold new_vectors/new_records into the base, replacing any existing
        record whose chunk_id matches one of the incoming records instead
        of duplicating it. This lets a single re-embedded/re-chunked book
        be folded back into an existing base without rebuilding the whole
        base from embedding_data/chunked_data.

        Implementation note: rather than reworking the FAISS index to
        support native removal (which search()'s row-based
        reconstruction assumes it does not need), every existing vector
        is reconstructed via self.index.reconstruct_n, any row whose
        chunk_id is being replaced is dropped, the new vectors are
        appended, and a fresh index is rebuilt via build_index(). This is
        cheap at this corpus's realistic scale (a handful of books, not
        millions of vectors).

        PARAMETERS
        ----------
        new_vectors   Matrix of new row vectors, one per new_records
                      entry. Normalized to unit length inside this
                      method — callers do not need to normalize first.
        new_records   One metadata dict per new vector, each carrying a
                      "chunk_id" key (the same key build_record uses).

        RETURNS
        -------
        A new FaissVectorBase holding every existing record whose
        chunk_id is not among the incoming chunk_ids, followed by
        new_records, with "row" renumbered to match — self is not
        modified.

        RAISES
        ------
        ValueError   len(new_vectors) != len(new_records).

        EXAMPLES
        --------
        >>> vectors = vb.normalize_rows(np.array([[1.0, 0.0], [0.0, 1.0]]))
        >>> records = [
        ...     {"row": 0, "chunk_id": "C1", "text": "old C1 text"},
        ...     {"row": 1, "chunk_id": "C2", "text": "C2 text"},
        ... ]
        >>> base = FaissVectorBase(build_index(vectors), records)
        >>> updated = base.upsert(
        ...     np.array([[0.0, 1.0]]), [{"chunk_id": "C1", "text": "new C1 text"}]
        ... )
        >>> len(updated)
        2
        >>> [r["chunk_id"] for r in updated.records]
        ['C2', 'C1']
        >>> [r["text"] for r in updated.records if r["chunk_id"] == "C1"]
        ['new C1 text']
        >>> [r["text"] for r in updated.records if r["chunk_id"] == "C2"]
        ['C2 text']
        >>> [r["row"] for r in updated.records]
        [0, 1]
        >>> len(base)
        2

        A genuinely new chunk_id is appended rather than replacing anything:

        >>> grown = updated.upsert(
        ...     np.array([[1.0, 1.0]]), [{"chunk_id": "C3", "text": "C3 text"}]
        ... )
        >>> len(grown)
        3
        >>> sorted(r["chunk_id"] for r in grown.records)
        ['C1', 'C2', 'C3']
        >>> [r["row"] for r in grown.records]
        [0, 1, 2]

        Records with no chunk_id are appended, never matched to each other:

        >>> len(grown.upsert(np.array([[1.0, 0.0]]), [{"text": "a"}])
        ...          .upsert(np.array([[0.0, 1.0]]), [{"text": "b"}]))
        5
        """
        if len(new_vectors) != len(new_records):
            raise ValueError(
                f"{len(new_vectors)} new vector(s) but "
                f"{len(new_records)} new record(s)."
            )

        new_vectors = vb.normalize_rows(
            np.asarray(new_vectors, dtype=vb.VECTOR_DATA_TYPE)
        )

        # A record without a chunk_id has no identity to match on, so it is
        # only ever appended; it must never wipe out other id-less records.
        incoming_chunk_ids = {
            record.get("chunk_id")
            for record in new_records
            if record.get("chunk_id")
        }

        all_existing_vectors = (
            self.index.reconstruct_n(0, len(self))
            if len(self)
            else np.zeros((0, self.index.d), dtype=np.float32)
        )

        kept_rows = [
            row
            for row, record in enumerate(self.records)
            if record.get("chunk_id") not in incoming_chunk_ids
        ]

        combined_vectors = np.vstack(
            [all_existing_vectors[kept_rows], new_vectors]
        ).astype(np.float32)

        combined_records = [
            dict(self.records[row]) for row in kept_rows
        ] + [dict(record) for record in new_records]

        for row, record in enumerate(combined_records):
            record["row"] = row

        index = build_index(combined_vectors)

        manifest = dict(self.manifest)
        manifest["vector_count"] = len(combined_records)
        manifest["dimension"] = int(index.d) if len(combined_records) else 0

        return FaissVectorBase(index, combined_records, manifest)


def build_index(vectors: np.ndarray) -> faiss.Index:
    """
    PURPOSE
    -------
    Build the FAISS index for a matrix of unit-length row vectors.
    IndexFlatIP does exact search via inner product, which equals cosine
    similarity here because every row is unit-length. This is the one
    function to change if a future corpus size calls for an approximate
    index (IndexIVFFlat, IndexHNSWFlat) instead — that swap happens here
    without touching search(), build_vector_base_faiss(), or callers, but
    approximate indexes trade a little recall for speed, and combining
    them with the field filters in search() needs care, since a small
    candidate pool from an approximate search can be filtered down to
    nothing.

    PARAMETERS
    ----------
    vectors   Matrix of unit-length row vectors to index.

    RETURNS
    -------
    A populated faiss.IndexFlatIP.
    """
    dimension = int(vectors.shape[1])
    index = faiss.IndexFlatIP(dimension)
    index.add(np.ascontiguousarray(vectors, dtype=np.float32))
    return index


def build_vector_base_faiss(
    embedding_folder: Path = EMBEDDING_FOLDER,
    chunked_folder: Path = CHUNKED_FOLDER,
    output_folder: Path = OUTPUT_FOLDER,
    datasets: tuple[str, ...] = DATASET_NAMES,
    deduplicate_rows: bool = True,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    PURPOSE
    -------
    Assemble the embedding outputs into a FAISS-backed vector base folder.
    Mirrors vector_base.build_vector_base's loading, text-join and
    deduplication behaviour exactly (it calls the same functions); only
    the final index construction and file formats differ.

    PARAMETERS
    ----------
    embedding_folder   Folder holding the .npy vectors and metadata.
    chunked_folder     Folder holding the passages, for the text join.
    output_folder      Folder to write the FAISS vector base into.
    datasets           Which embedded datasets to include.
    deduplicate_rows   Whether to collapse repeated passages.
    verbose            Whether to print progress messages.

    RETURNS
    -------
    The manifest dict written for the built base.

    RAISES
    ------
    FileNotFoundError   If embedding_folder does not exist.
    ValueError          If no vectors are found, or included datasets
                         have mismatched vector dimensions.
    """
    if not embedding_folder.is_dir():
        raise FileNotFoundError(
            f"Embedding folder does not exist: {embedding_folder}"
        )

    def announce(message: str) -> None:
        if verbose:
            print(message)

    join_lookup = vb.build_join_lookup(chunked_folder)

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
        vectors, metadata_records = vb.load_dataset(embedding_folder, dataset_name)

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

        all_vectors.append(vb.normalize_rows(vectors))
        all_records.extend(
            vb.build_record(metadata, dataset_name, start_row + offset, join_lookup)
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

    vectors = np.vstack(all_vectors).astype(vb.VECTOR_DATA_TYPE)
    removed = 0

    if deduplicate_rows:
        vectors, all_records, removed = vb.deduplicate(vectors, all_records)
        announce(f"Deduplicated: {removed} row(s) removed")

    index = build_index(vectors)

    manifest = {
        "program": "SoakinGarri FAISS Vector Base Program",
        "backend": "faiss.IndexFlatIP",
        "embedding_folder": str(embedding_folder),
        "chunked_folder": str(chunked_folder),
        "output_folder": str(output_folder),
        "source_label": vb.SOURCE_LABEL,
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

    base = FaissVectorBase(index, all_records, manifest)
    manifest = base.save(output_folder)

    if verbose:
        print(
            f"\nFAISS vector base built.\n"
            f"Vectors: {manifest['vector_count']} x {manifest['dimension']}\n"
            f"With text: {manifest['records_with_text']}\n"
            f"Folder: {output_folder}"
        )

    return manifest


def upsert_vector_base_faiss(
    folder: Path,
    new_vectors: np.ndarray,
    new_records: list[dict[str, Any]],
) -> dict[str, Any]:
    r"""
    PURPOSE
    -------
    Load the FAISS vector base at folder, upsert new_vectors/new_records
    into it (replacing any existing record whose chunk_id matches one of
    the incoming records), and save the result back to folder. This is
    the incremental-update entry point: it lets one re-embedded book be
    folded into an existing base without re-running
    build_vector_base_faiss over the whole embedding_data/chunked_data
    folders.

    If folder has no base yet — FaissVectorBase.load(folder) raises
    FileNotFoundError — this starts from an empty base sized to
    new_vectors' dimension, so the first call for a brand-new folder
    creates the store instead of erroring.

    PARAMETERS
    ----------
    folder        Vector base folder to update (or create, if missing).
    new_vectors   Matrix of new row vectors, one per new_records entry.
    new_records   One metadata dict per new vector, each carrying a
                  "chunk_id" key.

    RETURNS
    -------
    The manifest dict written for the updated base.

    RAISES
    ------
    ValueError   len(new_vectors) != len(new_records).

    EXAMPLES
    --------
    >>> import tempfile
    >>> with tempfile.TemporaryDirectory() as folder:
    ...     path = Path(folder) / "base"
    ...     first_records = [
    ...         {"chunk_id": "C1", "text": "C1 text"},
    ...         {"chunk_id": "C2", "text": "C2 text"},
    ...     ]
    ...     first_manifest = upsert_vector_base_faiss(
    ...         path, np.array([[1.0, 0.0], [0.0, 1.0]]), first_records
    ...     )
    ...     second_manifest = upsert_vector_base_faiss(
    ...         path,
    ...         np.array([[0.0, 1.0]]),
    ...         [{"chunk_id": "C1", "text": "new C1 text"}],
    ...     )
    ...     reloaded = FaissVectorBase.load(path)
    >>> first_manifest["vector_count"]
    2
    >>> second_manifest["vector_count"]
    2
    >>> sorted(r["chunk_id"] for r in reloaded.records)
    ['C1', 'C2']
    >>> sorted(reloaded.records, key=lambda r: r["chunk_id"])[0]["text"]
    'new C1 text'
    """
    try:
        base = FaissVectorBase.load(folder)
    except FileNotFoundError:
        dimension = int(np.asarray(new_vectors).shape[1])
        base = FaissVectorBase(
            build_index(np.zeros((0, dimension), dtype=np.float32)), []
        )

    updated = base.upsert(new_vectors, new_records)

    return updated.save(folder)


def parse_arguments(argument_list: list[str] | None = None) -> argparse.Namespace:
    r"""
    PURPOSE
    -------
    Parse the command line.

    PARAMETERS
    ----------
    argument_list   Argument strings to parse; defaults to sys.argv.

    RETURNS
    -------
    The parsed argparse.Namespace.

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
        description="Assemble embedding outputs into a FAISS-backed searchable vector base."
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
        help="folder to write the FAISS vector base into",
    )

    parser.add_argument(
        "--no-deduplicate",
        dest="deduplicate",
        action="store_false",
        help="keep repeated passages instead of collapsing them",
    )

    return parser.parse_args(argument_list)


def main() -> None:
    """PURPOSE -- Build the FAISS vector base using the options given on the command line."""
    options = parse_arguments()

    build_vector_base_faiss(
        embedding_folder=options.embeddings,
        chunked_folder=options.chunked,
        output_folder=options.output,
        datasets=tuple(options.datasets),
        deduplicate_rows=options.deduplicate,
    )


if __name__ == "__main__":
    main()
