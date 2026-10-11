#!/usr/bin/env python3

"""
SoakinGarri Keyword Index Program

PURPOSE
-------
A BM25 keyword index over the same records vector_base.py indexes. A dense
embedding can miss an exact name, a formula written in symbols, or a local-
language term it was never trained to weight highly; a plain keyword index
catches those. This program is not a replacement for the vector base — it is
a second, cheap-to-build index meant to be combined with vector search
results by a hybrid-search caller (not this module).

This module deliberately reuses vector_base.py's dataset loading, text join
and deduplication logic rather than re-implementing it, so the two indexes
are built from the exact same records, in the exact same row order, with the
exact same ids. That is what lets a future hybrid-search merge line the two
result sets up by chunk_id/record_id without ambiguity.

WHY NO VECTORS HERE
--------------------
vector_base.py checks that every included dataset's vectors share one
dimension, because it stacks them into a single numpy matrix. A keyword
index never looks at vector values at all, so that constraint does not
apply here: datasets with embeddings from different models (even different
dimensions) can still share one BM25 index, since only their text matters.
load_dataset() is still called per dataset (and still reused as-is) because
it also validates that vector and metadata counts agree and that
vector_index values are in order — a correctness check worth keeping even
though the vectors themselves are discarded right after.

INPUT FOLDERS
-------------
<repo>/data/ask/embedding_data      metadata (vectors are loaded only to
                                     reuse vector_base.py's validation; their
                                     values are never used)
<repo>/data/ask/chunked_data        passages, for the text join

OUTPUT FOLDER
-------------
<repo>/data/ask/keyword_index

OUTPUT FILES
------------
records.jsonl                one record per row, in row order (identical in
                              shape to vector_base.py's records.jsonl)
corpus_tokens.jsonl           one JSON array of tokens per row, same row order
keyword_index_manifest.json

INSTALL
-------
python3 -m pip install rank-bm25

RUN
---
python3 keyword_index.py
python3 keyword_index.py --datasets rag
python3 keyword_index.py --no-deduplicate
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi

sys.path.insert(
    0, str(Path(__file__).resolve().parent.parent / "vector_database")
)
sys.path.insert(
    0, str(Path(__file__).resolve().parent.parent.parent.parent / "source")
)

import vector_base as vb  # noqa: E402
from common import DATA_ROOT  # noqa: E402

EMBEDDING_FOLDER = vb.EMBEDDING_FOLDER
CHUNKED_FOLDER = vb.CHUNKED_FOLDER

OUTPUT_FOLDER = DATA_ROOT / "keyword_index"

RECORD_FILE_NAME = vb.RECORD_FILE_NAME
TOKENS_FILE_NAME = "corpus_tokens.jsonl"
MANIFEST_FILE_NAME = "keyword_index_manifest.json"

DATASET_NAMES = vb.DATASET_NAMES
SOURCE_LABEL = vb.SOURCE_LABEL

DEFAULT_TOP_K = vb.DEFAULT_TOP_K

# Casefold first, then split on word-boundary runs of word characters,
# apostrophes and hyphens, so "Newton's" and "long-term" stay single tokens.
# No stemming, no stopword removal: BM25's own term-frequency weighting
# already discounts words that appear in nearly every record, and keeping
# tokenization this plain and explicit makes a result easy to explain by
# just looking at which words it shares with the query.
TOKEN_PATTERN = re.compile(r"\b[\w'-]+\b", re.UNICODE)


def tokenize(text: str) -> list[str]:
    r"""
    PURPOSE
    -------
    Split text into lowercase tokens for BM25 scoring.

    PARAMETERS
    ----------
    text   Text to tokenize.

    RETURNS
    -------
    The list of tokens found, in order; empty if text is empty.

    EXAMPLES
    --------
    >>> tokenize("Newton's Second Law: F = ma")
    ["newton's", 'second', 'law', 'f', 'ma']
    >>> tokenize("long-term memory")
    ['long-term', 'memory']
    >>> tokenize("")
    []
    >>> tokenize(None)
    []
    """
    if not text:
        return []

    return TOKEN_PATTERN.findall(str(text).casefold())


class KeywordIndex:
    r"""
    PURPOSE
    -------
    An in-memory BM25 view of a keyword index folder: a row-aligned list of
    records, their tokenized text, and the manifest describing how the
    index was built. This is the object that both builds an index (via
    ``build_keyword_index``) and is reloaded to search one (via
    ``KeywordIndex.load``).

    ATTRIBUTES
    ----------
    records        List of record dicts, aligned by row with corpus_tokens.
    corpus_tokens  List of token lists, aligned by row with records.
    manifest       Dict describing how the index was built.
    bm25           The rank_bm25.BM25Okapi built from corpus_tokens, or None
                   if the index is empty.

    EXAMPLES
    --------
    >>> records = [
    ...     {"row": 0, "dataset": "rag", "topic": "Heat", "text": "Heat flows from hot to cold."},
    ...     {"row": 1, "dataset": "llm", "topic": "Light", "text": "Light travels in straight lines."},
    ... ]
    >>> corpus_tokens = [tokenize(r["text"]) for r in records]
    >>> index = KeywordIndex(records, corpus_tokens, {"program": "stub"})
    >>> len(index)
    2
    >>> results = index.search("heat", top_k=1)
    >>> results[0].rank, results[0].row, results[0].text
    (1, 0, 'Heat flows from hot to cold.')
    >>> [r.record["topic"] for r in index.search("light", top_k=2, filters={"dataset": "llm"})]
    ['Light']
    >>> index.search("heat", top_k=2, filters={"topic": "nothing"})
    []
    """

    def __init__(
        self,
        records: list[dict[str, Any]],
        corpus_tokens: list[list[str]],
        manifest: dict[str, Any] | None = None,
    ):
        if len(records) != len(corpus_tokens):
            raise ValueError(
                f"{len(records)} record(s) but {len(corpus_tokens)} "
                "tokenized row(s)."
            )

        self.records = records
        self.corpus_tokens = corpus_tokens
        self.manifest = manifest or {}
        # BM25Okapi divides by the number of distinct terms it saw, so a
        # non-empty corpus of entirely empty token lists (every record had
        # no text to join) would still blow up with ZeroDivisionError.
        self.bm25 = BM25Okapi(corpus_tokens) if any(corpus_tokens) else None

    def __len__(self) -> int:
        """
        PURPOSE
        -------
        Number of records in the index.
        """
        return len(self.records)

    def search(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        filters: dict[str, Any] | None = None,
    ) -> list[vb.SearchResult]:
        """
        PURPOSE
        -------
        Return the ``top_k`` records that best match ``query`` by BM25
        score, optionally filtered.

        PARAMETERS
        ----------
        query     Text to search for.
        top_k     Maximum number of results to return.
        filters   Dict of field name to wanted value (or list of values),
                  or None for no filtering.

        RETURNS
        -------
        Up to top_k SearchResult objects, best match first. Mirrors
        vector_base.VectorBase.search's return shape exactly, so a caller
        can treat this and a vector search's results uniformly.
        """
        if len(self) == 0 or top_k <= 0 or self.bm25 is None:
            return []

        scores = self.bm25.get_scores(tokenize(query))

        candidates = [
            row
            for row in range(len(self))
            if vb.matches_filters(self.records[row], filters)
        ]

        if not candidates:
            return []

        chosen = sorted(candidates, key=lambda row: -scores[row])[:top_k]

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
        Write the index to ``folder`` as records.jsonl, corpus_tokens.jsonl
        and a manifest.

        PARAMETERS
        ----------
        folder   Folder to write the index's files into.

        RETURNS
        -------
        The manifest that was written, with build statistics merged in.
        """
        folder.mkdir(parents=True, exist_ok=True)

        vb.atomic_write_text(
            folder / RECORD_FILE_NAME,
            "".join(
                json.dumps(record, ensure_ascii=False) + "\n"
                for record in self.records
            ),
        )
        vb.atomic_write_text(
            folder / TOKENS_FILE_NAME,
            "".join(
                json.dumps(tokens, ensure_ascii=False) + "\n"
                for tokens in self.corpus_tokens
            ),
        )

        manifest = dict(self.manifest)
        manifest.update(
            {
                "record_count": len(self),
                "record_file": str(folder / RECORD_FILE_NAME),
                "tokens_file": str(folder / TOKENS_FILE_NAME),
            }
        )

        vb.atomic_write_text(
            folder / MANIFEST_FILE_NAME,
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        )

        self.manifest = manifest

        return manifest

    @classmethod
    def load(cls, folder: Path) -> KeywordIndex:
        """
        PURPOSE
        -------
        Open a keyword index folder, rebuilding BM25Okapi from the stored
        tokens. The BM25Okapi object itself is never pickled: rebuilding it
        from plain JSON tokens is more portable and avoids pickle version
        fragility, matching this repo's plain-files-no-database philosophy.

        PARAMETERS
        ----------
        folder   Folder previously written by ``save``.

        RETURNS
        -------
        A KeywordIndex over the folder's records, tokens and manifest.

        RAISES
        ------
        FileNotFoundError   The folder has no keyword index files in it.
        """
        record_path = folder / RECORD_FILE_NAME
        tokens_path = folder / TOKENS_FILE_NAME
        manifest_path = folder / MANIFEST_FILE_NAME

        if not record_path.exists() or not tokens_path.exists():
            raise FileNotFoundError(
                f"No keyword index found in {folder}. Run keyword_index.py first."
            )

        records = list(vb.read_jsonl(record_path))

        corpus_tokens: list[list[str]] = []

        with tokens_path.open("r", encoding="utf-8") as input_file:
            for line in input_file:
                line = line.strip()

                if not line:
                    continue

                corpus_tokens.append(json.loads(line))

        manifest: dict[str, Any] = {}

        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        return cls(records, corpus_tokens, manifest)


def build_keyword_index(
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
    Assemble the embedding program's metadata (text, not vectors) into a
    BM25 keyword index folder and return its manifest. Reuses
    vector_base.py's dataset loading, text join, record building and
    deduplication, so the records here are identical in order, content and
    ids to vector_base.build_vector_base's output over the same inputs.

    PARAMETERS
    ----------
    embedding_folder   Folder holding the .npy vectors and metadata (the
                        vectors are loaded only so vector_base.load_dataset
                        can validate them; their values are discarded).
    chunked_folder     Folder holding the passages, for the text join.
    output_folder      Folder to write the keyword index into.
    datasets           Which embedded datasets to include.
    deduplicate_rows   Whether to collapse rows with duplicate text.
    verbose            Whether to print progress as the index is built.

    RETURNS
    -------
    The manifest describing the built keyword index.

    RAISES
    ------
    FileNotFoundError   embedding_folder does not exist.
    ValueError           No records were found across the requested
                          datasets.

    EXAMPLES
    --------
    >>> import tempfile
    >>> with tempfile.TemporaryDirectory() as folder:
    ...     root = Path(folder)
    ...     embedded, chunked, index_folder = root / "emb", root / "chunk", root / "idx"
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
    ...     manifest = build_keyword_index(embedded, chunked, index_folder,
    ...                                     datasets=("rag",), verbose=False)
    ...     written = sorted(p.name for p in index_folder.iterdir())
    ...     loaded = KeywordIndex.load(index_folder)
    >>> manifest["record_count"], manifest["records_with_text"]
    (2, 2)
    >>> written
    ['corpus_tokens.jsonl', 'keyword_index_manifest.json', 'records.jsonl']
    >>> loaded.search("heat", top_k=1)[0].text
    'Heat flows from hot to cold.'
    >>> build_keyword_index(Path("/definitely/not/here"), Path("/tmp"), Path("/tmp/x"))
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

    join_lookup = vb.build_join_lookup(chunked_folder)

    if not join_lookup:
        announce(
            f"Warning: no passages found in {chunked_folder}. "
            "Records will carry identifiers but no text."
        )

    all_records: list[dict[str, Any]] = []
    dataset_results: list[dict[str, Any]] = []

    for dataset_name in datasets:
        vectors, metadata_records = vb.load_dataset(embedding_folder, dataset_name)

        if len(vectors) == 0:
            dataset_results.append(
                {
                    "dataset": dataset_name,
                    "status": "skipped",
                    "reason": "No records found for this dataset.",
                    "record_count": 0,
                }
            )
            announce(f"{dataset_name}: skipped (no records)")
            continue

        start_row = len(all_records)

        all_records.extend(
            vb.build_record(metadata, dataset_name, start_row + offset, join_lookup)
            for offset, metadata in enumerate(metadata_records)
        )

        dataset_results.append(
            {
                "dataset": dataset_name,
                "status": "included",
                "record_count": len(metadata_records),
            }
        )

        announce(f"{dataset_name}: {len(metadata_records)} record(s)")

    if not all_records:
        raise ValueError(
            f"No records were found in {embedding_folder}. "
            "Run embedding.py first."
        )

    removed = 0

    if deduplicate_rows:
        # Only the row count needs to line up with all_records; the actual
        # vector values are irrelevant to deduplicate(), which keys solely
        # on each record's text_sha256. A dummy single-column placeholder
        # avoids forcing datasets with different embedding dimensions (or
        # none at all) to agree on a width, which vector_base.py must do
        # but a keyword index never needs to.
        placeholder = np.zeros((len(all_records), 1), dtype=np.float32)
        _, all_records, removed = vb.deduplicate(placeholder, all_records)
        announce(f"Deduplicated: {removed} row(s) removed")

    corpus_tokens = [tokenize(record.get("text") or "") for record in all_records]

    manifest = {
        "program": "SoakinGarri Keyword Index Program",
        "backend": "rank_bm25.BM25Okapi",
        "embedding_folder": str(embedding_folder),
        "chunked_folder": str(chunked_folder),
        "output_folder": str(output_folder),
        "source_label": SOURCE_LABEL,
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

    index = KeywordIndex(all_records, corpus_tokens, manifest)
    manifest = index.save(output_folder)

    if verbose:
        print(
            f"\nKeyword index built.\n"
            f"Records: {manifest['record_count']}\n"
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
        description="Assemble embedding outputs into a searchable BM25 keyword index."
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
        help="folder to write the keyword index into",
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
    Build the keyword index using the options given on the command line.
    """
    options = parse_arguments()

    build_keyword_index(
        embedding_folder=options.embeddings,
        chunked_folder=options.chunked,
        output_folder=options.output,
        datasets=tuple(options.datasets),
        deduplicate_rows=options.deduplicate,
    )


if __name__ == "__main__":
    main()
