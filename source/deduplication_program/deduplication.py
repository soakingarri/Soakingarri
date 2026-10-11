#!/usr/bin/env python3

"""
SoakinGarri Cross-Document Deduplication Program

PURPOSE
-------
Remove duplicate paragraphs that appear across *different* cleaned
documents -- the same passage showing up in two different books, or in a
book and a scraped web page. This runs one stage after data_cleaning.py
(which only removes *consecutive* duplicate paragraphs within a single
document) and one stage before chunking: it works on whole paragraphs,
not embedded vectors, so duplicates never reach the chunker, embedder, or
vector store in the first place.

Two kinds of duplicates are caught:

    * Exact duplicates   Identical text (after normalization). Detected
                          by comparing sha256 hashes of normalized text.
    * Near duplicates    The same passage with minor rewording or OCR
                          noise. Detected with MinHash/MinHashLSH over
                          word shingles.

In both cases the first occurrence (in document/chapter/paragraph order)
is kept and every later repeat is removed.

Input:
    <repo>/data/ask/cleaned_data

Output:
    <repo>/data/ask/deduplicated_data

Install dependencies:
    python3 -m pip install datasketch

Run:
    python3 deduplication.py
    python3 deduplication.py --input /path/to/cleaned --output /path/to/out
    python3 deduplication.py --threshold 0.9

Notes:
    Output files reuse the input cleaned_data JSON structure (a
    document dict with "chapters", each chapter with "paragraphs"); only
    duplicate paragraphs are removed. A chapter left with no paragraphs
    is dropped, and a document left with no chapters is dropped, rather
    than raising -- a book that is wholly a duplicate of another book is
    simply absent from the output, not an error.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common import DATA_ROOT, load_documents, normalize_text  # noqa: E402

from datasketch import MinHash, MinHashLSH  # noqa: E402

INPUT_FOLDER = DATA_ROOT / "cleaned_data"
OUTPUT_FOLDER = DATA_ROOT / "deduplicated_data"

DEFAULT_SHINGLE_SIZE = 5
DEFAULT_NUM_PERM = 128
DEFAULT_NEAR_DUPLICATE_THRESHOLD = 0.85


def paragraph_shingles(
    text: str,
    shingle_size: int = DEFAULT_SHINGLE_SIZE,
) -> set[str]:
    r"""
    PURPOSE
    -------
    Split a paragraph's text into overlapping word k-shingles, for use as
    MinHash input. The text is run through common.normalize_text first,
    so whitespace differences and invisible characters never cause a
    shingle mismatch on their own.

    PARAMETERS
    ----------
    text           The paragraph text to shingle.
    shingle_size   Number of words per shingle.

    RETURNS
    -------
    The set of space-joined, overlapping shingle_size-word windows. A
    text with fewer words than shingle_size produces one shingle holding
    all of its words; empty (or whitespace-only) text produces an empty
    set.

    EXAMPLES
    --------
    >>> sorted(paragraph_shingles("one two three four five six", 5))
    ['one two three four five', 'two three four five six']
    >>> paragraph_shingles("one two")
    {'one two'}
    >>> paragraph_shingles("   ")
    set()
    """
    words = [word for word in normalize_text(text).split(" ") if word]

    if len(words) < shingle_size:
        return {" ".join(words)} if words else set()

    return {
        " ".join(words[index : index + shingle_size])
        for index in range(len(words) - shingle_size + 1)
    }


def build_minhash(text: str, num_perm: int = DEFAULT_NUM_PERM) -> MinHash:
    r"""
    PURPOSE
    -------
    Build a MinHash signature from a paragraph's word shingles.

    PARAMETERS
    ----------
    text       The paragraph text to hash.
    num_perm   Number of permutation functions (MinHash signature size);
               higher is more accurate but slower and larger.

    RETURNS
    -------
    The MinHash built from the text's shingles (see
    :func:`paragraph_shingles`).

    EXAMPLES
    --------
    >>> first = build_minhash("the quick brown fox jumps over the lazy dog")
    >>> second = build_minhash("the quick brown fox jumps over the lazy dog")
    >>> first.jaccard(second)
    1.0
    >>> type(first).__name__
    'MinHash'
    """
    minhash = MinHash(num_perm=num_perm)

    for shingle in paragraph_shingles(text):
        minhash.update(shingle.encode("utf-8"))

    return minhash


def find_exact_duplicates(
    documents: list[dict[str, Any]],
) -> dict[str, str]:
    r"""
    PURPOSE
    -------
    Find paragraphs whose normalized text is byte-for-byte identical to
    an earlier paragraph's, across every document. Walks paragraphs in
    document/chapter/paragraph order, hashing each one's normalized text
    with sha256; a paragraph whose hash was already seen is a duplicate
    of whichever paragraph_id first produced that hash.

    PARAMETERS
    ----------
    documents   Cleaned documents to scan, each with a "chapters" list of
                chapters, each with a "paragraphs" list of {"paragraph_id",
                "text"} dicts.

    RETURNS
    -------
    A dict mapping each duplicate paragraph_id to the paragraph_id of its
    first (kept) occurrence. The first occurrence of any text is never a
    key in this dict -- only later repeats are.

    EXAMPLES
    --------
    >>> documents = [
    ...     {"chapters": [{"paragraphs": [
    ...         {"paragraph_id": "P-1", "text": "Hello world."},
    ...         {"paragraph_id": "P-2", "text": "Hello   world."},
    ...     ]}]},
    ...     {"chapters": [{"paragraphs": [
    ...         {"paragraph_id": "P-3", "text": "Something else entirely."},
    ...         {"paragraph_id": "P-4", "text": "Hello world."},
    ...     ]}]},
    ... ]
    >>> find_exact_duplicates(documents)
    {'P-2': 'P-1', 'P-4': 'P-1'}
    """
    seen_hashes: dict[str, str] = {}
    duplicates: dict[str, str] = {}

    for document in documents:
        for chapter in document.get("chapters", []):
            for paragraph in chapter.get("paragraphs", []):
                paragraph_id = paragraph["paragraph_id"]
                digest = hashlib.sha256(
                    normalize_text(paragraph.get("text")).encode("utf-8")
                ).hexdigest()

                if digest in seen_hashes:
                    duplicates[paragraph_id] = seen_hashes[digest]
                else:
                    seen_hashes[digest] = paragraph_id

    return duplicates


def find_near_duplicates(
    documents: list[dict[str, Any]],
    threshold: float = DEFAULT_NEAR_DUPLICATE_THRESHOLD,
    num_perm: int = DEFAULT_NUM_PERM,
    exclude: set[str] | None = None,
) -> dict[str, str]:
    r"""
    PURPOSE
    -------
    Find paragraphs that are near-duplicates (same passage, minor
    rewording/OCR noise) of an earlier paragraph's, across every
    document, via MinHashLSH. Paragraphs are processed in
    document/chapter/paragraph order: each one is first queried against
    an LSH index of every paragraph inserted so far. If a match comes
    back, the paragraph is recorded as a duplicate of that match (the
    earliest-inserted match, if several come back) and is *not* itself
    inserted into the index -- so a chain of near-duplicate rewordings
    all point back to one true original, rather than to each other.

    Call this only on paragraphs that survived
    :func:`find_exact_duplicates` (via `exclude`), so exact duplicates
    are not redundantly flagged again as near-duplicates too.

    PARAMETERS
    ----------
    documents   Cleaned documents to scan, same shape as in
                :func:`find_exact_duplicates`.
    threshold   Minimum (approximate) Jaccard similarity for two
                paragraphs to count as near-duplicates.
    num_perm    MinHash signature size; see :func:`build_minhash`.
    exclude     paragraph_ids to skip entirely (neither queried nor
                inserted) -- typically the exact-duplicate ids already
                found by :func:`find_exact_duplicates`.

    RETURNS
    -------
    A dict mapping each near-duplicate paragraph_id to the paragraph_id
    of its earliest matching (kept) occurrence.

    EXAMPLES
    --------
    >>> base = (
    ...     "In many African societies time was traditionally understood "
    ...     "through the rhythm of agricultural seasons rather than "
    ...     "through fixed clock hours and elders passed this "
    ...     "understanding down through oral tradition across many "
    ...     "generations of herders and farmers alike who relied on the "
    ...     "position of the sun and the changing of the weather to plan "
    ...     "planting harvesting and the movement of cattle between dry "
    ...     "season and wet season grazing lands throughout the year "
    ...     "while also watching the stars at night to predict the "
    ...     "coming of the rains and to decide when ceremonies marking "
    ...     "the new season should begin for the whole community to "
    ...     "gather and celebrate together near the riverbanks where the "
    ...     "soil was richest and the cattle could graze safely away "
    ...     "from predators that roamed the drier hills further inland "
    ...     "during the hottest months of the dry season every single "
    ...     "year"
    ... )
    >>> reworded = base.replace("understanding", "knowledge", 1)
    >>> unrelated = (
    ...     "A completely different paragraph discussing colonial trade "
    ...     "routes along the coast of West Africa during the nineteenth "
    ...     "century and the economic consequences for local merchants "
    ...     "and port cities that depended heavily on the export of palm "
    ...     "oil rubber and other goods"
    ... )
    >>> documents = [{"chapters": [{"paragraphs": [
    ...     {"paragraph_id": "P-1", "text": base},
    ...     {"paragraph_id": "P-2", "text": reworded},
    ...     {"paragraph_id": "P-3", "text": unrelated},
    ... ]}]}]
    >>> find_near_duplicates(documents)
    {'P-2': 'P-1'}
    """
    exclude = exclude or set()
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    insertion_order: dict[str, int] = {}
    duplicates: dict[str, str] = {}

    for document in documents:
        for chapter in document.get("chapters", []):
            for paragraph in chapter.get("paragraphs", []):
                paragraph_id = paragraph["paragraph_id"]

                if paragraph_id in exclude:
                    continue

                minhash = build_minhash(
                    paragraph.get("text"), num_perm=num_perm
                )
                matches = lsh.query(minhash)

                if matches:
                    duplicates[paragraph_id] = min(
                        matches, key=lambda key: insertion_order[key]
                    )
                else:
                    lsh.insert(paragraph_id, minhash)
                    insertion_order[paragraph_id] = len(insertion_order)

    return duplicates


def prune_duplicate_paragraphs(
    document: dict[str, Any],
    removed_ids: set[str],
) -> dict[str, Any] | None:
    r"""
    PURPOSE
    -------
    Remove duplicate paragraphs from one document's chapters. A chapter
    left with no paragraphs is dropped; if that leaves the document with
    no chapters, the whole document is dropped.

    PARAMETERS
    ----------
    document      A cleaned document dict with a "chapters" list.
    removed_ids   paragraph_ids to remove wherever they appear.

    RETURNS
    -------
    A new document dict with the duplicate paragraphs removed, or None
    if nothing is left of it.

    EXAMPLES
    --------
    >>> document = {"document_id": "DOC-1", "chapters": [
    ...     {"chapter_id": "CH-1", "paragraphs": [
    ...         {"paragraph_id": "P-1", "text": "a"},
    ...         {"paragraph_id": "P-2", "text": "b"},
    ...     ]},
    ...     {"chapter_id": "CH-2", "paragraphs": [
    ...         {"paragraph_id": "P-3", "text": "c"},
    ...     ]},
    ... ]}
    >>> pruned = prune_duplicate_paragraphs(document, {"P-2", "P-3"})
    >>> [c["chapter_id"] for c in pruned["chapters"]]
    ['CH-1']
    >>> [p["paragraph_id"] for c in pruned["chapters"] for p in c["paragraphs"]]
    ['P-1']
    >>> prune_duplicate_paragraphs(document, {"P-1", "P-2", "P-3"}) is None
    True
    """
    new_chapters: list[dict[str, Any]] = []

    for chapter in document.get("chapters", []):
        kept_paragraphs = [
            paragraph
            for paragraph in chapter.get("paragraphs", [])
            if paragraph["paragraph_id"] not in removed_ids
        ]

        if kept_paragraphs:
            new_chapter = dict(chapter)
            new_chapter["paragraphs"] = kept_paragraphs
            new_chapters.append(new_chapter)

    if not new_chapters:
        return None

    new_document = dict(document)
    new_document["chapters"] = new_chapters
    return new_document


def discover_cleaned_files(input_folder: Path) -> list[Path]:
    r"""
    PURPOSE
    -------
    List cleaned document files in a folder (the output of
    data_cleaning.py), skipping anything else -- including its own
    cleaning_manifest.json.

    PARAMETERS
    ----------
    input_folder   Root folder to search for cleaned document files.

    RETURNS
    -------
    The sorted list of "*_cleaned.json" / "*_cleaned.jsonl" file paths
    found.

    EXAMPLES
    --------
    >>> import tempfile
    >>> with tempfile.TemporaryDirectory() as folder:
    ...     root = Path(folder)
    ...     _ = (root / "book_cleaned.json").write_text("{}", encoding="utf-8")
    ...     _ = (root / "cleaning_manifest.json").write_text("{}", encoding="utf-8")
    ...     [p.name for p in discover_cleaned_files(root)]
    ['book_cleaned.json']
    """
    return sorted(
        path
        for pattern in ("*_cleaned.json", "*_cleaned.jsonl")
        for path in input_folder.rglob(pattern)
        if path.is_file()
    )


def deduplicate_corpus(
    input_folder: Path = INPUT_FOLDER,
    output_folder: Path = OUTPUT_FOLDER,
    near_duplicate_threshold: float = DEFAULT_NEAR_DUPLICATE_THRESHOLD,
) -> dict[str, Any]:
    r"""
    PURPOSE
    -------
    Run the full cross-document deduplication pipeline: load every
    cleaned document in input_folder, find exact and near duplicate
    paragraphs across the whole corpus, write a deduplicated copy of
    each document (minus its duplicate paragraphs, and minus any chapter
    or document that is left empty) to output_folder, and write a
    manifest of what was removed and why.

    PARAMETERS
    ----------
    input_folder               Folder of cleaned "*_cleaned.json(l)"
                                files (the output of data_cleaning.py).
    output_folder               Folder to write deduplicated files and
                                the manifest to.
    near_duplicate_threshold   Minimum (approximate) Jaccard similarity
                                for two paragraphs to count as
                                near-duplicates; see
                                :func:`find_near_duplicates`.

    RETURNS
    -------
    The manifest dict (also written to
    output_folder/"deduplication_manifest.json"): input/output folder
    paths, total_paragraphs seen, exact_duplicates_removed and
    near_duplicates_removed counts, and a "removed" list of
    {"paragraph_id", "duplicate_of", "reason"} entries.

    EXAMPLES
    --------
    >>> import tempfile
    >>> with tempfile.TemporaryDirectory() as folder:
    ...     root = Path(folder)
    ...     inp, out = root / "cleaned", root / "deduped"
    ...     inp.mkdir()
    ...     book_one = {"document_id": "DOC-1", "chapters": [{"chapter_id": "CH-1",
    ...         "paragraphs": [
    ...             {"paragraph_id": "P-1", "text": "A shared passage repeated in two books."},
    ...             {"paragraph_id": "P-2", "text": "Only in the first book."},
    ...         ]}]}
    ...     book_two = {"document_id": "DOC-2", "chapters": [{"chapter_id": "CH-2",
    ...         "paragraphs": [
    ...             {"paragraph_id": "P-3", "text": "A shared passage repeated in two books."},
    ...         ]}]}
    ...     _ = (inp / "one_cleaned.json").write_text(json.dumps(book_one), encoding="utf-8")
    ...     _ = (inp / "two_cleaned.json").write_text(json.dumps(book_two), encoding="utf-8")
    ...     manifest = deduplicate_corpus(inp, out)
    ...     remaining = json.loads((out / "one_cleaned.json").read_text(encoding="utf-8"))
    ...     (manifest["total_paragraphs"], manifest["exact_duplicates_removed"],
    ...      manifest["removed"],
    ...      sorted(p.name for p in out.glob("*_cleaned.json")),
    ...      [p["paragraph_id"] for p in remaining["chapters"][0]["paragraphs"]])
    (3, 1, [{'paragraph_id': 'P-3', 'duplicate_of': 'P-1', 'reason': 'exact'}], ['one_cleaned.json'], ['P-1', 'P-2'])
    """
    files = discover_cleaned_files(input_folder)

    documents: list[dict[str, Any]] = []
    source_files: list[Path] = []

    for path in files:
        for document in load_documents(path):
            documents.append(document)
            source_files.append(path)

    total_paragraphs = sum(
        len(chapter.get("paragraphs", []))
        for document in documents
        for chapter in document.get("chapters", [])
    )

    exact_duplicates = find_exact_duplicates(documents)
    near_duplicates = find_near_duplicates(
        documents,
        threshold=near_duplicate_threshold,
        exclude=set(exact_duplicates),
    )
    removed_ids = set(exact_duplicates) | set(near_duplicates)

    removed = [
        {"paragraph_id": paragraph_id, "duplicate_of": original, "reason": "exact"}
        for paragraph_id, original in exact_duplicates.items()
    ] + [
        {"paragraph_id": paragraph_id, "duplicate_of": original, "reason": "near"}
        for paragraph_id, original in near_duplicates.items()
    ]

    survivors_by_file: dict[Path, list[dict[str, Any]]] = {}

    for path, document in zip(source_files, documents):
        pruned = prune_duplicate_paragraphs(document, removed_ids)

        if pruned is not None:
            survivors_by_file.setdefault(path, []).append(pruned)

    output_folder.mkdir(parents=True, exist_ok=True)

    for path in files:
        survivors = survivors_by_file.get(path, [])

        if not survivors:
            continue

        output_path = output_folder / path.name

        if output_path.suffix.casefold() == ".jsonl":
            temporary_path = output_path.with_suffix(".jsonl.tmp")

            with temporary_path.open("w", encoding="utf-8") as output_file:
                for survivor in survivors:
                    output_file.write(
                        json.dumps(survivor, ensure_ascii=False) + "\n"
                    )
        else:
            payload = survivors[0] if len(survivors) == 1 else survivors
            temporary_path = output_path.with_suffix(".json.tmp")

            with temporary_path.open("w", encoding="utf-8") as output_file:
                json.dump(payload, output_file, ensure_ascii=False, indent=2)

        temporary_path.replace(output_path)

    manifest = {
        "input_folder": str(input_folder),
        "output_folder": str(output_folder),
        "total_paragraphs": total_paragraphs,
        "exact_duplicates_removed": len(exact_duplicates),
        "near_duplicates_removed": len(near_duplicates),
        "removed": removed,
    }

    manifest_path = output_folder / "deduplication_manifest.json"
    temporary_manifest_path = manifest_path.with_suffix(".json.tmp")

    with temporary_manifest_path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)

    temporary_manifest_path.replace(manifest_path)

    return manifest


def parse_arguments(
    argument_list: list[str] | None = None,
) -> argparse.Namespace:
    r"""
    PURPOSE
    -------
    Parse the command line.

    PARAMETERS
    ----------
    argument_list   Arguments to parse; None reads from sys.argv.

    RETURNS
    -------
    The parsed arguments namespace.

    EXAMPLES
    --------
    >>> parse_arguments([]).threshold
    0.85
    >>> parse_arguments(["--threshold", "0.9"]).threshold
    0.9
    """
    parser = argparse.ArgumentParser(
        description=(
            "Remove exact and near-duplicate paragraphs across cleaned "
            "documents."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=INPUT_FOLDER,
        help="folder of cleaned JSON files",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_FOLDER,
        help="folder for deduplicated JSON files",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_NEAR_DUPLICATE_THRESHOLD,
        help="Jaccard similarity threshold for near-duplicate detection",
    )

    return parser.parse_args(argument_list)


def main() -> None:
    """
    PURPOSE
    -------
    Deduplicate the configured folders using the options given on the
    command line, and print a summary.
    """
    options = parse_arguments()

    manifest = deduplicate_corpus(
        input_folder=options.input,
        output_folder=options.output,
        near_duplicate_threshold=options.threshold,
    )

    print(
        "Deduplication completed.\n"
        f"Total paragraphs seen: {manifest['total_paragraphs']}\n"
        f"Exact duplicates removed: {manifest['exact_duplicates_removed']}\n"
        f"Near duplicates removed: {manifest['near_duplicates_removed']}\n"
        f"Output folder: {manifest['output_folder']}"
    )


if __name__ == "__main__":
    main()
