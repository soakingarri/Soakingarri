#!/usr/bin/env python3

"""
SoakinGarri Shared Pipeline Utilities

PURPOSE
-------
Helpers shared by the data_cleaning, data_chunking, embedding, tokenization,
and summarization programs, plus the vector_database and retrieval programs
under data/ask/. Centralizes three things that used to be copy-pasted
(and quietly drifting apart) across those files: text normalization,
deterministic id generation, and resolving the project's data folders
without hardcoding a developer's home directory in every script.

USAGE
-----
Each consumer lives in its own folder with no shared package structure, so
import this the same way data/ask/retrieval/retrieval.py already imports
data/ask/vector_database/vector_base.py: insert this file's directory into
sys.path, then import it as a plain top-level module.

    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    from common import DATA_ROOT, normalize_text, stable_id, load_documents
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data" / "ask"

SPACE_PATTERN = re.compile(r"\s+")


def normalize_text(value: Any) -> str:
    r"""
    PURPOSE
    -------
    Collapse whitespace to single spaces and strip invisible unicode
    characters (soft hyphen, zero-width space, BOM); ``None`` becomes
    ``""``, and non-string values are stringified first.

    PARAMETERS
    ----------
    value   Value to normalize; converted via ``str`` first.

    RETURNS
    -------
    The normalized string.

    EXAMPLES
    --------
    >>> normalize_text("  hello \n\t world​ ")
    'hello world'
    >>> normalize_text(None)
    ''
    >>> normalize_text(42)
    '42'
    """
    if value is None:
        return ""

    text = str(value)
    text = text.replace("­", "")
    text = text.replace("​", "")
    text = text.replace("﻿", "")
    text = SPACE_PATTERN.sub(" ", text)

    return text.strip()


def stable_id(prefix: str, *values: str) -> str:
    r"""
    PURPOSE
    -------
    Build a deterministic id from a prefix and a sequence of values: the
    same values in always produce the same id out, and order matters.
    Use this instead of a random id (``uuid4``) anywhere the id needs to
    survive a re-run unchanged — for example, re-cleaning the same source
    file should produce the same document/chapter/paragraph/chunk ids, so
    a vector store can be updated in place instead of growing a duplicate
    entry on every run.

    PARAMETERS
    ----------
    prefix   Short label prepended to the generated id (e.g. "DOC", "P").
    values   The values to hash together, in order.

    RETURNS
    -------
    A string of the form "{prefix}-{24-character sha256 hex digest}".

    EXAMPLES
    --------
    >>> stable_id("DOC", "1", "hello") == stable_id("DOC", "1", "hello")
    True
    >>> stable_id("DOC", "a", "b") != stable_id("DOC", "b", "a")
    True
    >>> stable_id("DOC", "x")
    'DOC-2d711642b726b04401627ca9'
    """
    content = "\x1f".join(values)
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()[:24]

    return f"{prefix}-{digest}"


def load_documents(path: Path) -> list[dict[str, Any]]:
    r"""
    PURPOSE
    -------
    Read a .json or .jsonl file into a list of document dictionaries,
    silently dropping any top-level entries that are not JSON objects.

    PARAMETERS
    ----------
    path   Path to the .json or .jsonl file to read.

    RETURNS
    -------
    The list of document dictionaries found in the file; empty for an
    unsupported extension.

    RAISES
    ------
    ValueError   A .jsonl file has a line that is not valid JSON.

    EXAMPLES
    --------
    >>> import tempfile
    >>> with tempfile.TemporaryDirectory() as folder:
    ...     path = Path(folder) / "a.jsonl"
    ...     _ = path.write_text('{"a": 1}\n\n{"b": 2}\n', encoding="utf-8")
    ...     load_documents(path)
    [{'a': 1}, {'b': 2}]
    >>> with tempfile.TemporaryDirectory() as folder:
    ...     path = Path(folder) / "a.json"
    ...     _ = path.write_text('[{"a": 1}, "junk"]', encoding="utf-8")
    ...     load_documents(path)
    [{'a': 1}]
    >>> load_documents(Path("notes.txt"))
    []
    """
    extension = path.suffix.casefold()

    if extension == ".json":
        with path.open("r", encoding="utf-8") as input_file:
            data = json.load(input_file)

        if isinstance(data, dict):
            return [data]

        if isinstance(data, list):
            return [item for item in data if isinstance(item, dict)]

        return []

    if extension == ".jsonl":
        documents: list[dict[str, Any]] = []

        with path.open("r", encoding="utf-8") as input_file:
            for line_number, line in enumerate(input_file, start=1):
                line = line.strip()

                if not line:
                    continue

                try:
                    item = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"Invalid JSON in {path.name} on line "
                        f"{line_number}: {error}"
                    ) from error

                if isinstance(item, dict):
                    documents.append(item)

        return documents

    return []
