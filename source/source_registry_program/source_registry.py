#!/usr/bin/env python3

"""
SoakinGarri Source Registry Program

PURPOSE
-------
A catalogue of every source the pipeline ingests: title, author, the
author's country, an African-authored flag, year, publisher, language,
ISBN, and licence/permission status.

THE HARD RULE: nothing is treated as cleared for use unless its registry
entry's licence_status is explicitly "cleared". A source with no registry
entry at all, or an entry whose licence_status is "pending" or "rejected",
must never be treated as cleared -- see is_cleared and filter_cleared_files,
which encode exactly this rule and nothing looser.

This is a standalone CRUD tool over the registry file. It is deliberately
NOT wired into data_cleaning_program/data_cleaning.py's pipeline yet; that
integration is a follow-up task, not part of this module.

INPUT
-----
None required to run the CRUD tool itself. The `check` subcommand takes
--raw-data pointing at a folder of candidate source files (default:
<repo>/data/ask/raw_data).

OUTPUT
------
<repo>/data/ask/metadata/source_registry.json

INSTALL
-------
No extra dependencies beyond the standard library and common.py.

RUN
---
python3 source_registry.py add --title "Time and Social Anthropology" \
    --author "Alfred Gell" --author-country "United Kingdom" \
    --year 1998 --publisher Berg --language English \
    --licence-status pending --licence-notes "Awaiting publisher reply." \
    --source-file "Gell_Alfred_1998_2000_Time_and_Social_Anthropology.pdf"

python3 source_registry.py list
python3 source_registry.py list --json
python3 source_registry.py check --raw-data /path/to/raw_data

NOTES
-----
- Storage is a single JSON object keyed by source_id, written atomically
  via a temp-file-then-rename, same pattern as
  data_chunking_programs/data_chunking.py's write_jsonl and
  data/ask/vector_database/vector_base.py's atomic_write_text.
- make_source_id is deterministic from title+author, so re-registering the
  same source is idempotent: it overwrites the existing entry instead of
  creating a duplicate.
- Labelling a source here does not itself grant permission to use it;
  licence_status should only be set to "cleared" once that permission is
  actually confirmed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common import DATA_ROOT, normalize_text, stable_id  # noqa: E402

REGISTRY_FOLDER = DATA_ROOT / "metadata"
REGISTRY_FILE = REGISTRY_FOLDER / "source_registry.json"
RAW_DATA_FOLDER = DATA_ROOT / "raw_data"

LICENCE_STATUSES = {"cleared", "pending", "rejected"}


@dataclass(slots=True)
class SourceRecord:
    """
    PURPOSE
    -------
    One catalogued source: who wrote it, where, when, and whether the
    pipeline has permission to use it. A record's `licence_status` is the
    single source of truth for whether its `source_file` may be treated
    as cleared for use -- see is_cleared.

    ATTRIBUTES
    ----------
    source_id             Deterministic id derived from title+author (see
                           make_source_id).
    title                 The source's title.
    author                The source's author.
    author_country        The author's country.
    is_african_authored   Whether the author is African.
    year                  Publication year, if known.
    publisher             The source's publisher.
    language              The source's language.
    isbn                  The source's ISBN, if known.
    licence_status        One of "cleared", "pending", "rejected".
    licence_notes         Free-text notes on the licence/permission.
    source_file           Name of the file this record corresponds to,
                           matching what appears under data/ask/raw_data.
    date_added            ISO date string the record was added/updated.
    """

    source_id: str
    title: str
    author: str
    author_country: str
    is_african_authored: bool
    year: int | None
    publisher: str
    language: str
    isbn: str | None
    licence_status: str
    licence_notes: str
    source_file: str
    date_added: str


def make_source_id(title: str, author: str) -> str:
    r"""
    PURPOSE
    -------
    Build a deterministic source id from a title and author, via
    common.stable_id. The same title+author always produces the same id,
    so re-registering the same source updates its existing registry entry
    instead of creating a duplicate.

    PARAMETERS
    ----------
    title    The source's title.
    author   The source's author.

    RETURNS
    -------
    A stable "SRC-..." id string.

    EXAMPLES
    --------
    >>> make_source_id("Time and Social Anthropology", "Alfred Gell")
    'SRC-72cbbed96449e0e05d385723'
    >>> make_source_id("Time and Social Anthropology", "Alfred Gell") == \
    ...     make_source_id("Time and Social Anthropology", "Alfred Gell")
    True
    >>> make_source_id("A", "B") != make_source_id("B", "A")
    True
    """
    return stable_id("SRC", normalize_text(title), normalize_text(author))


def load_registry(path: Path) -> dict[str, SourceRecord]:
    r"""
    PURPOSE
    -------
    Load the source registry from a JSON file, keyed by source_id.
    Returns an empty registry when the file does not exist yet, which is
    the normal state on the very first run.

    PARAMETERS
    ----------
    path   Path to the registry JSON file.

    RETURNS
    -------
    Dict mapping source_id to its SourceRecord.

    EXAMPLES
    --------
    >>> load_registry(Path("/no/such/source_registry.json"))
    {}
    >>> import tempfile
    >>> with tempfile.TemporaryDirectory() as folder:
    ...     path = Path(folder) / "registry.json"
    ...     registry: dict[str, SourceRecord] = {}
    ...     _ = add_or_update_source(
    ...         registry, title="A", author="X", author_country="Nigeria",
    ...         is_african_authored=True, year=2020, publisher="P",
    ...         language="English", isbn=None, licence_status="cleared",
    ...         licence_notes="", source_file="a.pdf", date_added="2024-01-01",
    ...     )
    ...     save_registry(path, registry)
    ...     loaded = load_registry(path)
    ...     (len(loaded) == len(registry), loaded[next(iter(loaded))].title)
    (True, 'A')
    """
    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8") as input_file:
        payload = json.load(input_file)

    return {
        source_id: SourceRecord(**fields)
        for source_id, fields in payload.items()
    }


def save_registry(path: Path, registry: dict[str, SourceRecord]) -> None:
    r"""
    PURPOSE
    -------
    Atomically write the registry to a JSON file (temp file then
    rename), keyed by source_id with each record's fields as a dict, so
    a reader never sees a partially written file.

    PARAMETERS
    ----------
    path       Destination path for the registry JSON file.
    registry   The registry to write.

    EXAMPLES
    --------
    >>> import tempfile
    >>> with tempfile.TemporaryDirectory() as folder:
    ...     path = Path(folder) / "sub" / "registry.json"
    ...     registry: dict[str, SourceRecord] = {}
    ...     _ = add_or_update_source(
    ...         registry, title="A", author="X", author_country="Nigeria",
    ...         is_african_authored=True, year=2020, publisher="P",
    ...         language="English", isbn=None, licence_status="pending",
    ...         licence_notes="", source_file="a.pdf", date_added="2024-01-01",
    ...     )
    ...     save_registry(path, registry)
    ...     (path.exists(), path.with_suffix(".json.tmp").exists())
    (True, False)
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        source_id: asdict(record)
        for source_id, record in registry.items()
    }

    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)

    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as output_file:
            json.dump(payload, output_file, ensure_ascii=False, indent=2)
            output_file.flush()
            os.fsync(output_file.fileno())

        temporary_path.replace(path)

    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def add_or_update_source(
    registry: dict[str, SourceRecord],
    **fields: Any,
) -> SourceRecord:
    r"""
    PURPOSE
    -------
    Catalogue a source: validate its licence_status, build (or overwrite)
    its SourceRecord via make_source_id, insert it into `registry`, and
    return it. Re-registering the same title+author is idempotent -- the
    existing entry is overwritten in place rather than duplicated, since
    make_source_id always derives the same source_id from the same
    title+author.

    PARAMETERS
    ----------
    registry   The registry dict to insert into (mutated in place).
    fields     The SourceRecord's fields as keyword arguments: title,
               author, author_country, is_african_authored, year,
               publisher, language, isbn, licence_status, licence_notes,
               source_file. date_added is optional and defaults to
               today's date; other omitted fields default to "", None, or
               False as appropriate for their type.

    RETURNS
    -------
    The inserted (or updated) SourceRecord.

    RAISES
    ------
    ValueError   licence_status is missing or not one of LICENCE_STATUSES.

    EXAMPLES
    --------
    >>> registry: dict[str, SourceRecord] = {}
    >>> record = add_or_update_source(
    ...     registry,
    ...     title="Time and Social Anthropology",
    ...     author="Alfred Gell",
    ...     author_country="United Kingdom",
    ...     is_african_authored=False,
    ...     year=1998,
    ...     publisher="Berg",
    ...     language="English",
    ...     isbn="978-0-85496-000-0",
    ...     licence_status="pending",
    ...     licence_notes="Awaiting publisher reply.",
    ...     source_file="Gell_Alfred_1998_2000_Time_and_Social_Anthropology.pdf",
    ...     date_added="2026-10-10",
    ... )
    >>> record.source_id in registry, record.licence_status, len(registry)
    (True, 'pending', 1)
    >>> # re-registering the same title+author overwrites, not duplicates
    >>> updated = add_or_update_source(
    ...     registry, title="Time and Social Anthropology", author="Alfred Gell",
    ...     author_country="United Kingdom", is_african_authored=False, year=1998,
    ...     publisher="Berg", language="English", isbn=None,
    ...     licence_status="cleared", licence_notes="Publisher granted permission.",
    ...     source_file="Gell_Alfred_1998_2000_Time_and_Social_Anthropology.pdf",
    ...     date_added="2026-10-11",
    ... )
    >>> updated.source_id == record.source_id, len(registry), updated.licence_status
    (True, 1, 'cleared')
    >>> add_or_update_source(
    ...     registry, title="X", author="Y", author_country="", is_african_authored=False,
    ...     year=None, publisher="", language="", isbn=None,
    ...     licence_status="unknown", licence_notes="", source_file="x.pdf",
    ... )
    Traceback (most recent call last):
        ...
    ValueError: licence_status must be one of ['cleared', 'pending', 'rejected'], got 'unknown'
    """
    licence_status = fields.get("licence_status")

    if licence_status not in LICENCE_STATUSES:
        raise ValueError(
            "licence_status must be one of "
            f"{sorted(LICENCE_STATUSES)}, got {licence_status!r}"
        )

    title = normalize_text(fields.get("title"))
    author = normalize_text(fields.get("author"))
    source_id = make_source_id(title, author)

    record = SourceRecord(
        source_id=source_id,
        title=title,
        author=author,
        author_country=normalize_text(fields.get("author_country")),
        is_african_authored=bool(fields.get("is_african_authored", False)),
        year=fields.get("year"),
        publisher=normalize_text(fields.get("publisher")),
        language=normalize_text(fields.get("language")),
        isbn=fields.get("isbn") or None,
        licence_status=licence_status,
        licence_notes=normalize_text(fields.get("licence_notes")),
        source_file=normalize_text(fields.get("source_file")),
        date_added=fields.get("date_added") or date.today().isoformat(),
    )

    registry[source_id] = record
    return record


def is_cleared(registry: dict[str, SourceRecord], source_file: str) -> bool:
    r"""
    PURPOSE
    -------
    The hard rule, as code: a source_file is cleared for use only if some
    record in the registry names it and that record's licence_status is
    exactly "cleared". A source_file absent from the registry, or present
    only with a "pending" or "rejected" record, is NOT cleared.

    PARAMETERS
    ----------
    registry      The registry to check.
    source_file   The file name to check (matches SourceRecord.source_file).

    RETURNS
    -------
    True only if a "cleared" record exists for source_file.

    EXAMPLES
    --------
    >>> registry: dict[str, SourceRecord] = {}
    >>> _ = add_or_update_source(
    ...     registry, title="A", author="X", author_country="Nigeria",
    ...     is_african_authored=True, year=2020, publisher="P",
    ...     language="English", isbn=None, licence_status="cleared",
    ...     licence_notes="", source_file="a.pdf", date_added="2024-01-01",
    ... )
    >>> _ = add_or_update_source(
    ...     registry, title="B", author="Y", author_country="Kenya",
    ...     is_african_authored=True, year=2021, publisher="P",
    ...     language="English", isbn=None, licence_status="pending",
    ...     licence_notes="", source_file="b.pdf", date_added="2024-01-01",
    ... )
    >>> is_cleared(registry, "a.pdf")
    True
    >>> is_cleared(registry, "b.pdf")
    False
    >>> is_cleared(registry, "missing.pdf")
    False
    """
    return any(
        record.source_file == source_file and record.licence_status == "cleared"
        for record in registry.values()
    )


def filter_cleared_files(
    registry: dict[str, SourceRecord],
    candidate_files: list[Path],
) -> tuple[list[Path], list[Path]]:
    r"""
    PURPOSE
    -------
    Partition candidate files into (cleared, not_cleared) by checking
    each file's name against is_cleared. This is the utility a future
    pipeline integration would call before feeding files to
    data_cleaning_program/data_cleaning.py; it does not touch
    data_cleaning.py itself.

    PARAMETERS
    ----------
    registry          The registry to check against.
    candidate_files   Candidate file paths to partition.

    RETURNS
    -------
    A (cleared, not_cleared) tuple of file paths, each list keeping the
    relative order of candidate_files.

    EXAMPLES
    --------
    >>> registry: dict[str, SourceRecord] = {}
    >>> _ = add_or_update_source(
    ...     registry, title="A", author="X", author_country="Nigeria",
    ...     is_african_authored=True, year=2020, publisher="P",
    ...     language="English", isbn=None, licence_status="cleared",
    ...     licence_notes="", source_file="a.pdf", date_added="2024-01-01",
    ... )
    >>> files = [Path("a.pdf"), Path("b.pdf"), Path("missing.pdf")]
    >>> cleared, not_cleared = filter_cleared_files(registry, files)
    >>> [p.name for p in cleared], [p.name for p in not_cleared]
    (['a.pdf'], ['b.pdf', 'missing.pdf'])
    """
    cleared: list[Path] = []
    not_cleared: list[Path] = []

    for path in candidate_files:
        if is_cleared(registry, path.name):
            cleared.append(path)
        else:
            not_cleared.append(path)

    return cleared, not_cleared


def registry_summary(registry: dict[str, SourceRecord]) -> dict[str, Any]:
    r"""
    PURPOSE
    -------
    Quick audit of the registry: counts by licence_status, counts of
    African-authored vs not, and counts by language.

    PARAMETERS
    ----------
    registry   The registry to summarize.

    RETURNS
    -------
    A dict with "total_sources", "by_licence_status" (a count for every
    status in LICENCE_STATUSES, including zero), "african_authored_count",
    "not_african_authored_count", and "by_language" (counts keyed by
    language, sorted by language name).

    EXAMPLES
    --------
    >>> registry: dict[str, SourceRecord] = {}
    >>> _ = add_or_update_source(
    ...     registry, title="A", author="X", author_country="Nigeria",
    ...     is_african_authored=True, year=2020, publisher="P",
    ...     language="English", isbn=None, licence_status="cleared",
    ...     licence_notes="", source_file="a.pdf", date_added="2024-01-01",
    ... )
    >>> _ = add_or_update_source(
    ...     registry, title="B", author="Y", author_country="France",
    ...     is_african_authored=False, year=2021, publisher="P",
    ...     language="French", isbn=None, licence_status="pending",
    ...     licence_notes="", source_file="b.pdf", date_added="2024-01-01",
    ... )
    >>> summary = registry_summary(registry)
    >>> summary["total_sources"], summary["by_licence_status"]
    (2, {'cleared': 1, 'pending': 1, 'rejected': 0})
    >>> summary["african_authored_count"], summary["not_african_authored_count"]
    (1, 1)
    >>> summary["by_language"]
    {'English': 1, 'French': 1}
    >>> registry_summary({})["total_sources"]
    0
    """
    by_status = {status: 0 for status in sorted(LICENCE_STATUSES)}
    african_authored_count = 0
    not_african_authored_count = 0
    by_language: dict[str, int] = {}

    for record in registry.values():
        by_status[record.licence_status] = (
            by_status.get(record.licence_status, 0) + 1
        )

        if record.is_african_authored:
            african_authored_count += 1
        else:
            not_african_authored_count += 1

        language = record.language or "unknown"
        by_language[language] = by_language.get(language, 0) + 1

    return {
        "total_sources": len(registry),
        "by_licence_status": by_status,
        "african_authored_count": african_authored_count,
        "not_african_authored_count": not_african_authored_count,
        "by_language": dict(sorted(by_language.items())),
    }


def parse_arguments(argument_list: list[str] | None = None) -> argparse.Namespace:
    r"""
    PURPOSE
    -------
    Parse the command line for the add/list/check subcommands.

    PARAMETERS
    ----------
    argument_list   Arguments to parse, or None to use sys.argv.

    RETURNS
    -------
    The parsed options, with `command` set to "add", "list", or "check".

    EXAMPLES
    --------
    >>> options = parse_arguments([
    ...     "add", "--title", "T", "--author", "A", "--author-country", "Kenya",
    ...     "--african-authored", "--year", "2020", "--publisher", "P",
    ...     "--language", "English", "--licence-status", "cleared",
    ...     "--source-file", "t.pdf",
    ... ])
    >>> options.command, options.title, options.is_african_authored, options.year
    ('add', 'T', True, 2020)
    >>> parse_arguments(["list"]).command
    'list'
    >>> parse_arguments(["list", "--json"]).as_json
    True
    >>> str(parse_arguments(["check", "--raw-data", "/tmp/raw"]).raw_data)
    '/tmp/raw'
    """
    parser = argparse.ArgumentParser(
        description=(
            "Catalogue sources ingested by the pipeline, and their "
            "licence/permission status."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    add_parser = subparsers.add_parser("add", help="register or update a source")
    add_parser.add_argument("--title", required=True, help="the source's title")
    add_parser.add_argument("--author", required=True, help="the source's author")
    add_parser.add_argument(
        "--author-country", default="", help="the author's country"
    )
    add_parser.add_argument(
        "--african-authored",
        dest="is_african_authored",
        action="store_true",
        help="flag the author as African",
    )
    add_parser.add_argument(
        "--year", type=int, default=None, help="publication year"
    )
    add_parser.add_argument(
        "--publisher", default="", help="the source's publisher"
    )
    add_parser.add_argument(
        "--language", default="", help="the source's language"
    )
    add_parser.add_argument(
        "--isbn", default=None, help="the source's ISBN, if known"
    )
    add_parser.add_argument(
        "--licence-status",
        required=True,
        choices=sorted(LICENCE_STATUSES),
        help="one of: cleared, pending, rejected",
    )
    add_parser.add_argument(
        "--licence-notes", default="", help="free-text licence/permission notes"
    )
    add_parser.add_argument(
        "--source-file",
        required=True,
        help="file name this record corresponds to, matching a file under "
             "data/ask/raw_data",
    )
    add_parser.add_argument(
        "--date-added",
        default=None,
        help="ISO date string (YYYY-MM-DD); defaults to today",
    )
    add_parser.add_argument(
        "--registry",
        type=Path,
        default=REGISTRY_FILE,
        help="path to the registry JSON file",
    )

    list_parser = subparsers.add_parser("list", help="print a registry summary")
    list_parser.add_argument(
        "--registry",
        type=Path,
        default=REGISTRY_FILE,
        help="path to the registry JSON file",
    )
    list_parser.add_argument(
        "--json",
        dest="as_json",
        action="store_true",
        help="print the raw summary dict (as JSON) instead of a table",
    )

    check_parser = subparsers.add_parser(
        "check",
        help="report which files under --raw-data have no cleared registry entry",
    )
    check_parser.add_argument(
        "--raw-data",
        type=Path,
        default=RAW_DATA_FOLDER,
        help="folder of candidate source files to check",
    )
    check_parser.add_argument(
        "--registry",
        type=Path,
        default=REGISTRY_FILE,
        help="path to the registry JSON file",
    )

    return parser.parse_args(argument_list)


def main() -> None:
    """
    PURPOSE
    -------
    Dispatch to the add/list/check subcommand named on the command line.

    RAISES
    ------
    FileNotFoundError   The `check` subcommand's --raw-data folder does
                         not exist.
    """
    options = parse_arguments()

    if options.command == "add":
        registry = load_registry(options.registry)
        record = add_or_update_source(
            registry,
            title=options.title,
            author=options.author,
            author_country=options.author_country,
            is_african_authored=options.is_african_authored,
            year=options.year,
            publisher=options.publisher,
            language=options.language,
            isbn=options.isbn,
            licence_status=options.licence_status,
            licence_notes=options.licence_notes,
            source_file=options.source_file,
            date_added=options.date_added,
        )
        save_registry(options.registry, registry)
        print(
            f"Saved {record.source_id}: {record.title!r} by "
            f"{record.author!r} -> licence_status={record.licence_status}\n"
            f"Registry: {options.registry}"
        )

    elif options.command == "list":
        registry = load_registry(options.registry)
        summary = registry_summary(registry)

        if options.as_json:
            print(json.dumps(summary, ensure_ascii=False, indent=2))
        else:
            print(f"Registry: {options.registry}")
            print(f"Total sources: {summary['total_sources']}")
            print("By licence status:")
            for status, count in summary["by_licence_status"].items():
                print(f"  {status}: {count}")
            print(
                f"African-authored: {summary['african_authored_count']} | "
                f"Not African-authored: {summary['not_african_authored_count']}"
            )
            print("By language:")
            for language, count in summary["by_language"].items():
                print(f"  {language}: {count}")

    elif options.command == "check":
        if not options.raw_data.is_dir():
            raise FileNotFoundError(
                f"Raw data folder does not exist: {options.raw_data}"
            )

        registry = load_registry(options.registry)
        candidate_files = sorted(
            path
            for path in options.raw_data.rglob("*")
            if path.is_file() and not path.name.startswith(".")
        )
        cleared, not_cleared = filter_cleared_files(registry, candidate_files)

        print(f"Checked {len(candidate_files)} file(s) under {options.raw_data}")
        print(f"Cleared: {len(cleared)}")
        print(f"NOT cleared (no 'cleared' registry entry): {len(not_cleared)}")

        for path in not_cleared:
            print(f"  - {path.name}")


if __name__ == "__main__":
    main()
