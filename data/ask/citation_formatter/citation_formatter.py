#!/usr/bin/env python3

"""
SoakinGarri Citation Formatter Program

PURPOSE
-------
Turns the bare source numbers in a generated answer ([1], [2][3], [1, 4])
into readable references, and checks them:

    Books       "Title, Author, p. 47" (title and author from
                source_registry.py's registry; pages from the passage)
    Web pages   The page's URL (from web_ingestion.py's manifest)
    Otherwise   The source file name, plus pages when known

Every number is checked against the passages the prompt actually gave the
model (prompt_builder.py's numbered passages). A number that points to no
passage is an invented citation: it is reported, and by default removed
from the answer text so the reader never sees a citation that leads
nowhere.

INPUT
-----
The answer text and the PromptPackage passages it was generated from
(anything with `.number` and `.record` works).

RUN
---
python3 citation_formatter.py --answer "Momentum is mass times velocity [1]." \\
    --passages passages.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(
    0, str(Path(__file__).resolve().parent.parent.parent.parent / "source")
)

from common import DATA_ROOT  # noqa: E402

REGISTRY_FILE = DATA_ROOT / "metadata" / "source_registry.json"
WEB_MANIFEST_FILE = DATA_ROOT / "raw_data" / "web" / "web_ingestion_manifest.json"

CLEANED_SUFFIX = "_cleaned"

NOT_IN_SOURCES_MARKERS = (
    "do not contain the answer",
    "does not contain the answer",
    "don't contain the answer",
    "doesn't contain the answer",
)

# A bracket holding only numbers, commas, spaces and ranges: [1], [1, 3], [2-4].
CITATION_PATTERN = re.compile(r"\[(\d+(?:\s*[-–,]\s*\d+)*)\]")


@dataclass(slots=True)
class CitationReport:
    """
    PURPOSE
    -------
    What the citation check found in one answer.

    ATTRIBUTES
    ----------
    cited                Distinct valid source numbers, in first-use order.
    invalid              Distinct numbers pointing to no passage.
    uncited              Passage numbers the answer never cites.
    says_not_in_sources  Whether the answer says the sources lack the answer.
    """

    cited: list[int] = field(default_factory=list)
    invalid: list[int] = field(default_factory=list)
    uncited: list[int] = field(default_factory=list)
    says_not_in_sources: bool = False

    @property
    def ok(self) -> bool:
        """
        PURPOSE
        -------
        True when no citation is invented, and the answer either cites at
        least one source or says the sources do not contain the answer.
        """
        return not self.invalid and (bool(self.cited) or self.says_not_in_sources)


@dataclass(slots=True)
class FormattedAnswer:
    """
    PURPOSE
    -------
    An answer ready to show.

    ATTRIBUTES
    ----------
    text         The answer text, with invalid citations removed when asked.
    references   One {"number", "reference", "chunk_id"} dict per cited
                  source, in citation-number order.
    report       The CitationReport for the original answer.
    """

    text: str
    references: list[dict[str, Any]]
    report: CitationReport

    def render(self) -> str:
        """
        PURPOSE
        -------
        The answer followed by a "Sources:" list of its references.
        """
        if not self.references:
            return self.text

        lines = [f"[{ref['number']}] {ref['reference']}" for ref in self.references]

        return f"{self.text}\n\nSources:\n" + "\n".join(lines)


def raw_stem(file_name: str | None) -> str:
    r"""
    PURPOSE
    -------
    The stem of the original source file behind a retrieval record. Records
    name the cleaned JSON ("book_cleaned.json"); the registry and web
    manifest name the raw file ("book.pdf", "WEB-1.txt"). Both reduce to
    the same stem.

    PARAMETERS
    ----------
    file_name   A raw or cleaned file name, or None.

    RETURNS
    -------
    The stem, or '' for no name.

    EXAMPLES
    --------
    >>> raw_stem("Gell_1998_cleaned.json"), raw_stem("Gell_1998.pdf")
    ('Gell_1998', 'Gell_1998')
    >>> raw_stem("WEB-ab12.txt"), raw_stem(None)
    ('WEB-ab12', '')
    """
    if not file_name:
        return ""

    stem = Path(str(file_name)).stem

    return stem[: -len(CLEANED_SUFFIX)] if stem.endswith(CLEANED_SUFFIX) else stem


def load_source_lookup(
    registry_file: Path = REGISTRY_FILE,
    web_manifest_file: Path = WEB_MANIFEST_FILE,
) -> dict[str, dict[str, Any]]:
    r"""
    PURPOSE
    -------
    Map each source's file stem to what a reference needs: title, author
    and year from the source registry, URL from the web ingestion manifest.
    A missing or unreadable file contributes nothing.

    PARAMETERS
    ----------
    registry_file       source_registry.json written by source_registry.py.
    web_manifest_file   web_ingestion_manifest.json written by
                         web_ingestion.py.

    RETURNS
    -------
    A dict of stem -> {"title", "author", "year", "url"} (keys present only
    when known).

    EXAMPLES
    --------
    >>> import tempfile
    >>> with tempfile.TemporaryDirectory() as folder:
    ...     registry = Path(folder) / "registry.json"
    ...     manifest = Path(folder) / "manifest.json"
    ...     _ = registry.write_text(json.dumps({"S1": {"source_file": "gell.pdf",
    ...         "title": "The Anthropology of Time", "author": "Alfred Gell", "year": 1992}}))
    ...     _ = manifest.write_text(json.dumps({"known_pages": [
    ...         {"url": "https://example.org/ori", "output_file": "WEB-1.txt", "content_hash": "h"}]}))
    ...     lookup = load_source_lookup(registry, manifest)
    >>> lookup["gell"]["author"], lookup["WEB-1"]["url"]
    ('Alfred Gell', 'https://example.org/ori')
    >>> load_source_lookup(Path("/no/registry.json"), Path("/no/manifest.json"))
    {}
    """
    lookup: dict[str, dict[str, Any]] = {}

    try:
        registry = json.loads(registry_file.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        registry = {}

    for record in registry.values() if isinstance(registry, dict) else []:
        stem = raw_stem(record.get("source_file"))

        if stem:
            lookup.setdefault(stem, {}).update(
                {
                    key: record[key]
                    for key in ("title", "author", "year")
                    if record.get(key)
                }
            )

    try:
        manifest = json.loads(web_manifest_file.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        manifest = {}

    for page in manifest.get("known_pages") or manifest.get("results") or []:
        stem = raw_stem(page.get("output_file"))

        if stem and page.get("url"):
            lookup.setdefault(stem, {})["url"] = page["url"]

    return lookup


def page_label(record: dict[str, Any]) -> str:
    r"""
    PURPOSE
    -------
    "p. 47" or "pp. 47–49" from a record's page fields.

    PARAMETERS
    ----------
    record   A retrieval record.

    RETURNS
    -------
    The page label, or '' when no page is known.

    EXAMPLES
    --------
    >>> page_label({"source_page_start": 47}), page_label({"source_page_start": 47, "source_page_end": 49})
    ('p. 47', 'pp. 47–49')
    >>> page_label({"source_page_start": 5, "source_page_end": 5}), page_label({})
    ('p. 5', '')
    """
    start = record.get("source_page_start")
    end = record.get("source_page_end")

    if start is None:
        return ""

    if end is not None and end != start:
        return f"pp. {start}–{end}"

    return f"p. {start}"


def format_reference(record: dict[str, Any], lookup: dict[str, dict[str, Any]]) -> str:
    r"""
    PURPOSE
    -------
    One readable reference for a retrieval record.

    PARAMETERS
    ----------
    record   A retrieval record (needs at least "source_file").
    lookup   Output of load_source_lookup.

    RETURNS
    -------
    "Title, Author, p. 47" for a registered book, the URL for a web page,
    or "file name, p. 47" when the source is not registered.

    EXAMPLES
    --------
    >>> lookup = {"gell": {"title": "The Anthropology of Time", "author": "Alfred Gell"},
    ...           "WEB-1": {"url": "https://example.org/ori"}}
    >>> format_reference({"source_file": "gell_cleaned.json", "source_page_start": 47}, lookup)
    'The Anthropology of Time, Alfred Gell, p. 47'
    >>> format_reference({"source_file": "WEB-1_cleaned.json", "source_page_start": 1}, lookup)
    'https://example.org/ori'
    >>> format_reference({"source_file": "notes_cleaned.json", "source_page_start": 3,
    ...                   "source_page_end": 4}, lookup)
    'notes, pp. 3–4'
    >>> format_reference({}, lookup)
    'unknown source'
    """
    stem = raw_stem(record.get("source_file"))
    source = lookup.get(stem, {})

    if source.get("url"):
        return str(source["url"])

    parts = [str(source["title"])] if source.get("title") else ([stem] if stem else [])

    if source.get("author"):
        parts.append(str(source["author"]))

    pages = page_label(record)

    if pages and parts:
        parts.append(pages)

    return ", ".join(parts) if parts else "unknown source"


def extract_citations(answer: str) -> list[int]:
    r"""
    PURPOSE
    -------
    Every source number cited in an answer, in order, with repeats.
    Handles [1], [1][2], [1, 3] and ranges such as [2-4] / [2–4].

    PARAMETERS
    ----------
    answer   The generated answer.

    RETURNS
    -------
    The cited numbers.

    EXAMPLES
    --------
    >>> extract_citations("Momentum is p = mv [1]. It is conserved [2][3], see [1, 4] and [5-7].")
    [1, 2, 3, 1, 4, 5, 6, 7]
    >>> extract_citations("An array a[i] and a list [x] are not citations.")
    []
    """
    numbers: list[int] = []

    for match in CITATION_PATTERN.finditer(answer or ""):
        for part in re.split(r"\s*,\s*", match.group(1)):
            bounds = re.split(r"\s*[-–]\s*", part)

            if len(bounds) == 2:
                low, high = int(bounds[0]), int(bounds[1])
                # A reversed or huge range is not a real citation range.
                if low <= high and high - low < 50:
                    numbers.extend(range(low, high + 1))
            else:
                numbers.append(int(bounds[0]))

    return numbers


def check_citations(answer: str, passages: list[Any]) -> CitationReport:
    r"""
    PURPOSE
    -------
    Check every citation in an answer against the numbered passages.

    PARAMETERS
    ----------
    answer     The generated answer.
    passages   The numbered passages (anything with `.number`).

    RETURNS
    -------
    A CitationReport.

    EXAMPLES
    --------
    >>> from types import SimpleNamespace as P
    >>> passages = [P(number=1), P(number=2), P(number=3)]
    >>> report = check_citations("Mass [1] and velocity [1][4].", passages)
    >>> report.cited, report.invalid, report.uncited, report.ok
    ([1], [4], [2, 3], False)
    >>> check_citations("The provided sources do not contain the answer.", passages).ok
    True
    >>> check_citations("Momentum is mass times velocity.", passages).ok
    False
    """
    valid_numbers = {passage.number for passage in passages}
    cited: list[int] = []
    invalid: list[int] = []

    for number in extract_citations(answer):
        target = cited if number in valid_numbers else invalid

        if number not in target:
            target.append(number)

    lowered = (answer or "").casefold()

    return CitationReport(
        cited=cited,
        invalid=invalid,
        uncited=sorted(valid_numbers - set(cited)),
        says_not_in_sources=any(marker in lowered for marker in NOT_IN_SOURCES_MARKERS),
    )


def remove_invalid_citations(answer: str, valid_numbers: set[int]) -> str:
    r"""
    PURPOSE
    -------
    Drop invented numbers from the answer's citation brackets. A bracket
    left empty disappears with the space before it.

    PARAMETERS
    ----------
    answer          The generated answer.
    valid_numbers   The passage numbers that exist.

    RETURNS
    -------
    The cleaned answer.

    EXAMPLES
    --------
    >>> remove_invalid_citations("Mass [1] and velocity [4]. Both [1, 9].", {1, 2})
    'Mass [1] and velocity. Both [1].'
    >>> remove_invalid_citations("Range [1-3].", {1, 2})
    'Range [1, 2].'
    """

    def replace(match: re.Match) -> str:
        kept: list[int] = []

        for number in extract_citations(match.group(0)):
            if number in valid_numbers and number not in kept:
                kept.append(number)

        return f"[{', '.join(str(n) for n in kept)}]" if kept else ""

    cleaned = CITATION_PATTERN.sub(replace, answer or "")

    # Tidy the space an emptied bracket leaves before punctuation.
    return re.sub(r"\s+([.,;:!?])", r"\1", re.sub(r"[ \t]{2,}", " ", cleaned)).strip()


def format_answer(
    answer: str,
    passages: list[Any],
    lookup: dict[str, dict[str, Any]] | None = None,
    drop_invalid: bool = True,
) -> FormattedAnswer:
    r"""
    PURPOSE
    -------
    Check an answer's citations and build its reference list.

    PARAMETERS
    ----------
    answer         The generated answer.
    passages       The numbered passages (`.number` and `.record`).
    lookup         Output of load_source_lookup; None loads the default
                    registry and web manifest.
    drop_invalid   Whether to remove invented citations from the text.

    RETURNS
    -------
    A FormattedAnswer; its references cover only the cited sources.

    EXAMPLES
    --------
    >>> from types import SimpleNamespace as P
    >>> passages = [
    ...     P(number=1, record={"chunk_id": "C1", "source_file": "gell_cleaned.json", "source_page_start": 47}),
    ...     P(number=2, record={"chunk_id": "C2", "source_file": "WEB-1_cleaned.json"}),
    ... ]
    >>> lookup = {"gell": {"title": "The Anthropology of Time", "author": "Alfred Gell"},
    ...           "WEB-1": {"url": "https://example.org/ori"}}
    >>> formatted = format_answer("Time is social [1][3]. Ori is destiny [2].", passages, lookup)
    >>> print(formatted.render())
    Time is social [1]. Ori is destiny [2].
    <BLANKLINE>
    Sources:
    [1] The Anthropology of Time, Alfred Gell, p. 47
    [2] https://example.org/ori
    >>> formatted.report.invalid, formatted.references[0]["chunk_id"]
    ([3], 'C1')
    """
    if lookup is None:
        lookup = load_source_lookup()

    report = check_citations(answer, passages)
    by_number = {passage.number: passage for passage in passages}

    text = (
        remove_invalid_citations(answer, set(by_number))
        if drop_invalid and report.invalid
        else (answer or "").strip()
    )

    references = [
        {
            "number": number,
            "reference": format_reference(by_number[number].record, lookup),
            "chunk_id": by_number[number].record.get("chunk_id")
            or by_number[number].record.get("record_id"),
        }
        for number in sorted(report.cited)
    ]

    return FormattedAnswer(text=text, references=references, report=report)


def parse_arguments(argument_list: list[str] | None = None) -> argparse.Namespace:
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
    >>> options = parse_arguments(["--answer", "A [1].", "--passages", "p.json"])
    >>> options.answer, options.passages.name, options.keep_invalid
    ('A [1].', 'p.json', False)
    """
    parser = argparse.ArgumentParser(
        description="Check an answer's citations and print it with its references."
    )
    parser.add_argument("--answer", required=True, help="the generated answer text")
    parser.add_argument(
        "--passages",
        type=Path,
        required=True,
        help='JSON list of {"number": n, "record": {...}} passages',
    )
    parser.add_argument(
        "--keep-invalid", action="store_true", help="leave invented citations in the text"
    )

    return parser.parse_args(argument_list)


def main() -> None:
    """
    PURPOSE
    -------
    Format the answer given on the command line and print it, followed by
    the citation check.
    """
    from types import SimpleNamespace

    options = parse_arguments()

    passages = [
        SimpleNamespace(number=int(item["number"]), record=item.get("record") or {})
        for item in json.loads(options.passages.read_text(encoding="utf-8"))
    ]

    formatted = format_answer(options.answer, passages, drop_invalid=not options.keep_invalid)

    print(formatted.render())
    print(
        f"\nCitation check: {'ok' if formatted.report.ok else 'PROBLEMS'} "
        f"(cited {formatted.report.cited}, invalid {formatted.report.invalid}, "
        f"uncited {formatted.report.uncited})"
    )


if __name__ == "__main__":
    main()
