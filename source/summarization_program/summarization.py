#!/usr/bin/env python3

"""
SoakinGarri Technical Summarization Program

PURPOSE
-------
Scans cleaned documents and keeps only the technical material:

    formulas             equations, LaTeX maths, numbered expressions
    concepts             definitions and named principles
    graphs               figure, chart and plot captions and references
    technical_drawings   schematics, diagrams, cross-sections, plans

Everything else is discarded. Each source document produces a plain-text summary
file containing the extracted passages, grouped by category and labelled with the
page and chapter they came from.

WHAT THIS PROGRAM CAN AND CANNOT DO
-----------------------------------
It reads text. For graphs and technical drawings it therefore extracts the
caption and the surrounding description, not the picture itself. To also export
the pictures, run with --export-images and keep the original PDFs reachable at
the "source_path" recorded by the cleaning program; PyMuPDF then writes every
image from the pages where a figure caption was found.

INPUT FOLDER
------------
<repo>/data/ask/cleaned_data

OUTPUT FOLDER
-------------
<repo>/data/ask/summarized_data

OUTPUT FILES
------------
<book>_summary.txt          human-readable extract, one per document
<book>_findings.jsonl       the same findings as structured records
summarization_manifest.json
images/<book>/page_<n>_<i>.png    only with --export-images

INSTALL
-------
python3 -m pip install pymupdf        (only needed for --export-images)

RUN
---
python3 summarization.py
python3 summarization.py --categories formula graph
python3 summarization.py --export-images
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common import DATA_ROOT, normalize_text, stable_id, load_documents  # noqa: E402

INPUT_FOLDER = DATA_ROOT / "cleaned_data"

OUTPUT_FOLDER = DATA_ROOT / "summarized_data"

MANIFEST_FILE = OUTPUT_FOLDER / "summarization_manifest.json"

SOURCE_LABEL = "Borrowed content and open source"

CATEGORIES = (
    "formula",
    "concept",
    "graph",
    "technical_drawing",
)

CATEGORY_HEADINGS = {
    "formula": "FORMULAS AND EQUATIONS",
    "concept": "CONCEPTS AND DEFINITIONS",
    "graph": "GRAPHS, CHARTS AND PLOTS",
    "technical_drawing": "TECHNICAL DRAWINGS AND SCHEMATICS",
}

SUPPORTED_EXTENSIONS = {".json", ".jsonl"}

MAXIMUM_CONTEXT_CHARACTERS = 400

SENTENCE_PATTERN = re.compile(
    r"(?<=[.!?])\s+(?=[A-Z0-9“\"'(\[])"
)

LATEX_PATTERNS = (
    re.compile(r"\$\$.+?\$\$", re.DOTALL),
    re.compile(r"\$[^$\n]{2,200}\$"),
    re.compile(r"\\\[.+?\\\]", re.DOTALL),
    re.compile(r"\\begin\{(?:equation|align|gather)\*?\}.+?"
               r"\\end\{(?:equation|align|gather)\*?\}", re.DOTALL),
)

FORMULA_PATTERNS = (
    re.compile(
        r"(?<!\w)"
        r"[A-Za-z][A-Za-z0-9_]*(?:\s*\([^()\n]{1,80}\))?"
        r"\s*=\s*"
        r"[^.\n;]{1,180}"
    ),
    re.compile(
        r"(?<!\w)"
        r"(?:[A-Za-z0-9_()]+(?:\s*[+\-*/^]\s*[A-Za-z0-9_().]+)+)"
        r"(?!\w)"
    ),
    re.compile(
        r"(?<!\w)[A-Za-z][A-Za-z0-9_]*\s*(?:≈|≠|≤|≥|∝|∼)\s*[^.\n;]{1,160}"
    ),
)

HYPHENATED_WORD_PATTERN = re.compile(r"^[A-Za-z]+(?:-[A-Za-z]+)+$")

FORMULA_TAIL_PATTERN = re.compile(
    r"\s+(?:for|where|when|with|and|in|of|to|on|as|by|from|that|which|"
    r"so|then|if|is|are|was|gives|giving|becomes|yields)\s+.*$",
    re.IGNORECASE,
)

EQUATION_NUMBER_PATTERN = re.compile(
    r"\(\s*\d{1,3}(?:\.\d{1,3}){0,2}\s*\)\s*$"
)

DEFINITION_PATTERN = re.compile(
    r"""(?x)
    ^\s*
    (?:The\s+|A\s+|An\s+)?
    (?P<term>[A-Z][A-Za-z0-9'’\-\s]{1,80}?)
    \s+
    (?:is|are)\s+
    (?:defined\s+as|known\s+as|called|referred\s+to\s+as)
    \b
    |
    ^\s*
    (?:We\s+define|Define)\s+
    (?P<term2>[A-Za-z][A-Za-z0-9'’\-\s]{1,80}?)
    \s+as\b
    |
    ^\s*
    (?P<term3>[A-Z][A-Za-z0-9'’\-\s]{1,60}?)
    \s+
    (?:is|are|means|refers\s+to)
    \s+
    (?:the|a|an)\b
    """
)

PRINCIPLE_PATTERN = re.compile(
    r"(?i)\b(?:law|principle|theorem|rule|axiom|postulate|equation|hypothesis)"
    r"\s+of\s+[A-Z][A-Za-z'’\-]+"
    r"|\b[A-Z][A-Za-z'’\-]+(?:'s|’s)\s+"
    r"(?:law|principle|theorem|rule|equation|constant)\b"
)

CAPTION_PATTERN = re.compile(
    r"""(?ix)^\s*
    (?P<kind>figure|fig\.?|graph|chart|plot|diagram|schematic|
             illustration|drawing|plate|scheme|exhibit)
    \s*
    (?P<label>\d{1,3}(?:[.\-]\d{1,3}){0,2}|[ivxlcdm]{1,6})
    \s*[:.\-–—]?\s*
    (?P<caption>.{0,200})
    $"""
)

GRAPH_MARKERS = re.compile(
    r"(?i)\b(?:graph|chart|plot|histogram|scatter\s*plot|bar\s*chart|"
    r"pie\s*chart|curve|axis|axes|x-axis|y-axis|abscissa|ordinate|"
    r"plotted\s+against|as\s+a\s+function\s+of|trend\s*line|data\s*points)\b"
)

DRAWING_MARKERS = re.compile(
    r"(?i)\b(?:schematic|circuit\s+diagram|wiring\s+diagram|block\s+diagram|"
    r"flow\s*chart|cross[-\s]?section|sectional\s+view|exploded\s+view|"
    r"isometric|orthographic|elevation\s+view|plan\s+view|floor\s+plan|"
    r"blueprint|assembly\s+drawing|cad\s+drawing|technical\s+drawing|"
    r"free[-\s]?body\s+diagram|layout\s+drawing)\b"
)

CONCEPT_MARKERS = re.compile(
    r"(?i)\b(?:is\s+defined\s+as|are\s+defined\s+as|refers\s+to|"
    r"is\s+known\s+as|are\s+known\s+as|is\s+called|are\s+called|"
    r"by\s+definition|we\s+define|the\s+definition\s+of|"
    r"can\s+be\s+described\s+as)\b"
)


@dataclass(slots=True)
class Finding:
    """
    PURPOSE
    -------
    One extracted piece of technical content, together with everything
    needed to render it in a summary and trace it back to its source.

    ATTRIBUTES
    ----------
    finding_id          Stable identifier derived from the source file, category and excerpt.
    category            One of "formula", "concept", "graph", "technical_drawing".
    excerpt             The extracted fragment worth keeping.
    context             The full sentence the excerpt was drawn from, shortened.
    label               Figure/diagram label such as "Figure 4.1", or None.
    source_file         Name of the cleaned document file the finding came from.
    document_id         Identifier of the source document, or None.
    chapter_id          Identifier of the chapter, or None.
    chapter_title       Title of the chapter the finding came from.
    paragraph_id        Identifier of the paragraph, or None.
    source_page_start   First page the finding appears on, or None.
    source_page_end     Last page the finding appears on, or None.
    source              Attribution label recorded on every finding.
    """

    finding_id: str
    category: str
    excerpt: str
    context: str
    label: str | None
    source_file: str
    document_id: str | None
    chapter_id: str | None
    chapter_title: str
    paragraph_id: str | None
    source_page_start: int | None
    source_page_end: int | None
    source: str


def split_sentences(text: str) -> list[str]:
    r"""
    PURPOSE
    -------
    Split text into sentences.

    PARAMETERS
    ----------
    text   The text to split.

    RETURNS
    -------
    The list of sentences, or an empty list when there is nothing to split.

    EXAMPLES
    --------
    >>> split_sentences("One. Two! Three?")
    ['One.', 'Two!', 'Three?']
    >>> split_sentences("  ")
    []
    """
    normalized = normalize_text(text)

    if not normalized:
        return []

    sentences = [
        sentence.strip()
        for sentence in SENTENCE_PATTERN.split(normalized)
        if sentence.strip()
    ]

    return sentences or [normalized]


def shorten(text: str, limit: int = MAXIMUM_CONTEXT_CHARACTERS) -> str:
    r"""
    PURPOSE
    -------
    Trim text to ``limit`` characters at a word boundary, adding an
    ellipsis when it was cut.

    PARAMETERS
    ----------
    text    The text to trim.
    limit   Maximum number of characters to keep.

    RETURNS
    -------
    The trimmed text.

    EXAMPLES
    --------
    >>> shorten("short text", 40)
    'short text'
    >>> shorten("one two three four five six seven", 15)
    'one two three …'
    """
    text = normalize_text(text)

    if len(text) <= limit:
        return text

    cut = text[:limit].rsplit(" ", 1)[0]

    return f"{cut} …"


def trim_formula_tail(formula: str) -> str:
    r"""
    PURPOSE
    -------
    Cut prose that trails an equation.

    PARAMETERS
    ----------
    formula   The candidate formula text, possibly with trailing prose.

    RETURNS
    -------
    The formula with any trailing prose removed.

    EXAMPLES
    --------
    >>> trim_formula_tail("sigma = F/A for a uniform bar")
    'sigma = F/A'
    >>> trim_formula_tail("v = u + at")
    'v = u + at'
    >>> trim_formula_tail("E = mc^2 where c is the speed of light")
    'E = mc^2'
    """
    trimmed = FORMULA_TAIL_PATTERN.sub("", normalize_text(formula))

    return trimmed.strip(" ,.;:") or formula


def extract_formulas(text: str) -> list[str]:
    r"""
    PURPOSE
    -------
    Return equation-like fragments, LaTeX first, deduplicated
    case-insensitively. Hyphenated words such as "cross-section" and
    "x-axis" look like subtraction to a naive pattern, so they are
    rejected explicitly.

    PARAMETERS
    ----------
    text   The text to search for formulas.

    RETURNS
    -------
    The list of distinct formula fragments found, in order of discovery.

    EXAMPLES
    --------
    >>> extract_formulas("Newton wrote F = ma. Rearranged, a = F/m.")
    ['F = ma', 'a = F/m']
    >>> extract_formulas("The energy is $E = mc^2$ in every frame.")
    ['$E = mc^2$']
    >>> extract_formulas("It follows that sigma = F/A for a uniform bar.")
    ['sigma = F/A']
    >>> extract_formulas("The cross-section along the x-axis is shown.")
    []
    >>> extract_formulas("Ordinary prose with no mathematics.")
    []
    """
    text = normalize_text(text)
    formulas: list[str] = []
    seen: set[str] = set()

    def remember(candidate: str) -> None:
        candidate = trim_formula_tail(candidate.strip(" ,.;:"))

        if len(candidate) < 3:
            return

        if HYPHENATED_WORD_PATTERN.match(candidate):
            return

        key = candidate.casefold()

        if key in seen:
            return

        # a sub-expression of a formula already found adds nothing
        if any(key in existing.casefold() for existing in formulas):
            return

        seen.add(key)
        formulas.append(candidate[:240])

    for pattern in LATEX_PATTERNS:
        for match in pattern.finditer(text):
            remember(match.group(0))

    if formulas:
        return formulas

    for pattern in FORMULA_PATTERNS:
        for match in pattern.finditer(text):
            remember(match.group(0))

    return formulas


def extract_caption(line: str) -> dict[str, str] | None:
    r"""
    PURPOSE
    -------
    Parse a figure-style caption into its kind, label and text.

    PARAMETERS
    ----------
    line   The line of text to parse as a caption.

    RETURNS
    -------
    A dict with "kind", "label" and "caption" keys, or None when the
    line is not a caption.

    EXAMPLES
    --------
    >>> extract_caption("Figure 3.2: Stress against strain for mild steel")
    {'kind': 'Figure', 'label': '3.2', 'caption': 'Stress against strain for mild steel'}
    >>> extract_caption("Fig. 4 — Circuit schematic")
    {'kind': 'Fig', 'label': '4', 'caption': 'Circuit schematic'}
    >>> extract_caption("This paragraph mentions figure 2 in passing.") is None
    True
    """
    match = CAPTION_PATTERN.match(normalize_text(line))

    if not match:
        return None

    kind = match.group("kind").strip().rstrip(".").title()

    return {
        "kind": kind,
        "label": match.group("label"),
        "caption": normalize_text(match.group("caption")),
    }


def classify_sentence(sentence: str) -> list[str]:
    r"""
    PURPOSE
    -------
    Return every category a sentence belongs to, in a stable order. A
    sentence can carry more than one kind of technical content.

    PARAMETERS
    ----------
    sentence   The sentence to classify.

    RETURNS
    -------
    The list of matching categories, in CATEGORIES order.

    EXAMPLES
    --------
    >>> classify_sentence("Figure 3.2: current plotted against voltage")
    ['graph']
    >>> classify_sentence("Entropy is defined as the measure of disorder.")
    ['concept']
    >>> classify_sentence("The cross-section shows the bearing housing.")
    ['technical_drawing']
    >>> classify_sentence("Since F = ma, the plot of force against acceleration is linear.")
    ['formula', 'graph']
    >>> classify_sentence("The x-axis shows time.")            # not mistaken for a formula
    ['graph']
    >>> classify_sentence("They walked to the market that morning.")
    []
    """
    sentence = normalize_text(sentence)
    categories: list[str] = []

    if extract_formulas(sentence) or EQUATION_NUMBER_PATTERN.search(sentence):
        categories.append("formula")

    if DEFINITION_PATTERN.match(sentence) or CONCEPT_MARKERS.search(sentence):
        categories.append("concept")
    elif PRINCIPLE_PATTERN.search(sentence):
        categories.append("concept")

    caption = extract_caption(sentence)
    caption_kind = caption["kind"].casefold() if caption else ""

    if caption_kind in {"figure", "fig", "graph", "chart", "plot", "exhibit"}:
        categories.append("graph")
    elif GRAPH_MARKERS.search(sentence):
        categories.append("graph")

    if caption_kind in {
        "diagram",
        "schematic",
        "drawing",
        "plate",
        "scheme",
        "illustration",
    }:
        categories.append("technical_drawing")
    elif DRAWING_MARKERS.search(sentence):
        categories.append("technical_drawing")

    return [category for category in CATEGORIES if category in categories]


def concept_term(sentence: str) -> str | None:
    r"""
    PURPOSE
    -------
    Pull the term being defined out of a definition sentence.

    PARAMETERS
    ----------
    sentence   The sentence to extract a defined term from.

    RETURNS
    -------
    The defined term, or None when the sentence does not define one.

    EXAMPLES
    --------
    >>> concept_term("Entropy is defined as a measure of disorder.")
    'Entropy'
    >>> concept_term("The Reynolds number is defined as the ratio of forces.")
    'Reynolds number'
    >>> concept_term("We define admittance as the inverse of impedance.")
    'admittance'
    >>> concept_term("They walked to the market.") is None
    True
    """
    match = DEFINITION_PATTERN.match(normalize_text(sentence))

    if not match:
        return None

    for group_name in ("term", "term2", "term3"):
        term = match.group(group_name)

        if term:
            return normalize_text(term)

    return None


def iterate_chapters(document: dict[str, Any]) -> list[dict[str, Any]]:
    r"""
    PURPOSE
    -------
    Return the document's chapters, wrapping a flat text field into a
    single synthetic chapter when the document has no chapter structure.

    PARAMETERS
    ----------
    document   The parsed document dictionary.

    RETURNS
    -------
    The list of chapter dictionaries, possibly empty.

    EXAMPLES
    --------
    >>> iterate_chapters({"chapters": [{"title": "A"}, "junk"]})
    [{'title': 'A'}]
    >>> iterate_chapters({"title": "Essay", "text": "Body"})[0]["paragraphs"]
    [{'paragraph_id': None, 'text': 'Body'}]
    >>> iterate_chapters({})
    []
    """
    chapters = document.get("chapters")

    if isinstance(chapters, list):
        return [
            chapter for chapter in chapters if isinstance(chapter, dict)
        ]

    text = normalize_text(
        document.get("text")
        or document.get("content")
        or document.get("body")
    )

    if not text:
        return []

    return [
        {
            "chapter_id": None,
            "title": document.get("title") or "Main Text",
            "source_page_start": None,
            "source_page_end": None,
            "paragraphs": [{"paragraph_id": None, "text": text}],
        }
    ]


def chapter_paragraphs(
    chapter: dict[str, Any],
) -> list[tuple[str | None, str]]:
    r"""
    PURPOSE
    -------
    Return ``(paragraph_id, text)`` pairs for a chapter.

    PARAMETERS
    ----------
    chapter   The chapter dictionary to read paragraphs from.

    RETURNS
    -------
    The list of (paragraph_id, text) pairs, skipping empty paragraphs.

    EXAMPLES
    --------
    >>> chapter_paragraphs({"paragraphs": [{"paragraph_id": "P-1", "text": "Hi"}, "plain", 7]})
    [('P-1', 'Hi'), (None, 'plain')]
    >>> chapter_paragraphs({"paragraphs": 5})
    []
    """
    raw = chapter.get("paragraphs", [])

    if isinstance(raw, str):
        raw = [raw]

    if not isinstance(raw, list):
        return []

    paragraphs: list[tuple[str | None, str]] = []

    for item in raw:
        if isinstance(item, str):
            paragraph_id, text = None, normalize_text(item)
        elif isinstance(item, dict):
            paragraph_id = item.get("paragraph_id")
            text = normalize_text(item.get("text") or item.get("content"))
        else:
            continue

        if text:
            paragraphs.append((paragraph_id, text))

    return paragraphs


def scan_document(
    document: dict[str, Any],
    source_file: Path,
    categories: Iterable[str] = CATEGORIES,
) -> list[Finding]:
    r"""
    PURPOSE
    -------
    Scan one cleaned document and return every technical finding in it.

    PARAMETERS
    ----------
    document      The parsed document dictionary to scan.
    source_file   Path to the source file; used to label findings and derive their ids.
    categories    Which categories to keep; findings outside this set are discarded.

    RETURNS
    -------
    The list of findings, in the order they were discovered.

    EXAMPLES
    --------
    >>> document = {
    ...     "document_id": "DOC-1",
    ...     "chapters": [{
    ...         "chapter_id": "CH-1",
    ...         "title": "Chapter 4: Materials",
    ...         "source_page_start": 61,
    ...         "source_page_end": 66,
    ...         "paragraphs": [
    ...             {"paragraph_id": "P-1",
    ...              "text": "Stress is defined as force per unit area. "
    ...                      "It follows that sigma = F/A for a uniform bar. "
    ...                      "They ate lunch outside."},
    ...             {"paragraph_id": "P-2",
    ...              "text": "Figure 4.1: stress plotted against strain. "
    ...                      "The cross-section of the specimen is shown alongside."},
    ...         ],
    ...     }],
    ... }
    >>> findings = scan_document(document, Path("materials_cleaned.json"))
    >>> [(f.category, f.excerpt) for f in findings]
    [('concept', 'Stress'), ('formula', 'sigma = F/A'), ('graph', 'Figure 4.1'), ('technical_drawing', 'The cross-section of the specimen is shown alongside.')]
    >>> findings[1].source_page_start, findings[1].chapter_title
    (61, 'Chapter 4: Materials')
    >>> findings[2].label, findings[2].context
    ('Figure 4.1', 'Figure 4.1: stress plotted against strain.')
    >>> [f.category for f in scan_document(document, Path("x.json"), categories=["concept"])]
    ['concept']
    >>> scan_document({}, Path("x.json"))
    []
    """
    wanted = [category for category in CATEGORIES if category in set(categories)]
    findings: list[Finding] = []
    seen: set[tuple[str, str]] = set()

    for chapter in iterate_chapters(document):
        chapter_title = normalize_text(chapter.get("title") or "Main Text")

        for paragraph_id, text in chapter_paragraphs(chapter):
            for sentence in split_sentences(text):
                for category in classify_sentence(sentence):
                    if category not in wanted:
                        continue

                    excerpt, label = summarize_finding(category, sentence)
                    key = (category, excerpt.casefold())

                    if key in seen:
                        continue

                    seen.add(key)

                    findings.append(
                        Finding(
                            finding_id=stable_id(
                                "FND", source_file.name, category, excerpt
                            ),
                            category=category,
                            excerpt=excerpt,
                            context=shorten(sentence),
                            label=label,
                            source_file=source_file.name,
                            document_id=document.get("document_id"),
                            chapter_id=chapter.get("chapter_id"),
                            chapter_title=chapter_title,
                            paragraph_id=paragraph_id,
                            source_page_start=chapter.get("source_page_start"),
                            source_page_end=chapter.get("source_page_end"),
                            source=SOURCE_LABEL,
                        )
                    )

    return findings


def summarize_finding(category: str, sentence: str) -> tuple[str, str | None]:
    r"""
    PURPOSE
    -------
    Reduce a sentence to the fragment worth keeping, plus a label when
    there is one.

    PARAMETERS
    ----------
    category   The category the sentence was classified as.
    sentence   The sentence to reduce.

    RETURNS
    -------
    A (excerpt, label) pair; label is None unless the excerpt is a
    figure-style caption.

    EXAMPLES
    --------
    >>> summarize_finding("formula", "It follows that v = u + at for constant acceleration.")
    ('v = u + at', None)
    >>> summarize_finding("concept", "Entropy is defined as a measure of disorder.")
    ('Entropy', None)
    >>> summarize_finding("graph", "Figure 2.1: output against time")
    ('Figure 2.1', 'Figure 2.1')
    >>> summarize_finding("technical_drawing", "The wiring diagram is on the next page.")
    ('The wiring diagram is on the next page.', None)
    """
    sentence = normalize_text(sentence)

    if category == "formula":
        formulas = extract_formulas(sentence)

        if formulas:
            return formulas[0], None

    if category == "concept":
        term = concept_term(sentence)

        if term:
            return term, None

    caption = extract_caption(sentence)

    if caption:
        label = f"{caption['kind']} {caption['label']}"
        return label, label

    return shorten(sentence), None


def group_by_category(
    findings: Iterable[Finding],
) -> dict[str, list[Finding]]:
    r"""
    PURPOSE
    -------
    Bucket findings by category, preserving order within each bucket.

    PARAMETERS
    ----------
    findings   The findings to bucket.

    RETURNS
    -------
    A dict mapping category to its findings, in CATEGORIES order, with
    empty categories omitted.

    EXAMPLES
    --------
    >>> findings = scan_document(
    ...     {"chapters": [{"title": "T", "paragraphs": [
    ...         "Entropy is defined as disorder. The plot of S against T rises."]}]},
    ...     Path("b.json"))
    >>> {category: len(items) for category, items in group_by_category(findings).items()}
    {'concept': 1, 'graph': 1}
    """
    grouped: dict[str, list[Finding]] = {}

    for finding in findings:
        grouped.setdefault(finding.category, []).append(finding)

    return {
        category: grouped[category]
        for category in CATEGORIES
        if category in grouped
    }


def render_summary(
    findings: list[Finding],
    source_file: str,
    document_id: str | None = None,
) -> str:
    r"""
    PURPOSE
    -------
    Render findings as the plain-text summary document.

    PARAMETERS
    ----------
    findings      The findings to render.
    source_file   Name of the source file, shown in the header.
    document_id   Document identifier, shown in the header when given.

    RETURNS
    -------
    The rendered summary text.

    EXAMPLES
    --------
    >>> findings = scan_document(
    ...     {"document_id": "DOC-1", "chapters": [{"title": "Ch 1", "source_page_start": 7,
    ...      "paragraphs": ["Entropy is defined as a measure of disorder."]}]},
    ...     Path("thermo_cleaned.json"))
    >>> print(render_summary(findings, "thermo_cleaned.json", "DOC-1"))
    TECHNICAL SUMMARY
    Source file: thermo_cleaned.json
    Document id: DOC-1
    Source: Borrowed content and open source
    Findings: 1 (concept: 1)
    <BLANKLINE>
    ================================================================
    CONCEPTS AND DEFINITIONS (1)
    ================================================================
    <BLANKLINE>
    [p. 7 | Ch 1]
    Entropy
        Entropy is defined as a measure of disorder.
    <BLANKLINE>
    >>> print(render_summary([], "empty.json"))
    TECHNICAL SUMMARY
    Source file: empty.json
    Source: Borrowed content and open source
    Findings: 0
    <BLANKLINE>
    No formulas, concepts, graphs or technical drawings were found.
    """
    grouped = group_by_category(findings)

    counts = ", ".join(
        f"{category}: {len(items)}" for category, items in grouped.items()
    )

    lines = ["TECHNICAL SUMMARY", f"Source file: {source_file}"]

    if document_id:
        lines.append(f"Document id: {document_id}")

    lines.append(f"Source: {SOURCE_LABEL}")
    lines.append(
        f"Findings: {len(findings)}" + (f" ({counts})" if counts else "")
    )
    lines.append("")

    if not findings:
        lines.append(
            "No formulas, concepts, graphs or technical drawings were found."
        )
        return "\n".join(lines)

    for category, items in grouped.items():
        lines.append("=" * 64)
        lines.append(f"{CATEGORY_HEADINGS[category]} ({len(items)})")
        lines.append("=" * 64)
        lines.append("")

        for finding in items:
            location = describe_location(finding)

            if location:
                lines.append(f"[{location}]")

            lines.append(finding.excerpt)

            if finding.context and finding.context != finding.excerpt:
                lines.append(f"    {finding.context}")

            lines.append("")

    return "\n".join(lines)


def describe_location(finding: Finding) -> str:
    r"""
    PURPOSE
    -------
    Build the ``p. 7 | Chapter 1`` location label for a finding.

    PARAMETERS
    ----------
    finding   The finding to describe the location of.

    RETURNS
    -------
    The location label, with empty parts omitted.

    EXAMPLES
    --------
    >>> findings = scan_document(
    ...     {"chapters": [{"title": "Ch 2", "source_page_start": 10, "source_page_end": 14,
    ...      "paragraphs": ["Entropy is defined as disorder."]}]}, Path("b.json"))
    >>> describe_location(findings[0])
    'pp. 10-14 | Ch 2'
    >>> findings[0].source_page_end = None
    >>> describe_location(findings[0])
    'p. 10 | Ch 2'
    >>> findings[0].source_page_start = None
    >>> describe_location(findings[0])
    'Ch 2'
    """
    parts: list[str] = []

    start = finding.source_page_start
    end = finding.source_page_end

    if start is not None and end is not None and end != start:
        parts.append(f"pp. {start}-{end}")
    elif start is not None:
        parts.append(f"p. {start}")

    if finding.chapter_title:
        parts.append(finding.chapter_title)

    return " | ".join(parts)


def export_figure_images(
    source_path: Path,
    pages: Iterable[int],
    destination_folder: Path,
) -> list[str]:
    """
    PURPOSE
    -------
    Export images from the given pages of a source PDF. Requires
    PyMuPDF and the original PDF still being where the cleaning program
    recorded it.

    PARAMETERS
    ----------
    source_path           Path to the original source PDF.
    pages                 The 1-based page numbers to export images from.
    destination_folder    Folder the exported PNG files are written to.

    RETURNS
    -------
    The written file paths; an empty list when the PDF is missing or
    PyMuPDF is not installed.
    """
    try:
        import fitz
    except ImportError:
        return []

    if not source_path.exists() or source_path.suffix.casefold() != ".pdf":
        return []

    destination_folder.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    with fitz.open(source_path) as document:
        for page_number in sorted({page for page in pages if page}):
            if not 1 <= page_number <= document.page_count:
                continue

            page = document[page_number - 1]

            for image_index, image in enumerate(page.get_images(full=True), start=1):
                pixmap = fitz.Pixmap(document, image[0])

                if pixmap.n - pixmap.alpha >= 4:
                    pixmap = fitz.Pixmap(fitz.csRGB, pixmap)

                target = destination_folder / (
                    f"page_{page_number}_{image_index}.png"
                )
                pixmap.save(target)
                written.append(str(target))

    return written


def atomic_write_text(destination: Path, text: str) -> None:
    r"""
    PURPOSE
    -------
    Write text through a temporary file, then rename it over the
    target, so readers never see a partially written file.

    PARAMETERS
    ----------
    destination   Path the text is ultimately written to.
    text          The text to write.

    EXAMPLES
    --------
    >>> import tempfile
    >>> with tempfile.TemporaryDirectory() as folder:
    ...     path = Path(folder) / "out" / "summary.txt"
    ...     atomic_write_text(path, "café\n")
    ...     (path.read_text("utf-8"), [p.name for p in path.parent.iterdir()])
    ('café\n', ['summary.txt'])
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


def atomic_write_json(destination: Path, data: dict[str, Any]) -> None:
    r"""
    PURPOSE
    -------
    Write JSON atomically.

    PARAMETERS
    ----------
    destination   Path the JSON is written to.
    data          The data to serialize and write.

    EXAMPLES
    --------
    >>> import tempfile
    >>> with tempfile.TemporaryDirectory() as folder:
    ...     path = Path(folder) / "manifest.json"
    ...     atomic_write_json(path, {"findings": 4})
    ...     json.loads(path.read_text("utf-8"))
    {'findings': 4}
    """
    atomic_write_text(
        destination,
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
    )


def write_findings_jsonl(path: Path, findings: list[Finding]) -> int:
    r"""
    PURPOSE
    -------
    Write findings as JSONL, one finding per line.

    PARAMETERS
    ----------
    path       Path the JSONL file is written to.
    findings   The findings to write.

    RETURNS
    -------
    The number of lines written.

    EXAMPLES
    --------
    >>> import tempfile
    >>> findings = scan_document(
    ...     {"chapters": [{"title": "T", "paragraphs": ["Entropy is defined as disorder."]}]},
    ...     Path("b.json"))
    >>> with tempfile.TemporaryDirectory() as folder:
    ...     path = Path(folder) / "findings.jsonl"
    ...     count = write_findings_jsonl(path, findings)
    ...     (count, json.loads(path.read_text("utf-8"))["category"])
    (1, 'concept')
    """
    lines = "".join(
        json.dumps(asdict(finding), ensure_ascii=False) + "\n"
        for finding in findings
    )

    atomic_write_text(path, lines)

    return len(findings)


def discover_input_files(input_folder: Path = INPUT_FOLDER) -> list[Path]:
    r"""
    PURPOSE
    -------
    List the .json/.jsonl files to scan, skipping hidden and manifest
    files.

    PARAMETERS
    ----------
    input_folder   Root folder to search for input files.

    RETURNS
    -------
    The matching files, sorted.

    EXAMPLES
    --------
    >>> import tempfile
    >>> with tempfile.TemporaryDirectory() as folder:
    ...     root = Path(folder)
    ...     for name in ("b.json", "a.JSONL", "cleaning_manifest.json", ".x.json", "n.txt"):
    ...         _ = (root / name).write_text("{}")
    ...     [p.name for p in discover_input_files(root)]
    ['a.JSONL', 'b.json']
    """
    return sorted(
        path
        for path in input_folder.rglob("*")
        if path.is_file()
        and path.suffix.casefold() in SUPPORTED_EXTENSIONS
        and not path.name.startswith(".")
        and not path.name.endswith("_manifest.json")
    )


def summary_stem(path: Path) -> str:
    r"""
    PURPOSE
    -------
    Strip the cleaning program's suffix so outputs read naturally.

    PARAMETERS
    ----------
    path   Path to the input file.

    RETURNS
    -------
    The file stem with a trailing "_cleaned" removed.

    EXAMPLES
    --------
    >>> summary_stem(Path("physics_cleaned.json"))
    'physics'
    >>> summary_stem(Path("notes.jsonl"))
    'notes'
    """
    stem = path.stem

    if stem.endswith("_cleaned"):
        stem = stem[: -len("_cleaned")]

    return stem


def run_pipeline(
    input_folder: Path = INPUT_FOLDER,
    output_folder: Path = OUTPUT_FOLDER,
    categories: Iterable[str] = CATEGORIES,
    export_images: bool = False,
    verbose: bool = True,
) -> dict[str, Any]:
    r"""
    PURPOSE
    -------
    Scan every input file, write one summary per file, and write a
    manifest.

    PARAMETERS
    ----------
    input_folder    Root folder of cleaned JSON/JSONL files to scan.
    output_folder   Folder summaries, findings and the manifest are written to.
    categories      Which categories of content to keep.
    export_images   Also export page images from the original PDFs when True.
    verbose         Print per-file progress and a final summary when True.

    RETURNS
    -------
    The manifest dictionary describing what was written.

    RAISES
    ------
    FileNotFoundError   input_folder does not exist.
    ValueError          categories contains no valid category.

    EXAMPLES
    --------
    >>> import tempfile
    >>> document = {"document_id": "DOC-1", "chapters": [{
    ...     "chapter_id": "CH-1", "title": "Chapter 1: Stress", "source_page_start": 3,
    ...     "paragraphs": ["Stress is defined as force per unit area, so sigma = F/A. "
    ...                    "Figure 1.2: stress plotted against strain. "
    ...                    "They went home afterwards."]}]}
    >>> with tempfile.TemporaryDirectory() as folder:
    ...     inp, out = Path(folder) / "cleaned", Path(folder) / "summarized"
    ...     inp.mkdir()
    ...     _ = (inp / "materials_cleaned.json").write_text(json.dumps(document), encoding="utf-8")
    ...     manifest = run_pipeline(inp, out, verbose=False)
    ...     written = sorted(p.name for p in out.iterdir())
    ...     summary = (out / "materials_summary.txt").read_text("utf-8")
    >>> written
    ['materials_findings.jsonl', 'materials_summary.txt', 'summarization_manifest.json']
    >>> manifest["total_findings"], manifest["findings_by_category"]
    (3, {'formula': 1, 'concept': 1, 'graph': 1})
    >>> "FORMULAS AND EQUATIONS (1)" in summary, "They went home" in summary
    (True, False)
    >>> run_pipeline(Path("/definitely/not/here"), Path("/tmp/x"))
    Traceback (most recent call last):
        ...
    FileNotFoundError: Input folder does not exist: /definitely/not/here
    """
    if not input_folder.is_dir():
        raise FileNotFoundError(
            f"Input folder does not exist: {input_folder}"
        )

    wanted = [category for category in CATEGORIES if category in set(categories)]

    if not wanted:
        raise ValueError(
            f"No valid categories selected. Choose from: {', '.join(CATEGORIES)}"
        )

    output_folder.mkdir(parents=True, exist_ok=True)

    files = discover_input_files(input_folder)
    file_results: list[dict[str, Any]] = []
    category_totals: dict[str, int] = {}
    total_findings = 0

    for path in files:
        try:
            documents = load_documents(path)
            findings: list[Finding] = []

            for document in documents:
                findings.extend(scan_document(document, path, wanted))

            stem = summary_stem(path)
            summary_path = output_folder / f"{stem}_summary.txt"
            findings_path = output_folder / f"{stem}_findings.jsonl"

            atomic_write_text(
                summary_path,
                render_summary(
                    findings,
                    source_file=path.name,
                    document_id=(
                        documents[0].get("document_id") if documents else None
                    ),
                )
                + "\n",
            )

            write_findings_jsonl(findings_path, findings)

            images: list[str] = []

            if export_images:
                for document in documents:
                    source_path = Path(str(document.get("source_path") or ""))
                    pages = {
                        finding.source_page_start
                        for finding in findings
                        if finding.category in {"graph", "technical_drawing"}
                    }

                    images.extend(
                        export_figure_images(
                            source_path,
                            pages,
                            output_folder / "images" / stem,
                        )
                    )

            for finding in findings:
                category_totals[finding.category] = (
                    category_totals.get(finding.category, 0) + 1
                )

            total_findings += len(findings)

            file_results.append(
                {
                    "status": "success",
                    "input_file": str(path),
                    "summary_file": str(summary_path),
                    "findings_file": str(findings_path),
                    "documents": len(documents),
                    "findings": len(findings),
                    "exported_images": len(images),
                }
            )

            if verbose:
                print(f"{path.name}: {len(findings)} finding(s)")

        except Exception as error:
            file_results.append(
                {
                    "status": "failed",
                    "input_file": str(path),
                    "error": f"{type(error).__name__}: {error}",
                }
            )

            if verbose:
                print(f"{path.name}: FAILED ({type(error).__name__})")

    manifest = {
        "program": "SoakinGarri Technical Summarization Program",
        "input_folder": str(input_folder),
        "output_folder": str(output_folder),
        "source_label": SOURCE_LABEL,
        "categories": wanted,
        "images_exported": export_images,
        "total_files": len(files),
        "successful_files": sum(
            result["status"] == "success" for result in file_results
        ),
        "failed_files": sum(
            result["status"] == "failed" for result in file_results
        ),
        "total_findings": total_findings,
        "findings_by_category": {
            category: category_totals[category]
            for category in CATEGORIES
            if category in category_totals
        },
        "files": file_results,
    }

    atomic_write_json(output_folder / MANIFEST_FILE.name, manifest)

    if verbose:
        print(
            f"\nSummarization completed.\n"
            f"Findings: {total_findings}\n"
            f"Output folder: {output_folder}"
        )

    return manifest


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
    >>> parse_arguments([]).categories
    ['formula', 'concept', 'graph', 'technical_drawing']
    >>> parse_arguments(["--categories", "formula", "graph"]).categories
    ['formula', 'graph']
    >>> parse_arguments(["--export-images"]).export_images
    True
    """
    parser = argparse.ArgumentParser(
        description=(
            "Extract formulas, concepts, graphs and technical drawings from "
            "cleaned documents into plain-text summaries."
        )
    )

    parser.add_argument(
        "--categories",
        nargs="+",
        choices=list(CATEGORIES),
        default=list(CATEGORIES),
        help="which kinds of content to keep",
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
        help="folder for summaries",
    )

    parser.add_argument(
        "--export-images",
        action="store_true",
        help="also export page images from the original PDFs (needs PyMuPDF)",
    )

    return parser.parse_args(argument_list)


def main() -> None:
    """
    PURPOSE
    -------
    Summarize the configured folders using the options given on the
    command line.
    """
    options = parse_arguments()

    run_pipeline(
        input_folder=options.input,
        output_folder=options.output,
        categories=options.categories,
        export_images=options.export_images,
    )


if __name__ == "__main__":
    main()
