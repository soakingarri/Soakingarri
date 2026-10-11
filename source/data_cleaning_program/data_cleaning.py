#!/usr/bin/env python3

"""
SoakinGarri Book and Publication Cleaning Program

Input:
    <repo>/data/ask/raw_data

Output:
    <repo>/data/ask/cleaned_data

Supported input formats:
    PDF, EPUB, DOCX, TXT, MD

Output format:
    One UTF-8 JSON file per book.

Install dependencies:
    python3 -m pip install pymupdf python-docx ebooklib beautifulsoup4 lxml

Run:
    python3 data_cleaning_program.py
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import unicodedata
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(
    0, str(Path(__file__).resolve().parent.parent / "ocr_extraction_program")
)

from common import DATA_ROOT, PROJECT_ROOT, stable_id  # noqa: E402
from ocr_extraction import recognize_text  # noqa: E402

INPUT_FOLDER = DATA_ROOT / "raw_data"
OUTPUT_FOLDER = DATA_ROOT / "cleaned_data"

SUPPORTED_EXTENSIONS = {".pdf", ".epub", ".docx", ".txt", ".md"}

MIN_PARAGRAPH_LENGTH = 2
MIN_PAGE_CONTENT_LENGTH = 35
HEADER_FOOTER_ZONE_LINES = 3
REPEATED_LINE_MIN_PAGES = 3
REPEATED_LINE_RATIO = 0.25
MAX_WORKERS = max(1, min((os.cpu_count() or 2) - 1, 8))

OCR_MIN_TEXT_LENGTH = 20
OCR_RENDER_DPI = 300

PAGE_NUMBER_PATTERN = re.compile(
    r"""(?ix)^\s*
    (?:
        page\s*
    )?
    (?:
        \d{1,5}
        |
        [ivxlcdm]{1,12}
    )
    (?:
        \s*(?:of|/)\s*
        (?:
            \d{1,5}
            |
            [ivxlcdm]{1,12}
        )
    )?
    \s*$"""
)

CHAPTER_PATTERN = re.compile(
    r"""(?ix)^\s*
    (?:
        chapter|chapitre|cap[ií]tulo|part|book|section|unit
    )
    \s+
    (?:
        \d{1,4}|[ivxlcdm]{1,12}|(?-i:[A-Z])
    )
    \b
    (?:
        \s*[:.\-–—]\s*|\s+
    )?
    .{0,160}
    $"""
)
# The number/letter must be a whole token ("Chapter 3", "Part IV", "Part A").
# Without the \b, "chapter expresses..." matched as "Chapter E" + "xpresses",
# and "part of the..." as "Part O" + "f the...", turning ordinary sentences
# that start with those words into chapter titles.

NUMBERED_HEADING_PATTERN = re.compile(
    r"^\s*\d{1,3}(?:\.\d{1,3}){0,4}\.?\s+[A-Z][^\n]{1,150}$"
)
# The optional "\.?" accepts the "13.1. Introduction" style, with a full stop
# after the section number, which the original pattern rejected.

COPYRIGHT_LINE_PATTERN = re.compile(
    r"""(?ix)
    ©
    | \ball\s+rights\s+reserved\b
    | ^\s*(?:\(c\)\s*)?\d{4}\s+[A-Z][\w&.,'’ -]{1,80}?
      \b(?:B\.?V|Ltd|Inc|LLC|GmbH|Press|Publishers?|Publishing|Elsevier|
          Springer|Wiley|Routledge)\b\.?\s*$
    """
)

EDITOR_BYLINE_PATTERN = re.compile(
    r"(?i)^[^.!?]{2,80}\((?:ed|eds|editor|editors)\.?\)\s*$"
)

MAX_BOILERPLATE_LINE_LENGTH = 150

COPYRIGHT_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bcopyright\b",
        r"\ball rights reserved\b",
        r"\bno part of this (?:book|publication|work)\b",
        r"\bpermissions?\b",
        r"\bpublished by\b",
        r"\bfirst published\b",
        r"\bisbn(?:-1[03])?\b",
        r"\blibrary of congress\b",
        r"\bcatalog(?:u)?ing[- ]in[- ]publication\b",
        r"\bprinted in\b",
        r"\blegal notice\b",
        r"\btrademark(?:s)?\b",
    )
]

TOC_TITLE_PATTERN = re.compile(
    r"(?i)^\s*(?:table\s+of\s+contents|contents|contents\s+at\s+a\s+glance)\s*$"
)

TOC_ENTRY_PATTERN = re.compile(
    r"(?i)^.{2,150}(?:\.{2,}|\s{3,})\s*(?:\d{1,5}|[ivxlcdm]{1,12})\s*$"
)

INDEX_TITLE_PATTERN = re.compile(
    r"(?i)^\s*(?:index|subject\s+index|author\s+index|general\s+index)\s*$"
)

INDEX_ENTRY_PATTERN = re.compile(
    r"""(?ix)^
    [a-z][a-z0-9'’\- ]{1,80}
    ,\s*
    \d{1,5}
    (?:
        (?:\s*[-–]\s*|\s*,\s*)\d{1,5}
    )*
    (?:\s*,\s*\d{1,5})*
    \s*$"""
)

FOOTNOTE_PATTERN = re.compile(r"^\s*(?:\[\d+\]|\d{1,3}[.)])\s+\S+")

CONTROL_CHARACTER_PATTERN = re.compile(
    r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]"
)

MULTIPLE_SPACES_PATTERN = re.compile(r"[ \t]{2,}")
MULTIPLE_BLANK_LINES_PATTERN = re.compile(r"\n{3,}")
WORD_PATTERN = re.compile(r"\b[\w'’\-]+\b", re.UNICODE)


@dataclass(slots=True)
class Page:
    """
    PURPOSE
    -------
    A single page (or section) of extracted book text, tagged with its
    position in the source document.

    ATTRIBUTES
    ----------
    number   1-based position of this page/section in the source document.
    text     The page's extracted text content.
    """

    number: int
    text: str


def normalize_unicode(text: str) -> str:
    r"""
    PURPOSE
    -------
    Fold text to a consistent, control-character-free form.

    PARAMETERS
    ----------
    text   The raw text to normalize.

    RETURNS
    -------
    The text in NFKC form with control characters, soft hyphens, BOM, and
    zero-width spaces removed, and line endings normalized to `\n`.

    EXAMPLES
    --------
    >>> normalize_unicode("caf\u00e9\r\n")
    'café\n'
    >>> normalize_unicode("a\x0cb")
    'ab'
    >>> normalize_unicode("soft\u00adhyphen")
    'softhyphen'
    """
    text = unicodedata.normalize("NFKC", text)
    text = CONTROL_CHARACTER_PATTERN.sub("", text)
    text = text.replace("\u00ad", "")
    text = text.replace("\ufeff", "")
    text = text.replace("\u200b", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return text


def normalize_line_for_comparison(line: str) -> str:
    r"""
    PURPOSE
    -------
    Fold a line to a form suitable for repeated-line and dedup comparisons.
    Digits collapse to a single `#` so that "Page 12" and "Page 97" compare
    equal, punctuation is dropped, and case and whitespace are normalized.

    PARAMETERS
    ----------
    line   The line of text to normalize.

    RETURNS
    -------
    The normalized comparison key for the line.

    EXAMPLES
    --------
    >>> normalize_line_for_comparison("  Page 12  ")
    'page #'
    >>> normalize_line_for_comparison("Chapter 3: The Beginning!")
    'chapter # the beginning'
    """
    line = normalize_unicode(line).casefold()
    line = re.sub(r"\d+", "#", line)
    line = re.sub(r"\s+", " ", line)
    line = re.sub(r"[^\w# ]+", "", line)
    return line.strip()


def nonempty_lines(text: str) -> list[str]:
    r"""
    PURPOSE
    -------
    Split text into its non-blank, stripped lines.

    PARAMETERS
    ----------
    text   The text to split into lines.

    RETURNS
    -------
    The list of non-blank lines, each stripped of surrounding whitespace.

    EXAMPLES
    --------
    >>> nonempty_lines("first\n\n  second  \n\nthird\n")
    ['first', 'second', 'third']
    """
    return [line.strip() for line in text.splitlines() if line.strip()]


def ocr_page_image(page: Any, dpi: int = OCR_RENDER_DPI) -> str:
    r"""
    PURPOSE
    -------
    Run OCR over a rendered image of one PDF page. This is the fallback
    for pages with no usable text layer — a photographed or scanned page
    has no embedded text for extract_pdf's normal extraction to find, so
    it would otherwise come out empty.

    PARAMETERS
    ----------
    page   A PyMuPDF (fitz) page object to render and recognize text on.
    dpi    Resolution to render the page at before recognition; higher
           values improve accuracy on small print at the cost of speed.

    RETURNS
    -------
    The OCR-recognized text for the page (possibly empty, if the page is
    genuinely blank).

    RAISES
    ------
    RuntimeError   If pytesseract/Pillow are not installed, or the
                   Tesseract OCR engine itself is not installed.
    """
    try:
        from PIL import Image
    except ImportError as error:
        raise RuntimeError(
            "OCR support requires pytesseract and Pillow: "
            "python3 -m pip install pytesseract pillow"
        ) from error

    pixmap = page.get_pixmap(dpi=dpi)
    image = Image.frombytes(
        "RGB", (pixmap.width, pixmap.height), pixmap.samples
    )

    return recognize_text(image)


def extract_pdf(path: Path) -> list[Page]:
    r"""
    PURPOSE
    -------
    Extract text from a PDF file, one Page per PDF page. A page whose
    normal text-layer extraction comes back near-empty is assumed to be a
    photographed or scanned page and falls back to OCR over a rendered
    image of it (see :func:`ocr_page_image`), so books with no embedded
    text layer still produce real content instead of empty pages.

    PARAMETERS
    ----------
    path   Path to the PDF file.

    RETURNS
    -------
    The list of extracted pages, in document order.

    RAISES
    ------
    RuntimeError   If PyMuPDF (fitz) is not installed, or a page needed
                   OCR and pytesseract/Tesseract are not installed.
    """
    try:
        import fitz
    except ImportError as error:
        raise RuntimeError(
            "PDF support requires PyMuPDF: "
            "python3 -m pip install pymupdf"
        ) from error

    pages: list[Page] = []

    with fitz.open(path) as document:
        for page_number, page in enumerate(document, start=1):
            blocks = page.get_text("blocks", sort=True)
            text = "\n".join(
                block[4].strip()
                for block in blocks
                if len(block) > 4 and block[4].strip()
            )

            if len(text) < OCR_MIN_TEXT_LENGTH:
                text = ocr_page_image(page)

            pages.append(Page(page_number, normalize_unicode(text)))

    return pages


def extract_docx(path: Path) -> list[Page]:
    r"""
    PURPOSE
    -------
    Extract text from a DOCX file, splitting into sections at heading and
    title paragraphs.

    PARAMETERS
    ----------
    path   Path to the DOCX file.

    RETURNS
    -------
    The list of extracted, non-empty sections, in document order.

    RAISES
    ------
    RuntimeError   If python-docx is not installed.
    """
    try:
        from docx import Document
    except ImportError as error:
        raise RuntimeError(
            "DOCX support requires python-docx: "
            "python3 -m pip install python-docx"
        ) from error

    document = Document(path)
    sections: list[list[str]] = [[]]
    current_section = sections[0]

    for paragraph in document.paragraphs:
        text = normalize_unicode(paragraph.text).strip()

        if not text:
            if current_section and current_section[-1] != "":
                current_section.append("")
            continue

        style_name = (
            paragraph.style.name.casefold()
            if paragraph.style and paragraph.style.name
            else ""
        )

        if (
            ("heading" in style_name or "title" in style_name)
            and current_section
            and any(current_section)
        ):
            current_section = []
            sections.append(current_section)

        current_section.append(text)

    return [
        Page(number=index, text="\n".join(section))
        for index, section in enumerate(sections, start=1)
        if any(line.strip() for line in section)
    ]


def extract_epub(path: Path) -> list[Page]:
    r"""
    PURPOSE
    -------
    Extract text from an EPUB file, one Page per document item that has
    non-empty content.

    PARAMETERS
    ----------
    path   Path to the EPUB file.

    RETURNS
    -------
    The list of extracted pages, in document order.

    RAISES
    ------
    RuntimeError   If ebooklib, beautifulsoup4, or lxml is not installed.
    """
    try:
        from bs4 import BeautifulSoup
        from ebooklib import ITEM_DOCUMENT, epub
    except ImportError as error:
        raise RuntimeError(
            "EPUB support requires ebooklib, beautifulsoup4, and lxml: "
            "python3 -m pip install ebooklib beautifulsoup4 lxml"
        ) from error

    book = epub.read_epub(str(path), options={"ignore_ncx": True})
    pages: list[Page] = []
    page_number = 1

    for item in book.get_items_of_type(ITEM_DOCUMENT):
        soup = BeautifulSoup(item.get_content(), "lxml")

        for unwanted in soup(
            ["script", "style", "nav", "svg", "noscript", "form"]
        ):
            unwanted.decompose()

        blocks: list[str] = []

        for element in soup.find_all(
            ["h1", "h2", "h3", "h4", "h5", "h6", "p", "blockquote", "li"]
        ):
            text = " ".join(element.get_text(" ", strip=True).split())

            if text:
                if element.name.startswith("h"):
                    blocks.extend(["", text, ""])
                else:
                    blocks.append(text)

        text = normalize_unicode("\n".join(blocks)).strip()

        if text:
            pages.append(Page(page_number, text))
            page_number += 1

    return pages


def extract_text_file(path: Path) -> list[Page]:
    r"""
    PURPOSE
    -------
    Split a plain-text or Markdown file into pages. Pages are split on
    form-feed characters first; if none are found, on lines that are a
    bare page number instead.

    PARAMETERS
    ----------
    path   Path to the text or Markdown file.

    RETURNS
    -------
    The list of extracted, non-empty pages, in document order.

    EXAMPLES
    --------
    >>> import tempfile
    >>> with tempfile.TemporaryDirectory() as folder:
    ...     path = Path(folder) / "book.txt"
    ...     _ = path.write_text("Only one page, no separators.\n", encoding="utf-8")
    ...     [(p.number, p.text) for p in extract_text_file(path)]
    [(1, 'Only one page, no separators.')]
    """
    raw_bytes = path.read_bytes()

    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            text = raw_bytes.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        text = raw_bytes.decode("utf-8", errors="replace")

    page_chunks = re.split(r"\f+", text)

    if len(page_chunks) == 1:
        page_chunks = re.split(
            r"(?im)^\s*(?:page\s+)?\d{1,5}\s*$",
            text,
        )

    return [
        Page(number=index, text=normalize_unicode(chunk).strip())
        for index, chunk in enumerate(page_chunks, start=1)
        if normalize_unicode(chunk).strip()
    ]


def extract_pages(path: Path) -> list[Page]:
    r"""
    PURPOSE
    -------
    Extract pages from a book file, dispatching on its extension.

    PARAMETERS
    ----------
    path   Path to the book file.

    RETURNS
    -------
    The list of extracted pages, in document order.

    RAISES
    ------
    ValueError   If the file's extension is not a supported format.

    EXAMPLES
    --------
    >>> import tempfile
    >>> with tempfile.TemporaryDirectory() as folder:
    ...     path = Path(folder) / "book.txt"
    ...     _ = path.write_text("Page one text.\n1\n\fPage two text.\n2\n", encoding="utf-8")
    ...     [(p.number, p.text) for p in extract_pages(path)]
    [(1, 'Page one text.\n1'), (2, 'Page two text.\n2')]
    """
    extension = path.suffix.casefold()

    if extension == ".pdf":
        return extract_pdf(path)

    if extension == ".docx":
        return extract_docx(path)

    if extension == ".epub":
        return extract_epub(path)

    if extension in {".txt", ".md"}:
        return extract_text_file(path)

    raise ValueError(f"Unsupported file type: {extension}")


def identify_repeated_marginal_lines(pages: list[Page]) -> set[str]:
    r"""
    PURPOSE
    -------
    Find lines near the top or bottom of pages that repeat often enough
    to be running headers, footers, or page numbers.

    PARAMETERS
    ----------
    pages   The book's pages to scan for repeated marginal lines.

    RETURNS
    -------
    The set of normalized line values that qualify as repeated marginal
    lines.

    EXAMPLES
    --------
    >>> pages = [Page(1, "Header\nBody one\n1"),
    ...          Page(2, "Header\nBody two\n2"),
    ...          Page(3, "Header\nBody three\n3")]
    >>> sorted(identify_repeated_marginal_lines(pages))
    ['#', 'header']
    >>> identify_repeated_marginal_lines([Page(1, "x")])
    set()
    """
    if len(pages) < REPEATED_LINE_MIN_PAGES:
        return set()

    candidate_counts: Counter[str] = Counter()

    for page in pages:
        lines = nonempty_lines(page.text)

        if not lines:
            continue

        candidates = (
            lines[:HEADER_FOOTER_ZONE_LINES]
            + lines[-HEADER_FOOTER_ZONE_LINES:]
        )

        normalized_candidates = {
            normalize_line_for_comparison(line)
            for line in candidates
            if 0 < len(line) <= 180
        }

        candidate_counts.update(
            value for value in normalized_candidates if value
        )

    required_count = max(
        REPEATED_LINE_MIN_PAGES,
        int(len(pages) * REPEATED_LINE_RATIO + 0.999),
    )

    return {
        line
        for line, count in candidate_counts.items()
        if count >= required_count
    }


def remove_headers_footers_and_page_numbers(
    page: Page,
    repeated_lines: set[str],
) -> Page:
    r"""
    PURPOSE
    -------
    Strip a page's running headers/footers and bare page numbers.

    PARAMETERS
    ----------
    page             The page to clean.
    repeated_lines   Normalized marginal-line values to strip, as returned
                      by identify_repeated_marginal_lines.

    RETURNS
    -------
    A new Page with marginal header/footer/page-number lines removed.

    EXAMPLES
    --------
    >>> page = Page(3, "Header\nChapter 1: The Start\n3")
    >>> remove_headers_footers_and_page_numbers(page, {"header", "#"}).text
    'Chapter 1: The Start'
    """
    lines = page.text.splitlines()
    nonempty_positions = [
        index for index, line in enumerate(lines) if line.strip()
    ]

    if not nonempty_positions:
        return Page(page.number, "")

    marginal_positions = set(
        nonempty_positions[:HEADER_FOOTER_ZONE_LINES]
        + nonempty_positions[-HEADER_FOOTER_ZONE_LINES:]
    )

    cleaned_lines: list[str] = []

    for position, line in enumerate(lines):
        stripped = line.strip()

        if not stripped:
            cleaned_lines.append("")
            continue

        if position in marginal_positions:
            normalized = normalize_line_for_comparison(stripped)

            if normalized in repeated_lines:
                continue

            if PAGE_NUMBER_PATTERN.fullmatch(stripped):
                continue

        cleaned_lines.append(stripped)

    return Page(
        page.number,
        MULTIPLE_BLANK_LINES_PATTERN.sub(
            "\n\n",
            "\n".join(cleaned_lines),
        ).strip(),
    )


def toc_score(lines: list[str]) -> int:
    r"""
    PURPOSE
    -------
    Score how much a page looks like a table of contents.

    PARAMETERS
    ----------
    lines   The page's non-blank lines.

    RETURNS
    -------
    An integer score; higher means more likely to be a table of contents.

    EXAMPLES
    --------
    >>> toc_score(["Table of Contents",
    ...            "Chapter 1 .......... 3",
    ...            "Chapter 2 .......... 9"])
    8
    """
    score = 0

    if any(TOC_TITLE_PATTERN.fullmatch(line) for line in lines[:8]):
        score += 6

    toc_entries = sum(bool(TOC_ENTRY_PATTERN.match(line)) for line in lines)
    score += min(toc_entries, 10)

    short_numbered_lines = sum(
        bool(re.search(r"(?:\d{1,5}|[ivxlcdm]{1,12})\s*$", line, re.I))
        and len(line) < 150
        for line in lines
    )

    if len(lines) >= 5 and short_numbered_lines / len(lines) >= 0.45:
        score += 4

    return score


def copyright_score(text: str, lines: list[str]) -> int:
    r"""
    PURPOSE
    -------
    Score how much a page looks like a copyright or legal notice page.

    PARAMETERS
    ----------
    text    The page's full text.
    lines   The page's non-blank lines.

    RETURNS
    -------
    An integer score; higher means more likely to be a copyright/legal
    page.

    EXAMPLES
    --------
    >>> lines = ["My Great Book",
    ...          "© 2021 Example Press. All rights reserved.",
    ...          "ISBN 978-0-00-000000-0. Printed in Lagos."]
    >>> copyright_score("\n".join(lines), lines)
    10
    """
    sample = "\n".join(lines[:80])
    matches = sum(bool(pattern.search(sample)) for pattern in COPYRIGHT_PATTERNS)
    score = matches * 2

    if len(text) < 3000 and matches >= 2:
        score += 4

    if re.search(r"(?i)\b©\s*\d{4}\b", sample):
        score += 3

    return score


def index_score(lines: list[str]) -> int:
    r"""
    PURPOSE
    -------
    Score how much a page looks like a back-of-book index.

    PARAMETERS
    ----------
    lines   The page's non-blank lines.

    RETURNS
    -------
    An integer score; higher means more likely to be an index page.

    EXAMPLES
    --------
    >>> index_score(["Index", "apples, 3, 7", "bananas, 12-14",
    ...              "cherries, 20", "dates, 21"])
    16
    """
    score = 0

    if any(INDEX_TITLE_PATTERN.fullmatch(line) for line in lines[:8]):
        score += 7

    entries = sum(bool(INDEX_ENTRY_PATTERN.match(line)) for line in lines)

    if entries >= 4:
        score += min(entries, 10)

    if lines and entries / len(lines) >= 0.35:
        score += 5

    return score


def classify_removable_page(
    page: Page,
    page_position: int,
    total_pages: int,
    index_sequence_active: bool,
) -> tuple[str | None, bool]:
    r"""
    PURPOSE
    -------
    Decide whether a page should be dropped as non-content, and why.

    PARAMETERS
    ----------
    page                     The page to classify.
    page_position            The page's 0-based position in the book.
    total_pages              Total number of pages in the book.
    index_sequence_active    Whether the previous page was classified as
                              part of a back-of-book index run.

    RETURNS
    -------
    A (reason, index_sequence_active) tuple: `reason` is the removal
    reason (or None to keep the page), and `index_sequence_active`
    reports whether this page continues/starts an index run.

    EXAMPLES
    --------
    >>> classify_removable_page(Page(1, ""), 0, 10, False)
    ('blank_or_nearly_blank', False)
    >>> toc_page = Page(1, "Table of Contents\n" +
    ...     "\n".join(f"Chapter {i} .......... {i}" for i in range(1, 6)))
    >>> classify_removable_page(toc_page, 0, 10, False)
    ('table_of_contents', False)
    """
    text = page.text.strip()
    lines = nonempty_lines(text)

    if not text or len(text) < MIN_PAGE_CONTENT_LENGTH:
        return "blank_or_nearly_blank", False

    front_matter_limit = max(12, int(total_pages * 0.12))
    back_matter_start = max(0, int(total_pages * 0.82))

    current_toc_score = toc_score(lines)
    current_copyright_score = copyright_score(text, lines)
    current_index_score = index_score(lines)

    if page_position < front_matter_limit:
        if current_toc_score >= 7:
            return "table_of_contents", False

        if current_copyright_score >= 6:
            return "copyright_or_legal_page", False

    is_index_page = (
        page_position >= back_matter_start
        and current_index_score >= 7
    )

    if is_index_page:
        return "index", True

    if index_sequence_active and page_position >= back_matter_start:
        index_entries = sum(
            bool(INDEX_ENTRY_PATTERN.match(line)) for line in lines
        )

        if index_entries >= 3:
            return "index", True

    return None, False


def remove_non_content_pages(
    pages: list[Page],
) -> tuple[list[Page], list[dict]]:
    r"""
    PURPOSE
    -------
    Drop blank, table-of-contents, copyright and index pages.

    PARAMETERS
    ----------
    pages   The book's pages to filter.

    RETURNS
    -------
    A (retained, removed) tuple: the pages kept as content, and a list of
    {"page_number", "reason"} records for the pages dropped.

    EXAMPLES
    --------
    >>> pages = [Page(1, "This page has enough real content to be kept as-is."),
    ...          Page(2, "")]
    >>> retained, removed = remove_non_content_pages(pages)
    >>> [(p.number, p.text) for p in retained]
    [(1, 'This page has enough real content to be kept as-is.')]
    >>> removed
    [{'page_number': 2, 'reason': 'blank_or_nearly_blank'}]
    """
    retained: list[Page] = []
    removed: list[dict] = []
    index_sequence_active = False
    total_pages = len(pages)

    for position, page in enumerate(pages):
        reason, index_sequence_active = classify_removable_page(
            page=page,
            page_position=position,
            total_pages=total_pages,
            index_sequence_active=index_sequence_active,
        )

        if reason:
            removed.append(
                {
                    "page_number": page.number,
                    "reason": reason,
                }
            )
        else:
            retained.append(page)

    return retained, removed


def dehyphenate_lines(text: str) -> str:
    r"""
    PURPOSE
    -------
    Rejoin words split across a line break by a hyphen. Hyphenated proper
    nouns (a capital letter after the break, like "Jean-\nPaul") are left
    alone; a lowercase continuation is treated as a wrapped word and
    rejoined.

    PARAMETERS
    ----------
    text   The text to dehyphenate.

    RETURNS
    -------
    The text with wrapped-word hyphenation removed.

    EXAMPLES
    --------
    >>> dehyphenate_lines("re-\nuse")
    'reuse'
    >>> dehyphenate_lines("Jean-\nPaul")
    'Jean-\nPaul'
    """
    return re.sub(
        r"(?<=\w)-\n(?=[a-zà-öø-ÿ])",
        "",
        text,
    )


def looks_like_heading(text: str) -> bool:
    r"""
    PURPOSE
    -------
    Guess whether a line of text is a heading rather than body prose.

    PARAMETERS
    ----------
    text   The line of text to classify.

    RETURNS
    -------
    True if the line looks like a heading, False otherwise.

    EXAMPLES
    --------
    >>> looks_like_heading("Chapter 1: The Start")
    True
    >>> looks_like_heading("THE MIDDLE")
    True
    >>> looks_like_heading("This is a normal sentence.")
    False
    >>> looks_like_heading("13.1. Introduction")
    True
    >>> looks_like_heading("Part IV The Results")
    True
    >>> looks_like_heading("chapter expresses a more cautious view than the one which I think is still")
    False
    >>> looks_like_heading("part of the essential fabric of social life. But when should a gift be returned? Not")
    False
    """
    text = " ".join(text.split())

    if not text or len(text) > 180:
        return False

    if CHAPTER_PATTERN.match(text):
        return True

    if NUMBERED_HEADING_PATTERN.match(text):
        return True

    words = WORD_PATTERN.findall(text)

    if not 1 <= len(words) <= 14:
        return False

    if text.endswith((".", "?", "!", ";", ",")):
        return False

    letters = [character for character in text if character.isalpha()]

    if not letters:
        return False

    uppercase_ratio = sum(character.isupper() for character in letters) / len(
        letters
    )

    title_case_words = sum(
        word[0].isupper()
        for word in words
        if word and word[0].isalpha()
    )

    return (
        uppercase_ratio >= 0.75
        or title_case_words / len(words) >= 0.80
    )


def is_strong_heading(text: str) -> bool:
    r"""
    PURPOSE
    -------
    True for headings with an explicit structural marker ("Chapter 3",
    "Part IV", "13.2 ..."). These are trusted wherever they appear;
    weaker cues (ALL CAPS, Title Case) are only trusted at a paragraph
    boundary, see :func:`paragraphs_from_pages`.

    PARAMETERS
    ----------
    text   The line of text to classify.

    RETURNS
    -------
    True if the line carries an explicit chapter/section marker.

    EXAMPLES
    --------
    >>> is_strong_heading("13.2. Social construction of time")
    True
    >>> is_strong_heading("Chapter 1: The Start")
    True
    >>> is_strong_heading("Kedang")
    False
    """
    text = " ".join(text.split())

    if not text or len(text) > 180:
        return False

    return bool(
        CHAPTER_PATTERN.match(text) or NUMBERED_HEADING_PATTERN.match(text)
    )


def is_boilerplate_line(line: str) -> bool:
    r"""
    PURPOSE
    -------
    True for short standalone lines that are publishing boilerplate rather
    than content: copyright/publisher notices and editor bylines. Whole
    copyright *pages* are already dropped by remove_non_content_pages;
    this catches single notice lines printed on content pages, such as
    the first page of a chapter in an edited volume.

    PARAMETERS
    ----------
    line   One line of page text.

    RETURNS
    -------
    True if the line should be dropped.

    EXAMPLES
    --------
    >>> is_boilerplate_line("2000 Elsevier Science B.V.")
    True
    >>> is_boilerplate_line("© 2021 Example Press. All rights reserved.")
    True
    >>> is_boilerplate_line("Patrick Baert (Editor)")
    True
    >>> is_boilerplate_line("In 1939 Evans-Pritchard argued that time was social.")
    False
    """
    line = " ".join(line.split())

    if not line or len(line) > MAX_BOILERPLATE_LINE_LENGTH:
        return False

    return bool(
        COPYRIGHT_LINE_PATTERN.search(line)
        or EDITOR_BYLINE_PATTERN.match(line)
    )


def clean_line(line: str) -> str:
    r"""
    PURPOSE
    -------
    Normalize whitespace and spacing around punctuation in one line.

    PARAMETERS
    ----------
    line   The line of text to clean.

    RETURNS
    -------
    The cleaned line.

    EXAMPLES
    --------
    >>> clean_line("  Hello   , world  !  ")
    'Hello, world!'
    >>> clean_line("( hello )")
    '(hello)'
    """
    line = normalize_unicode(line)
    line = MULTIPLE_SPACES_PATTERN.sub(" ", line)
    line = re.sub(r"\s+([,.;:!?])", r"\1", line)
    line = re.sub(r"([(\[“‘])\s+", r"\1", line)
    line = re.sub(r"\s+([)\]”’])", r"\1", line)
    return line.strip()


SENTENCE_END_PATTERN = re.compile(r'[.!?]["”’\')\]]*$')


def paragraphs_from_pages(pages: list[Page]) -> list[dict]:
    r"""
    PURPOSE
    -------
    Join wrapped lines into paragraphs across the whole book, splitting on
    headings and blank lines.

    Two rules keep sentences intact and stop stray words becoming
    chapter titles:

    1. Paragraphs continue across page breaks. A sentence that starts at
       the bottom of page 4 and finishes on page 5 stays one paragraph
       (page=4, page_end=5) instead of being cut in two.
    2. A line only counts as a weak heading (ALL CAPS / Title Case) when
       it starts a new paragraph, i.e. the previous line ended a sentence
       or was blank. Mid-sentence, a lone "Kedang" or "Drinking" is just
       a word on its own line. Strong headings ("Chapter 3", "13.2 ...")
       count anywhere.

    Short publisher/copyright notice lines and editor bylines are dropped.

    PARAMETERS
    ----------
    pages   The book's content pages, in reading order.

    RETURNS
    -------
    A list of {"text", "page", "page_end", "is_heading"} dicts, with
    consecutive duplicates removed.

    EXAMPLES
    --------
    >>> pages = [Page(4, "13.1. Introduction\nThe argument runs onto"),
    ...          Page(5, "the next page. Among the\nKedang\npeople, time is cyclic.")]
    >>> [(p["text"], p["page"], p["page_end"], p["is_heading"])
    ...  for p in paragraphs_from_pages(pages)]
    [('13.1. Introduction', 4, 4, True), ('The argument runs onto the next page. Among the Kedang people, time is cyclic.', 4, 5, False)]
    >>> pages = [Page(1, "2000 Elsevier Science B.V.\nPatrick Baert (Editor)\nTHE MIDDLE\nBody text here.")]
    >>> [(p["text"], p["is_heading"]) for p in paragraphs_from_pages(pages)]
    [('THE MIDDLE', True), ('Body text here.', False)]
    """
    paragraphs: list[dict] = []
    buffer: list[str] = []
    buffer_start_page: int | None = None
    buffer_end_page: int | None = None

    def flush_buffer() -> None:
        """Flush the buffered line(s) as one cleaned paragraph, if any."""
        nonlocal buffer_start_page, buffer_end_page

        if not buffer:
            return

        paragraph = clean_line(" ".join(buffer))
        buffer.clear()

        if len(paragraph) >= MIN_PARAGRAPH_LENGTH:
            paragraphs.append(
                {
                    "text": paragraph,
                    "page": buffer_start_page,
                    "page_end": buffer_end_page,
                    "is_heading": False,
                }
            )

        buffer_start_page = None
        buffer_end_page = None

    def add_heading(line: str, page_number: int) -> None:
        """Record a standalone heading paragraph."""
        paragraphs.append(
            {
                "text": line,
                "page": page_number,
                "page_end": page_number,
                "is_heading": True,
            }
        )

    for page in pages:
        text = dehyphenate_lines(normalize_unicode(page.text))

        for raw_line in text.splitlines():
            line = clean_line(raw_line)

            if not line:
                flush_buffer()
                continue

            if is_boilerplate_line(line):
                continue

            if is_strong_heading(line):
                flush_buffer()
                add_heading(line, page.number)
                continue

            if not buffer and looks_like_heading(line):
                add_heading(line, page.number)
                continue

            if FOOTNOTE_PATTERN.match(line) and buffer:
                flush_buffer()

            if not buffer:
                buffer_start_page = page.number

            buffer.append(line)
            buffer_end_page = page.number

            if SENTENCE_END_PATTERN.search(line):
                flush_buffer()

        # Deliberately no flush at the end of a page: an unfinished
        # sentence carries over to the next page.

    flush_buffer()

    deduplicated: list[dict] = []

    for paragraph in paragraphs:
        if (
            not deduplicated
            or normalize_line_for_comparison(paragraph["text"])
            != normalize_line_for_comparison(deduplicated[-1]["text"])
        ):
            deduplicated.append(paragraph)

    return deduplicated


def paragraphs_from_text(text: str) -> list[str]:
    r"""
    PURPOSE
    -------
    Join wrapped lines of a single page into paragraphs; a convenience
    wrapper around :func:`paragraphs_from_pages`.

    PARAMETERS
    ----------
    text   The page text to split into paragraphs.

    RETURNS
    -------
    The list of paragraphs (including standalone headings), deduplicated.

    EXAMPLES
    --------
    >>> text = ("Chapter 1: The Start\n"
    ...         "The first sentence runs onto\n"
    ...         "the next line.\n\n"
    ...         "A second paragraph.")
    >>> paragraphs_from_text(text)
    ['Chapter 1: The Start', 'The first sentence runs onto the next line.', 'A second paragraph.']
    >>> paragraphs_from_text("a\n\nb\n")
    []
    """
    return [
        paragraph["text"]
        for paragraph in paragraphs_from_pages([Page(1, text)])
    ]


def restructure_into_chapters(
    pages: list[Page],
    document_id: str,
) -> list[dict]:
    r"""
    PURPOSE
    -------
    Group a book's pages into chapters of headings and paragraphs. Chapter
    and paragraph ids are derived from ``document_id`` plus each chapter's
    position and title (and each paragraph's position and text), so
    re-cleaning the same file produces the same ids every time — a vector
    store built from them can be updated in place instead of growing a
    fresh duplicate entry on every run.

    PARAMETERS
    ----------
    pages         The book's retained content pages, in reading order.
    document_id   Stable id of the document these pages belong to.

    RETURNS
    -------
    The list of chapter dicts, each with chapter_id, title,
    source_page_start, source_page_end, and paragraphs (each paragraph
    dict also carries its own ``page`` number, for per-paragraph
    citations).

    EXAMPLES
    --------
    >>> pages = [Page(1, "Chapter 1: The Start\nFirst paragraph here."),
    ...          Page(2, "Second paragraph continues.")]
    >>> chapters = restructure_into_chapters(pages, "DOC-test")
    >>> [c["title"] for c in chapters]
    ['Chapter 1: The Start']
    >>> [p["text"] for c in chapters for p in c["paragraphs"]]
    ['First paragraph here.', 'Second paragraph continues.']
    >>> [p["page"] for c in chapters for p in c["paragraphs"]]
    [1, 2]
    >>> chapters[0]["source_page_start"], chapters[0]["source_page_end"]
    (1, 2)
    >>> restructure_into_chapters(pages, "DOC-test") == chapters
    True
    >>> # a sentence crossing a page break stays one paragraph, pages 3-4
    >>> pages = [Page(3, "13.1. Introduction\nAnthropologists have written a whole"),
    ...          Page(4, "chapter on this topic.\n\nA new sentence starts here.")]
    >>> chapters = restructure_into_chapters(pages, "DOC-test")
    >>> chapters[0]["title"]
    '13.1. Introduction'
    >>> [(p["text"], p["page"], p["page_end"]) for p in chapters[0]["paragraphs"]]
    [('Anthropologists have written a whole chapter on this topic.', 3, 4), ('A new sentence starts here.', 4, 4)]
    """
    chapters: list[dict] = []
    chapter_index = 0
    current_title = "Main Text"
    current_chapter_id = stable_id("CH", document_id, "0", current_title)
    current_paragraphs: list[dict] = []
    current_start_page: int | None = None
    current_end_page: int | None = None

    def save_current_chapter() -> None:
        """Save the in-progress chapter's paragraphs as a chapter record,
        if any."""
        nonlocal current_paragraphs
        nonlocal current_start_page
        nonlocal current_end_page

        if not current_paragraphs:
            return

        chapters.append(
            {
                "chapter_id": current_chapter_id,
                "title": current_title,
                "source_page_start": current_start_page,
                "source_page_end": current_end_page,
                "paragraphs": current_paragraphs,
            }
        )

        current_paragraphs = []
        current_start_page = None
        current_end_page = None

    for item in paragraphs_from_pages(pages):
        paragraph = item["text"]

        if item["is_heading"]:
            save_current_chapter()
            chapter_index += 1
            current_title = paragraph
            current_chapter_id = stable_id(
                "CH", document_id, str(chapter_index), paragraph
            )
            current_start_page = item["page"]
            current_end_page = item["page"]
            continue

        if current_start_page is None:
            current_start_page = item["page"]

        current_end_page = item["page_end"]
        paragraph_index = len(current_paragraphs)
        current_paragraphs.append(
            {
                "paragraph_id": stable_id(
                    "P", current_chapter_id, str(paragraph_index), paragraph
                ),
                "text": paragraph,
                "page": item["page"],
                "page_end": item["page_end"],
            }
        )

    save_current_chapter()

    return chapters


def count_paragraphs(chapters: Iterable[dict]) -> int:
    r"""
    PURPOSE
    -------
    Count paragraphs across every chapter.

    PARAMETERS
    ----------
    chapters   The chapters to count paragraphs in.

    RETURNS
    -------
    The total number of paragraphs.

    EXAMPLES
    --------
    >>> count_paragraphs([{"paragraphs": [1, 2]}, {"paragraphs": [3]}])
    3
    """
    return sum(len(chapter["paragraphs"]) for chapter in chapters)


WEB_FOLDER_NAME = "web"
WEB_MANIFEST_NAME = "web_ingestion_manifest.json"
REGISTRY_RELATIVE_PATH = Path("metadata") / "source_registry.json"


def source_attributes(
    path: Path,
    data_folder: Path = PROJECT_ROOT / "data",
) -> dict[str, Any]:
    r"""
    PURPOSE
    -------
    Work out the retrieval filter fields for one raw source file:

        tool          The product folder the file sits in:
                       data/ask/... -> "ask", data/meme_generator/... ->
                       "meme_generator"; None outside data_folder.
        source_type   "web" for pages web_ingestion.py saved under a
                       raw_data/web folder, otherwise "book".
        date          For a web page, its fetch timestamp from
                       web_ingestion_manifest.json; for a book, its year
                       (as a string) from that tool's source_registry.json,
                       matched by file name. None when unknown.

    Missing or unreadable manifests and registries are treated as empty,
    so cleaning never fails because of them.

    PARAMETERS
    ----------
    path          The raw source file.
    data_folder   The repo's data folder, whose children are tool folders.

    RETURNS
    -------
    A dict with "tool", "source_type" and "date".

    EXAMPLES
    --------
    >>> import json, tempfile
    >>> with tempfile.TemporaryDirectory() as folder:
    ...     data = Path(folder)
    ...     web = data / "ask" / "raw_data" / "web"
    ...     web.mkdir(parents=True)
    ...     _ = (web / WEB_MANIFEST_NAME).write_text(json.dumps({"known_pages": [
    ...         {"url": "u", "output_file": "WEB-1.txt", "content_hash": "h",
    ...          "fetch_timestamp": "2026-10-01T09:00:00+00:00"}]}), encoding="utf-8")
    ...     registry = data / "ask" / REGISTRY_RELATIVE_PATH
    ...     registry.parent.mkdir(parents=True)
    ...     _ = registry.write_text(json.dumps({"SRC-1": {
    ...         "source_file": "gell.pdf", "year": 1998}}), encoding="utf-8")
    ...     page = source_attributes(web / "WEB-1.txt", data)
    ...     book = source_attributes(data / "ask" / "raw_data" / "gell.pdf", data)
    ...     unknown = source_attributes(data / "meme_generator" / "raw_data" / "x.pdf", data)
    ...     outside = source_attributes(Path("/elsewhere/x.pdf"), data)
    >>> page
    {'tool': 'ask', 'source_type': 'web', 'date': '2026-10-01T09:00:00+00:00'}
    >>> book
    {'tool': 'ask', 'source_type': 'book', 'date': '1998'}
    >>> unknown
    {'tool': 'meme_generator', 'source_type': 'book', 'date': None}
    >>> outside
    {'tool': None, 'source_type': 'book', 'date': None}
    """
    path = path.resolve()
    data_folder = data_folder.resolve()

    try:
        tool: str | None = path.relative_to(data_folder).parts[0]
    except (ValueError, IndexError):
        tool = None

    is_web = path.parent.name == WEB_FOLDER_NAME
    date: str | None = None

    if is_web:
        try:
            manifest = json.loads(
                (path.parent / WEB_MANIFEST_NAME).read_text(encoding="utf-8")
            )
        except (OSError, ValueError):
            manifest = {}

        pages = manifest.get("known_pages") or manifest.get("results") or []

        for page in pages:
            if page.get("output_file") == path.name and page.get("fetch_timestamp"):
                date = str(page["fetch_timestamp"])
                break

    elif tool is not None:
        try:
            registry = json.loads(
                (data_folder / tool / REGISTRY_RELATIVE_PATH).read_text(encoding="utf-8")
            )
        except (OSError, ValueError):
            registry = {}

        for record in registry.values() if isinstance(registry, dict) else []:
            if record.get("source_file") == path.name and record.get("year"):
                date = str(record["year"])
                break

    return {
        "tool": tool,
        "source_type": WEB_FOLDER_NAME if is_web else "book",
        "date": date,
    }


def clean_book(path: Path) -> dict:
    r"""
    PURPOSE
    -------
    Run the full cleaning pipeline on one book file.

    PARAMETERS
    ----------
    path   Path to the source book file.

    RETURNS
    -------
    The cleaned document dict, with document_id, source metadata, a
    processing_summary, and the restructured chapters.

    RAISES
    ------
    ValueError   If no readable text was extracted, or no usable chapters
                 or paragraphs remained after cleaning.

    EXAMPLES
    --------
    >>> import tempfile
    >>> with tempfile.TemporaryDirectory() as folder:
    ...     path = Path(folder) / "book.txt"
    ...     _ = path.write_text(
    ...         "My Book\nChapter 1: Start\n"
    ...         "A short paragraph that is long enough to keep.\n",
    ...         encoding="utf-8",
    ...     )
    ...     doc = clean_book(path)
    ...     first_id = doc["document_id"]
    ...     second_id = clean_book(path)["document_id"]
    ...     [c["title"] for c in doc["chapters"]], first_id == second_id
    (['Chapter 1: Start'], True)
    """
    pages = extract_pages(path)

    if not pages:
        raise ValueError("No readable text was extracted.")

    content_hash = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    document_id = stable_id("DOC", content_hash, path.name)

    repeated_lines = identify_repeated_marginal_lines(pages)

    margin_cleaned_pages = [
        remove_headers_footers_and_page_numbers(page, repeated_lines)
        for page in pages
    ]

    retained_pages, removed_pages = remove_non_content_pages(
        margin_cleaned_pages
    )

    chapters = restructure_into_chapters(retained_pages, document_id)

    if not chapters:
        raise ValueError(
            "No usable chapters or paragraphs remained after cleaning."
        )

    return {
        "document_id": document_id,
        "source_file": path.name,
        "source_path": str(path),
        "source_format": path.suffix.casefold().lstrip("."),
        **source_attributes(path),
        "processing_summary": {
            "original_page_or_section_count": len(pages),
            "retained_page_or_section_count": len(retained_pages),
            "removed_page_or_section_count": len(removed_pages),
            "detected_repeated_header_footer_count": len(repeated_lines),
            "chapter_count": len(chapters),
            "paragraph_count": count_paragraphs(chapters),
            "removed_pages_or_sections": removed_pages,
        },
        "chapters": chapters,
    }


def safe_output_path(
    input_path: Path,
    input_folder: Path,
    output_folder: Path,
) -> Path:
    r"""
    PURPOSE
    -------
    Build a flat, collision-free output path for a cleaned book. Nested
    input folders are flattened into the file name with `__`, so a
    subfolder structure cannot collide with a differently-nested file
    that happens to share a stem.

    PARAMETERS
    ----------
    input_path      Path to the source book file.
    input_folder    Root folder the book was discovered under.
    output_folder   Root folder cleaned JSON files are written to.

    RETURNS
    -------
    The flattened destination path under output_folder.

    EXAMPLES
    --------
    >>> safe_output_path(Path("/in/fiction/book.txt"), Path("/in"), Path("/out")).name
    'fiction__book_cleaned.json'
    >>> safe_output_path(Path("/in/book.txt"), Path("/in"), Path("/out")).name
    'book_cleaned.json'
    """
    relative_path = input_path.relative_to(input_folder)
    parent_name = "__".join(relative_path.parent.parts)

    if parent_name:
        output_name = (
            f"{parent_name}__{input_path.stem}_cleaned.json"
        )
    else:
        output_name = f"{input_path.stem}_cleaned.json"

    return output_folder / output_name


def process_file(
    path_string: str,
    input_folder: Path,
    output_folder: Path,
) -> dict:
    r"""
    PURPOSE
    -------
    Clean one book and write it to disk, reporting success or failure.
    Never raises: a failure to clean or write comes back as a
    {"status": "failed", "error": ...} result instead, so one bad book in
    a batch cannot take down the rest of the run.

    PARAMETERS
    ----------
    path_string      Path to the source book file, as a string.
    input_folder     Root folder the book was discovered under.
    output_folder    Root folder cleaned JSON files are written to.

    RETURNS
    -------
    A result dict describing the outcome; see PURPOSE for the failure
    shape, and the example below for the success shape.

    EXAMPLES
    --------
    >>> import tempfile
    >>> with tempfile.TemporaryDirectory() as folder:
    ...     root = Path(folder)
    ...     inp, out = root / "raw", root / "clean"
    ...     inp.mkdir()
    ...     book = inp / "book.txt"
    ...     _ = book.write_text(
    ...         "My Book\nChapter 1: Start\n"
    ...         "A short paragraph that is long enough to keep.\n",
    ...         encoding="utf-8",
    ...     )
    ...     result = process_file(str(book), inp, out)
    ...     result["status"], Path(result["output"]).name, result["chapters"]
    ('success', 'book_cleaned.json', 1)
    """
    path = Path(path_string)
    output_path = safe_output_path(path, input_folder, output_folder)

    try:
        cleaned_document = clean_book(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        temporary_path = output_path.with_suffix(".json.tmp")

        with temporary_path.open("w", encoding="utf-8") as output_file:
            json.dump(
                cleaned_document,
                output_file,
                ensure_ascii=False,
                indent=2,
            )

        temporary_path.replace(output_path)

        return {
            "status": "success",
            "input": str(path),
            "output": str(output_path),
            "chapters": cleaned_document["processing_summary"][
                "chapter_count"
            ],
            "paragraphs": cleaned_document["processing_summary"][
                "paragraph_count"
            ],
        }

    except Exception as error:
        return {
            "status": "failed",
            "input": str(path),
            "output": str(output_path),
            "error": f"{type(error).__name__}: {error}",
        }


def discover_books(input_folder: Path) -> list[Path]:
    r"""
    PURPOSE
    -------
    List supported book files in a folder, skipping hidden and lock files.

    PARAMETERS
    ----------
    input_folder   Root folder to search for book files.

    RETURNS
    -------
    The sorted list of supported book file paths found.

    EXAMPLES
    --------
    >>> import tempfile
    >>> with tempfile.TemporaryDirectory() as folder:
    ...     root = Path(folder)
    ...     _ = (root / "notes.txt").write_text("hi", encoding="utf-8")
    ...     _ = (root / "cover.png").write_bytes(b"\x89PNG")
    ...     _ = (root / ".hidden.txt").write_text("hi", encoding="utf-8")
    ...     [p.name for p in discover_books(root)]
    ['notes.txt']
    """
    return sorted(
        path
        for path in input_folder.rglob("*")
        if path.is_file()
        and path.suffix.casefold() in SUPPORTED_EXTENSIONS
        and not path.name.startswith("~$")
        and not path.name.startswith(".")
    )


def main() -> None:
    """
    PURPOSE
    -------
    Discover books in INPUT_FOLDER, clean them in parallel worker
    processes, and write cleaned JSON files plus a manifest to
    OUTPUT_FOLDER.

    RAISES
    ------
    FileNotFoundError    If INPUT_FOLDER does not exist.
    NotADirectoryError   If INPUT_FOLDER is not a folder.
    """
    if not INPUT_FOLDER.exists():
        raise FileNotFoundError(
            f"Input folder does not exist: {INPUT_FOLDER}"
        )

    if not INPUT_FOLDER.is_dir():
        raise NotADirectoryError(
            f"Input path is not a folder: {INPUT_FOLDER}"
        )

    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    books = discover_books(INPUT_FOLDER)

    if not books:
        print(
            "No supported files were found.\n"
            f"Input folder: {INPUT_FOLDER}\n"
            f"Supported formats: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )
        return

    worker_count = min(MAX_WORKERS, len(books))
    results: list[dict] = []

    print(f"Found {len(books)} book(s).")
    print(f"Using {worker_count} worker process(es).")

    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        future_to_path = {
            executor.submit(
                process_file, str(path), INPUT_FOLDER, OUTPUT_FOLDER
            ): path
            for path in books
        }

        for completed_count, future in enumerate(
            as_completed(future_to_path),
            start=1,
        ):
            result = future.result()
            results.append(result)

            if result["status"] == "success":
                print(
                    f"[{completed_count}/{len(books)}] CLEANED: "
                    f"{Path(result['input']).name} | "
                    f"{result['chapters']} chapters | "
                    f"{result['paragraphs']} paragraphs"
                )
            else:
                print(
                    f"[{completed_count}/{len(books)}] FAILED: "
                    f"{Path(result['input']).name} | "
                    f"{result['error']}"
                )

    results.sort(key=lambda item: item["input"].casefold())

    manifest = {
        "input_folder": str(INPUT_FOLDER),
        "output_folder": str(OUTPUT_FOLDER),
        "total_files": len(results),
        "successful_files": sum(
            result["status"] == "success" for result in results
        ),
        "failed_files": sum(
            result["status"] == "failed" for result in results
        ),
        "results": results,
    }

    manifest_path = OUTPUT_FOLDER / "cleaning_manifest.json"
    temporary_manifest_path = manifest_path.with_suffix(".json.tmp")

    with temporary_manifest_path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)

    temporary_manifest_path.replace(manifest_path)

    print(
        "\nCleaning completed.\n"
        f"Successful: {manifest['successful_files']}\n"
        f"Failed: {manifest['failed_files']}\n"
        f"Output folder: {OUTPUT_FOLDER}\n"
        f"Manifest: {manifest_path}"
    )


if __name__ == "__main__":
    main()