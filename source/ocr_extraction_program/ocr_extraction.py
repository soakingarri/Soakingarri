#!/usr/bin/env python3

"""
SoakinGarri OCR Extraction Program

A richer, standalone OCR module than data_cleaning_program's basic
fallback (see data_cleaning.py's ocr_page_image/extract_pdf). Given a
scanned/photographed PDF, this program keeps page numbers, flags pages
whose OCR confidence is low enough that a human should review them
instead of silently shipping garbled text, and classifies each
recognized block of text as body prose, a code block, a table, or a
figure/table caption rather than treating a whole page as one
undifferentiated blob.

Input:
    <repo>/data/ask/raw_data

Output:
    <repo>/data/ask/ocr_extraction

Supported input formats:
    PDF (scanned/photographed; no text layer required)

Output format:
    One UTF-8 JSON file per PDF: {"source_file", "source_path",
    "page_count", "pages"}, where each page is {"page_number", "text",
    "mean_confidence", "is_low_confidence", "blocks"}. Plus a manifest
    summarizing the run.

Install dependencies:
    python3 -m pip install pymupdf pytesseract pillow
    (and the Tesseract OCR engine itself: `brew install tesseract` on
    macOS, or `conda install -c conda-forge tesseract`)

Run:
    python3 ocr_extraction.py
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common import DATA_ROOT  # noqa: E402

INPUT_FOLDER = DATA_ROOT / "raw_data"
OUTPUT_FOLDER = DATA_ROOT / "ocr_extraction"

DEFAULT_DPI = 300

# Mean/word confidence below this (0-100 scale) is treated as low enough
# to flag for human review rather than ship silently.
LOW_CONFIDENCE_PAGE_THRESHOLD = 60.0
LOW_CONFIDENCE_WORD_THRESHOLD = 60.0

NUMERIC_TOKEN_PATTERN = re.compile(r"^[-+]?[$€£]?\d[\d,.]*%?$")
SHORT_TOKEN_MAX_LENGTH = 4

TABLE_MIN_QUALIFYING_TOKENS = 3
TABLE_MIN_RUN_LINES = 2

CODE_SYMBOL_PATTERN = re.compile(r"[{}();=<>]")
CODE_SYMBOL_MIN_MATCHES = 2
CODE_INDENT_MIN_SPACES = 4

FIGURE_CAPTION_LINE_PATTERN = re.compile(
    r"(?i)^\s*(?:figure|fig\.|table)\s+\d+"
)
FIGURE_CAPTION_CONTINUATION_MAX_LENGTH = 160


@dataclass(slots=True)
class OcrWord:
    """
    PURPOSE
    -------
    One word recognized by Tesseract, as given by one entry of
    pytesseract.image_to_data's per-word columns.

    ATTRIBUTES
    ----------
    text                        The recognized word text.
    confidence                  Tesseract's confidence for this word,
                                 0-100 (-1 means no text was recognized
                                 at this position).
    left, top, width, height    The word's bounding box in pixels, in
                                 the rendered page image's coordinates.
    """

    text: str
    confidence: float
    left: int
    top: int
    width: int
    height: int


@dataclass(slots=True)
class OcrPageResult:
    """
    PURPOSE
    -------
    The OCR result for one rendered page image. mean_confidence is what
    lets a caller flag a whole page for human review instead of
    silently shipping garbled OCR output into the training/RAG corpus.

    ATTRIBUTES
    ----------
    text                   The page's recognized text, with line breaks
                            preserved.
    mean_confidence        Mean confidence across words that had any
                            text recognized (entries with confidence -1
                            are excluded, since -1 is not a real amount
                            of confidence to average in).
    words                   Every recognized word, with its confidence
                            and position.
    low_confidence_words   The subset of recognized words' text whose
                            confidence fell below the low-confidence
                            threshold.
    """

    text: str
    mean_confidence: float
    words: list[OcrWord] = field(default_factory=list)
    low_confidence_words: list[str] = field(default_factory=list)


def recognize_text(image: Any, config: str = "") -> str:
    r"""
    PURPOSE
    -------
    Run Tesseract OCR over an already-rendered image and return its
    recognized text. This is the one place pytesseract.image_to_string
    is called from in this module (and, via data_cleaning.py's
    ocr_page_image, in data_cleaning.py too), so the install-instruction
    messages and error handling live in one place instead of being
    copy-pasted at every call site.

    PARAMETERS
    ----------
    image    A Pillow Image (or anything pytesseract.image_to_string
             accepts) to recognize text on.
    config   Extra Tesseract configuration flags, passed through
             unchanged (e.g. "--psm 6" to assume a single uniform block
             of text).

    RETURNS
    -------
    The OCR-recognized text (possibly empty, if the image is blank).

    RAISES
    ------
    RuntimeError   If pytesseract/Pillow are not installed, or the
                   Tesseract OCR engine itself is not installed.
    """
    try:
        import pytesseract
    except ImportError as error:
        raise RuntimeError(
            "OCR support requires pytesseract and Pillow: "
            "python3 -m pip install pytesseract pillow"
        ) from error

    try:
        return pytesseract.image_to_string(image, config=config)
    except pytesseract.TesseractNotFoundError as error:
        raise RuntimeError(
            "OCR support also requires the Tesseract engine itself, not "
            "just the pytesseract wrapper: install it with "
            "`conda install -c conda-forge tesseract` or "
            "`brew install tesseract` on macOS."
        ) from error


def recognize_with_confidence(
    image: Any,
    low_confidence_word_threshold: float = LOW_CONFIDENCE_WORD_THRESHOLD,
) -> OcrPageResult:
    r"""
    PURPOSE
    -------
    Run Tesseract OCR over an image and report not just the recognized
    text but how confident Tesseract was in it, word by word. A single
    page in a batch of thousands of scanned documents can come out
    almost unreadable (poor scan quality, a skewed photograph, faint
    print) while still producing *some* text; mean_confidence is what
    lets a caller flag that whole page for human review.

    PARAMETERS
    ----------
    image                            A Pillow Image to recognize text
                                      on.
    low_confidence_word_threshold    Words recognized below this
                                      confidence are collected into
                                      low_confidence_words.

    RETURNS
    -------
    An OcrPageResult with the recognized text, the mean confidence
    across words that had any text recognized, the full per-word
    detail, and the subset of words below the threshold.

    RAISES
    ------
    RuntimeError   If pytesseract/Pillow are not installed, or the
                   Tesseract OCR engine itself is not installed.
    """
    try:
        import pytesseract
    except ImportError as error:
        raise RuntimeError(
            "OCR support requires pytesseract and Pillow: "
            "python3 -m pip install pytesseract pillow"
        ) from error

    try:
        text = pytesseract.image_to_string(image)
        data = pytesseract.image_to_data(
            image, output_type=pytesseract.Output.DICT
        )
    except pytesseract.TesseractNotFoundError as error:
        raise RuntimeError(
            "OCR support also requires the Tesseract engine itself, not "
            "just the pytesseract wrapper: install it with "
            "`conda install -c conda-forge tesseract` or "
            "`brew install tesseract` on macOS."
        ) from error

    words: list[OcrWord] = []

    for index in range(len(data["text"])):
        word_text = data["text"][index].strip()

        if not word_text:
            continue

        words.append(
            OcrWord(
                text=word_text,
                confidence=float(data["conf"][index]),
                left=data["left"][index],
                top=data["top"][index],
                width=data["width"][index],
                height=data["height"][index],
            )
        )

    confident_words = [word for word in words if word.confidence >= 0]
    mean_confidence = (
        sum(word.confidence for word in confident_words)
        / len(confident_words)
        if confident_words
        else 0.0
    )

    low_confidence_words = [
        word.text
        for word in confident_words
        if word.confidence < low_confidence_word_threshold
    ]

    return OcrPageResult(
        text=text,
        mean_confidence=mean_confidence,
        words=words,
        low_confidence_words=low_confidence_words,
    )


def is_low_confidence_page(
    mean_confidence: float,
    threshold: float = LOW_CONFIDENCE_PAGE_THRESHOLD,
) -> bool:
    r"""
    PURPOSE
    -------
    Decide whether a page's OCR confidence is low enough to flag for
    human review, rather than shipping it into the corpus untouched.
    Centralizing the threshold here means the review policy lives in
    one obvious place instead of being re-guessed at each call site.

    PARAMETERS
    ----------
    mean_confidence   A page's mean OCR confidence (0-100), e.g. from
                       OcrPageResult.mean_confidence.
    threshold         Mean confidence below which a page is considered
                       low-confidence.

    RETURNS
    -------
    True if the page should be flagged for human review.

    EXAMPLES
    --------
    >>> is_low_confidence_page(45.0)
    True
    >>> is_low_confidence_page(85.0)
    False
    >>> is_low_confidence_page(60.0, threshold=60.0)
    False
    """
    return mean_confidence < threshold


def looks_like_figure_caption(line: str) -> bool:
    r"""
    PURPOSE
    -------
    Guess whether a line starts a figure, image, or table caption.

    PARAMETERS
    ----------
    line   The line of text to classify.

    RETURNS
    -------
    True if the line looks like the start of a figure/table caption.

    EXAMPLES
    --------
    >>> looks_like_figure_caption("Figure 1: Rainfall by region.")
    True
    >>> looks_like_figure_caption("Fig. 2 shows the trend.")
    True
    >>> looks_like_figure_caption("Table 3: Summary of results.")
    True
    >>> looks_like_figure_caption("This figure illustrates the point.")
    False
    """
    return bool(FIGURE_CAPTION_LINE_PATTERN.match(line.strip()))


def token_looks_numeric_or_short(token: str) -> bool:
    r"""
    PURPOSE
    -------
    Guess whether one whitespace-separated token is the kind of short
    label or number that fills a table cell, as opposed to a word of
    ordinary prose.

    PARAMETERS
    ----------
    token   The token to classify.

    RETURNS
    -------
    True if the token looks numeric-or-short.

    EXAMPLES
    --------
    >>> token_looks_numeric_or_short("5.99")
    True
    >>> token_looks_numeric_or_short("cat")
    True
    >>> token_looks_numeric_or_short("Introduction")
    False
    """
    return bool(
        NUMERIC_TOKEN_PATTERN.match(token)
        or len(token) <= SHORT_TOKEN_MAX_LENGTH
    )


def count_numeric_or_short_tokens(line: str) -> int:
    r"""
    PURPOSE
    -------
    Count how many of a line's whitespace-separated tokens look
    numeric-or-short; the building block for looks_like_table_row's
    crude multi-column heuristic.

    PARAMETERS
    ----------
    line   The line of text to count tokens in.

    RETURNS
    -------
    The number of qualifying tokens.

    EXAMPLES
    --------
    >>> count_numeric_or_short_tokens("1      10    5.99")
    3
    >>> count_numeric_or_short_tokens("Anthropology examines cultural practices")
    0
    """
    return sum(token_looks_numeric_or_short(token) for token in line.split())


def looks_like_table_row(line: str) -> bool:
    r"""
    PURPOSE
    -------
    Guess whether one line is a row of a table: 3+ whitespace-separated
    tokens that each look numeric-or-short. On its own this is a weak
    signal (a short English sentence can pass it too); classify_text_blocks
    only trusts it once 2+ consecutive lines pass it with the *same*
    qualifying-token count, which ordinary prose rarely does since
    sentences wrap at different points line to line.

    PARAMETERS
    ----------
    line   The line of text to classify.

    RETURNS
    -------
    True if the line has 3 or more numeric-or-short tokens.

    EXAMPLES
    --------
    >>> looks_like_table_row("1      10    5.99")
    True
    >>> looks_like_table_row("Anthropology examines cultural practices")
    False
    """
    return (
        len(line.split()) >= TABLE_MIN_QUALIFYING_TOKENS
        and count_numeric_or_short_tokens(line) >= TABLE_MIN_QUALIFYING_TOKENS
    )


def looks_like_code_line(line: str) -> bool:
    r"""
    PURPOSE
    -------
    Guess whether one line looks like source code rather than prose:
    either a high density of code-typical symbols ({ } ( ) ; = < >), or
    indentation of 4+ spaces (the way a nested code block is printed).

    PARAMETERS
    ----------
    line   The line of text to classify (not pre-stripped; leading
           whitespace matters for the indentation check).

    RETURNS
    -------
    True if the line looks like code.

    EXAMPLES
    --------
    >>> looks_like_code_line("def add(a, b):")
    True
    >>> looks_like_code_line("    return a + b")
    True
    >>> looks_like_code_line("This is a normal sentence.")
    False
    """
    stripped = line.strip()

    if not stripped:
        return False

    symbol_matches = len(CODE_SYMBOL_PATTERN.findall(stripped))
    indent = len(line) - len(line.lstrip(" "))

    return (
        symbol_matches >= CODE_SYMBOL_MIN_MATCHES
        or indent >= CODE_INDENT_MIN_SPACES
    )


def classify_text_blocks(text: str) -> list[dict]:
    r"""
    PURPOSE
    -------
    Split OCR'd page text into blocks classified as body prose, a code
    block, a table, or a figure/table caption, rather than treating a
    whole page as one undifferentiated blob of text. Heuristics are
    explicit regex/structure checks (consistent with this repo's
    looks_like_heading/toc_score style in data_cleaning.py) rather than
    a model, since OCR'd text is noisy enough without adding a second
    source of unpredictability on top of it.

    PARAMETERS
    ----------
    text   The page's OCR'd text (plain text, not an image).

    RETURNS
    -------
    A list of {"type", "text"} dicts, "type" being one of "body",
    "code", "table", or "figure_caption", in reading order.

    EXAMPLES
    --------
    >>> text = ("Anthropology examines cultural practices extensively throughout history.\n"
    ...         "Scholars continue investigating these community traditions carefully.")
    >>> classify_text_blocks(text)
    [{'type': 'body', 'text': 'Anthropology examines cultural practices extensively throughout history.\nScholars continue investigating these community traditions carefully.'}]

    >>> text = "Figure 1: Rainfall by region.\nSource: National Weather Service."
    >>> classify_text_blocks(text)
    [{'type': 'figure_caption', 'text': 'Figure 1: Rainfall by region. Source: National Weather Service.'}]

    >>> text = ("Item inventory follows.\n"
    ...         "1      10    5.99\n"
    ...         "2      20    3.50\n"
    ...         "3      15    7.25")
    >>> classify_text_blocks(text)
    [{'type': 'body', 'text': 'Item inventory follows.'}, {'type': 'table', 'text': '1      10    5.99\n2      20    3.50\n3      15    7.25'}]

    >>> text = ("def add(a, b):\n"
    ...         "    return a + b\n"
    ...         "result = add(1, 2)")
    >>> classify_text_blocks(text)
    [{'type': 'code', 'text': 'def add(a, b):\n    return a + b\nresult = add(1, 2)'}]
    """
    lines = text.splitlines()
    blocks: list[dict] = []
    body_buffer: list[str] = []

    def flush_body() -> None:
        if body_buffer:
            joined = "\n".join(body_buffer).strip()

            if joined:
                blocks.append({"type": "body", "text": joined})

            body_buffer.clear()

    index = 0
    total_lines = len(lines)

    while index < total_lines:
        line = lines[index]
        stripped = line.strip()

        if not stripped:
            index += 1
            continue

        if looks_like_figure_caption(stripped):
            flush_body()
            caption_lines = [stripped]
            index += 1

            while (
                index < total_lines
                and lines[index].strip()
                and len(lines[index].strip())
                <= FIGURE_CAPTION_CONTINUATION_MAX_LENGTH
                and not looks_like_figure_caption(lines[index])
                and not looks_like_table_row(lines[index])
                and not looks_like_code_line(lines[index])
            ):
                caption_lines.append(lines[index].strip())
                index += 1

            blocks.append(
                {"type": "figure_caption", "text": " ".join(caption_lines)}
            )
            continue

        # Code is checked before table rows: short tokens made of bare
        # operators/brackets ("{", "(x", "0)") are "short" by length and
        # would otherwise be miscounted as numeric-or-short table cells,
        # which code-typical symbol density is a more reliable signal
        # for than the table heuristic is for ruling it out.
        if looks_like_code_line(line):
            run_lines = [line]
            index += 1

            while index < total_lines and looks_like_code_line(lines[index]):
                run_lines.append(lines[index])
                index += 1

            flush_body()
            blocks.append(
                {
                    "type": "code",
                    "text": "\n".join(
                        code_line.rstrip() for code_line in run_lines
                    ),
                }
            )
            continue

        if looks_like_table_row(line):
            run_start = index
            qualifying_count = count_numeric_or_short_tokens(line)
            run_lines = [line]
            index += 1

            while (
                index < total_lines
                and looks_like_table_row(lines[index])
                and count_numeric_or_short_tokens(lines[index])
                == qualifying_count
            ):
                run_lines.append(lines[index])
                index += 1

            if len(run_lines) >= TABLE_MIN_RUN_LINES:
                flush_body()
                blocks.append(
                    {
                        "type": "table",
                        "text": "\n".join(
                            table_line.strip() for table_line in run_lines
                        ),
                    }
                )
                continue

            # A lone line with qualifying tokens isn't a table by itself;
            # treat it as an ordinary body line and re-examine the line
            # right after it.
            body_buffer.append(run_lines[0].strip())
            index = run_start + 1
            continue

        body_buffer.append(stripped)
        index += 1

    flush_body()

    return blocks


def ocr_pdf(path: Path, dpi: int = DEFAULT_DPI) -> list[dict]:
    r"""
    PURPOSE
    -------
    Run full OCR extraction over every page of a scanned/photographed
    PDF: render each page to an image, recognize its text with per-word
    confidence, flag low-confidence pages for review, and classify the
    recognized text into body/code/table/figure_caption blocks.

    PARAMETERS
    ----------
    path   Path to the PDF file.
    dpi    Resolution to render each page at before recognition; higher
           values improve accuracy on small print at the cost of speed.

    RETURNS
    -------
    One dict per page, in page order: {"page_number", "text",
    "mean_confidence", "is_low_confidence", "blocks"}.

    RAISES
    ------
    RuntimeError   If PyMuPDF is not installed, or pytesseract/Pillow
                   or the Tesseract engine itself are not installed.
    """
    try:
        import fitz
    except ImportError as error:
        raise RuntimeError(
            "PDF support requires PyMuPDF: "
            "python3 -m pip install pymupdf"
        ) from error

    try:
        from PIL import Image
    except ImportError as error:
        raise RuntimeError(
            "OCR support requires pytesseract and Pillow: "
            "python3 -m pip install pytesseract pillow"
        ) from error

    pages: list[dict] = []

    with fitz.open(path) as document:
        for page_number, page in enumerate(document, start=1):
            pixmap = page.get_pixmap(dpi=dpi)
            image = Image.frombytes(
                "RGB", (pixmap.width, pixmap.height), pixmap.samples
            )

            result = recognize_with_confidence(image)
            blocks = classify_text_blocks(result.text)

            pages.append(
                {
                    "page_number": page_number,
                    "text": result.text,
                    "mean_confidence": result.mean_confidence,
                    "is_low_confidence": is_low_confidence_page(
                        result.mean_confidence
                    ),
                    "blocks": blocks,
                }
            )

    return pages


def discover_pdfs(input_folder: Path) -> list[Path]:
    r"""
    PURPOSE
    -------
    List PDF files in a folder, skipping hidden and lock files.

    PARAMETERS
    ----------
    input_folder   Root folder to search for PDF files.

    RETURNS
    -------
    The sorted list of PDF file paths found.

    EXAMPLES
    --------
    >>> import tempfile
    >>> with tempfile.TemporaryDirectory() as folder:
    ...     root = Path(folder)
    ...     _ = (root / "scan.pdf").write_bytes(b"%PDF-1.4")
    ...     _ = (root / "notes.txt").write_text("hi", encoding="utf-8")
    ...     _ = (root / ".hidden.pdf").write_bytes(b"%PDF-1.4")
    ...     [p.name for p in discover_pdfs(root)]
    ['scan.pdf']
    """
    return sorted(
        path
        for path in input_folder.rglob("*")
        if path.is_file()
        and path.suffix.casefold() == ".pdf"
        and not path.name.startswith("~$")
        and not path.name.startswith(".")
    )


def safe_output_path(
    input_path: Path,
    input_folder: Path,
    output_folder: Path,
) -> Path:
    r"""
    PURPOSE
    -------
    Build a flat, collision-free output path for one PDF's OCR result.
    Nested input folders are flattened into the file name with "__", so
    a subfolder structure cannot collide with a differently-nested file
    that happens to share a stem.

    PARAMETERS
    ----------
    input_path      Path to the source PDF file.
    input_folder    Root folder the PDF was discovered under.
    output_folder   Root folder OCR JSON files are written to.

    RETURNS
    -------
    The flattened destination path under output_folder.

    EXAMPLES
    --------
    >>> safe_output_path(Path("/in/scans/book.pdf"), Path("/in"), Path("/out")).name
    'scans__book_ocr.json'
    >>> safe_output_path(Path("/in/book.pdf"), Path("/in"), Path("/out")).name
    'book_ocr.json'
    """
    relative_path = input_path.relative_to(input_folder)
    parent_name = "__".join(relative_path.parent.parts)

    if parent_name:
        output_name = f"{parent_name}__{input_path.stem}_ocr.json"
    else:
        output_name = f"{input_path.stem}_ocr.json"

    return output_folder / output_name


def process_pdf_file(
    path_string: str,
    input_folder: Path,
    output_folder: Path,
    dpi: int = DEFAULT_DPI,
) -> dict:
    r"""
    PURPOSE
    -------
    OCR one PDF and write its page results to disk, reporting success
    or failure. Never raises: a failure comes back as a
    {"status": "failed", "error": ...} result instead, so one bad scan
    in a batch cannot take down the rest of the run.

    PARAMETERS
    ----------
    path_string      Path to the source PDF file, as a string.
    input_folder     Root folder the PDF was discovered under.
    output_folder    Root folder OCR JSON files are written to.
    dpi              Resolution to render each page at before
                      recognition.

    RETURNS
    -------
    On success: {"status": "success", "input", "output", "page_count",
    "low_confidence_page_count"}. On failure: {"status": "failed",
    "input", "output", "error"}.
    """
    path = Path(path_string)
    output_path = safe_output_path(path, input_folder, output_folder)

    try:
        pages = ocr_pdf(path, dpi=dpi)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        document = {
            "source_file": path.name,
            "source_path": str(path),
            "page_count": len(pages),
            "pages": pages,
        }

        temporary_path = output_path.with_suffix(".json.tmp")

        with temporary_path.open("w", encoding="utf-8") as output_file:
            json.dump(document, output_file, ensure_ascii=False, indent=2)

        temporary_path.replace(output_path)

        low_confidence_page_count = sum(
            page["is_low_confidence"] for page in pages
        )

        return {
            "status": "success",
            "input": str(path),
            "output": str(output_path),
            "page_count": len(pages),
            "low_confidence_page_count": low_confidence_page_count,
        }

    except Exception as error:
        return {
            "status": "failed",
            "input": str(path),
            "output": str(output_path),
            "error": f"{type(error).__name__}: {error}",
        }


def run_pipeline(
    input_folder: Path = INPUT_FOLDER,
    output_folder: Path = OUTPUT_FOLDER,
    dpi: int = DEFAULT_DPI,
) -> dict:
    r"""
    PURPOSE
    -------
    OCR every PDF found in input_folder and write one JSON file per PDF
    plus a manifest summarizing the run (including how many pages came
    back low-confidence, for a human reviewer to triage) to
    output_folder.

    PARAMETERS
    ----------
    input_folder    Root folder containing scanned/photographed PDFs.
    output_folder   Root folder OCR JSON files and the manifest are
                     written to.
    dpi             Resolution to render each page at before
                     recognition.

    RETURNS
    -------
    The manifest dict describing the run.

    RAISES
    ------
    FileNotFoundError    If input_folder does not exist.
    NotADirectoryError   If input_folder exists but is not a folder.

    EXAMPLES
    --------
    >>> import tempfile
    >>> with tempfile.TemporaryDirectory() as folder:
    ...     inp, out = Path(folder) / "in", Path(folder) / "out"
    ...     inp.mkdir()
    ...     manifest = run_pipeline(inp, out)
    ...     manifest["total_files"], manifest["successful_files"]
    (0, 0)
    """
    if not input_folder.exists():
        raise FileNotFoundError(
            f"Input folder does not exist: {input_folder}"
        )

    if not input_folder.is_dir():
        raise NotADirectoryError(
            f"Input path is not a folder: {input_folder}"
        )

    output_folder.mkdir(parents=True, exist_ok=True)

    pdf_paths = discover_pdfs(input_folder)
    results = [
        process_pdf_file(str(path), input_folder, output_folder, dpi=dpi)
        for path in pdf_paths
    ]

    manifest = {
        "input_folder": str(input_folder),
        "output_folder": str(output_folder),
        "total_files": len(results),
        "successful_files": sum(
            result["status"] == "success" for result in results
        ),
        "failed_files": sum(
            result["status"] == "failed" for result in results
        ),
        "total_pages": sum(
            result.get("page_count", 0)
            for result in results
            if result["status"] == "success"
        ),
        "low_confidence_pages": sum(
            result.get("low_confidence_page_count", 0)
            for result in results
            if result["status"] == "success"
        ),
        "results": results,
    }

    manifest_path = output_folder / "ocr_extraction_manifest.json"
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
    Parse the command line into the options run_pipeline needs.

    PARAMETERS
    ----------
    argument_list   Arguments to parse, or None to read from sys.argv.

    RETURNS
    -------
    The parsed arguments namespace.

    EXAMPLES
    --------
    >>> parse_arguments([]).dpi
    300
    >>> options = parse_arguments(["--dpi", "400"])
    >>> options.dpi
    400
    """
    parser = argparse.ArgumentParser(
        description=(
            "OCR scanned/photographed PDFs into page-level text with "
            "per-page confidence flags and body/code/table/"
            "figure_caption block classification."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=INPUT_FOLDER,
        help="folder of PDFs to OCR",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_FOLDER,
        help="folder for OCR JSON files and the manifest",
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=DEFAULT_DPI,
        help="resolution to render each page at before recognition",
    )

    return parser.parse_args(argument_list)


def main() -> None:
    """
    PURPOSE
    -------
    OCR the configured folders using the options given on the command
    line, and print a summary.
    """
    options = parse_arguments()

    manifest = run_pipeline(
        input_folder=options.input,
        output_folder=options.output,
        dpi=options.dpi,
    )

    print(
        "OCR extraction completed.\n"
        f"Files: {manifest['successful_files']} ok, "
        f"{manifest['failed_files']} failed\n"
        f"Pages: {manifest['total_pages']} total, "
        f"{manifest['low_confidence_pages']} low-confidence\n"
        f"Output folder: {manifest['output_folder']}"
    )


if __name__ == "__main__":
    main()
