#!/usr/bin/env python3

"""
SoakinGarri Tokenization Program

PURPOSE
-------
Reads the cleaned JSON books produced by the cleaning program and packs their
sentences into fixed-size token windows of either 500 or 1000 tokens.

Windows are sentence-aware: a sentence is never cut in half unless the sentence
alone is longer than the whole window. Consecutive windows share a small overlap
so that meaning is not lost at the seams.

500 is the default because it is a conservative window size that works well
for sentence-aware packing; the embedding program's model (BAAI/bge-m3)
supports sequences of up to 8192 tokens, so both 500 and 1000 sit well within
its limit. Choose 1000 when the downstream model benefits from more context
per window.

INPUT FOLDER
------------
data/ask/cleaned_data (resolved from DATA_ROOT)

OUTPUT FOLDER
-------------
data/ask/tokenized_data (resolved from DATA_ROOT)

OUTPUT FILES
------------
token_windows_<size>.jsonl
tokenization_manifest.json

INSTALL
-------
python3 -m pip install transformers

The transformers package is optional. When it is absent the program falls back
to a built-in approximate tokenizer, and says so in the manifest.

RUN
---
python3 tokenization.py                 (500-token windows)
python3 tokenization.py --tokens 1000
python3 tokenization.py --tokens 1000 --overlap 80
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
from typing import Any, Iterator, Protocol

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common import DATA_ROOT, normalize_text, stable_id, load_documents  # noqa: E402

INPUT_FOLDER = DATA_ROOT / "cleaned_data"

OUTPUT_FOLDER = DATA_ROOT / "tokenized_data"

MANIFEST_FILE = OUTPUT_FOLDER / "tokenization_manifest.json"

SOURCE_LABEL = "Borrowed content and open source"

TOKENIZER_MODEL_NAME = "BAAI/bge-m3"

WINDOW_SIZE_CHOICES = (500, 1000)
DEFAULT_WINDOW_SIZE = 500

OVERLAP_RATIO = 0.10
MINIMUM_WINDOW_TOKENS = 40

SUPPORTED_EXTENSIONS = {".json", ".jsonl"}

SENTENCE_PATTERN = re.compile(
    r"(?<=[.!?])\s+(?=[A-Z0-9“\"'(\[])"
)

APPROXIMATE_TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]")

APPROXIMATE_CHARACTERS_PER_TOKEN = 5


class Tokenizer(Protocol):
    """
    PURPOSE
    -------
    Minimal interface this program needs from a tokenizer. Anything exposing
    `name` and `count(text)` works, which is what lets the tests run without
    installing transformers.
    """

    name: str

    def count(self, text: str) -> int:
        """
        PURPOSE
        -------
        Count how many tokens `text` would occupy.

        PARAMETERS
        ----------
        text   The text to count tokens for.

        RETURNS
        -------
        The token count.
        """
        ...


class ApproximateTokenizer:
    r"""
    PURPOSE
    -------
    Deterministic fallback tokenizer used when transformers is unavailable.
    Words are split into pieces of about five characters, which tracks real
    subword tokenizers closely enough for windowing. It is an approximation,
    not a substitute: counts typically land within ten percent of a real
    tokenizer.

    EXAMPLES
    --------
    >>> tokenizer = ApproximateTokenizer()
    >>> tokenizer.name
    'approximate'
    >>> tokenizer.count("Momentum is mass times velocity.")
    8
    >>> tokenizer.count("")
    0
    >>> tokenizer.count("a" * 25)
    5
    """

    name = "approximate"

    def count(self, text: str) -> int:
        """
        PURPOSE
        -------
        Count tokens by splitting text into words/punctuation pieces and
        estimating roughly one token per five characters within each piece.

        PARAMETERS
        ----------
        text   The text to count tokens for.

        RETURNS
        -------
        The estimated token count.
        """
        total = 0

        for piece in APPROXIMATE_TOKEN_PATTERN.findall(text):
            if len(piece) <= APPROXIMATE_CHARACTERS_PER_TOKEN:
                total += 1
            else:
                total += -(-len(piece) // APPROXIMATE_CHARACTERS_PER_TOKEN)

        return total


class TransformersTokenizer:
    """
    PURPOSE
    -------
    Wraps a Hugging Face tokenizer so it matches the Tokenizer interface.
    """

    def __init__(self, tokenizer, name: str):
        """
        PURPOSE
        -------
        Store the wrapped Hugging Face tokenizer and its display name.
        """
        self._tokenizer = tokenizer
        self.name = name

    def count(self, text: str) -> int:
        """
        PURPOSE
        -------
        Count tokens using the wrapped Hugging Face tokenizer, without
        adding any special tokens.

        PARAMETERS
        ----------
        text   The text to count tokens for.

        RETURNS
        -------
        The exact token count.
        """
        return len(
            self._tokenizer.encode(text, add_special_tokens=False)
        )


def load_tokenizer(
    model_name: str = TOKENIZER_MODEL_NAME,
    allow_fallback: bool = True,
) -> Tokenizer:
    """
    PURPOSE
    -------
    Load the model's real tokenizer, or fall back to the approximate one
    when transformers is not installed.

    PARAMETERS
    ----------
    model_name       Name of the Hugging Face model whose tokenizer to load.
    allow_fallback   When False, a missing transformers install raises
                      instead of silently downgrading to the approximate
                      tokenizer.

    RETURNS
    -------
    A tokenizer implementing the Tokenizer interface.

    RAISES
    ------
    RuntimeError   transformers is not installed and allow_fallback is
                   False.
    """
    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        if not allow_fallback:
            raise RuntimeError(
                "Exact token counts require transformers: "
                "python3 -m pip install transformers"
            ) from error

        return ApproximateTokenizer()

    return TransformersTokenizer(
        AutoTokenizer.from_pretrained(model_name),
        name=model_name,
    )


@dataclass(slots=True)
class TokenWindow:
    """
    PURPOSE
    -------
    One packed window of consecutive sentences, ready to be written out as a
    training/RAG record, together with the metadata needed to trace it back
    to its source book, chapter, and paragraphs.
    """

    window_id: str
    window_index: int
    document_id: str | None
    source_file: str
    chapter_id: str | None
    chapter_title: str
    paragraph_ids: list[str]
    source_page_start: int | None
    source_page_end: int | None
    sentence_count: int
    token_count: int
    window_size: int
    overlap_tokens: int
    tokenizer: str
    source: str
    text: str


def overlap_token_count(
    window_size: int,
    overlap_tokens: int | None = None,
) -> int:
    r"""
    PURPOSE
    -------
    Resolve how many tokens carry over between consecutive windows. When
    `overlap_tokens` is not given, it defaults to `OVERLAP_RATIO` of the
    window size; the result is always clamped so it never exceeds half the
    window.

    PARAMETERS
    ----------
    window_size      Size of a token window.
    overlap_tokens   Requested overlap, or None to use the default ratio.

    RETURNS
    -------
    The resolved overlap, between 0 and half of window_size.

    EXAMPLES
    --------
    >>> overlap_token_count(500), overlap_token_count(1000)
    (50, 100)
    >>> overlap_token_count(500, 0), overlap_token_count(500, 120)
    (0, 120)
    >>> overlap_token_count(500, 900)      # never larger than half the window
    250
    """
    if overlap_tokens is None:
        overlap_tokens = int(window_size * OVERLAP_RATIO)

    return max(0, min(overlap_tokens, window_size // 2))


def split_sentences(text: str) -> list[str]:
    r"""
    PURPOSE
    -------
    Split text into sentences on terminal punctuation (`.`, `!`, `?`)
    followed by whitespace and a capital letter, digit, or opening quote or
    bracket.

    PARAMETERS
    ----------
    text   The text to split.

    RETURNS
    -------
    The list of sentences; a single-element list holding the whole
    normalized text when no split point is found, or an empty list when
    text is blank.

    EXAMPLES
    --------
    >>> split_sentences("First one. Second one! Third? yes.")
    ['First one.', 'Second one!', 'Third? yes.']
    >>> split_sentences("   ")
    []
    >>> split_sentences("no terminal punctuation")
    ['no terminal punctuation']
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


def split_oversized_sentence(
    sentence: str,
    window_size: int,
    tokenizer: Tokenizer,
) -> list[str]:
    r"""
    PURPOSE
    -------
    Break a single sentence that is longer than one window into word-safe
    pieces. Splitting happens at word boundaries, so no word is ever cut in
    half and the original wording is preserved exactly.

    PARAMETERS
    ----------
    sentence      The sentence to split, if needed.
    window_size   Maximum tokens allowed per piece.
    tokenizer     Tokenizer used to count tokens.

    RETURNS
    -------
    A list of one or more pieces, each at or under window_size tokens
    (unless a single word alone exceeds it).

    EXAMPLES
    --------
    >>> tokenizer = ApproximateTokenizer()
    >>> split_oversized_sentence("short enough", 50, tokenizer)
    ['short enough']
    >>> pieces = split_oversized_sentence(" ".join(["alpha"] * 20), 6, tokenizer)
    >>> [len(piece.split()) for piece in pieces]
    [6, 6, 6, 2]
    >>> all(tokenizer.count(piece) <= 6 for piece in pieces)
    True
    """
    if tokenizer.count(sentence) <= window_size:
        return [sentence]

    pieces: list[str] = []
    current_words: list[str] = []
    current_tokens = 0

    for word in sentence.split():
        word_tokens = tokenizer.count(word)

        if current_words and current_tokens + word_tokens > window_size:
            pieces.append(" ".join(current_words))
            current_words = []
            current_tokens = 0

        current_words.append(word)
        current_tokens += word_tokens

    if current_words:
        pieces.append(" ".join(current_words))

    return pieces


def pack_sentences(
    sentences: list[str],
    window_size: int,
    overlap_tokens: int,
    tokenizer: Tokenizer,
) -> list[list[str]]:
    r"""
    PURPOSE
    -------
    Group sentences into windows of at most window_size tokens. Each window
    after the first starts with trailing sentences from the previous one, up
    to overlap_tokens tokens' worth.

    PARAMETERS
    ----------
    sentences        Sentences to pack, in order.
    window_size      Maximum tokens allowed per window.
    overlap_tokens   Tokens' worth of trailing sentences to carry into the
                      next window.
    tokenizer        Tokenizer used to count tokens.

    RETURNS
    -------
    A list of windows, each a list of the sentences it contains.

    EXAMPLES
    --------
    >>> tokenizer = ApproximateTokenizer()
    >>> sentences = ["Alpha beta gamma.", "Delta epsilon zeta.", "Eta theta iota."]
    >>> [tokenizer.count(s) for s in sentences]
    [4, 5, 4]
    >>> pack_sentences(sentences, 12, 0, tokenizer)
    [['Alpha beta gamma.', 'Delta epsilon zeta.'], ['Eta theta iota.']]
    >>> pack_sentences(sentences, 12, 6, tokenizer)
    [['Alpha beta gamma.', 'Delta epsilon zeta.'], ['Delta epsilon zeta.', 'Eta theta iota.']]
    >>> pack_sentences(sentences, 100, 0, tokenizer)
    [['Alpha beta gamma.', 'Delta epsilon zeta.', 'Eta theta iota.']]
    >>> pack_sentences([], 500, 50, tokenizer)
    []
    """
    windows: list[list[str]] = []
    current: list[str] = []
    current_tokens = 0

    def carry_over(finished: list[str]) -> tuple[list[str], int]:
        """
        PURPOSE
        -------
        Pick the trailing sentences of a just-finished window to carry over
        into the next one, staying within overlap_tokens and never carrying
        the entire window over (which would stop packing from advancing).

        PARAMETERS
        ----------
        finished   The window that was just closed off.

        RETURNS
        -------
        A (carried_sentences, carried_tokens) pair.
        """
        if overlap_tokens <= 0:
            return [], 0

        carried: list[str] = []
        carried_tokens = 0

        for sentence in reversed(finished):
            sentence_tokens = tokenizer.count(sentence)

            if carried_tokens + sentence_tokens > overlap_tokens:
                break

            carried.insert(0, sentence)
            carried_tokens += sentence_tokens

        # never carry the whole window over, or packing would not advance
        if len(carried) >= len(finished):
            carried = carried[1:]
            carried_tokens = sum(tokenizer.count(s) for s in carried)

        return carried, carried_tokens

    for sentence in sentences:
        for piece in split_oversized_sentence(sentence, window_size, tokenizer):
            piece_tokens = tokenizer.count(piece)

            if current and current_tokens + piece_tokens > window_size:
                windows.append(current)
                current, current_tokens = carry_over(current)

            current.append(piece)
            current_tokens += piece_tokens

    if current:
        windows.append(current)

    return windows


def iterate_chapters(document: dict[str, Any]) -> list[dict[str, Any]]:
    r"""
    PURPOSE
    -------
    Return the document's chapters, wrapping a flat text field (text,
    content, or body) into a single synthetic chapter when the document has
    no chapters list of its own.

    PARAMETERS
    ----------
    document   The document dictionary to read chapters from.

    RETURNS
    -------
    A list of chapter dictionaries; empty when the document has neither
    chapters nor any usable flat text.

    EXAMPLES
    --------
    >>> iterate_chapters({"chapters": [{"title": "A"}, "junk"]})
    [{'title': 'A'}]
    >>> chapters = iterate_chapters({"title": "Essay", "text": "Body  text"})
    >>> chapters[0]["title"], chapters[0]["paragraphs"][0]["text"]
    ('Essay', 'Body text')
    >>> iterate_chapters({"nothing": True})
    []
    """
    chapters = document.get("chapters")

    if isinstance(chapters, list):
        return [
            chapter
            for chapter in chapters
            if isinstance(chapter, dict)
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
    Return (paragraph_id, text) pairs for a chapter, normalizing string,
    dict, and single-string paragraph shapes and dropping anything without
    usable text.

    PARAMETERS
    ----------
    chapter   The chapter dictionary to read paragraphs from.

    RETURNS
    -------
    A list of (paragraph_id, text) pairs, in order.

    EXAMPLES
    --------
    >>> chapter_paragraphs({"paragraphs": [{"paragraph_id": "P-1", "text": "Hi"},
    ...                                    "plain", {"text": " "}, 7]})
    [('P-1', 'Hi'), (None, 'plain')]
    >>> chapter_paragraphs({"paragraphs": "single string"})
    [(None, 'single string')]
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


def tokenize_document(
    document: dict[str, Any],
    source_file: Path,
    window_size: int,
    tokenizer: Tokenizer,
    overlap_tokens: int | None = None,
    window_offset: int = 0,
) -> list[TokenWindow]:
    r"""
    PURPOSE
    -------
    Turn one cleaned document into a list of token windows. Windows never
    span two chapters, so every window keeps a single chapter's page range
    and title; an undersized trailing window is folded into the previous
    one within the same chapter.

    PARAMETERS
    ----------
    document         The cleaned document to tokenize.
    source_file      Path to the file the document came from, used for ids
                      and the source_file field.
    window_size      Maximum tokens allowed per window.
    tokenizer        Tokenizer used to count tokens.
    overlap_tokens   Requested overlap, or None to use the default ratio.
    window_offset    Starting value for window_index, so windows from
                      multiple documents in one file can be numbered
                      continuously.

    RETURNS
    -------
    A list of TokenWindow records, in order.

    EXAMPLES
    --------
    >>> tokenizer = ApproximateTokenizer()
    >>> document = {
    ...     "document_id": "DOC-1",
    ...     "chapters": [{
    ...         "chapter_id": "CH-1",
    ...         "title": "Chapter 1: Motion",
    ...         "source_page_start": 3,
    ...         "source_page_end": 6,
    ...         "paragraphs": [
    ...             {"paragraph_id": "P-1", "text": "Alpha beta gamma. Delta epsilon zeta."},
    ...             {"paragraph_id": "P-2", "text": "Eta theta iota."},
    ...         ],
    ...     }],
    ... }
    >>> windows = tokenize_document(document, Path("book.json"), 12, tokenizer, overlap_tokens=0)
    >>> [(w.window_index, w.token_count, w.sentence_count) for w in windows]
    [(0, 9, 2), (1, 4, 1)]
    >>> windows[0].text
    'Alpha beta gamma. Delta epsilon zeta.'
    >>> windows[0].chapter_title, windows[0].source_page_start, windows[0].paragraph_ids
    ('Chapter 1: Motion', 3, ['P-1'])
    >>> windows[0].window_id.startswith("WIN-"), windows[0].tokenizer
    (True, 'approximate')
    >>> tokenize_document({}, Path("b.json"), 500, tokenizer)
    []
    """
    resolved_overlap = overlap_token_count(window_size, overlap_tokens)
    windows: list[TokenWindow] = []
    window_index = window_offset

    for chapter in iterate_chapters(document):
        paragraphs = chapter_paragraphs(chapter)

        if not paragraphs:
            continue

        sentences: list[str] = []
        sentence_owners: list[str | None] = []

        for paragraph_id, text in paragraphs:
            for sentence in split_sentences(text):
                sentences.append(sentence)
                sentence_owners.append(paragraph_id)

        owner_of = dict(zip(sentences, sentence_owners))
        chapter_title = normalize_text(chapter.get("title") or "Main Text")
        first_window_of_chapter = len(windows)

        for group in pack_sentences(
            sentences, window_size, resolved_overlap, tokenizer
        ):
            text = " ".join(group)
            token_count = tokenizer.count(text)

            if (
                token_count < MINIMUM_WINDOW_TOKENS
                and len(windows) > first_window_of_chapter
            ):
                # fold an undersized tail into the window before it, but only
                # within the same chapter: windows never span two chapters
                previous = windows[-1]

                if previous.token_count + token_count <= window_size:
                    previous.text = f"{previous.text} {text}"
                    previous.token_count = tokenizer.count(previous.text)
                    previous.sentence_count += len(group)
                    continue

            paragraph_ids: list[str] = []

            for sentence in group:
                owner = owner_of.get(sentence)

                if owner and owner not in paragraph_ids:
                    paragraph_ids.append(owner)

            windows.append(
                TokenWindow(
                    window_id=stable_id(
                        "WIN", source_file.name, str(window_index), text
                    ),
                    window_index=window_index,
                    document_id=document.get("document_id"),
                    source_file=source_file.name,
                    chapter_id=chapter.get("chapter_id"),
                    chapter_title=chapter_title,
                    paragraph_ids=paragraph_ids,
                    source_page_start=chapter.get("source_page_start"),
                    source_page_end=chapter.get("source_page_end"),
                    sentence_count=len(group),
                    token_count=token_count,
                    window_size=window_size,
                    overlap_tokens=resolved_overlap,
                    tokenizer=tokenizer.name,
                    source=SOURCE_LABEL,
                    text=text,
                )
            )

            window_index += 1

    return windows


def discover_input_files(input_folder: Path = INPUT_FOLDER) -> list[Path]:
    r"""
    PURPOSE
    -------
    List the .json/.jsonl files to tokenize under input_folder, skipping
    hidden files and files ending in "_manifest.json".

    PARAMETERS
    ----------
    input_folder   Root folder to search recursively.

    RETURNS
    -------
    A sorted list of matching file paths.

    EXAMPLES
    --------
    >>> import tempfile
    >>> with tempfile.TemporaryDirectory() as folder:
    ...     root = Path(folder)
    ...     for name in ("b.json", "a.JSONL", "cleaning_manifest.json",
    ...                  ".hidden.json", "notes.txt"):
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


def atomic_write_json(destination: Path, data: dict[str, Any]) -> None:
    r"""
    PURPOSE
    -------
    Write JSON through a temporary file in the same folder, then rename it
    over the target, so a crash mid-write never leaves a truncated file at
    destination. Creates parent folders as needed and cleans up the
    temporary file on failure.

    PARAMETERS
    ----------
    destination   Final path to write the JSON file to.
    data          The JSON-serializable data to write.

    EXAMPLES
    --------
    >>> import tempfile
    >>> with tempfile.TemporaryDirectory() as folder:
    ...     path = Path(folder) / "out" / "manifest.json"
    ...     atomic_write_json(path, {"windows": 3, "label": "café"})
    ...     (json.loads(path.read_text("utf-8")), [p.name for p in path.parent.iterdir()])
    ({'windows': 3, 'label': 'café'}, ['manifest.json'])
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
            json.dump(data, output_file, ensure_ascii=False, indent=2)
            output_file.write("\n")
            output_file.flush()
            os.fsync(output_file.fileno())

        temporary_path.replace(destination)

    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def write_jsonl(path: Path, records: Iterator[dict[str, Any]]) -> int:
    r"""
    PURPOSE
    -------
    Write records as JSONL (one JSON object per line) through a temporary
    file in the same folder, then atomically replace path with it.

    PARAMETERS
    ----------
    path      Final path to write the JSONL file to.
    records   Iterator of JSON-serializable dictionaries to write.

    RETURNS
    -------
    The number of records written.

    EXAMPLES
    --------
    >>> import tempfile
    >>> with tempfile.TemporaryDirectory() as folder:
    ...     path = Path(folder) / "out" / "windows.jsonl"
    ...     count = write_jsonl(path, iter([{"a": 1}, {"b": "é"}]))
    ...     (count, path.read_text("utf-8"), [p.name for p in path.parent.iterdir()])
    (2, '{"a": 1}\n{"b": "é"}\n', ['windows.jsonl'])
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0

    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as temporary_file:
        for record in records:
            temporary_file.write(json.dumps(record, ensure_ascii=False))
            temporary_file.write("\n")
            count += 1

        temporary_name = temporary_file.name

    os.replace(temporary_name, path)

    return count


def summarize_token_counts(windows: list[TokenWindow]) -> dict[str, Any]:
    r"""
    PURPOSE
    -------
    Report window count and token statistics across a list of token
    windows.

    PARAMETERS
    ----------
    windows   The token windows to summarize.

    RETURNS
    -------
    A dictionary with window_count, total_tokens, minimum_tokens,
    maximum_tokens, and average_tokens.

    EXAMPLES
    --------
    >>> summarize_token_counts([])
    {'window_count': 0, 'total_tokens': 0, 'minimum_tokens': 0, 'maximum_tokens': 0, 'average_tokens': 0.0}
    >>> tokenizer = ApproximateTokenizer()
    >>> document = {"chapters": [{"title": "T", "paragraphs": ["Alpha beta. Gamma delta."]}]}
    >>> stats = summarize_token_counts(tokenize_document(document, Path("b.json"), 500, tokenizer))
    >>> stats["window_count"], stats["total_tokens"] > 0
    (1, True)
    """
    if not windows:
        return {
            "window_count": 0,
            "total_tokens": 0,
            "minimum_tokens": 0,
            "maximum_tokens": 0,
            "average_tokens": 0.0,
        }

    counts = [window.token_count for window in windows]

    return {
        "window_count": len(counts),
        "total_tokens": sum(counts),
        "minimum_tokens": min(counts),
        "maximum_tokens": max(counts),
        "average_tokens": round(sum(counts) / len(counts), 2),
    }


def run_pipeline(
    input_folder: Path = INPUT_FOLDER,
    output_folder: Path = OUTPUT_FOLDER,
    window_size: int = DEFAULT_WINDOW_SIZE,
    overlap_tokens: int | None = None,
    tokenizer: Tokenizer | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    r"""
    PURPOSE
    -------
    Tokenize every input file into token windows and write both the windows
    (JSONL) and a manifest describing the run. A file that fails to load or
    tokenize is recorded as failed in the manifest rather than aborting the
    whole run.

    PARAMETERS
    ----------
    input_folder     Folder of cleaned JSON/JSONL files to tokenize.
    output_folder    Folder to write the token windows and manifest to.
    window_size      Maximum tokens allowed per window.
    overlap_tokens   Requested overlap, or None to use the default ratio.
    tokenizer        Tokenizer to use, or None to load the default one.
    verbose          Whether to print progress and a summary.

    RETURNS
    -------
    The manifest dictionary that was also written to disk.

    RAISES
    ------
    FileNotFoundError   input_folder does not exist.

    EXAMPLES
    --------
    >>> import tempfile
    >>> document = {"document_id": "DOC-1", "chapters": [{
    ...     "chapter_id": "CH-1", "title": "Heat",
    ...     "paragraphs": [{"paragraph_id": "P-1",
    ...                     "text": "Water boils at one hundred degrees. " * 30}]}]}
    >>> with tempfile.TemporaryDirectory() as folder:
    ...     inp, out = Path(folder) / "cleaned", Path(folder) / "tokenized"
    ...     inp.mkdir()
    ...     _ = (inp / "physics_cleaned.json").write_text(json.dumps(document), encoding="utf-8")
    ...     manifest = run_pipeline(inp, out, window_size=500,
    ...                             tokenizer=ApproximateTokenizer(), verbose=False)
    ...     written = sorted(p.name for p in out.iterdir())
    >>> manifest["window_size"], manifest["overlap_tokens"], manifest["tokenizer"]
    (500, 50, 'approximate')
    >>> manifest["statistics"]["maximum_tokens"] <= 500
    True
    >>> written
    ['token_windows_500.jsonl', 'tokenization_manifest.json']
    >>> run_pipeline(Path("/definitely/not/here"), Path("/tmp/x"))
    Traceback (most recent call last):
        ...
    FileNotFoundError: Input folder does not exist: /definitely/not/here
    """
    if not input_folder.is_dir():
        raise FileNotFoundError(
            f"Input folder does not exist: {input_folder}"
        )

    if tokenizer is None:
        tokenizer = load_tokenizer()

    resolved_overlap = overlap_token_count(window_size, overlap_tokens)

    def announce(message: str) -> None:
        """
        PURPOSE
        -------
        Print message when verbose output is enabled.
        """
        if verbose:
            print(message)

    announce(f"Tokenizer: {tokenizer.name}")
    announce(f"Window size: {window_size} tokens")
    announce(f"Overlap: {resolved_overlap} tokens")

    files = discover_input_files(input_folder)
    all_windows: list[TokenWindow] = []
    file_results: list[dict[str, Any]] = []

    for path in files:
        try:
            documents = load_documents(path)
            file_windows: list[TokenWindow] = []

            for document in documents:
                file_windows.extend(
                    tokenize_document(
                        document=document,
                        source_file=path,
                        window_size=window_size,
                        tokenizer=tokenizer,
                        overlap_tokens=resolved_overlap,
                        window_offset=len(all_windows) + len(file_windows),
                    )
                )

            all_windows.extend(file_windows)

            file_results.append(
                {
                    "status": "success",
                    "input_file": str(path),
                    "documents": len(documents),
                    "windows": len(file_windows),
                }
            )

            announce(f"{path.name}: {len(file_windows)} window(s)")

        except Exception as error:
            file_results.append(
                {
                    "status": "failed",
                    "input_file": str(path),
                    "error": f"{type(error).__name__}: {error}",
                }
            )

            announce(f"{path.name}: FAILED ({type(error).__name__})")

    output_file = output_folder / f"token_windows_{window_size}.jsonl"

    written = write_jsonl(
        output_file,
        (asdict(window) for window in all_windows),
    )

    manifest = {
        "program": "SoakinGarri Tokenization Program",
        "input_folder": str(input_folder),
        "output_folder": str(output_folder),
        "output_file": str(output_file),
        "source_label": SOURCE_LABEL,
        "tokenizer": tokenizer.name,
        "exact_token_counts": tokenizer.name != ApproximateTokenizer.name,
        "window_size": window_size,
        "overlap_tokens": resolved_overlap,
        "total_files": len(files),
        "successful_files": sum(
            result["status"] == "success" for result in file_results
        ),
        "failed_files": sum(
            result["status"] == "failed" for result in file_results
        ),
        "records_written": written,
        "statistics": summarize_token_counts(all_windows),
        "files": file_results,
    }

    atomic_write_json(output_folder / MANIFEST_FILE.name, manifest)

    if verbose:
        statistics = manifest["statistics"]
        print(
            f"\nTokenization completed.\n"
            f"Windows: {statistics['window_count']}\n"
            f"Tokens: {statistics['total_tokens']}\n"
            f"Average window: {statistics['average_tokens']} tokens\n"
            f"Output: {output_file}"
        )

    return manifest


def parse_arguments(argument_list: list[str] | None = None) -> argparse.Namespace:
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
    >>> parse_arguments([]).tokens
    500
    >>> options = parse_arguments(["--tokens", "1000", "--overlap", "80"])
    >>> options.tokens, options.overlap
    (1000, 80)
    >>> parse_arguments(["--tokens", "750"]).tokens      # other sizes are allowed
    750
    """
    parser = argparse.ArgumentParser(
        description=(
            "Pack cleaned book text into fixed-size token windows "
            f"(default {DEFAULT_WINDOW_SIZE}; "
            f"{' or '.join(str(size) for size in WINDOW_SIZE_CHOICES)} are the "
            "sizes this pipeline is tuned for)."
        )
    )

    parser.add_argument(
        "--tokens",
        type=int,
        default=DEFAULT_WINDOW_SIZE,
        help="tokens per window",
    )

    parser.add_argument(
        "--overlap",
        type=int,
        default=None,
        help="tokens shared between consecutive windows "
             f"(default {int(OVERLAP_RATIO * 100)}%% of the window)",
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
        help="folder for token windows",
    )

    parser.add_argument(
        "--exact-tokens",
        action="store_true",
        help="fail instead of falling back to the approximate tokenizer",
    )

    return parser.parse_args(argument_list)


def main() -> None:
    """
    PURPOSE
    -------
    Tokenize the configured folders using the options given on the command
    line.
    """
    options = parse_arguments()

    if options.tokens < MINIMUM_WINDOW_TOKENS:
        raise SystemExit(
            f"--tokens must be at least {MINIMUM_WINDOW_TOKENS}."
        )

    run_pipeline(
        input_folder=options.input,
        output_folder=options.output,
        window_size=options.tokens,
        overlap_tokens=options.overlap,
        tokenizer=load_tokenizer(allow_fallback=not options.exact_tokens),
    )


if __name__ == "__main__":
    main()
