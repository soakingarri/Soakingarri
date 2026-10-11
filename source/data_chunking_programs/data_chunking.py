#!/usr/bin/env python3

"""
SoakinGarri Data Chunking Program

PURPOSE
-------
Reads the cleaned JSON books created by the SoakinGarri cleaning program and
produces two text datasets:

1. LLM training dataset (heuristic, template-generated -- see
   build_llm_records; not reviewed or model-generated, and every record is
   marked generation_method: "template" so a training pipeline can filter
   these out or flag them for review):
   Question -> reasoning -> answer
   Concept -> explanation
   Formula -> derivation -> example

2. RAG dataset:
   Document ID
   Topic
   Concept
   Formula
   Source metadata (including a per-chunk page range narrowed to the
   pages its own paragraphs came from)

Embedding is deliberately NOT done here: embedding_program/embedding.py is
the one place vectors get produced, so a chunk is embedded exactly once.

INPUT
-----
<repo>/data/ask/cleaned_data

OUTPUT
------
<repo>/data/ask/chunked_data

INSTALL
-------
python3 -m pip install numpy

RUN
---
python3 data_chunking.py

NOTES
-----
- The output uses JSONL, where every line is one valid JSON object.
- Existing output files are replaced safely.
- The program does not claim that labeling content changes its copyright status.
  Confirm that material is licensed or otherwise authorized before training or
  redistribution.
"""

from __future__ import annotations

import json
import os
import re
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common import (  # noqa: E402
    DATA_ROOT,
    load_documents,
    normalize_text,
    stable_id as stable_record_id,
)

INPUT_FOLDER = DATA_ROOT / "cleaned_data"
OUTPUT_FOLDER = DATA_ROOT / "chunked_data"

# Pipeline stages that may sit between cleaning and chunking, latest first:
# cleaning -> deduplication.py -> language_detection.py -> chunking.
UPSTREAM_FOLDERS = (
    DATA_ROOT / "language_tagged_data",
    DATA_ROOT / "deduplicated_data",
    INPUT_FOLDER,
)

LLM_OUTPUT_FILE = OUTPUT_FOLDER / "llm_training_data.jsonl"
RAG_OUTPUT_FILE = OUTPUT_FOLDER / "rag_data.jsonl"
MANIFEST_FILE = OUTPUT_FOLDER / "chunking_manifest.json"

SOURCE_LABEL = "Borrowed content and open source"

TARGET_CHUNK_WORDS = 350
MIN_CHUNK_WORDS = 60
CHUNK_OVERLAP_WORDS = 50

SUPPORTED_EXTENSIONS = {".json", ".jsonl"}

SENTENCE_PATTERN = re.compile(
    r"(?<=[.!?])\s+(?=[A-Z0-9“\"'(\[])"
)

ABBREVIATION_END_PATTERN = re.compile(
    r"(?:^|[\s(\[])"
    r"(?:pp?|vol|vols|no|nos|ch|chap|fig|figs|eq|eqs|ed|eds|cf|al|"
    r"e\.g|i\.e|viz|approx|ca|dr|mr|mrs|ms|prof|st|jr|sr|vs|etc)\.$",
    re.IGNORECASE,
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
        r"(?:[A-Za-z0-9_()]+(?:\s*[+\-*/^]\s*"
        r"[A-Za-z0-9_().]+)+)"
        r"(?!\w)"
    ),
    re.compile(
        r"(?<!\w)"
        r"(?:[A-Za-z][A-Za-z0-9_]*)"
        r"\s*(?:≈|≠|≤|≥|∝)\s*"
        r"[^.\n;]{1,160}"
    ),
)

DERIVATION_MARKERS = re.compile(
    r"\b("
    r"derive|derived|derivation|therefore|thus|hence|because|"
    r"substitut|rearrang|simplif|solv|proof|given that|"
    r"starting with|from the equation"
    r")\b",
    re.IGNORECASE,
)

EXAMPLE_MARKERS = re.compile(
    r"\b("
    r"example|for instance|for example|suppose|consider|"
    r"e\.g\.|such as"
    r")\b",
    re.IGNORECASE,
)

STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being",
    "but", "by", "can", "could", "did", "do", "does", "for",
    "from", "had", "has", "have", "he", "her", "here", "hers",
    "him", "his", "how", "i", "if", "in", "into", "is", "it",
    "its", "may", "more", "most", "not", "of", "on", "or", "our",
    "out", "over", "she", "should", "so", "some", "such", "than",
    "that", "the", "their", "them", "then", "there", "these",
    "they", "this", "those", "through", "to", "under", "up", "us",
    "was", "we", "were", "what", "when", "where", "which", "who",
    "why", "will", "with", "would", "you", "your",
}


@dataclass(slots=True)
class SourceChunk:
    """
    PURPOSE
    -------
    One chunk of source text pulled from a cleaned book, annotated with
    the topic, concept, formula, derivation and example metadata used to
    build the LLM training and RAG records.

    ATTRIBUTES
    ----------
    document_id          Deterministic id for the source document.
    chunk_id             Deterministic id for this chunk.
    source_file          Name of the file the chunk was read from.
    source_path          Original source path recorded in the document.
    source_document_id   Document id declared in the source data, if any.
    chapter_id           Chapter id the chunk belongs to, if any.
    chapter_title        Title of the chapter the chunk belongs to.
    paragraph_ids        Paragraph ids contributing to the chunk.
    source_page_start    First source page the chunk covers, if known.
    source_page_end      Last source page the chunk covers, if known.
    text                 The chunk's normalized text.
    word_count           Word count of `text`.
    topic                Inferred topic for the chunk.
    concept              Inferred concept for the chunk.
    formula              First formula found in the chunk, if any.
    derivation           First derivation-marker sentence found, if any.
    example              First example-marker sentence found, if any.
    """

    document_id: str
    chunk_id: str
    source_file: str
    source_path: str
    source_document_id: str | None
    chapter_id: str | None
    chapter_title: str
    paragraph_ids: list[str]
    source_page_start: int | None
    source_page_end: int | None
    text: str
    word_count: int
    topic: str
    concept: str
    formula: str | None
    derivation: str | None
    example: str | None
    language: str | None = None
    source_type: str | None = None
    tool: str | None = None
    date: str | None = None


def word_count(text: str) -> int:
    r"""
    PURPOSE
    -------
    Number of whitespace-separated words.

    PARAMETERS
    ----------
    text   Text to count words in.

    RETURNS
    -------
    The word count.

    EXAMPLES
    --------
    >>> word_count("one two  three")
    3
    >>> word_count("")
    0
    """
    return len(text.split())


def split_sentences(text: str) -> list[str]:
    r"""
    PURPOSE
    -------
    Split on sentence-ending punctuation followed by a capital/digit/quote.

    PARAMETERS
    ----------
    text   Text to split into sentences.

    RETURNS
    -------
    List of sentences found in the text.

    EXAMPLES
    --------
    >>> split_sentences("First one. Second one! Is this third? yes.")
    ['First one.', 'Second one!', 'Is this third? yes.']
    >>> split_sentences("   ")
    []
    >>> split_sentences("no terminal punctuation here")
    ['no terminal punctuation here']
    >>> split_sentences("See pp. 59-60 and Fig. 3 for details. Then e.g. Dr. Ade agrees.")
    ['See pp. 59-60 and Fig. 3 for details.', 'Then e.g. Dr. Ade agrees.']
    """
    normalized = normalize_text(text)

    if not normalized:
        return []

    pieces = [
        piece.strip()
        for piece in SENTENCE_PATTERN.split(normalized)
        if piece.strip()
    ]

    # Re-join pieces that were split after an abbreviation ("pp. 59",
    # "Fig. 3", "Dr. Ade") rather than at a real sentence end.
    sentences: list[str] = []

    for piece in pieces:
        if sentences and ABBREVIATION_END_PATTERN.search(sentences[-1]):
            sentences[-1] = f"{sentences[-1]} {piece}"
        else:
            sentences.append(piece)

    return sentences or [normalized]


def split_long_text(text: str, maximum_words: int) -> list[str]:
    r"""
    PURPOSE
    -------
    Cut text into consecutive pieces of at most ``maximum_words`` words.

    PARAMETERS
    ----------
    text            Text to split.
    maximum_words   Maximum number of words allowed per piece.

    RETURNS
    -------
    List of word-bounded pieces.

    EXAMPLES
    --------
    >>> split_long_text("a b c d e", 2)
    ['a b', 'c d', 'e']
    >>> split_long_text("a b c", 10)
    ['a b c']
    >>> split_long_text("", 3)
    []
    """
    words = normalize_text(text).split()

    if not words:
        return []

    if len(words) <= maximum_words:
        return [" ".join(words)]

    pieces: list[str] = []

    for start in range(0, len(words), maximum_words):
        piece = " ".join(words[start:start + maximum_words]).strip()

        if piece:
            pieces.append(piece)

    return pieces


def clean_formula(formula: str) -> str:
    r"""
    PURPOSE
    -------
    Tidy a matched formula string.

    PARAMETERS
    ----------
    formula   Raw formula text to tidy.

    RETURNS
    -------
    The cleaned, length-capped formula string.

    EXAMPLES
    --------
    >>> clean_formula("  E = m c^2 ,  ")
    'E = m c^2'
    >>> len(clean_formula("x = " + "1+" * 200))
    240
    """
    formula = normalize_text(formula)
    formula = formula.strip(" ,.;:")
    return formula[:240]


FORMULA_TAIL_PATTERN = re.compile(
    r"\s+(?:for|where|when|with|and|or|in|of|to|on|as|by|from|that|which|"
    r"so|then|if|is|are|was|gives|giving|becomes|yields)\s+.*$",
    re.IGNORECASE,
)

OPERATOR_SPLIT_PATTERN = re.compile(r"\s*[+\-*/^]\s*")


def trim_formula_tail(formula: str) -> str:
    r"""
    PURPOSE
    -------
    Cut prose that trails an equation ("F = ma for a uniform body").

    PARAMETERS
    ----------
    formula   The candidate formula, possibly with trailing prose.

    RETURNS
    -------
    The formula with trailing prose removed.

    EXAMPLES
    --------
    >>> trim_formula_tail("a = F/m and x-1 or n - 1")
    'a = F/m'
    >>> trim_formula_tail("v = u + at")
    'v = u + at'
    """
    trimmed = FORMULA_TAIL_PATTERN.sub("", normalize_text(formula))
    return trimmed.strip(" ,.;:") or formula


def looks_mathematical(candidate: str) -> bool:
    r"""
    PURPOSE
    -------
    Reject operator-pattern matches that are really ordinary words.
    In prose, "-" and "/" mostly join words ("Evans-Pritchard",
    "210-day", "world/process") or number ranges ("59-60"), not
    variables. So a fragment whose only operators are "-" or "/" is kept
    only when every operand is a short symbol or number (at most two
    characters, like "F/m", "x-1", "n - 1") and at least one operand is
    a letter. Anything with "=", "+", "*" or "^" is kept.

    PARAMETERS
    ----------
    candidate   A cleaned operator-pattern match.

    RETURNS
    -------
    True if the fragment plausibly is mathematics.

    EXAMPLES
    --------
    >>> [looks_mathematical(s) for s in ["F/m", "x-1", "n - 1", "a/b + c", "2^n"]]
    [True, True, True, True, True]
    >>> [looks_mathematical(s) for s in
    ...  ["Evans-Pritchard", "210-day", "59-60)", "world/process - again", "Pitt- Bloch"]]
    [False, False, False, False, False]
    """
    if re.search(r"[=+*^]", candidate):
        return True

    operands = [
        operand.strip("() ")
        for operand in OPERATOR_SPLIT_PATTERN.split(candidate)
        if operand.strip("() ")
    ]

    if len(operands) < 2:
        return False

    return all(
        len(operand) <= 2 or operand.replace(".", "").isdigit()
        for operand in operands
    ) and any(
        operand.isalpha() for operand in operands
    )


def extract_formulas(text: str) -> list[str]:
    r"""
    PURPOSE
    -------
    Find equation-like fragments, deduplicated case-insensitively.
    Hyphenated words, slashed word pairs and number ranges are rejected
    (see :func:`looks_mathematical`), trailing prose is trimmed, and a
    sub-expression of a formula already found is dropped.

    PARAMETERS
    ----------
    text   Text to search for formulas.

    RETURNS
    -------
    List of distinct formula strings found, in order of first appearance.

    EXAMPLES
    --------
    >>> extract_formulas("Newton stated F = ma. Again, f = MA. So a = F/m.")
    ['F = ma', 'a = F/m']
    >>> extract_formulas("The ratio is a/b + c.")
    ['a/b + c']
    >>> extract_formulas("Plain prose with no maths at all.")
    []
    >>> extract_formulas("As Evans-Pritchard noted, this hi-jacking of a tidied-up, "
    ...                  "non-western 210-day calendar (pp. 59-60) is a world/process.")
    []
    """
    formulas: list[str] = []
    seen: set[str] = set()

    for pattern_index, pattern in enumerate(FORMULA_PATTERNS):
        for match in pattern.finditer(text):
            formula = clean_formula(trim_formula_tail(match.group(0)))
            normalized = formula.casefold()

            if len(formula) < 3:
                continue

            # pattern 1 is the bare-operator pattern, the one that confuses
            # hyphenated words and ranges with subtraction and division
            if pattern_index == 1 and not looks_mathematical(formula):
                continue

            if normalized in seen:
                continue

            if any(normalized in existing.casefold() for existing in formulas):
                continue

            seen.add(normalized)
            formulas.append(formula)

    return formulas


def select_sentence(
    sentences: Iterable[str],
    marker_pattern: re.Pattern[str],
) -> str | None:
    r"""
    PURPOSE
    -------
    Return the first sentence matching ``marker_pattern``, or ``None``.

    PARAMETERS
    ----------
    sentences        Sentences to search, in order.
    marker_pattern   Compiled regex used to test each sentence.

    RETURNS
    -------
    The first matching sentence (normalized), or ``None`` if none match.

    EXAMPLES
    --------
    >>> sentences = ["We define x.", "Therefore y follows.", "For example, z."]
    >>> select_sentence(sentences, DERIVATION_MARKERS)
    'Therefore y follows.'
    >>> select_sentence(sentences, EXAMPLE_MARKERS)
    'For example, z.'
    >>> select_sentence(["Nothing here."], EXAMPLE_MARKERS) is None
    True
    """
    for sentence in sentences:
        if marker_pattern.search(sentence):
            return normalize_text(sentence)

    return None


def infer_topic(chapter_title: str, text: str) -> str:
    r"""
    PURPOSE
    -------
    Use the chapter title as topic, or fall back to the most frequent
    keywords.

    PARAMETERS
    ----------
    chapter_title   Chapter title to prefer as the topic, if usable.
    text            Chapter text to fall back on for keyword extraction.

    RETURNS
    -------
    The inferred topic string.

    EXAMPLES
    --------
    >>> infer_topic("Chapter 2: Thermodynamics", "irrelevant body")
    'Chapter 2: Thermodynamics'
    >>> infer_topic("Main Text", "Photosynthesis converts light. Photosynthesis needs chlorophyll and light.")
    'Photosynthesis Light Converts Needs'
    >>> infer_topic("", "the and of")
    'General Knowledge'
    """
    title = normalize_text(chapter_title)

    if title and title.casefold() not in {
        "main text",
        "untitled",
        "unknown",
    }:
        return title[:200]

    words = re.findall(r"[A-Za-z][A-Za-z0-9'’-]{2,}", text.casefold())

    frequencies = Counter(
        word
        for word in words
        if word not in STOP_WORDS and not word.isdigit()
    )

    keywords = [
        word
        for word, _ in frequencies.most_common(4)
    ]

    if keywords:
        return " ".join(word.title() for word in keywords)

    return "General Knowledge"


def infer_concept(text: str, topic: str) -> str:
    r"""
    PURPOSE
    -------
    Pull a defined term from a leading "X is ..." sentence, else use the
    topic.

    PARAMETERS
    ----------
    text    Text to search for a defining sentence.
    topic   Fallback concept when no definition is found.

    RETURNS
    -------
    The inferred concept string.

    EXAMPLES
    --------
    >>> infer_concept("Osmosis is the movement of water across a membrane.", "Biology")
    'Osmosis'
    >>> infer_concept("The cat sat on the mat.", "Biology")
    'Biology'
    >>> infer_concept("", "Biology")
    'Biology'
    """
    sentences = split_sentences(text)

    if not sentences:
        return topic

    first_sentence = sentences[0]

    definition_match = re.match(
        r"^\s*"
        r"([A-Z][A-Za-z0-9'’\-\s]{1,100}?)"
        r"\s+(?:is|are|means|refers to|can be defined as)\s+",
        first_sentence,
    )

    if definition_match:
        concept = normalize_text(definition_match.group(1))

        if 1 <= len(concept.split()) <= 12:
            return concept[:160]

    return topic[:160]


def infer_answer(text: str) -> str:
    r"""
    PURPOSE
    -------
    Take leading sentences until roughly 70 words (never beyond 140).

    PARAMETERS
    ----------
    text   Text to draw the answer from.

    RETURNS
    -------
    The assembled answer text.

    EXAMPLES
    --------
    >>> infer_answer("Short answer. Another sentence.")
    'Short answer. Another sentence.'
    >>> long = " ".join(["Sentence number %d has exactly seven words." % i for i in range(30)])
    >>> 70 <= word_count(infer_answer(long)) <= 140
    True
    >>> infer_answer("")
    ''
    """
    sentences = split_sentences(text)

    if not sentences:
        return normalize_text(text)

    selected: list[str] = []
    current_words = 0

    for sentence in sentences:
        sentence_words = word_count(sentence)

        if selected and current_words + sentence_words > 140:
            break

        selected.append(sentence)
        current_words += sentence_words

        if current_words >= 70:
            break

    return normalize_text(" ".join(selected))


def build_question(topic: str, concept: str, formula: str | None) -> str:
    r"""
    PURPOSE
    -------
    Compose a training question from topic/concept/formula.

    PARAMETERS
    ----------
    topic     Topic the question is about.
    concept   Concept the question is about.
    formula   Formula to reference, if any.

    RETURNS
    -------
    The composed question string.

    EXAMPLES
    --------
    >>> build_question("Mechanics", "Momentum", "p = mv")
    'What does the formula p = mv represent in the context of Mechanics?'
    >>> build_question("Mechanics", "Momentum", None)
    'What is Momentum, and how is it related to Mechanics?'
    >>> build_question("Mechanics", "mechanics", None)
    'What is the main idea explained about Mechanics?'
    """
    if formula:
        return (
            f"What does the formula {formula} represent in the context "
            f"of {topic}?"
        )

    if concept and concept.casefold() != topic.casefold():
        return f"What is {concept}, and how is it related to {topic}?"

    return f"What is the main idea explained about {topic}?"


def build_reasoning(chunk: SourceChunk) -> str:
    r"""
    PURPOSE
    -------
    Compose the reasoning hint that pairs with :func:`build_question`.
    Grounds the hint in the chunk's own extracted derivation/example
    sentence when one was found, rather than a bare template sentence
    with no connection to the specific passage — the same three-way
    topic/concept/formula branch still picks the shape of the hint, but
    the hint itself now quotes the actual text where available.

    PARAMETERS
    ----------
    chunk   Source chunk the reasoning is about; uses its topic, concept,
            formula, and (if present) derivation/example sentences.

    RETURNS
    -------
    The composed reasoning string.

    EXAMPLES
    --------
    >>> body = ("Momentum is mass times velocity, p = mv. "
    ...         "Therefore doubling mass doubles momentum.")
    >>> chunk = finalize_chunk({}, Path("b.json"), {"title": "Mechanics"}, [], [body * 5])
    >>> build_reasoning(chunk).startswith("Identify the stated formula p = mv")
    True
    >>> 'Therefore doubling mass doubles momentum' in build_reasoning(chunk)
    True
    >>> no_formula = finalize_chunk({}, Path("b.json"), {"title": "Mechanics"},
    ...     [], ["Momentum is the quantity of motion an object has. " * 20])
    >>> build_reasoning(no_formula)
    'Locate the passage defining or describing Momentum, then relate that explanation to the broader topic of Mechanics.'
    >>> same_topic = finalize_chunk({}, Path("b.json"), {"title": "Mechanics"},
    ...     [], ["The general principles discussed here apply broadly. " * 20])
    >>> build_reasoning(same_topic).startswith("Identify the central claim")
    True
    """
    topic, concept, formula = chunk.topic, chunk.concept, chunk.formula

    if formula:
        hint = (
            f"Identify the stated formula {formula}, connect its "
            f"variables or terms to the surrounding explanation, and "
            f"interpret it within the topic of {topic}."
        )

        if chunk.derivation:
            hint += f' The passage reasons: "{chunk.derivation}"'

        return hint

    if concept and concept.casefold() != topic.casefold():
        return (
            f"Locate the passage defining or describing {concept}, then "
            f"relate that explanation to the broader topic of {topic}."
        )

    return (
        f"Identify the central claim in the passage and summarize the "
        f"supporting explanation about {topic}."
    )


def build_answer(chunk: SourceChunk) -> str:
    r"""
    PURPOSE
    -------
    Compose the answer that pairs with :func:`build_question`, grounded
    in the chunk's own extracted formula/derivation/example/concept
    content rather than just the chunk's leading sentences regardless of
    what the question actually asked — an answer about a formula should
    state and derive that formula, not whatever text happens to open the
    passage. Falls back to :func:`infer_answer`'s leading-sentence
    summary when no such structured signal is available.

    PARAMETERS
    ----------
    chunk   Source chunk to build the answer from.

    RETURNS
    -------
    The composed answer string.

    EXAMPLES
    --------
    >>> body = ("Momentum is mass times velocity, p = mv. "
    ...         "Therefore doubling mass doubles momentum. "
    ...         "For example, a heavier truck has more momentum at the same speed.")
    >>> chunk = finalize_chunk({}, Path("b.json"), {"title": "Mechanics"}, [], [body * 5])
    >>> answer = build_answer(chunk)
    >>> "p = mv" in answer and "doubles momentum" in answer and "heavier truck" in answer
    True
    >>> no_formula = finalize_chunk({}, Path("b.json"), {"title": "Mechanics"},
    ...     [], ["The general principles discussed here apply broadly. " * 20])
    >>> build_answer(no_formula) == infer_answer(no_formula.text)
    True
    """
    if chunk.formula:
        pieces = [f"The formula is {chunk.formula}."]

        if chunk.derivation:
            pieces.append(chunk.derivation)

        if chunk.example:
            pieces.append(chunk.example)

        if len(pieces) > 1:
            return normalize_text(" ".join(pieces))

    if chunk.concept and chunk.concept.casefold() != chunk.topic.casefold():
        concept_sentence = next(
            (
                sentence
                for sentence in split_sentences(chunk.text)
                if chunk.concept.casefold() in sentence.casefold()
            ),
            None,
        )

        if concept_sentence:
            return normalize_text(concept_sentence)

    return infer_answer(chunk.text)


def build_explanation(text: str) -> str:
    r"""
    PURPOSE
    -------
    Take leading sentences until roughly 120 words (never beyond 220).

    PARAMETERS
    ----------
    text   Text to draw the explanation from.

    RETURNS
    -------
    The assembled explanation text.

    EXAMPLES
    --------
    >>> build_explanation("One. Two.")
    'One. Two.'
    >>> long = " ".join(["Sentence number %d has exactly seven words." % i for i in range(60)])
    >>> 120 <= word_count(build_explanation(long)) <= 220
    True
    """
    sentences = split_sentences(text)

    if not sentences:
        return normalize_text(text)

    selected: list[str] = []
    current_words = 0

    for sentence in sentences:
        sentence_words = word_count(sentence)

        if selected and current_words + sentence_words > 220:
            break

        selected.append(sentence)
        current_words += sentence_words

        if current_words >= 120:
            break

    return normalize_text(" ".join(selected))


def extract_chapters(
    document: dict[str, Any],
) -> list[dict[str, Any]]:
    r"""
    PURPOSE
    -------
    Return the document's chapter dicts, or wrap a flat text field as one
    chapter.

    PARAMETERS
    ----------
    document   Parsed document dict.

    RETURNS
    -------
    List of chapter dicts.

    EXAMPLES
    --------
    >>> extract_chapters({"chapters": [{"title": "A"}, "junk"]})
    [{'title': 'A'}]
    >>> chapters = extract_chapters({"title": "Essay", "text": "Body  text"})
    >>> chapters[0]["title"], chapters[0]["paragraphs"][0]["text"]
    ('Essay', 'Body text')
    >>> extract_chapters({"nothing": True})
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
            "paragraphs": [
                {
                    "paragraph_id": None,
                    "text": text,
                }
            ],
        }
    ]


def extract_paragraphs(
    chapter: dict[str, Any],
) -> list[tuple[str | None, str, int | None, int | None]]:
    r"""
    PURPOSE
    -------
    Return ``(paragraph_id, text, page, page_end)`` tuples; accepts
    strings, dicts or a single string. ``page``/``page_end`` carry
    data_cleaning.py's per-paragraph page numbers through to chunking, so
    a chunk's citation can narrow to the pages its own text came from. A
    paragraph that crosses a page break has page_end = page + 1; older
    cleaned files without ``page_end`` fall back to ``page``.

    Long paragraphs are NOT cut here any more; :func:`build_chunks`
    splits them at sentence boundaries instead of every N words.

    PARAMETERS
    ----------
    chapter   Chapter dict to pull paragraphs from.

    RETURNS
    -------
    List of ``(paragraph_id, text, page, page_end)`` tuples; the pages are
    ``None`` when the source paragraph carries no page number (e.g. a
    DOCX/EPUB section, which has no real page concept).

    EXAMPLES
    --------
    >>> extract_paragraphs({"paragraphs": [
    ...     {"paragraph_id": "P-1", "text": "Hi", "page": 4, "page_end": 5},
    ...     {"paragraph_id": "P-2", "text": "Old", "page": 6},
    ...     "plain", {"text": ""}, 7]})
    [('P-1', 'Hi', 4, 5), ('P-2', 'Old', 6, 6), (None, 'plain', None, None)]
    >>> extract_paragraphs({"paragraphs": "just one"})
    [(None, 'just one', None, None)]
    >>> extract_paragraphs({"paragraphs": 123})
    []
    """
    raw_paragraphs = chapter.get("paragraphs", [])
    paragraphs: list[tuple[str | None, str, int | None, int | None]] = []

    if isinstance(raw_paragraphs, str):
        raw_paragraphs = [raw_paragraphs]

    if not isinstance(raw_paragraphs, list):
        return paragraphs

    for item in raw_paragraphs:
        if isinstance(item, str):
            paragraph_id = None
            page = page_end = None
            text = normalize_text(item)

        elif isinstance(item, dict):
            paragraph_id = item.get("paragraph_id")
            page = item.get("page")
            page = page if isinstance(page, int) else None
            page_end = item.get("page_end")
            page_end = page_end if isinstance(page_end, int) else page
            text = normalize_text(
                item.get("text")
                or item.get("content")
            )

        else:
            continue

        if text:
            paragraphs.append((paragraph_id, text, page, page_end))

    return paragraphs


@dataclass(slots=True)
class SentenceUnit:
    """
    PURPOSE
    -------
    One sentence (or, for a sentence longer than a whole chunk, one
    word-bounded piece of it) with the paragraph and pages it came from.
    Chunks are packed from these, so a chunk only ever starts and ends
    on a sentence boundary.
    """

    text: str
    words: int
    paragraph_id: str | None
    page: int | None
    page_end: int | None


def sentence_units(
    paragraphs: list[tuple[str | None, str, int | None, int | None]],
    target_words: int = TARGET_CHUNK_WORDS,
) -> list[SentenceUnit]:
    r"""
    PURPOSE
    -------
    Break paragraphs into sentence units. Only a single sentence longer
    than ``target_words`` is cut by word count, the one case where there
    is no sentence boundary to use.

    PARAMETERS
    ----------
    paragraphs     Output of :func:`extract_paragraphs`.
    target_words   Maximum words per unit.

    RETURNS
    -------
    List of :class:`SentenceUnit`, in reading order.

    EXAMPLES
    --------
    >>> units = sentence_units([("P-1", "One two three. Four five.", 3, 4)])
    >>> [(u.text, u.words, u.paragraph_id, u.page) for u in units]
    [('One two three.', 3, 'P-1', 3), ('Four five.', 2, 'P-1', 3)]
    >>> [u.text for u in sentence_units([(None, "a b c d e", None, None)], target_words=2)]
    ['a b', 'c d', 'e']
    """
    units: list[SentenceUnit] = []

    for paragraph_id, text, page, page_end in paragraphs:
        for sentence in split_sentences(text):
            for piece in split_long_text(sentence, target_words):
                units.append(
                    SentenceUnit(
                        text=piece,
                        words=word_count(piece),
                        paragraph_id=paragraph_id,
                        page=page,
                        page_end=page_end,
                    )
                )

    return units


def chunk_language(
    document: dict[str, Any],
    chapter: dict[str, Any],
    paragraph_ids: list[str],
) -> str | None:
    r"""
    PURPOSE
    -------
    The language of a chunk, from the "language" tags language_detection.py
    puts on each paragraph: the most common label among the chunk's own
    paragraphs (ties go to the first seen), else the chapter's, else the
    document's. Untagged input gives None, so chunking still works on
    cleaned_data that never went through language detection.

    PARAMETERS
    ----------
    document        Parsed document dict the chunk belongs to.
    chapter         Chapter dict the chunk belongs to.
    paragraph_ids   Paragraph ids contributing to the chunk.

    RETURNS
    -------
    A language label such as "yor_Latn", or None.

    EXAMPLES
    --------
    >>> chapter = {"language": "eng_Latn", "paragraphs": [
    ...     {"paragraph_id": "P-1", "language": "yor_Latn"},
    ...     {"paragraph_id": "P-2", "language": "eng_Latn"},
    ...     {"paragraph_id": "P-3", "language": "yor_Latn"}]}
    >>> chunk_language({}, chapter, ["P-1", "P-2", "P-3"])
    'yor_Latn'
    >>> chunk_language({}, chapter, ["P-2"])
    'eng_Latn'
    >>> chunk_language({"language": "hau_Latn"}, {"paragraphs": ["plain"]}, [])
    'hau_Latn'
    >>> chunk_language({}, {"paragraphs": []}, []) is None
    True
    """
    wanted = set(paragraph_ids)
    counts: dict[str, int] = {}

    paragraphs = chapter.get("paragraphs")

    for paragraph in paragraphs if isinstance(paragraphs, list) else []:
        if not isinstance(paragraph, dict):
            continue

        label = paragraph.get("language")

        if label and paragraph.get("paragraph_id") in wanted:
            counts[label] = counts.get(label, 0) + 1

    if counts:
        return max(counts, key=lambda label: counts[label])

    return chapter.get("language") or document.get("language") or None


def finalize_chunk(
    document: dict[str, Any],
    source_file: Path,
    chapter: dict[str, Any],
    paragraph_ids: list[str],
    text_parts: list[str],
    paragraph_pages: list[int] | None = None,
) -> SourceChunk | None:
    r"""
    PURPOSE
    -------
    Turn accumulated paragraph text into an annotated
    :class:`SourceChunk`. Returns ``None`` when the text has fewer than
    ``MIN_CHUNK_WORDS`` words. The chunk's page range is the min/max of
    ``paragraph_pages`` — the pages its own contributing paragraphs
    actually came from — rather than the whole chapter's range, so a
    citation can narrow to e.g. "p. 47" instead of "pp. 40-62". Falls
    back to the chapter's page range when no paragraph page numbers are
    available (e.g. a DOCX/EPUB section, which has no real page concept).

    PARAMETERS
    ----------
    document          Parsed document dict the chunk belongs to.
    source_file       Path to the source file the chunk was read from.
    chapter           Chapter dict the chunk belongs to.
    paragraph_ids     Paragraph ids contributing to the chunk.
    text_parts        Text pieces to join into the chunk's text.
    paragraph_pages   Page number of each contributing paragraph (same
                       order as the source paragraphs); entries may be
                       ``None``. Defaults to no page information.

    RETURNS
    -------
    The built ``SourceChunk``, or ``None`` if it is too short.

    EXAMPLES
    --------
    >>> finalize_chunk({}, Path("b.json"), {"title": "T"}, [], ["too short"]) is None
    True
    >>> body = ("Momentum is the product of mass and velocity, written p = mv. "
    ...         "Therefore doubling mass doubles momentum. For example, a truck has more. ")
    >>> chunk = finalize_chunk(
    ...     {"document_id": "DOC-1"}, Path("b.json"),
    ...     {"chapter_id": "CH-1", "title": "Chapter 3: Dynamics", "source_page_start": 4, "source_page_end": 5},
    ...     ["P-1", "P-2"], [body * 4], paragraph_pages=[4, 4],
    ... )
    >>> chunk.topic, chunk.concept, chunk.formula
    ('Chapter 3: Dynamics', 'Momentum', 'p = mv')
    >>> chunk.derivation, chunk.example
    ('Therefore doubling mass doubles momentum.', 'For example, a truck has more.')
    >>> chunk.paragraph_ids, chunk.source_page_start, chunk.word_count >= MIN_CHUNK_WORDS
    (['P-1', 'P-2'], 4, True)
    >>> chunk.chunk_id.startswith("CHUNK-") and chunk.document_id.startswith("DOC-")
    True
    >>> # a chunk spanning pages 6-7 of paragraphs narrows below the chapter's 4-5 range
    >>> wide_chapter = {"chapter_id": "CH-1", "title": "T", "source_page_start": 4, "source_page_end": 10}
    >>> narrow = finalize_chunk(
    ...     {"document_id": "DOC-1"}, Path("b.json"), wide_chapter,
    ...     ["P-3"], [body * 4], paragraph_pages=[6, 7],
    ... )
    >>> narrow.source_page_start, narrow.source_page_end
    (6, 7)
    >>> # falls back to the chapter's range when no paragraph pages are known
    >>> no_pages = finalize_chunk(
    ...     {"document_id": "DOC-1"}, Path("b.json"), wide_chapter,
    ...     ["P-4"], [body * 4], paragraph_pages=[None, None],
    ... )
    >>> no_pages.source_page_start, no_pages.source_page_end
    (4, 10)
    """
    text = normalize_text(" ".join(text_parts))

    if word_count(text) < MIN_CHUNK_WORDS:
        return None

    chapter_title = normalize_text(
        chapter.get("title") or "Main Text"
    )

    topic = infer_topic(chapter_title, text)
    concept = infer_concept(text, topic)
    sentences = split_sentences(text)
    formulas = extract_formulas(text)
    formula = formulas[0] if formulas else None
    derivation = select_sentence(sentences, DERIVATION_MARKERS)
    example = select_sentence(sentences, EXAMPLE_MARKERS)

    source_document_id = document.get("document_id")
    chapter_id = chapter.get("chapter_id")

    document_id = stable_record_id(
        "DOC",
        str(source_document_id or ""),
        source_file.name,
    )
    chunk_id = stable_record_id(
        "CHUNK",
        document_id,
        str(chapter_id or chapter_title),
        text,
    )

    known_pages = [page for page in (paragraph_pages or []) if page is not None]

    if known_pages:
        source_page_start = min(known_pages)
        source_page_end = max(known_pages)
    else:
        source_page_start = chapter.get("source_page_start")
        source_page_end = chapter.get("source_page_end")

    return SourceChunk(
        language=chunk_language(document, chapter, paragraph_ids),
        source_type=document.get("source_type"),
        tool=document.get("tool"),
        date=document.get("date"),
        document_id=document_id,
        chunk_id=chunk_id,
        source_file=source_file.name,
        source_path=str(document.get("source_path") or source_file),
        source_document_id=source_document_id,
        chapter_id=chapter_id,
        chapter_title=chapter_title,
        paragraph_ids=[pid for pid in paragraph_ids if pid],
        source_page_start=source_page_start,
        source_page_end=source_page_end,
        text=text,
        word_count=word_count(text),
        topic=topic,
        concept=concept,
        formula=formula,
        derivation=derivation,
        example=example,
    )


def overlap_units(
    units: list[SentenceUnit],
    overlap_words: int,
) -> list[SentenceUnit]:
    r"""
    PURPOSE
    -------
    The trailing whole sentences of a finished chunk, up to
    ``overlap_words`` words in total, carried into the next chunk so
    meaning is not lost at the seam. Whole sentences only, so the next
    chunk still starts at the beginning of a sentence. Never carries the
    entire chunk, or packing would not advance.

    PARAMETERS
    ----------
    units           The finished chunk's sentence units.
    overlap_words   Maximum words to carry over.

    RETURNS
    -------
    The units to carry, in reading order (possibly empty).

    EXAMPLES
    --------
    >>> units = sentence_units([("P", "One two three. Four five. Six seven.", 1, 1)])
    >>> [u.text for u in overlap_units(units, 5)]
    ['Four five.', 'Six seven.']
    >>> [u.text for u in overlap_units(units, 1)]
    []
    >>> [u.text for u in overlap_units(units, 0)]
    []
    """
    if overlap_words <= 0:
        return []

    carried: list[SentenceUnit] = []
    carried_words = 0

    for unit in reversed(units):
        if carried_words + unit.words > overlap_words:
            break

        carried.insert(0, unit)
        carried_words += unit.words

    if len(carried) >= len(units):
        carried = carried[1:]

    return carried


def chunk_from_units(
    document: dict[str, Any],
    source_file: Path,
    chapter: dict[str, Any],
    units: list[SentenceUnit],
) -> SourceChunk | None:
    r"""
    PURPOSE
    -------
    Build a :class:`SourceChunk` from sentence units, collecting their
    paragraph ids (in order, without repeats) and page numbers.

    PARAMETERS
    ----------
    document      Parsed document dict the chunk belongs to.
    source_file   Path to the source file the chunk was read from.
    chapter       Chapter dict the chunk belongs to.
    units         The sentence units making up the chunk.

    RETURNS
    -------
    The built chunk, or ``None`` if it is shorter than MIN_CHUNK_WORDS.
    """
    paragraph_ids: list[str] = []
    pages: list[int | None] = []

    for unit in units:
        if unit.paragraph_id and unit.paragraph_id not in paragraph_ids:
            paragraph_ids.append(unit.paragraph_id)

        pages.extend([unit.page, unit.page_end])

    return finalize_chunk(
        document,
        source_file,
        chapter,
        paragraph_ids,
        [unit.text for unit in units],
        pages,
    )


def build_chunks(
    document: dict[str, Any],
    source_file: Path,
    target_words: int = TARGET_CHUNK_WORDS,
    overlap_words: int = CHUNK_OVERLAP_WORDS,
) -> Iterator[SourceChunk]:
    r"""
    PURPOSE
    -------
    Group a document's sentences into overlapping chunks of at most
    ~``target_words``, never crossing a chapter. Chunks always start and
    end on sentence boundaries, and consecutive chunks share up to
    ``overlap_words`` words of whole trailing sentences. Undersized
    leftovers at the end of a chapter are merged into the previous chunk.

    PARAMETERS
    ----------
    document         Parsed document dict to chunk.
    source_file      Path to the source file the document was read from.
    target_words     Target number of words per chunk.
    overlap_words    Number of words carried over between consecutive
                      chunks.

    RETURNS
    -------
    An iterator of ``SourceChunk`` objects.

    EXAMPLES
    --------
    >>> sentence = "Water boils at one hundred degrees Celsius at sea level. "   # 10 words
    >>> paragraphs = [{"paragraph_id": f"P-{i}", "text": sentence * 6, "page": i + 1}
    ...               for i in range(3)]                                          # 60 words each
    >>> doc = {"document_id": "D", "chapters": [{"chapter_id": "C", "title": "Heat", "paragraphs": paragraphs}]}
    >>> chunks = list(build_chunks(doc, Path("b.json"), target_words=100, overlap_words=15))
    >>> [(c.word_count, c.paragraph_ids, c.source_page_start, c.source_page_end) for c in chunks]
    [(100, ['P-0', 'P-1'], 1, 2), (90, ['P-1', 'P-2'], 2, 3)]
    >>> # the next chunk opens with the last whole sentence of the previous one
    >>> chunks[0].text.endswith(sentence.strip()) and chunks[1].text.startswith(sentence.strip())
    True
    >>> # a long paragraph is split between sentences, never inside one
    >>> long_paragraph = " ".join(f"Sentence {i} talks about the calendar of the Kedang people." for i in range(80))
    >>> doc = {"chapters": [{"title": "Time", "paragraphs": [long_paragraph]}]}
    >>> chunks = list(build_chunks(doc, Path("b.json")))
    >>> len(chunks) > 1 and all(c.text.startswith("Sentence ") and c.text.endswith("people.") for c in chunks)
    True
    >>> list(build_chunks({"chapters": []}, Path("b.json")))
    []
    """
    for chapter in extract_chapters(document):
        units = sentence_units(extract_paragraphs(chapter), target_words)

        if not units:
            continue

        current: list[SentenceUnit] = []
        current_words = 0
        carried_count = 0
        previous: tuple[SourceChunk, list[SentenceUnit]] | None = None

        for unit in units:
            has_new_text = len(current) > carried_count

            if has_new_text and current_words + unit.words > target_words:
                chunk = chunk_from_units(document, source_file, chapter, current)

                if chunk is not None:
                    if previous is not None:
                        yield previous[0]

                    previous = (chunk, current)
                    carried = overlap_units(current, overlap_words)
                    current = list(carried)
                    current_words = sum(u.words for u in current)
                    carried_count = len(carried)

                # a too-short chunk (only possible with one huge unit) keeps
                # accumulating instead of being dropped

            current.append(unit)
            current_words += unit.words

        if len(current) > carried_count:
            chunk = chunk_from_units(document, source_file, chapter, current)

            if chunk is not None:
                if previous is not None:
                    yield previous[0]

                previous = (chunk, current)

            elif previous is not None:
                merged_units = previous[1] + current[carried_count:]
                merged = chunk_from_units(
                    document, source_file, chapter, merged_units
                )

                if merged is not None:
                    previous = (merged, merged_units)

        if previous is not None:
            yield previous[0]


def build_llm_records(chunk: SourceChunk) -> list[dict[str, Any]]:
    r"""
    PURPOSE
    -------
    Produce the 2-3 LLM training records for one chunk. These are
    heuristically generated from the chunk's own extracted structure
    (topic/concept/formula/derivation/example), not written or reviewed
    by a person and not produced by a language model — every record
    carries ``generation_method: "template"`` so a training pipeline can
    filter them out (or flag them for human review) instead of training
    directly on template-generated text.

    PARAMETERS
    ----------
    chunk   Source chunk to build records from.

    RETURNS
    -------
    List of LLM training record dicts.

    EXAMPLES
    --------
    >>> body = "Momentum is mass times velocity, written p = mv. Therefore it grows with mass. "
    >>> chunk = finalize_chunk({}, Path("b.json"), {"title": "Dynamics"}, [], [body * 5])
    >>> records = build_llm_records(chunk)
    >>> [r["type"] for r in records]
    ['question_reasoning_answer', 'concept_explanation', 'formula_derivation_example']
    >>> records[0]["question"]
    'What does the formula p = mv represent in the context of Dynamics?'
    >>> records[2]["formula"], records[2]["derivation"]
    ('p = mv', 'Therefore it grows with mass.')
    >>> {r["generation_method"] for r in records}
    {'template'}
    >>> plain = finalize_chunk({}, Path("b.json"), {"title": "Prose"}, [], ["Just words here. " * 40])
    >>> [r["type"] for r in build_llm_records(plain)]
    ['question_reasoning_answer', 'concept_explanation']
    """
    base = {
        "document_id": chunk.document_id,
        "chunk_id": chunk.chunk_id,
        "topic": chunk.topic,
        "source": SOURCE_LABEL,
        "source_file": chunk.source_file,
        "chapter_title": chunk.chapter_title,
        "language": chunk.language,
        "source_type": chunk.source_type,
        "tool": chunk.tool,
        "date": chunk.date,
        "generation_method": "template",
    }

    records: list[dict[str, Any]] = [
        {
            "record_id": stable_record_id("QA", chunk.chunk_id),
            "type": "question_reasoning_answer",
            "question": build_question(chunk.topic, chunk.concept, chunk.formula),
            "reasoning": build_reasoning(chunk),
            "answer": build_answer(chunk),
            **base,
        },
        {
            "record_id": stable_record_id("CE", chunk.chunk_id),
            "type": "concept_explanation",
            "concept": chunk.concept,
            "explanation": build_explanation(chunk.text),
            **base,
        },
    ]

    if chunk.formula:
        records.append(
            {
                "record_id": stable_record_id("FD", chunk.chunk_id),
                "type": "formula_derivation_example",
                "formula": chunk.formula,
                "derivation": chunk.derivation
                or "No explicit derivation was stated in the source passage.",
                "example": chunk.example
                or "No explicit worked example was stated in the source passage.",
                **base,
            }
        )

    return records


def build_rag_record(chunk: SourceChunk) -> dict[str, Any]:
    r"""
    PURPOSE
    -------
    Produce the RAG record (text + source metadata) for one chunk. This
    program's job ends at producing clean, citable text — embedding it
    is embedding_program/embedding.py's job, not this one's. Embedding
    chunks here as well as there would mean embedding every chunk twice
    and shipping stale/duplicate vectors in two places.

    PARAMETERS
    ----------
    chunk   Source chunk to build the record from.

    RETURNS
    -------
    The RAG record dict.

    EXAMPLES
    --------
    >>> chunk = finalize_chunk({"document_id": "DOC-9"}, Path("b.json"), {"title": "Prose"}, ["P-1"], ["Just words here. " * 40])
    >>> record = build_rag_record(chunk)
    >>> record["topic"], record["source_metadata"]["paragraph_ids"]
    ('Prose', ['P-1'])
    >>> sorted(record)
    ['chunk_id', 'concept', 'document_id', 'formula', 'source', 'source_metadata', 'text', 'topic']
    """
    return {
        "document_id": chunk.document_id,
        "chunk_id": chunk.chunk_id,
        "topic": chunk.topic,
        "concept": chunk.concept,
        "formula": chunk.formula,
        "text": chunk.text,
        "source": SOURCE_LABEL,
        "source_metadata": {
            "source_file": chunk.source_file,
            "source_path": chunk.source_path,
            "source_document_id": chunk.source_document_id,
            "chapter_id": chunk.chapter_id,
            "chapter_title": chunk.chapter_title,
            "paragraph_ids": chunk.paragraph_ids,
            "source_page_start": chunk.source_page_start,
            "source_page_end": chunk.source_page_end,
            "word_count": chunk.word_count,
            "language": chunk.language,
            "source_type": chunk.source_type,
            "tool": chunk.tool,
            "date": chunk.date,
        },
    }


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> int:
    r"""
    PURPOSE
    -------
    Atomically write records as JSONL; returns the number of lines
    written.

    PARAMETERS
    ----------
    path      Destination path for the JSONL file.
    records   Records to write, one JSON object per line.

    RETURNS
    -------
    The number of lines written.

    EXAMPLES
    --------
    >>> import tempfile
    >>> with tempfile.TemporaryDirectory() as folder:
    ...     path = Path(folder) / "out" / "data.jsonl"
    ...     n = write_jsonl(path, [{"a": 1}, {"b": "é"}])
    ...     (n, path.read_text(encoding="utf-8"), [p.name for p in path.parent.iterdir()])
    (2, '{"a": 1}\n{"b": "é"}\n', ['data.jsonl'])
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


def discover_input_files(input_folder: Path = INPUT_FOLDER) -> list[Path]:
    r"""
    PURPOSE
    -------
    List .json/.jsonl files under ``input_folder``, skipping hidden
    files and every pipeline stage's ``*_manifest.json``.

    PARAMETERS
    ----------
    input_folder   Root folder to search for input files.

    RETURNS
    -------
    Sorted list of matching file paths.

    EXAMPLES
    --------
    >>> import tempfile
    >>> with tempfile.TemporaryDirectory() as folder:
    ...     root = Path(folder)
    ...     for name in ("b.json", "a.JSONL", ".hidden.json", "notes.txt",
    ...                  "cleaning_manifest.json", "language_detection_manifest.json"):
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
        and path.name != MANIFEST_FILE.name
        and not path.stem.casefold().endswith("_manifest")
    )


def run_pipeline(
    input_folder: Path = INPUT_FOLDER,
    output_folder: Path = OUTPUT_FOLDER,
) -> dict[str, Any]:
    r"""
    PURPOSE
    -------
    Chunk every input file and write both datasets (rag_data.jsonl,
    llm_training_data.jsonl) and a manifest. This program's output is
    plain text plus metadata, not vectors — embedding is
    embedding_program/embedding.py's job, done exactly once downstream,
    rather than here as well (which used to mean every chunk got
    embedded twice, with this program's copy going stale the moment
    embedding.py's model changed).

    PARAMETERS
    ----------
    input_folder    Root folder containing cleaned input documents.
    output_folder   Root folder to write output datasets and the
                     manifest to.

    RETURNS
    -------
    The manifest dict describing the run.

    RAISES
    ------
    FileNotFoundError   Raised when ``input_folder`` does not exist.

    EXAMPLES
    --------
    >>> import tempfile
    >>> body = "Force is mass times acceleration, so F = ma. Hence heavier things need more force. "
    >>> doc = {"document_id": "DOC-1", "chapters": [{"chapter_id": "CH-1", "title": "Forces",
    ...        "paragraphs": [{"paragraph_id": "P-1", "text": body * 6}]}]}
    >>> with tempfile.TemporaryDirectory() as folder:
    ...     inp, out = Path(folder) / "in", Path(folder) / "out"
    ...     inp.mkdir()
    ...     _ = (inp / "book_cleaned.json").write_text(json.dumps(doc), encoding="utf-8")
    ...     _ = (inp / "broken.json").write_text("{not json", encoding="utf-8")
    ...     manifest = run_pipeline(inp, out)
    ...     written = sorted(p.name for p in out.iterdir())
    >>> written
    ['chunking_manifest.json', 'llm_training_data.jsonl', 'rag_data.jsonl']
    >>> manifest["successful_files"], manifest["failed_files"], manifest["chunk_count"]
    (1, 1, 1)
    >>> manifest["llm_record_count"], manifest["rag_record_count"]
    (3, 1)
    >>> run_pipeline(Path("/definitely/not/here"), Path("/tmp/x"))
    Traceback (most recent call last):
        ...
    FileNotFoundError: Input folder does not exist: /definitely/not/here
    """
    if not input_folder.is_dir():
        raise FileNotFoundError(f"Input folder does not exist: {input_folder}")

    files = discover_input_files(input_folder)
    chunks: list[SourceChunk] = []
    file_results: list[dict[str, Any]] = []

    for path in files:
        try:
            documents = load_documents(path)
            file_chunks = [
                chunk
                for document in documents
                for chunk in build_chunks(document, path)
            ]
            chunks.extend(file_chunks)
            file_results.append(
                {
                    "status": "success",
                    "input": str(path),
                    "documents": len(documents),
                    "chunks": len(file_chunks),
                }
            )
        except Exception as error:
            file_results.append(
                {
                    "status": "failed",
                    "input": str(path),
                    "error": f"{type(error).__name__}: {error}",
                }
            )

    llm_file = output_folder / LLM_OUTPUT_FILE.name
    rag_file = output_folder / RAG_OUTPUT_FILE.name
    manifest_file = output_folder / MANIFEST_FILE.name

    llm_count = write_jsonl(
        llm_file,
        (record for chunk in chunks for record in build_llm_records(chunk)),
    )
    rag_count = write_jsonl(
        rag_file,
        (build_rag_record(chunk) for chunk in chunks),
    )

    manifest = {
        "input_folder": str(input_folder),
        "output_folder": str(output_folder),
        "source_label": SOURCE_LABEL,
        "total_files": len(files),
        "successful_files": sum(r["status"] == "success" for r in file_results),
        "failed_files": sum(r["status"] == "failed" for r in file_results),
        "chunk_count": len(chunks),
        "llm_record_count": llm_count,
        "rag_record_count": rag_count,
        "llm_output_file": str(llm_file),
        "rag_output_file": str(rag_file),
        "files": file_results,
    }

    output_folder.mkdir(parents=True, exist_ok=True)
    temporary_manifest = manifest_file.with_suffix(".json.tmp")

    with temporary_manifest.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)

    temporary_manifest.replace(manifest_file)
    return manifest


def pick_input_folder(candidates: Iterable[Path] = UPSTREAM_FOLDERS) -> Path:
    r"""
    PURPOSE
    -------
    Choose the latest pipeline stage that has documents in it, so chunking
    reads deduplicated, language-tagged text once those stages have run,
    and plain cleaned_data until then.

    PARAMETERS
    ----------
    candidates   Folders to try, latest stage first.

    RETURNS
    -------
    The first folder holding at least one input file; the last candidate
    when none do.

    EXAMPLES
    --------
    >>> import tempfile
    >>> with tempfile.TemporaryDirectory() as folder:
    ...     root = Path(folder)
    ...     tagged, cleaned = root / "tagged", root / "cleaned"
    ...     cleaned.mkdir()
    ...     _ = (cleaned / "a_cleaned.json").write_text("{}", encoding="utf-8")
    ...     before = pick_input_folder([tagged, cleaned]).name
    ...     tagged.mkdir()
    ...     _ = (tagged / "a_cleaned.json").write_text("{}", encoding="utf-8")
    ...     after = pick_input_folder([tagged, cleaned]).name
    >>> before, after
    ('cleaned', 'tagged')
    """
    candidates = list(candidates)

    for folder in candidates:
        if folder.is_dir() and discover_input_files(folder):
            return folder

    return candidates[-1]


def main() -> None:
    """
    PURPOSE
    -------
    Run the pipeline on the latest upstream stage's folder and print a
    summary.
    """
    input_folder = pick_input_folder()
    manifest = run_pipeline(input_folder)

    print(
        f"Input: {input_folder}\n"
        "Chunking completed.\n"
        f"Files: {manifest['successful_files']} ok, "
        f"{manifest['failed_files']} failed\n"
        f"Chunks: {manifest['chunk_count']}\n"
        f"LLM records: {manifest['llm_record_count']} -> "
        f"{manifest['llm_output_file']}\n"
        f"RAG records: {manifest['rag_record_count']} -> "
        f"{manifest['rag_output_file']}"
    )


if __name__ == "__main__":
    main()