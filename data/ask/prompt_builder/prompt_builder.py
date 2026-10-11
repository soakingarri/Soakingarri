#!/usr/bin/env python3

"""
SoakinGarri Prompt Builder Program

PURPOSE
-------
Packs retrieved passages into a grounded prompt for the answer generator.
Passages are numbered [1], [2], ... in rank order, and the prompt carries
three rules:

    1. Answer only from the numbered sources.
    2. Cite every claim with its source number, e.g. [1] or [1][3].
    3. When the sources do not contain the answer, say so plainly instead
       of guessing.

The answer is also requested in the question's language when it is known
(query_processing.py detects it).

OUTPUT
------
A PromptPackage holding chat messages in the common
[{"role": ..., "content": ...}] format that OpenAI-style chat models (and
most others) accept, plus the numbered passages, so citation_formatter.py
can later turn each [n] in the answer back into its source.

INPUT
-----
Any list of results with `.text` and `.record` attributes: SearchResult
objects from retrieval.py (vector, keyword or hybrid) or RerankedResult
objects from reranker.py.

RUN
---
python3 prompt_builder.py --question "how is momentum defined" --hybrid
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(
    0, str(Path(__file__).resolve().parent.parent.parent.parent / "source")
)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "retrieval"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "vector_database"))

from common import normalize_text  # noqa: E402

DEFAULT_MAX_PASSAGES = 8
DEFAULT_PASSAGE_CHARACTERS = 1500
DEFAULT_CONTEXT_CHARACTERS = 9000
MAX_HISTORY_TURNS = 4

NOT_IN_SOURCES = "The provided sources do not contain the answer to this question."

# GlotLID labels for the languages most likely to come up, so the
# instruction can name the language rather than give a code.
LANGUAGE_NAMES = {
    "eng_Latn": "English",
    "yor_Latn": "Yoruba",
    "ibo_Latn": "Igbo",
    "hau_Latn": "Hausa",
    "pcm_Latn": "Nigerian Pidgin",
    "swh_Latn": "Swahili",
    "swa_Latn": "Swahili",
    "amh_Ethi": "Amharic",
    "twi_Latn": "Twi",
    "aka_Latn": "Akan",
    "wol_Latn": "Wolof",
    "zul_Latn": "Zulu",
    "xho_Latn": "Xhosa",
    "som_Latn": "Somali",
    "fra_Latn": "French",
    "por_Latn": "Portuguese",
    "arb_Arab": "Arabic",
}

SYSTEM_RULES = """You answer questions using only the numbered sources provided.

Rules:
1. Use only information found in the sources. Do not use outside knowledge.
2. Cite the source of every claim with its number in square brackets, such as [1], placed right after the claim. Cite several sources as [1][3].
3. Only cite numbers that appear in the sources list.
4. If the sources do not contain the answer, reply exactly: "{not_in_sources}" Do not guess.
5. If the sources only partly answer the question, answer that part, cite it, and say what is missing."""


@dataclass(slots=True)
class Passage:
    """
    PURPOSE
    -------
    One numbered source in a prompt.

    ATTRIBUTES
    ----------
    number     The citation number, starting at 1.
    text       The passage text as placed in the prompt (possibly trimmed).
    citation   Short human-readable source label shown to the model.
    record     The full retrieval record, carried through unchanged.
    """

    number: int
    text: str
    citation: str
    record: dict[str, Any] = field(repr=False)


@dataclass(slots=True)
class PromptPackage:
    """
    PURPOSE
    -------
    Everything the generator and the citation formatter need.

    ATTRIBUTES
    ----------
    messages   Chat messages: one system message with the rules, earlier
                turns if any, and one user message with sources + question.
    passages   The numbered passages, in citation order.
    question   The question the prompt asks.
    """

    messages: list[dict[str, str]]
    passages: list[Passage]
    question: str


def language_instruction(language: str | None) -> str:
    r"""
    PURPOSE
    -------
    The sentence telling the model which language to answer in.

    PARAMETERS
    ----------
    language   A GlotLID label such as "yor_Latn", "und", or None.

    RETURNS
    -------
    The instruction, or '' when the language is unknown.

    EXAMPLES
    --------
    >>> language_instruction("yor_Latn")
    'Write the answer in Yoruba, the language of the question.'
    >>> language_instruction("kin_Latn")
    'Write the answer in the language of the question (kin_Latn).'
    >>> language_instruction("und"), language_instruction(None)
    ('', '')
    """
    if not language or language == "und":
        return ""

    name = LANGUAGE_NAMES.get(language)

    if name:
        return f"Write the answer in {name}, the language of the question."

    return f"Write the answer in the language of the question ({language})."


def trim_passage(text: str, limit: int) -> str:
    r"""
    PURPOSE
    -------
    Collapse whitespace and cut a passage to `limit` characters, at a
    sentence end when one falls in the last third, otherwise at a word
    boundary.

    PARAMETERS
    ----------
    text    The passage text.
    limit   Maximum characters to keep.

    RETURNS
    -------
    The trimmed text, ending in " …" when it was cut mid-sentence.

    EXAMPLES
    --------
    >>> trim_passage("Short passage.", 100)
    'Short passage.'
    >>> trim_passage("First sentence here. Second sentence is longer.", 30)
    'First sentence here.'
    >>> trim_passage("one two three four five six", 12)
    'one two …'
    """
    text = normalize_text(text)

    if len(text) <= limit:
        return text

    cut = text[:limit]
    sentence_end = max(cut.rfind(". "), cut.rfind("? "), cut.rfind("! "))

    if sentence_end >= limit * 2 // 3 - 1:
        return cut[: sentence_end + 1]

    return f"{cut.rsplit(' ', 1)[0]} …"


def number_passages(
    results: list[Any],
    max_passages: int = DEFAULT_MAX_PASSAGES,
    passage_characters: int = DEFAULT_PASSAGE_CHARACTERS,
    context_characters: int = DEFAULT_CONTEXT_CHARACTERS,
) -> list[Passage]:
    r"""
    PURPOSE
    -------
    Turn ranked results into numbered passages, best first. Results with no
    text, and exact repeats of an earlier passage, are skipped so every
    number points to something distinct. Passages stop being added once the
    character budget is spent, but the first one is always kept.

    PARAMETERS
    ----------
    results              Ranked results with `.text` and `.record`.
    max_passages         Maximum passages to number.
    passage_characters   Per-passage character limit.
    context_characters   Total character budget across passages.

    RETURNS
    -------
    The numbered passages.

    EXAMPLES
    --------
    >>> from vector_base import SearchResult
    >>> results = [
    ...     SearchResult(1, 0.9, 0, "Momentum is mass times velocity.",
    ...                  {"source_file": "physics.pdf", "source_page_start": 47}),
    ...     SearchResult(2, 0.8, 1, "", {"source_file": "empty.pdf"}),
    ...     SearchResult(3, 0.7, 2, "Momentum is mass times velocity.", {"source_file": "copy.pdf"}),
    ...     SearchResult(4, 0.6, 3, "Impulse changes momentum.", {"source_file": "notes.pdf"}),
    ... ]
    >>> [(p.number, p.citation, p.text) for p in number_passages(results)]
    [(1, 'physics.pdf | p. 47', 'Momentum is mass times velocity.'), (2, 'notes.pdf', 'Impulse changes momentum.')]
    >>> len(number_passages(results, max_passages=1))
    1
    """
    from retrieval import describe_source

    passages: list[Passage] = []
    seen: set[str] = set()
    used = 0

    for result in results:
        text = trim_passage(getattr(result, "text", "") or "", passage_characters)
        key = text.casefold()

        if not text or key in seen:
            continue

        if passages and used + len(text) > context_characters:
            break

        record = getattr(result, "record", {}) or {}
        seen.add(key)
        used += len(text)
        passages.append(
            Passage(
                number=len(passages) + 1,
                text=text,
                citation=describe_source(record),
                record=record,
            )
        )

        if len(passages) >= max_passages:
            break

    return passages


def format_sources(passages: list[Passage]) -> str:
    r"""
    PURPOSE
    -------
    Render numbered passages as the sources block of the prompt.

    PARAMETERS
    ----------
    passages   The numbered passages.

    RETURNS
    -------
    The sources block, or a line saying there are none.

    EXAMPLES
    --------
    >>> print(format_sources([Passage(1, "Heat flows.", "physics.pdf | p. 9", {})]))
    [1] (physics.pdf | p. 9)
    Heat flows.
    >>> format_sources([])
    '(no sources were found)'
    """
    if not passages:
        return "(no sources were found)"

    return "\n\n".join(
        f"[{passage.number}] ({passage.citation})\n{passage.text}" for passage in passages
    )


def build_prompt(
    question: str,
    results: list[Any],
    language: str | None = None,
    history: list[dict[str, str]] | None = None,
    max_passages: int = DEFAULT_MAX_PASSAGES,
    passage_characters: int = DEFAULT_PASSAGE_CHARACTERS,
    context_characters: int = DEFAULT_CONTEXT_CHARACTERS,
) -> PromptPackage:
    r"""
    PURPOSE
    -------
    Build the grounded chat prompt for one question.

    PARAMETERS
    ----------
    question             The (standalone) question to answer; must not be
                          empty.
    results              Ranked retrieval results with `.text` and
                          `.record`.
    language             GlotLID label of the question, or None.
    history              Earlier turns as {"role", "content"} messages;
                          only the last few are kept, and they are given to
                          the model for tone and continuity, not as sources.
    max_passages         Maximum passages to include.
    passage_characters   Per-passage character limit.
    context_characters   Total passage character budget.

    RETURNS
    -------
    A PromptPackage.

    RAISES
    ------
    ValueError   When the question is empty.

    EXAMPLES
    --------
    >>> from vector_base import SearchResult
    >>> results = [SearchResult(1, 0.9, 0, "Momentum is mass times velocity.",
    ...                         {"source_file": "physics.pdf", "source_page_start": 47})]
    >>> package = build_prompt("How is momentum defined?", results, language="yor_Latn")
    >>> [m["role"] for m in package.messages]
    ['system', 'user']
    >>> "Cite the source of every claim" in package.messages[0]["content"]
    True
    >>> print(package.messages[1]["content"])
    Sources:
    <BLANKLINE>
    [1] (physics.pdf | p. 47)
    Momentum is mass times velocity.
    <BLANKLINE>
    Question: How is momentum defined?
    <BLANKLINE>
    Write the answer in Yoruba, the language of the question.
    >>> build_prompt(" ", results)
    Traceback (most recent call last):
        ...
    ValueError: The question is empty.
    """
    question = normalize_text(question)

    if not question:
        raise ValueError("The question is empty.")

    passages = number_passages(
        results,
        max_passages=max_passages,
        passage_characters=passage_characters,
        context_characters=context_characters,
    )

    messages: list[dict[str, str]] = [
        {"role": "system", "content": SYSTEM_RULES.format(not_in_sources=NOT_IN_SOURCES)}
    ]

    # Only user/assistant turns are kept: a "system" turn from the client
    # must never be able to override the grounding rules.
    usable_turns = [
        {"role": turn["role"], "content": str(turn["content"])}
        for turn in history or []
        if turn.get("role") in {"user", "assistant"} and turn.get("content")
    ]
    messages.extend(usable_turns[-MAX_HISTORY_TURNS:])

    parts = [f"Sources:\n\n{format_sources(passages)}", f"Question: {question}"]
    instruction = language_instruction(language)

    if instruction:
        parts.append(instruction)

    messages.append({"role": "user", "content": "\n\n".join(parts)})

    return PromptPackage(messages=messages, passages=passages, question=question)


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
    >>> options = parse_arguments(["--question", "What is heat?", "--top-k", "6"])
    >>> options.question, options.top_k, options.hybrid
    ('What is heat?', 6, False)
    """
    parser = argparse.ArgumentParser(
        description="Retrieve passages for a question and print the grounded prompt."
    )
    parser.add_argument("--question", required=True, help="the question to answer")
    parser.add_argument("--top-k", type=int, default=DEFAULT_MAX_PASSAGES, help="passages to include")
    parser.add_argument("--hybrid", action="store_true", help="use hybrid vector+BM25 retrieval")
    parser.add_argument("--language", help="GlotLID label of the question, e.g. yor_Latn")

    return parser.parse_args(argument_list)


def main() -> None:
    """
    PURPOSE
    -------
    Retrieve passages for the question on the command line and print the
    resulting chat messages as JSON, for inspecting what the model will see.
    """
    import retrieval as ret
    from vector_base import VectorBase

    options = parse_arguments()

    base = VectorBase.load(ret.VECTOR_BASE_FOLDER)
    encode, _ = ret.load_query_encoder()

    filters = {"dataset": ret.DEFAULT_DATASET}

    if options.hybrid:
        from keyword_index import KeywordIndex

        results = ret.hybrid_retrieve(
            base, KeywordIndex.load(ret.KEYWORD_INDEX_FOLDER), options.question, encode,
            top_k=options.top_k, filters=filters,
        )
    else:
        results = ret.retrieve(base, options.question, encode, top_k=options.top_k, filters=filters)

    package = build_prompt(options.question, results, language=options.language)

    print(json.dumps(package.messages, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
