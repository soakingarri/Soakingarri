#!/usr/bin/env python3

"""
SoakinGarri Query Processing Program

PURPOSE
-------
Prepares a user's question for retrieval, before retrieval.py runs:

    1. Rewrite      A follow-up such as "what about its formula?" means
                    nothing to a search index on its own. It is rewritten
                    into a standalone question using the conversation
                    history.
    2. Language     The question's language is detected with GlotLID (the
                    same model language_detection.py uses for the corpus),
                    so retrieval can filter on it and the answer can be
                    written in it.
    3. Expansion    Optionally, extra phrasings of the question are
                    produced and searched too; their result lists are
                    merged with Reciprocal Rank Fusion.

MODELS
------
Rewriting and expansion work without any language model, using simple,
explainable heuristics. Either step can instead be handed to a model
through a plain callable, `text_generator(prompt) -> str`, so any
generator (an OpenAI model, or your own later) can be plugged in without
this module depending on it.

INSTALL
-------
python3 -m pip install fasttext huggingface_hub     (language detection)

RUN
---
python3 query_processing.py --query "what about its formula?" \\
    --history "What is momentum?"
python3 query_processing.py --query "Kí ni orí?" --expand
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

sys.path.insert(
    0, str(Path(__file__).resolve().parent.parent.parent.parent / "source")
)
sys.path.insert(
    0,
    str(
        Path(__file__).resolve().parent.parent.parent.parent
        / "source"
        / "language_detection_program"
    ),
)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "retrieval"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "vector_database"))

from common import normalize_text  # noqa: E402

TextGenerator = Callable[[str], str]

# Queries are short, so the corpus tagger's 20-character floor is too high.
MIN_QUERY_DETECTION_CHARACTERS = 8
MIN_QUERY_CONFIDENCE = 0.5
UNDETERMINED = "und"

MAX_HISTORY_TURNS = 3
MAX_EXPANSIONS = 3

# Words that, at the start of a short question, usually point back to
# something said earlier ("it", "what about ...").
FOLLOW_UP_OPENERS = (
    "it", "its", "it's", "they", "them", "their", "this", "that", "these",
    "those", "he", "she", "his", "her", "and", "also", "so", "then",
    "what about", "how about", "why is that", "why", "same", "more",
    "explain more", "tell me more", "and what", "what else",
)

# Pronouns anywhere in a question that cannot be resolved without context.
DANGLING_PRONOUNS = {"it", "its", "they", "them", "their", "this", "that", "these", "those"}

STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "of", "in",
    "on", "at", "to", "for", "and", "or", "but", "with", "by", "from", "as",
    "what", "which", "who", "whom", "whose", "when", "where", "why", "how",
    "do", "does", "did", "can", "could", "would", "should", "will", "shall",
    "i", "you", "we", "me", "my", "your", "our", "please", "about", "tell",
    "explain", "describe", "define", "give", "show", "it", "its", "they",
    "them", "their", "this", "that", "these", "those", "there", "here",
    "more", "also", "so", "then", "same", "else", "into", "than", "not",
    "ok", "okay", "yes", "no", "thanks", "thank", "great", "cool", "sure",
}

WORD_PATTERN = re.compile(r"[\w'-]+", re.UNICODE)

REWRITE_PROMPT = (
    "Rewrite the user's last question as one standalone question that can "
    "be understood without the conversation. Keep its language. Return only "
    "the question.\n\nConversation:\n{history}\n\nLast question: {question}\n"
    "Standalone question:"
)

EXPAND_PROMPT = (
    "Write {count} different short search queries, one per line, that would "
    "find passages answering this question. Keep its language. Return only "
    "the queries.\n\nQuestion: {question}\nQueries:"
)


@dataclass(slots=True)
class ProcessedQuery:
    """
    PURPOSE
    -------
    The result of processing one question.

    ATTRIBUTES
    ----------
    original      The question exactly as the user typed it.
    standalone    The question rewritten to stand on its own (identical to
                   the original when no rewrite was needed).
    language      GlotLID label of the question, e.g. "yor_Latn", or "und".
    confidence    The language prediction's confidence.
    rewritten     Whether the standalone question differs from the original.
    expansions    Extra search phrasings, not including `standalone`.
    """

    original: str
    standalone: str
    language: str = UNDETERMINED
    confidence: float = 0.0
    rewritten: bool = False
    expansions: list[str] = field(default_factory=list)

    @property
    def search_queries(self) -> list[str]:
        """
        PURPOSE
        -------
        Every query to search with: the standalone question first, then
        each expansion.
        """
        return [self.standalone, *self.expansions]


def content_words(text: str) -> list[str]:
    r"""
    PURPOSE
    -------
    The words of a text that carry meaning for search, in order, without
    stopwords or repeats.

    PARAMETERS
    ----------
    text   The text to read.

    RETURNS
    -------
    The lowercase content words.

    EXAMPLES
    --------
    >>> content_words("What is the formula for kinetic energy? Kinetic!")
    ['formula', 'kinetic', 'energy']
    >>> content_words("What about it?")
    []
    """
    words: list[str] = []

    for word in WORD_PATTERN.findall(normalize_text(text).casefold()):
        if word not in STOPWORDS and word not in words:
            words.append(word)

    return words


def is_follow_up(question: str, history: list[str] | None) -> bool:
    r"""
    PURPOSE
    -------
    Decide whether a question leans on earlier turns: it opens with a
    pointing word ("it", "what about", "and"), or uses a pronoun while
    having too few content words of its own to search with.

    PARAMETERS
    ----------
    question   The latest question.
    history    Earlier user questions, oldest first; None or [] means this
                is the first question, which is never a follow-up.

    RETURNS
    -------
    True when the question should be rewritten.

    EXAMPLES
    --------
    >>> is_follow_up("What about its formula?", ["What is momentum?"])
    True
    >>> is_follow_up("And for light?", ["What is refraction in water?"])
    True
    >>> is_follow_up("How is it measured?", ["What is entropy?"])
    True
    >>> is_follow_up("What is the formula for kinetic energy?", ["What is momentum?"])
    False
    >>> is_follow_up("What about its formula?", [])
    False
    """
    if not history:
        return False

    text = normalize_text(question).casefold().strip(" ?!.")

    for opener in FOLLOW_UP_OPENERS:
        if text == opener or text.startswith(opener + " "):
            return True

    words = WORD_PATTERN.findall(text)
    has_pronoun = any(word in DANGLING_PRONOUNS for word in words)

    return has_pronoun and len(content_words(text)) <= 2


def rewrite_standalone(
    question: str,
    history: list[str] | None,
    rewriter: TextGenerator | None = None,
) -> str:
    r"""
    PURPOSE
    -------
    Turn a follow-up into a standalone question. With a rewriter (any
    `prompt -> text` callable, such as a chat model) the model does the
    rewrite. Without one, the subject words of the most recent earlier
    question that has any are appended in brackets, which is enough for
    retrieval to find the right passages.

    PARAMETERS
    ----------
    question   The latest question.
    history    Earlier user questions, oldest first.
    rewriter   Optional `prompt -> text` callable. If it fails or returns
                nothing, the heuristic rewrite is used instead.

    RETURNS
    -------
    The standalone question; the question itself, normalised, when it is
    not a follow-up.

    EXAMPLES
    --------
    >>> rewrite_standalone("What about its formula?", ["What is momentum?"])
    'What about its formula? (momentum)'
    >>> rewrite_standalone("And why?", ["Define kinetic energy", "ok"])
    'And why? (kinetic energy)'
    >>> rewrite_standalone("What is momentum?", ["What is light?"])
    'What is momentum?'
    >>> rewrite_standalone("What about its formula?", ["What is momentum?"],
    ...                    rewriter=lambda prompt: "What is the formula for momentum?")
    'What is the formula for momentum?'
    >>> def broken(prompt):
    ...     raise RuntimeError("model offline")
    >>> rewrite_standalone("What about its formula?", ["What is momentum?"], rewriter=broken)
    'What about its formula? (momentum)'
    """
    question = normalize_text(question)

    if not is_follow_up(question, history):
        return question

    recent = [normalize_text(turn) for turn in (history or [])][-MAX_HISTORY_TURNS:]

    if rewriter is not None:
        prompt = REWRITE_PROMPT.format(
            history="\n".join(f"User: {turn}" for turn in recent),
            question=question,
        )

        try:
            rewritten = normalize_text(rewriter(prompt)).strip().strip('"')
        except Exception:  # a model failure must never block the question
            rewritten = ""

        if rewritten:
            return rewritten

    for turn in reversed(recent):
        subject = content_words(turn)

        if subject:
            return f"{question} ({' '.join(subject)})"

    return question


def detect_query_language(
    question: str,
    model: Any = None,
    min_confidence: float = MIN_QUERY_CONFIDENCE,
) -> tuple[str, float]:
    r"""
    PURPOSE
    -------
    Identify a question's language with GlotLID, through
    language_detection.detect_language so queries and corpus share one
    labelling scheme.

    PARAMETERS
    ----------
    question         The question text.
    model            A loaded language model; None loads GlotLID.
    min_confidence   Predictions below this give "und".

    RETURNS
    -------
    A (label, confidence) tuple such as ("yor_Latn", 0.97).

    EXAMPLES
    --------
    >>> class StubModel:
    ...     def predict(self, text, k=1):
    ...         return (("__label__yor_Latn",), (0.97,))
    >>> detect_query_language("Kí ni orí?", StubModel())
    ('yor_Latn', 0.97)
    >>> detect_query_language("Hi?", StubModel())
    ('und', 0.0)
    """
    import language_detection as ld

    if model is None:
        model = ld.load_language_model()

    return ld.detect_language(
        question,
        model,
        min_characters=MIN_QUERY_DETECTION_CHARACTERS,
        min_confidence=min_confidence,
    )


def expand_query(
    question: str,
    expander: TextGenerator | None = None,
    max_expansions: int = MAX_EXPANSIONS,
) -> list[str]:
    r"""
    PURPOSE
    -------
    Produce extra search phrasings of a question. Without an expander the
    only extra phrasing is the bare content words, which helps the BM25
    side of hybrid search. With one (a `prompt -> text` callable), its
    lines are used as additional phrasings.

    PARAMETERS
    ----------
    question         The standalone question.
    expander         Optional `prompt -> text` callable returning one query
                      per line. Failures are ignored.
    max_expansions   Maximum phrasings to return.

    RETURNS
    -------
    Distinct phrasings, none equal to the question itself.

    EXAMPLES
    --------
    >>> expand_query("What is the formula for kinetic energy?")
    ['formula kinetic energy']
    >>> expand_query("Momentum?")
    []
    >>> fake = lambda prompt: "1. kinetic energy equation\n- KE = 1/2 m v^2\n\nkinetic energy equation"
    >>> expand_query("What is the formula for kinetic energy?", expander=fake)
    ['kinetic energy equation', 'KE = 1/2 m v^2', 'formula kinetic energy']
    """
    question = normalize_text(question)
    candidates: list[str] = []

    if expander is not None:
        try:
            raw = expander(EXPAND_PROMPT.format(count=max_expansions, question=question))
        except Exception:
            raw = ""

        for line in str(raw or "").splitlines():
            line = re.sub(r"^\s*(?:\d+[.)]|[-*•])\s*", "", line).strip().strip('"')

            if line:
                candidates.append(line)

    keywords = " ".join(content_words(question))

    if keywords:
        candidates.append(keywords)

    expansions: list[str] = []
    seen = {question.casefold()}

    for candidate in candidates:
        key = candidate.casefold()

        # A keyword list identical to the question's own words adds nothing.
        if key in seen or key == question.casefold().strip(" ?!."):
            continue

        seen.add(key)
        expansions.append(candidate)

    return expansions[:max_expansions]


def process_query(
    question: str,
    history: list[str] | None = None,
    language_model: Any = None,
    rewriter: TextGenerator | None = None,
    expand: bool = False,
    expander: TextGenerator | None = None,
) -> ProcessedQuery:
    r"""
    PURPOSE
    -------
    Run the full preparation: rewrite, detect language, optionally expand.
    The language is detected on the user's original wording (a rewrite may
    have appended English-looking subject words), falling back to the
    standalone question when the original is too short to judge.

    PARAMETERS
    ----------
    question         The latest question; must not be empty.
    history          Earlier user questions, oldest first.
    language_model   A loaded language model; None loads GlotLID.
    rewriter         Optional `prompt -> text` callable for rewriting.
    expand           Whether to produce extra phrasings.
    expander         Optional `prompt -> text` callable for expansion.

    RETURNS
    -------
    A ProcessedQuery.

    RAISES
    ------
    ValueError   When the question is empty.

    EXAMPLES
    --------
    >>> class StubModel:
    ...     def predict(self, text, k=1):
    ...         return (("__label__eng_Latn",), (0.99,))
    >>> result = process_query("What about its formula?", ["What is momentum?"],
    ...                        language_model=StubModel(), expand=True)
    >>> result.standalone, result.rewritten, result.language
    ('What about its formula? (momentum)', True, 'eng_Latn')
    >>> result.search_queries
    ['What about its formula? (momentum)', 'formula momentum']
    >>> process_query("  ", language_model=StubModel())
    Traceback (most recent call last):
        ...
    ValueError: The question is empty.
    """
    question = normalize_text(question)

    if not question:
        raise ValueError("The question is empty.")

    standalone = rewrite_standalone(question, history, rewriter)

    language, confidence = detect_query_language(question, language_model)

    if language == UNDETERMINED and standalone != question:
        language, confidence = detect_query_language(standalone, language_model)

    expansions = expand_query(standalone, expander) if expand else []

    return ProcessedQuery(
        original=question,
        standalone=standalone,
        language=language,
        confidence=confidence,
        rewritten=standalone != question,
        expansions=expansions,
    )


def retrieve_for_query(
    processed: ProcessedQuery,
    search: Callable[[str], list[Any]],
    top_k: int | None = None,
) -> list[Any]:
    r"""
    PURPOSE
    -------
    Search with every phrasing of a processed query and merge the result
    lists with retrieval.reciprocal_rank_fusion. With no expansions this is
    exactly one search.

    PARAMETERS
    ----------
    processed   The ProcessedQuery to search for.
    search      `query -> list[SearchResult]`, for example
                 `lambda q: retrieval.hybrid_retrieve(base, index, q, encode)`.
    top_k       Keep at most this many merged results, or None for all.

    RETURNS
    -------
    The merged SearchResult list, best first.

    EXAMPLES
    --------
    >>> from vector_base import SearchResult
    >>> def hit(rank, chunk_id):
    ...     return SearchResult(rank, 0.0, rank, chunk_id, {"dataset": "rag", "chunk_id": chunk_id})
    >>> hits = {"a": [hit(1, "C1"), hit(2, "C2")], "b": [hit(1, "C2"), hit(2, "C3")]}
    >>> processed = ProcessedQuery("a", "a", expansions=["b"])
    >>> [r.text for r in retrieve_for_query(processed, hits.get)]
    ['C2', 'C1', 'C3']
    >>> [r.text for r in retrieve_for_query(ProcessedQuery("a", "a"), hits.get)]
    ['C1', 'C2']
    """
    from retrieval import reciprocal_rank_fusion

    result_lists = [search(query) for query in processed.search_queries]

    if len(result_lists) == 1:
        return result_lists[0][:top_k] if top_k is not None else result_lists[0]

    return reciprocal_rank_fusion(result_lists, top_k=top_k)


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
    >>> options = parse_arguments(["--query", "And why?", "--history", "What is heat?", "--expand"])
    >>> options.query, options.history, options.expand
    ('And why?', ['What is heat?'], True)
    """
    parser = argparse.ArgumentParser(
        description="Rewrite, language-tag and expand a question for retrieval."
    )
    parser.add_argument("--query", required=True, help="the latest question")
    parser.add_argument(
        "--history",
        action="append",
        default=[],
        help="an earlier user question; repeat, oldest first",
    )
    parser.add_argument(
        "--expand", action="store_true", help="also produce extra search phrasings"
    )

    return parser.parse_args(argument_list)


def main() -> None:
    """
    PURPOSE
    -------
    Process the question given on the command line and print the result
    as JSON.
    """
    options = parse_arguments()

    processed = process_query(options.query, options.history, expand=options.expand)

    print(json.dumps(asdict(processed), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
