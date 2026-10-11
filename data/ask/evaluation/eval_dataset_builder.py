#!/usr/bin/env python3

"""
SoakinGarri Evaluation Dataset Builder

PURPOSE
-------
Builds and maintains the "golden" evaluation set: 200-500 questions with a
known answer and the exact source passages (chunk ids) that answer them.
rag_evaluation.py scores the system against this set after every change,
so its quality decides how much the scores can be trusted.

A golden set must be checked by a person. This tool therefore never marks
anything as golden by itself. It produces DRAFTS from three places and
lets a reviewer accept, edit or reject each one:

    draft             A question and reference answer drafted from a
                      sampled chunk, by template or by any text generator
                      you plug in (`prompt -> text`).
    import-feedback   Real user questions that got a thumbs up, from
                      feedback_logger.py, with the passages they were
                      answered from. These are the most realistic entries.
    add               A question you write yourself (stored as verified).

Only entries with status "verified" are used by rag_evaluation.py.

CHUNK IDS
---------
Chunk ids are derived from chunk text, so re-chunking a changed document
gives new ids. `validate` lists entries whose chunks no longer exist, so
they can be re-reviewed instead of silently scoring as misses.

OUTPUT
------
<repo>/data/ask/evaluation/golden_set.jsonl   (one entry per line)

RUN
---
python3 eval_dataset_builder.py draft --count 50
python3 eval_dataset_builder.py import-feedback
python3 eval_dataset_builder.py review
python3 eval_dataset_builder.py add --question "..." --answer "..." --chunk-id CHUNK-...
python3 eval_dataset_builder.py validate
python3 eval_dataset_builder.py stats
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

sys.path.insert(
    0, str(Path(__file__).resolve().parent.parent.parent.parent / "source")
)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "feedback"))

from common import DATA_ROOT, normalize_text, stable_id  # noqa: E402

CHUNKED_FOLDER = DATA_ROOT / "chunked_data"
RAG_FILE_NAME = "rag_data.jsonl"
GOLDEN_SET_FILE = DATA_ROOT / "evaluation" / "golden_set.jsonl"

STATUSES = ("draft", "verified", "rejected")
TARGET_MINIMUM = 200
TARGET_MAXIMUM = 500

MIN_CHUNK_WORDS = 60
REFERENCE_ANSWER_WORDS = 60

TextGenerator = Callable[[str], str]

DRAFT_PROMPT = """Read the passage and write ONE question a student might ask that this passage answers, and the answer, using only the passage.
Write the question and answer in the passage's language.
Reply with JSON only: {{"question": "...", "answer": "..."}}

Passage:
{text}"""


def utc_now() -> str:
    """
    PURPOSE
    -------
    The current UTC time as an ISO 8601 string.
    """
    return datetime.now(timezone.utc).isoformat()


def make_entry(
    question: str,
    reference_answer: str,
    source_chunk_ids: list[str],
    origin: str,
    status: str = "draft",
    **extra: Any,
) -> dict[str, Any]:
    r"""
    PURPOSE
    -------
    Build one golden-set entry. The id comes from the question and its
    chunks, so drafting the same thing twice gives the same id.

    PARAMETERS
    ----------
    question           The question text.
    reference_answer   The known correct answer.
    source_chunk_ids   Chunk ids of the passages that answer it.
    origin             "template", "generator", "feedback" or "manual".
    status             "draft", "verified" or "rejected".
    extra              Any further fields (source_file, language, ...).

    RETURNS
    -------
    The entry dict.

    RAISES
    ------
    ValueError   For an empty question, no chunk ids, or a bad status.

    EXAMPLES
    --------
    >>> entry = make_entry("What is momentum?", "Mass times velocity.", ["C1"], "manual",
    ...                    status="verified", language="eng_Latn")
    >>> entry["id"].startswith("GOLD-"), entry["status"], entry["language"]
    (True, 'verified', 'eng_Latn')
    >>> entry["id"] == make_entry("What is momentum?", "Other.", ["C1"], "template")["id"]
    True
    >>> make_entry("What?", "x", [], "manual")
    Traceback (most recent call last):
        ...
    ValueError: An entry needs a question and at least one source chunk id.
    """
    question = normalize_text(question)
    chunk_ids = [str(chunk_id) for chunk_id in source_chunk_ids if chunk_id]

    if not question or not chunk_ids:
        raise ValueError("An entry needs a question and at least one source chunk id.")

    if status not in STATUSES:
        raise ValueError(f"Status must be one of {STATUSES}, not {status!r}.")

    entry = {
        "id": stable_id("GOLD", question.casefold(), *sorted(chunk_ids)),
        "question": question,
        "reference_answer": normalize_text(reference_answer),
        "source_chunk_ids": chunk_ids,
        "status": status,
        "origin": origin,
        "created_at": utc_now(),
    }
    entry.update(extra)

    return entry


def load_golden_set(path: Path = GOLDEN_SET_FILE) -> list[dict[str, Any]]:
    r"""
    PURPOSE
    -------
    Read the golden set; a missing file is an empty set.

    PARAMETERS
    ----------
    path   The golden_set.jsonl file.

    RETURNS
    -------
    The entries, in file order.

    EXAMPLES
    --------
    >>> load_golden_set(Path("/no/such/golden_set.jsonl"))
    []
    """
    if not path.exists():
        return []

    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def save_golden_set(path: Path, entries: Iterable[dict[str, Any]]) -> int:
    r"""
    PURPOSE
    -------
    Write the golden set atomically (temporary file, then rename).

    PARAMETERS
    ----------
    path      Destination golden_set.jsonl.
    entries   The entries to write.

    RETURNS
    -------
    The number of entries written.

    EXAMPLES
    --------
    >>> import tempfile
    >>> with tempfile.TemporaryDirectory() as folder:
    ...     path = Path(folder) / "eval" / "golden_set.jsonl"
    ...     written = save_golden_set(path, [make_entry("Q?", "A.", ["C1"], "manual")])
    ...     loaded = load_golden_set(path)
    >>> written, loaded[0]["question"]
    (1, 'Q?')
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    count = 0

    with temporary_path.open("w", encoding="utf-8") as output_file:
        for entry in entries:
            output_file.write(json.dumps(entry, ensure_ascii=False) + "\n")
            count += 1

    temporary_path.replace(path)

    return count


def merge_entries(
    existing: list[dict[str, Any]],
    new: Iterable[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    r"""
    PURPOSE
    -------
    Add new entries, skipping any whose id already exists, so a reviewed
    entry is never overwritten by a fresh draft.

    PARAMETERS
    ----------
    existing   The current entries.
    new        Candidate entries to add.

    RETURNS
    -------
    The merged list and how many entries were added.

    EXAMPLES
    --------
    >>> kept = make_entry("Q?", "A.", ["C1"], "manual", status="verified")
    >>> merged, added = merge_entries([kept], [make_entry("Q?", "B.", ["C1"], "template"),
    ...                                        make_entry("R?", "C.", ["C2"], "template")])
    >>> added, [e["status"] for e in merged]
    (1, ['verified', 'draft'])
    """
    merged = list(existing)
    known = {entry["id"] for entry in merged}
    added = 0

    for entry in new:
        if entry["id"] not in known:
            merged.append(entry)
            known.add(entry["id"])
            added += 1

    return merged, added


def load_chunks(chunked_folder: Path = CHUNKED_FOLDER) -> list[dict[str, Any]]:
    r"""
    PURPOSE
    -------
    Read the RAG chunks written by data_chunking.py.

    PARAMETERS
    ----------
    chunked_folder   Folder holding rag_data.jsonl.

    RETURNS
    -------
    The chunk records; empty when the file is missing.

    EXAMPLES
    --------
    >>> load_chunks(Path("/no/such/folder"))
    []
    """
    path = chunked_folder / RAG_FILE_NAME

    if not path.exists():
        return []

    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def sample_chunks(
    chunks: list[dict[str, Any]],
    count: int,
    exclude_chunk_ids: set[str] | None = None,
    seed: int = 13,
    min_words: int = MIN_CHUNK_WORDS,
) -> list[dict[str, Any]]:
    r"""
    PURPOSE
    -------
    Pick chunks to draft questions from, spread evenly across documents (a
    round-robin over documents, each shuffled with a fixed seed), so a long
    book cannot crowd out the rest of the corpus. Short chunks and chunks
    already used are skipped.

    PARAMETERS
    ----------
    chunks              Chunk records from load_chunks.
    count               How many to pick.
    exclude_chunk_ids   Chunk ids already in the golden set.
    seed                Random seed, for reproducible samples.
    min_words           Skip chunks shorter than this.

    RETURNS
    -------
    Up to `count` chunk records.

    EXAMPLES
    --------
    >>> body = "word " * 80
    >>> chunks = ([{"chunk_id": f"A{n}", "document_id": "A", "text": body} for n in range(5)]
    ...           + [{"chunk_id": "B0", "document_id": "B", "text": body},
    ...              {"chunk_id": "B1", "document_id": "B", "text": "too short"}])
    >>> picked = sample_chunks(chunks, 3)
    >>> sorted({c["document_id"] for c in picked}), len(picked)
    (['A', 'B'], 3)
    >>> [c["chunk_id"] for c in sample_chunks(chunks, 10, exclude_chunk_ids={"B0"})
    ...  if c["document_id"] == "B"]
    []
    """
    exclude_chunk_ids = exclude_chunk_ids or set()
    by_document: dict[str, list[dict[str, Any]]] = {}

    for chunk in chunks:
        if (
            chunk.get("chunk_id") in exclude_chunk_ids
            or len(str(chunk.get("text") or "").split()) < min_words
        ):
            continue

        by_document.setdefault(str(chunk.get("document_id") or ""), []).append(chunk)

    generator = random.Random(seed)

    for document_chunks in by_document.values():
        generator.shuffle(document_chunks)

    picked: list[dict[str, Any]] = []
    queues = [by_document[key] for key in sorted(by_document)]

    while len(picked) < count and any(queues):
        for queue in queues:
            if queue and len(picked) < count:
                picked.append(queue.pop())

    return picked


def leading_words(text: str, limit: int = REFERENCE_ANSWER_WORDS) -> str:
    r"""
    PURPOSE
    -------
    Whole leading sentences of a text, up to about `limit` words.

    PARAMETERS
    ----------
    text    The text.
    limit   Word budget.

    RETURNS
    -------
    The leading sentences (at least the first one, cut at `limit` words).

    EXAMPLES
    --------
    >>> leading_words("One two. Three four five. Six.", 4)
    'One two.'
    >>> leading_words("a b c d e f", 3)
    'a b c'
    """
    sentences = re.split(r"(?<=[.!?])\s+", normalize_text(text))
    kept: list[str] = []
    used = 0

    for sentence in sentences:
        words = len(sentence.split())

        if kept and used + words > limit:
            break

        kept.append(sentence if words <= limit else " ".join(sentence.split()[:limit]))
        used += words

    return " ".join(kept)


def draft_from_template(chunk: dict[str, Any]) -> dict[str, Any]:
    r"""
    PURPOSE
    -------
    Draft a question from a chunk's own structure, with no model: its
    formula when it has one, otherwise its concept. The reference answer
    is the chunk's opening sentences. Templates are formulaic, so these
    drafts usually need editing in review.

    PARAMETERS
    ----------
    chunk   A rag_data.jsonl record.

    RETURNS
    -------
    A draft entry.

    EXAMPLES
    --------
    >>> chunk = {"chunk_id": "C1", "document_id": "D1", "concept": "Momentum",
    ...          "formula": "p = mv", "text": "Momentum is mass times velocity, p = mv. It is conserved."}
    >>> draft = draft_from_template(chunk)
    >>> draft["question"], draft["origin"], draft["status"]
    ('What does the formula p = mv describe for Momentum?', 'template', 'draft')
    >>> draft_from_template({"chunk_id": "C2", "concept": "Ori", "text": "Ori is destiny."})["question"]
    'What does the source say about Ori?'
    """
    concept = normalize_text(chunk.get("concept") or chunk.get("topic") or "this topic")
    formula = normalize_text(chunk.get("formula") or "")

    question = (
        f"What does the formula {formula} describe for {concept}?"
        if formula
        else f"What does the source say about {concept}?"
    )

    return make_entry(
        question,
        leading_words(str(chunk.get("text") or "")),
        [chunk["chunk_id"]],
        origin="template",
        source_document_id=chunk.get("document_id"),
        source_file=(chunk.get("source_metadata") or {}).get("source_file"),
        language=(chunk.get("source_metadata") or {}).get("language"),
    )


def parse_json_reply(reply: str) -> dict[str, Any] | None:
    r"""
    PURPOSE
    -------
    Pull the first JSON object out of a model reply, tolerating code
    fences and chatter around it.

    PARAMETERS
    ----------
    reply   The model's raw reply.

    RETURNS
    -------
    The parsed object, or None when there is none.

    EXAMPLES
    --------
    >>> parse_json_reply('Sure! ```json\n{"question": "Q?", "answer": "A."}\n```')
    {'question': 'Q?', 'answer': 'A.'}
    >>> parse_json_reply("no json here") is None
    True
    """
    match = re.search(r"\{.*\}", reply or "", re.DOTALL)

    if not match:
        return None

    try:
        value = json.loads(match.group(0))
    except ValueError:
        return None

    return value if isinstance(value, dict) else None


def draft_from_generator(chunk: dict[str, Any], generator: TextGenerator) -> dict[str, Any]:
    r"""
    PURPOSE
    -------
    Ask a text generator to draft a natural question and answer from a
    chunk. Falls back to the template draft when the generator fails or
    does not return usable JSON, so one bad reply never stops a batch.

    PARAMETERS
    ----------
    chunk       A rag_data.jsonl record.
    generator   Any `prompt -> text` callable.

    RETURNS
    -------
    A draft entry (origin "generator", or "template" on fallback).

    EXAMPLES
    --------
    >>> chunk = {"chunk_id": "C1", "concept": "Ori", "text": "Ori is the seat of destiny."}
    >>> good = lambda prompt: '{"question": "What is ori?", "answer": "The seat of destiny."}'
    >>> draft = draft_from_generator(chunk, good)
    >>> draft["question"], draft["origin"]
    ('What is ori?', 'generator')
    >>> draft_from_generator(chunk, lambda prompt: "I can't")["origin"]
    'template'
    """
    try:
        reply = parse_json_reply(generator(DRAFT_PROMPT.format(text=chunk.get("text") or "")))
    except Exception:  # a model failure falls back to the template
        reply = None

    if not reply or not str(reply.get("question") or "").strip():
        return draft_from_template(chunk)

    return make_entry(
        str(reply["question"]),
        str(reply.get("answer") or ""),
        [chunk["chunk_id"]],
        origin="generator",
        source_document_id=chunk.get("document_id"),
        source_file=(chunk.get("source_metadata") or {}).get("source_file"),
        language=(chunk.get("source_metadata") or {}).get("language"),
    )


def draft_entries(
    entries: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
    count: int,
    generator: TextGenerator | None = None,
    seed: int = 13,
) -> tuple[list[dict[str, Any]], int]:
    r"""
    PURPOSE
    -------
    Draft `count` new entries from chunks not yet in the golden set.

    PARAMETERS
    ----------
    entries     The current golden set.
    chunks      Chunk records from load_chunks.
    count       How many drafts to add.
    generator   Optional `prompt -> text` callable; templates when None.
    seed        Sampling seed.

    RETURNS
    -------
    The updated golden set and the number of drafts added.

    EXAMPLES
    --------
    >>> body = "Momentum is mass times velocity. " * 20
    >>> chunks = [{"chunk_id": f"C{n}", "document_id": "D", "concept": f"Idea {n}", "text": body}
    ...           for n in range(4)]
    >>> entries, added = draft_entries([], chunks, 3)
    >>> added, len(entries)
    (3, 3)
    >>> entries, added = draft_entries(entries, chunks, 3)
    >>> added, len(entries)
    (1, 4)
    """
    used = {chunk_id for entry in entries for chunk_id in entry["source_chunk_ids"]}

    drafts = [
        draft_from_generator(chunk, generator) if generator else draft_from_template(chunk)
        for chunk in sample_chunks(chunks, count, exclude_chunk_ids=used, seed=seed)
    ]

    return merge_entries(entries, drafts)


def import_feedback(
    entries: list[dict[str, Any]],
    interactions: Iterable[dict[str, Any]],
    max_sources: int = 3,
) -> tuple[list[dict[str, Any]], int]:
    r"""
    PURPOSE
    -------
    Turn real user questions that got a thumbs up into draft entries. The
    answer the user liked becomes the draft reference answer, and the
    passages it cited (or, when there are no citations, the top retrieved
    ones) become the source chunks. Review still decides what is golden.

    PARAMETERS
    ----------
    entries        The current golden set.
    interactions   Interaction dicts from feedback_logger
                    (FeedbackLogger.iter_interactions(rating=1)).
    max_sources    Most retrieved passages to take when nothing was cited.

    RETURNS
    -------
    The updated golden set and the number of drafts added.

    EXAMPLES
    --------
    >>> interaction = {"interaction_id": "INT-1", "question": "What is ori?",
    ...                "standalone_question": None, "answer": "Destiny [2].", "rating": 1,
    ...                "language": "eng_Latn",
    ...                "retrieved": [{"chunk_id": "C1"}, {"chunk_id": "C2"}],
    ...                "citations": {"references": [{"number": 2, "chunk_id": "C2"}]}}
    >>> entries, added = import_feedback([], [interaction])
    >>> added, entries[0]["source_chunk_ids"], entries[0]["origin"], entries[0]["reference_answer"]
    (1, ['C2'], 'feedback', 'Destiny.')
    >>> import_feedback([], [dict(interaction, rating=-1)])[1]
    0
    """
    drafts: list[dict[str, Any]] = []

    for interaction in interactions:
        if interaction.get("rating") != 1:
            continue

        references = (interaction.get("citations") or {}).get("references") or []
        chunk_ids = [ref.get("chunk_id") for ref in references if ref.get("chunk_id")]

        if not chunk_ids:
            chunk_ids = [
                item.get("chunk_id") or item.get("record_id")
                for item in (interaction.get("retrieved") or [])[:max_sources]
            ]

        chunk_ids = [chunk_id for chunk_id in chunk_ids if chunk_id]

        if not chunk_ids or not str(interaction.get("question") or "").strip():
            continue

        # The reference answer should not carry this run's citation numbers.
        answer = re.sub(r"\s*\[\d+(?:\s*[-–,]\s*\d+)*\]", "", str(interaction.get("answer") or ""))

        drafts.append(
            make_entry(
                interaction.get("standalone_question") or interaction["question"],
                answer,
                chunk_ids,
                origin="feedback",
                language=interaction.get("language"),
                interaction_id=interaction.get("interaction_id"),
            )
        )

    return merge_entries(entries, drafts)


def review_entries(
    entries: list[dict[str, Any]],
    chunks_by_id: dict[str, dict[str, Any]],
    input_function: Callable[[str], str] = input,
    output_function: Callable[[str], None] = print,
) -> Counter:
    r"""
    PURPOSE
    -------
    Walk through draft entries one at a time, showing each question,
    reference answer and source passage, and record the reviewer's
    decision in place:

        a  accept as is (verified)
        e  edit the question and/or answer, then accept
        r  reject
        s  skip for now (stays a draft)
        q  stop reviewing

    PARAMETERS
    ----------
    entries          The golden set; drafts are changed in place.
    chunks_by_id     chunk_id -> chunk record, to show the source passage.
    input_function   Reads the reviewer's replies (injectable for tests).
    output_function  Shows text to the reviewer.

    RETURNS
    -------
    A Counter of decisions made ("verified", "rejected", "skipped").

    EXAMPLES
    --------
    >>> entries = [make_entry("Q1?", "A1.", ["C1"], "template"),
    ...            make_entry("Q2?", "A2.", ["C1"], "template"),
    ...            make_entry("Q3?", "A3.", ["C1"], "template")]
    >>> replies = iter(["a", "e", "Better Q2?", "", "q"])
    >>> counts = review_entries(entries, {"C1": {"text": "Passage."}},
    ...                         input_function=lambda prompt: next(replies),
    ...                         output_function=lambda text: None)
    >>> [(e["question"], e["status"]) for e in entries]
    [('Q1?', 'verified'), ('Better Q2?', 'verified'), ('Q3?', 'draft')]
    >>> dict(counts)
    {'verified': 2}
    """
    counts: Counter = Counter()
    drafts = [entry for entry in entries if entry["status"] == "draft"]

    for position, entry in enumerate(drafts, start=1):
        passages = "\n\n".join(
            str((chunks_by_id.get(chunk_id) or {}).get("text") or f"(chunk {chunk_id} not found)")
            for chunk_id in entry["source_chunk_ids"]
        )
        output_function(
            f"\n--- Draft {position}/{len(drafts)} ({entry['origin']}) ---\n"
            f"SOURCE:\n{passages}\n\n"
            f"QUESTION: {entry['question']}\n"
            f"ANSWER:   {entry['reference_answer']}"
        )

        try:
            choice = input_function("[a]ccept [e]dit [r]eject [s]kip [q]uit: ").strip().casefold()
        except (EOFError, KeyboardInterrupt, StopIteration):
            break

        if choice == "q":
            break

        if choice == "s" or choice not in {"a", "e", "r"}:
            counts["skipped"] += 1
            continue

        if choice == "r":
            entry["status"] = "rejected"
            counts["rejected"] += 1
            continue

        if choice == "e":
            new_question = input_function("New question (blank keeps it): ").strip()
            new_answer = input_function("New answer (blank keeps it): ").strip()

            if new_question:
                entry["question"] = normalize_text(new_question)

            if new_answer:
                entry["reference_answer"] = normalize_text(new_answer)

        entry["status"] = "verified"
        entry["verified_at"] = utc_now()
        counts["verified"] += 1

    return counts


def validate_entries(
    entries: list[dict[str, Any]],
    chunk_ids: set[str],
) -> list[dict[str, Any]]:
    r"""
    PURPOSE
    -------
    Find entries that cite chunks no longer in the corpus (re-chunking
    changes ids), so they can be re-reviewed.

    PARAMETERS
    ----------
    entries     The golden set.
    chunk_ids   Every chunk id currently in rag_data.jsonl.

    RETURNS
    -------
    One {"id", "question", "missing_chunk_ids"} dict per stale entry,
    rejected entries excluded.

    EXAMPLES
    --------
    >>> entries = [make_entry("Q1?", "A.", ["C1", "C9"], "manual", status="verified"),
    ...            make_entry("Q2?", "A.", ["C2"], "manual", status="verified")]
    >>> [problem["missing_chunk_ids"] for problem in validate_entries(entries, {"C1", "C2"})]
    [['C9']]
    """
    problems: list[dict[str, Any]] = []

    for entry in entries:
        if entry["status"] == "rejected":
            continue

        missing = [chunk_id for chunk_id in entry["source_chunk_ids"] if chunk_id not in chunk_ids]

        if missing:
            problems.append(
                {"id": entry["id"], "question": entry["question"], "missing_chunk_ids": missing}
            )

    return problems


def golden_set_stats(entries: list[dict[str, Any]]) -> dict[str, Any]:
    r"""
    PURPOSE
    -------
    Counts by status, origin and language, and how far the verified set is
    from the 200-500 target.

    PARAMETERS
    ----------
    entries   The golden set.

    RETURNS
    -------
    A stats dict.

    EXAMPLES
    --------
    >>> stats = golden_set_stats([make_entry("Q?", "A.", ["C1"], "manual", status="verified",
    ...                                      language="yor_Latn")])
    >>> stats["by_status"], stats["verified_needed"], stats["by_language"]
    ({'verified': 1}, 199, {'yor_Latn': 1})
    """
    verified = [entry for entry in entries if entry["status"] == "verified"]

    return {
        "total": len(entries),
        "by_status": dict(Counter(entry["status"] for entry in entries)),
        "by_origin": dict(Counter(entry["origin"] for entry in entries)),
        "by_language": dict(Counter(entry.get("language") or "unknown" for entry in verified)),
        "verified_needed": max(0, TARGET_MINIMUM - len(verified)),
        "over_maximum": len(verified) > TARGET_MAXIMUM,
    }


def load_text_generator(spec: str) -> TextGenerator:
    r"""
    PURPOSE
    -------
    Load your own generator from "path/to/file.py:function_name". The
    function must take a prompt string and return the reply text. This is
    how an existing generator (for example the one behind the website demo)
    is plugged in without this module depending on it.

    PARAMETERS
    ----------
    spec   "file.py:function".

    RETURNS
    -------
    The callable.

    RAISES
    ------
    ValueError   When the spec is malformed or the function is missing.

    EXAMPLES
    --------
    >>> import tempfile
    >>> with tempfile.TemporaryDirectory() as folder:
    ...     path = Path(folder) / "my_generator.py"
    ...     _ = path.write_text("def ask(prompt):\n    return 'reply to ' + prompt\n")
    ...     generate = load_text_generator(f"{path}:ask")
    ...     generate("hi")
    'reply to hi'
    >>> load_text_generator("no_colon")
    Traceback (most recent call last):
        ...
    ValueError: Use the form path/to/file.py:function_name, not 'no_colon'.
    """
    file_part, separator, function_name = spec.rpartition(":")

    if not separator or not file_part or not function_name:
        raise ValueError(f"Use the form path/to/file.py:function_name, not {spec!r}.")

    path = Path(file_part).expanduser().resolve()
    module_spec = importlib.util.spec_from_file_location(f"user_generator_{path.stem}", path)

    if module_spec is None or module_spec.loader is None:
        raise ValueError(f"Cannot load a Python module from {path}.")

    module = importlib.util.module_from_spec(module_spec)
    sys.path.insert(0, str(path.parent))
    module_spec.loader.exec_module(module)

    function = getattr(module, function_name, None)

    if not callable(function):
        raise ValueError(f"{path} has no function named {function_name!r}.")

    return function


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
    >>> options = parse_arguments(["draft", "--count", "25"])
    >>> options.command, options.count, options.generator
    ('draft', 25, None)
    >>> options = parse_arguments(["add", "--question", "Q?", "--answer", "A.",
    ...                            "--chunk-id", "C1", "--chunk-id", "C2"])
    >>> options.chunk_id
    ['C1', 'C2']
    """
    parser = argparse.ArgumentParser(description="Build and maintain the golden evaluation set.")
    parser.add_argument("--golden-set", type=Path, default=GOLDEN_SET_FILE, help="golden_set.jsonl")
    parser.add_argument("--chunks", type=Path, default=CHUNKED_FOLDER, help="chunked_data folder")

    commands = parser.add_subparsers(dest="command", required=True)

    draft = commands.add_parser("draft", help="draft entries from sampled chunks")
    draft.add_argument("--count", type=int, default=50, help="how many drafts to add")
    draft.add_argument("--generator", help="your generator as file.py:function (prompt -> text)")
    draft.add_argument("--seed", type=int, default=13, help="sampling seed")

    feedback = commands.add_parser("import-feedback", help="draft entries from thumbs-up answers")
    feedback.add_argument("--database", type=Path, help="feedback.db (default: feedback_logger's)")

    commands.add_parser("review", help="accept, edit or reject drafts")

    add = commands.add_parser("add", help="add a verified entry you wrote")
    add.add_argument("--question", required=True)
    add.add_argument("--answer", required=True)
    add.add_argument("--chunk-id", action="append", required=True, help="repeat for several")
    add.add_argument("--language", help="GlotLID label, e.g. yor_Latn")

    commands.add_parser("validate", help="list entries whose chunks no longer exist")
    commands.add_parser("stats", help="counts and progress toward 200-500 verified")

    return parser.parse_args(argument_list)


def main() -> None:
    """
    PURPOSE
    -------
    Run the subcommand given on the command line.
    """
    options = parse_arguments()
    entries = load_golden_set(options.golden_set)

    if options.command == "draft":
        chunks = load_chunks(options.chunks)
        generator = load_text_generator(options.generator) if options.generator else None
        entries, added = draft_entries(entries, chunks, options.count, generator, options.seed)
        save_golden_set(options.golden_set, entries)
        print(f"Added {added} draft(s). Review them with: eval_dataset_builder.py review")

    elif options.command == "import-feedback":
        from feedback_logger import DATABASE_FILE, FeedbackLogger

        logger = FeedbackLogger(options.database or DATABASE_FILE)
        entries, added = import_feedback(entries, logger.iter_interactions(rating=1))
        save_golden_set(options.golden_set, entries)
        print(f"Added {added} draft(s) from thumbs-up answers.")

    elif options.command == "review":
        chunks_by_id = {chunk["chunk_id"]: chunk for chunk in load_chunks(options.chunks)}
        try:
            counts = review_entries(entries, chunks_by_id)
        finally:
            save_golden_set(options.golden_set, entries)
        print(f"Review saved: {dict(counts)}")

    elif options.command == "add":
        entries, added = merge_entries(
            entries,
            [make_entry(options.question, options.answer, options.chunk_id, "manual",
                        status="verified", verified_at=utc_now(), language=options.language)],
        )
        save_golden_set(options.golden_set, entries)
        print("Added." if added else "An identical entry already exists.")

    elif options.command == "validate":
        chunk_ids = {chunk["chunk_id"] for chunk in load_chunks(options.chunks)}
        problems = validate_entries(entries, chunk_ids)
        print(json.dumps(problems, ensure_ascii=False, indent=2) if problems else "All entries are valid.")

    else:
        print(json.dumps(golden_set_stats(entries), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
