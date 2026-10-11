#!/usr/bin/env python3

"""
SoakinGarri RAG Evaluation Program

PURPOSE
-------
Scores the question-answering system against the verified golden set
(eval_dataset_builder.py). Run it after every change — a new chunk size, a
different embedding model, a prompt edit — and compare with the last run.

RETRIEVAL METRICS  (free, deterministic, always computed)
-----------------
    hit@k       share of questions with at least one gold passage in the
                top k
    recall@k    share of each question's gold passages found in the top k,
                averaged over questions
    MRR         mean reciprocal rank of the first gold passage (0 when
                none is retrieved)

ANSWER METRICS  (computed when an answer function is given)
--------------
    citation_validity    share of answers citing no invented source number
    citation_precision   share of cited passages that are gold passages
    refusal_rate         share of answers saying the sources lack the answer
    faithfulness         share of answers a judge model finds fully
                         supported by the passages (only with a judge)

The answer and judge functions are plain callables, so your own generator
plugs in without this module depending on it. RAGAS is not required; its
metrics need an LLM on every run, while the scores above cost nothing
except faithfulness.

OUTPUT
------
<repo>/data/ask/evaluation/runs/eval_<UTC timestamp>.json
Each run is compared with the previous one, and the change per metric is
printed. --fail-on-drop makes the program exit with an error when any
metric falls by more than the given amount, so it can gate a change.

RUN
---
python3 rag_evaluation.py                         (retrieval only, hybrid)
python3 rag_evaluation.py --mode vector
python3 rag_evaluation.py --rerank
python3 rag_evaluation.py --answer-with path/to/generator.py:function
python3 rag_evaluation.py --fail-on-drop 0.02
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Callable

HERE = Path(__file__).resolve().parent

for folder in (
    HERE.parent.parent.parent / "source",
    HERE.parent / "retrieval",
    HERE.parent / "vector_database",
    HERE.parent / "keyword_index",
    HERE.parent / "reranker",
    HERE.parent / "prompt_builder",
    HERE.parent / "citation_formatter",
):
    sys.path.insert(0, str(folder))

from common import DATA_ROOT  # noqa: E402

import eval_dataset_builder as edb  # noqa: E402

RUNS_FOLDER = DATA_ROOT / "evaluation" / "runs"

DEFAULT_K_VALUES = (1, 3, 5, 10)
DEFAULT_TOP_K = 10

Retriever = Callable[[str], list[Any]]
Answerer = Callable[[str, list[Any]], tuple[str, list[Any]]]
Judge = Callable[[str], str]

JUDGE_PROMPT = """You check whether an answer is supported by its sources.

Sources:
{sources}

Answer:
{answer}

Is every factual claim in the answer supported by the sources? Reply with one word: YES or NO."""


def result_ids(result: Any) -> set[str]:
    r"""
    PURPOSE
    -------
    The ids a retrieved result can match a gold passage by: its chunk id
    and, for LLM-dataset records, its record id.

    PARAMETERS
    ----------
    result   A SearchResult, RerankedResult, Passage, or record dict.

    RETURNS
    -------
    The set of ids.

    EXAMPLES
    --------
    >>> from types import SimpleNamespace as R
    >>> sorted(result_ids(R(record={"chunk_id": "C1", "record_id": "QA-1"})))
    ['C1', 'QA-1']
    >>> result_ids({"chunk_id": "C2"})
    {'C2'}
    """
    record = result if isinstance(result, dict) else (getattr(result, "record", {}) or {})

    return {str(record[key]) for key in ("chunk_id", "record_id") if record.get(key)}


def first_relevant_rank(results: list[Any], gold: set[str]) -> int | None:
    r"""
    PURPOSE
    -------
    The 1-based rank of the first result matching a gold passage.

    PARAMETERS
    ----------
    results   Retrieved results, best first.
    gold      The gold chunk ids.

    RETURNS
    -------
    The rank, or None when no result matches.

    EXAMPLES
    --------
    >>> first_relevant_rank([{"chunk_id": "A"}, {"chunk_id": "B"}], {"B"})
    2
    >>> first_relevant_rank([{"chunk_id": "A"}], {"B"}) is None
    True
    """
    for rank, result in enumerate(results, start=1):
        if result_ids(result) & gold:
            return rank

    return None


def retrieval_scores(
    results: list[Any],
    gold: set[str],
    k_values: tuple[int, ...] = DEFAULT_K_VALUES,
) -> dict[str, float]:
    r"""
    PURPOSE
    -------
    hit@k, recall@k and reciprocal rank for one question.

    PARAMETERS
    ----------
    results    Retrieved results, best first.
    gold       The gold chunk ids for the question.
    k_values   Cut-offs to score at.

    RETURNS
    -------
    {"hit@1": ..., "recall@1": ..., ..., "reciprocal_rank": ...}.

    EXAMPLES
    --------
    >>> results = [{"chunk_id": "X"}, {"chunk_id": "A"}, {"chunk_id": "Y"}, {"chunk_id": "B"}]
    >>> scores = retrieval_scores(results, {"A", "B"}, k_values=(1, 2, 4))
    >>> scores["hit@1"], scores["hit@2"], scores["recall@2"], scores["recall@4"], scores["reciprocal_rank"]
    (0.0, 1.0, 0.5, 1.0, 0.5)
    """
    scores: dict[str, float] = {}

    for k in k_values:
        found: set[str] = set()

        for result in results[:k]:
            found |= result_ids(result) & gold

        scores[f"hit@{k}"] = 1.0 if found else 0.0
        scores[f"recall@{k}"] = len(found) / len(gold) if gold else 0.0

    rank = first_relevant_rank(results, gold)
    scores["reciprocal_rank"] = 1.0 / rank if rank else 0.0

    return scores


def judge_faithfulness(answer: str, passages: list[Any], judge: Judge) -> float | None:
    r"""
    PURPOSE
    -------
    Ask a judge model whether every claim in an answer is supported by
    the passages it was given.

    PARAMETERS
    ----------
    answer     The generated answer.
    passages   The passages the answer was generated from (`.text`).
    judge      Any `prompt -> text` callable.

    RETURNS
    -------
    1.0 for YES, 0.0 for NO, None when the judge fails or is unclear.

    EXAMPLES
    --------
    >>> from types import SimpleNamespace as P
    >>> judge_faithfulness("Mass times velocity [1].", [P(number=1, text="p = mv")], lambda p: "YES.")
    1.0
    >>> judge_faithfulness("x", [], lambda p: "no")
    0.0
    >>> judge_faithfulness("x", [], lambda p: "maybe") is None
    True
    """
    sources = "\n\n".join(
        f"[{getattr(passage, 'number', position)}] {getattr(passage, 'text', '')}"
        for position, passage in enumerate(passages, start=1)
    )

    try:
        verdict = judge(JUDGE_PROMPT.format(sources=sources or "(none)", answer=answer))
    except Exception:
        return None

    word = re.match(r"\W*(\w+)", str(verdict or ""))

    if not word:
        return None

    return {"yes": 1.0, "no": 0.0}.get(word.group(1).casefold())


def answer_scores(
    answer: str,
    passages: list[Any],
    gold: set[str],
    judge: Judge | None = None,
) -> dict[str, float | None]:
    r"""
    PURPOSE
    -------
    Citation and faithfulness scores for one answer.

    PARAMETERS
    ----------
    answer     The generated answer, with [n] citations.
    passages   The numbered passages it was generated from (`.number`,
                `.record`, `.text`).
    gold       The question's gold chunk ids.
    judge      Optional `prompt -> text` judge for faithfulness.

    RETURNS
    -------
    {"citation_valid", "citation_precision", "refused", "faithfulness"};
    citation_precision is None when nothing was cited, faithfulness is
    None without a judge.

    EXAMPLES
    --------
    >>> from types import SimpleNamespace as P
    >>> passages = [P(number=1, record={"chunk_id": "A"}, text="..."),
    ...             P(number=2, record={"chunk_id": "B"}, text="...")]
    >>> answer_scores("Claim [1]. Other [2]. Made up [7].", passages, {"A"})
    {'citation_valid': 0.0, 'citation_precision': 0.5, 'refused': 0.0, 'faithfulness': None}
    >>> answer_scores("The provided sources do not contain the answer.", passages, {"A"})["refused"]
    1.0
    """
    from citation_formatter import check_citations

    report = check_citations(answer, passages)
    by_number = {passage.number: passage for passage in passages}

    precision = (
        mean(1.0 if result_ids(by_number[number]) & gold else 0.0 for number in report.cited)
        if report.cited
        else None
    )

    return {
        "citation_valid": 0.0 if report.invalid else 1.0,
        "citation_precision": precision,
        "refused": 1.0 if report.says_not_in_sources else 0.0,
        "faithfulness": judge_faithfulness(answer, passages, judge) if judge else None,
    }


def average(rows: list[dict[str, Any]], key: str) -> float | None:
    r"""
    PURPOSE
    -------
    Mean of one metric over the rows that have it.

    PARAMETERS
    ----------
    rows   Per-question score dicts.
    key    The metric name.

    RETURNS
    -------
    The mean rounded to 4 places, or None when no row has the metric.

    EXAMPLES
    --------
    >>> average([{"m": 1.0}, {"m": 0.0}, {"m": None}, {}], "m")
    0.5
    >>> average([{}], "m") is None
    True
    """
    values = [row[key] for row in rows if row.get(key) is not None]

    return round(mean(values), 4) if values else None


def evaluate(
    entries: list[dict[str, Any]],
    retriever: Retriever,
    answerer: Answerer | None = None,
    judge: Judge | None = None,
    k_values: tuple[int, ...] = DEFAULT_K_VALUES,
) -> dict[str, Any]:
    r"""
    PURPOSE
    -------
    Score every verified golden entry.

    PARAMETERS
    ----------
    entries     The golden set; only "verified" entries are scored.
    retriever   `question -> ranked results`.
    answerer    Optional `(question, results) -> (answer, passages)`;
                 `passages` are the numbered passages the answer cites
                 (prompt_builder's PromptPackage.passages).
    judge       Optional `prompt -> text` judge for faithfulness.
    k_values    Retrieval cut-offs.

    RETURNS
    -------
    {"summary": {metric: mean, ...}, "questions": [per-question rows]}.

    EXAMPLES
    --------
    >>> entries = [
    ...     {"id": "G1", "question": "q1", "source_chunk_ids": ["A"], "status": "verified"},
    ...     {"id": "G2", "question": "q2", "source_chunk_ids": ["B"], "status": "verified"},
    ...     {"id": "G3", "question": "q3", "source_chunk_ids": ["C"], "status": "draft"},
    ... ]
    >>> ranked = {"q1": [{"chunk_id": "A"}], "q2": [{"chunk_id": "X"}, {"chunk_id": "B"}]}
    >>> report = evaluate(entries, ranked.get, k_values=(1, 2))
    >>> report["summary"]["questions"], report["summary"]["hit@1"], report["summary"]["mrr"]
    (2, 0.5, 0.75)
    """
    rows: list[dict[str, Any]] = []

    for entry in entries:
        if entry.get("status") != "verified":
            continue

        gold = {str(chunk_id) for chunk_id in entry["source_chunk_ids"]}
        started = time.perf_counter()
        results = retriever(entry["question"]) or []
        row: dict[str, Any] = {
            "id": entry["id"],
            "question": entry["question"],
            "language": entry.get("language"),
            "first_relevant_rank": first_relevant_rank(results, gold),
            **retrieval_scores(results, gold, k_values),
        }

        if answerer is not None:
            answer, passages = answerer(entry["question"], results)
            row["answer"] = answer
            row.update(answer_scores(answer, passages, gold, judge))

        row["seconds"] = round(time.perf_counter() - started, 3)
        rows.append(row)

    summary: dict[str, Any] = {"questions": len(rows)}

    for k in k_values:
        summary[f"hit@{k}"] = average(rows, f"hit@{k}")
        summary[f"recall@{k}"] = average(rows, f"recall@{k}")

    summary["mrr"] = average(rows, "reciprocal_rank")

    if answerer is not None:
        summary["citation_validity"] = average(rows, "citation_valid")
        summary["citation_precision"] = average(rows, "citation_precision")
        summary["refusal_rate"] = average(rows, "refused")
        summary["faithfulness"] = average(rows, "faithfulness")

    summary["mean_seconds"] = average(rows, "seconds")

    return {"summary": summary, "questions": rows}


# Metrics where a lower value is better; every other metric is higher-is-better.
LOWER_IS_BETTER = {"refusal_rate", "mean_seconds"}


def compare_runs(
    current: dict[str, Any],
    previous: dict[str, Any] | None,
) -> dict[str, dict[str, float]]:
    r"""
    PURPOSE
    -------
    The change in each summary metric since the previous run.

    PARAMETERS
    ----------
    current    This run's summary.
    previous   The previous run's summary, or None.

    RETURNS
    -------
    metric -> {"previous", "current", "change", "worse"} for every numeric
    metric both runs have ("worse" accounts for lower-is-better metrics).

    EXAMPLES
    --------
    >>> changes = compare_runs({"mrr": 0.6, "refusal_rate": 0.3}, {"mrr": 0.7, "refusal_rate": 0.1})
    >>> changes["mrr"]["change"], changes["mrr"]["worse"], changes["refusal_rate"]["worse"]
    (-0.1, 0.1, 0.2)
    >>> compare_runs({"mrr": 0.6}, None)
    {}
    """
    if not previous:
        return {}

    changes: dict[str, dict[str, float]] = {}

    for metric, value in current.items():
        old = previous.get(metric)

        if metric == "questions" or not isinstance(value, (int, float)) or not isinstance(old, (int, float)):
            continue

        change = round(value - old, 4)
        worse = change if metric in LOWER_IS_BETTER else -change

        changes[metric] = {
            "previous": old,
            "current": value,
            "change": change,
            "worse": round(max(0.0, worse), 4),
        }

    return changes


def latest_run(runs_folder: Path = RUNS_FOLDER) -> dict[str, Any] | None:
    r"""
    PURPOSE
    -------
    The most recent saved run, or None.

    PARAMETERS
    ----------
    runs_folder   Folder of eval_*.json files.

    RETURNS
    -------
    The run's JSON.

    EXAMPLES
    --------
    >>> latest_run(Path("/no/such/folder")) is None
    True
    """
    runs = sorted(runs_folder.glob("eval_*.json")) if runs_folder.is_dir() else []

    if not runs:
        return None

    return json.loads(runs[-1].read_text(encoding="utf-8"))


def save_run(report: dict[str, Any], runs_folder: Path = RUNS_FOLDER) -> Path:
    r"""
    PURPOSE
    -------
    Save a run atomically as eval_<UTC timestamp>.json. Timestamps sort in
    time order, so the newest run is always last.

    PARAMETERS
    ----------
    report        The run report.
    runs_folder   Destination folder.

    RETURNS
    -------
    The saved file's path.

    EXAMPLES
    --------
    >>> import tempfile
    >>> with tempfile.TemporaryDirectory() as folder:
    ...     first = save_run({"summary": {"mrr": 0.5}}, Path(folder))
    ...     second = save_run({"summary": {"mrr": 0.6}}, Path(folder))
    ...     latest = latest_run(Path(folder))
    >>> latest["summary"]["mrr"], first != second
    (0.6, True)
    """
    runs_folder.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = runs_folder / f"eval_{stamp}.json"
    temporary_path = path.with_suffix(".json.tmp")
    temporary_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary_path.replace(path)

    return path


def format_summary(summary: dict[str, Any], changes: dict[str, dict[str, float]]) -> str:
    r"""
    PURPOSE
    -------
    A terminal table of the summary and its change since the last run.

    PARAMETERS
    ----------
    summary   This run's summary.
    changes   Output of compare_runs.

    RETURNS
    -------
    The table text.

    EXAMPLES
    --------
    >>> print(format_summary({"questions": 2, "mrr": 0.75, "faithfulness": None},
    ...                      {"mrr": {"previous": 0.7, "current": 0.75, "change": 0.05, "worse": 0.0}}))
    questions              2
    mrr               0.7500   (+0.0500)
    faithfulness           -
    """
    lines = []

    for metric, value in summary.items():
        if value is None:
            shown = "-"
        elif metric == "questions":
            shown = str(value)
        else:
            shown = f"{value:.4f}"

        change = changes.get(metric)
        suffix = f"   ({change['change']:+.4f})" if change else ""
        lines.append(f"{metric:<16}{shown:>8}{suffix}")

    return "\n".join(lines)


def build_answerer(generate_messages: Callable[[list[dict[str, str]]], str]) -> Answerer:
    r"""
    PURPOSE
    -------
    Wrap a chat generator into an answerer: build the grounded prompt with
    prompt_builder, call the generator, return its answer and the numbered
    passages. This is the same prompt the live system uses.

    PARAMETERS
    ----------
    generate_messages   Your generator: `messages -> answer text`, where
                         messages are [{"role", "content"}] dicts.

    RETURNS
    -------
    An Answerer for evaluate().

    EXAMPLES
    --------
    >>> from vector_base import SearchResult
    >>> answer, passages = build_answerer(lambda messages: "Mass times velocity [1].")(
    ...     "What is momentum?", [SearchResult(1, 0.9, 0, "p = mv", {"chunk_id": "A"})])
    >>> answer, [p.number for p in passages]
    ('Mass times velocity [1].', [1])
    """
    from prompt_builder import build_prompt

    def answer(question: str, results: list[Any]) -> tuple[str, list[Any]]:
        package = build_prompt(question, results)
        return str(generate_messages(package.messages) or ""), package.passages

    return answer


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
    >>> options = parse_arguments([])
    >>> options.mode, options.top_k, options.answer_with, options.fail_on_drop
    ('hybrid', 10, None, None)
    >>> parse_arguments(["--mode", "vector", "--fail-on-drop", "0.02"]).fail_on_drop
    0.02
    """
    parser = argparse.ArgumentParser(description="Score retrieval and answers on the golden set.")
    parser.add_argument("--golden-set", type=Path, default=edb.GOLDEN_SET_FILE)
    parser.add_argument("--runs", type=Path, default=RUNS_FOLDER, help="folder for saved runs")
    parser.add_argument("--mode", choices=["hybrid", "vector"], default="hybrid")
    parser.add_argument("--rerank", action="store_true", help="rerank with bge-reranker-v2-m3")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="results per question")
    parser.add_argument(
        "--dataset", default="rag", help="dataset to search: rag (default), llm, or all"
    )
    parser.add_argument(
        "--answer-with",
        help="your chat generator as file.py:function (messages -> answer) to score answers",
    )
    parser.add_argument(
        "--judge-with",
        help="a text generator as file.py:function (prompt -> text) to score faithfulness",
    )
    parser.add_argument(
        "--fail-on-drop",
        type=float,
        help="exit with an error if any metric is this much worse than the last run",
    )

    return parser.parse_args(argument_list)


def main() -> None:
    """
    PURPOSE
    -------
    Evaluate the current system on the golden set, save the run, and print
    it with its change since the previous run.
    """
    import retrieval as ret
    from vector_base import VectorBase

    options = parse_arguments()

    entries = edb.load_golden_set(options.golden_set)

    if not any(entry["status"] == "verified" for entry in entries):
        raise SystemExit(
            "No verified golden entries yet. Build them with eval_dataset_builder.py "
            "(draft / import-feedback, then review)."
        )

    base = VectorBase.load(ret.VECTOR_BASE_FOLDER)
    encode, _ = ret.load_query_encoder()
    reranker_model = None
    filters = {} if options.dataset == ret.ALL_DATASETS else {"dataset": options.dataset}

    if options.rerank:
        from reranker import load_reranker

        reranker_model = load_reranker()

    if options.mode == "hybrid" or options.rerank:
        from keyword_index import KeywordIndex

        index = KeywordIndex.load(ret.KEYWORD_INDEX_FOLDER)

        def retriever(question: str) -> list[Any]:
            return ret.hybrid_retrieve(
                base, index, question, encode, top_k=options.top_k, filters=filters,
                reranker_model=reranker_model,
            )
    else:

        def retriever(question: str) -> list[Any]:
            return ret.retrieve(
                base, question, encode, top_k=options.top_k, filters=filters, relevance_weight=1.0
            )

    answerer = (
        build_answerer(edb.load_text_generator(options.answer_with)) if options.answer_with else None
    )
    judge = edb.load_text_generator(options.judge_with) if options.judge_with else None

    previous = latest_run(options.runs)
    report = evaluate(entries, retriever, answerer, judge)
    report["settings"] = {
        "mode": options.mode,
        "rerank": options.rerank,
        "top_k": options.top_k,
        "dataset": options.dataset,
        "answer_with": options.answer_with,
        "judge_with": options.judge_with,
        "embedding_model": base.embedding_model,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
    }
    changes = compare_runs(report["summary"], previous["summary"] if previous else None)
    report["changes_since_previous"] = changes
    path = save_run(report, options.runs)

    print(format_summary(report["summary"], changes))
    print(f"\nSaved: {path}")

    if options.fail_on_drop is not None:
        dropped = {m: c["worse"] for m, c in changes.items() if c["worse"] > options.fail_on_drop}

        if dropped:
            raise SystemExit(f"Metrics got worse by more than {options.fail_on_drop}: {dropped}")


if __name__ == "__main__":
    main()
