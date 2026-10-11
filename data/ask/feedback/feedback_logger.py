#!/usr/bin/env python3

"""
SoakinGarri Feedback Logger Program

PURPOSE
-------
Records every answered question: the question (and its standalone
rewrite), the passages retrieved for it, the answer, its citations, and
the user's thumbs up or down. These logs are the most valuable data the
system produces — real questions, with the passages and answers users
judged good or bad — so they are kept in a form that exports straight
into training and evaluation data.

STORAGE
-------
A single SQLite database file (Python's built-in sqlite3, nothing to
install). Each call opens its own short-lived connection, and the file uses
write-ahead logging, so several API workers can log at the same time.

    interactions   one row per answered question
    feedback       one row per vote; a user may change their vote, and the
                   latest vote counts

OUTPUT
------
<repo>/data/ask/feedback/feedback.db
export:  JSONL, one interaction per line, with its latest rating

RUN
---
python3 feedback_logger.py stats
python3 feedback_logger.py export --output rated.jsonl --rated-only
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import uuid
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

sys.path.insert(
    0, str(Path(__file__).resolve().parent.parent.parent.parent / "source")
)

from common import DATA_ROOT  # noqa: E402

DATABASE_FILE = DATA_ROOT / "feedback" / "feedback.db"

RATINGS = {"up": 1, "down": -1, "+1": 1, "-1": -1, "1": 1, 1: 1, -1: -1}

SCHEMA = """
CREATE TABLE IF NOT EXISTS interactions (
    interaction_id      TEXT PRIMARY KEY,
    created_at          TEXT NOT NULL,
    session_id          TEXT,
    question            TEXT NOT NULL,
    standalone_question TEXT,
    language            TEXT,
    answer              TEXT NOT NULL,
    model               TEXT,
    latency_ms          INTEGER,
    citation_ok         INTEGER,
    retrieved_json      TEXT NOT NULL,
    citations_json      TEXT,
    metadata_json       TEXT
);
CREATE INDEX IF NOT EXISTS interactions_session ON interactions(session_id);
CREATE INDEX IF NOT EXISTS interactions_created ON interactions(created_at);

CREATE TABLE IF NOT EXISTS feedback (
    feedback_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    interaction_id  TEXT NOT NULL REFERENCES interactions(interaction_id),
    rating          INTEGER NOT NULL CHECK (rating IN (-1, 1)),
    comment         TEXT,
    created_at      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS feedback_interaction ON feedback(interaction_id);
"""

# The latest vote per interaction (highest feedback_id wins).
LATEST_FEEDBACK = """
SELECT f.interaction_id, f.rating, f.comment, f.created_at
FROM feedback AS f
JOIN (SELECT interaction_id, MAX(feedback_id) AS latest
      FROM feedback GROUP BY interaction_id) AS l
  ON f.feedback_id = l.latest
"""


def utc_now() -> str:
    """
    PURPOSE
    -------
    The current UTC time as an ISO 8601 string.
    """
    return datetime.now(timezone.utc).isoformat()


def normalize_rating(rating: Any) -> int:
    r"""
    PURPOSE
    -------
    Turn a thumbs value into +1 or -1.

    PARAMETERS
    ----------
    rating   "up"/"down", +1/-1, or True/False.

    RETURNS
    -------
    1 for thumbs up, -1 for thumbs down.

    RAISES
    ------
    ValueError   For anything else.

    EXAMPLES
    --------
    >>> normalize_rating("up"), normalize_rating("DOWN"), normalize_rating(-1), normalize_rating(True)
    (1, -1, -1, 1)
    >>> normalize_rating(0)
    Traceback (most recent call last):
        ...
    ValueError: A rating must be up/down or +1/-1, not 0.
    """
    if isinstance(rating, bool):
        return 1 if rating else -1

    key = rating.strip().casefold() if isinstance(rating, str) else rating

    if key in RATINGS:
        return RATINGS[key]

    raise ValueError(f"A rating must be up/down or +1/-1, not {rating!r}.")


def summarize_retrieved(results: list[Any]) -> list[dict[str, Any]]:
    r"""
    PURPOSE
    -------
    Reduce retrieved results to what is worth keeping: rank, score, ids,
    source and passage text. Accepts SearchResult / RerankedResult objects,
    prompt_builder Passages, or plain dicts.

    PARAMETERS
    ----------
    results   The retrieved results, best first.

    RETURNS
    -------
    One plain, JSON-ready dict per result.

    EXAMPLES
    --------
    >>> from types import SimpleNamespace as R
    >>> summarize_retrieved([R(rank=1, score=0.91, text="Heat flows.",
    ...                        record={"chunk_id": "C1", "source_file": "p.pdf", "source_page_start": 9})])
    [{'rank': 1, 'score': 0.91, 'chunk_id': 'C1', 'record_id': None, 'source_file': 'p.pdf', 'source_page_start': 9, 'text': 'Heat flows.'}]
    >>> summarize_retrieved([{"chunk_id": "C2", "text": "Light."}])[0]["rank"]
    1
    """
    summary: list[dict[str, Any]] = []

    for position, result in enumerate(results, start=1):
        if isinstance(result, dict):
            record = result.get("record") or result
            text = result.get("text") or record.get("text") or ""
            rank = result.get("rank") or position
            score = result.get("score")
        else:
            record = getattr(result, "record", {}) or {}
            text = getattr(result, "text", "") or ""
            rank = getattr(result, "rank", None) or getattr(result, "number", None) or position
            score = getattr(result, "score", None)

        summary.append(
            {
                "rank": rank,
                "score": None if score is None else round(float(score), 6),
                "chunk_id": record.get("chunk_id"),
                "record_id": record.get("record_id"),
                "source_file": record.get("source_file"),
                "source_page_start": record.get("source_page_start"),
                "text": str(text),
            }
        )

    return summary


class FeedbackLogger:
    r"""
    PURPOSE
    -------
    Log interactions and votes to a SQLite file, and read them back.

    ATTRIBUTES
    ----------
    path   The database file.

    EXAMPLES
    --------
    >>> import tempfile
    >>> with tempfile.TemporaryDirectory() as folder:
    ...     logger = FeedbackLogger(Path(folder) / "feedback.db")
    ...     interaction_id = logger.log_interaction(
    ...         question="What is momentum?", answer="Mass times velocity [1].",
    ...         retrieved=[{"chunk_id": "C1", "text": "Momentum is mass times velocity."}],
    ...         session_id="S1", language="eng_Latn")
    ...     logger.record_feedback(interaction_id, "down")
    ...     logger.record_feedback(interaction_id, "up", comment="changed my mind")
    ...     row = logger.get_interaction(interaction_id)
    ...     stats = logger.stats()
    >>> row["rating"], row["comment"], row["retrieved"][0]["chunk_id"]
    (1, 'changed my mind', 'C1')
    >>> stats["interactions"], stats["rated"], stats["thumbs_up"], stats["thumbs_down"]
    (1, 1, 1, 0)
    """

    def __init__(self, path: Path = DATABASE_FILE):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

        with closing(self.connect()) as connection, connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.executescript(SCHEMA)

    def connect(self) -> sqlite3.Connection:
        """
        PURPOSE
        -------
        Open a new connection. One per operation keeps the logger safe to
        share between threads and processes.
        """
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def log_interaction(
        self,
        question: str,
        answer: str,
        retrieved: list[Any],
        session_id: str | None = None,
        standalone_question: str | None = None,
        language: str | None = None,
        model: str | None = None,
        latency_ms: int | None = None,
        citations: Any = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """
        PURPOSE
        -------
        Store one answered question.

        PARAMETERS
        ----------
        question              What the user typed.
        answer                The answer shown to the user.
        retrieved             The results the answer was built from.
        session_id            Conversation id, if any.
        standalone_question   query_processing.py's rewrite, if different.
        language              GlotLID label of the question.
        model                 Name of the model that answered.
        latency_ms            Time from question to finished answer.
        citations             citation_formatter's FormattedAnswer, or a
                               JSON-ready dict; its references and check
                               result are stored.
        metadata              Any extra JSON-ready fields (app version...).

        RETURNS
        -------
        The new interaction id, to send with a later vote.

        RAISES
        ------
        ValueError   When the question or answer is empty.
        """
        if not str(question or "").strip() or not str(answer or "").strip():
            raise ValueError("Both the question and the answer are required.")

        citation_ok = None
        citations_payload: Any = citations

        if citations is not None and hasattr(citations, "report"):
            citation_ok = int(citations.report.ok)
            citations_payload = {
                "references": citations.references,
                "cited": citations.report.cited,
                "invalid": citations.report.invalid,
                "uncited": citations.report.uncited,
                "says_not_in_sources": citations.report.says_not_in_sources,
            }

        interaction_id = f"INT-{uuid.uuid4().hex}"

        with closing(self.connect()) as connection, connection:
            connection.execute(
                """INSERT INTO interactions VALUES
                   (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    interaction_id,
                    utc_now(),
                    session_id,
                    question,
                    standalone_question,
                    language,
                    answer,
                    model,
                    latency_ms,
                    citation_ok,
                    json.dumps(summarize_retrieved(retrieved), ensure_ascii=False),
                    None if citations_payload is None
                    else json.dumps(citations_payload, ensure_ascii=False),
                    None if metadata is None else json.dumps(metadata, ensure_ascii=False),
                ),
            )

        return interaction_id

    def record_feedback(
        self,
        interaction_id: str,
        rating: Any,
        comment: str | None = None,
    ) -> None:
        """
        PURPOSE
        -------
        Store a thumbs up or down for an interaction. Voting again keeps
        the history but the latest vote counts.

        PARAMETERS
        ----------
        interaction_id   The id log_interaction returned.
        rating           "up"/"down" or +1/-1.
        comment          Optional free-text comment.

        RAISES
        ------
        KeyError     When the interaction id is unknown.
        ValueError   When the rating is not up/down.
        """
        value = normalize_rating(rating)

        with closing(self.connect()) as connection, connection:
            known = connection.execute(
                "SELECT 1 FROM interactions WHERE interaction_id = ?", (interaction_id,)
            ).fetchone()

            if known is None:
                raise KeyError(f"Unknown interaction id: {interaction_id}")

            connection.execute(
                "INSERT INTO feedback (interaction_id, rating, comment, created_at) "
                "VALUES (?, ?, ?, ?)",
                (interaction_id, value, comment, utc_now()),
            )

    def iter_interactions(
        self,
        rated_only: bool = False,
        rating: int | None = None,
    ) -> Iterator[dict[str, Any]]:
        """
        PURPOSE
        -------
        Yield interactions, oldest first, each with its latest vote.

        PARAMETERS
        ----------
        rated_only   Only interactions that have a vote.
        rating       Only interactions whose latest vote is this (+1/-1).

        RETURNS
        -------
        An iterator of plain dicts with JSON fields decoded.
        """
        conditions = []
        parameters: list[Any] = []

        if rated_only or rating is not None:
            conditions.append("f.rating IS NOT NULL")

        if rating is not None:
            conditions.append("f.rating = ?")
            parameters.append(normalize_rating(rating))

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""

        with closing(self.connect()) as connection:
            rows = connection.execute(
                f"""SELECT i.*, f.rating, f.comment, f.created_at AS rated_at
                    FROM interactions AS i
                    LEFT JOIN ({LATEST_FEEDBACK}) AS f USING (interaction_id)
                    {where}
                    ORDER BY i.created_at, i.rowid""",
                parameters,
            ).fetchall()

        for row in rows:
            yield row_to_dict(row)

    def get_interaction(self, interaction_id: str) -> dict[str, Any] | None:
        """
        PURPOSE
        -------
        One interaction with its latest vote, or None if unknown.
        """
        with closing(self.connect()) as connection:
            row = connection.execute(
                f"""SELECT i.*, f.rating, f.comment, f.created_at AS rated_at
                    FROM interactions AS i
                    LEFT JOIN ({LATEST_FEEDBACK}) AS f USING (interaction_id)
                    WHERE i.interaction_id = ?""",
                (interaction_id,),
            ).fetchone()

        return None if row is None else row_to_dict(row)

    def stats(self) -> dict[str, Any]:
        """
        PURPOSE
        -------
        Counts for a quick health check: interactions, votes, the share of
        thumbs up among voted answers, failed citation checks, and
        interactions per language.
        """
        with closing(self.connect()) as connection:
            total = connection.execute("SELECT COUNT(*) FROM interactions").fetchone()[0]
            up, down = connection.execute(
                f"""SELECT COALESCE(SUM(rating = 1), 0), COALESCE(SUM(rating = -1), 0)
                    FROM ({LATEST_FEEDBACK})"""
            ).fetchone()
            citation_failures = connection.execute(
                "SELECT COUNT(*) FROM interactions WHERE citation_ok = 0"
            ).fetchone()[0]
            languages = connection.execute(
                """SELECT COALESCE(language, 'unknown'), COUNT(*) FROM interactions
                   GROUP BY 1 ORDER BY 2 DESC"""
            ).fetchall()

        rated = up + down

        return {
            "interactions": total,
            "rated": rated,
            "thumbs_up": up,
            "thumbs_down": down,
            "thumbs_up_rate": round(up / rated, 4) if rated else None,
            "citation_failures": citation_failures,
            "by_language": {language: count for language, count in languages},
        }

    def export_jsonl(
        self,
        output_path: Path,
        rated_only: bool = False,
        rating: int | None = None,
    ) -> int:
        """
        PURPOSE
        -------
        Write interactions to a JSONL file (atomically), ready to become
        training or evaluation data.

        PARAMETERS
        ----------
        output_path   Destination .jsonl file.
        rated_only    Only interactions that have a vote.
        rating        Only this latest vote (+1 for good answers, -1 for
                       bad ones).

        RETURNS
        -------
        The number of interactions written.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
        count = 0

        with temporary_path.open("w", encoding="utf-8") as output_file:
            for interaction in self.iter_interactions(rated_only=rated_only, rating=rating):
                output_file.write(json.dumps(interaction, ensure_ascii=False) + "\n")
                count += 1

        temporary_path.replace(output_path)

        return count


def row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    """
    PURPOSE
    -------
    Turn a joined interaction row into a plain dict, decoding its JSON
    columns ("retrieved_json" becomes "retrieved", and so on).
    """
    data = dict(row)

    for column in ("retrieved_json", "citations_json", "metadata_json"):
        raw = data.pop(column, None)
        data[column.removesuffix("_json")] = None if raw is None else json.loads(raw)

    if data.get("citation_ok") is not None:
        data["citation_ok"] = bool(data["citation_ok"])

    return data


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
    >>> parse_arguments(["stats"]).command
    'stats'
    >>> options = parse_arguments(["export", "--output", "good.jsonl", "--rating", "up"])
    >>> options.command, options.output.name, options.rating, options.rated_only
    ('export', 'good.jsonl', 'up', False)
    """
    parser = argparse.ArgumentParser(description="Inspect and export logged feedback.")
    parser.add_argument("--database", type=Path, default=DATABASE_FILE, help="SQLite file")

    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("stats", help="print counts and the thumbs-up rate")

    export = commands.add_parser("export", help="write interactions to JSONL")
    export.add_argument("--output", type=Path, required=True, help="destination .jsonl")
    export.add_argument("--rated-only", action="store_true", help="only voted interactions")
    export.add_argument("--rating", choices=["up", "down"], help="only this latest vote")

    return parser.parse_args(argument_list)


def main() -> None:
    """
    PURPOSE
    -------
    Run the subcommand given on the command line.
    """
    options = parse_arguments()
    logger = FeedbackLogger(options.database)

    if options.command == "stats":
        print(json.dumps(logger.stats(), ensure_ascii=False, indent=2))
        return

    count = logger.export_jsonl(options.output, rated_only=options.rated_only, rating=options.rating)
    print(f"Exported {count} interaction(s) to {options.output}")


if __name__ == "__main__":
    main()
