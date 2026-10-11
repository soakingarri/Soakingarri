#!/usr/bin/env python3
"""
Test cases for feedback_logger.py

Run with either:
    python3 test_feedback_logger.py          (unittest, no extra installs)
    python3 -m pytest test_feedback_logger.py

Runs every docstring example plus logging, voting, export, concurrency and
citation_formatter integration checks, each against a fresh temporary
SQLite file.
"""

from __future__ import annotations

import doctest
import json
import sys
import tempfile
import threading
import unittest
from pathlib import Path

import feedback_logger as fl

HERE = Path(fl.__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "citation_formatter"))
sys.path.insert(0, str(HERE.parent / "vector_database"))


def load_tests(loader, tests, ignore):
    """Hook so `python3 test_feedback_logger.py` also runs every docstring example."""
    tests.addTests(doctest.DocTestSuite(fl))
    return tests


RETRIEVED = [
    {"chunk_id": "C1", "text": "Momentum is mass times velocity.", "score": 0.9},
    {"chunk_id": "C2", "text": "Impulse changes momentum.", "score": 0.7},
]


class LoggerTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.logger = fl.FeedbackLogger(Path(self.tmp.name) / "nested" / "feedback.db")

    def tearDown(self):
        self.tmp.cleanup()

    def log(self, question="What is momentum?", **kwargs):
        return self.logger.log_interaction(
            question=question, answer="Mass times velocity [1].", retrieved=RETRIEVED, **kwargs
        )


class LoggingTests(LoggerTestCase):
    def test_round_trip(self):
        interaction_id = self.log(
            session_id="S1", standalone_question="What is momentum? (physics)",
            language="eng_Latn", model="gpt-x", latency_ms=420, metadata={"app": "1.2"},
        )
        row = self.logger.get_interaction(interaction_id)
        self.assertEqual(row["session_id"], "S1")
        self.assertEqual(row["latency_ms"], 420)
        self.assertEqual(row["metadata"], {"app": "1.2"})
        self.assertEqual([r["rank"] for r in row["retrieved"]], [1, 2])
        self.assertEqual(row["retrieved"][1]["score"], 0.7)
        self.assertIsNone(row["rating"])

    def test_ids_are_unique(self):
        self.assertNotEqual(self.log(), self.log())

    def test_empty_question_or_answer_is_rejected(self):
        with self.assertRaises(ValueError):
            self.logger.log_interaction(question=" ", answer="x", retrieved=[])
        with self.assertRaises(ValueError):
            self.logger.log_interaction(question="q", answer="", retrieved=[])

    def test_unknown_interaction_returns_none(self):
        self.assertIsNone(self.logger.get_interaction("INT-missing"))

    def test_database_survives_reopening(self):
        interaction_id = self.log()
        reopened = fl.FeedbackLogger(self.logger.path)
        self.assertEqual(reopened.get_interaction(interaction_id)["question"], "What is momentum?")


class FeedbackTests(LoggerTestCase):
    def test_unknown_interaction_is_rejected(self):
        with self.assertRaises(KeyError):
            self.logger.record_feedback("INT-missing", "up")

    def test_invalid_rating_is_rejected(self):
        interaction_id = self.log()
        for rating in ("meh", 0, 5, None):
            with self.assertRaises(ValueError):
                self.logger.record_feedback(interaction_id, rating)

    def test_latest_vote_counts_in_stats(self):
        first, second, third = self.log(), self.log(), self.log()
        self.logger.record_feedback(first, "up")
        self.logger.record_feedback(second, "up")
        self.logger.record_feedback(second, "down")
        stats = self.logger.stats()
        self.assertEqual(
            (stats["interactions"], stats["rated"], stats["thumbs_up"], stats["thumbs_down"]),
            (3, 2, 1, 1),
        )
        self.assertEqual(stats["thumbs_up_rate"], 0.5)

    def test_stats_on_empty_database(self):
        stats = self.logger.stats()
        self.assertEqual((stats["interactions"], stats["thumbs_up_rate"]), (0, None))


class ExportTests(LoggerTestCase):
    def test_filters(self):
        good, bad, unrated = self.log("good?"), self.log("bad?"), self.log("unrated?")
        self.logger.record_feedback(good, "up")
        self.logger.record_feedback(bad, "down")

        output = Path(self.tmp.name) / "out" / "export.jsonl"

        self.assertEqual(self.logger.export_jsonl(output), 3)
        self.assertEqual(self.logger.export_jsonl(output, rated_only=True), 2)
        self.assertEqual(self.logger.export_jsonl(output, rating="up"), 1)

        lines = output.read_text(encoding="utf-8").splitlines()
        row = json.loads(lines[0])
        self.assertEqual((row["question"], row["rating"]), ("good?", 1))
        self.assertEqual(row["retrieved"][0]["chunk_id"], "C1")
        self.assertFalse(list(output.parent.glob("*.tmp")))

    def test_order_is_oldest_first(self):
        ids = [self.log(f"q{n}") for n in range(5)]
        self.assertEqual([r["interaction_id"] for r in self.logger.iter_interactions()], ids)


class ConcurrencyTests(LoggerTestCase):
    def test_parallel_writers(self):
        errors: list[Exception] = []

        def worker(n):
            try:
                for _ in range(10):
                    self.logger.record_feedback(self.log(f"thread {n}"), "up")
            except Exception as error:  # pragma: no cover - reported below
                errors.append(error)

        threads = [threading.Thread(target=worker, args=(n,)) for n in range(6)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        self.assertEqual(errors, [])
        self.assertEqual(self.logger.stats()["thumbs_up"], 60)


class CitationIntegrationTests(LoggerTestCase):
    def test_formatted_answer_is_stored(self):
        from types import SimpleNamespace

        import citation_formatter as cf

        passages = [SimpleNamespace(number=1, record={"chunk_id": "C1", "source_file": "p_cleaned.json",
                                                      "source_page_start": 4})]
        formatted = cf.format_answer("Mass times velocity [1][2].", passages, lookup={})
        interaction_id = self.logger.log_interaction(
            question="What is momentum?", answer=formatted.text, retrieved=passages, citations=formatted
        )
        row = self.logger.get_interaction(interaction_id)
        self.assertFalse(row["citation_ok"])
        self.assertEqual(row["citations"]["invalid"], [2])
        self.assertEqual(row["citations"]["references"][0]["reference"], "p, p. 4")
        self.assertEqual(row["retrieved"][0]["rank"], 1)
        self.assertEqual(self.logger.stats()["citation_failures"], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
