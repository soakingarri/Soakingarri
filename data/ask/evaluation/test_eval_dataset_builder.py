#!/usr/bin/env python3
"""
Test cases for eval_dataset_builder.py

Run with either:
    python3 test_eval_dataset_builder.py          (unittest, no extra installs)
    python3 -m pytest test_eval_dataset_builder.py

Runs every docstring example plus drafting, review, feedback-import and
validation checks. Stub generators replace any model.
"""

from __future__ import annotations

import doctest
import json
import tempfile
import unittest
from pathlib import Path

import eval_dataset_builder as edb


def load_tests(loader, tests, ignore):
    """Hook so `python3 test_eval_dataset_builder.py` also runs every docstring example."""
    tests.addTests(doctest.DocTestSuite(edb))
    return tests


BODY = "Momentum is the product of mass and velocity. It is conserved in collisions. " * 8


def make_chunks(documents: int = 3, per_document: int = 5) -> list[dict]:
    return [
        {
            "chunk_id": f"CHUNK-{d}-{n}",
            "document_id": f"DOC-{d}",
            "concept": f"Concept {d}.{n}",
            "formula": "p = mv" if n == 0 else None,
            "text": BODY,
            "source_metadata": {"source_file": f"book{d}_cleaned.json", "language": "eng_Latn"},
        }
        for d in range(documents)
        for n in range(per_document)
    ]


class DraftTests(unittest.TestCase):
    def test_sampling_is_reproducible_and_balanced(self):
        chunks = make_chunks()
        first = [c["chunk_id"] for c in edb.sample_chunks(chunks, 6, seed=1)]
        second = [c["chunk_id"] for c in edb.sample_chunks(chunks, 6, seed=1)]
        self.assertEqual(first, second)
        per_document = {d: sum(cid.startswith(f"CHUNK-{d}-") for cid in first) for d in range(3)}
        self.assertEqual(per_document, {0: 2, 1: 2, 2: 2})

    def test_drafts_carry_source_metadata(self):
        entries, _ = edb.draft_entries([], make_chunks(1, 1), 1)
        entry = entries[0]
        self.assertEqual(entry["source_file"], "book0_cleaned.json")
        self.assertEqual(entry["language"], "eng_Latn")
        self.assertEqual(entry["status"], "draft")
        self.assertLessEqual(len(entry["reference_answer"].split()), edb.REFERENCE_ANSWER_WORDS)

    def test_generator_drafts_and_partial_failures(self):
        calls = {"n": 0}

        def flaky(prompt):
            calls["n"] += 1
            if calls["n"] % 2:
                raise ConnectionError
            return '{"question": "Why is momentum conserved?", "answer": "No external force."}'

        entries, added = edb.draft_entries([], make_chunks(1, 4), 4, generator=flaky)
        self.assertEqual(added, 4)
        self.assertEqual(sorted(e["origin"] for e in entries), ["generator"] * 2 + ["template"] * 2)

    def test_drafting_never_reuses_a_chunk(self):
        chunks = make_chunks(1, 3)
        entries, _ = edb.draft_entries([], chunks, 2)
        entries, added = edb.draft_entries(entries, chunks, 5)
        self.assertEqual(added, 1)
        used = [e["source_chunk_ids"][0] for e in entries]
        self.assertEqual(len(used), len(set(used)))


class ReviewTests(unittest.TestCase):
    def test_reject_skip_and_unknown_reply(self):
        entries = [edb.make_entry(f"Q{n}?", "A.", ["C1"], "template") for n in range(3)]
        replies = iter(["r", "s", "x"])
        counts = edb.review_entries(entries, {}, lambda prompt: next(replies), lambda text: None)
        self.assertEqual([e["status"] for e in entries], ["rejected", "draft", "draft"])
        self.assertEqual(dict(counts), {"rejected": 1, "skipped": 2})

    def test_only_drafts_are_shown(self):
        entries = [
            edb.make_entry("Done?", "A.", ["C1"], "manual", status="verified"),
            edb.make_entry("New?", "A.", ["C1"], "template"),
        ]
        shown = []
        edb.review_entries(entries, {"C1": {"text": "Passage text."}}, lambda p: "a", shown.append)
        self.assertEqual(len(shown), 1)
        self.assertIn("New?", shown[0])
        self.assertIn("Passage text.", shown[0])

    def test_end_of_input_stops_cleanly(self):
        entries = [edb.make_entry("Q?", "A.", ["C1"], "template")]

        def closed(prompt):
            raise EOFError

        edb.review_entries(entries, {}, closed, lambda text: None)
        self.assertEqual(entries[0]["status"], "draft")


class FeedbackImportTests(unittest.TestCase):
    def test_real_feedback_database(self):
        import sys

        sys.path.insert(0, str(Path(edb.__file__).resolve().parent.parent / "feedback"))
        from feedback_logger import FeedbackLogger

        with tempfile.TemporaryDirectory() as folder:
            logger = FeedbackLogger(Path(folder) / "feedback.db")
            liked = logger.log_interaction(
                question="And why?", standalone_question="Why is momentum conserved?",
                answer="No external force acts [1].", retrieved=[{"chunk_id": "C1"}, {"chunk_id": "C2"}],
            )
            disliked = logger.log_interaction(question="q", answer="a", retrieved=[{"chunk_id": "C3"}])
            logger.record_feedback(liked, "up")
            logger.record_feedback(disliked, "down")

            entries, added = edb.import_feedback([], logger.iter_interactions(rating=1))

        self.assertEqual(added, 1)
        self.assertEqual(entries[0]["question"], "Why is momentum conserved?")
        self.assertEqual(entries[0]["reference_answer"], "No external force acts.")
        self.assertEqual(entries[0]["source_chunk_ids"], ["C1", "C2"])
        self.assertEqual(entries[0]["interaction_id"], liked)


class StorageTests(unittest.TestCase):
    def test_round_trip_preserves_unicode(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "golden_set.jsonl"
            entry = edb.make_entry("Kí ni orí?", "Orí ni ìpín.", ["C1"], "manual", status="verified")
            edb.save_golden_set(path, [entry])
            self.assertIn("Kí ni orí?", path.read_text(encoding="utf-8"))
            self.assertEqual(edb.load_golden_set(path), [entry])

    def test_invalid_status_is_rejected(self):
        with self.assertRaises(ValueError):
            edb.make_entry("Q?", "A.", ["C1"], "manual", status="golden")

    def test_stats_target(self):
        entries = [
            edb.make_entry(f"Q{n}?", "A.", ["C1"], "manual", status="verified") for n in range(3)
        ] + [edb.make_entry("D?", "A.", ["C1"], "template")]
        stats = edb.golden_set_stats(entries)
        self.assertEqual(stats["by_status"], {"verified": 3, "draft": 1})
        self.assertEqual(stats["verified_needed"], 197)
        self.assertFalse(stats["over_maximum"])


class ChunkFileTests(unittest.TestCase):
    def test_load_chunks_reads_rag_data(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / edb.RAG_FILE_NAME
            path.write_text("\n".join(json.dumps(c) for c in make_chunks(1, 2)) + "\n", encoding="utf-8")
            self.assertEqual(len(edb.load_chunks(Path(folder))), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
