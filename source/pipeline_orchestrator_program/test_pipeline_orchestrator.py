#!/usr/bin/env python3
"""
Test cases for pipeline_orchestrator.py

Run with either:
    python3 test_pipeline_orchestrator.py          (unittest, no extra installs)
    python3 -m pytest test_pipeline_orchestrator.py

Runs every docstring example plus end-to-end runs of the REAL stages
(cleaning, deduplication, language detection, chunking, embedding, vector
base, keyword index) on a small temporary corpus. Stub language and
embedding models stand in for GlotLID and bge-m3, so nothing is downloaded.
"""

from __future__ import annotations

import doctest
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

import pipeline_orchestrator as po


def load_tests(loader, tests, ignore):
    """Hook so `python3 test_pipeline_orchestrator.py` also runs every docstring example."""
    tests.addTests(doctest.DocTestSuite(po))
    return tests


class StubLanguageModel:
    def predict(self, text, k=1):
        return (("__label__eng_Latn",), (0.99,))


class StubEmbeddingModel:
    """Deterministic unit vectors from a text hash."""

    dimension = 8

    def get_sentence_embedding_dimension(self):
        return self.dimension

    def encode(self, sentences, **kwargs):
        vectors = np.array(
            [
                np.frombuffer(hashlib.sha256(text.encode()).digest()[: self.dimension], dtype=np.uint8)
                for text in sentences
            ],
            dtype="float32",
        ) + 1.0
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def book_text(subject: str) -> str:
    sentence = f"{subject} is described in this chapter with a long and careful explanation. "
    return f"A Book About {subject}\nChapter 1: {subject}\n" + sentence * 40 + "\n"


class EndToEndTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.raw = self.root / "raw_data"
        self.raw.mkdir()
        (self.raw / "momentum.txt").write_text(book_text("Momentum"), encoding="utf-8")
        (self.raw / "entropy.txt").write_text(book_text("Entropy"), encoding="utf-8")
        self.stages = po.build_stages(
            self.root, language_model=StubLanguageModel(), embedding_model=StubEmbeddingModel()
        )
        self.lines: list[str] = []

    def tearDown(self):
        self.tmp.cleanup()

    def run_pipeline(self, **kwargs):
        self.lines = []
        return po.run_pipeline(self.root, self.stages, output_function=self.lines.append, **kwargs)

    def statuses(self, report):
        return {name: info["status"] for name, info in report["stages"].items()}

    def test_first_run_builds_everything_searchable(self):
        report = self.run_pipeline()
        self.assertTrue(report["ok"], self.lines)
        self.assertEqual(sorted(report["cleaning"]["cleaned"]), ["entropy.txt", "momentum.txt"])
        self.assertEqual(set(self.statuses(report).values()), {"success"})

        import keyword_index as ki
        import vector_base as vb

        base = vb.VectorBase.load(self.root / "vector_base")
        index = ki.KeywordIndex.load(self.root / "keyword_index")
        self.assertGreater(len(base), 0)
        self.assertEqual(len(base), len(index))
        hits = index.search("entropy", top_k=1)
        self.assertIn("Entropy", hits[0].text)
        self.assertEqual(hits[0].record["language"], "eng_Latn")

        state = json.loads((self.root / "metadata" / po.STATE_FILE_NAME).read_text("utf-8"))
        self.assertEqual(state["stages"]["chunking"]["status"], "success")
        self.assertIn("chunk_count", state["stages"]["chunking"]["manifest"])

    def test_unchanged_rerun_skips_everything(self):
        self.run_pipeline()
        report = self.run_pipeline()
        self.assertEqual(report["cleaning"]["cleaned"], [])
        self.assertEqual(report["cleaning"]["unchanged"], 2)
        self.assertEqual(set(self.statuses(report).values()), {"skipped"})

    def test_changing_one_file_recleans_only_that_file(self):
        self.run_pipeline()
        (self.raw / "entropy.txt").write_text(book_text("Enthalpy"), encoding="utf-8")
        report = self.run_pipeline()
        self.assertEqual(report["cleaning"]["cleaned"], ["entropy.txt"])
        self.assertEqual(report["cleaning"]["unchanged"], 1)
        self.assertEqual(set(self.statuses(report).values()), {"success"})

        import keyword_index as ki

        index = ki.KeywordIndex.load(self.root / "keyword_index")
        self.assertTrue(index.search("enthalpy", top_k=1))
        self.assertFalse([h for h in index.search("entropy", top_k=5) if h.score > 0])

    def test_deleted_raw_file_disappears_downstream(self):
        self.run_pipeline()
        (self.raw / "momentum.txt").unlink()
        report = self.run_pipeline()
        self.assertEqual(report["cleaning"]["removed"], ["momentum.txt"])
        self.assertFalse((self.root / "cleaned_data" / "momentum_cleaned.json").exists())

        import keyword_index as ki

        index = ki.KeywordIndex.load(self.root / "keyword_index")
        self.assertFalse([h for h in index.search("momentum", top_k=5) if h.score > 0])

    def test_dry_run_changes_nothing(self):
        report = self.run_pipeline(dry_run=True)
        self.assertEqual(set(self.statuses(report).values()), {"would_run"})
        self.assertFalse((self.root / "cleaned_data").exists())
        self.assertFalse((self.root / "metadata" / po.STATE_FILE_NAME).exists())

    def test_force_from_reruns_that_stage_and_after(self):
        self.run_pipeline()
        report = self.run_pipeline(force_from="embedding")
        statuses = self.statuses(report)
        self.assertEqual(statuses["chunking"], "skipped")
        self.assertEqual(
            [statuses[name] for name in ("embedding", "vector_base", "keyword_index")],
            ["success"] * 3,
        )

    def test_deleted_output_is_rebuilt(self):
        self.run_pipeline()
        for path in (self.root / "chunked_data").iterdir():
            path.unlink()
        report = self.run_pipeline()
        self.assertEqual(self.statuses(report)["chunking"], "success")

    def test_unknown_stage_name(self):
        with self.assertRaises(ValueError):
            self.run_pipeline(force_from="nonsense")


class FailureTests(unittest.TestCase):
    def test_failure_stops_later_stages_and_retries_next_time(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "raw_data").mkdir()
            calls: list[str] = []
            fail = {"now": True}

            def make(name, source, target, may_fail=False):
                def run():
                    calls.append(name)
                    if may_fail and fail["now"]:
                        raise RuntimeError("disk full")
                    (root / target).mkdir(exist_ok=True)
                    (root / target / "out.json").write_text(name)
                    return {}

                return po.Stage(name, [root / source], root / target, run)

            stages = [
                make("first", "cleaned_data", "first_out"),
                make("second", "first_out", "second_out", may_fail=True),
                make("third", "second_out", "third_out"),
            ]
            quiet = lambda line: None

            report = po.run_pipeline(root, stages, output_function=quiet)
            self.assertFalse(report["ok"])
            self.assertEqual(calls, ["first", "second"])
            self.assertIn("disk full", report["stages"]["second"]["error"])

            fail["now"] = False
            report = po.run_pipeline(root, stages, output_function=quiet)
            self.assertTrue(report["ok"])
            self.assertEqual(calls, ["first", "second", "second", "third"])

    def test_unreadable_raw_file_is_reported_per_file(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "raw_data").mkdir()
            (root / "raw_data" / "good.txt").write_text(book_text("Heat"), encoding="utf-8")
            (root / "raw_data" / "broken.pdf").write_bytes(b"not really a pdf")
            report = po.run_pipeline(root, [], output_function=lambda line: None)
            self.assertEqual(report["cleaning"]["cleaned"], ["good.txt"])
            self.assertEqual([f["file"] for f in report["cleaning"]["failed"]], ["broken.pdf"])
            self.assertFalse(report["ok"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
