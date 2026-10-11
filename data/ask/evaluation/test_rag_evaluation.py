#!/usr/bin/env python3
"""
Test cases for rag_evaluation.py

Run with either:
    python3 test_rag_evaluation.py          (unittest, no extra installs)
    python3 -m pytest test_rag_evaluation.py

Runs every docstring example plus metric, comparison and end-to-end checks.
The end-to-end test scores a real hybrid retriever (vector_base +
keyword_index + retrieval) and answers built with the real prompt_builder,
using a stub generator and stub embeddings so nothing is downloaded.
"""

from __future__ import annotations

import doctest
import tempfile
import unittest
from pathlib import Path

import numpy as np

import rag_evaluation as re_
import eval_dataset_builder as edb


def load_tests(loader, tests, ignore):
    """Hook so `python3 test_rag_evaluation.py` also runs every docstring example."""
    tests.addTests(doctest.DocTestSuite(re_))
    return tests


class MetricTests(unittest.TestCase):
    def test_no_results_scores_zero(self):
        scores = re_.retrieval_scores([], {"A"}, k_values=(1, 5))
        self.assertEqual(scores, {"hit@1": 0.0, "recall@1": 0.0, "hit@5": 0.0, "recall@5": 0.0,
                                  "reciprocal_rank": 0.0})

    def test_llm_record_matches_by_record_id(self):
        scores = re_.retrieval_scores([{"record_id": "QA-1"}], {"QA-1"}, k_values=(1,))
        self.assertEqual(scores["hit@1"], 1.0)

    def test_duplicate_hits_do_not_inflate_recall(self):
        results = [{"chunk_id": "A"}, {"chunk_id": "A"}, {"chunk_id": "A"}]
        self.assertEqual(re_.retrieval_scores(results, {"A", "B"}, k_values=(3,))["recall@3"], 0.5)

    def test_judge_failure_is_unknown_not_zero(self):
        def broken(prompt):
            raise TimeoutError

        self.assertIsNone(re_.judge_faithfulness("x", [], broken))


class ComparisonTests(unittest.TestCase):
    def test_improvements_are_not_flagged(self):
        changes = re_.compare_runs({"mrr": 0.8, "mean_seconds": 0.5}, {"mrr": 0.7, "mean_seconds": 0.9})
        self.assertEqual(changes["mrr"]["worse"], 0.0)
        self.assertEqual(changes["mean_seconds"]["worse"], 0.0)

    def test_metrics_missing_from_one_run_are_skipped(self):
        changes = re_.compare_runs({"mrr": 0.8, "faithfulness": 0.9}, {"mrr": 0.7, "faithfulness": None})
        self.assertEqual(list(changes), ["mrr"])

    def test_saved_runs_are_compared_in_order(self):
        with tempfile.TemporaryDirectory() as folder:
            runs = Path(folder)
            re_.save_run({"summary": {"mrr": 0.9}}, runs)
            re_.save_run({"summary": {"mrr": 0.7}}, runs)
            self.assertEqual(re_.latest_run(runs)["summary"]["mrr"], 0.7)
            self.assertFalse(list(runs.glob("*.tmp")))


class EndToEndTests(unittest.TestCase):
    """Golden set -> real hybrid retrieval -> real prompt -> stub answers."""

    def setUp(self):
        import keyword_index as ki
        import retrieval as ret
        import vector_base as vb

        texts = {
            "C-momentum": "Momentum is the product of mass and velocity.",
            "C-energy": "Kinetic energy grows with the square of velocity.",
            "C-light": "Refraction bends light when it enters water.",
            "C-ori": "Ori is the spiritual head and seat of destiny in Yoruba thought.",
        }
        axes = {"C-momentum": [1, 0, 0], "C-energy": [0.8, 0.6, 0], "C-light": [0, 1, 0], "C-ori": [0, 0, 1]}
        records = [
            {"row": n, "dataset": "rag", "chunk_id": cid, "text": text, "source_file": "b_cleaned.json"}
            for n, (cid, text) in enumerate(texts.items())
        ]
        base = vb.VectorBase(
            vb.normalize_rows(np.array([axes[r["chunk_id"]] for r in records], dtype="float32")),
            records,
            {"embedding_model": "stub"},
        )
        index = ki.KeywordIndex(records, [ki.tokenize(r["text"]) for r in records])

        def encode(text):
            lowered = text.casefold()
            return np.array(
                [float("momentum" in lowered or "mass" in lowered),
                 float("light" in lowered),
                 float("ori" in lowered.split() or "destiny" in lowered)],
                dtype="float32",
            ) + 0.01

        self.retriever = lambda q: ret.hybrid_retrieve(base, index, q, encode, top_k=3, minimum_score=0.5)

        self.entries = [
            edb.make_entry("What is momentum?", "Mass times velocity.", ["C-momentum"], "manual",
                           status="verified", language="eng_Latn"),
            edb.make_entry("How does light bend in water?", "By refraction.", ["C-light"], "manual",
                           status="verified"),
            edb.make_entry("What is ori?", "The seat of destiny.", ["C-ori"], "manual", status="verified"),
            edb.make_entry("Unreviewed?", "x", ["C-energy"], "template"),
        ]

    def test_retrieval_only(self):
        report = re_.evaluate(self.entries, self.retriever, k_values=(1, 3))
        summary = report["summary"]
        self.assertEqual(summary["questions"], 3)
        self.assertEqual(summary["hit@1"], 1.0)
        self.assertEqual(summary["mrr"], 1.0)
        self.assertNotIn("citation_validity", summary)

    def test_answers_citations_and_faithfulness(self):
        def generator(messages):
            # Cites [1] correctly for momentum, invents [9] for light, and
            # refuses for ori.
            question = messages[-1]["content"]
            if "momentum" in question:
                return "Momentum is mass times velocity [1]."
            if "light" in question:
                return "Light bends by refraction [1][9]."
            return "The provided sources do not contain the answer to this question."

        judge = lambda prompt: "YES" if "[9]" not in prompt else "NO"

        report = re_.evaluate(
            self.entries, self.retriever, re_.build_answerer(generator), judge, k_values=(1, 3)
        )
        summary = report["summary"]
        self.assertEqual(summary["citation_validity"], round(2 / 3, 4))
        self.assertEqual(summary["citation_precision"], 1.0)
        self.assertEqual(summary["refusal_rate"], round(1 / 3, 4))
        self.assertEqual(summary["faithfulness"], round(2 / 3, 4))
        light = next(row for row in report["questions"] if "light" in row["question"])
        self.assertEqual(light["citation_valid"], 0.0)

    def test_a_worse_retriever_is_flagged_against_the_previous_run(self):
        good = re_.evaluate(self.entries, self.retriever, k_values=(1, 3))["summary"]
        bad = re_.evaluate(self.entries, lambda q: list(reversed(self.retriever(q))), k_values=(1, 3))["summary"]
        changes = re_.compare_runs(bad, good)
        self.assertGreater(changes["hit@1"]["worse"], 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
