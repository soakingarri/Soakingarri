#!/usr/bin/env python3
"""
Test cases for reranker.py

Run with either:
    python3 test_reranker.py          (unittest, no extra installs)
    python3 -m pytest test_reranker.py

Every function in reranker.py carries doctest examples in its docstring, and
those use a fake cross-encoder, so this file (and the doctests) run fast and
offline. A real end-to-end check against BAAI/bge-reranker-v2-m3 is included
too, but it is opt-in (set SOAKINGARRI_REAL_MODELS=1) because it downloads
a ~2.3 GB model on first run.
"""

from __future__ import annotations

import doctest
import importlib.util
import os
import unittest

import reranker as rr


def load_tests(loader, tests, ignore):
    """Hook so `python3 test_reranker.py` also runs every docstring example."""
    tests.addTests(doctest.DocTestSuite(rr))
    return tests


def has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


class FakeModel:
    """Deterministic stand-in for CrossEncoder.

    Scores a (query, text) pair by how many of the query's words appear in
    the text, so relevance is exactly predictable without any model.
    """

    def __init__(self):
        self.seen_pairs: list[tuple[str, str]] = []

    def predict(self, pairs):
        self.seen_pairs.extend(pairs)
        scores = []

        for query, text in pairs:
            query_words = set(query.casefold().split())
            text_words = set(text.casefold().split())
            scores.append(float(len(query_words & text_words)))

        return scores


class Candidate:
    """A minimal vb.SearchResult-shaped stand-in: rank, text, record."""

    def __init__(self, rank: int, text: str, record: dict | None = None):
        self.rank = rank
        self.text = text
        self.record = record if record is not None else {"text": text}


class RerankSortingTests(unittest.TestCase):
    def test_reorders_by_cross_encoder_score(self):
        # The candidate that started last should end up first, since it is
        # the only one that actually mentions the query terms.
        candidates = [
            Candidate(1, "Refraction bends light as it crosses a boundary."),
            Candidate(2, "Momentum is the product of mass and velocity."),
            Candidate(3, "Heat transfer moves energy from hot to cold."),
        ]

        results = rr.rerank(
            "heat transfer", candidates, model=FakeModel(), top_k=3
        )

        self.assertEqual(results[0].text, candidates[2].text)
        self.assertEqual(results[0].rank, 1)
        self.assertEqual(results[0].original_rank, 3)
        # Scores should be sorted descending.
        scores = [result.score for result in results]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_ranks_are_sequential_and_one_indexed(self):
        candidates = [Candidate(1, "a"), Candidate(2, "b"), Candidate(3, "c")]
        results = rr.rerank("a b c", candidates, model=FakeModel(), top_k=3)
        self.assertEqual([r.rank for r in results], [1, 2, 3])

    def test_record_and_text_are_carried_through_unchanged(self):
        record = {"topic": "Heat", "text": "Heat transfer."}
        candidates = [Candidate(1, "Heat transfer.", record=record)]
        results = rr.rerank("heat", candidates, model=FakeModel(), top_k=1)
        self.assertIs(results[0].record, record)
        self.assertEqual(results[0].text, "Heat transfer.")

    def test_original_rank_falls_back_to_position_without_rank_attribute(self):
        class NoRankCandidate:
            def __init__(self, text):
                self.text = text
                self.record = {"text": text}

        candidates = [NoRankCandidate("first"), NoRankCandidate("second")]
        results = rr.rerank("first second", candidates, model=FakeModel(), top_k=2)
        self.assertEqual({r.original_rank for r in results}, {1, 2})


class RerankTruncationTests(unittest.TestCase):
    def test_top_k_truncates(self):
        candidates = [Candidate(i, f"text {i}") for i in range(1, 6)]
        results = rr.rerank("text", candidates, model=FakeModel(), top_k=2)
        self.assertEqual(len(results), 2)

    def test_top_k_larger_than_candidates_returns_everything(self):
        candidates = [Candidate(1, "a"), Candidate(2, "b")]
        results = rr.rerank("a b", candidates, model=FakeModel(), top_k=50)
        self.assertEqual(len(results), 2)

    def test_top_k_zero_or_negative_returns_empty(self):
        candidates = [Candidate(1, "a")]
        self.assertEqual(rr.rerank("a", candidates, model=FakeModel(), top_k=0), [])
        self.assertEqual(rr.rerank("a", candidates, model=FakeModel(), top_k=-1), [])


class RerankEmptyInputTests(unittest.TestCase):
    def test_empty_candidates_returns_empty(self):
        self.assertEqual(rr.rerank("anything", [], model=FakeModel()), [])

    def test_duplicate_candidates_are_all_scored(self):
        candidates = [Candidate(1, "heat"), Candidate(2, "heat")]
        model = FakeModel()
        results = rr.rerank("heat", candidates, model=model, top_k=2)
        self.assertEqual(len(results), 2)
        self.assertEqual(len(model.seen_pairs), 2)


class LoadRerankerDependencyTests(unittest.TestCase):
    @unittest.skipIf(
        has_module("sentence_transformers"),
        "sentence-transformers is installed; dependency error path not reachable",
    )
    def test_raises_runtime_error_without_sentence_transformers(self):
        with self.assertRaises(RuntimeError):
            rr.load_reranker()


@unittest.skipUnless(
    has_module("sentence_transformers") and os.environ.get("SOAKINGARRI_REAL_MODELS") == "1",
    "real-model test is opt-in: set SOAKINGARRI_REAL_MODELS=1 to run it",
)
class RealCrossEncoderEndToEndTests(unittest.TestCase):
    """
    Downloads and runs BAAI/bge-reranker-v2-m3 once (~2.3 GB). Opt-in only,
    so the normal suite stays fast and offline:
        SOAKINGARRI_REAL_MODELS=1 python3 -m pytest data/ask/reranker
    """

    @classmethod
    def setUpClass(cls):
        cls.model = rr.load_reranker()

    def test_relevant_passage_scores_above_irrelevant_passage(self):
        query = "How is momentum defined in physics?"

        candidates = [
            Candidate(
                1,
                "Bananas are a good source of potassium and are often eaten "
                "as a quick snack between meals.",
            ),
            Candidate(
                2,
                "Momentum is defined as the product of an object's mass and "
                "its velocity, a vector quantity conserved in collisions.",
            ),
        ]

        results = rr.rerank(query, candidates, model=self.model, top_k=2)

        self.assertEqual(results[0].text, candidates[1].text)
        self.assertGreater(results[0].score, results[1].score)


if __name__ == "__main__":
    unittest.main()
