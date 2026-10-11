#!/usr/bin/env python3
"""
Test cases for query_processing.py

Run with either:
    python3 test_query_processing.py          (unittest, no extra installs)
    python3 -m pytest test_query_processing.py

Runs every docstring example plus rewrite, language and expansion checks.
A stub language model and stub text generators stand in for GlotLID and a
chat model, so nothing is downloaded or called over the network.
"""

from __future__ import annotations

import doctest
import unittest

import numpy as np

import query_processing as qp


def load_tests(loader, tests, ignore):
    """Hook so `python3 test_query_processing.py` also runs every docstring example."""
    tests.addTests(doctest.DocTestSuite(qp))
    return tests


class StubLanguageModel:
    """Yoruba when a Yoruba marker word appears, English otherwise."""

    def __init__(self):
        self.seen: list[str] = []

    def predict(self, text, k=1):
        self.seen.append(text)
        words = set(text.casefold().replace("?", " ").split())

        if words & {"ni", "orí", "ori", "kí"}:
            return (("__label__yor_Latn",), (0.97,))

        return (("__label__eng_Latn",), (0.99,))


class FollowUpTests(unittest.TestCase):
    def test_first_question_is_never_a_follow_up(self):
        self.assertFalse(qp.is_follow_up("What about it?", None))

    def test_pointing_openers(self):
        history = ["What is momentum?"]
        for question in ("And why?", "What about light?", "Tell me more", "its unit?"):
            self.assertTrue(qp.is_follow_up(question, history), question)

    def test_self_contained_questions_are_left_alone(self):
        history = ["What is momentum?"]
        for question in (
            "What is the formula for kinetic energy?",
            "How does refraction bend light in water?",
        ):
            self.assertFalse(qp.is_follow_up(question, history), question)

    def test_pronoun_with_its_own_subject_is_not_a_follow_up(self):
        self.assertFalse(
            qp.is_follow_up(
                "How is entropy related to heat and how is it measured in kelvin?",
                ["What is momentum?"],
            )
        )


class RewriteTests(unittest.TestCase):
    def test_heuristic_uses_most_recent_meaningful_turn(self):
        rewritten = qp.rewrite_standalone(
            "What about its unit?", ["What is entropy?", "What is momentum?", "thanks"]
        )
        self.assertEqual(rewritten, "What about its unit? (momentum)")

    def test_only_recent_turns_are_used(self):
        history = ["What is entropy?", "ok", "yes", "thanks"]
        self.assertEqual(qp.rewrite_standalone("And why?", history), "And why?")

    def test_rewriter_receives_history_and_question(self):
        prompts = []

        def rewriter(prompt):
            prompts.append(prompt)
            return '"What is the SI unit of momentum?"'

        rewritten = qp.rewrite_standalone("its unit?", ["What is momentum?"], rewriter)
        self.assertEqual(rewritten, "What is the SI unit of momentum?")
        self.assertIn("User: What is momentum?", prompts[0])
        self.assertIn("Last question: its unit?", prompts[0])

    def test_empty_rewriter_output_falls_back(self):
        rewritten = qp.rewrite_standalone("its unit?", ["What is momentum?"], lambda p: "  ")
        self.assertEqual(rewritten, "its unit? (momentum)")


class LanguageTests(unittest.TestCase):
    def test_language_is_detected_on_the_original_wording(self):
        model = StubLanguageModel()
        processed = qp.process_query(
            "Kí ni orí ninu asa Yoruba?", ["What is destiny?"], language_model=model
        )
        self.assertEqual(processed.language, "yor_Latn")
        self.assertEqual(model.seen[0], "Kí ni orí ninu asa Yoruba?")

    def test_too_short_original_falls_back_to_standalone(self):
        model = StubLanguageModel()
        processed = qp.process_query("And why?", ["What is momentum?"], language_model=model)
        self.assertEqual(processed.standalone, "And why? (momentum)")
        self.assertEqual(processed.language, "eng_Latn")


class ExpansionTests(unittest.TestCase):
    def test_no_expansion_unless_asked(self):
        processed = qp.process_query(
            "What is the formula for kinetic energy?", language_model=StubLanguageModel()
        )
        self.assertEqual(processed.expansions, [])
        self.assertEqual(processed.search_queries, [processed.standalone])

    def test_expander_failure_keeps_keyword_variant(self):
        def broken(prompt):
            raise TimeoutError

        self.assertEqual(
            qp.expand_query("What is the formula for kinetic energy?", broken),
            ["formula kinetic energy"],
        )

    def test_expansions_are_capped(self):
        many = lambda prompt: "\n".join(f"query {n}" for n in range(10))
        self.assertEqual(len(qp.expand_query("What is momentum?", many, max_expansions=2)), 2)


class RetrievalIntegrationTests(unittest.TestCase):
    """A processed query drives the real hybrid retriever."""

    def test_expansion_finds_passage_the_question_alone_misses(self):
        import keyword_index as ki
        import retrieval as ret
        import vector_base as vb

        records = [
            {"row": 0, "dataset": "rag", "chunk_id": "C0", "text": "Momentum is mass times velocity."},
            {"row": 1, "dataset": "rag", "chunk_id": "C1", "text": "The joule measures energy in SI units."},
            # A third passage: with only two, a word in one of them gets a
            # BM25 idf of exactly zero.
            {"row": 2, "dataset": "rag", "chunk_id": "C2", "text": "Light bends at a boundary."},
        ]
        base = vb.VectorBase(
            vb.normalize_rows(np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])),
            records,
            {"embedding_model": "stub"},
        )
        index = ki.KeywordIndex(records, [ki.tokenize(r["text"]) for r in records])
        encode = lambda text: np.array([1.0, 0.0], dtype="float32")
        search = lambda query: ret.hybrid_retrieve(base, index, query, encode, minimum_score=0.5)

        plain = qp.process_query("What is momentum?", language_model=StubLanguageModel())
        self.assertEqual([r.record["chunk_id"] for r in qp.retrieve_for_query(plain, search)], ["C0"])

        expanded = qp.process_query(
            "What is momentum?",
            language_model=StubLanguageModel(),
            expand=True,
            expander=lambda prompt: "joule unit of energy",
        )
        found = [r.record["chunk_id"] for r in qp.retrieve_for_query(expanded, search)]
        self.assertEqual(sorted(found), ["C0", "C1"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
