#!/usr/bin/env python3
"""
Test cases for retrieval.py

Run with either:
    python3 test_retrieval.py          (unittest, no extra installs)
    python3 -m pytest test_retrieval.py

Runs every docstring example plus query, filtering, formatting and CLI checks.
A keyword-based stub encoder replaces sentence-transformers, so nothing is
downloaded and the expected rankings are exact.
"""

from __future__ import annotations

import doctest
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

import retrieval as ret
import vector_base as vb


def load_tests(loader, tests, ignore):
    """Hook so `python3 test_retrieval.py` also runs every docstring example."""
    tests.addTests(doctest.DocTestSuite(ret))
    return tests


# Three orthogonal "meanings", so similarity is exactly predictable.
AXES = ("momentum", "light", "heat")

PASSAGES = [
    {
        "axis": "momentum",
        "record": {
            "row": 0,
            "dataset": "rag",
            "chunk_id": "CHUNK-1",
            "topic": "Mechanics",
            "concept": "Momentum",
            "formula": "p = mv",
            "chapter_title": "Chapter 3",
            "source_file": "physics.pdf",
            "source_page_start": 40,
            "source_page_end": 52,
            "text": "Momentum is the product of mass and velocity.",
        },
    },
    {
        "axis": "light",
        "record": {
            "row": 1,
            "dataset": "rag",
            "chunk_id": "CHUNK-2",
            "topic": "Optics",
            "concept": "Refraction",
            "chapter_title": "Chapter 9",
            "source_file": "physics.pdf",
            "source_page_start": 120,
            "text": "Refraction bends light as it crosses a boundary.",
        },
    },
    {
        "axis": "heat",
        "record": {
            "row": 2,
            "dataset": "llm",
            "record_id": "CE-1",
            "topic": "Thermodynamics",
            "concept": "Entropy",
            "chapter_title": "Chapter 12",
            "source_file": "thermo.pdf",
            "source_page_start": 8,
            "text": "Entropy measures how much energy is unavailable for work.",
        },
    },
    {
        "axis": "momentum",
        "record": {
            "row": 3,
            "dataset": "llm",
            "record_id": "QA-1",
            "topic": "Mechanics",
            "concept": "Impulse",
            "chapter_title": "Chapter 3",
            "source_file": "notes.pdf",
            "text": "Impulse is the change in momentum over an interval.",
        },
    },
]


def axis_vector(axis: str, jitter: float = 0.0) -> np.ndarray:
    vector = np.zeros(len(AXES), dtype="float32")
    vector[AXES.index(axis)] = 1.0

    if jitter:
        vector[(AXES.index(axis) + 1) % len(AXES)] = jitter

    return vector


def stub_encoder(text: str) -> np.ndarray:
    """Encode by keyword: a query mentioning an axis points along that axis."""
    vector = np.zeros(len(AXES), dtype="float32")

    for index, axis in enumerate(AXES):
        if axis in text.casefold():
            vector[index] = 1.0

    if not vector.any():
        vector[:] = 1.0

    return vector


def make_base(manifest: dict | None = None) -> vb.VectorBase:
    vectors = vb.normalize_rows(
        np.array(
            [axis_vector(p["axis"], jitter=0.1 * index) for index, p in enumerate(PASSAGES)],
            dtype="float32",
        )
    )
    records = [dict(p["record"]) for p in PASSAGES]

    return vb.VectorBase(
        vectors,
        records,
        manifest if manifest is not None else {"embedding_model": "stub-model"},
    )


class ModelCheckTests(unittest.TestCase):
    def test_matching_model_is_silent(self):
        self.assertIsNone(ret.check_model_match(make_base(), "stub-model"))

    def test_mismatch_raises_by_default(self):
        with self.assertRaises(ValueError) as caught:
            ret.check_model_match(make_base(), "other-model")
        self.assertIn("--allow-model-mismatch", str(caught.exception))

    def test_mismatch_can_be_overridden(self):
        warning = ret.check_model_match(
            make_base(), "other-model", allow_mismatch=True
        )
        self.assertIn("Model mismatch ignored", warning)

    def test_missing_model_in_manifest_warns(self):
        warning = ret.check_model_match(make_base({}), "stub-model")
        self.assertIn("does not record", warning)


class FilterParsingTests(unittest.TestCase):
    def test_pairs_and_repeats(self):
        self.assertEqual(ret.parse_filter_arguments([]), {})
        self.assertEqual(
            ret.parse_filter_arguments(["topic=Mechanics", "topic=Optics"]),
            {"topic": ["Mechanics", "Optics"]},
        )
        self.assertEqual(
            ret.parse_filter_arguments(["dataset=rag", "source_file=a.pdf"]),
            {"dataset": "rag", "source_file": "a.pdf"},
        )

    def test_values_containing_equals_signs_survive(self):
        self.assertEqual(
            ret.parse_filter_arguments(["formula=p = mv"]), {"formula": "p = mv"}
        )

    def test_malformed_filter_raises(self):
        with self.assertRaises(ValueError):
            ret.parse_filter_arguments(["topic"])


class RetrieveTests(unittest.TestCase):
    def setUp(self):
        self.base = make_base()

    def retrieve(self, query, **kwargs):
        options = dict(encode=stub_encoder, minimum_score=0.0)
        options.update(kwargs)
        return ret.retrieve(self.base, query, **options)

    def test_query_finds_the_matching_axis(self):
        results = self.retrieve("what is momentum", top_k=2)
        self.assertEqual(
            {r.record["concept"] for r in results}, {"Momentum", "Impulse"}
        )

    def test_results_are_ordered_by_score(self):
        results = self.retrieve("heat", top_k=4)
        scores = [r.score for r in results]
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertEqual(results[0].record["concept"], "Entropy")

    def test_filters_are_applied(self):
        results = self.retrieve("momentum", top_k=4, filters={"dataset": "llm"})
        self.assertEqual({r.record["dataset"] for r in results}, {"llm"})

        results = self.retrieve(
            "momentum", top_k=4, filters={"source_file": "notes.pdf"}
        )
        self.assertEqual([r.record["record_id"] for r in results], ["QA-1"])

    def test_minimum_score_filters_weak_matches(self):
        self.assertEqual(self.retrieve("light", top_k=4, minimum_score=0.99)[0].row, 1)
        self.assertEqual(len(self.retrieve("light", top_k=4, minimum_score=0.99)), 1)

    def test_no_match_returns_empty_list(self):
        self.assertEqual(
            self.retrieve("momentum", filters={"topic": "Geology"}), []
        )

    def test_empty_query_raises(self):
        for query in ("", "   ", None):
            with self.assertRaises(ValueError):
                self.retrieve(query)

    def test_text_is_returned_with_the_result(self):
        result = self.retrieve("momentum", top_k=1)[0]
        self.assertEqual(result.text, result.record["text"])
        self.assertTrue(result.text)

    def test_diversity_weight_is_passed_through(self):
        focused = self.retrieve("momentum", top_k=2, relevance_weight=1.0)
        varied = self.retrieve("momentum", top_k=2, relevance_weight=0.0)
        self.assertNotEqual([r.row for r in focused], [r.row for r in varied])


class FormattingTests(unittest.TestCase):
    def setUp(self):
        self.base = make_base()
        self.results = ret.retrieve(
            self.base, "momentum", stub_encoder, top_k=2, minimum_score=0.0
        )

    def test_text_format_has_ranks_scores_and_citations(self):
        rendered = ret.format_results("momentum", self.results)
        self.assertIn("Query: momentum", rendered)
        self.assertIn("Results: 2", rendered)
        self.assertIn("1. [", rendered)
        self.assertIn("physics.pdf | Chapter 3 | pp. 40-52", rendered)

    def test_empty_results_explain_what_to_try(self):
        rendered = ret.format_results("nothing", [])
        self.assertIn("No matching passages", rendered)
        self.assertIn("--minimum-score", rendered)

    def test_citation_shapes(self):
        self.assertEqual(
            ret.describe_source(
                {"source_file": "a.pdf", "source_page_start": 3, "source_page_end": 3}
            ),
            "a.pdf | p. 3",
        )
        self.assertEqual(ret.describe_source({"chapter_title": "Ch 1"}), "Ch 1")
        self.assertEqual(ret.describe_source({}), "unknown source")

    def test_context_block_is_citation_labelled(self):
        context = ret.build_context(self.results)
        self.assertTrue(context.startswith("[1] "))
        self.assertIn(self.results[0].text, context)
        self.assertIn("[2] ", context)

    def test_context_respects_the_budget(self):
        context = ret.build_context(self.results, character_budget=10)
        self.assertNotIn("[2] ", context)

    def test_context_skips_records_without_text(self):
        for record in self.base.records:
            record["text"] = ""
        results = ret.retrieve(
            self.base, "momentum", stub_encoder, top_k=2, minimum_score=0.0
        )
        self.assertEqual(ret.build_context(results), "")

    def test_json_output_is_machine_readable(self):
        payload = json.loads(ret.results_as_json("momentum", self.results))
        self.assertEqual(payload["query"], "momentum")
        self.assertEqual(payload["result_count"], 2)
        self.assertEqual(payload["source"], vb.SOURCE_LABEL)
        first = payload["results"][0]
        self.assertEqual(first["rank"], 1)
        self.assertIn("citation", first)
        self.assertIn("text", first["record"])

    def test_run_query_formats(self):
        for output_format, marker in (
            ("text", "Results:"),
            ("context", "[1] "),
            ("json", '"result_count"'),
        ):
            rendered = ret.run_query(
                self.base,
                "momentum",
                stub_encoder,
                top_k=2,
                minimum_score=0.0,
                output_format=output_format,
            )
            self.assertIn(marker, rendered, output_format)


class InteractiveTests(unittest.TestCase):
    def test_loop_answers_until_quit(self):
        typed = iter(["momentum", "", "heat", "quit", "never reached"])
        printed: list[str] = []
        answered = ret.interactive_loop(
            make_base(),
            stub_encoder,
            input_function=lambda prompt: next(typed),
            output_function=printed.append,
        )
        self.assertEqual(answered, 2)
        self.assertEqual(len(printed), 2)

    def test_end_of_input_stops_cleanly(self):
        def raise_eof(prompt):
            raise EOFError

        self.assertEqual(
            ret.interactive_loop(make_base(), stub_encoder, input_function=raise_eof),
            0,
        )


class EndToEndTests(unittest.TestCase):
    """Build a base on disk with vector_base, then query it with retrieval."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.folder = Path(self.tmp.name) / "base"
        make_base().save(self.folder)

    def tearDown(self):
        self.tmp.cleanup()

    def test_saved_base_can_be_queried(self):
        base = vb.VectorBase.load(self.folder)
        self.assertEqual(len(base), len(PASSAGES))
        self.assertIsNone(ret.check_model_match(base, "stub-model"))

        results = ret.retrieve(base, "heat", stub_encoder, top_k=1, minimum_score=0.0)
        self.assertEqual(results[0].record["concept"], "Entropy")
        self.assertIn("Entropy measures", results[0].text)

    def test_filters_work_against_the_loaded_base(self):
        base = vb.VectorBase.load(self.folder)
        results = ret.retrieve(
            base,
            "momentum",
            stub_encoder,
            top_k=5,
            filters={"topic": "Mechanics", "dataset": "rag"},
            minimum_score=0.0,
        )
        self.assertEqual([r.record["chunk_id"] for r in results], ["CHUNK-1"])


class CommandLineTests(unittest.TestCase):
    def test_defaults(self):
        options = ret.parse_arguments(["--query", "x"])
        self.assertEqual(options.top_k, vb.DEFAULT_TOP_K)
        self.assertEqual(options.format, "text")
        self.assertFalse(options.allow_model_mismatch)
        self.assertEqual(options.model, ret.DEFAULT_MODEL_NAME)

    def test_output_formats_are_mutually_exclusive(self):
        self.assertEqual(ret.parse_arguments(["--query", "x", "--json"]).format, "json")
        self.assertEqual(
            ret.parse_arguments(["--query", "x", "--context"]).format, "context"
        )
        with self.assertRaises(SystemExit):
            ret.parse_arguments(["--query", "x", "--json", "--context"])

    def test_repeatable_filters(self):
        options = ret.parse_arguments(
            ["--query", "x", "--filter", "topic=A", "--filter", "topic=B"]
        )
        self.assertEqual(options.filter, ["topic=A", "topic=B"])

    def test_interactive_flag(self):
        self.assertTrue(ret.parse_arguments(["--interactive"]).interactive)


class EncoderTests(unittest.TestCase):
    def test_missing_dependency_is_reported(self):
        if importlib.util.find_spec("sentence_transformers") is not None:
            self.skipTest("sentence-transformers is installed")
        with self.assertRaises(RuntimeError) as caught:
            ret.load_query_encoder()
        self.assertIn("pip install sentence-transformers", str(caught.exception))


class HybridRetrieveTests(unittest.TestCase):
    """Vector + BM25 fusion, filters, date range and optional reranking."""

    def setUp(self):
        import keyword_index as ki

        self.base = make_base()
        dates = ["2019-05-01", "2023-02-10", "2024-07-15", "2025-01-03"]
        for record, date in zip(self.base.records, dates):
            record["date"] = date
            record["language"] = "eng_Latn"
        self.base.records[1]["language"] = "yor_Latn"

        self.index = ki.KeywordIndex(
            self.base.records, [ki.tokenize(r["text"]) for r in self.base.records]
        )

    def hybrid(self, query, **kwargs):
        options = dict(encode=stub_encoder, minimum_score=0.0)
        options.update(kwargs)
        return ret.hybrid_retrieve(self.base, self.index, query, **options)

    def test_keyword_side_rescues_what_vector_side_misses(self):
        # The stub encoder knows nothing about "refraction", so with a
        # strict cosine floor the vector side returns nothing at all.
        self.assertEqual(
            ret.retrieve(self.base, "refraction", stub_encoder, minimum_score=0.9), []
        )
        results = self.hybrid("refraction", minimum_score=0.9)
        self.assertEqual([r.record["chunk_id"] for r in results], ["CHUNK-2"])

    def test_passage_found_by_both_sides_ranks_first(self):
        results = self.hybrid("impulse momentum", top_k=4)
        self.assertEqual(results[0].record["record_id"], "QA-1")
        scores = [r.score for r in results]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_each_passage_appears_once(self):
        results = self.hybrid("momentum", top_k=10)
        keys = [ret.result_key(r) for r in results]
        self.assertEqual(len(keys), len(set(keys)))

    def test_field_filters_apply_to_both_indexes(self):
        results = self.hybrid("refraction light", top_k=4, filters={"language": "eng_Latn"})
        self.assertNotIn("CHUNK-2", [r.record.get("chunk_id") for r in results])

        results = self.hybrid("refraction light", top_k=4, filters={"language": "yor_Latn"})
        self.assertEqual([r.record["chunk_id"] for r in results], ["CHUNK-2"])

    def test_date_range_is_inclusive(self):
        results = self.hybrid("momentum", top_k=4, date_from="2025-01-03")
        self.assertEqual([r.record["record_id"] for r in results], ["QA-1"])

        results = self.hybrid("momentum", top_k=4, date_to="2019-05-01")
        self.assertEqual([r.record["chunk_id"] for r in results], ["CHUNK-1"])

    def test_reranker_reorders_and_truncates(self):
        class PrefersImpulse:
            def predict(self, pairs):
                return [2.0 if "Impulse" in text else 0.5 for _, text in pairs]

        results = self.hybrid("momentum", top_k=1, reranker_model=PrefersImpulse())
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].record["record_id"], "QA-1")
        self.assertEqual(results[0].row, 3)
        self.assertEqual(results[0].score, 2.0)

    def test_run_query_switches_to_hybrid_with_an_index(self):
        payload = json.loads(
            ret.run_query(
                self.base,
                "refraction",
                stub_encoder,
                minimum_score=0.9,
                output_format="json",
                keyword_index=self.index,
            )
        )
        self.assertEqual(payload["result_count"], 1)

    def test_empty_query_raises(self):
        with self.assertRaises(ValueError):
            self.hybrid("  ")


if __name__ == "__main__":
    unittest.main(verbosity=2)
