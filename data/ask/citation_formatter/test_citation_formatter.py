#!/usr/bin/env python3
"""
Test cases for citation_formatter.py

Run with either:
    python3 test_citation_formatter.py          (unittest, no extra installs)
    python3 -m pytest test_citation_formatter.py

Runs every docstring example plus parsing, checking and formatting tests,
and an end-to-end check that reads a registry written by the real
source_registry.py and passages built by the real prompt_builder.py.
"""

from __future__ import annotations

import doctest
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import citation_formatter as cf

HERE = Path(cf.__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "prompt_builder"))
sys.path.insert(0, str(HERE.parent.parent.parent / "source" / "source_registry_program"))


def load_tests(loader, tests, ignore):
    """Hook so `python3 test_citation_formatter.py` also runs every docstring example."""
    tests.addTests(doctest.DocTestSuite(cf))
    return tests


def passage(number: int, **record) -> SimpleNamespace:
    return SimpleNamespace(number=number, record=record)


class ExtractTests(unittest.TestCase):
    def test_spaces_and_en_dash_ranges(self):
        self.assertEqual(cf.extract_citations("x [ 1 ,2 ] y"), [])
        self.assertEqual(cf.extract_citations("x [1 ,2] y [3 – 4]"), [1, 2, 3, 4])

    def test_absurd_ranges_are_ignored(self):
        self.assertEqual(cf.extract_citations("[5-2] [1-999]"), [])

    def test_years_in_brackets_are_numbers_not_skipped(self):
        # A bracketed year is indistinguishable from a citation; the check
        # then flags it as invalid rather than silently accepting it.
        report = cf.check_citations("Published [1998].", [passage(1)])
        self.assertEqual(report.invalid, [1998])


class CheckTests(unittest.TestCase):
    def setUp(self):
        self.passages = [passage(n) for n in (1, 2, 3)]

    def test_all_valid(self):
        report = cf.check_citations("A [1]. B [2][3].", self.passages)
        self.assertEqual((report.cited, report.invalid, report.uncited), ([1, 2, 3], [], []))
        self.assertTrue(report.ok)

    def test_cited_order_is_first_use(self):
        self.assertEqual(cf.check_citations("B [3]. A [1]. B [3].", self.passages).cited, [3, 1])

    def test_no_passages_makes_every_citation_invalid(self):
        report = cf.check_citations("A [1].", [])
        self.assertEqual(report.invalid, [1])
        self.assertFalse(report.ok)

    def test_refusal_wording_variants(self):
        for text in (
            "The provided sources do not contain the answer to this question.",
            "Sorry, the source doesn't contain the answer.",
        ):
            self.assertTrue(cf.check_citations(text, self.passages).says_not_in_sources, text)


class FormatTests(unittest.TestCase):
    def test_keep_invalid_leaves_text_alone(self):
        formatted = cf.format_answer(
            "A [1] and B [7].", [passage(1, source_file="a.pdf")], lookup={}, drop_invalid=False
        )
        self.assertEqual(formatted.text, "A [1] and B [7].")
        self.assertEqual(formatted.report.invalid, [7])

    def test_references_only_for_cited_sources_in_number_order(self):
        passages = [passage(n, source_file=f"book{n}_cleaned.json", chunk_id=f"C{n}") for n in (1, 2, 3)]
        formatted = cf.format_answer("X [3]. Y [1].", passages, lookup={})
        self.assertEqual([r["number"] for r in formatted.references], [1, 3])
        self.assertEqual(formatted.references[1]["reference"], "book3")

    def test_llm_record_falls_back_to_record_id(self):
        formatted = cf.format_answer(
            "X [1].", [passage(1, source_file="b_cleaned.json", record_id="QA-1")], lookup={}
        )
        self.assertEqual(formatted.references[0]["chunk_id"], "QA-1")

    def test_answer_without_citations_renders_without_sources(self):
        formatted = cf.format_answer("No idea.", [passage(1, source_file="a.pdf")], lookup={})
        self.assertEqual(formatted.render(), "No idea.")
        self.assertFalse(formatted.report.ok)


class PipelineIntegrationTests(unittest.TestCase):
    """Registry from source_registry.py + passages from prompt_builder.py."""

    def test_registry_and_prompt_passages_produce_full_references(self):
        import prompt_builder as pb
        import source_registry as sr
        from vector_base import SearchResult

        with tempfile.TemporaryDirectory() as folder:
            registry_path = Path(folder) / "source_registry.json"
            registry: dict = {}
            sr.add_or_update_source(
                registry,
                title="Time and Social Anthropology",
                author="Alfred Gell",
                author_country="United Kingdom",
                is_african_authored=False,
                year=1998,
                publisher="Berg",
                language="English",
                isbn=None,
                licence_status="cleared",
                licence_notes="",
                source_file="Gell_1998.pdf",
            )
            sr.save_registry(registry_path, registry)

            manifest_path = Path(folder) / "web_ingestion_manifest.json"
            manifest_path.write_text(json.dumps({"known_pages": [
                {"url": "https://example.org/ori", "output_file": "WEB-9.txt",
                 "content_hash": "h", "fetch_timestamp": "2026-10-01T00:00:00+00:00"}
            ]}), encoding="utf-8")

            lookup = cf.load_source_lookup(registry_path, manifest_path)

        results = [
            SearchResult(1, 0.9, 0, "Time is socially constructed.",
                         {"chunk_id": "C1", "source_file": "Gell_1998_cleaned.json",
                          "source_page_start": 47, "source_page_end": 49}),
            SearchResult(2, 0.8, 1, "Ori is the seat of destiny.",
                         {"chunk_id": "C2", "source_file": "WEB-9_cleaned.json"}),
        ]
        package = pb.build_prompt("What is time?", results)
        formatted = cf.format_answer(
            "Time is social [1]. Ori relates to destiny [2]. Also [5].",
            package.passages,
            lookup,
        )

        self.assertEqual(formatted.text, "Time is social [1]. Ori relates to destiny [2]. Also.")
        self.assertEqual(
            [r["reference"] for r in formatted.references],
            ["Time and Social Anthropology, Alfred Gell, pp. 47–49", "https://example.org/ori"],
        )
        self.assertEqual(formatted.report.invalid, [5])


if __name__ == "__main__":
    unittest.main(verbosity=2)
