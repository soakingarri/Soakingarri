#!/usr/bin/env python3
"""
Test cases for language_detection.py

Run with either:
    python3 test_language_detection.py          (unittest, no extra installs)
    python3 -m pytest test_language_detection.py

Every function in language_detection.py carries doctest examples in its
docstring; this file runs those plus end-to-end checks against a stub
model, so nothing is downloaded. A real GlotLID check is included too, but
it is opt-in (set SOAKINGARRI_REAL_MODELS=1) because it downloads ~1.7 GB.
"""

from __future__ import annotations

import doctest
import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path

import language_detection as ld


def load_tests(loader, tests, ignore):
    """Hook so `python3 test_language_detection.py` also runs every docstring example."""
    tests.addTests(doctest.DocTestSuite(ld))
    return tests


YORUBA_WORDS = {"ni", "ati", "awon", "ori"}
HAUSA_WORDS = {"da", "shi", "yana", "ina"}


class StubModel:
    """Label by marker words; mimics fastText's one-line rule and output shape."""

    def __init__(self):
        self.calls = 0

    def predict(self, text, k=1):
        if "\n" in text:
            raise ValueError("predict processes one line at a time")

        self.calls += 1
        words = set(text.casefold().replace(",", " ").replace(".", " ").split())

        if words & YORUBA_WORDS:
            return (("__label__yor_Latn",), (0.96,))

        if words & HAUSA_WORDS:
            return (("__label__hau_Latn",), (0.93,))

        if "mumble" in words:
            return (("__label__eng_Latn",), (0.2,))

        return (("__label__eng_Latn",), (0.99,))


def make_document() -> dict:
    return {
        "document_id": "DOC-1",
        "source_file": "mixed.pdf",
        "chapters": [
            {
                "chapter_id": "CH-1",
                "title": "Ori",
                "paragraphs": [
                    {"paragraph_id": "P-1", "page": 3,
                     "text": "Ori ni ipin eniyan ati ayanmo ninu asa Yoruba."},
                    {"paragraph_id": "P-2", "page": 3, "text": "Ori ni."},
                    {"paragraph_id": "P-3", "page": 4,
                     "text": "mumble mumble mumble mumble mumble"},
                ],
            },
            {
                "chapter_id": "CH-2",
                "title": "Translation",
                "paragraphs": [
                    {"paragraph_id": "P-4", "page": 5,
                     "text": "The head is the seat of a person's destiny in Yoruba thought, "
                             "and this chapter discusses it at some length."},
                    {"paragraph_id": "P-5", "page": 5, "text": "Figure 2."},
                ],
            },
        ],
    }


class DetectLanguageTests(unittest.TestCase):
    def test_multiline_text_is_collapsed_to_one_line(self):
        label, confidence = ld.detect_language(
            "Ori ni ipin eniyan\nati ayanmo.", StubModel()
        )
        self.assertEqual((label, confidence), ("yor_Latn", 0.96))

    def test_short_text_is_never_sent_to_the_model(self):
        model = StubModel()
        self.assertEqual(ld.detect_language("Fig. 2", model), ("und", 0.0))
        self.assertEqual(model.calls, 0)

    def test_low_confidence_is_undetermined(self):
        label, confidence = ld.detect_language(
            "mumble mumble mumble mumble", StubModel()
        )
        self.assertEqual(label, "und")
        self.assertEqual(confidence, 0.2)


class TagDocumentTests(unittest.TestCase):
    def setUp(self):
        self.document = make_document()
        self.tagged = ld.tag_document(self.document, StubModel())

    def paragraphs(self):
        return [
            paragraph
            for chapter in self.tagged["chapters"]
            for paragraph in chapter["paragraphs"]
        ]

    def test_every_paragraph_gets_a_real_language(self):
        for paragraph in self.paragraphs():
            self.assertNotEqual(paragraph["language"], "und")
            self.assertIn(paragraph["language_source"], {"glotlid", "document"})

    def test_uncertain_paragraphs_inherit_their_chapters_language(self):
        by_id = {p["paragraph_id"]: p for p in self.paragraphs()}
        self.assertEqual(by_id["P-2"]["language"], "yor_Latn")
        self.assertEqual(by_id["P-2"]["language_source"], "document")
        self.assertEqual(by_id["P-3"]["language"], "yor_Latn")
        self.assertEqual(by_id["P-5"]["language"], "eng_Latn")

    def test_chapter_and_document_languages(self):
        self.assertEqual(
            [c["language"] for c in self.tagged["chapters"]], ["yor_Latn", "eng_Latn"]
        )
        # The English chapter has more confidently detected characters.
        self.assertEqual(self.tagged["language"], "eng_Latn")

    def test_existing_fields_are_preserved(self):
        by_id = {p["paragraph_id"]: p for p in self.paragraphs()}
        self.assertEqual(by_id["P-4"]["page"], 5)
        self.assertEqual(self.tagged["source_file"], "mixed.pdf")
        self.assertEqual(self.tagged["chapters"][0]["title"], "Ori")

    def test_input_document_is_not_modified(self):
        self.assertEqual(self.document, make_document())

    def test_chapter_with_nothing_detectable_uses_document_language(self):
        document = {"chapters": [
            {"paragraphs": ["Yana da kyau sosai, ina son shi da yawa."]},
            {"paragraphs": ["Table 1."]},
        ]}
        tagged = ld.tag_document(document, StubModel())
        self.assertEqual(tagged["chapters"][1]["language"], "hau_Latn")
        self.assertEqual(tagged["chapters"][1]["paragraphs"][0]["language"], "hau_Latn")

    def test_all_undetermined_document_stays_undetermined(self):
        tagged = ld.tag_document({"chapters": [{"paragraphs": ["Fig. 1"]}]}, StubModel())
        self.assertEqual(tagged["language"], "und")


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.inp = root / "deduplicated_data"
        self.out = root / "language_tagged_data"
        (self.inp / "nested").mkdir(parents=True)
        (self.inp / "mixed_cleaned.json").write_text(
            json.dumps(make_document()), encoding="utf-8"
        )
        with (self.inp / "nested" / "notes_cleaned.jsonl").open("w", encoding="utf-8") as f:
            f.write(json.dumps({"text": "Ina son karatu da rubutu kowace rana."}) + "\n")
            f.write(json.dumps(make_document()) + "\n")
        (self.inp / "deduplication_manifest.json").write_text("{}", encoding="utf-8")

    def tearDown(self):
        self.tmp.cleanup()

    def test_end_to_end(self):
        manifest = ld.tag_corpus(self.inp, self.out, model=StubModel())

        self.assertEqual(manifest["file_count"], 2)
        self.assertEqual(manifest["document_count"], 3)
        self.assertEqual(manifest["paragraph_count"], 10)
        self.assertEqual(manifest["failed_files"], [])
        self.assertEqual(sum(manifest["paragraphs_by_language"].values()), 10)
        self.assertNotIn("und", manifest["paragraphs_by_language"])

        tagged = json.loads((self.out / "mixed_cleaned.json").read_text("utf-8"))
        self.assertEqual(tagged["document_id"], "DOC-1")

        lines = (self.out / "nested" / "notes_cleaned.jsonl").read_text("utf-8").splitlines()
        self.assertEqual(len(lines), 2)
        self.assertEqual(json.loads(lines[0])["language"], "hau_Latn")

        self.assertFalse(list(self.out.rglob("*.tmp")))
        stored = json.loads((self.out / ld.MANIFEST_FILE_NAME).read_text("utf-8"))
        self.assertEqual(stored["document_count"], 3)

    def test_unreadable_file_is_reported_not_fatal(self):
        (self.inp / "broken_cleaned.jsonl").write_text("{not json\n", encoding="utf-8")
        manifest = ld.tag_corpus(self.inp, self.out, model=StubModel())
        self.assertEqual(len(manifest["failed_files"]), 1)
        self.assertEqual(manifest["document_count"], 3)

    def test_rerun_is_idempotent(self):
        first = ld.tag_corpus(self.inp, self.out, model=StubModel())
        second = ld.tag_corpus(self.inp, self.out, model=StubModel())
        self.assertEqual(first, second)


class LoadModelDependencyTests(unittest.TestCase):
    @unittest.skipIf(
        importlib.util.find_spec("fasttext") is not None
        and importlib.util.find_spec("huggingface_hub") is not None,
        "fasttext and huggingface_hub are installed; dependency error path not reachable",
    )
    def test_raises_runtime_error_without_dependencies(self):
        with self.assertRaises(RuntimeError):
            ld.load_language_model()


@unittest.skipUnless(
    os.environ.get("SOAKINGARRI_REAL_MODELS") == "1",
    "real-model test is opt-in: set SOAKINGARRI_REAL_MODELS=1 to run it",
)
class RealGlotLIDTests(unittest.TestCase):
    """Downloads GlotLID (~1.7 GB) on first run."""

    @classmethod
    def setUpClass(cls):
        cls.model = ld.load_language_model()

    def test_english_and_yoruba(self):
        self.assertEqual(
            ld.detect_language("This is a sentence written in plain English.", self.model)[0],
            "eng_Latn",
        )
        self.assertEqual(
            ld.detect_language(
                "Ẹ kú àárọ̀, ṣé dáadáa ni? Mo fẹ́ lọ sí ọjà lónìí.", self.model
            )[0],
            "yor_Latn",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
