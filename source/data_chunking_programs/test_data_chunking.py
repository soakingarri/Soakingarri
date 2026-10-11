#!/usr/bin/env python3
"""
Test cases for data_chunking.py

Run with either:
    python3 test_data_chunking.py          (unittest, no extra installs)
    python3 -m pytest test_data_chunking.py

Every function in data_chunking.py carries doctest examples in its docstring;
this file runs all of those plus end-to-end checks. data_chunking.py does not
embed anything itself (that's embedding_program/embedding.py's job), so these
tests never need sentence-transformers.
"""

from __future__ import annotations

import doctest
import json
import tempfile
import unittest
from pathlib import Path

import data_chunking as dcx


def load_tests(loader, tests, ignore):
    """Hook so `python3 test_data_chunking.py` also runs every docstring example."""
    tests.addTests(doctest.DocTestSuite(dcx))
    return tests


PHYSICS_PARAGRAPH = (
    "Kinetic energy is the energy of motion, given by KE = 1/2 m v^2. "
    "Therefore doubling the velocity quadruples the kinetic energy. "
    "For example, a car at 60 km/h has four times the kinetic energy it "
    "has at 30 km/h. This is why braking distance grows so quickly. "
)


def make_cleaned_document(paragraph_count: int = 6, title: str = "Chapter 5: Energy") -> dict:
    """Mimic the JSON emitted by data_cleaning.py."""
    return {
        "document_id": "DOC-abc123",
        "source_file": "physics.pdf",
        "source_path": "/raw/physics.pdf",
        "source_format": "pdf",
        "chapters": [
            {
                "chapter_id": "CH-1",
                "title": title,
                "source_page_start": 40,
                "source_page_end": 44,
                "paragraphs": [
                    {"paragraph_id": f"P-{i}", "text": PHYSICS_PARAGRAPH * 2}
                    for i in range(paragraph_count)
                ],
            }
        ],
    }


class ChunkingTests(unittest.TestCase):
    def test_chunks_respect_target_and_minimum(self):
        doc = make_cleaned_document(paragraph_count=10)
        chunks = list(dcx.build_chunks(doc, Path("physics_cleaned.json")))
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertGreaterEqual(chunk.word_count, dcx.MIN_CHUNK_WORDS)
            # a chunk may exceed the target only by the overlap plus one paragraph
            self.assertLessEqual(
                chunk.word_count,
                dcx.TARGET_CHUNK_WORDS + dcx.CHUNK_OVERLAP_WORDS + dcx.word_count(PHYSICS_PARAGRAPH * 2),
            )

    def test_chunk_ids_are_stable_and_unique(self):
        doc = make_cleaned_document()
        first = [c.chunk_id for c in dcx.build_chunks(doc, Path("x.json"))]
        second = [c.chunk_id for c in dcx.build_chunks(doc, Path("x.json"))]
        self.assertEqual(first, second)
        self.assertEqual(len(first), len(set(first)))

    def test_annotations_are_filled(self):
        doc = make_cleaned_document()
        chunk = next(dcx.build_chunks(doc, Path("x.json")))
        self.assertEqual(chunk.topic, "Chapter 5: Energy")
        self.assertEqual(chunk.concept, "Kinetic energy")
        self.assertEqual(chunk.formula, "KE = 1/2 m v^2")
        self.assertTrue(chunk.derivation.startswith("Therefore"))
        self.assertTrue(chunk.example.startswith("For example"))
        self.assertEqual(chunk.source_document_id, "DOC-abc123")
        self.assertEqual(chunk.source_path, "/raw/physics.pdf")
        self.assertEqual((chunk.source_page_start, chunk.source_page_end), (40, 44))

    def test_chunk_page_range_narrows_to_its_own_paragraphs(self):
        # The chapter spans pages 40-44, but each individual paragraph now
        # carries its own page (as data_cleaning.py produces); a chunk
        # built from only the first two paragraphs should cite just their
        # pages (40-41), not the whole chapter's wider 40-44 range.
        doc = make_cleaned_document(paragraph_count=2, title="Chapter 5: Energy")
        doc["chapters"][0]["paragraphs"][0]["page"] = 40
        doc["chapters"][0]["paragraphs"][1]["page"] = 41
        chunk = next(dcx.build_chunks(doc, Path("x.json"), target_words=10_000))
        self.assertEqual((chunk.source_page_start, chunk.source_page_end), (40, 41))

    def test_language_tags_flow_into_chunks_and_records(self):
        # language_detection.py tags each paragraph; the chunk, its RAG
        # record and its LLM records must all carry that label so retrieval
        # can filter on it.
        doc = make_cleaned_document(paragraph_count=2)
        for paragraph in doc["chapters"][0]["paragraphs"]:
            paragraph["language"] = "yor_Latn"
        chunk = next(dcx.build_chunks(doc, Path("x.json"), target_words=10_000))
        self.assertEqual(chunk.language, "yor_Latn")
        self.assertEqual(
            dcx.build_rag_record(chunk)["source_metadata"]["language"], "yor_Latn"
        )
        self.assertEqual(
            {r["language"] for r in dcx.build_llm_records(chunk)}, {"yor_Latn"}
        )

    def test_document_filter_fields_flow_into_records(self):
        # data_cleaning.py sets tool/source_type/date per document; every
        # chunk and record from that document must carry them.
        doc = make_cleaned_document()
        doc.update({"tool": "ask", "source_type": "web", "date": "2026-10-01"})
        chunk = next(dcx.build_chunks(doc, Path("x.json")))
        self.assertEqual((chunk.tool, chunk.source_type, chunk.date), ("ask", "web", "2026-10-01"))
        metadata = dcx.build_rag_record(chunk)["source_metadata"]
        self.assertEqual((metadata["tool"], metadata["source_type"]), ("ask", "web"))
        self.assertEqual({r["date"] for r in dcx.build_llm_records(chunk)}, {"2026-10-01"})

    def test_untagged_input_has_no_language(self):
        chunk = next(dcx.build_chunks(make_cleaned_document(), Path("x.json")))
        self.assertIsNone(chunk.language)

    def test_flat_text_document_is_supported(self):
        doc = {"title": "Essay", "text": PHYSICS_PARAGRAPH * 4}
        chunks = list(dcx.build_chunks(doc, Path("essay.json")))
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].chapter_title, "Essay")
        self.assertEqual(chunks[0].paragraph_ids, [])

    def test_tiny_document_produces_no_chunks(self):
        doc = {"chapters": [{"title": "T", "paragraphs": ["Too short to keep."]}]}
        self.assertEqual(list(dcx.build_chunks(doc, Path("t.json"))), [])


class RecordTests(unittest.TestCase):
    def setUp(self):
        self.chunk = next(dcx.build_chunks(make_cleaned_document(), Path("x.json")))

    def test_llm_records_are_json_serialisable_and_labelled(self):
        for record in dcx.build_llm_records(self.chunk):
            json.dumps(record)
            self.assertEqual(record["source"], dcx.SOURCE_LABEL)
            self.assertEqual(record["chunk_id"], self.chunk.chunk_id)
            self.assertTrue(record["record_id"])
            self.assertEqual(record["generation_method"], "template")

    def test_answer_is_grounded_in_the_chunks_own_content_not_its_opening(self):
        # PHYSICS_PARAGRAPH's formula/derivation/example sentences are not
        # its first sentence, so an answer that actually targets the
        # question (about the formula) looks different from just taking
        # the chunk's leading words.
        qa_record = next(
            r for r in dcx.build_llm_records(self.chunk)
            if r["type"] == "question_reasoning_answer"
        )
        self.assertIn(self.chunk.formula, qa_record["answer"])
        self.assertNotEqual(qa_record["answer"], dcx.infer_answer(self.chunk.text))

    def test_reasoning_quotes_the_passages_own_derivation(self):
        qa_record = next(
            r for r in dcx.build_llm_records(self.chunk)
            if r["type"] == "question_reasoning_answer"
        )
        self.assertIn(self.chunk.derivation, qa_record["reasoning"])

    def test_rag_record_has_no_embedding(self):
        # Embedding is embedding_program/embedding.py's job, done exactly
        # once; a RAG record out of data_chunking.py must carry text and
        # metadata only, never a vector, or a chunk ends up embedded twice.
        record = dcx.build_rag_record(self.chunk)
        json.dumps(record)
        self.assertNotIn("embedding", record)
        self.assertNotIn("embedding_model", record)
        self.assertEqual(record["source_metadata"]["chapter_id"], "CH-1")


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.inp = Path(self.tmp.name) / "cleaned_data"
        self.out = Path(self.tmp.name) / "chunked_data"
        self.inp.mkdir()
        (self.inp / "physics_cleaned.json").write_text(
            json.dumps(make_cleaned_document(8)), encoding="utf-8"
        )
        with (self.inp / "more.jsonl").open("w", encoding="utf-8") as f:
            f.write(json.dumps(make_cleaned_document(4, "Chapter 6: Work")) + "\n")
            f.write(json.dumps({"title": "Note", "text": PHYSICS_PARAGRAPH * 3}) + "\n")

    def tearDown(self):
        self.tmp.cleanup()

    def test_end_to_end(self):
        manifest = dcx.run_pipeline(self.inp, self.out)

        self.assertEqual(manifest["failed_files"], 0)
        self.assertEqual(manifest["successful_files"], 2)
        self.assertGreater(manifest["chunk_count"], 0)

        llm_lines = (self.out / "llm_training_data.jsonl").read_text("utf-8").splitlines()
        rag_lines = (self.out / "rag_data.jsonl").read_text("utf-8").splitlines()
        self.assertEqual(len(llm_lines), manifest["llm_record_count"])
        self.assertEqual(len(rag_lines), manifest["rag_record_count"])
        self.assertEqual(len(rag_lines), manifest["chunk_count"])

        # every line is valid JSON and the two datasets share chunk ids
        llm_ids = {json.loads(line)["chunk_id"] for line in llm_lines}
        rag_ids = {json.loads(line)["chunk_id"] for line in rag_lines}
        self.assertEqual(llm_ids, rag_ids)

        # no temp files left behind
        self.assertFalse([p for p in self.out.iterdir() if p.suffix == ".tmp"])

        stored = json.loads((self.out / "chunking_manifest.json").read_text("utf-8"))
        self.assertEqual(stored["chunk_count"], manifest["chunk_count"])

    def test_rerun_replaces_outputs_safely(self):
        first = dcx.run_pipeline(self.inp, self.out)
        second = dcx.run_pipeline(self.inp, self.out)
        self.assertEqual(first["chunk_count"], second["chunk_count"])
        rag_lines = (self.out / "rag_data.jsonl").read_text("utf-8").splitlines()
        self.assertEqual(len(rag_lines), second["chunk_count"])

    def test_manifest_file_is_not_reprocessed_as_input(self):
        # write the manifest into the input folder to simulate in-place output
        dcx.run_pipeline(self.inp, self.inp)
        files = [p.name for p in dcx.discover_input_files(self.inp)]
        self.assertNotIn("chunking_manifest.json", files)


if __name__ == "__main__":
    unittest.main(verbosity=2)
