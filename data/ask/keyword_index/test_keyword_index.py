#!/usr/bin/env python3
"""
Test cases for keyword_index.py

Run with either:
    python3 test_keyword_index.py          (unittest, no extra installs)
    python3 -m pytest test_keyword_index.py

Mirrors test_vector_base.py's fixtures and expectations as closely as
possible, so the two indexes can be checked for the same behaviour on the
same inputs. No real embedding model is needed: vectors are written by
hand and never actually scored by keyword_index.py.
"""

from __future__ import annotations

import doctest
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

import keyword_index as ki
import vector_base as vb


def load_tests(loader, tests, ignore):
    """Hook so `python3 test_keyword_index.py` also runs every docstring example."""
    tests.addTests(doctest.DocTestSuite(ki))
    return tests


def unit(*values: float) -> list[float]:
    vector = np.array(values, dtype="float32")
    return (vector / np.linalg.norm(vector)).tolist()


RAG_VECTORS = np.array(
    [unit(1, 0, 0), unit(0, 1, 0), unit(0.9, 0.1, 0)], dtype="float32"
)

RAG_METADATA = [
    {
        "vector_index": 0,
        "document_id": "DOC-1",
        "chunk_id": "CHUNK-1",
        "topic": "Mechanics",
        "concept": "Momentum",
        "formula": "p = mv",
        "text_sha256": "aaa",
        "source_metadata": {
            "chapter_title": "Chapter 3",
            "source_file": "physics.pdf",
            "source_page_start": 40,
            "source_page_end": 52,
        },
    },
    {
        "vector_index": 1,
        "document_id": "DOC-1",
        "chunk_id": "CHUNK-2",
        "topic": "Optics",
        "concept": "Refraction",
        "formula": None,
        "text_sha256": "bbb",
        "source_metadata": {"chapter_title": "Chapter 9", "source_file": "physics.pdf"},
    },
    {
        "vector_index": 2,
        "document_id": "DOC-2",
        "chunk_id": "CHUNK-3",
        "topic": "Mechanics",
        "concept": "Impulse",
        "formula": None,
        "text_sha256": "ccc",
        "source_metadata": {"chapter_title": "Chapter 3", "source_file": "notes.pdf"},
    },
]

LLM_VECTORS = np.array([unit(1, 0, 0), unit(0, 0, 1)], dtype="float32")

LLM_METADATA = [
    {
        "vector_index": 0,
        "record_id": "CE-1",
        "training_type": "concept_explanation",
        "text_sha256": "aaa",  # same passage as CHUNK-1: a duplicate
        "source_metadata": {"source_file": "physics.pdf"},
    },
    {
        "vector_index": 1,
        "record_id": "QA-1",
        "training_type": "question_reasoning_answer",
        "text_sha256": "ddd",
        "source_metadata": {"source_file": "physics.pdf"},
    },
]

CHUNK_TEXTS = [
    {"chunk_id": "CHUNK-1", "text": "Momentum is the product of mass and velocity."},
    {"chunk_id": "CHUNK-2", "text": "Refraction bends light at a boundary."},
    {"chunk_id": "CHUNK-3", "text": "Impulse is the change in momentum."},
]

LLM_TEXTS = [
    {
        "record_id": "CE-1",
        "type": "concept_explanation",
        "concept": "Momentum",
        "explanation": "Mass times velocity.",
    },
    {
        "record_id": "QA-1",
        "type": "question_reasoning_answer",
        "question": "What is impulse?",
        "reasoning": "Find the definition.",
        "answer": "The change in momentum.",
    },
]


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record) + "\n")


class BuildTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.embedded = root / "embedding_data"
        self.chunked = root / "chunked_data"
        self.index_folder = root / "keyword_index"
        self.embedded.mkdir()

        np.save(self.embedded / "rag_embeddings.npy", RAG_VECTORS)
        np.save(self.embedded / "llm_embeddings.npy", LLM_VECTORS)
        write_jsonl(self.embedded / "rag_embedding_metadata.jsonl", RAG_METADATA)
        write_jsonl(self.embedded / "llm_embedding_metadata.jsonl", LLM_METADATA)
        write_jsonl(self.chunked / "rag_data.jsonl", CHUNK_TEXTS)
        write_jsonl(self.chunked / "llm_training_data.jsonl", LLM_TEXTS)

    def tearDown(self):
        self.tmp.cleanup()

    def build(self, **kwargs):
        options = dict(verbose=False)
        options.update(kwargs)
        return ki.build_keyword_index(
            self.embedded, self.chunked, self.index_folder, **options
        )

    def test_both_datasets_are_merged(self):
        manifest = self.build(deduplicate_rows=False)
        self.assertEqual(manifest["record_count"], 5)
        self.assertEqual(
            [d["dataset"] for d in manifest["datasets"]], ["llm", "rag"]
        )

    def test_duplicate_passages_are_collapsed(self):
        manifest = self.build()
        self.assertEqual(manifest["duplicates_removed"], 1)
        self.assertEqual(manifest["record_count"], 4)

        index = ki.KeywordIndex.load(self.index_folder)
        digests = [r["text_sha256"] for r in index.records]
        self.assertEqual(len(digests), len(set(digests)))

    def test_rows_are_renumbered_after_deduplication(self):
        self.build()
        index = ki.KeywordIndex.load(self.index_folder)
        self.assertEqual([r["row"] for r in index.records], list(range(len(index))))

    def test_text_is_joined_from_the_chunked_folder(self):
        manifest = self.build()
        self.assertEqual(manifest["records_without_text"], 0)

        index = ki.KeywordIndex.load(self.index_folder)
        by_chunk = {r["chunk_id"]: r["text"] for r in index.records if r["chunk_id"]}
        self.assertEqual(by_chunk["CHUNK-2"], "Refraction bends light at a boundary.")

        by_record = {r["record_id"]: r["text"] for r in index.records if r["record_id"]}
        self.assertIn("Question: What is impulse?", by_record["QA-1"])
        self.assertEqual(by_record["CE-1"], "Concept: Momentum\nExplanation: Mass times velocity.")

    def test_records_match_vector_base_exactly(self):
        # The whole point of reusing vector_base.py's loading pipeline: a
        # hybrid-search caller must be able to line this index's rows up
        # with vector_base's rows by id.
        vector_base_folder = Path(self.tmp.name) / "vector_base"
        vector_manifest = vb.build_vector_base(
            self.embedded, self.chunked, vector_base_folder, verbose=False
        )
        keyword_manifest = self.build()

        vector_base = vb.VectorBase.load(vector_base_folder)
        keyword_index = ki.KeywordIndex.load(self.index_folder)

        self.assertEqual(vector_manifest["vector_count"], keyword_manifest["record_count"])
        self.assertEqual(vector_base.records, keyword_index.records)

    def test_corpus_tokens_align_with_records(self):
        self.build()
        index = ki.KeywordIndex.load(self.index_folder)
        self.assertEqual(len(index.corpus_tokens), len(index.records))
        by_chunk = {r["chunk_id"]: t for r, t in zip(index.records, index.corpus_tokens) if r["chunk_id"]}
        self.assertIn("refraction", by_chunk["CHUNK-2"])

    def test_missing_chunked_folder_still_builds(self):
        manifest = ki.build_keyword_index(
            self.embedded, Path("/no/such/folder"), self.index_folder, verbose=False
        )
        self.assertEqual(manifest["records_with_text"], 0)
        self.assertGreater(manifest["records_without_text"], 0)

        index = ki.KeywordIndex.load(self.index_folder)
        self.assertTrue(all(r["text"] == "" for r in index.records))

    def test_dataset_selection(self):
        manifest = self.build(datasets=("rag",))
        self.assertEqual(manifest["record_count"], 3)
        index = ki.KeywordIndex.load(self.index_folder)
        self.assertEqual({r["dataset"] for r in index.records}, {"rag"})

    def test_datasets_with_different_dimensions_are_allowed(self):
        # Unlike vector_base.py, keyword_index.py never stacks real vector
        # values, so datasets whose embeddings have different widths (or
        # come from different models entirely) can still share one index.
        np.save(
            self.embedded / "llm_embeddings.npy",
            np.ones((2, 7), dtype="float32"),
        )
        manifest = self.build()
        self.assertEqual(manifest["record_count"], 4)

    def test_vector_metadata_length_mismatch_is_rejected(self):
        write_jsonl(self.embedded / "rag_embedding_metadata.jsonl", RAG_METADATA[:1])
        with self.assertRaises(ValueError) as caught:
            self.build(datasets=("rag",))
        self.assertIn("3 vectors but 1 metadata", str(caught.exception))

    def test_empty_embedding_folder_is_rejected(self):
        for path in self.embedded.iterdir():
            path.unlink()
        with self.assertRaises(ValueError) as caught:
            self.build()
        self.assertIn("Run embedding.py first", str(caught.exception))

    def test_missing_embedding_folder_raises(self):
        with self.assertRaises(FileNotFoundError):
            ki.build_keyword_index(Path("/no/such"), self.chunked, self.index_folder)

    def test_no_temporary_files_left(self):
        self.build()
        self.assertFalse(
            [p for p in self.index_folder.iterdir() if p.name.startswith(".")]
        )

    def test_rebuild_is_idempotent(self):
        first = self.build()
        content = (self.index_folder / "records.jsonl").read_text("utf-8")
        second = self.build()
        self.assertEqual(content, (self.index_folder / "records.jsonl").read_text("utf-8"))
        self.assertEqual(first["record_count"], second["record_count"])


class SearchTests(unittest.TestCase):
    def setUp(self):
        records = [
            vb.build_record(metadata, "rag", row, {})
            for row, metadata in enumerate(RAG_METADATA)
        ]
        for record, chunk in zip(records, CHUNK_TEXTS):
            record["text"] = chunk["text"]
            record["has_text"] = True

        corpus_tokens = [ki.tokenize(r["text"]) for r in records]
        self.index = ki.KeywordIndex(records, corpus_tokens, {"program": "stub"})

    def test_exact_keyword_match_wins(self):
        # "Refraction" appears only in CHUNK-2's passage; it must rank first
        # even though a dense embedding might not favour it at all.
        results = self.index.search("refraction", top_k=3)
        self.assertEqual(results[0].row, 1)
        self.assertEqual(results[0].text, CHUNK_TEXTS[1]["text"])
        self.assertGreater(results[0].score, 0.0)

    def test_ranks_are_sequential(self):
        results = self.index.search("momentum", top_k=3)
        self.assertEqual([r.rank for r in results], [1, 2, 3])

    def test_shared_term_ranks_both_momentum_passages_above_the_rest(self):
        results = self.index.search("momentum", top_k=3)
        top_two_rows = {results[0].row, results[1].row}
        self.assertEqual(top_two_rows, {0, 2})  # CHUNK-1 and CHUNK-3 both say "momentum"

    def test_filters_narrow_the_result_set(self):
        results = self.index.search("momentum", top_k=5, filters={"topic": "Mechanics"})
        self.assertEqual({r.record["topic"] for r in results}, {"Mechanics"})

        results = self.index.search(
            "momentum", top_k=5, filters={"source_file": "notes.pdf"}
        )
        self.assertEqual([r.record["chunk_id"] for r in results], ["CHUNK-3"])

    def test_filter_accepts_a_list_of_values(self):
        results = self.index.search(
            "light", top_k=5, filters={"topic": ["Optics", "Mechanics"]}
        )
        self.assertEqual(len(results), 3)

    def test_degenerate_requests_return_nothing(self):
        self.assertEqual(self.index.search("momentum", top_k=0), [])
        self.assertEqual(
            self.index.search("momentum", filters={"topic": "Nonexistent"}), []
        )
        empty = ki.KeywordIndex([], [])
        self.assertEqual(empty.search("momentum"), [])

    def test_top_k_larger_than_the_index(self):
        results = self.index.search("momentum", top_k=50)
        self.assertEqual(len(results), 3)

    def test_mismatched_lengths_are_rejected_at_construction(self):
        with self.assertRaises(ValueError):
            ki.KeywordIndex([{"row": 0}], [])

    def test_len(self):
        self.assertEqual(len(self.index), 3)


class PersistenceTests(unittest.TestCase):
    def test_save_and_load_round_trip(self):
        records = [
            {"row": 0, "dataset": "rag", "topic": "Heat", "text": "Heat flows."},
            {"row": 1, "dataset": "rag", "topic": "Light", "text": "Light travels."},
        ]
        corpus_tokens = [ki.tokenize(r["text"]) for r in records]
        index = ki.KeywordIndex(records, corpus_tokens, {"program": "stub"})

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "index"
            manifest = index.save(path)
            reloaded = ki.KeywordIndex.load(path)

            self.assertEqual(manifest["record_count"], 2)
            self.assertEqual(len(reloaded), 2)
            self.assertEqual(reloaded.records, records)
            self.assertEqual(reloaded.corpus_tokens, corpus_tokens)
            self.assertIsNotNone(reloaded.bm25)

            # Search works against the index rebuilt purely from stored
            # tokens, with no pickled BM25Okapi object involved.
            results = reloaded.search("heat", top_k=1)
            self.assertEqual(results[0].row, 0)

    def test_loading_a_missing_index_raises(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(FileNotFoundError) as caught:
                ki.KeywordIndex.load(Path(folder))
            self.assertIn("Run keyword_index.py first", str(caught.exception))


class CommandLineTests(unittest.TestCase):
    def test_defaults(self):
        options = ki.parse_arguments([])
        self.assertEqual(options.datasets, ["llm", "rag"])
        self.assertTrue(options.deduplicate)

    def test_dataset_choice_is_validated(self):
        with self.assertRaises(SystemExit):
            ki.parse_arguments(["--datasets", "nonsense"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
