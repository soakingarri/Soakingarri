#!/usr/bin/env python3
"""
Test cases for vector_base.py

Run with either:
    python3 test_vector_base.py          (unittest, no extra installs)
    python3 -m pytest test_vector_base.py

Runs every docstring example plus build, search and persistence checks. Vectors
are written by hand, so no embedding model is needed.
"""

from __future__ import annotations

import doctest
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

import vector_base as vb


def load_tests(loader, tests, ignore):
    """Hook so `python3 test_vector_base.py` also runs every docstring example."""
    tests.addTests(doctest.DocTestSuite(vb))
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
        self.base = root / "vector_base"
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
        return vb.build_vector_base(self.embedded, self.chunked, self.base, **options)

    def test_both_datasets_are_merged(self):
        manifest = self.build(deduplicate_rows=False)
        self.assertEqual(manifest["vector_count"], 5)
        self.assertEqual(manifest["dimension"], 3)
        self.assertEqual(
            [d["dataset"] for d in manifest["datasets"]], ["llm", "rag"]
        )

    def test_duplicate_passages_are_collapsed(self):
        manifest = self.build()
        self.assertEqual(manifest["duplicates_removed"], 1)
        self.assertEqual(manifest["vector_count"], 4)

        base = vb.VectorBase.load(self.base)
        digests = [r["text_sha256"] for r in base.records]
        self.assertEqual(len(digests), len(set(digests)))

    def test_rows_are_renumbered_after_deduplication(self):
        self.build()
        base = vb.VectorBase.load(self.base)
        self.assertEqual([r["row"] for r in base.records], list(range(len(base))))

    def test_text_is_joined_from_the_chunked_folder(self):
        manifest = self.build()
        self.assertEqual(manifest["records_without_text"], 0)

        base = vb.VectorBase.load(self.base)
        by_chunk = {r["chunk_id"]: r["text"] for r in base.records if r["chunk_id"]}
        self.assertEqual(by_chunk["CHUNK-2"], "Refraction bends light at a boundary.")

        by_record = {r["record_id"]: r["text"] for r in base.records if r["record_id"]}
        self.assertIn("Question: What is impulse?", by_record["QA-1"])
        self.assertEqual(by_record["CE-1"], "Concept: Momentum\nExplanation: Mass times velocity.")

    def test_deduplication_keeps_the_first_dataset_seen(self):
        # CE-1 (llm) and CHUNK-1 (rag) share a text_sha256; llm is processed first
        self.build()
        base = vb.VectorBase.load(self.base)
        chunk_ids = {r["chunk_id"] for r in base.records}
        record_ids = {r["record_id"] for r in base.records}
        self.assertIn("CE-1", record_ids)
        self.assertNotIn("CHUNK-1", chunk_ids)

    def test_text_join_without_deduplication_keeps_both_copies(self):
        self.build(deduplicate_rows=False)
        base = vb.VectorBase.load(self.base)
        by_chunk = {r["chunk_id"]: r["text"] for r in base.records if r["chunk_id"]}
        self.assertEqual(
            by_chunk["CHUNK-1"], "Momentum is the product of mass and velocity."
        )

    def test_missing_chunked_folder_still_builds(self):
        manifest = vb.build_vector_base(
            self.embedded, Path("/no/such/folder"), self.base, verbose=False
        )
        self.assertEqual(manifest["records_with_text"], 0)
        self.assertGreater(manifest["records_without_text"], 0)

        base = vb.VectorBase.load(self.base)
        self.assertTrue(all(r["text"] == "" for r in base.records))

    def test_dataset_selection(self):
        manifest = self.build(datasets=("rag",))
        self.assertEqual(manifest["vector_count"], 3)
        base = vb.VectorBase.load(self.base)
        self.assertEqual(base.datasets(), ["rag"])

    def test_vectors_are_normalized_on_the_way_in(self):
        unnormalized = np.array([[3.0, 4.0, 0.0], [0.0, 5.0, 0.0]], dtype="float32")
        np.save(self.embedded / "rag_embeddings.npy", unnormalized)
        write_jsonl(
            self.embedded / "rag_embedding_metadata.jsonl", RAG_METADATA[:2]
        )
        self.build(datasets=("rag",))
        vectors = np.load(self.base / "vectors.npy")
        np.testing.assert_allclose(
            np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5
        )

    def test_mismatched_dimensions_are_rejected(self):
        np.save(
            self.embedded / "llm_embeddings.npy",
            np.ones((2, 7), dtype="float32"),
        )
        with self.assertRaises(ValueError) as caught:
            self.build()
        self.assertIn("different vector dimensions", str(caught.exception))

    def test_vector_metadata_length_mismatch_is_rejected(self):
        write_jsonl(self.embedded / "rag_embedding_metadata.jsonl", RAG_METADATA[:1])
        with self.assertRaises(ValueError) as caught:
            self.build(datasets=("rag",))
        self.assertIn("3 vectors but 1 metadata", str(caught.exception))

    def test_out_of_order_vector_index_is_rejected(self):
        shuffled = [dict(r) for r in RAG_METADATA]
        shuffled[0]["vector_index"] = 2
        shuffled[2]["vector_index"] = 0
        write_jsonl(self.embedded / "rag_embedding_metadata.jsonl", shuffled)
        with self.assertRaises(ValueError):
            self.build(datasets=("rag",))

    def test_empty_embedding_folder_is_rejected(self):
        for path in self.embedded.iterdir():
            path.unlink()
        with self.assertRaises(ValueError) as caught:
            self.build()
        self.assertIn("Run embedding.py first", str(caught.exception))

    def test_missing_embedding_folder_raises(self):
        with self.assertRaises(FileNotFoundError):
            vb.build_vector_base(Path("/no/such"), self.chunked, self.base)

    def test_no_temporary_files_left(self):
        self.build()
        self.assertFalse([p for p in self.base.iterdir() if p.name.startswith(".")])

    def test_rebuild_is_idempotent(self):
        first = self.build()
        content = (self.base / "records.jsonl").read_text("utf-8")
        second = self.build()
        self.assertEqual(content, (self.base / "records.jsonl").read_text("utf-8"))
        self.assertEqual(first["vector_count"], second["vector_count"])


class SearchTests(unittest.TestCase):
    def setUp(self):
        records = [
            vb.build_record(metadata, "rag", row, {})
            for row, metadata in enumerate(RAG_METADATA)
        ]
        for record, chunk in zip(records, CHUNK_TEXTS):
            record["text"] = chunk["text"]
            record["has_text"] = True

        self.base = vb.VectorBase(
            vb.normalize_rows(RAG_VECTORS), records, {"embedding_model": "stub"}
        )

    def test_closest_row_wins(self):
        results = self.base.search(np.array([1.0, 0.0, 0.0]), top_k=3)
        self.assertEqual([r.row for r in results], [0, 2, 1])
        self.assertAlmostEqual(results[0].score, 1.0, places=5)
        self.assertEqual(results[0].text, CHUNK_TEXTS[0]["text"])

    def test_ranks_are_sequential(self):
        results = self.base.search(np.array([1.0, 0.0, 0.0]), top_k=3)
        self.assertEqual([r.rank for r in results], [1, 2, 3])

    def test_unnormalized_query_is_handled(self):
        normalized = self.base.search(np.array([1.0, 0.0, 0.0]), top_k=1)
        scaled = self.base.search(np.array([50.0, 0.0, 0.0]), top_k=1)
        self.assertAlmostEqual(normalized[0].score, scaled[0].score, places=5)

    def test_filters_narrow_the_result_set(self):
        results = self.base.search(
            np.array([1.0, 0.0, 0.0]), top_k=5, filters={"topic": "Mechanics"}
        )
        self.assertEqual({r.record["topic"] for r in results}, {"Mechanics"})

        results = self.base.search(
            np.array([1.0, 0.0, 0.0]), top_k=5, filters={"source_file": "notes.pdf"}
        )
        self.assertEqual([r.record["chunk_id"] for r in results], ["CHUNK-3"])

    def test_filter_accepts_a_list_of_values(self):
        results = self.base.search(
            np.array([1.0, 0.0, 0.0]),
            top_k=5,
            filters={"topic": ["Optics", "Mechanics"]},
        )
        self.assertEqual(len(results), 3)

    def test_minimum_score_drops_weak_matches(self):
        results = self.base.search(
            np.array([0.0, 1.0, 0.0]), top_k=5, minimum_score=0.9
        )
        self.assertEqual([r.row for r in results], [1])

    def test_diversity_reranking_changes_the_order(self):
        relevance_only = self.base.search(
            np.array([1.0, 0.0, 0.0]), top_k=2, relevance_weight=1.0
        )
        diversified = self.base.search(
            np.array([1.0, 0.0, 0.0]), top_k=2, relevance_weight=0.1
        )
        self.assertEqual([r.row for r in relevance_only], [0, 2])
        self.assertEqual([r.row for r in diversified], [0, 1])

    def test_wrong_dimension_query_raises(self):
        with self.assertRaises(ValueError):
            self.base.search(np.array([1.0, 0.0]), top_k=1)

    def test_degenerate_requests_return_nothing(self):
        self.assertEqual(self.base.search(np.array([1.0, 0.0, 0.0]), top_k=0), [])
        self.assertEqual(
            self.base.search(
                np.array([1.0, 0.0, 0.0]), filters={"topic": "Nonexistent"}
            ),
            [],
        )
        empty = vb.VectorBase(np.zeros((0, 3), dtype="float32"), [])
        self.assertEqual(empty.search(np.array([1.0, 0.0, 0.0])), [])

    def test_top_k_larger_than_the_base(self):
        results = self.base.search(np.array([1.0, 0.0, 0.0]), top_k=50)
        self.assertEqual(len(results), 3)

    def test_helpers(self):
        self.assertEqual(self.base.topics(), ["Mechanics", "Optics"])
        self.assertEqual(self.base.datasets(), ["rag"])
        self.assertEqual(self.base.dimension, 3)
        self.assertEqual(len(self.base), 3)


class PersistenceTests(unittest.TestCase):
    def test_save_and_load_round_trip(self):
        records = [
            {"row": 0, "dataset": "rag", "topic": "Heat", "text": "Heat flows."},
            {"row": 1, "dataset": "rag", "topic": "Light", "text": "Light travels."},
        ]
        vectors = vb.normalize_rows(np.array([[1.0, 0.0], [0.0, 1.0]]))
        base = vb.VectorBase(vectors, records, {"embedding_model": "stub"})

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "base"
            manifest = base.save(path)
            reloaded = vb.VectorBase.load(path)

            self.assertEqual(manifest["vector_count"], 2)
            self.assertEqual(len(reloaded), 2)
            self.assertEqual(reloaded.embedding_model, "stub")
            self.assertEqual(reloaded.records, records)
            np.testing.assert_allclose(np.asarray(reloaded.vectors), vectors)

    def test_loading_a_missing_base_raises(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(FileNotFoundError) as caught:
                vb.VectorBase.load(Path(folder))
            self.assertIn("Run vector_base.py first", str(caught.exception))

    def test_mismatched_lengths_are_rejected_at_construction(self):
        with self.assertRaises(ValueError):
            vb.VectorBase(np.zeros((3, 2), dtype="float32"), [{"row": 0}])


class CommandLineTests(unittest.TestCase):
    def test_defaults(self):
        options = vb.parse_arguments([])
        self.assertEqual(options.datasets, ["llm", "rag"])
        self.assertTrue(options.deduplicate)

    def test_dataset_choice_is_validated(self):
        with self.assertRaises(SystemExit):
            vb.parse_arguments(["--datasets", "nonsense"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
