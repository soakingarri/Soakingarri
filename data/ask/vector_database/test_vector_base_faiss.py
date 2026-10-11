#!/usr/bin/env python3
"""
Test cases for vector_base_faiss.py

Run with either:
    python3 test_vector_base_faiss.py          (unittest)
    python3 -m pytest test_vector_base_faiss.py

Mirrors test_vector_base.py's fixtures and expectations, so the two backends
can be checked for the same behaviour on the same inputs.
"""

from __future__ import annotations

import doctest
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

import vector_base as vb
import vector_base_faiss as vbf


def load_tests(loader, tests, ignore):
    """Hook so `python3 test_vector_base_faiss.py` also runs every docstring example."""
    tests.addTests(doctest.DocTestSuite(vbf))
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
        self.base = root / "vector_base_faiss"
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
        return vbf.build_vector_base_faiss(
            self.embedded, self.chunked, self.base, **options
        )

    def test_both_datasets_are_merged(self):
        manifest = self.build(deduplicate_rows=False)
        self.assertEqual(manifest["vector_count"], 5)
        self.assertEqual(manifest["dimension"], 3)
        self.assertEqual(manifest["backend"], "faiss.IndexFlatIP")

    def test_duplicate_passages_are_collapsed(self):
        manifest = self.build()
        self.assertEqual(manifest["duplicates_removed"], 1)
        self.assertEqual(manifest["vector_count"], 4)

        base = vbf.FaissVectorBase.load(self.base)
        digests = [r["text_sha256"] for r in base.records]
        self.assertEqual(len(digests), len(set(digests)))

    def test_text_is_joined_from_the_chunked_folder(self):
        manifest = self.build()
        self.assertEqual(manifest["records_without_text"], 0)

        base = vbf.FaissVectorBase.load(self.base)
        by_chunk = {r["chunk_id"]: r["text"] for r in base.records if r["chunk_id"]}
        self.assertEqual(by_chunk["CHUNK-2"], "Refraction bends light at a boundary.")

    def test_dataset_selection(self):
        manifest = self.build(datasets=("rag",))
        self.assertEqual(manifest["vector_count"], 3)
        base = vbf.FaissVectorBase.load(self.base)
        self.assertEqual(base.datasets(), ["rag"])

    def test_mismatched_dimensions_are_rejected(self):
        np.save(
            self.embedded / "llm_embeddings.npy",
            np.ones((2, 7), dtype="float32"),
        )
        with self.assertRaises(ValueError) as caught:
            self.build()
        self.assertIn("different vector dimensions", str(caught.exception))

    def test_empty_embedding_folder_is_rejected(self):
        for path in self.embedded.iterdir():
            path.unlink()
        with self.assertRaises(ValueError) as caught:
            self.build()
        self.assertIn("Run embedding.py first", str(caught.exception))

    def test_missing_embedding_folder_raises(self):
        with self.assertRaises(FileNotFoundError):
            vbf.build_vector_base_faiss(Path("/no/such"), self.chunked, self.base)

    def test_no_temporary_files_left(self):
        self.build()
        self.assertFalse([p for p in self.base.iterdir() if p.name.startswith(".")])

    def test_rebuild_is_idempotent(self):
        first = self.build()
        content = (self.base / "records.jsonl").read_text("utf-8")
        second = self.build()
        self.assertEqual(content, (self.base / "records.jsonl").read_text("utf-8"))
        self.assertEqual(first["vector_count"], second["vector_count"])

    def test_output_files_written(self):
        self.build()
        written = sorted(p.name for p in self.base.iterdir())
        self.assertEqual(
            written,
            ["records.jsonl", "vector_base_manifest.json", "vectors.faiss"],
        )


class SearchTests(unittest.TestCase):
    def setUp(self):
        records = [
            vb.build_record(metadata, "rag", row, {})
            for row, metadata in enumerate(RAG_METADATA)
        ]
        for record, chunk in zip(records, CHUNK_TEXTS):
            record["text"] = chunk["text"]
            record["has_text"] = True

        index = vbf.build_index(vb.normalize_rows(RAG_VECTORS))
        self.base = vbf.FaissVectorBase(
            index, records, {"embedding_model": "stub"}
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
        empty = vbf.FaissVectorBase(vbf.build_index(np.zeros((0, 3), dtype="float32")), [])
        self.assertEqual(empty.search(np.array([1.0, 0.0, 0.0])), [])

    def test_top_k_larger_than_the_base(self):
        results = self.base.search(np.array([1.0, 0.0, 0.0]), top_k=50)
        self.assertEqual(len(results), 3)

    def test_helpers(self):
        self.assertEqual(self.base.topics(), ["Mechanics", "Optics"])
        self.assertEqual(self.base.datasets(), ["rag"])
        self.assertEqual(self.base.dimension, 3)
        self.assertEqual(len(self.base), 3)

    def test_matches_numpy_backend_on_the_same_data(self):
        """Same vectors, same query: the two backends must agree."""
        numpy_base = vb.VectorBase(
            vb.normalize_rows(RAG_VECTORS),
            [
                vb.build_record(metadata, "rag", row, {})
                for row, metadata in enumerate(RAG_METADATA)
            ],
        )
        query = np.array([0.9, 0.2, 0.0])
        numpy_rows = [r.row for r in numpy_base.search(query, top_k=3)]
        faiss_rows = [r.row for r in self.base.search(query, top_k=3)]
        self.assertEqual(numpy_rows, faiss_rows)


class PersistenceTests(unittest.TestCase):
    def test_save_and_load_round_trip(self):
        records = [
            {"row": 0, "dataset": "rag", "topic": "Heat", "text": "Heat flows."},
            {"row": 1, "dataset": "rag", "topic": "Light", "text": "Light travels."},
        ]
        vectors = vb.normalize_rows(np.array([[1.0, 0.0], [0.0, 1.0]]))
        index = vbf.build_index(vectors)
        base = vbf.FaissVectorBase(index, records, {"embedding_model": "stub"})

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "base"
            manifest = base.save(path)
            reloaded = vbf.FaissVectorBase.load(path)

            self.assertEqual(manifest["vector_count"], 2)
            self.assertEqual(len(reloaded), 2)
            self.assertEqual(reloaded.embedding_model, "stub")
            self.assertEqual(reloaded.records, records)

    def test_loading_a_missing_base_raises(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(FileNotFoundError) as caught:
                vbf.FaissVectorBase.load(Path(folder))
            self.assertIn("Run vector_base_faiss.py first", str(caught.exception))

    def test_mismatched_lengths_are_rejected_at_construction(self):
        index = vbf.build_index(np.zeros((3, 2), dtype="float32"))
        with self.assertRaises(ValueError):
            vbf.FaissVectorBase(index, [{"row": 0}])


class UpsertTests(unittest.TestCase):
    def setUp(self):
        records = [
            vb.build_record(metadata, "rag", row, {})
            for row, metadata in enumerate(RAG_METADATA)
        ]
        for record, chunk in zip(records, CHUNK_TEXTS):
            record["text"] = chunk["text"]
            record["has_text"] = True

        index = vbf.build_index(vb.normalize_rows(RAG_VECTORS))
        self.base = vbf.FaissVectorBase(
            index, records, {"embedding_model": "stub"}
        )

    def test_upsert_replaces_an_existing_chunk_id(self):
        updated = self.base.upsert(
            np.array([[0.0, 0.0, 1.0]]),
            [{"chunk_id": "CHUNK-1", "text": "Momentum, revised."}],
        )

        self.assertEqual(len(updated), 3)
        self.assertEqual(len(self.base), 3)  # self is unmodified

        by_chunk = {r["chunk_id"]: r for r in updated.records}
        self.assertEqual(by_chunk["CHUNK-1"]["text"], "Momentum, revised.")
        self.assertEqual(
            by_chunk["CHUNK-2"]["text"], "Refraction bends light at a boundary."
        )
        self.assertEqual(sorted(r["row"] for r in updated.records), [0, 1, 2])

    def test_upsert_adds_a_genuinely_new_chunk_id(self):
        updated = self.base.upsert(
            np.array([[0.0, 0.0, 1.0]]),
            [{"chunk_id": "CHUNK-4", "text": "A brand new passage."}],
        )

        self.assertEqual(len(updated), 4)
        self.assertEqual(
            sorted(r["chunk_id"] for r in updated.records),
            ["CHUNK-1", "CHUNK-2", "CHUNK-3", "CHUNK-4"],
        )
        self.assertEqual(sorted(r["row"] for r in updated.records), [0, 1, 2, 3])

    def test_upsert_rejects_mismatched_lengths(self):
        with self.assertRaises(ValueError):
            self.base.upsert(
                np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]),
                [{"chunk_id": "CHUNK-4"}],
            )

    def test_upsert_result_is_searchable(self):
        updated = self.base.upsert(
            np.array([[0.0, 0.0, 1.0]]),
            [{"chunk_id": "CHUNK-1", "text": "Momentum, revised."}],
        )
        results = updated.search(np.array([0.0, 0.0, 1.0]), top_k=1)
        self.assertEqual(results[0].record["chunk_id"], "CHUNK-1")
        self.assertEqual(results[0].text, "Momentum, revised.")


class UpsertVectorBaseFaissTests(unittest.TestCase):
    def test_creates_a_store_from_nothing(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "vector_base_faiss"
            manifest = vbf.upsert_vector_base_faiss(
                path,
                np.array([[1.0, 0.0], [0.0, 1.0]]),
                [
                    {"chunk_id": "C1", "text": "C1 text"},
                    {"chunk_id": "C2", "text": "C2 text"},
                ],
            )

            self.assertEqual(manifest["vector_count"], 2)
            self.assertEqual(manifest["dimension"], 2)

            reloaded = vbf.FaissVectorBase.load(path)
            self.assertEqual(
                sorted(r["chunk_id"] for r in reloaded.records), ["C1", "C2"]
            )

    def test_updates_a_saved_store_on_disk(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            embedded, chunked, base = root / "emb", root / "chunk", root / "base"
            embedded.mkdir()

            np.save(embedded / "rag_embeddings.npy", RAG_VECTORS)
            write_jsonl(
                embedded / "rag_embedding_metadata.jsonl", RAG_METADATA
            )
            write_jsonl(chunked / "rag_data.jsonl", CHUNK_TEXTS)

            vbf.build_vector_base_faiss(
                embedded, chunked, base, datasets=("rag",), verbose=False
            )

            manifest = vbf.upsert_vector_base_faiss(
                base,
                np.array([[0.0, 0.0, 1.0]]),
                [{"chunk_id": "CHUNK-1", "text": "Momentum, revised."}],
            )

            self.assertEqual(manifest["vector_count"], 3)

            reloaded = vbf.FaissVectorBase.load(base)
            by_chunk = {r["chunk_id"]: r for r in reloaded.records}
            self.assertEqual(by_chunk["CHUNK-1"]["text"], "Momentum, revised.")
            self.assertEqual(
                by_chunk["CHUNK-3"]["text"], "Impulse is the change in momentum."
            )


class CommandLineTests(unittest.TestCase):
    def test_defaults(self):
        options = vbf.parse_arguments([])
        self.assertEqual(options.datasets, ["llm", "rag"])
        self.assertTrue(options.deduplicate)

    def test_dataset_choice_is_validated(self):
        with self.assertRaises(SystemExit):
            vbf.parse_arguments(["--datasets", "nonsense"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
