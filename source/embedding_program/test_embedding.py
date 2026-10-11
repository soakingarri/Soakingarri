#!/usr/bin/env python3
"""
Test cases for embedding.py

Run with either:
    python3 test_embedding.py          (unittest, no extra installs)
    python3 -m pytest test_embedding.py

Every function in embedding.py carries doctest examples in its docstring; this
file runs all of those plus end-to-end checks. A stub model replaces
sentence-transformers, so no model download and no torch install is needed.
"""

from __future__ import annotations

import doctest
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

import embedding as emb


def load_tests(loader, tests, ignore):
    """Hook so `python3 test_embedding.py` also runs every docstring example."""
    tests.addTests(doctest.DocTestSuite(emb))
    return tests


class StubModel:
    """Deterministic stand-in for SentenceTransformer.

    Produces unit-length vectors derived from the text, so two identical texts
    embed identically and validate_vector_output's norm check is exercised.
    """

    def __init__(self, dimension: int = 8, normalize: bool = True):
        self.dimension = dimension
        self.normalize = normalize
        self.batch_sizes: list[int] = []
        self.seen_texts: list[str] = []

    def get_sentence_embedding_dimension(self) -> int:
        return self.dimension

    def encode(self, sentences, **kwargs):
        self.batch_sizes.append(len(sentences))
        self.seen_texts.extend(sentences)

        vectors = np.zeros((len(sentences), self.dimension), dtype="float32")

        for row, text in enumerate(sentences):
            for position, character in enumerate(text[: self.dimension * 4]):
                vectors[row, position % self.dimension] += ord(character) % 13

        if kwargs.get("normalize_embeddings", self.normalize):
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            vectors = vectors / norms

        return vectors


LLM_RECORDS = [
    {
        "record_id": "QA-1",
        "type": "question_reasoning_answer",
        "question": "What is momentum?",
        "reasoning": "Locate the definition.",
        "answer": "Mass times velocity.",
        "chunk_id": "CHUNK-1",
    },
    {
        "record_id": "CE-1",
        "type": "concept_explanation",
        "concept": "Momentum",
        "explanation": "Momentum is mass times velocity.",
        "chunk_id": "CHUNK-1",
    },
    {
        "record_id": "FD-1",
        "type": "formula_derivation_example",
        "formula": "p = mv",
        "derivation": "Therefore it scales with mass.",
        "example": "A truck versus a bicycle.",
        "chunk_id": "CHUNK-1",
    },
]

RAG_RECORDS = [
    {
        "document_id": "DOC-1",
        "chunk_id": "CHUNK-1",
        "topic": "Mechanics",
        "concept": "Momentum",
        "formula": "p = mv",
        "text": "Momentum is the product of mass and velocity.",
        "source_metadata": {"chapter_title": "Chapter 3", "source_file": "physics.pdf"},
    },
    {
        "document_id": "DOC-1",
        "chunk_id": "CHUNK-2",
        "topic": "Mechanics",
        "concept": "Impulse",
        "formula": None,
        "text": "Impulse is the change in momentum over time.",
        "source_metadata": {"chapter_title": "Chapter 3"},
    },
]


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record) + "\n")


class TextExtractionTests(unittest.TestCase):
    def test_llm_record_types_all_produce_text(self):
        for record in LLM_RECORDS:
            self.assertTrue(emb.extract_llm_text(record), record["type"])

    def test_chat_messages_take_priority(self):
        record = dict(LLM_RECORDS[0])
        record["messages"] = [{"role": "user", "content": "Hi"}]
        self.assertEqual(emb.extract_llm_text(record), "user: Hi")

    def test_rag_falls_back_to_fields_when_text_missing(self):
        record = {k: v for k, v in RAG_RECORDS[0].items() if k != "text"}
        self.assertEqual(
            emb.extract_rag_text(record),
            "Topic: Mechanics\nConcept: Momentum\nFormula: p = mv",
        )

    def test_empty_record_is_replaced_by_placeholder_downstream(self):
        # extract_* returns "" and generate_file_embeddings substitutes a marker
        self.assertEqual(emb.extract_llm_text({}), "")
        self.assertEqual(emb.extract_rag_text({}), "")


class MetadataTests(unittest.TestCase):
    def test_input_records_are_not_mutated(self):
        record = {"document_id": "D", "source_metadata": {"a": 1}}
        snapshot = json.dumps(record, sort_keys=True)
        emb.create_rag_metadata(record, 1, 0, "text")
        self.assertEqual(json.dumps(record, sort_keys=True), snapshot)

    def test_generated_ids_are_stable_for_same_input(self):
        first = emb.create_llm_metadata({}, 3, 0, "same text")
        second = emb.create_llm_metadata({}, 3, 0, "same text")
        self.assertEqual(first["record_id"], second["record_id"])
        self.assertEqual(first["text_sha256"], second["text_sha256"])

    def test_type_key_is_accepted_as_training_type(self):
        metadata = emb.create_llm_metadata(LLM_RECORDS[0], 1, 0, "t")
        self.assertEqual(metadata["training_type"], "question_reasoning_answer")


class GenerateEmbeddingsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.source = self.root / "rag_data.jsonl"
        write_jsonl(self.source, RAG_RECORDS)
        self.vectors = self.root / "rag_embeddings.npy"
        self.metadata = self.root / "rag_embedding_metadata.jsonl"

    def tearDown(self):
        self.tmp.cleanup()

    def run_generation(self, model=None, batch_size=8, dataset_type="rag"):
        return emb.generate_file_embeddings(
            input_path=self.source,
            vector_path=self.vectors,
            metadata_path=self.metadata,
            model=model or StubModel(),
            batch_size=batch_size,
            dataset_type=dataset_type,
            verbose=False,
        )

    def test_vector_rows_align_with_metadata_lines(self):
        result = self.run_generation()
        vectors = np.load(self.vectors)
        lines = [json.loads(l) for l in self.metadata.read_text("utf-8").splitlines()]

        self.assertEqual(result["record_count"], len(RAG_RECORDS))
        self.assertEqual(vectors.shape, (len(RAG_RECORDS), 8))
        self.assertEqual([m["vector_index"] for m in lines], list(range(len(lines))))
        self.assertEqual([m["chunk_id"] for m in lines], ["CHUNK-1", "CHUNK-2"])
        self.assertEqual([m["source_line"] for m in lines], [1, 2])
        self.assertEqual(lines[0]["source_file"], "rag_data.jsonl")

    def test_batching_does_not_change_results(self):
        self.run_generation(batch_size=1)
        one_at_a_time = np.load(self.vectors).copy()
        self.run_generation(batch_size=100)
        all_at_once = np.load(self.vectors)
        np.testing.assert_allclose(one_at_a_time, all_at_once)

    def test_batch_size_is_respected(self):
        model = StubModel()
        self.run_generation(model=model, batch_size=1)
        self.assertEqual(model.batch_sizes, [1, 1])

    def test_empty_record_gets_placeholder_text(self):
        write_jsonl(self.source, [{"document_id": "D"}])
        model = StubModel()
        self.run_generation(model=model)
        self.assertEqual(model.seen_texts, ["[EMPTY RECORD]"])

    def test_vectors_are_unit_length_and_validate(self):
        result = self.run_generation()
        emb.validate_vector_output(
            self.vectors, result["record_count"], result["embedding_dimension"]
        )
        norms = np.linalg.norm(np.load(self.vectors), axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_missing_and_empty_inputs_are_skipped(self):
        result = emb.generate_file_embeddings(
            input_path=self.root / "nope.jsonl",
            vector_path=self.vectors,
            metadata_path=self.metadata,
            model=StubModel(),
            batch_size=8,
            dataset_type="rag",
            verbose=False,
        )
        self.assertEqual(result["status"], "skipped")

        self.source.write_text("\n\n", encoding="utf-8")
        self.assertEqual(self.run_generation()["status"], "skipped")

    def test_unknown_dataset_type_raises(self):
        with self.assertRaises(ValueError):
            self.run_generation(dataset_type="unknown")

    def test_failure_leaves_no_partial_output(self):
        class BrokenModel(StubModel):
            def encode(self, sentences, **kwargs):
                raise RuntimeError("model exploded")

        with self.assertRaises(RuntimeError) as caught:
            self.run_generation(model=BrokenModel())

        # the original error must survive, not be masked by cleanup
        self.assertIn("model exploded", str(caught.exception))
        self.assertFalse(self.vectors.exists())
        self.assertFalse(self.metadata.exists())
        self.assertFalse([p for p in self.root.iterdir() if p.name.startswith(".")])

    def test_invalid_json_input_is_reported_with_line_number(self):
        self.source.write_text('{"text": "ok"}\nnot json\n', encoding="utf-8")
        with self.assertRaises(ValueError) as caught:
            self.run_generation()
        self.assertIn("line 2", str(caught.exception))


class ValidationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "v.npy"

    def tearDown(self):
        self.tmp.cleanup()

    def test_zero_vectors_are_allowed(self):
        np.save(self.path, np.zeros((2, 4), dtype="float32"))
        emb.validate_vector_output(self.path, 2, 4)  # zero norms are skipped

    def test_infinite_values_rejected(self):
        np.save(self.path, np.array([[np.inf, 0.0]], dtype="float32"))
        with self.assertRaises(RuntimeError):
            emb.validate_vector_output(self.path, 1, 2)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.inp = Path(self.tmp.name) / "chunked_data"
        self.out = Path(self.tmp.name) / "embedding_data"
        write_jsonl(self.inp / "llm_training_data.jsonl", LLM_RECORDS)
        write_jsonl(self.inp / "rag_data.jsonl", RAG_RECORDS)

    def tearDown(self):
        self.tmp.cleanup()

    def run_it(self, **kwargs):
        options = dict(model=StubModel(), device="cpu", batch_size=2, verbose=False)
        options.update(kwargs)
        return emb.run_pipeline(self.inp, self.out, **options)

    def test_end_to_end(self):
        manifest = self.run_it()

        self.assertEqual(manifest["total_embedded_records"], 5)
        self.assertEqual([d["status"] for d in manifest["datasets"]], ["success"] * 2)

        llm = np.load(self.out / "llm_embeddings.npy")
        rag = np.load(self.out / "rag_embeddings.npy")
        self.assertEqual(llm.shape, (3, 8))
        self.assertEqual(rag.shape, (2, 8))

        llm_meta = (self.out / "llm_embedding_metadata.jsonl").read_text("utf-8").splitlines()
        rag_meta = (self.out / "rag_embedding_metadata.jsonl").read_text("utf-8").splitlines()
        self.assertEqual(len(llm_meta), 3)
        self.assertEqual(len(rag_meta), 2)
        self.assertEqual(
            [json.loads(l)["record_id"] for l in llm_meta], ["QA-1", "CE-1", "FD-1"]
        )
        self.assertEqual(json.loads(llm_meta[0])["source_file"], "llm_training_data.jsonl")

        stored = json.loads((self.out / "embedding_manifest.json").read_text("utf-8"))
        self.assertEqual(stored["total_embedded_records"], 5)
        self.assertEqual(stored["device"], "cpu")
        self.assertEqual(stored["vector_dtype"], "float32")

        self.assertFalse([p for p in self.out.iterdir() if p.name.startswith(".")])

    def test_missing_dataset_is_skipped_not_fatal(self):
        (self.inp / "rag_data.jsonl").unlink()
        manifest = self.run_it()
        statuses = [d["status"] for d in manifest["datasets"]]
        self.assertEqual(statuses, ["success", "skipped"])
        self.assertEqual(manifest["total_embedded_records"], 3)

    def test_rerun_is_idempotent(self):
        first = self.run_it()
        first_vectors = np.load(self.out / "rag_embeddings.npy").copy()
        second = self.run_it()
        np.testing.assert_allclose(
            first_vectors, np.load(self.out / "rag_embeddings.npy")
        )
        self.assertEqual(
            first["total_embedded_records"], second["total_embedded_records"]
        )

    def test_missing_input_folder_raises(self):
        with self.assertRaises(FileNotFoundError):
            emb.run_pipeline(
                Path(self.tmp.name) / "nope", self.out, model=StubModel(), verbose=False
            )

    def test_input_path_that_is_a_file_raises(self):
        with self.assertRaises(NotADirectoryError):
            emb.run_pipeline(
                self.inp / "rag_data.jsonl", self.out, model=StubModel(), verbose=False
            )

    def test_dimension_mismatch_is_caught_by_validation(self):
        class LyingModel(StubModel):
            def get_sentence_embedding_dimension(self):
                return 8

            def encode(self, sentences, **kwargs):
                return np.ones((len(sentences), 8), dtype="float32")

        # unit-norm check: all-ones rows have norm sqrt(8), so validation must fail
        with self.assertRaises(RuntimeError):
            self.run_it(model=LyingModel())


if __name__ == "__main__":
    unittest.main(verbosity=2)
