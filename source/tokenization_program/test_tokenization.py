#!/usr/bin/env python3
"""
Test cases for tokenization.py

Run with either:
    python3 test_tokenization.py          (unittest, no extra installs)
    python3 -m pytest test_tokenization.py

Runs every docstring example plus windowing invariants. A one-token-per-word
stub tokenizer makes the arithmetic in the assertions exact; the approximate
tokenizer is exercised separately, and the real transformers tokenizer is used
only if it happens to be installed.
"""

from __future__ import annotations

import doctest
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import tokenization as tok


def load_tests(loader, tests, ignore):
    """Hook so `python3 test_tokenization.py` also runs every docstring example."""
    tests.addTests(doctest.DocTestSuite(tok))
    return tests


class WordTokenizer:
    """One token per whitespace-separated word — makes expected counts exact."""

    name = "one-token-per-word"

    def count(self, text: str) -> int:
        return len(text.split())


def sentences(count: int, words_each: int = 10) -> str:
    """Build `count` sentences of `words_each` words (including the full stop)."""
    return " ".join(
        " ".join([f"W{index}"] * (words_each - 1)) + " end."
        for index in range(count)
    )


def make_document(paragraph_texts: list[str], title: str = "Chapter 1: Motion") -> dict:
    return {
        "document_id": "DOC-1",
        "chapters": [
            {
                "chapter_id": "CH-1",
                "title": title,
                "source_page_start": 5,
                "source_page_end": 9,
                "paragraphs": [
                    {"paragraph_id": f"P-{index}", "text": text}
                    for index, text in enumerate(paragraph_texts)
                ],
            }
        ],
    }


class TokenizerTests(unittest.TestCase):
    def test_approximate_counts_are_deterministic(self):
        tokenizer = tok.ApproximateTokenizer()
        text = "Thermodynamics describes energy transfer between systems."
        self.assertEqual(tokenizer.count(text), tokenizer.count(text))
        self.assertGreater(tokenizer.count(text), len(text.split()))

    def test_fallback_when_transformers_missing(self):
        if importlib.util.find_spec("transformers") is not None:
            self.skipTest("transformers is installed")

        self.assertIsInstance(tok.load_tokenizer(), tok.ApproximateTokenizer)

        with self.assertRaises(RuntimeError):
            tok.load_tokenizer(allow_fallback=False)

    def test_overlap_resolution(self):
        self.assertEqual(tok.overlap_token_count(1000), 100)
        self.assertEqual(tok.overlap_token_count(500, 0), 0)
        self.assertEqual(tok.overlap_token_count(500, 10_000), 250)
        self.assertEqual(tok.overlap_token_count(500, -5), 0)


class PackingTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = WordTokenizer()

    def test_no_window_exceeds_the_limit(self):
        for window_size in (240, 500):
            groups = tok.pack_sentences(
                tok.split_sentences(sentences(200)),
                window_size,
                tok.overlap_token_count(window_size),
                self.tokenizer,
            )
            for group in groups:
                self.assertLessEqual(
                    self.tokenizer.count(" ".join(group)), window_size
                )

    def test_every_sentence_appears_at_least_once(self):
        source = tok.split_sentences(sentences(40))
        groups = tok.pack_sentences(source, 100, 20, self.tokenizer)
        packed = {sentence for group in groups for sentence in group}
        self.assertEqual(packed, set(source))

    def test_overlap_is_shared_between_neighbours(self):
        source = tok.split_sentences(sentences(40))
        groups = tok.pack_sentences(source, 100, 30, self.tokenizer)
        self.assertGreater(len(groups), 1)
        for previous, current in zip(groups, groups[1:]):
            shared = set(previous) & set(current)
            self.assertTrue(shared, "consecutive windows should overlap")
            self.assertLessEqual(
                self.tokenizer.count(" ".join(shared)), 30
            )

    def test_zero_overlap_produces_disjoint_windows(self):
        source = tok.split_sentences(sentences(40))
        groups = tok.pack_sentences(source, 100, 0, self.tokenizer)
        for previous, current in zip(groups, groups[1:]):
            self.assertFalse(set(previous) & set(current))

    def test_packing_always_advances(self):
        # a pathological case: overlap as large as the window is allowed to ask for
        source = tok.split_sentences(sentences(20, words_each=50))
        groups = tok.pack_sentences(source, 50, 25, self.tokenizer)
        self.assertEqual(len(groups), 20)

    def test_sentence_longer_than_window_is_split_at_word_boundaries(self):
        long_sentence = " ".join(["alpha"] * 1200) + "."
        groups = tok.pack_sentences([long_sentence], 500, 50, self.tokenizer)
        rebuilt = " ".join(" ".join(group) for group in groups)
        self.assertEqual(rebuilt.count("alpha"), 1200)
        for group in groups:
            self.assertLessEqual(self.tokenizer.count(" ".join(group)), 500)


class DocumentTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = WordTokenizer()

    def windows_for(self, document, window_size=500, overlap=None):
        return tok.tokenize_document(
            document,
            Path("book_cleaned.json"),
            window_size,
            self.tokenizer,
            overlap_tokens=overlap,
        )

    def test_metadata_is_carried_through(self):
        windows = self.windows_for(make_document([sentences(100)]))
        self.assertGreater(len(windows), 1)
        for index, window in enumerate(windows):
            self.assertEqual(window.window_index, index)
            self.assertEqual(window.document_id, "DOC-1")
            self.assertEqual(window.chapter_title, "Chapter 1: Motion")
            self.assertEqual(window.source_page_start, 5)
            self.assertEqual(window.tokenizer, "one-token-per-word")
            self.assertEqual(window.window_size, 500)
            self.assertEqual(window.source, tok.SOURCE_LABEL)
            self.assertEqual(window.paragraph_ids, ["P-0"])

    def test_windows_never_span_two_chapters(self):
        document = {
            "chapters": [
                {"chapter_id": "CH-1", "title": "One", "paragraphs": [sentences(3)]},
                {"chapter_id": "CH-2", "title": "Two", "paragraphs": [sentences(3)]},
            ]
        }
        windows = self.windows_for(document, window_size=500)
        self.assertEqual([w.chapter_id for w in windows], ["CH-1", "CH-2"])

    def test_undersized_tail_is_folded_into_previous_window(self):
        # 120 words then a 5-word paragraph: the tail is below MINIMUM_WINDOW_TOKENS
        document = make_document([sentences(12), "One two three four end."])
        windows = self.windows_for(document, window_size=500)
        self.assertEqual(len(windows), 1)
        self.assertIn("One two three four end.", windows[0].text)

    def test_window_ids_are_stable_and_unique(self):
        document = make_document([sentences(100)])
        first = [w.window_id for w in self.windows_for(document)]
        second = [w.window_id for w in self.windows_for(document)]
        self.assertEqual(first, second)
        self.assertEqual(len(first), len(set(first)))

    def test_500_produces_more_windows_than_1000(self):
        document = make_document([sentences(400)])
        five_hundred = self.windows_for(document, 500)
        one_thousand = self.windows_for(document, 1000)
        self.assertGreater(len(five_hundred), len(one_thousand))
        for window in five_hundred:
            self.assertLessEqual(window.token_count, 500)
        for window in one_thousand:
            self.assertLessEqual(window.token_count, 1000)

    def test_empty_and_malformed_documents(self):
        self.assertEqual(self.windows_for({}), [])
        self.assertEqual(self.windows_for({"chapters": [{"paragraphs": []}]}), [])
        self.assertEqual(self.windows_for({"chapters": "not a list"}), [])


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.inp = Path(self.tmp.name) / "cleaned_data"
        self.out = Path(self.tmp.name) / "tokenized_data"
        self.inp.mkdir()
        (self.inp / "a_cleaned.json").write_text(
            json.dumps(make_document([sentences(150)])), encoding="utf-8"
        )
        with (self.inp / "b_cleaned.jsonl").open("w", encoding="utf-8") as file:
            file.write(json.dumps(make_document([sentences(60)], "Chapter 2")) + "\n")

    def tearDown(self):
        self.tmp.cleanup()

    def run_it(self, **kwargs):
        options = dict(tokenizer=WordTokenizer(), verbose=False)
        options.update(kwargs)
        return tok.run_pipeline(self.inp, self.out, **options)

    def test_end_to_end_500(self):
        manifest = self.run_it(window_size=500)
        output = self.out / "token_windows_500.jsonl"
        records = [json.loads(line) for line in output.read_text("utf-8").splitlines()]

        self.assertEqual(manifest["failed_files"], 0)
        self.assertEqual(manifest["successful_files"], 2)
        self.assertEqual(manifest["records_written"], len(records))
        self.assertEqual(manifest["statistics"]["window_count"], len(records))
        self.assertLessEqual(manifest["statistics"]["maximum_tokens"], 500)
        self.assertEqual(manifest["overlap_tokens"], 50)

        # window_index is a single sequence across all files
        self.assertEqual(
            [r["window_index"] for r in records], list(range(len(records)))
        )
        self.assertEqual({r["window_size"] for r in records}, {500})

    def test_end_to_end_1000(self):
        manifest = self.run_it(window_size=1000)
        self.assertTrue((self.out / "token_windows_1000.jsonl").exists())
        self.assertEqual(manifest["window_size"], 1000)
        self.assertEqual(manifest["overlap_tokens"], 100)
        self.assertLessEqual(manifest["statistics"]["maximum_tokens"], 1000)

    def test_both_sizes_can_coexist(self):
        self.run_it(window_size=500)
        self.run_it(window_size=1000)
        written = sorted(p.name for p in self.out.iterdir())
        self.assertEqual(
            written,
            [
                "token_windows_1000.jsonl",
                "token_windows_500.jsonl",
                "tokenization_manifest.json",
            ],
        )

    def test_broken_file_is_reported_not_fatal(self):
        (self.inp / "broken.json").write_text("{not json", encoding="utf-8")
        manifest = self.run_it()
        self.assertEqual(manifest["failed_files"], 1)
        self.assertEqual(manifest["successful_files"], 2)
        failure = next(f for f in manifest["files"] if f["status"] == "failed")
        self.assertIn("JSONDecodeError", failure["error"])

    def test_rerun_is_idempotent(self):
        first = self.run_it()
        content = (self.out / "token_windows_500.jsonl").read_text("utf-8")
        second = self.run_it()
        self.assertEqual(
            content, (self.out / "token_windows_500.jsonl").read_text("utf-8")
        )
        self.assertEqual(first["records_written"], second["records_written"])

    def test_approximate_tokenizer_is_flagged_in_the_manifest(self):
        manifest = self.run_it(tokenizer=tok.ApproximateTokenizer())
        self.assertFalse(manifest["exact_token_counts"])
        self.assertEqual(manifest["tokenizer"], "approximate")

    def test_no_temporary_files_left(self):
        self.run_it()
        self.assertFalse([p for p in self.out.iterdir() if p.name.startswith(".")])

    def test_missing_input_folder_raises(self):
        with self.assertRaises(FileNotFoundError):
            tok.run_pipeline(
                Path(self.tmp.name) / "nope", self.out, tokenizer=WordTokenizer()
            )


class CommandLineTests(unittest.TestCase):
    def test_defaults_and_overrides(self):
        self.assertEqual(tok.parse_arguments([]).tokens, tok.DEFAULT_WINDOW_SIZE)
        self.assertIsNone(tok.parse_arguments([]).overlap)
        options = tok.parse_arguments(
            ["--tokens", "1000", "--overlap", "0", "--exact-tokens"]
        )
        self.assertEqual((options.tokens, options.overlap), (1000, 0))
        self.assertTrue(options.exact_tokens)

    def test_both_documented_sizes_are_accepted(self):
        for size in tok.WINDOW_SIZE_CHOICES:
            self.assertEqual(
                tok.parse_arguments(["--tokens", str(size)]).tokens, size
            )


@unittest.skipUnless(
    importlib.util.find_spec("transformers") is not None,
    "transformers not installed",
)
class RealTokenizerTests(unittest.TestCase):
    def test_real_tokenizer_respects_the_window(self):
        tokenizer = tok.load_tokenizer()
        self.assertNotEqual(tokenizer.name, "approximate")
        windows = tok.tokenize_document(
            make_document([sentences(200)]), Path("b.json"), 500, tokenizer
        )
        for window in windows:
            self.assertLessEqual(tokenizer.count(window.text), 500)


if __name__ == "__main__":
    unittest.main(verbosity=2)
