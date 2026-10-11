#!/usr/bin/env python3
"""
Test cases for deduplication.py

Run with either:
    python3 test_deduplication.py          (unittest, no extra installs)
    python3 -m pytest test_deduplication.py

Every function in deduplication.py carries doctest examples in its
docstring; this file runs all of those plus a handful of end-to-end
checks.
"""

from __future__ import annotations

import doctest
import json
import tempfile
import unittest
from pathlib import Path

import deduplication as dd


def load_tests(loader, tests, ignore):
    """Hook so `python3 test_deduplication.py` also runs every docstring example."""
    tests.addTests(doctest.DocTestSuite(dd))
    return tests


# A long passage (well over the ~25-word floor where a single changed word
# still leaves estimated Jaccard similarity comfortably above the default
# 0.85 near-duplicate threshold) and a one-word reworded copy of it.
BASE_PASSAGE = (
    "In many African societies time was traditionally understood "
    "through the rhythm of agricultural seasons rather than through "
    "fixed clock hours and elders passed this understanding down "
    "through oral tradition across many generations of herders and "
    "farmers alike who relied on the position of the sun and the "
    "changing of the weather to plan planting harvesting and the "
    "movement of cattle between dry season and wet season grazing "
    "lands throughout the year while also watching the stars at "
    "night to predict the coming of the rains and to decide when "
    "ceremonies marking the new season should begin for the whole "
    "community to gather and celebrate together near the riverbanks "
    "where the soil was richest and the cattle could graze safely "
    "away from predators that roamed the drier hills further inland "
    "during the hottest months of the dry season every single year"
)
REWORDED_PASSAGE = BASE_PASSAGE.replace("understanding", "knowledge", 1)
UNRELATED_PASSAGE = (
    "A completely different paragraph discussing colonial trade "
    "routes along the coast of West Africa during the nineteenth "
    "century and the economic consequences for local merchants and "
    "port cities that depended heavily on the export of palm oil "
    "rubber and other goods"
)


def make_document(document_id: str, chapter_id: str, paragraphs: list[tuple[str, str]]) -> dict:
    """Build a minimal cleaned-document dict with one chapter."""
    return {
        "document_id": document_id,
        "chapters": [
            {
                "chapter_id": chapter_id,
                "paragraphs": [
                    {"paragraph_id": paragraph_id, "text": text}
                    for paragraph_id, text in paragraphs
                ],
            }
        ],
    }


class ExactDuplicateTests(unittest.TestCase):
    def test_exact_duplicate_across_two_documents(self):
        documents = [
            make_document("DOC-1", "CH-1", [("P-1", "A shared sentence here."), ("P-2", "Unique to book one.")]),
            make_document("DOC-2", "CH-2", [("P-3", "A shared sentence here."), ("P-4", "Unique to book two.")]),
        ]
        duplicates = dd.find_exact_duplicates(documents)
        self.assertEqual(duplicates, {"P-3": "P-1"})

    def test_exact_duplicate_ignores_whitespace_differences(self):
        documents = [
            make_document("DOC-1", "CH-1", [("P-1", "Hello   world.")]),
            make_document("DOC-2", "CH-2", [("P-2", "Hello world.")]),
        ]
        duplicates = dd.find_exact_duplicates(documents)
        self.assertEqual(duplicates, {"P-2": "P-1"})

    def test_keep_first_occurrence_ordering(self):
        """A chain of three identical paragraphs across three documents all
        point back to the very first one, not to each other."""
        documents = [
            make_document("DOC-1", "CH-1", [("P-1", "Repeated passage.")]),
            make_document("DOC-2", "CH-2", [("P-2", "Repeated passage.")]),
            make_document("DOC-3", "CH-3", [("P-3", "Repeated passage.")]),
        ]
        duplicates = dd.find_exact_duplicates(documents)
        self.assertEqual(duplicates, {"P-2": "P-1", "P-3": "P-1"})
        self.assertNotIn("P-1", duplicates)

    def test_distinct_text_not_flagged(self):
        documents = [
            make_document("DOC-1", "CH-1", [("P-1", "First distinct sentence."), ("P-2", "Second distinct sentence.")]),
        ]
        self.assertEqual(dd.find_exact_duplicates(documents), {})


class NearDuplicateTests(unittest.TestCase):
    def test_reworded_passage_flagged_above_threshold(self):
        documents = [
            make_document(
                "DOC-1",
                "CH-1",
                [("P-1", BASE_PASSAGE), ("P-2", REWORDED_PASSAGE)],
            )
        ]
        near_duplicates = dd.find_near_duplicates(documents)
        self.assertEqual(near_duplicates, {"P-2": "P-1"})

    def test_unrelated_passage_not_flagged(self):
        documents = [
            make_document(
                "DOC-1",
                "CH-1",
                [("P-1", BASE_PASSAGE), ("P-2", UNRELATED_PASSAGE)],
            )
        ]
        self.assertEqual(dd.find_near_duplicates(documents), {})

    def test_excluded_ids_are_skipped(self):
        """A paragraph already flagged as an exact duplicate is not
        re-flagged as a near-duplicate, and does not become an insertion
        in the LSH index that some later paragraph could match against."""
        documents = [
            make_document(
                "DOC-1",
                "CH-1",
                [("P-1", BASE_PASSAGE), ("P-2", BASE_PASSAGE)],
            )
        ]
        near_duplicates = dd.find_near_duplicates(documents, exclude={"P-2"})
        self.assertEqual(near_duplicates, {})

    def test_chain_of_near_duplicates_points_to_earliest(self):
        """Several reworded copies all point back to the first occurrence,
        not to the previous reworded copy."""
        second_reworded = BASE_PASSAGE.replace("elders", "women", 1)
        documents = [
            make_document(
                "DOC-1",
                "CH-1",
                [
                    ("P-1", BASE_PASSAGE),
                    ("P-2", REWORDED_PASSAGE),
                    ("P-3", second_reworded),
                ],
            )
        ]
        near_duplicates = dd.find_near_duplicates(documents)
        self.assertEqual(near_duplicates.get("P-2"), "P-1")
        self.assertEqual(near_duplicates.get("P-3"), "P-1")


class PruneTests(unittest.TestCase):
    def test_empty_chapter_is_dropped(self):
        document = make_document("DOC-1", "CH-1", [("P-1", "a"), ("P-2", "b")])
        pruned = dd.prune_duplicate_paragraphs(document, {"P-1", "P-2"})
        self.assertIsNone(pruned)

    def test_chapter_survives_if_one_paragraph_remains(self):
        document = {
            "document_id": "DOC-1",
            "chapters": [
                {"chapter_id": "CH-1", "paragraphs": [{"paragraph_id": "P-1", "text": "a"}, {"paragraph_id": "P-2", "text": "b"}]},
                {"chapter_id": "CH-2", "paragraphs": [{"paragraph_id": "P-3", "text": "c"}]},
            ],
        }
        pruned = dd.prune_duplicate_paragraphs(document, {"P-1"})
        self.assertEqual([c["chapter_id"] for c in pruned["chapters"]], ["CH-1", "CH-2"])
        self.assertEqual(
            [p["paragraph_id"] for p in pruned["chapters"][0]["paragraphs"]], ["P-2"]
        )

    def test_whole_document_dropped_when_all_chapters_empty(self):
        document = {
            "document_id": "DOC-1",
            "chapters": [
                {"chapter_id": "CH-1", "paragraphs": [{"paragraph_id": "P-1", "text": "a"}]},
                {"chapter_id": "CH-2", "paragraphs": [{"paragraph_id": "P-2", "text": "b"}]},
            ],
        }
        self.assertIsNone(dd.prune_duplicate_paragraphs(document, {"P-1", "P-2"}))


class EndToEndTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.inp = self.root / "cleaned"
        self.out = self.root / "deduped"
        self.inp.mkdir()

    def tearDown(self):
        self.tmp.cleanup()

    def write_cleaned_file(self, name: str, document: dict) -> None:
        (self.inp / name).write_text(json.dumps(document), encoding="utf-8")

    def test_deduplicate_corpus_removes_exact_and_near_duplicates(self):
        self.write_cleaned_file(
            "one_cleaned.json",
            make_document(
                "DOC-1",
                "CH-1",
                [
                    ("P-1", "An exact duplicate sentence shared by two books."),
                    ("P-2", BASE_PASSAGE),
                    ("P-3", "Only ever appears in book one."),
                ],
            ),
        )
        self.write_cleaned_file(
            "two_cleaned.json",
            make_document(
                "DOC-2",
                "CH-2",
                [
                    ("P-4", "An exact duplicate sentence shared by two books."),
                    ("P-5", REWORDED_PASSAGE),
                ],
            ),
        )

        manifest = dd.deduplicate_corpus(self.inp, self.out)

        self.assertEqual(manifest["total_paragraphs"], 5)
        self.assertEqual(manifest["exact_duplicates_removed"], 1)
        self.assertEqual(manifest["near_duplicates_removed"], 1)

        reasons = {entry["paragraph_id"]: entry["reason"] for entry in manifest["removed"]}
        self.assertEqual(reasons, {"P-4": "exact", "P-5": "near"})

        manifest_path = self.out / "deduplication_manifest.json"
        self.assertTrue(manifest_path.exists())
        self.assertFalse(manifest_path.with_suffix(".json.tmp").exists())
        on_disk_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(on_disk_manifest, manifest)

        # book one survives whole; book two loses both of its paragraphs
        # (both were duplicates), so it is dropped entirely.
        output_files = sorted(p.name for p in self.out.glob("*_cleaned.json"))
        self.assertEqual(output_files, ["one_cleaned.json"])

        surviving = json.loads((self.out / "one_cleaned.json").read_text(encoding="utf-8"))
        surviving_ids = [p["paragraph_id"] for p in surviving["chapters"][0]["paragraphs"]]
        self.assertEqual(surviving_ids, ["P-1", "P-2", "P-3"])

    def test_output_files_are_valid_json_with_no_duplicate_text(self):
        self.write_cleaned_file(
            "only_cleaned.json",
            make_document(
                "DOC-1",
                "CH-1",
                [("P-1", "Repeated line."), ("P-2", "Repeated line."), ("P-3", "Different line.")],
            ),
        )

        dd.deduplicate_corpus(self.inp, self.out)

        data = json.loads((self.out / "only_cleaned.json").read_text(encoding="utf-8"))
        texts = [p["text"] for c in data["chapters"] for p in c["paragraphs"]]
        self.assertEqual(texts, ["Repeated line.", "Different line."])

    def test_no_duplicates_leaves_everything_in_place(self):
        self.write_cleaned_file(
            "only_cleaned.json",
            make_document("DOC-1", "CH-1", [("P-1", "a"), ("P-2", "b")]),
        )

        manifest = dd.deduplicate_corpus(self.inp, self.out)

        self.assertEqual(manifest["exact_duplicates_removed"], 0)
        self.assertEqual(manifest["near_duplicates_removed"], 0)
        self.assertEqual(manifest["removed"], [])
        data = json.loads((self.out / "only_cleaned.json").read_text(encoding="utf-8"))
        self.assertEqual(len(data["chapters"][0]["paragraphs"]), 2)

    def test_discover_cleaned_files_skips_manifest(self):
        self.write_cleaned_file("book_cleaned.json", make_document("DOC-1", "CH-1", [("P-1", "a")]))
        (self.inp / "cleaning_manifest.json").write_text("{}", encoding="utf-8")
        names = [p.name for p in dd.discover_cleaned_files(self.inp)]
        self.assertEqual(names, ["book_cleaned.json"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
