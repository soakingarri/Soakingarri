#!/usr/bin/env python3
"""
Test cases for summarization.py

Run with either:
    python3 test_summarization.py          (unittest, no extra installs)
    python3 -m pytest test_summarization.py

Runs every docstring example plus classification, rendering and pipeline checks.
The PDF image-export test is skipped when PyMuPDF is not installed.
"""

from __future__ import annotations

import doctest
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import summarization as smz


def load_tests(loader, tests, ignore):
    """Hook so `python3 test_summarization.py` also runs every docstring example."""
    tests.addTests(doctest.DocTestSuite(smz))
    return tests


TEXTBOOK_PARAGRAPHS = [
    "Stress is defined as the force carried per unit of cross-sectional area. "
    "For a uniform bar this gives sigma = F/A. "
    "The workshop closed early that Friday afternoon.",

    "Figure 4.1: engineering stress plotted against engineering strain for mild steel. "
    "The curve is linear until the yield point. "
    "Young's modulus is the slope of that linear region.",

    "Diagram 4.2: exploded view of the tensile test rig. "
    "A cross-section through the grip assembly is shown alongside. "
    "Everyone agreed the rig looked complicated.",

    "Hooke's law relates the two quantities. "
    "It follows that sigma = E * epsilon below the proportional limit. (4.3)",
]


def make_document(paragraphs=None, **overrides) -> dict:
    document = {
        "document_id": "DOC-1",
        "source_file": "materials.pdf",
        "source_path": "/raw/materials.pdf",
        "chapters": [
            {
                "chapter_id": "CH-4",
                "title": "Chapter 4: Mechanical Properties",
                "source_page_start": 61,
                "source_page_end": 78,
                "paragraphs": [
                    {"paragraph_id": f"P-{index}", "text": text}
                    for index, text in enumerate(
                        paragraphs if paragraphs is not None else TEXTBOOK_PARAGRAPHS
                    )
                ],
            }
        ],
    }
    document.update(overrides)
    return document


class ClassificationTests(unittest.TestCase):
    def test_each_category_is_detected(self):
        cases = {
            "formula": "It follows that v = u + at over the interval.",
            "concept": "Viscosity is defined as resistance to shear flow.",
            "graph": "Figure 2.3: pressure plotted against volume.",
            "technical_drawing": "Schematic 5: the wiring diagram of the controller.",
        }
        for category, sentence in cases.items():
            self.assertIn(category, smz.classify_sentence(sentence), sentence)

    def test_ordinary_prose_is_ignored(self):
        for sentence in (
            "They walked to the market that morning.",
            "The author thanks his colleagues for their patience.",
            "It rained for most of the week.",
        ):
            self.assertEqual(smz.classify_sentence(sentence), [], sentence)

    def test_hyphenated_words_are_not_formulas(self):
        for phrase in ("cross-section", "x-axis", "free-body", "well-known"):
            self.assertEqual(smz.extract_formulas(f"The {phrase} is shown."), [])

    def test_one_sentence_can_carry_two_categories(self):
        categories = smz.classify_sentence(
            "Since F = ma, the graph of force against acceleration is a straight line."
        )
        self.assertEqual(categories, ["formula", "graph"])

    def test_categories_are_returned_in_canonical_order(self):
        categories = smz.classify_sentence(
            "Impedance is defined as V = IR, plotted against frequency in the schematic."
        )
        self.assertEqual(categories, sorted(categories, key=smz.CATEGORIES.index))

    def test_latex_maths_is_preferred_over_plain_patterns(self):
        self.assertEqual(
            smz.extract_formulas(r"We write $E = mc^2$ and also E = mc^2 again."),
            ["$E = mc^2$"],
        )

    def test_equation_numbers_count_as_formulas(self):
        self.assertIn("formula", smz.classify_sentence("The result follows. (4.12)"))

    def test_caption_kinds_route_to_the_right_category(self):
        self.assertIn("graph", smz.classify_sentence("Chart 2: sales by quarter"))
        self.assertIn(
            "technical_drawing",
            smz.classify_sentence("Plate 7: cutaway of the gearbox"),
        )


class ScanTests(unittest.TestCase):
    def setUp(self):
        self.findings = smz.scan_document(
            make_document(), Path("materials_cleaned.json")
        )

    def test_all_four_categories_found(self):
        found = {finding.category for finding in self.findings}
        self.assertEqual(found, set(smz.CATEGORIES))

    def test_prose_sentences_are_absent(self):
        joined = " ".join(f.context for f in self.findings)
        self.assertNotIn("workshop closed early", joined)
        self.assertNotIn("looked complicated", joined)
        self.assertNotIn("rained", joined)

    def test_metadata_is_attached(self):
        for finding in self.findings:
            self.assertEqual(finding.source_file, "materials_cleaned.json")
            self.assertEqual(finding.document_id, "DOC-1")
            self.assertEqual(finding.chapter_id, "CH-4")
            self.assertEqual(finding.source_page_start, 61)
            self.assertEqual(finding.source, smz.SOURCE_LABEL)
            self.assertTrue(finding.paragraph_id)
            self.assertTrue(finding.finding_id.startswith("FND-"))

    def test_duplicate_findings_are_collapsed(self):
        repeated = ["Entropy is defined as disorder."] * 5
        findings = smz.scan_document(
            make_document(repeated), Path("b.json")
        )
        self.assertEqual(len(findings), 1)

    def test_finding_ids_are_stable(self):
        again = smz.scan_document(make_document(), Path("materials_cleaned.json"))
        self.assertEqual(
            [f.finding_id for f in self.findings], [f.finding_id for f in again]
        )

    def test_category_filter(self):
        only_formulas = smz.scan_document(
            make_document(), Path("b.json"), categories=["formula"]
        )
        self.assertTrue(only_formulas)
        self.assertEqual({f.category for f in only_formulas}, {"formula"})

    def test_empty_and_malformed_documents(self):
        self.assertEqual(smz.scan_document({}, Path("b.json")), [])
        self.assertEqual(
            smz.scan_document({"chapters": "not a list"}, Path("b.json")), []
        )
        self.assertEqual(
            smz.scan_document(
                make_document(["Nothing technical happened here at all."]),
                Path("b.json"),
            ),
            [],
        )


class RenderTests(unittest.TestCase):
    def setUp(self):
        self.findings = smz.scan_document(make_document(), Path("m_cleaned.json"))
        self.text = smz.render_summary(self.findings, "m_cleaned.json", "DOC-1")

    def test_headings_and_counts(self):
        grouped = smz.group_by_category(self.findings)
        for category, items in grouped.items():
            self.assertIn(
                f"{smz.CATEGORY_HEADINGS[category]} ({len(items)})", self.text
            )

    def test_categories_appear_in_canonical_order(self):
        positions = [
            self.text.index(smz.CATEGORY_HEADINGS[category])
            for category in smz.CATEGORIES
            if smz.CATEGORY_HEADINGS[category] in self.text
        ]
        self.assertEqual(positions, sorted(positions))

    def test_locations_are_shown(self):
        self.assertIn("pp. 61-78 | Chapter 4: Mechanical Properties", self.text)

    def test_prose_is_not_carried_into_the_summary(self):
        self.assertNotIn("workshop closed early", self.text)

    def test_empty_findings_produce_a_readable_note(self):
        text = smz.render_summary([], "empty.json")
        self.assertIn("No formulas, concepts, graphs or technical drawings", text)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.inp = Path(self.tmp.name) / "cleaned_data"
        self.out = Path(self.tmp.name) / "summarized_data"
        self.inp.mkdir()
        (self.inp / "materials_cleaned.json").write_text(
            json.dumps(make_document()), encoding="utf-8"
        )
        (self.inp / "notes_cleaned.jsonl").write_text(
            json.dumps(make_document(["Torque is defined as force times distance."]))
            + "\n",
            encoding="utf-8",
        )

    def tearDown(self):
        self.tmp.cleanup()

    def run_it(self, **kwargs):
        options = dict(verbose=False)
        options.update(kwargs)
        return smz.run_pipeline(self.inp, self.out, **options)

    def test_end_to_end(self):
        manifest = self.run_it()

        self.assertEqual(manifest["failed_files"], 0)
        self.assertEqual(manifest["successful_files"], 2)
        self.assertEqual(set(manifest["findings_by_category"]), set(smz.CATEGORIES))
        self.assertEqual(
            sum(manifest["findings_by_category"].values()), manifest["total_findings"]
        )

        written = sorted(p.name for p in self.out.iterdir())
        self.assertEqual(
            written,
            [
                "materials_findings.jsonl",
                "materials_summary.txt",
                "notes_findings.jsonl",
                "notes_summary.txt",
                "summarization_manifest.json",
            ],
        )

        summary = (self.out / "materials_summary.txt").read_text("utf-8")
        self.assertTrue(summary.startswith("TECHNICAL SUMMARY"))
        self.assertIn(smz.SOURCE_LABEL, summary)

        findings = [
            json.loads(line)
            for line in (self.out / "materials_findings.jsonl")
            .read_text("utf-8")
            .splitlines()
        ]
        self.assertTrue(findings)
        self.assertEqual(
            len(findings),
            next(
                f["findings"]
                for f in manifest["files"]
                if f["input_file"].endswith("materials_cleaned.json")
            ),
        )

    def test_summary_file_naming_strips_cleaned_suffix(self):
        self.run_it()
        self.assertTrue((self.out / "materials_summary.txt").exists())
        self.assertFalse((self.out / "materials_cleaned_summary.txt").exists())

    def test_category_selection_is_recorded_and_applied(self):
        manifest = self.run_it(categories=["formula", "graph"])
        self.assertEqual(manifest["categories"], ["formula", "graph"])
        self.assertEqual(
            set(manifest["findings_by_category"]), {"formula", "graph"}
        )
        summary = (self.out / "materials_summary.txt").read_text("utf-8")
        self.assertNotIn(smz.CATEGORY_HEADINGS["concept"], summary)

    def test_invalid_category_selection_raises(self):
        with self.assertRaises(ValueError):
            self.run_it(categories=["nonsense"])

    def test_broken_file_is_reported_not_fatal(self):
        (self.inp / "broken.json").write_text("{not json", encoding="utf-8")
        manifest = self.run_it()
        self.assertEqual(manifest["failed_files"], 1)
        self.assertEqual(manifest["successful_files"], 2)

    def test_rerun_is_idempotent(self):
        self.run_it()
        first = (self.out / "materials_summary.txt").read_text("utf-8")
        self.run_it()
        self.assertEqual(first, (self.out / "materials_summary.txt").read_text("utf-8"))

    def test_no_temporary_files_left(self):
        self.run_it()
        self.assertFalse([p for p in self.out.iterdir() if p.name.startswith(".")])

    def test_missing_input_folder_raises(self):
        with self.assertRaises(FileNotFoundError):
            smz.run_pipeline(Path(self.tmp.name) / "nope", self.out, verbose=False)

    def test_export_images_without_source_pdf_is_harmless(self):
        manifest = self.run_it(export_images=True)
        self.assertTrue(manifest["images_exported"])
        for result in manifest["files"]:
            self.assertEqual(result["exported_images"], 0)


class CommandLineTests(unittest.TestCase):
    def test_defaults(self):
        options = smz.parse_arguments([])
        self.assertEqual(options.categories, list(smz.CATEGORIES))
        self.assertFalse(options.export_images)

    def test_category_subset(self):
        options = smz.parse_arguments(["--categories", "graph", "technical_drawing"])
        self.assertEqual(options.categories, ["graph", "technical_drawing"])

    def test_unknown_category_is_rejected(self):
        with self.assertRaises(SystemExit):
            smz.parse_arguments(["--categories", "poetry"])


@unittest.skipUnless(
    importlib.util.find_spec("fitz") is not None, "PyMuPDF not installed"
)
class ImageExportTests(unittest.TestCase):
    def test_images_are_written_from_the_named_pages(self):
        import fitz

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            pdf_path = root / "source.pdf"

            document = fitz.open()
            for _ in range(3):
                document.new_page()

            pixmap = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 40, 40))
            pixmap.set_rect(pixmap.irect, (200, 40, 40))
            document[1].insert_image(fitz.Rect(10, 10, 90, 90), pixmap=pixmap)
            document.save(pdf_path)
            document.close()

            written = smz.export_figure_images(pdf_path, [2], root / "images")
            self.assertEqual(len(written), 1)
            self.assertTrue(Path(written[0]).exists())
            self.assertIn("page_2_", written[0])

            # a page with no images, and a page number out of range
            self.assertEqual(
                smz.export_figure_images(pdf_path, [1, 99], root / "images2"), []
            )

    def test_missing_or_non_pdf_source_returns_empty(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.assertEqual(
                smz.export_figure_images(root / "gone.pdf", [1], root / "out"), []
            )
            text_file = root / "book.txt"
            text_file.write_text("not a pdf")
            self.assertEqual(
                smz.export_figure_images(text_file, [1], root / "out"), []
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
