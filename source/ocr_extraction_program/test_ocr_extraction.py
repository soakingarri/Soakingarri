#!/usr/bin/env python3
"""
Test cases for ocr_extraction.py

Run with either:
    python3 test_ocr_extraction.py          (unittest, no extra installs)
    python3 -m pytest test_ocr_extraction.py

Every function in ocr_extraction.py carries doctest examples in its
docstring where that is deterministic to do; this file runs all of those
plus a handful of end-to-end checks. Tests that need optional libraries
(PyMuPDF, pytesseract, the Tesseract engine) are skipped automatically
when those are not installed.
"""

from __future__ import annotations

import doctest
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import ocr_extraction as ocr


def load_tests(loader, tests, ignore):
    """Hook so `python3 test_ocr_extraction.py` also runs every docstring
    example."""
    tests.addTests(doctest.DocTestSuite(ocr))
    return tests


def has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def has_tesseract() -> bool:
    if not has_module("pytesseract"):
        return False

    import pytesseract

    try:
        pytesseract.get_tesseract_version()
    except Exception:
        return False

    return True


def make_scanned_pdf(folder: Path, text: str) -> Path:
    """Build a PDF page with no text layer: the text is drawn onto a
    plain image and inserted as a picture, the way a scanned or
    photographed page would come out of a scanner. Mirrors
    data_cleaning_program/test_data_cleaning.py's OcrFallbackTests
    pattern, reused here so this module doesn't need its own real
    scanned document to test against."""
    import fitz
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (800, 200), "white")
    draw = ImageDraw.Draw(image)
    draw.text((20, 80), text, fill="black")

    image_path = folder / "scan.png"
    image.save(image_path)

    document = fitz.open()
    page = document.new_page(width=800, height=200)
    page.insert_image(page.rect, filename=str(image_path))

    pdf_path = folder / "scanned.pdf"
    document.save(pdf_path)
    document.close()

    return pdf_path


class BlockClassificationTests(unittest.TestCase):
    """Pure-text heuristics; no OCR or optional libraries involved, so
    these always run."""

    def test_body_prose_stays_one_block(self):
        text = "A plain sentence about history.\nAnother plain sentence follows."
        blocks = ocr.classify_text_blocks(text)
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0]["type"], "body")

    def test_figure_caption_detected_with_continuation(self):
        text = "Figure 2: Population growth.\nSource: UN Statistics Division."
        blocks = ocr.classify_text_blocks(text)
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0]["type"], "figure_caption")
        self.assertIn("Figure 2", blocks[0]["text"])
        self.assertIn("UN Statistics Division", blocks[0]["text"])

    def test_table_run_requires_two_consistent_lines(self):
        text = "1   10   5.99\n2   20   3.50\n3   15   7.25"
        blocks = ocr.classify_text_blocks(text)
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0]["type"], "table")

    def test_single_numeric_line_is_not_a_table(self):
        text = "Only one short numeric-ish line here: 1 2 3"
        blocks = ocr.classify_text_blocks(text)
        self.assertTrue(all(b["type"] == "body" for b in blocks))

    def test_code_block_detected(self):
        text = "def is_even(n):\n    return n % 2 == 0"
        blocks = ocr.classify_text_blocks(text)
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0]["type"], "code")

    def test_mixed_page_produces_multiple_block_types(self):
        text = (
            "This chapter introduces the dataset used throughout.\n"
            "Table 1: Sample counts by region.\n"
            "Region   Count   Pct\n"
            "North    120     40%\n"
            "South    80      27%\n"
            "West     100     33%\n"
            "\n"
            "The analysis continues below with a short discussion."
        )
        blocks = ocr.classify_text_blocks(text)
        types = [b["type"] for b in blocks]
        self.assertIn("figure_caption", types)
        self.assertIn("body", types)


class ConfidenceHelperTests(unittest.TestCase):
    def test_is_low_confidence_page_threshold(self):
        self.assertTrue(ocr.is_low_confidence_page(10.0))
        self.assertFalse(ocr.is_low_confidence_page(95.0))
        self.assertFalse(ocr.is_low_confidence_page(60.0))
        self.assertTrue(ocr.is_low_confidence_page(59.9))


class DiscoverAndPathTests(unittest.TestCase):
    def test_discover_pdfs_skips_unsupported_and_hidden(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "scan.pdf").write_bytes(b"%PDF-1.4")
            (root / "notes.txt").write_text("hi", encoding="utf-8")
            (root / ".hidden.pdf").write_bytes(b"%PDF-1.4")
            (root / "~$lock.pdf").write_bytes(b"%PDF-1.4")
            names = [p.name for p in ocr.discover_pdfs(root)]
            self.assertEqual(names, ["scan.pdf"])

    def test_safe_output_path_flattens_nested_folders(self):
        path = ocr.safe_output_path(
            Path("/in/africa/history.pdf"), Path("/in"), Path("/out")
        )
        self.assertEqual(path.name, "africa__history_ocr.json")

    def test_safe_output_path_flat_input(self):
        path = ocr.safe_output_path(
            Path("/in/history.pdf"), Path("/in"), Path("/out")
        )
        self.assertEqual(path.name, "history_ocr.json")


class ArgumentParsingTests(unittest.TestCase):
    def test_defaults(self):
        options = ocr.parse_arguments([])
        self.assertEqual(options.dpi, ocr.DEFAULT_DPI)
        self.assertEqual(options.input, ocr.INPUT_FOLDER)
        self.assertEqual(options.output, ocr.OUTPUT_FOLDER)

    def test_overrides(self):
        options = ocr.parse_arguments(
            ["--dpi", "150", "--input", "/tmp/in", "--output", "/tmp/out"]
        )
        self.assertEqual(options.dpi, 150)
        self.assertEqual(options.input, Path("/tmp/in"))
        self.assertEqual(options.output, Path("/tmp/out"))


@unittest.skipUnless(has_module("fitz"), "PyMuPDF not installed")
@unittest.skipUnless(has_tesseract(), "Tesseract OCR engine not installed")
class OcrEndToEndTests(unittest.TestCase):
    def test_recognize_text_recovers_words(self):
        from PIL import Image, ImageDraw

        image = Image.new("RGB", (400, 100), "white")
        draw = ImageDraw.Draw(image)
        draw.text((10, 30), "PLAIN TEXT TEST", fill="black")

        text = ocr.recognize_text(image)
        self.assertIn("TEXT", text.upper())

    def test_recognize_with_confidence_shape(self):
        from PIL import Image, ImageDraw

        image = Image.new("RGB", (400, 100), "white")
        draw = ImageDraw.Draw(image)
        draw.text((10, 30), "CONFIDENCE CHECK", fill="black")

        result = ocr.recognize_with_confidence(image)
        self.assertIsInstance(result, ocr.OcrPageResult)
        self.assertIn("CONFIDENCE", result.text.upper())
        self.assertGreaterEqual(result.mean_confidence, 0.0)
        self.assertLessEqual(result.mean_confidence, 100.0)
        self.assertTrue(all(isinstance(w, ocr.OcrWord) for w in result.words))

    def test_ocr_pdf_returns_page_dicts(self):
        with tempfile.TemporaryDirectory() as folder:
            path = make_scanned_pdf(Path(folder), "HELLO OCR WORLD")
            pages = ocr.ocr_pdf(path)

            self.assertEqual(len(pages), 1)
            page = pages[0]
            self.assertEqual(page["page_number"], 1)
            self.assertIn("HELLO", page["text"].upper())
            self.assertIn("mean_confidence", page)
            self.assertIn("is_low_confidence", page)
            self.assertIn("blocks", page)
            self.assertIsInstance(page["blocks"], list)

    def test_run_pipeline_writes_output_and_manifest(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            inp, out = root / "raw", root / "ocr_out"
            inp.mkdir()
            make_scanned_pdf(inp, "PIPELINE TEST PAGE")
            # make_scanned_pdf always writes to "scanned.pdf"; rename so
            # discover_pdfs finds it under the real input folder.

            manifest = ocr.run_pipeline(inp, out)

            self.assertEqual(manifest["total_files"], 1)
            self.assertEqual(manifest["successful_files"], 1)
            self.assertEqual(manifest["failed_files"], 0)
            self.assertGreaterEqual(manifest["total_pages"], 1)

            output_files = sorted(p.name for p in out.iterdir())
            self.assertIn("scanned_ocr.json", output_files)
            self.assertIn("ocr_extraction_manifest.json", output_files)

            data = json.loads((out / "scanned_ocr.json").read_text(encoding="utf-8"))
            self.assertEqual(data["page_count"], 1)
            self.assertIn("PIPELINE", data["pages"][0]["text"].upper())

    def test_process_pdf_file_reports_failure_instead_of_raising(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            inp, out = root / "raw", root / "ocr_out"
            inp.mkdir()
            bad = inp / "not_a_real.pdf"
            bad.write_bytes(b"this is not a pdf")

            result = ocr.process_pdf_file(str(bad), inp, out)
            self.assertEqual(result["status"], "failed")
            self.assertIn("error", result)


if __name__ == "__main__":
    unittest.main(verbosity=2)
