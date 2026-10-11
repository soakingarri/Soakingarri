#!/usr/bin/env python3
"""
Test cases for data_cleaning.py

Run with either:
    python3 test_data_cleaning.py          (unittest, no extra installs)
    python3 -m pytest test_data_cleaning.py

Every function in data_cleaning.py carries doctest examples in its docstring;
this file runs all of those plus a handful of end-to-end checks. Tests that
need optional libraries (PyMuPDF, python-docx, ebooklib) are skipped
automatically when those libraries are not installed.
"""

from __future__ import annotations

import doctest
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import data_cleaning as dc


def load_tests(loader, tests, ignore):
    """Hook so `python3 test_data_cleaning.py` also runs every docstring example."""
    tests.addTests(doctest.DocTestSuite(dc))
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


SAMPLE_BOOK = (
    "My Great Book\n"
    "Contents\n"
    "Chapter 1 ........ 3\n"
    "Chapter 2 ........ 9\n"
    "Chapter 3 ........ 15\n"
    "1\n"
    "\f"
    "My Great Book\n"
    "\u00a9 2021 Example Press. All rights reserved.\n"
    "ISBN 978-0-00-000000-0. Printed in Lagos.\n"
    "2\n"
    "\f"
    "My Great Book\n"
    "Chapter 1: The Start\n"
    "The first sentence of the book runs onto the\n"
    "next line and then finishes here.\n"
    "\n"
    "A second paragraph that is self-con-\n"
    "tained on this page.\n"
    "3\n"
    "\f"
    "My Great Book\n"
    "Chapter 2: The Middle\n"
    "Something happens in the middle of the story.\n"
    "4\n"
    "\f"
    "My Great Book\n"
    "Index\n"
    "apples, 3, 7\n"
    "bananas, 12-14\n"
    "cherries, 20\n"
    "dates, 21\n"
    "5\n"
)


class TextPipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.inp = self.root / "raw"
        self.out = self.root / "clean"
        self.inp.mkdir()
        self.book = self.inp / "great_book.txt"
        self.book.write_text(SAMPLE_BOOK, encoding="utf-8")

    def tearDown(self):
        self.tmp.cleanup()

    def test_extract_text_file_splits_on_form_feed(self):
        pages = dc.extract_pages(self.book)
        self.assertEqual([p.number for p in pages], [1, 2, 3, 4, 5])
        self.assertTrue(pages[0].text.startswith("My Great Book"))

    def test_repeated_header_detected(self):
        pages = dc.extract_pages(self.book)
        repeated = dc.identify_repeated_marginal_lines(pages)
        self.assertIn("my great book", repeated)
        self.assertIn("#", repeated)  # bare page numbers collapse to "#"

    def test_headers_and_page_numbers_removed(self):
        pages = dc.extract_pages(self.book)
        repeated = dc.identify_repeated_marginal_lines(pages)
        cleaned = dc.remove_headers_footers_and_page_numbers(pages[2], repeated)
        self.assertNotIn("My Great Book", cleaned.text)
        self.assertFalse(cleaned.text.endswith("\n3"))
        self.assertIn("Chapter 1: The Start", cleaned.text)

    def test_clean_book_end_to_end(self):
        doc = dc.clean_book(self.book)
        summary = doc["processing_summary"]

        reasons = {r["reason"] for r in summary["removed_pages_or_sections"]}
        self.assertIn("table_of_contents", reasons)
        self.assertIn("copyright_or_legal_page", reasons)
        self.assertIn("index", reasons)

        titles = [c["title"] for c in doc["chapters"]]
        self.assertEqual(titles, ["Chapter 1: The Start", "Chapter 2: The Middle"])

        texts = [p["text"] for c in doc["chapters"] for p in c["paragraphs"]]
        self.assertIn(
            "The first sentence of the book runs onto the next line and then finishes here.",
            texts,
        )
        self.assertIn("A second paragraph that is self-contained on this page.", texts)
        self.assertEqual(summary["paragraph_count"], len(texts))

        for chapter in doc["chapters"]:
            self.assertRegex(chapter["chapter_id"], r"^CH-[0-9a-f]{24}$")
            for paragraph in chapter["paragraphs"]:
                self.assertRegex(paragraph["paragraph_id"], r"^P-[0-9a-f]{24}$")
                self.assertIsInstance(paragraph["page"], int)

    def test_ids_are_stable_across_reruns(self):
        first = dc.clean_book(self.book)
        second = dc.clean_book(self.book)

        self.assertEqual(first["document_id"], second["document_id"])
        self.assertEqual(
            [c["chapter_id"] for c in first["chapters"]],
            [c["chapter_id"] for c in second["chapters"]],
        )
        self.assertEqual(
            [p["paragraph_id"] for c in first["chapters"] for p in c["paragraphs"]],
            [p["paragraph_id"] for c in second["chapters"] for p in c["paragraphs"]],
        )

    def test_paragraphs_carry_their_own_page_number(self):
        doc = dc.clean_book(self.book)
        pages_by_text = {
            p["text"]: p["page"]
            for c in doc["chapters"]
            for p in c["paragraphs"]
        }
        self.assertEqual(
            pages_by_text["A second paragraph that is self-contained on this page."],
            3,
        )
        self.assertEqual(
            pages_by_text["Something happens in the middle of the story."],
            4,
        )

    def test_process_file_writes_valid_json(self):
        result = dc.process_file(str(self.book), self.inp, self.out)
        self.assertEqual(result["status"], "success", result)
        output = Path(result["output"])
        self.assertEqual(output, self.out / "great_book_cleaned.json")
        self.assertTrue(output.exists())
        self.assertFalse(output.with_suffix(".json.tmp").exists())

        data = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(data["source_file"], "great_book.txt")
        self.assertEqual(data["processing_summary"]["chapter_count"], result["chapters"])

    def test_process_file_reports_failure_instead_of_raising(self):
        bad = self.inp / "empty.txt"
        bad.write_text("", encoding="utf-8")
        result = dc.process_file(str(bad), self.inp, self.out)
        self.assertEqual(result["status"], "failed")
        self.assertIn("No readable text", result["error"])

    def test_nested_input_gets_flat_output_name(self):
        nested = self.inp / "fiction" / "classics"
        nested.mkdir(parents=True)
        path = nested / "book.txt"
        self.assertEqual(
            dc.safe_output_path(path, self.inp, self.out).name,
            "fiction__classics__book_cleaned.json",
        )

    def test_discover_books_skips_unsupported_and_hidden(self):
        (self.inp / "cover.png").write_bytes(b"\x89PNG")
        (self.inp / "~$temp.docx").write_text("x")
        (self.inp / ".DS_Store").write_text("x")
        (self.inp / "notes.MD").write_text("hello")
        names = [p.name for p in dc.discover_books(self.inp)]
        self.assertEqual(names, ["great_book.txt", "notes.MD"])


class HeuristicEdgeCaseTests(unittest.TestCase):
    def test_page_number_pattern(self):
        for line in ("3", "Page 12", "iv", "12 of 300", "7/20"):
            self.assertIsNotNone(dc.PAGE_NUMBER_PATTERN.fullmatch(line), line)
        for line in ("3 apples", "Page", "chapter 3"):
            self.assertIsNone(dc.PAGE_NUMBER_PATTERN.fullmatch(line), line)

    def test_dehyphenation_keeps_proper_noun_hyphens(self):
        self.assertEqual(dc.dehyphenate_lines("re-\nuse"), "reuse")
        self.assertEqual(dc.dehyphenate_lines("Jean-\nPaul"), "Jean-\nPaul")

    def test_looks_like_heading_rejects_long_sentences(self):
        long_line = " ".join(["Word"] * 20)
        self.assertFalse(dc.looks_like_heading(long_line))
        self.assertFalse(dc.looks_like_heading("Title Case Line, With Comma,"))

    def test_footnote_starts_new_paragraph(self):
        text = "Body sentence ends here\n1. A footnote follows.\n"
        self.assertEqual(
            dc.paragraphs_from_text(text),
            ["Body sentence ends here", "1. A footnote follows."],
        )

    def test_short_paragraphs_dropped(self):
        self.assertEqual(dc.paragraphs_from_text("a\n\nb\n"), [])


@unittest.skipUnless(has_module("fitz"), "PyMuPDF not installed")
class PdfTests(unittest.TestCase):
    def test_extract_pdf(self):
        import fitz

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "sample.pdf"
            document = fitz.open()
            for number in (1, 2):
                page = document.new_page()
                page.insert_text((72, 72), f"Page {number} body text.")
            document.save(path)
            document.close()

            pages = dc.extract_pdf(path)
            self.assertEqual([p.number for p in pages], [1, 2])
            self.assertIn("Page 1 body text.", pages[0].text)


@unittest.skipUnless(has_module("fitz"), "PyMuPDF not installed")
@unittest.skipUnless(has_tesseract(), "Tesseract OCR engine not installed")
class OcrFallbackTests(unittest.TestCase):
    def make_scanned_pdf(self, folder: Path, text: str) -> Path:
        """Build a PDF page with no text layer: the text is drawn onto a
        plain image and inserted as a picture, the way a scanned or
        photographed page would come out of a scanner."""
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

    def test_image_only_page_recovers_text_via_ocr(self):
        with tempfile.TemporaryDirectory() as folder:
            path = self.make_scanned_pdf(Path(folder), "HELLO SCANNED WORLD")
            pages = dc.extract_pdf(path)

            self.assertEqual(len(pages), 1)
            self.assertIn("HELLO", pages[0].text.upper())

    def test_text_layer_page_does_not_invoke_ocr(self):
        import fitz

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "normal.pdf"
            document = fitz.open()
            page = document.new_page()
            page.insert_text((72, 72), "Ordinary embedded text.")
            document.save(path)
            document.close()

            pages = dc.extract_pdf(path)
            self.assertIn("Ordinary embedded text.", pages[0].text)


@unittest.skipUnless(has_module("docx"), "python-docx not installed")
class DocxTests(unittest.TestCase):
    def test_extract_docx_splits_on_headings(self):
        from docx import Document

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "sample.docx"
            document = Document()
            document.add_heading("Chapter One", level=1)
            document.add_paragraph("First body paragraph.")
            document.add_heading("Chapter Two", level=1)
            document.add_paragraph("Second body paragraph.")
            document.save(path)

            pages = dc.extract_docx(path)
            self.assertEqual(len(pages), 2)
            self.assertTrue(pages[0].text.startswith("Chapter One"))
            self.assertIn("Second body paragraph.", pages[1].text)


@unittest.skipUnless(
    has_module("ebooklib") and has_module("bs4"), "ebooklib/bs4 not installed"
)
class EpubTests(unittest.TestCase):
    def test_extract_epub(self):
        from ebooklib import epub

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "sample.epub"
            book = epub.EpubBook()
            book.set_identifier("test-id")
            book.set_title("Sample")
            book.set_language("en")
            chapter = epub.EpubHtml(title="Ch1", file_name="ch1.xhtml", lang="en")
            chapter.content = "<h1>Chapter One</h1><p>Hello epub world.</p>"
            book.add_item(chapter)
            book.spine = ["nav", chapter]
            book.add_item(epub.EpubNcx())
            book.add_item(epub.EpubNav())
            epub.write_epub(str(path), book)

            pages = dc.extract_epub(path)
            self.assertTrue(any("Hello epub world." in p.text for p in pages))


if __name__ == "__main__":
    unittest.main(verbosity=2)
