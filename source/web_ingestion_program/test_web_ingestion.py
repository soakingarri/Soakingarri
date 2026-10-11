#!/usr/bin/env python3
"""
Test cases for web_ingestion.py

Run with either:
    python3 test_web_ingestion.py          (unittest, no extra installs)
    python3 -m pytest test_web_ingestion.py

Every function in web_ingestion.py carries doctest examples in its
docstring; this file runs all of those plus a handful of end-to-end
checks. No test here makes a real network request: every HTTP fetch and
every robots.txt fetch is replaced with a canned, injected callable, the
same way test_embedding.py uses a StubModel instead of downloading a real
model.
"""

from __future__ import annotations

import doctest
import tempfile
import unittest
from pathlib import Path

import web_ingestion as wi


def load_tests(loader, tests, ignore):
    """Hook so `python3 test_web_ingestion.py` also runs every docstring example."""
    tests.addTests(doctest.DocTestSuite(wi))
    return tests


ARTICLE_HTML = """
<html><head><title>A Real Article</title></head>
<body>
<nav>Home | About | Contact</nav>
<article>
<h1>A Real Article</h1>
<p>This is the first real paragraph of the article, long enough for
trafilatura's extraction heuristics to keep it as genuine content rather
than discarding it as boilerplate.</p>
<p>This is a second paragraph continuing the same article for good
measure and length.</p>
</article>
<footer>Copyright 2024 Example Corp. All rights reserved.</footer>
</body></html>
"""

EMPTY_HTML = "<html><body></body></html>"

ALLOW_ALL_ROBOTS = "User-agent: *\nAllow: /\n"
DISALLOW_PRIVATE_ROBOTS = "User-agent: *\nDisallow: /private/\n"


class ExtractContentTests(unittest.TestCase):
    def test_extracts_title_text_and_url(self):
        result = wi.extract_content(ARTICLE_HTML, "https://example.com/article")
        self.assertIsNotNone(result)
        self.assertEqual(result["title"], "A Real Article")
        self.assertIn("first real paragraph", result["text"])
        self.assertEqual(result["url"], "https://example.com/article")

    def test_returns_none_for_empty_page(self):
        result = wi.extract_content(EMPTY_HTML, "https://example.com/empty")
        self.assertIsNone(result)


class ContentHashTests(unittest.TestCase):
    def test_deterministic_for_same_normalized_text(self):
        self.assertEqual(
            wi.content_hash("Hello   world"),
            wi.content_hash("Hello world"),
        )

    def test_differs_for_different_text(self):
        self.assertNotEqual(
            wi.content_hash("Hello world"),
            wi.content_hash("Goodbye world"),
        )


class OutputFilenameTests(unittest.TestCase):
    def test_same_url_gives_same_filename(self):
        self.assertEqual(
            wi.output_filename("https://example.com/a"),
            wi.output_filename("https://example.com/a"),
        )

    def test_different_urls_give_different_filenames(self):
        self.assertNotEqual(
            wi.output_filename("https://example.com/a"),
            wi.output_filename("https://example.com/b"),
        )

    def test_filename_has_txt_extension(self):
        self.assertTrue(wi.output_filename("https://example.com/a").endswith(".txt"))


class IsAllowedByRobotsTests(unittest.TestCase):
    def test_allowed_when_robots_permits(self):
        self.assertTrue(
            wi.is_allowed_by_robots(
                "https://example.com/page",
                fetch_robots_txt=lambda robots_url: ALLOW_ALL_ROBOTS,
            )
        )

    def test_disallowed_path_is_blocked(self):
        self.assertFalse(
            wi.is_allowed_by_robots(
                "https://example.com/private/page",
                fetch_robots_txt=lambda robots_url: DISALLOW_PRIVATE_ROBOTS,
            )
        )

    def test_other_paths_still_allowed(self):
        self.assertTrue(
            wi.is_allowed_by_robots(
                "https://example.com/public/page",
                fetch_robots_txt=lambda robots_url: DISALLOW_PRIVATE_ROBOTS,
            )
        )

    def test_fetch_error_defaults_to_allowed(self):
        def broken_fetch(robots_url):
            raise RuntimeError("network is down")

        self.assertTrue(
            wi.is_allowed_by_robots(
                "https://example.com/anything",
                fetch_robots_txt=broken_fetch,
            )
        )

    def test_robots_url_is_derived_from_target_url(self):
        seen = {}

        def capturing_fetch(robots_url):
            seen["robots_url"] = robots_url
            return ALLOW_ALL_ROBOTS

        wi.is_allowed_by_robots(
            "https://example.com/some/deep/page",
            fetch_robots_txt=capturing_fetch,
        )
        self.assertEqual(seen["robots_url"], "https://example.com/robots.txt")


class WritePageTests(unittest.TestCase):
    def test_writes_text_atomically(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = wi.write_page(
                root, "https://example.com/a", {"text": "Hello world."}
            )
            self.assertEqual(path.name, wi.output_filename("https://example.com/a"))
            self.assertEqual(path.read_text(encoding="utf-8"), "Hello world.")
            self.assertFalse(path.with_suffix(".txt.tmp").exists())

    def test_rewriting_same_url_overwrites_in_place(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            first = wi.write_page(root, "https://example.com/a", {"text": "First."})
            second = wi.write_page(root, "https://example.com/a", {"text": "Second."})
            self.assertEqual(first, second)
            self.assertEqual(second.read_text(encoding="utf-8"), "Second.")
            self.assertEqual(len(list(root.glob("*.txt"))), 1)


class LoadUrlsFromFileTests(unittest.TestCase):
    def test_skips_comments_and_blank_lines(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "urls.txt"
            path.write_text(
                "# seed urls\n"
                "https://example.com/a\n"
                "\n"
                "   \n"
                "# another comment\n"
                "https://example.com/b  \n",
                encoding="utf-8",
            )
            self.assertEqual(
                wi.load_urls_from_file(path),
                ["https://example.com/a", "https://example.com/b"],
            )


class IngestUrlsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.output_folder = Path(self.tmp.name) / "web"

    def tearDown(self):
        self.tmp.cleanup()

    def allow_all(self, robots_url):
        return ALLOW_ALL_ROBOTS

    def test_normal_page_is_written(self):
        fake_pages = {"https://example.com/a": ARTICLE_HTML}
        manifest = wi.ingest_urls(
            list(fake_pages),
            output_folder=self.output_folder,
            fetch=fake_pages.get,
            fetch_robots_txt=self.allow_all,
        )
        self.assertEqual(manifest["written"], 1)
        result = manifest["results"][0]
        self.assertEqual(result["status"], "written")
        self.assertIn("fetch_timestamp", result)
        self.assertIn("content_hash", result)
        written_path = self.output_folder / result["output_file"]
        self.assertTrue(written_path.exists())
        self.assertIn("first real paragraph", written_path.read_text(encoding="utf-8"))

    def test_duplicate_content_is_skipped_on_second_url(self):
        fake_pages = {
            "https://example.com/a": ARTICLE_HTML,
            "https://example.com/mirror-of-a": ARTICLE_HTML,
        }
        manifest = wi.ingest_urls(
            list(fake_pages),
            output_folder=self.output_folder,
            fetch=fake_pages.get,
            fetch_robots_txt=self.allow_all,
        )
        statuses = [r["status"] for r in manifest["results"]]
        self.assertEqual(statuses, ["written", "skipped_duplicate"])
        self.assertEqual(manifest["written"], 1)
        self.assertEqual(manifest["skipped_duplicate"], 1)
        # only one file on disk despite two URLs
        self.assertEqual(len(list(self.output_folder.glob("*.txt"))), 1)

    def test_failed_fetch_is_reported_not_raised(self):
        fake_pages = {
            "https://example.com/a": ARTICLE_HTML,
            "https://example.com/dead-link": None,
        }
        manifest = wi.ingest_urls(
            list(fake_pages),
            output_folder=self.output_folder,
            fetch=fake_pages.get,
            fetch_robots_txt=self.allow_all,
        )
        statuses = {r["url"]: r["status"] for r in manifest["results"]}
        self.assertEqual(statuses["https://example.com/a"], "written")
        self.assertEqual(statuses["https://example.com/dead-link"], "failed_fetch")
        self.assertEqual(manifest["failed_fetch"], 1)

    def test_robots_disallowed_page_is_skipped_without_fetching(self):
        called = []

        def fetch_spy(url):
            called.append(url)
            return ARTICLE_HTML

        manifest = wi.ingest_urls(
            ["https://example.com/private/page"],
            output_folder=self.output_folder,
            fetch=fetch_spy,
            fetch_robots_txt=lambda robots_url: DISALLOW_PRIVATE_ROBOTS,
        )
        self.assertEqual(manifest["skipped_robots"], 1)
        self.assertEqual(called, [])  # fetch never invoked for a disallowed URL

    def test_empty_extraction_is_skipped(self):
        fake_pages = {"https://example.com/blank": EMPTY_HTML}
        manifest = wi.ingest_urls(
            list(fake_pages),
            output_folder=self.output_folder,
            fetch=fake_pages.get,
            fetch_robots_txt=self.allow_all,
        )
        self.assertEqual(manifest["skipped_empty"], 1)

    def test_manifest_file_is_written_atomically(self):
        fake_pages = {"https://example.com/a": ARTICLE_HTML}
        wi.ingest_urls(
            list(fake_pages),
            output_folder=self.output_folder,
            fetch=fake_pages.get,
            fetch_robots_txt=self.allow_all,
        )
        manifest_path = self.output_folder / wi.MANIFEST_FILENAME
        self.assertTrue(manifest_path.exists())
        self.assertFalse(manifest_path.with_suffix(".json.tmp").exists())

    def test_rerunning_same_urls_overwrites_not_duplicates(self):
        fake_pages = {"https://example.com/a": ARTICLE_HTML}
        wi.ingest_urls(
            list(fake_pages),
            output_folder=self.output_folder,
            fetch=fake_pages.get,
            fetch_robots_txt=self.allow_all,
        )
        wi.ingest_urls(
            list(fake_pages),
            output_folder=self.output_folder,
            fetch=fake_pages.get,
            fetch_robots_txt=self.allow_all,
        )
        self.assertEqual(len(list(self.output_folder.glob("*.txt"))), 1)


class ParseArgumentsTests(unittest.TestCase):
    def test_urls_flag_parsed(self):
        options = wi.parse_arguments(["--urls", "https://a.example/", "https://b.example/"])
        self.assertEqual(options.urls, ["https://a.example/", "https://b.example/"])

    def test_defaults(self):
        options = wi.parse_arguments([])
        self.assertIsNone(options.urls)
        self.assertIsNone(options.urls_file)
        self.assertEqual(options.output, wi.OUTPUT_FOLDER)
        self.assertEqual(options.user_agent, wi.DEFAULT_USER_AGENT)

    def test_urls_file_flag_parsed(self):
        options = wi.parse_arguments(["--urls-file", "seeds.txt"])
        self.assertEqual(options.urls_file, Path("seeds.txt"))


class MainNoNetworkTests(unittest.TestCase):
    """Confirms main() refuses to run (rather than silently doing nothing
    or reaching the network) when no URLs were supplied at all."""

    def test_main_raises_systemexit_without_urls(self):
        import sys

        old_argv = sys.argv
        sys.argv = ["web_ingestion.py"]
        try:
            with self.assertRaises(SystemExit):
                wi.main()
        finally:
            sys.argv = old_argv


if __name__ == "__main__":
    unittest.main(verbosity=2)
