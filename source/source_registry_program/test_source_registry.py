#!/usr/bin/env python3
"""
Test cases for source_registry.py

Run with either:
    python3 test_source_registry.py          (unittest, no extra installs)
    python3 -m pytest test_source_registry.py

Every function in source_registry.py carries doctest examples in its
docstring; this file runs all of those plus a handful of end-to-end checks.
"""

from __future__ import annotations

import doctest
import json
import tempfile
import unittest
from pathlib import Path

import source_registry as sr


def load_tests(loader, tests, ignore):
    """Hook so `python3 test_source_registry.py` also runs every docstring
    example."""
    tests.addTests(doctest.DocTestSuite(sr))
    return tests


def make_registry() -> dict[str, sr.SourceRecord]:
    """Build a small registry with one record of each licence status, plus
    one African-authored and one non-African-authored source."""
    registry: dict[str, sr.SourceRecord] = {}

    sr.add_or_update_source(
        registry,
        title="Time and Social Anthropology",
        author="Alfred Gell",
        author_country="United Kingdom",
        is_african_authored=False,
        year=1998,
        publisher="Berg",
        language="English",
        isbn="978-0-85496-000-0",
        licence_status="cleared",
        licence_notes="Publisher granted permission.",
        source_file="cleared.pdf",
        date_added="2024-01-01",
    )
    sr.add_or_update_source(
        registry,
        title="A Pending Work",
        author="Ngozi Adichie",
        author_country="Nigeria",
        is_african_authored=True,
        year=2010,
        publisher="Lagos Press",
        language="English",
        isbn=None,
        licence_status="pending",
        licence_notes="Awaiting reply.",
        source_file="pending.pdf",
        date_added="2024-01-01",
    )
    sr.add_or_update_source(
        registry,
        title="A Rejected Work",
        author="Jean Dupont",
        author_country="France",
        is_african_authored=False,
        year=2015,
        publisher="Paris Press",
        language="French",
        isbn=None,
        licence_status="rejected",
        licence_notes="Publisher declined.",
        source_file="rejected.pdf",
        date_added="2024-01-01",
    )

    return registry


class RegistryIOTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.registry_path = self.root / "metadata" / "source_registry.json"

    def tearDown(self):
        self.tmp.cleanup()

    def test_load_missing_file_returns_empty_dict(self):
        self.assertEqual(sr.load_registry(self.registry_path), {})

    def test_round_trip_save_and_load(self):
        registry = make_registry()
        sr.save_registry(self.registry_path, registry)

        self.assertTrue(self.registry_path.exists())
        self.assertFalse(
            self.registry_path.with_suffix(".json.tmp").exists()
        )

        loaded = sr.load_registry(self.registry_path)
        self.assertEqual(set(loaded), set(registry))

        for source_id, record in registry.items():
            self.assertEqual(loaded[source_id], record)

    def test_saved_file_is_valid_json_keyed_by_source_id(self):
        registry = make_registry()
        sr.save_registry(self.registry_path, registry)

        payload = json.loads(self.registry_path.read_text(encoding="utf-8"))
        self.assertEqual(set(payload), set(registry))

        for source_id, fields in payload.items():
            self.assertEqual(fields["source_id"], source_id)
            self.assertIn(fields["licence_status"], sr.LICENCE_STATUSES)

    def test_save_creates_parent_directories(self):
        nested_path = self.root / "a" / "b" / "c" / "source_registry.json"
        sr.save_registry(nested_path, {})
        self.assertTrue(nested_path.exists())
        self.assertEqual(json.loads(nested_path.read_text(encoding="utf-8")), {})


class AddOrUpdateSourceTests(unittest.TestCase):
    def test_rejects_invalid_licence_status(self):
        registry: dict[str, sr.SourceRecord] = {}
        with self.assertRaises(ValueError) as context:
            sr.add_or_update_source(
                registry,
                title="X",
                author="Y",
                author_country="",
                is_african_authored=False,
                year=None,
                publisher="",
                language="",
                isbn=None,
                licence_status="approved",  # not a valid status
                licence_notes="",
                source_file="x.pdf",
            )
        self.assertIn("licence_status must be one of", str(context.exception))
        self.assertEqual(registry, {})

    def test_rejects_missing_licence_status(self):
        registry: dict[str, sr.SourceRecord] = {}
        with self.assertRaises(ValueError):
            sr.add_or_update_source(
                registry,
                title="X",
                author="Y",
                author_country="",
                is_african_authored=False,
                year=None,
                publisher="",
                language="",
                isbn=None,
                licence_notes="",
                source_file="x.pdf",
            )

    def test_registering_same_title_author_is_idempotent(self):
        registry: dict[str, sr.SourceRecord] = {}

        first = sr.add_or_update_source(
            registry,
            title="Repeated Title",
            author="Same Author",
            author_country="Ghana",
            is_african_authored=True,
            year=2000,
            publisher="P",
            language="English",
            isbn=None,
            licence_status="pending",
            licence_notes="",
            source_file="repeated.pdf",
            date_added="2024-01-01",
        )
        self.assertEqual(len(registry), 1)

        second = sr.add_or_update_source(
            registry,
            title="Repeated Title",
            author="Same Author",
            author_country="Ghana",
            is_african_authored=True,
            year=2000,
            publisher="P",
            language="English",
            isbn=None,
            licence_status="cleared",
            licence_notes="Now cleared.",
            source_file="repeated.pdf",
            date_added="2024-02-01",
        )

        self.assertEqual(len(registry), 1)
        self.assertEqual(first.source_id, second.source_id)
        self.assertEqual(registry[first.source_id].licence_status, "cleared")

    def test_date_added_defaults_to_today_when_omitted(self):
        from datetime import date

        registry: dict[str, sr.SourceRecord] = {}
        record = sr.add_or_update_source(
            registry,
            title="No Date Given",
            author="Someone",
            author_country="Kenya",
            is_african_authored=True,
            year=None,
            publisher="",
            language="",
            isbn=None,
            licence_status="pending",
            licence_notes="",
            source_file="nodate.pdf",
        )
        self.assertEqual(record.date_added, date.today().isoformat())


class ClearanceTests(unittest.TestCase):
    def setUp(self):
        self.registry = make_registry()

    def test_is_cleared_true_only_for_cleared_record(self):
        self.assertTrue(sr.is_cleared(self.registry, "cleared.pdf"))

    def test_is_cleared_false_for_pending(self):
        self.assertFalse(sr.is_cleared(self.registry, "pending.pdf"))

    def test_is_cleared_false_for_rejected(self):
        self.assertFalse(sr.is_cleared(self.registry, "rejected.pdf"))

    def test_is_cleared_false_for_absent_file(self):
        self.assertFalse(sr.is_cleared(self.registry, "never_registered.pdf"))

    def test_is_cleared_false_on_empty_registry(self):
        self.assertFalse(sr.is_cleared({}, "cleared.pdf"))

    def test_filter_cleared_files_partitions_correctly(self):
        candidates = [
            Path("cleared.pdf"),
            Path("pending.pdf"),
            Path("rejected.pdf"),
            Path("never_registered.pdf"),
        ]
        cleared, not_cleared = sr.filter_cleared_files(self.registry, candidates)

        self.assertEqual([p.name for p in cleared], ["cleared.pdf"])
        self.assertEqual(
            [p.name for p in not_cleared],
            ["pending.pdf", "rejected.pdf", "never_registered.pdf"],
        )

    def test_filter_cleared_files_uses_file_name_not_full_path(self):
        candidates = [Path("/some/nested/dir/cleared.pdf")]
        cleared, not_cleared = sr.filter_cleared_files(self.registry, candidates)
        self.assertEqual(len(cleared), 1)
        self.assertEqual(not_cleared, [])

    def test_filter_cleared_files_empty_input(self):
        self.assertEqual(sr.filter_cleared_files(self.registry, []), ([], []))


class RegistrySummaryTests(unittest.TestCase):
    def test_summary_counts_on_populated_registry(self):
        registry = make_registry()
        summary = sr.registry_summary(registry)

        self.assertEqual(summary["total_sources"], 3)
        self.assertEqual(
            summary["by_licence_status"],
            {"cleared": 1, "pending": 1, "rejected": 1},
        )
        self.assertEqual(summary["african_authored_count"], 1)
        self.assertEqual(summary["not_african_authored_count"], 2)
        self.assertEqual(
            summary["by_language"], {"English": 2, "French": 1}
        )

    def test_summary_on_empty_registry(self):
        summary = sr.registry_summary({})
        self.assertEqual(summary["total_sources"], 0)
        self.assertEqual(
            summary["by_licence_status"],
            {"cleared": 0, "pending": 0, "rejected": 0},
        )
        self.assertEqual(summary["african_authored_count"], 0)
        self.assertEqual(summary["not_african_authored_count"], 0)
        self.assertEqual(summary["by_language"], {})

    def test_summary_every_licence_status_key_present_even_if_unused(self):
        registry: dict[str, sr.SourceRecord] = {}
        sr.add_or_update_source(
            registry,
            title="Only Cleared",
            author="A",
            author_country="Nigeria",
            is_african_authored=True,
            year=2020,
            publisher="P",
            language="English",
            isbn=None,
            licence_status="cleared",
            licence_notes="",
            source_file="only.pdf",
            date_added="2024-01-01",
        )
        summary = sr.registry_summary(registry)
        self.assertEqual(set(summary["by_licence_status"]), sr.LICENCE_STATUSES)


class CliTests(unittest.TestCase):
    def test_add_subcommand_parses_all_flags(self):
        options = sr.parse_arguments(
            [
                "add",
                "--title", "My Title",
                "--author", "My Author",
                "--author-country", "Kenya",
                "--african-authored",
                "--year", "2022",
                "--publisher", "My Publisher",
                "--language", "English",
                "--isbn", "123-456",
                "--licence-status", "cleared",
                "--licence-notes", "All good.",
                "--source-file", "book.pdf",
                "--date-added", "2024-05-05",
            ]
        )
        self.assertEqual(options.command, "add")
        self.assertEqual(options.title, "My Title")
        self.assertEqual(options.author, "My Author")
        self.assertEqual(options.author_country, "Kenya")
        self.assertTrue(options.is_african_authored)
        self.assertEqual(options.year, 2022)
        self.assertEqual(options.publisher, "My Publisher")
        self.assertEqual(options.language, "English")
        self.assertEqual(options.isbn, "123-456")
        self.assertEqual(options.licence_status, "cleared")
        self.assertEqual(options.licence_notes, "All good.")
        self.assertEqual(options.source_file, "book.pdf")
        self.assertEqual(options.date_added, "2024-05-05")

    def test_add_subcommand_rejects_invalid_licence_status_choice(self):
        with self.assertRaises(SystemExit):
            sr.parse_arguments(
                [
                    "add",
                    "--title", "T",
                    "--author", "A",
                    "--licence-status", "not-a-real-status",
                    "--source-file", "f.pdf",
                ]
            )

    def test_add_subcommand_defaults(self):
        options = sr.parse_arguments(
            [
                "add",
                "--title", "T",
                "--author", "A",
                "--licence-status", "pending",
                "--source-file", "f.pdf",
            ]
        )
        self.assertEqual(options.author_country, "")
        self.assertFalse(options.is_african_authored)
        self.assertIsNone(options.year)
        self.assertEqual(options.publisher, "")
        self.assertEqual(options.language, "")
        self.assertIsNone(options.isbn)
        self.assertEqual(options.licence_notes, "")
        self.assertIsNone(options.date_added)
        self.assertEqual(options.registry, sr.REGISTRY_FILE)

    def test_list_subcommand(self):
        options = sr.parse_arguments(["list"])
        self.assertEqual(options.command, "list")
        self.assertFalse(options.as_json)
        self.assertEqual(options.registry, sr.REGISTRY_FILE)

    def test_list_subcommand_with_json_flag(self):
        options = sr.parse_arguments(["list", "--json"])
        self.assertTrue(options.as_json)

    def test_check_subcommand(self):
        options = sr.parse_arguments(["check", "--raw-data", "/some/folder"])
        self.assertEqual(options.command, "check")
        self.assertEqual(options.raw_data, Path("/some/folder"))
        self.assertEqual(options.registry, sr.REGISTRY_FILE)

    def test_no_subcommand_exits(self):
        with self.assertRaises(SystemExit):
            sr.parse_arguments([])


class EndToEndTests(unittest.TestCase):
    def test_main_add_then_check_reports_uncleared_files(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            registry_path = root / "metadata" / "source_registry.json"
            raw_data = root / "raw_data"
            raw_data.mkdir(parents=True)

            (raw_data / "cleared_book.pdf").write_text("x", encoding="utf-8")
            (raw_data / "uncleared_book.pdf").write_text("x", encoding="utf-8")

            registry = sr.load_registry(registry_path)
            sr.add_or_update_source(
                registry,
                title="Cleared Book",
                author="Someone",
                author_country="Nigeria",
                is_african_authored=True,
                year=2020,
                publisher="P",
                language="English",
                isbn=None,
                licence_status="cleared",
                licence_notes="",
                source_file="cleared_book.pdf",
                date_added="2024-01-01",
            )
            sr.save_registry(registry_path, registry)

            reloaded = sr.load_registry(registry_path)
            candidate_files = sorted(raw_data.iterdir())
            cleared, not_cleared = sr.filter_cleared_files(
                reloaded, candidate_files
            )

            self.assertEqual([p.name for p in cleared], ["cleared_book.pdf"])
            self.assertEqual(
                [p.name for p in not_cleared], ["uncleared_book.pdf"]
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
