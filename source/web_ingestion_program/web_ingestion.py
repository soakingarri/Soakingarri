#!/usr/bin/env python3

"""
SoakinGarri Web Ingestion Program

PURPOSE
-------
Fetches a bounded, operator-supplied list of seed URLs -- this is
deliberately NOT an open-ended, recursive, link-following crawler. That is
a scope decision, not an oversight: a fixed URL list is auditable (every
page that lands in the dataset was explicitly named ahead of time) and
safe (it cannot wander off a permitted site onto one that was never
reviewed). For each seed URL this program:

1. Checks robots.txt and skips the page if disallowed.
2. Fetches the raw HTML.
3. Extracts the main article text (not navigation/ads/boilerplate) with
   trafilatura.
4. Hashes the extracted text and skips exact duplicates -- the same
   content already ingested from this run or a previous one.
5. Writes the surviving text as a plain .txt file under the output
   folder, and records per-page provenance (source URL, fetch timestamp,
   content hash, output filename) in a manifest.

This tool identifies itself honestly over HTTP via DEFAULT_USER_AGENT; it
does not spoof a browser. Use it only against sources you already have
permission to crawl and reuse -- the same spirit as the licence-gating
language elsewhere in this project. Labeling or extracting a page's text
does not change its copyright status; confirm that the source is licensed
or otherwise authorized before training or redistribution.

INPUT
-----
A list of seed URLs, given on the command line via --urls and/or
--urls-file (a text file, one URL per line; lines starting with "#" are
comments).

OUTPUT FOLDER
-------------
<repo>/data/ask/raw_data/web (resolved from DATA_ROOT)

This is deliberately a subfolder under raw_data: data_cleaning.py's
discover_books already recursively globs raw_data for supported files, so
every .txt file written here is picked up and cleaned automatically, with
zero changes needed to data_cleaning.py.

OUTPUT FILES
------------
One UTF-8 .txt file per surviving page (deterministic name, see
output_filename), plus web_ingestion_manifest.json.

INSTALL
-------
python3 -m pip install trafilatura requests

requests is used for HTTP if already installed (as it is in this
environment); otherwise this program would fall back to urllib.request,
the stdlib equivalent. robots.txt parsing uses the stdlib's
urllib.robotparser, so no extra dependency is needed for that.

RUN
---
python3 web_ingestion.py --urls https://example.org/article-one https://example.org/article-two
python3 web_ingestion.py --urls-file seed_urls.txt

NOTES
-----
- Re-running ingestion on the same URL list is idempotent: output_filename
  is derived from the URL itself (via common.stable_id), so a re-fetch
  overwrites the same file in place instead of piling up duplicate files
  with random names.
- The manifest is deliberately useful raw material for a future
  source_registry.py integration; this program does not build that
  integration, it only makes the data available.
- On any robots.txt fetch error (network failure, 404, timeout), the page
  is treated as allowed -- the conventional, safe default, since the
  absence of a robots.txt file means no restriction was published.
- One bad URL (a fetch failure, an empty extraction) is recorded in the
  manifest and does not stop the rest of the batch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import requests
import trafilatura

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common import DATA_ROOT, normalize_text, stable_id  # noqa: E402

DEFAULT_USER_AGENT = "SoakinGarriBot/1.0 (+https://github.com/soakingarri/ask)"

OUTPUT_FOLDER = DATA_ROOT / "raw_data" / "web"

MANIFEST_FILENAME = "web_ingestion_manifest.json"

DEFAULT_TIMEOUT = 15.0


def fetch_robots_txt_over_network(
    robots_url: str,
    user_agent: str = DEFAULT_USER_AGENT,
    timeout: float = DEFAULT_TIMEOUT,
) -> str:
    r"""
    PURPOSE
    -------
    Fetch the raw text of a robots.txt file over the network. This is the
    real implementation; it is kept separate from is_allowed_by_robots (and
    named distinctly from its "fetch_robots_txt" parameter) so tests can
    inject a canned, offline replacement instead of calling it.

    PARAMETERS
    ----------
    robots_url   Full URL of the robots.txt file to fetch.
    user_agent   User-Agent header to send with the request.
    timeout      Seconds to wait before giving up.

    RETURNS
    -------
    The robots.txt file's text content.

    RAISES
    ------
    requests.RequestException   On any network error, timeout, or
                                 non-2xx/3xx response.
    """
    response = requests.get(
        robots_url,
        headers={"User-Agent": user_agent},
        timeout=timeout,
    )
    response.raise_for_status()
    return response.text


def is_allowed_by_robots(
    url: str,
    user_agent: str = DEFAULT_USER_AGENT,
    fetch_robots_txt: Callable[[str], str] | None = None,
) -> bool:
    r"""
    PURPOSE
    -------
    Check whether robots.txt permits user_agent to fetch url.

    PARAMETERS
    ----------
    url                 The URL to check.
    user_agent          The user agent that would do the fetching.
    fetch_robots_txt    Callable that takes the robots.txt URL and returns
                         its text content; defaults to the real,
                         network-fetching implementation of the same name.
                         Tests inject a canned replacement here so this
                         function never needs the network.

    RETURNS
    -------
    True if fetching url is allowed. Also True if robots.txt itself could
    not be fetched (network error, 404, timeout, ...): the conventional,
    safe default is that the absence of a robots.txt means no published
    restriction, so the fetch is allowed rather than blocked by an error
    that has nothing to do with the target page.

    EXAMPLES
    --------
    >>> allow_all = lambda robots_url: "User-agent: *\nAllow: /\n"
    >>> is_allowed_by_robots("https://example.com/page", fetch_robots_txt=allow_all)
    True
    >>> disallow_private = lambda robots_url: "User-agent: *\nDisallow: /private/\n"
    >>> is_allowed_by_robots("https://example.com/private/page", fetch_robots_txt=disallow_private)
    False
    >>> is_allowed_by_robots("https://example.com/public/page", fetch_robots_txt=disallow_private)
    True
    >>> def fetch_error(robots_url):
    ...     raise RuntimeError("no robots.txt here")
    >>> is_allowed_by_robots("https://example.com/anything", fetch_robots_txt=fetch_error)
    True
    """
    fetch = (
        fetch_robots_txt
        if fetch_robots_txt is not None
        else fetch_robots_txt_over_network
    )
    parsed = urlparse(url)
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"

    try:
        content = fetch(robots_url)
    except Exception:
        return True

    parser = RobotFileParser()
    parser.parse(content.splitlines())
    return parser.can_fetch(user_agent, url)


def fetch_page(
    url: str,
    user_agent: str = DEFAULT_USER_AGENT,
    timeout: float = DEFAULT_TIMEOUT,
) -> str | None:
    r"""
    PURPOSE
    -------
    Fetch a page's raw HTML. Never raises: one bad URL in a batch should
    be reported, not crash the run (mirrors data_cleaning.process_file's
    philosophy of catching per-item failures).

    PARAMETERS
    ----------
    url          The page URL to fetch.
    user_agent   User-Agent header to send with the request.
    timeout      Seconds to wait before giving up.

    RETURNS
    -------
    The response body as text, or None if the request failed for any
    reason (network error, timeout, non-2xx/3xx status).
    """
    try:
        response = requests.get(
            url,
            headers={"User-Agent": user_agent},
            timeout=timeout,
        )
        response.raise_for_status()
        return response.text
    except requests.RequestException:
        return None


def extract_content(html: str, url: str) -> dict | None:
    r"""
    PURPOSE
    -------
    Extract the main article text (not navigation/ads/boilerplate) from a
    page's raw HTML, using trafilatura. Pure function on a string: no
    network involved, so it is fully testable offline.

    PARAMETERS
    ----------
    html   The page's raw HTML.
    url    The page's original URL, passed to trafilatura as a hint and
           recorded in the result.

    RETURNS
    -------
    A dict with at least "text", "title", and "url", or None if
    trafilatura found nothing extractable on the page.

    EXAMPLES
    --------
    >>> html = '''
    ... <html><head><title>Big Title</title></head><body>
    ... <nav>Home | About | Contact</nav>
    ... <article>
    ... <h1>Big Title</h1>
    ... <p>This is the first real paragraph of the article, long enough
    ... for trafilatura's extraction heuristics to keep it as content.</p>
    ... <p>This is a second paragraph continuing the article for good
    ... measure and length.</p>
    ... </article>
    ... <footer>Copyright 2024 Example Corp</footer>
    ... </body></html>
    ... '''
    >>> result = extract_content(html, "https://example.com/article")
    >>> result["title"]
    'Big Title'
    >>> "first real paragraph" in result["text"]
    True
    >>> result["url"]
    'https://example.com/article'
    >>> extract_content("<html><body></body></html>", "https://example.com/empty") is None
    True
    """
    extracted = trafilatura.extract(
        html,
        url=url,
        with_metadata=True,
        output_format="json",
    )

    if not extracted:
        return None

    parsed = json.loads(extracted)
    text = normalize_text(parsed.get("text"))

    if not text:
        return None

    return {
        "text": text,
        "title": parsed.get("title"),
        "url": url,
    }


def content_hash(text: str) -> str:
    r"""
    PURPOSE
    -------
    Compute a deterministic fingerprint of a page's extracted text, so
    exact-duplicate pages -- even mirrored at different URLs -- can be
    caught and skipped.

    PARAMETERS
    ----------
    text   The extracted text to hash.

    RETURNS
    -------
    The sha256 hex digest of the text's normalized form.

    EXAMPLES
    --------
    >>> content_hash("Hello   world") == content_hash("Hello world")
    True
    >>> content_hash("Hello world") == content_hash("Goodbye world")
    False
    """
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()


def output_filename(url: str) -> str:
    r"""
    PURPOSE
    -------
    Build a deterministic output filename for a URL, so re-running
    ingestion on the same URL list is idempotent: the same URL always
    maps to the same file, which gets overwritten in place instead of
    piling up duplicate files with random names.

    PARAMETERS
    ----------
    url   The page URL.

    RETURNS
    -------
    The filename (no directory component) to write this page's text to.

    EXAMPLES
    --------
    >>> output_filename("https://example.com/a") == output_filename("https://example.com/a")
    True
    >>> output_filename("https://example.com/a") == output_filename("https://example.com/b")
    False
    >>> output_filename("https://example.com/a").endswith(".txt")
    True
    """
    return f"{stable_id('WEB', url)}.txt"


def write_page(output_folder: Path, url: str, extracted: dict) -> Path:
    r"""
    PURPOSE
    -------
    Write one page's extracted text to disk, atomically (temp file, then
    rename), matching the rest of this repo's write pattern.

    PARAMETERS
    ----------
    output_folder   Folder the .txt file is written into; created if
                     missing.
    url             The page's original URL (used to derive the filename).
    extracted       The dict returned by extract_content, at minimum
                     carrying "text".

    RETURNS
    -------
    The path the file was written to.

    EXAMPLES
    --------
    >>> import tempfile
    >>> with tempfile.TemporaryDirectory() as folder:
    ...     root = Path(folder)
    ...     path = write_page(root, "https://example.com/a", {"text": "Hello world."})
    ...     path.name == output_filename("https://example.com/a")
    ...     path.read_text(encoding="utf-8")
    ...     path.with_suffix(".txt.tmp").exists()
    True
    'Hello world.'
    False
    """
    output_folder.mkdir(parents=True, exist_ok=True)
    output_path = output_folder / output_filename(url)
    temporary_path = output_path.with_suffix(".txt.tmp")

    temporary_path.write_text(extracted["text"], encoding="utf-8")
    temporary_path.replace(output_path)

    return output_path


def load_urls_from_file(path: Path) -> list[str]:
    r"""
    PURPOSE
    -------
    Read a list of seed URLs from a text file, one per line, with
    "#"-prefixed lines treated as comments and blank lines ignored.

    PARAMETERS
    ----------
    path   Path to the text file of URLs.

    RETURNS
    -------
    The list of URLs, in file order.

    EXAMPLES
    --------
    >>> import tempfile
    >>> with tempfile.TemporaryDirectory() as folder:
    ...     path = Path(folder) / "urls.txt"
    ...     _ = path.write_text(
    ...         "# seed urls\n"
    ...         "https://example.com/a\n"
    ...         "\n"
    ...         "https://example.com/b  \n",
    ...         encoding="utf-8",
    ...     )
    ...     load_urls_from_file(path)
    ['https://example.com/a', 'https://example.com/b']
    """
    urls: list[str] = []

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()

        if not line or line.startswith("#"):
            continue

        urls.append(line)

    return urls


def load_known_pages(output_folder: Path) -> dict[str, dict]:
    r"""
    PURPOSE
    -------
    Read the pages earlier runs wrote, from the previous manifest in
    output_folder. Manifests from before "known_pages" existed are read
    from their "written" results instead.

    PARAMETERS
    ----------
    output_folder   Folder holding web_ingestion_manifest.json.

    RETURNS
    -------
    A dict mapping URL to its page record (url, fetch_timestamp,
    content_hash, output_file); empty when there is no readable manifest.

    EXAMPLES
    --------
    >>> import tempfile
    >>> with tempfile.TemporaryDirectory() as folder:
    ...     empty = load_known_pages(Path(folder))
    ...     _ = (Path(folder) / MANIFEST_FILENAME).write_text(json.dumps({"results": [
    ...         {"url": "u1", "status": "written", "content_hash": "h1"},
    ...         {"url": "u2", "status": "failed_fetch"}]}), encoding="utf-8")
    ...     legacy = load_known_pages(Path(folder))
    >>> empty, list(legacy), legacy["u1"]["content_hash"]
    ({}, ['u1'], 'h1')
    """
    manifest_path = output_folder / MANIFEST_FILENAME

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}

    pages = manifest.get("known_pages")

    if pages is None:
        pages = [
            {key: value for key, value in result.items() if key != "status"}
            for result in manifest.get("results", [])
            if result.get("status") == "written"
        ]

    return {
        page["url"]: page
        for page in pages
        if page.get("url") and page.get("content_hash")
    }


def ingest_urls(
    urls: list[str],
    output_folder: Path = OUTPUT_FOLDER,
    user_agent: str = DEFAULT_USER_AGENT,
    fetch: Callable[[str], str | None] | None = None,
    fetch_robots_txt: Callable[[str], str] | None = None,
) -> dict:
    r"""
    PURPOSE
    -------
    Run the full ingestion pipeline over a bounded list of seed URLs:
    check robots.txt, fetch, extract, deduplicate by content hash, and
    write survivors to output_folder. Writes web_ingestion_manifest.json
    (atomically) to output_folder and returns the same manifest dict.

    Never lets one bad URL stop the batch: every per-URL failure is
    captured as a result record with an explanatory status instead of an
    exception.

    PARAMETERS
    ----------
    urls               The seed URLs to ingest.
    output_folder      Folder .txt files and the manifest are written to.
    user_agent         User-Agent string identifying this tool to servers
                        and used for the robots.txt check.
    fetch              Callable taking a URL and returning its raw HTML
                        (or None on failure); defaults to fetch_page using
                        user_agent. Tests inject a canned replacement here
                        so this function never needs the network.
    fetch_robots_txt    Callable taking a robots.txt URL and returning its
                         text; forwarded to is_allowed_by_robots. Tests
                         inject a canned replacement here for the same
                         reason.

    RETURNS
    -------
    The manifest dict: output_folder, total_urls, per-status counts, and
    "results" (one record per URL with at least "url" and "status";
    written pages also carry "fetch_timestamp", "content_hash", and
    "output_file").

    EXAMPLES
    --------
    >>> import tempfile
    >>> pages = {
    ...     "https://example.com/a": "<html><body><article><p>" +
    ...         "First unique article with enough text to extract." +
    ...         "</p></article></body></html>",
    ...     "https://example.com/b": "<html><body><article><p>" +
    ...         "First unique article with enough text to extract." +
    ...         "</p></article></body></html>",
    ...     "https://example.com/c": None,
    ... }
    >>> allow_all = lambda robots_url: "User-agent: *\nAllow: /\n"
    >>> with tempfile.TemporaryDirectory() as folder:
    ...     manifest = ingest_urls(
    ...         list(pages),
    ...         output_folder=Path(folder),
    ...         fetch=pages.get,
    ...         fetch_robots_txt=allow_all,
    ...     )
    ...     manifest["written"], manifest["skipped_duplicate"], manifest["failed_fetch"]
    (1, 1, 1)
    >>> [r["status"] for r in manifest["results"]]
    ['written', 'skipped_duplicate', 'failed_fetch']

    A second run remembers what the first one wrote:

    >>> with tempfile.TemporaryDirectory() as folder:
    ...     first = ingest_urls(["https://example.com/a"], Path(folder),
    ...                         fetch=pages.get, fetch_robots_txt=allow_all)
    ...     second = ingest_urls(["https://example.com/a", "https://example.com/b"],
    ...                          Path(folder), fetch=pages.get, fetch_robots_txt=allow_all)
    >>> [r["status"] for r in second["results"]], len(second["known_pages"])
    (['skipped_unchanged', 'skipped_duplicate'], 1)
    """
    fetcher = fetch if fetch is not None else (
        lambda target_url: fetch_page(target_url, user_agent=user_agent)
    )

    # Pages written by earlier runs, keyed by URL, so a re-run skips text
    # it already has instead of re-writing it.
    known_pages = load_known_pages(output_folder)
    url_by_hash = {page["content_hash"]: url for url, page in known_pages.items()}

    seen_hashes: set[str] = set()
    results: list[dict] = []

    for url in urls:
        if not is_allowed_by_robots(
            url, user_agent=user_agent, fetch_robots_txt=fetch_robots_txt
        ):
            results.append({"url": url, "status": "skipped_robots"})
            continue

        html = fetcher(url)

        if html is None:
            results.append({"url": url, "status": "failed_fetch"})
            continue

        extracted = extract_content(html, url)

        if extracted is None:
            results.append({"url": url, "status": "skipped_empty"})
            continue

        digest = content_hash(extracted["text"])

        if digest in seen_hashes:
            results.append({"url": url, "status": "skipped_duplicate"})
            continue

        if digest in url_by_hash:
            status = (
                "skipped_unchanged" if url_by_hash[digest] == url
                else "skipped_duplicate"
            )
            results.append({"url": url, "status": status})
            seen_hashes.add(digest)
            continue

        seen_hashes.add(digest)
        output_path = write_page(output_folder, url, extracted)

        written = {
            "url": url,
            "status": "written",
            "fetch_timestamp": datetime.now(timezone.utc).isoformat(),
            "content_hash": digest,
            "output_file": output_path.name,
        }
        results.append(written)
        known_pages[url] = {
            key: value for key, value in written.items() if key != "status"
        }

    manifest = {
        "output_folder": str(output_folder),
        "total_urls": len(urls),
        "written": sum(r["status"] == "written" for r in results),
        "skipped_robots": sum(r["status"] == "skipped_robots" for r in results),
        "skipped_duplicate": sum(r["status"] == "skipped_duplicate" for r in results),
        "skipped_unchanged": sum(r["status"] == "skipped_unchanged" for r in results),
        "skipped_empty": sum(r["status"] == "skipped_empty" for r in results),
        "failed_fetch": sum(r["status"] == "failed_fetch" for r in results),
        "results": results,
        "known_pages": [known_pages[url] for url in sorted(known_pages)],
    }

    output_folder.mkdir(parents=True, exist_ok=True)
    manifest_path = output_folder / MANIFEST_FILENAME
    temporary_manifest_path = manifest_path.with_suffix(".json.tmp")

    with temporary_manifest_path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)

    temporary_manifest_path.replace(manifest_path)

    return manifest


def parse_arguments(argument_list: list[str] | None = None) -> argparse.Namespace:
    r"""
    PURPOSE
    -------
    Parse the command line into the options ingest_urls needs.

    PARAMETERS
    ----------
    argument_list   Arguments to parse, or None to read from sys.argv.

    RETURNS
    -------
    The parsed arguments namespace.

    EXAMPLES
    --------
    >>> parse_arguments(["--urls", "https://a.example/", "https://b.example/"]).urls
    ['https://a.example/', 'https://b.example/']
    >>> parse_arguments([]).output == OUTPUT_FOLDER
    True
    >>> parse_arguments([]).user_agent == DEFAULT_USER_AGENT
    True
    """
    parser = argparse.ArgumentParser(
        description=(
            "Fetch a bounded list of seed URLs, extract their main "
            "article text, and write it as .txt files for data_cleaning "
            "to pick up."
        )
    )

    parser.add_argument(
        "--urls",
        nargs="+",
        default=None,
        help="one or more seed URLs to ingest",
    )

    parser.add_argument(
        "--urls-file",
        type=Path,
        default=None,
        help="text file of seed URLs, one per line (# comments allowed)",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_FOLDER,
        help="folder for extracted .txt files and the manifest",
    )

    parser.add_argument(
        "--user-agent",
        default=DEFAULT_USER_AGENT,
        help="User-Agent string sent with every request",
    )

    return parser.parse_args(argument_list)


def main() -> None:
    """
    PURPOSE
    -------
    Ingest the URLs given on the command line (via --urls and/or
    --urls-file) using the options given, and print a short summary.

    RAISES
    ------
    SystemExit   If no URLs were given by either option.
    """
    options = parse_arguments()

    urls = list(options.urls) if options.urls else []

    if options.urls_file:
        urls.extend(load_urls_from_file(options.urls_file))

    if not urls:
        raise SystemExit("No URLs given. Use --urls or --urls-file.")

    manifest = ingest_urls(
        urls,
        output_folder=options.output,
        user_agent=options.user_agent,
    )

    print(
        "Web ingestion completed.\n"
        f"Total URLs: {manifest['total_urls']}\n"
        f"Written: {manifest['written']}\n"
        f"Skipped (robots): {manifest['skipped_robots']}\n"
        f"Skipped (duplicate): {manifest['skipped_duplicate']}\n"
        f"Skipped (empty): {manifest['skipped_empty']}\n"
        f"Failed fetch: {manifest['failed_fetch']}\n"
        f"Output folder: {options.output}"
    )


if __name__ == "__main__":
    main()
