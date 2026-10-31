"""
GENERIC AFRICAN AUTHOR SCRAPER

Install dependencies in your active Anaconda environment from Terminal:
    python -m pip install aiohttp beautifulsoup4 lxml orjson

Run:
    python african_author_scraper.py

Output:
    african_authors.json
"""

import asyncio
import json
import re
from collections import defaultdict
from pathlib import Path
from urllib.parse import urldefrag, urljoin, urlparse

import aiohttp
import orjson
from bs4 import BeautifulSoup


START_URLS = [
    "https://brittlepaper.com",
    "https://jaladaafrica.org",
    "https://writivism.org",
    "https://africanwriterstrust.org",
    "https://pan-africanwritersassociation.org",
    "https://iselemagazine.com",
    "https://lolwe.org",
    "https://bakwamagazine.com",
    "https://femrite.org",
    "https://kwani.org",
    "https://cassavarepublic.biz",
    "https://masobebooks.com",
    "https://mkukinanyota.com",
]

OUTPUT_FILE = Path(__file__).resolve().with_name("african_authors.json")

MAX_DEPTH = 3
MAX_PAGES_PER_SITE = 300
MAX_LINKS_PER_PAGE = 100
CONCURRENT_REQUESTS = 20
REQUEST_TIMEOUT_SECONDS = 30
REQUEST_DELAY_SECONDS = 0.15

AUTHOR_SELECTORS = [
    "[rel='author']",
    ".author",
    ".author-name",
    ".author_name",
    ".post-author",
    ".post_author",
    ".entry-author",
    ".entry_author",
    ".article-author",
    ".article_author",
    ".byline",
    ".by-line",
    ".writer",
    ".writer-name",
    ".contributor",
    ".contributor-name",
    ".profile-name",
    ".bio-name",
    ".book-author",
    ".book_author",
    ".product-author",
    ".product_author",
    "[class*='author-name']",
    "[class*='author_name']",
    "[class*='post-author']",
    "[class*='entry-author']",
    "[class*='article-author']",
    "[class*='byline']",
]

PERSON_NAME = (
    r"[A-ZÀ-ÖØ-Þ][A-Za-zÀ-ÖØ-öø-ÿ'’.\-]+"
    r"(?:\s+[A-ZÀ-ÖØ-Þ][A-Za-zÀ-ÖØ-öø-ÿ'’.\-]+){1,5}"
)

AUTHOR_PATTERNS = [
    re.compile(rf"\bwritten\s+by\s+({PERSON_NAME})", re.IGNORECASE),
    re.compile(
        rf"\b(?:story|poem|essay|fiction|review|interview)\s+by\s+({PERSON_NAME})",
        re.IGNORECASE,
    ),
    re.compile(rf"\bby\s+({PERSON_NAME})", re.IGNORECASE),
    re.compile(rf"\bauthor\s*:\s*({PERSON_NAME})", re.IGNORECASE),
    re.compile(rf"\bcontributor\s*:\s*({PERSON_NAME})", re.IGNORECASE),
]

INVALID_NAME_WORDS = {
    "admin",
    "administrator",
    "editor",
    "editors",
    "editorial team",
    "staff writer",
    "guest author",
    "guest writer",
    "anonymous",
    "unknown",
    "wordpress",
    "facebook",
    "twitter",
    "instagram",
    "linkedin",
    "read more",
    "click here",
    "learn more",
    "view more",
    "all rights reserved",
    "privacy policy",
    "terms of service",
    "latest posts",
    "related posts",
    "the author",
    "our contributors",
    "site administrator",
    "submissions editor",
}

BLOCKED_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg", ".pdf", ".zip",
    ".mp3", ".mp4", ".avi", ".mov", ".css", ".js", ".xml", ".json",
    ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
}


class GenericAuthorScraper:
    def __init__(self):
        self.semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)
        self.visited_by_site = defaultdict(set)
        self.authors = {}

    @staticmethod
    def normalize_domain(url):
        return urlparse(url).netloc.lower().removeprefix("www.")

    @staticmethod
    def normalize_url(url):
        clean_url, _ = urldefrag(url)
        parsed = urlparse(clean_url)
        normalized_path = re.sub(r"/+", "/", parsed.path or "/")
        normalized = parsed._replace(path=normalized_path, fragment="").geturl()
        return normalized.rstrip("/")

    @staticmethod
    def normalize_name(name):
        name = BeautifulSoup(str(name), "lxml").get_text(" ", strip=True)
        name = re.sub(
            r"^(?:by|written by|author|contributor)\s*:?\s*",
            "",
            name,
            flags=re.IGNORECASE,
        )
        name = re.sub(r"\s+", " ", name)
        return name.strip(" \t\r\n|:,;.!?–—-()[]{}")

    @staticmethod
    def name_key(name):
        clean = name.casefold().replace("’", "'")
        clean = re.sub(r"[^\w\s']", "", clean, flags=re.UNICODE)
        return re.sub(r"\s+", " ", clean).strip()

    @staticmethod
    def looks_like_person_name(name):
        if not name:
            return False

        valid_words = 0
        for word in words:
            cleaned_word = word.strip(".'’-,()[]{}")
            if re.fullmatch(
                r"[A-Za-zÀ-ÖØ-öø-ÿ][A-Za-zÀ-ÖØ-öø-ÿ'’.\-]*",
                cleaned_word,
            ):
                valid_words += 1

        return valid_words >= 2

    def add_author(self, name, source_url, extraction_method):
        clean_name = self.normalize_name(name)
        if not self.looks_like_person_name(clean_name):
            return

        key = self.name_key(clean_name)
        site_name = self.normalize_domain(source_url)

        if key not in self.authors:
            self.authors[key] = {"name": clean_name, "sources": []}

        source_record = {
            "site_name": site_name,
            "source_url": source_url,
            "extraction_method": extraction_method,
        }
        if source_record not in self.authors[key]["sources"]:
            self.authors[key]["sources"].append(source_record)

    async def fetch(self, session, url):
        async with self.semaphore:
            try:
                await asyncio.sleep(REQUEST_DELAY_SECONDS)
                async with session.get(url, allow_redirects=True) as response:
                    if response.status != 200:
                        return None, None

                    content_type = response.headers.get("Content-Type", "").lower()
                    if "text/html" not in content_type:
                        return None, None

                    html = await response.text(errors="ignore")
                    return str(response.url), html
            except (aiohttp.ClientError, asyncio.TimeoutError, UnicodeError) as error:
                print(f"Request failed: {url} ({error})")
                return None, None

    def extract_meta_authors(self, soup, source_url):
        selectors = [
            "meta[name='author']",
            "meta[property='author']",
            "meta[property='article:author']",
            "meta[name='byl']",
            "meta[name='dc.creator']",
            "meta[name='DC.creator']",
            "meta[name='citation_author']",
            "meta[name='twitter:creator']",
        ]
        for selector in selectors:
            for node in soup.select(selector):
                value = node.get("content")
                if value:
                    self.add_author(value, source_url, "metadata")

    def process_json_ld_item(self, item, source_url):
        if isinstance(item, list):
            for child in item:
                self.process_json_ld_item(child, source_url)
            return

        if not isinstance(item, dict):
            return

        for value in (item.get("author"), item.get("creator")):
            if isinstance(value, str):
                self.add_author(value, source_url, "json_ld")
            elif isinstance(value, dict):
                if value.get("name"):
                    self.add_author(value["name"], source_url, "json_ld")
            elif isinstance(value, list):
                for author_item in value:
                    if isinstance(author_item, str):
                        self.add_author(author_item, source_url, "json_ld")
                    elif isinstance(author_item, dict) and author_item.get("name"):
                        self.add_author(author_item["name"], source_url, "json_ld")

        for child_value in item.values():
            if isinstance(child_value, (dict, list)):
                self.process_json_ld_item(child_value, source_url)

    def extract_json_ld_authors(self, soup, source_url):
        for script in soup.select("script[type='application/ld+json']"):
            raw_json = script.string or script.get_text(" ", strip=True)
            if not raw_json:
                continue
            try:
                self.process_json_ld_item(json.loads(raw_json), source_url)
            except json.JSONDecodeError:
                continue

    def extract_selector_authors(self, soup, source_url):
        for selector in AUTHOR_SELECTORS:
            try:
                nodes = soup.select(selector)
            except Exception:
                continue
            for node in nodes:
                self.add_author(
                    node.get_text(" ", strip=True),
                    source_url,
                    f"selector:{selector}",
                )

    def extract_linked_author_profiles(self, soup, source_url):
        markers = (
            "/author/", "/authors/", "/writer/", "/writers/",
            "/contributor/", "/contributors/", "/profile/",
        )
        for node in soup.select("a[href]"):
            href = node.get("href", "").lower()
            if any(marker in href for marker in markers):
                self.add_author(
                    node.get_text(" ", strip=True),
                    source_url,
                    "author_profile_link",
                )

    def extract_regex_authors(self, soup, source_url):
        for unwanted in soup(["script", "style", "noscript", "svg", "template"]):
            unwanted.decompose()

        visible_text = re.sub(r"\s+", " ", soup.get_text(" ", strip=True))
        for pattern in AUTHOR_PATTERNS:
            for match in pattern.finditer(visible_text):
                self.add_author(match.group(1), source_url, "visible_text_pattern")

    def extract_authors(self, html, source_url):
        soup = BeautifulSoup(html, "lxml")
        self.extract_meta_authors(soup, source_url)
        self.extract_json_ld_authors(soup, source_url)
        self.extract_selector_authors(soup, source_url)
        self.extract_linked_author_profiles(soup, source_url)
        self.extract_regex_authors(soup, source_url)

    def extract_links(self, html, page_url, site_domain):
        soup = BeautifulSoup(html, "lxml")
        links = []

        for node in soup.select("a[href]"):
            href = node.get("href", "").strip()
            if not href or href.startswith(
                ("#", "mailto:", "tel:", "javascript:", "data:")
            ):
                continue

            full_url = self.normalize_url(urljoin(page_url, href))
            parsed = urlparse(full_url)

            if parsed.scheme not in ("http", "https"):
                continue
            if self.normalize_domain(full_url) != site_domain:
                continue
            if any(parsed.path.lower().endswith(ext) for ext in BLOCKED_EXTENSIONS):
                continue

            links.append(full_url)

        priority_words = (
            "author", "writer", "contributor", "book", "literature", "fiction",
            "poetry", "poem", "story", "interview", "profile", "people",
        )
        unique_links = list(dict.fromkeys(links))
        unique_links.sort(
            key=lambda link: (
                not any(word in link.lower() for word in priority_words),
                link,
            )
        )
        return unique_links[:MAX_LINKS_PER_PAGE]

    async def crawl_site(self, session, start_url):
        site_domain = self.normalize_domain(start_url)
        start_url = self.normalize_url(start_url)
        queue = asyncio.Queue()
        await queue.put((start_url, 0))
        scheduled = {start_url}
        page_count = 0

        while not queue.empty() and page_count < MAX_PAGES_PER_SITE:
            batch = []

            while (
                not queue.empty()
                and len(batch) < CONCURRENT_REQUESTS
                and page_count + len(batch) < MAX_PAGES_PER_SITE
            ):
                url, depth = await queue.get()
                queue.task_done()

                if url in self.visited_by_site[site_domain]:
                    continue

                self.visited_by_site[site_domain].add(url)
                batch.append((url, depth))

            if not batch:
                continue

            responses = await asyncio.gather(
                *(self.fetch(session, url) for url, _ in batch),
                return_exceptions=True,
            )

            for (_, depth), response in zip(batch, responses):
                page_count += 1

                if isinstance(response, Exception):
                    print(f"Unexpected fetch error on {site_domain}: {response}")
                    continue

                final_url, html = response
                if not final_url or not html:
                    continue

                print(
                    f"[{site_domain}] {page_count}/{MAX_PAGES_PER_SITE}: {final_url}"
                )
                self.extract_authors(html, final_url)

                if depth >= MAX_DEPTH:
                    continue

                for link in self.extract_links(html, final_url, site_domain):
                    if (
                        link not in scheduled
                        and link not in self.visited_by_site[site_domain]
                    ):
                        scheduled.add(link)
                        await queue.put((link, depth + 1))

        print(f"Finished {site_domain}: {page_count} pages visited")

    async def run(self):
        timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT_SECONDS)
        connector = aiohttp.TCPConnector(
            limit=CONCURRENT_REQUESTS,
            ttl_dns_cache=300,
        )
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0 Safari/537.36 "
                "SoakinGarriAuthorResearch/1.0"
            ),
            "Accept": (
                "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
            ),
            "Accept-Language": "en-US,en;q=0.9",
        }

        async with aiohttp.ClientSession(
            timeout=timeout,
            connector=connector,
            headers=headers,
        ) as session:
            results = await asyncio.gather(
                *(self.crawl_site(session, url) for url in START_URLS),
                return_exceptions=True,
            )

            for start_url, result in zip(START_URLS, results):
                if isinstance(result, Exception):
                    print(f"Site failed: {start_url} ({result})")

    def save(self):
        excluded_extensions = (".com", ".org", ".ng", ".biz")

        site_names = sorted({
            source["site_name"]
            for author in self.authors.values()
            for source in author["sources"]
            if not any(
                extension in source["site_name"].lower()
                for extension in excluded_extensions
            )
        })

        output_file = Path(__file__).resolve().with_name(
            "filtered_site_names.json"
        )

        output_file.write_bytes(
            orjson.dumps(
                site_names,
                option=orjson.OPT_INDENT_2,
            )
        )

        print(f"Saved {len(site_names)} site names to {output_file}")


async def main():
    scraper = GenericAuthorScraper()
    await scraper.run()
    scraper.save()


if __name__ == "__main__":
    asyncio.run(main())  

async def main():
    scraper = GenericAuthorScraper()
    await scraper.run()
    scraper.save()


if __name__ == "__main__":
    asyncio.run(main())

