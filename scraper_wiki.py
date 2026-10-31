import requests
from bs4 import BeautifulSoup
from pathlib import Path

FILE_NAME = "african_surnames.txt"

WIKIPEDIA_CATEGORIES = [
    "https://en.wikipedia.org/wiki/List_of_African-American_writers#Z"
]

# Load existing surnames if file already exists
all_surnames = set()

if Path(FILE_NAME).exists():
    with open(FILE_NAME, encoding="utf-8") as f:
        all_surnames = {
            line.strip().lower()
            for line in f
            if line.strip()
        }

print(f"Starting with {len(all_surnames)} existing surnames")

for url in WIKIPEDIA_CATEGORIES:

    print(f"Scraping {url}")

    html = requests.get(
        url,
        headers={"User-Agent": "Mozilla/5.0"}
    ).text

    soup = BeautifulSoup(html, "html.parser")

    added = 0

    for link in soup.find_all("a", rel="mw:WikiLink"):

        full_name = link.get_text(strip=True)

        if not full_name:
            continue

        words = full_name.split()

        if len(words) < 2:
            continue

        surname = words[-1].lower()

        print(f"Full name: {full_name}")
        print(f"Surname:  {surname}")
        print("-" * 40)

        if surname not in all_surnames:
            all_surnames.add(surname)
            added += 1 
with open(FILE_NAME, "w", encoding = "utf-8") as f:
    for surname in sorted(all_surnames):
        f.write(surname + "\n")
print(f"Saved {len(all_surnames)} total surnames")