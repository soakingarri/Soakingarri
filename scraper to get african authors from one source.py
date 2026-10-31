import pandas as pd
import requests

# ============================================================
# FILE PATHS
# ============================================================

excel_file = "/Users/t.baks/Soakingarri/list_of_books_and_sources.xlsx"

african_surnames_file = (
    "/Users/t.baks/Soakingarri/african_surnames.txt"
)

output_file = (
    "/Users/t.baks/Soakingarri/african_open_source_books.txt"
)

# ============================================================
# LOAD AFRICAN SURNAMES
# ============================================================

print("Loading African surnames...")

with open(
    african_surnames_file,
    "r",
    encoding="utf-8"
) as f:

    african_names = {
        line.strip().lower()
        for line in f
        if line.strip()
    }

print(
    f"African surnames loaded: "
    f"{len(african_names):,}"
)

# ============================================================
# READ EXCEL FILE
# ============================================================

print("\nReading Excel file...")

df = pd.read_excel(excel_file, header=2)

# Normalize column names

df.columns = [
    str(col).strip().lower()
    for col in df.columns
]

print(
    f"Total rows found: "
    f"{len(df):,}"
)

# ============================================================
# VERIFY REQUIRED COLUMNS EXIST
# ============================================================

required_columns = [
    "title of source",
    "open source"
]

for column in required_columns:

    if column not in df.columns:

        raise ValueError(
            f"\nMissing column: {column}\n\n"
            f"Columns found:\n"
            f"{list(df.columns)}"
        )

# ============================================================
# FILTER OPEN SOURCE BOOKS
# ============================================================

open_source_books = df[
    df["open source"]
    .astype(str)
    .str.strip()
    .str.lower()
    .isin([
        "true",
        "yes",
        "1"
    ])
]

book_titles = (
    open_source_books["title of source"]
    .dropna()
    .astype(str)
    .unique()
    .tolist()
)

print(
    f"Open-source books found: "
    f"{len(book_titles):,}"
)

# ============================================================
# FUNCTION TO SEARCH ONLINE FOR AUTHORS
# ============================================================

def get_authors(title):

    """
    Search Open Library using the title
    and return a list of authors.
    """

    try:

        response = requests.get(
            "https://openlibrary.org/search.json",
            params={
                "title": title
            },
            timeout=15
        )

        response.raise_for_status()

        data = response.json()

        docs = data.get("docs", [])

        if not docs:
            return []

        return docs[0].get(
            "author_name",
            []
        )

    except Exception as e:

        print(
            f"Error searching "
            f"'{title}'"
        )

        print(e)

        return []

# ============================================================
# SEARCH FOR AUTHORS ONLINE
# AND MATCH AGAINST AFRICAN SURNAME FILE
# ============================================================

matches = []

total_books = len(book_titles)

print(
    "\nStarting online author lookup..."
)

for index, title in enumerate(
    book_titles,
    start=1
):

    print("\n" + "=" * 70)

    print(
        f"[{index}/{total_books}] "
        f"Searching Open Library for:"
    )

    print(title)

    # --------------------------------------------------------
    # THIS IS WHERE THE ONLINE SEARCH HAPPENS
    # --------------------------------------------------------

    authors = get_authors(title)

    print(
        f"Authors found: {authors}"
    )

    # --------------------------------------------------------
    # CHECK AUTHORS AGAINST AFRICAN SURNAME FILE
    # --------------------------------------------------------

    for author in authors:

        author_parts = {
            part.lower().strip(
                ".,()[]{}'\""
            )
            for part in author.split()
        }

        matching_surnames = (
            author_parts.intersection(
                african_names
            )
        )

        if matching_surnames:

            matched_surname = (
                sorted(
                    matching_surnames
                )[0]
            )

            print(
                f"MATCH FOUND:"
            )

            print(
                f"Author: {author}"
            )

            print(
                f"Surname Match: "
                f"{matched_surname}"
            )

            matches.append(
                (
                    title,
                    author,
                    matched_surname
                )
            )

# ============================================================
# REMOVE DUPLICATES
# ============================================================

matches = list(set(matches))

matches.sort(
    key=lambda x: x[0].lower()
)

# ============================================================
# SAVE RESULTS
# ============================================================

print(
    "\nSaving results..."
)

with open(
    output_file,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "AFRICAN OPEN SOURCE BOOKS\n"
    )

    f.write(
        "=" * 80 + "\n\n"
    )

    for (
        title,
        author,
        surname
    ) in matches:

        f.write(
            f"BOOK: {title}\n"
        )

        f.write(
            f"AUTHOR: {author}\n"
        )

        f.write(
            f"MATCHED SURNAME: "
            f"{surname}\n"
        )

        f.write(
            "-" * 80 + "\n"
        )

# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 80)

print("PROCESS COMPLETE")

print("=" * 80)

print(
    f"Total open-source books checked: "
    f"{len(book_titles):,}"
)

print(
    f"African author matches found: "
    f"{len(matches):,}"
)

print(
    f"\nResults saved to:\n"
    f"{output_file}"
)