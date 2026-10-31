import json
import mimetypes
import os
import tempfile
import time
from pathlib import Path
from urllib.parse import urlparse

import boto3
import requests
from bs4 import BeautifulSoup


COMMONS_API = "https://commons.wikimedia.org/w/api.php"

# boto3 wants the bucket name, not the console URL
S3_BUCKET = "rag-african-history-images"
S3_REGION = "us-east-2"

# Wikimedia rejects requests without a descriptive User-Agent (HTTP 403)
HEADERS = {
    "User-Agent": "SoakinGarriImageScraper/1.0 (soakingarri@gmail.com)"
}

CATEGORIES = [
    "Category:Ancient Egyptian temples",
    "Category:Ancient Egyptian wall paintings",
    "Category:Ancient Egyptian hieroglyphs",
    "Category:Ancient Egyptian statues",
    "Category:Ancient Egyptian artifacts"
]

s3 = boto3.client("s3", region_name=S3_REGION)

session = requests.Session()
session.headers.update(HEADERS)


def upload_to_s3(local_path, key):
    content_type = mimetypes.guess_type(key)[0] or "application/octet-stream"
    s3.upload_file(
        local_path,
        S3_BUCKET,
        key,
        ExtraArgs={"ContentType": content_type}
    )
    return f"s3://{S3_BUCKET}/{key}"


def get_category_members(category, limit=500):
    params = {
        "action": "query",
        "list": "categorymembers",
        "cmtitle": category,
        "cmtype": "file",
        "cmlimit": limit,
        "format": "json"
    }

    members = []

    # Follow cmcontinue so categories with more files than one page are covered
    while True:
        r = session.get(COMMONS_API, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()

        members.extend(data.get("query", {}).get("categorymembers", []))

        if "continue" not in data:
            break
        params.update(data["continue"])

    return members


def get_image_info(title):
    params = {
        "action": "query",
        "titles": title,
        "prop": "imageinfo",
        "iiprop": "url|extmetadata",
        "format": "json"
    }

    r = session.get(COMMONS_API, params=params, timeout=30)
    r.raise_for_status()

    pages = r.json()["query"]["pages"]

    return list(pages.values())[0]


def download_file(url):
    r = session.get(url, timeout=120)
    r.raise_for_status()

    fd, path = tempfile.mkstemp()

    with os.fdopen(fd, "wb") as f:
        f.write(r.content)

    return path


def build_record(
        image_id,
        title,
        description,
        source_url,
        s3_url
):
    return {
        "image_id": str(image_id).zfill(9),
        "title": title,
        "description": description,
        "category": "Ancient Egypt",
        "tags": [],
        "source": {
            "name": "Wikimedia Commons",
            "url": source_url
        },
        "location": {
            "country": "Egypt",
            "original_name": "Kemet",
            "city": ""
        },
        "date_created": "",
        "images": [
            s3_url
        ],
        "metadata": {
            "dataset": "SoakinGarri Images",
            "version": "1.0",
            "continent": "Africa",
            "region": "North Africa",
            "language": "Meduneter"
        }
    }


def main():

    records = []
    counter = 1

    for category in CATEGORIES:

        members = get_category_members(category)

        for member in members:

            try:

                info = get_image_info(member["title"])

                image_info = info["imageinfo"][0]

                image_url = image_info["url"]

                metadata = image_info.get("extmetadata", {})

                title = member["title"]

                description = ""

                if "ImageDescription" in metadata:
                    # extmetadata values are HTML fragments
                    description = BeautifulSoup(
                        metadata["ImageDescription"].get("value", ""),
                        "html.parser"
                    ).get_text(" ", strip=True)

                local_file = download_file(image_url)

                # Commons appends ?utm_... query params, so take the suffix from the path only
                extension = Path(urlparse(image_url).path).suffix.lower()

                key = (
                    f"ancient-egypt/"
                    f"{counter:09d}"
                    f"{extension}"
                )

                try:
                    s3_url = upload_to_s3(
                        local_file,
                        key
                    )
                finally:
                    os.remove(local_file)

                record = build_record(
                    counter,
                    title,
                    description,
                    image_url,
                    s3_url
                )

                records.append(record)

                print(
                    f"{counter}: {title}"
                )

                counter += 1

                # Stay polite to upload.wikimedia.org (it rate-limits bulk downloads)
                time.sleep(1)

            except Exception as e:
                print(
                    "FAILED",
                    member["title"],
                    e
                )

    with open(
            "egypt_dataset.json",
            "w",
            encoding="utf-8"
    ) as f:

        json.dump(
            records,
            f,
            ensure_ascii=False,
            indent=2
        )


if __name__ == "__main__":
    main()