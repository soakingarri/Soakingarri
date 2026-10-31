import argparse
import fcntl
import json
import mimetypes
import os
import re
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlparse

import boto3
import requests

from met_people_filter import depicts_people


# ==================================================
# MET OPEN ACCESS API
# ==================================================

MET_API = "https://collectionapi.metmuseum.org/public/collection/v1"

# Egyptian Art Department
DEPARTMENT_ID = 10

S3_BUCKET = "rag-african-history-images"
S3_REGION = "us-east-2"

HEADERS = {
    "User-Agent": "SoakinGarriImageScraper/1.0 (soakingarri@gmail.com)"
}

# ==================================================
# AWS
# ==================================================

s3 = boto3.client(
    "s3",
    region_name=S3_REGION
)

# requests.Session isn't guaranteed thread-safe, so each worker gets its own.
thread_local = threading.local()


def get_session():

    if not hasattr(thread_local, "session"):

        thread_local.session = requests.Session()
        thread_local.session.headers.update(HEADERS)

    return thread_local.session


# Records are appended here one per line as soon as each image is uploaded,
# so the script can be stopped and resumed without losing anything.
RECORDS_FILE = Path(__file__).parent / "met_egypt_dataset.jsonl"
FINAL_FILE = Path(__file__).parent / "met_egypt_dataset.json"

# Object IDs already checked and skipped (no image, or not an original
# depiction of people/gods), so reruns don't fetch their metadata again.
SKIPPED_FILE = Path(__file__).parent / "met_egypt_skipped_ids.txt"

# Images are stored as <met_object_id><ext>, so "already uploaded?" is a
# lookup by object ID and reruns can never duplicate or misnumber anything.
S3_PREFIX = "ancient-egypt/met/"

MAX_RETRIES = 6

# Most objects are now rejected after one metadata request, and back-to-back
# requests trip the MET's bot protection (HTTP 403). Spacing requests evenly
# keeps roughly the same throughput (~85/min) without the bursts.
MIN_REQUEST_INTERVAL = 0.7

request_lock = threading.Lock()
last_request_time = 0.0


def wait_for_request_slot():

    global last_request_time

    with request_lock:

        wait = last_request_time + MIN_REQUEST_INTERVAL - time.monotonic()

        if wait > 0:
            time.sleep(wait)

        last_request_time = time.monotonic()

# Rewrite the full JSON dataset every this many records.
CHECKPOINT_EVERY = 100

# The MET's bot protection returns 403s with 4 parallel workers, so the
# default is one at a time.
DEFAULT_WORKERS = 1


# ==================================================
# HTTP with retry / backoff
# ==================================================

def get_with_retry(url, timeout=60):

    delay = 5

    for _ in range(MAX_RETRIES):

        wait_for_request_slot()

        try:

            r = get_session().get(
                url,
                timeout=timeout
            )

        except requests.RequestException as e:

            print(f"  network error ({e}), retrying in {delay}s")

            time.sleep(delay)
            delay = min(delay * 2, 300)
            continue

        if r.status_code in (403, 429) or r.status_code >= 500:

            print(
                f"  HTTP {r.status_code} for {url}, "
                f"retrying in {delay}s"
            )

            time.sleep(delay)
            delay = min(delay * 2, 300)
            continue

        return r

    print(f"  giving up on {url}")

    return None


# ==================================================
# S3 Upload
# ==================================================

def upload_to_s3(local_path, key):

    content_type = (
        mimetypes.guess_type(key)[0]
        or "application/octet-stream"
    )

    s3.upload_file(
        local_path,
        S3_BUCKET,
        key,
        ExtraArgs={
            "ContentType": content_type
        }
    )

    return f"s3://{S3_BUCKET}/{key}"


def list_uploaded_keys():

    # Maps met_object_id -> S3 key for every object-ID-named image.
    # Old counter-named files (9-digit names) are ignored.

    uploaded = {}

    paginator = s3.get_paginator("list_objects_v2")

    for page in paginator.paginate(
        Bucket=S3_BUCKET,
        Prefix=S3_PREFIX
    ):

        for o in page.get("Contents", []):

            stem = Path(o["Key"]).stem

            if re.fullmatch(r"\d+", stem) and len(stem) != 9:
                uploaded[int(stem)] = o["Key"]

    return uploaded


# ==================================================
# MET API FUNCTIONS
# ==================================================

def get_egyptian_object_ids():

    url = (
        f"{MET_API}/objects"
        f"?departmentIds={DEPARTMENT_ID}"
    )

    r = get_with_retry(url)

    if r is None:
        raise RuntimeError("Could not fetch MET object list")

    r.raise_for_status()

    data = r.json()

    return data.get("objectIDs", [])


def get_object(object_id):

    url = f"{MET_API}/objects/{object_id}"

    r = get_with_retry(url)

    if r is None or r.status_code != 200:

        print(
            f"SKIPPED {object_id}: "
            f"HTTP {r.status_code if r is not None else 'no response'}"
        )

        return None

    return r.json()


def get_image_info(obj):

    image_url = (
        obj.get("primaryImage")
        or obj.get("primaryImageSmall")
    )

    if not image_url:
        return None, None

    extension = (
        Path(
            urlparse(
                image_url
            ).path
        ).suffix.lower()
    )

    if not extension:
        extension = ".jpg"

    return image_url, extension


# ==================================================
# DOWNLOAD
# ==================================================

def download_file(url):

    r = get_with_retry(
        url,
        timeout=120
    )

    if r is None:
        raise RuntimeError(f"Could not download {url}")

    r.raise_for_status()

    fd, path = tempfile.mkstemp()

    with os.fdopen(fd, "wb") as f:
        f.write(r.content)

    return path


# ==================================================
# JSON RECORD
# ==================================================

def build_record(
    image_id,
    obj,
    s3_url,
    image_url
):

    tags = []

    for field in [
        obj.get("culture"),
        obj.get("period"),
        obj.get("dynasty"),
        obj.get("reign"),
        obj.get("objectName")
    ]:

        if field:
            tags.append(field)

    return {
        "image_id": str(image_id).zfill(9),

        "met_object_id": obj.get("objectID"),

        "depicts_people": True,

        "title": obj.get("title", ""),

        "description": (
            obj.get("objectName", "")
        ),

        "category": "Ancient Egypt",

        "tags": tags,

        "source": {
            "name": "Metropolitan Museum of Art Open Access",
            "url": obj.get("objectURL", "")
        },

        "location": {
            "country": obj.get("country", "Egypt"),
            "original_name": "Kemet",
            "city": obj.get("city", "")
        },

        "date_created": obj.get(
            "objectDate",
            ""
        ),

        "images": [
            s3_url
        ],

        "metadata": {

            "dataset":
                "SoakinGarri Images",

            "version":
                "1.0",

            "continent":
                "Africa",

            "region":
                "North Africa",

            "language":
                "Meduneter",

            "met_object_id":
                obj.get("objectID"),

            "met_image_url":
                image_url,

            "department":
                obj.get("department"),

            "culture":
                obj.get("culture"),

            "period":
                obj.get("period"),

            "dynasty":
                obj.get("dynasty"),

            "reign":
                obj.get("reign"),

            "medium":
                obj.get("medium"),

            "classification":
                obj.get("classification"),

            "repository":
                obj.get("repository"),

            "excavation":
                obj.get("excavation"),

            "rights":
                "CC0"
        }
    }


# ==================================================
# RECORDS
# ==================================================

def load_records():

    records = []

    if RECORDS_FILE.exists():

        with open(RECORDS_FILE, encoding="utf-8") as f:

            for line in f:

                if not line.strip():
                    continue

                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    # A line cut off by a kill mid-write; that object is
                    # redone on the next run.
                    print("Ignoring incomplete record line")

    return records


def write_dataset():

    records = load_records()

    # Write to a temp file first so a kill mid-write never corrupts the
    # previous checkpoint.
    tmp = FINAL_FILE.with_suffix(".json.tmp")

    with open(tmp, "w", encoding="utf-8") as f:

        json.dump(
            records,
            f,
            ensure_ascii=False,
            indent=2
        )

    os.replace(tmp, FINAL_FILE)

    return len(records)


class Progress:

    # Shared state for the workers: image numbering, duplicate tracking and
    # writes to the records file all happen under one lock.

    def __init__(self, records):

        self.lock = threading.Lock()

        # Records filtered out of the dataset leave gaps in image_id, so
        # number from the highest one rather than the record count.
        self.counter = max(
            (int(r["image_id"]) for r in records),
            default=0
        ) + 1

        self.done_ids = {
            r["metadata"]["met_object_id"] for r in records
        }

        self.image_urls = {
            r["metadata"].get("met_image_url") for r in records
        }

        self.skipped_ids = set()

        if SKIPPED_FILE.exists():

            with open(SKIPPED_FILE, encoding="utf-8") as f:

                self.skipped_ids = {
                    int(line) for line in f if line.strip().isdigit()
                }

    def mark_skipped(self, object_id):

        with self.lock:

            self.skipped_ids.add(object_id)

            with open(SKIPPED_FILE, "a", encoding="utf-8") as f:
                f.write(f"{object_id}\n")

    def claim_image_url(self, image_url):

        with self.lock:

            if image_url in self.image_urls:
                return False

            self.image_urls.add(image_url)

            return True

    def release_image_url(self, image_url):

        with self.lock:
            self.image_urls.discard(image_url)

    def save(self, obj, s3_url, image_url):

        with self.lock:

            image_id = self.counter

            record = build_record(
                image_id,
                obj,
                s3_url,
                image_url
            )

            with open(RECORDS_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

            self.done_ids.add(obj.get("objectID"))

            self.counter += 1

            print(f"{image_id}: {obj.get('title')}")

            if image_id % CHECKPOINT_EVERY == 0:
                print(f"Checkpoint: saved {write_dataset()} records")


# ==================================================
# WORKER
# ==================================================

def process_object(object_id, progress, uploaded):

    try:

        obj = get_object(object_id)

        if not obj:
            return

        image_url, extension = get_image_info(obj)

        if not image_url or not depicts_people(
            obj.get("title"),
            obj.get("objectName"),
            obj.get("medium"),
            obj.get("period"),
            obj.get("dynasty")
        ):

            progress.mark_skipped(object_id)

            return

        if not progress.claim_image_url(image_url):

            print(f"DUPLICATE image for {object_id}, skipping")

            return

        try:

            if object_id in uploaded:

                # Uploaded by a run that stopped before saving its record.
                s3_url = f"s3://{S3_BUCKET}/{uploaded[object_id]}"

            else:

                local_file = download_file(
                    image_url
                )

                try:

                    s3_url = upload_to_s3(
                        local_file,
                        f"{S3_PREFIX}{object_id}{extension}"
                    )

                finally:

                    os.remove(local_file)

        except Exception:

            progress.release_image_url(image_url)

            raise

        progress.save(obj, s3_url, image_url)

        time.sleep(0.1)

    except Exception as e:

        print(
            f"FAILED {object_id}",
            e
        )


# ==================================================
# MAIN
# ==================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help="number of objects downloaded in parallel"
    )

    args = parser.parse_args()

    ids = get_egyptian_object_ids()

    print(
        f"Found {len(ids)} Egyptian objects"
    )

    progress = Progress(load_records())

    uploaded = list_uploaded_keys()

    pending = [
        i for i in ids
        if i not in progress.done_ids
        and i not in progress.skipped_ids
    ]

    print(
        f"{len(progress.done_ids)} already done, "
        f"{len(progress.skipped_ids)} skipped earlier, "
        f"{len(pending)} objects left, "
        f"{args.workers} workers"
    )

    executor = ThreadPoolExecutor(args.workers)

    try:

        for object_id in pending:

            executor.submit(
                process_object,
                object_id,
                progress,
                uploaded
            )

        executor.shutdown(wait=True)

    except KeyboardInterrupt:

        # Drop queued objects; in-flight ones finish and save their records.
        print("Stopping after in-flight downloads finish...")

        executor.shutdown(wait=True, cancel_futures=True)

    print(
        f"Saved {write_dataset()} records"
    )


if __name__ == "__main__":

    # Two concurrent runs would race on the records file and numbering.
    lock = open(Path(__file__).parent / ".met_scraper.lock", "w")

    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("Another MET scraper run is already in progress")

    main()
