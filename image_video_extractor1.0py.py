
import os
import json
import time
import hashlib
import random
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np

# ============================================================
# SoakinGarri Image/Video Extractor (ImageIdea / Reality^Context)
# - Reads JSONL jobs (url + tags + extraction config)
# - Downloads via yt-dlp (YouTube supported + hardened)
# - Extracts frames/clips via ffmpeg
# - Updates data/media_catalog.json
# ============================================================

SOAKINGARRI_BASE = os.getenv("SOAKINGARRI_BASE")
BASE_DIR = Path(SOAKINGARRI_BASE).resolve() if SOAKINGARRI_BASE else Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

# ImageObject = ImageIdea folder (Reality media library)
IMAGEIDEA_DIR = DATA_DIR / "ImageIdea"
VIDEO_DIR = IMAGEIDEA_DIR / "videos"
IMAGE_DIR = IMAGEIDEA_DIR / "images"
CLIP_DIR  = IMAGEIDEA_DIR / "clips"

MEDIA_CATALOG_PATH = DATA_DIR / "media_catalog.json"

for d in (DATA_DIR, IMAGEIDEA_DIR, VIDEO_DIR, IMAGE_DIR, CLIP_DIR):
    d.mkdir(parents=True, exist_ok=True)

# Optional: cookies file for restricted YouTube videos (only if you need it later)
YTDLP_COOKIES = os.getenv("SOAKINGARRI_YTDLP_COOKIES", "").strip()  # path to cookies.txt

# -----------------------------
# Tag normalization (match app/suggestor)
# -----------------------------
KEY_ALIASES = {"heirachy": "hierarchy"}
PLACE_BOOLEAN_KEYS = ["mall", "party", "restaurant", "in_the_club"]

def normalize_tags(tags: Dict[str, Any]) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    tags = tags or {}
    for k, v in tags.items():
        k2 = KEY_ALIASES.get(k, k)
        if v is None:
            continue
        if isinstance(v, list):
            out[k2] = [str(x) for x in v]
        else:
            out[k2] = [str(v)]

    places = set(out.get("place", []))
    for k in PLACE_BOOLEAN_KEYS:
        k2 = KEY_ALIASES.get(k, k)
        if k2 in out:
            val = (out[k2][0] if out[k2] else "").lower()
            if val == "yes":
                places.add("club" if k2 == "in_the_club" else k2)
            out.pop(k2, None)
    if places:
        out["place"] = sorted(list(places))
    return out

# -----------------------------
# Catalog
# -----------------------------
def load_catalog() -> Dict[str, Any]:
    if not MEDIA_CATALOG_PATH.exists():
        return {"version": "1.0", "assets": []}
    with open(MEDIA_CATALOG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def save_catalog(catalog: Dict[str, Any]) -> None:
    catalog["version"] = catalog.get("version") or "1.0"
    catalog["updated_at"] = int(time.time())
    with open(MEDIA_CATALOG_PATH, "w", encoding="utf-8") as f:
        json.dump(catalog, f, ensure_ascii=False, indent=2)

def make_asset_id(path: str, tags: Dict[str, Any], source_url: str = "") -> str:
    raw = f"{path}||{json.dumps(tags, sort_keys=True)}||{source_url}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()

# -----------------------------
# Helpers
# -----------------------------
def run(cmd: List[str], check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=check)

def ensure_tools_available() -> None:
    for tool in ("yt-dlp", "ffmpeg", "ffprobe"):
        try:
            run([tool, "-version"], check=False)
        except Exception as e:
            raise RuntimeError(f"Missing required tool '{tool}'. Install it and ensure it's on PATH.") from e

def is_youtube_url(url: str) -> bool:
    u = (url or "").lower()
    return "youtube.com" in u or "youtu.be" in u

# -----------------------------
# YouTube-hardened download
# -----------------------------
def download_video(url: str, out_dir: Path) -> Path:
    """
    Downloads a video using yt-dlp.
    Hardened for YouTube:
      - no playlist
      - best video+audio selection
      - remux to mp4
      - retries
      - stable filename with video id
    Returns: downloaded mp4 (or best available) file path.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # stable naming prevents collisions and helps dedupe
    out_template = str(out_dir / "%(id)s_%(title).80s.%(ext)s")

    cmd = [
        "yt-dlp",
        "--no-playlist",
        "--retries", "10",
        "--fragment-retries", "10",
        "--concurrent-fragments", "4",
        "-o", out_template,
        "-f", "bv*+ba/b",          # best video+audio, else best single file
        "--remux-video", "mp4",     # ensure mp4 container when possible
        url
    ]

    # Optional cookies support (for restricted content)
    if YTDLP_COOKIES:
        cmd.extend(["--cookies", YTDLP_COOKIES])

    r = run(cmd, check=True)

    # Find most recent file in out_dir
    candidates = sorted(out_dir.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise RuntimeError("yt-dlp download produced no output file.")
    return candidates[0]

def get_video_duration(video_path: Path) -> float:
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(video_path)
    ]
    r = run(cmd, check=True)
    return float((r.stdout or "").strip() or 0.0)

# -----------------------------
# Timestamp generator
# -----------------------------
def generate_timestamps(duration: float, mode: str, count: int = 10, interval: float = 5.0,
                        custom: Optional[List[float]] = None) -> List[float]:
    if duration <= 0:
        return []
    if mode == "random":
        return sorted(random.uniform(0, duration) for _ in range(count))
    elif mode == "interval":
        return list(np.arange(0, duration, interval))
    elif mode == "custom":
        return sorted(custom or [])
    else:
        raise ValueError(f"Invalid mode: {mode}")

# -----------------------------
# Extraction
# -----------------------------
def extract_images(video_path: Path, timestamps: List[float], out_dir: Path, prefix: str) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs: List[Path] = []
    for i, t in enumerate(timestamps):
        output = out_dir / f"{prefix}_frame_{i:03d}.jpg"
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-ss", str(t), "-i", str(video_path),
            "-frames:v", "1", "-q:v", "2", "-y", str(output)
        ]
        run(cmd, check=True)
        if output.exists():
            outputs.append(output)
    return outputs

def extract_clips(video_path: Path, timestamps: List[float], clip_length: float, out_dir: Path, prefix: str) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs: List[Path] = []
    for i, t in enumerate(timestamps):
        output = out_dir / f"{prefix}_clip_{i:03d}.mp4"
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-ss", str(t), "-i", str(video_path),
            "-t", str(clip_length),
            "-c:v", "libx264", "-c:a", "aac", "-y", str(output)
        ]
        run(cmd, check=True)
        if output.exists():
            outputs.append(output)
    return outputs

# -----------------------------
# Jobs
# -----------------------------
def read_jobs(jsonl_path: Path) -> List[Dict[str, Any]]:
    if not jsonl_path.exists():
        raise FileNotFoundError(f"Jobs file not found: {jsonl_path}")
    jobs: List[Dict[str, Any]] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            jobs.append(json.loads(line))
    return jobs

def validate_job(job: Dict[str, Any]) -> None:
    url = (job.get("url") or "").strip()
    if not url:
        raise ValueError("Job is missing 'url' (paste the link into the job file).")

    # YouTube is supported explicitly; other URLs may also work if yt-dlp supports them
    if is_youtube_url(url):
        pass  # allowed

    tags = job.get("tags", {}) or {}
    pol = str(tags.get("polarity", "")).strip().lower()  # Option 1: string
    if pol not in ("positive", "negative"):
        raise ValueError(
            "Job.tags.polarity must be a STRING: 'positive' or 'negative'. "
            "This polarity represents Reality polarity (opposite of Expectation)."
        )

def job_prefix(job: Dict[str, Any]) -> str:
    url = (job.get("url") or "").strip()
    h = hashlib.sha1(url.encode("utf-8")).hexdigest()[:10]
    return f"sg_{h}"

def add_assets_to_catalog(catalog: Dict[str, Any], paths: List[Path], kind: str,
                          tags: Dict[str, Any], source_url: str,
                          extra_meta: Optional[Dict[str, Any]] = None) -> int:
    extra_meta = extra_meta or {}
    existing_paths = {a.get("path") for a in catalog.get("assets", [])}

    added = 0
    for p in paths:
        if not p.exists():
            continue
        sp = str(p.resolve())
        if sp in existing_paths:
            continue

        asset_tags = normalize_tags(tags)  # normalize into lists (catalog format)
        asset_id = make_asset_id(sp, asset_tags, source_url=source_url)

        catalog.setdefault("assets", []).append({
            "asset_id": asset_id,
            "kind": kind,               # "image" or "video"
            "path": sp,
            "tags": asset_tags,         # includes polarity (Reality polarity)
            "source_url": source_url,
            "created_at": int(time.time()),
            **extra_meta
        })
        added += 1
    return added

# -----------------------------
# Runner
# -----------------------------
def run_extractor_jobs(jobs_path: Path) -> Dict[str, Any]:
    ensure_tools_available()

    catalog = load_catalog()
    jobs = read_jobs(jobs_path)

    total_jobs = len(jobs)
    jobs_ok = 0
    jobs_failed = 0
    total_added_assets = 0

    for idx, job in enumerate(jobs, start=1):
        try:
            validate_job(job)
            url = (job["url"] or "").strip()

            # Tags: keep as-is (Option 1 polarity string); catalog normalization happens later
            tags = job.get("tags", {}) or {}

            # Download
            video_path = download_video(url, VIDEO_DIR)
            duration = get_video_duration(video_path)

            # Range window
            start = float(job.get("start", 0) or 0)
            end = job.get("end", None)
            end = float(end) if end not in (None, "", "None") else duration
            end = min(end, duration)
            if end <= start:
                raise ValueError(f"Invalid start/end. start={start}, end={end}, duration={duration}")

            usable_duration = end - start

            # Settings
            frames = bool(job.get("frames", True))
            frame_mode = str(job.get("frame_mode", "interval"))
            frame_interval = float(job.get("frame_interval", 5))
            frame_count = int(job.get("frame_count", 10))
            frame_custom = job.get("frame_custom_times", None)

            clips = bool(job.get("clips", True))
            clip_mode = str(job.get("clip_mode", "random"))
            clip_count = int(job.get("clip_count", 8))
            clip_length = float(job.get("clip_length", 2))

            prefix = job_prefix(job)

            extracted_images: List[Path] = []
            if frames:
                image_times = generate_timestamps(
                    usable_duration,
                    frame_mode,
                    count=frame_count,
                    interval=frame_interval,
                    custom=frame_custom
                )
                image_times = [t + start for t in image_times]
                extracted_images = extract_images(video_path, image_times, IMAGE_DIR, prefix)

            extracted_clips: List[Path] = []
            if clips:
                clip_times = generate_timestamps(
                    usable_duration,
                    clip_mode,
                    count=clip_count,
                    interval=frame_interval,
                    custom=None
                )
                clip_times = [t + start for t in clip_times]
                extracted_clips = extract_clips(video_path, clip_times, clip_length, CLIP_DIR, prefix)

            meta = {
                "job_index": idx,
                "job_source": str(jobs_path.name),
                "range_start": start,
                "range_end": end,
                "downloaded_video": str(video_path.resolve()),
                "duration": duration
            }

            added_imgs = add_assets_to_catalog(catalog, extracted_images, "image", tags, url, extra_meta=meta)
            added_vids = add_assets_to_catalog(catalog, extracted_clips, "video", tags, url, extra_meta=meta)

            total_added_assets += (added_imgs + added_vids)
            jobs_ok += 1
            print(f"[OK {idx}/{total_jobs}] added images={added_imgs}, clips={added_vids} url={url}")

        except Exception as e:
            jobs_failed += 1
            print(f"[FAIL {idx}/{total_jobs}] {e}")

    save_catalog(catalog)

    return {
        "jobs_total": total_jobs,
        "jobs_ok": jobs_ok,
        "jobs_failed": jobs_failed,
        "assets_added": total_added_assets,
        "media_catalog_path": str(MEDIA_CATALOG_PATH.resolve()),
        "imageidea_dir": str(IMAGEIDEA_DIR.resolve())
    }

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python image_video_extractor.py <jobs.jsonl>")

    jobs_path = Path(sys.argv[1]).resolve()
    report = run_extractor_jobs(jobs_path)
    print(json.dumps(report, ensure_ascii=False, indent=2))
