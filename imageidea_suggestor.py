
import os, json, sqlite3
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, Any, List, Tuple, Optional

SOAKINGARRI_BASE = os.getenv("SOAKINGARRI_BASE")
BASE_DIR = Path(SOAKINGARRI_BASE).resolve() if SOAKINGARRI_BASE else Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

DB_PATH = DATA_DIR / "soakin.db"
MEDIA_CATALOG_PATH = DATA_DIR / "media_catalog.json"

SUGGESTED_JOBS_BY_COUNT = DATA_DIR / "extractor_jobs_suggested_by_count.jsonl"
SUGGESTED_JOBS_BY_SCORE = DATA_DIR / "extractor_jobs_suggested_by_score.jsonl"

# -----------------------------
# Normalization
# -----------------------------
KEY_ALIASES = {"heirachy": "hierarchy"}
PLACE_BOOLEAN_KEYS = ["mall", "party", "restaurant", "in_the_club"]

def normalize_tags(tags: Dict[str, Any]) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for k, v in (tags or {}).items():
        k2 = KEY_ALIASES.get(k, k)
        if v is None:
            continue
        if isinstance(v, list):
            out[k2] = [str(x) for x in v]
        else:
            out[k2] = [str(v)]

    # Convert boolean place flags to place=...
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
# Coverage Profile (MVP)
# -----------------------------
def media_suggestor_keys_profile_1(draft_tags: Dict[str, List[str]]) -> List[str]:
    """
    MVP publish-easy coverage:
      Always require city + er_category.
      Then include whichever exist (up to 2 more) from:
        social_pressure, time_of_the_day, workplace, place, role
    """
    keys = ["city", "er_category"]
    candidates = ["social_pressure", "time_of_the_day", "workplace", "place", "role"]
    for k in candidates:
        if k in draft_tags:
            keys.append(k)
    return keys[:4]  # cap strictness so you can publish in 100s

def signature(tags: Dict[str, List[str]], keys: List[str]) -> Tuple[Tuple[str, str], ...]:
    sig = []
    for k in keys:
        vals = sorted(tags.get(k, []))
        sig.append((k, "|".join(vals) if vals else "__NONE__"))
    return tuple(sig)

def overlaps(a: List[str], b: List[str]) -> bool:
    return not set(a).isdisjoint(set(b))

def asset_matches(draft_tags: Dict[str, List[str]], asset_tags: Dict[str, List[str]], keys: List[str]) -> bool:
    for k in keys:
        if k not in asset_tags:
            return False
        if not overlaps(draft_tags.get(k, []), asset_tags.get(k, [])):
            return False
    return True

# -----------------------------
# IO
# -----------------------------
def db_connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def load_catalog() -> Dict[str, Any]:
    if not MEDIA_CATALOG_PATH.exists():
        return {"assets": []}
    with open(MEDIA_CATALOG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

# -----------------------------
# Core: compute missing + dual rankings
# -----------------------------
def compute_missing(top_n: int = 50) -> Dict[str, Any]:
    if not DB_PATH.exists():
        raise FileNotFoundError(f"Missing DB: {DB_PATH}")

    catalog = load_catalog()
    assets = catalog.get("assets", [])

    # drafts
    conn = db_connect()
    cur = conn.cursor()
    cur.execute("SELECT draft_id, tags_json, score FROM drafts")
    drafts = cur.fetchall()
    conn.close()

    draft_rows = []
    for d in drafts:
        tags = normalize_tags(json.loads(d["tags_json"]))
        draft_rows.append({
            "draft_id": d["draft_id"],
            "tags": tags,
            "score": float(d["score"] or 0.0)
        })

    # assets
    asset_rows = []
    for a in assets:
        asset_rows.append({
            "asset_id": a.get("asset_id"),
            "tags": normalize_tags(a.get("tags", {})),
            "kind": a.get("kind"),
            "path": a.get("path")
        })

    # Demand aggregate by signature (but keys differ per draft => store keys used)
    demand_count = Counter()
    demand_score = defaultdict(float)
    sig_keys_map: Dict[Tuple[Tuple[str,str],...], List[str]] = {}


    for d in draft_rows:
        keys = media_suggestor_keys_profile_1(d["tags"])
        sig = signature(d["tags"], keys)
        demand_count[sig] += 1
        demand_score[sig] += d["score"]
        sig_keys_map[sig] = keys

    # Coverage check
    covered = set()
    for sig in demand_count.keys():
        keys = sig_keys_map[sig]
        sig_dict = {k: (v.split("|") if v != "__NONE__" else []) for k, v in sig}

        for a in asset_rows:
            if a["path"] and os.path.exists(a["path"]) and asset_matches(sig_dict, a["tags"], keys):
                covered.add(sig)
                break

    missing = [sig for sig in demand_count.keys() if sig not in covered]

    # Ranking A: by demand frequency
    missing_by_count = sorted(
        missing,
        key=lambda s: (demand_count[s], demand_score[s]),
        reverse=True
    )[:top_n]

    # Ranking B: by total score sum
    missing_by_score = sorted(
        missing,
        key=lambda s: (demand_score[s], demand_count[s]),
        reverse=True
    )[:top_n]

    def pack(sig_list):
        out = []
        for sig in sig_list:
            out.append({
                "keys_used": sig_keys_map[sig],
                "signature": {k: v for k, v in sig},
                "draft_count": demand_count[sig],
                "draft_score_sum": round(demand_score[sig], 2)
            })
        return out

    return {
        "total_drafts": len(draft_rows),
        "total_assets": len(asset_rows),
        "unique_signatures": len(demand_count),
        "covered_signatures": len(covered),
        "missing_signatures": len(missing),
        "top_missing_by_count": pack(missing_by_count),
        "top_missing_by_score": pack(missing_by_score)
    }

def write_jobs(path: Path, items: List[Dict[str, Any]], template: Dict[str, Any]) -> int:
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            tags = {}
            sig = item["signature"]
            for k, joined in sig.items():
                if joined == "__NONE__":
                    continue
                tags[k] = joined.split("|")

            job = dict(template)
            job["url"] = ""      # you paste link manually
            job["tags"] = tags
            f.write(json.dumps(job, ensure_ascii=False) + "\n")
            n += 1
    return n

if __name__ == "__main__":
    report = compute_missing(top_n=50)
    print(json.dumps(report, ensure_ascii=False, indent=2))

    template = {
        "frames": True,
        "frame_mode": "interval",
        "frame_interval": 5,
        "frame_count": 10,
        "clips": True,
        "clip_mode": "random",
        "clip_count": 8,
        "clip_length": 2,
        "start": 0,
        "end": None
    }

    nA = write_jobs(SUGGESTED_JOBS_BY_COUNT, report["top_missing_by_count"], template)
    nB = write_jobs(SUGGESTED_JOBS_BY_SCORE, report["top_missing_by_score"], template)

    print(f"\nWrote {nA} suggested jobs (Rank A: by count) -> {SUGGESTED_JOBS_BY_COUNT}")
    print(f"Wrote {nB} suggested jobs (Rank B: by score) -> {SUGGESTED_JOBS_BY_SCORE}")
