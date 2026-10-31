
# app.py (SoakinGarri Meme Generator - SQL ER + Expectation Overlay Only)
# Updated: Hard Opposite Rule + Expectation^Context + Reality^Context enforcement

import os, re, json, time, base64, random, hashlib, sqlite3, subprocess
from typing import Dict, Any, List, Optional, Tuple

from fastapi import FastAPI, HTTPException, Query, Body
from fastapi.responses import FileResponse

try:
    from PIL import Image, ImageDraw, ImageFont
except Exception:
    Image = None

# =========================
# CONFIG
# =========================

SOAKINGARRI_BASE = os.getenv("SoakinGarri")
BASE_DIR = os.path.abspath(SOAKINGARRI_BASE) if SOAKINGARRI_BASE else os.path.abspath(os.path.dirname(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data")
MEME_TEXT_DIR = os.path.join(DATA_DIR, "meme_texts")
CONTEXTS_DIR = os.path.join(DATA_DIR, "contexts")
MANIFEST_PATH = os.path.join(DATA_DIR, "contexts_manifest.json")

DB_PATH = os.path.join(DATA_DIR, "soakin.db")  # drafts/published/metrics AND ER table by default
ER_DB_PATH = os.getenv("SOAKINGARRI_ER_DB", DB_PATH)  # option: store ER table in separate sqlite file
ER_TABLE = os.getenv("SOAKINGARRI_ER_TABLE", "expectation_reality")

MEDIA_CATALOG_PATH = os.path.join(DATA_DIR, "media_catalog.json")
PUBLISHED_DIR = os.path.join(DATA_DIR, "published_assets")
os.makedirs(PUBLISHED_DIR, exist_ok=True)

# =========================
# TEXT GENERATION MODE
# =========================
# as_is  = use ER expectation text (current behavior)
# llm    = use SoakinGarri LLM (future)
SOAKINGARRI_TEXTGEN_MODE = os.getenv("SOAKINGARRI_TEXTGEN_MODE", "as_is").lower()

DEFAULT_FEED_MODE = os.getenv("SOAKINGARRI_FEED_MODE", "newest")  # newest | trending

# Expectation overlay settings (images)
FONT_PATH = os.getenv("SOAKINGARRI_FONT_PATH", os.path.join(DATA_DIR, "assets", "fonts", "Impact.ttf"))
FONT_SIZE = int(os.getenv("SOAKINGARRI_FONT_SIZE", "64"))
TEXT_MARGIN = int(os.getenv("SOAKINGARRI_TEXT_MARGIN", "40"))

# Expectation overlay settings (videos via ffmpeg drawtext)
FFMPEG_FONTFILE = os.getenv("SOAKINGARRI_FFMPEG_FONTFILE", FONT_PATH)

# HARD OPPOSITE RULE enforcement strength
MAX_ER_PICK_TRIES = int(os.getenv("SOAKINGARRI_MAX_ER_PICK_TRIES", "50"))

# matches Context00001 etc
ID_RE = re.compile(r"^Context(\d{5})$")

# =========================
# TAG NORMALIZATION (must match extractor)
# =========================

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

# =========================
# POLARITY + OPPOSITE (HARD RULE)
# =========================

POSITIVE_HINTS = {
    "success", "win", "rich", "love", "peace", "calm", "easy", "smooth", "stable",
    "answer", "fix", "goal", "approved", "promotion", "perfect", "clean", "sorted",
    "under control", "nailed it", "all good", "finally"
}
NEGATIVE_HINTS = {
    "complicated", "hidden costs", "pressures", "struggle", "problem", "worse", "stress",
    "fatigue", "frustration", "disillusionment", "resignation", "fail", "chaos", "mess",
    "broke", "rejected", "late", "panic", "pain", "disaster", "crash", "stuck",
    "not", "nope", "never", "nothing works", "ruined"
}

def polarity(text: str) -> str:
    t = (text or "").lower()
    pos = sum(1 for w in POSITIVE_HINTS if w in t)
    neg = sum(1 for w in NEGATIVE_HINTS if w in t)
    if pos > neg:
        return "positive"
    if neg > pos:
        return "negative"
    return "neutral"

def opposite_polarity(p: str) -> Optional[str]:
    if p == "positive":
        return "negative"
    if p == "negative":
        return "positive"
    return None  # neutral can't be "opposite" reliably

def opposite_ok(expectation_text: str, reality_text: str) -> bool:
    """
    HARD RULE: Expectation must always be opposite to Reality.
    - Neutral is NOT acceptable (can't guarantee opposition)
    """
    pe = polarity(expectation_text)
    pr = polarity(reality_text)
    if pe == "neutral" or pr == "neutral":
        return False
    return pe != pr

# =========================
# MEME TEXT GENERATOR (Expectation^Context) + MODE SWITCH
# =========================

TEMPLATES_POSITIVE = [
    "THIS WILL BE EASY.",
    "TODAY WILL GO SMOOTHLY.",
    "I’VE GOT THIS UNDER CONTROL.",
    "CLEAN WIN. NO STRESS.",
    "EVERYTHING WILL WORK PERFECTLY."
]

def expectation_pow_context(C: Dict[str, Any], expectation: str) -> str:
    """
    MemeText = Expectation^Context
    Keep it short: add one cue max (city/place/workplace) if it fits.
    """
    exp = (expectation or "").strip()
    if not exp:
        exp = "EXPECTATION"

    cue = None
    if C.get("city"):
        cue = str(C["city"])
    elif C.get("place"):
        cue = str(C["place"])
    elif C.get("workplace"):
        cue = str(C["workplace"])

    if cue:
        candidate = f"{cue}: {exp}"
        if len(candidate) <= 110:
            return candidate
    return exp

def generate_expectation_as_is(C: Dict[str, Any], er: Dict[str, Any]) -> str:
    """
    Current mode: expectation comes from ER table, then shaped by Context.
    Ensures non-neutral expectation (so it can be opposite to Reality).
    """
    raw = (er.get("expectation") or "").strip()
    exp = expectation_pow_context(C, raw)

    # Hard rule support: expectation must be opposable (non-neutral)
    if polarity(exp) == "neutral":
        exp = expectation_pow_context(C, random.choice(TEMPLATES_POSITIVE))

    return exp

def generate_expectation_llm(C: Dict[str, Any], er: Dict[str, Any]) -> str:
    """
    Future mode: SoakinGarri LLM generates fresh Expectation^Context.
    For now: safe placeholder so the app still runs.
    """
    # TODO: replace with SoakinGarri LLM call
    exp = expectation_pow_context(C, random.choice(TEMPLATES_POSITIVE))
    return exp

def generate_expectation(C: Dict[str, Any], er: Dict[str, Any]) -> str:
    """
    Mode switch entry point.
    """
    if SOAKINGARRI_TEXTGEN_MODE == "llm":
        return generate_expectation_llm(C, er)
    return generate_expectation_as_is(C, er)

# =========================
# COVERAGE PROFILE (publish-easy, consistent with coverage_suggester)
# =========================

def coverage_keys_profile_1(draft_tags: Dict[str, List[str]]) -> List[str]:
    keys = ["city", "er_category"]
    candidates = ["social_pressure", "time_of_the_day", "workplace", "place", "role"]
    for k in candidates:
        if k in draft_tags:
            keys.append(k)
    return keys[:4]

def overlaps(a: List[str], b: List[str]) -> bool:
    return not set(a).isdisjoint(set(b))

def asset_matches(draft_tags: Dict[str, List[str]], asset_tags: Dict[str, List[str]], keys: List[str]) -> bool:
    for k in keys:
        if k not in asset_tags:
            return False
        if not overlaps(draft_tags.get(k, []), asset_tags.get(k, [])):
            return False
    return True

# =========================
# DB
# =========================

def db_connect(path: str) -> sqlite3.Connection:
    os.makedirs(DATA_DIR, exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def db_init() -> None:
    conn = db_connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS drafts (
        draft_id TEXT PRIMARY KEY,
        context_id TEXT NOT NULL,
        er_id INTEGER NOT NULL,
        schema_key TEXT,
        expectation_text TEXT NOT NULL,
        reality_text TEXT NOT NULL,
        combo_text TEXT NOT NULL,
        tags_json TEXT NOT NULL,
        score REAL DEFAULT 0,
        created_at INTEGER NOT NULL
    );
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS published (
        meme_id TEXT PRIMARY KEY,
        draft_id TEXT NOT NULL,
        context_id TEXT NOT NULL,
        er_id INTEGER NOT NULL,
        expectation_text TEXT NOT NULL,
        reality_text TEXT NOT NULL,
        tags_json TEXT NOT NULL,
        asset_id TEXT NOT NULL,
        asset_kind TEXT NOT NULL,
        asset_path TEXT NOT NULL,
        published_at INTEGER NOT NULL
    );
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS metrics (
        meme_id TEXT PRIMARY KEY,
        views INTEGER DEFAULT 0,
        shares INTEGER DEFAULT 0,
        last_viewed_at INTEGER,
        last_shared_at INTEGER
    );
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS app_config (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL
    );
    """)
    cur.execute("INSERT OR IGNORE INTO app_config(key,value) VALUES('feed_mode', ?)", (DEFAULT_FEED_MODE,))
    conn.commit()
    conn.close()

def ensure_er_table_exists() -> None:
    conn = db_connect(ER_DB_PATH)
    cur = conn.cursor()
    cur.execute(f"""
    CREATE TABLE IF NOT EXISTS {ER_TABLE} (
        id INTEGER PRIMARY KEY,
        category TEXT,
        theme TEXT,
        expectation TEXT,
        reality TEXT,
        emotion TEXT
    );
    """)
    conn.commit()
    conn.close()

# =========================
# Context
# =========================

def load_json(path: str) -> Any:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing JSON file: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_manifest() -> Dict[str, Any]:
    return load_json(MANIFEST_PATH)

def parse_ctx_id(ctx_id: str) -> int:
    m = ID_RE.match(ctx_id)
    if not m:
        raise ValueError(f"Invalid Context ID: {ctx_id}")
    return int(m.group(1))

def shard_for_id(ctx_id: str, manifest: Dict[str, Any]) -> str:
    n = parse_ctx_id(ctx_id)
    for shard in manifest.get("shards", []):
        start = parse_ctx_id(shard["start_id"])
        end = parse_ctx_id(shard["end_id"])
        if start <= n <= end:
            return shard["name"]
    raise KeyError(f"{ctx_id} not found in any shard range")

def load_shard(shard_name: str) -> Dict[str, Any]:
    return load_json(os.path.join(CONTEXTS_DIR, shard_name))

def get_random_context(manifest: Dict[str, Any]) -> Dict[str, Any]:
    shards = manifest.get("shards", [])
    chosen = random.choice(shards)
    shard = load_shard(chosen["name"])
    ctx_id = random.choice(list(shard.keys()))
    out = {"id": ctx_id}
    out.update(shard[ctx_id])
    return out

def schema_key_from_context(C: Dict[str, Any]) -> str:
    keys = sorted([k for k in C.keys() if k != "id"])
    return "schema__" + "__".join(keys) if keys else "schema__empty"

# =========================
# ER from SQL
# =========================

def pick_random_er_sql() -> Dict[str, Any]:
    conn = db_connect(ER_DB_PATH)
    cur = conn.cursor()
    cur.execute(f"""
        SELECT id, category, theme, expectation, reality, emotion
        FROM {ER_TABLE}
        ORDER BY RANDOM()
        LIMIT 1;
    """)
    r = cur.fetchone()
    conn.close()
    if not r:
        raise RuntimeError(f"{ER_TABLE} table is empty. Import your ER rows first.")
    return dict(r)

# =========================
# Draft creation
# =========================

def build_combo_text(C: Dict[str, Any], er: Dict[str, Any], expectation_final: str) -> str:
    ctx_bits = [f"{k}={v}" for k, v in C.items() if k != "id"]
    ctx_summary = ", ".join(ctx_bits) if ctx_bits else "context=unspecified"
    return (
        f"[{C.get('id')} | {ctx_summary}] "
        f"[ER#{er.get('id')} | {er.get('category')} | {er.get('theme')}] "
        f"EXPECTATION: {expectation_final} | REALITY: {er.get('reality')}"
    )

def tags_from(C: Dict[str, Any], er: Dict[str, Any], expectation_final: str) -> Dict[str, Any]:
    tags: Dict[str, Any] = {}
    for k, v in C.items():
        if k != "id":
            tags.setdefault(k, []).append(v)

    tags.setdefault("er_category", []).append(er.get("category", "unknown"))
    tags.setdefault("er_theme", []).append(er.get("theme", "unknown"))
    tags.setdefault("er_emotion", []).append(er.get("emotion", "unknown"))

    # polarity tags (HARD RULE: must be opposite, non-neutral)
    tags.setdefault("expectation_polarity", []).append(polarity(expectation_final))
    tags.setdefault("reality_polarity", []).append(polarity(er.get("reality", "")))

    return normalize_tags(tags)

def meme_score(C: Dict[str, Any], expectation_final: str, reality_text: str, er: Dict[str, Any]) -> float:
    base = 1.0

    if C.get("time_of_the_day") in ("evening", "night", "late_night"):
        base *= 1.10

    L = len((expectation_final or "").strip())
    if 20 <= L <= 85:
        base *= 1.10
    elif L > 140:
        base *= 0.90

    emo = (er.get("emotion") or "").lower()
    if any(x in emo for x in ["frustr", "stress", "irony", "panic", "chaos"]):
        base *= 1.05

    return round(base, 2)

def make_draft_id(context_id: str, er_id: int, combo_text: str) -> str:
    raw = f"{context_id}||{er_id}||{combo_text}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def save_meme_text_to_file(draft_id: str, expectation_text: str) -> None:
    """Save expectation^context text to a file for later use in meme generation."""

    path = os.path.join(MEME_TEXT_DIR, f"{draft_id}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write((expectation_text or "").strip() + "\n")

def create_draft(manifest: Dict[str, Any]) -> Dict[str, Any]:
    """
    HARD RULE: Expectation must always be opposite to Reality.
    We retry ER selection until:
      - expectation polarity is non-neutral
      - reality polarity is non-neutral
      - they are opposite
    """
    C = get_random_context(manifest)

    last_er = None
    for _ in range(MAX_ER_PICK_TRIES):
        er = pick_random_er_sql()

        # ✅ CHANGE (minimal): use mode-switch generator

        expectation_final = generate_expectation(C, er)

        if opposite_ok(expectation_final, er.get("reality", "")):
            combo = build_combo_text(C, er, expectation_final)
            tags = tags_from(C, er, expectation_final)
            draft = {
                "draft_id": make_draft_id(C["id"], er["id"], combo),
                "context_id": C["id"],
                "er_id": er["id"],
                "schema_key": schema_key_from_context(C),
                "expectation_text": expectation_final,
                "reality_text": er["reality"],
                "combo_text": combo,
                "tags_json": json.dumps(tags, ensure_ascii=False),
                "score": meme_score(C, expectation_final, er["reality"], er),
                "created_at": int(time.time())
            }
            save_meme_text_to_file(draft["draft_id"], draft["expectation_text"])
            return draft
        last_er = er

    raise RuntimeError(
        f"Could not find an opposite Expectation/Reality pair after {MAX_ER_PICK_TRIES} tries. "
        f"Last ER tried id={last_er.get('id') if last_er else 'unknown'} "
        f"(polarity exp={polarity(last_er.get('expectation','') if last_er else '')}, "
        f"real={polarity(last_er.get('reality','') if last_er else '')}). "
        f"Consider strengthening POSITIVE_HINTS/NEGATIVE_HINTS or adding explicit polarity tags in ER rows."
    )

def insert_draft(d: Dict[str, Any]) -> bool:
    conn = db_connect(DB_PATH)
    cur = conn.cursor()
    try:
        cur.execute("""
            INSERT INTO drafts(draft_id, context_id, er_id, schema_key,
                               expectation_text, reality_text, combo_text,
                               tags_json, score, created_at)
            VALUES(?,?,?,?,?,?,?,?,?,?)
        """, (
            d["draft_id"], d["context_id"], d["er_id"], d["schema_key"],
            d["expectation_text"], d["reality_text"], d["combo_text"],
            d["tags_json"], d["score"], d["created_at"]
        ))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

# =========================
# Media selection (Reality asset suggestion) + HARD "opposite"
# =========================

def load_media_catalog() -> Dict[str, Any]:
    if not os.path.exists(MEDIA_CATALOG_PATH):
        return {"version": "1.0", "assets": []}
    return load_json(MEDIA_CATALOG_PATH)

def suggest_asset_for_draft(
    draft_tags: Dict[str, List[str]],
    expectation_text: str,
    reality_text: str,
    er_emotion: str,
    catalog: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    keys = coverage_keys_profile_1(draft_tags)

    exp_pol = polarity(expectation_text)
    real_pol = polarity(reality_text)

    if exp_pol == "neutral" or real_pol == "neutral":
        return None

    if exp_pol == real_pol:
        return None