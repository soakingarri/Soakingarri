#!/usr/bin/env python3

"""
SoakinGarri Pipeline Orchestrator

PURPOSE
-------
Runs every data stage in order, and only redoes work whose inputs changed:

    1. cleaning            raw_data              -> cleaned_data
    2. deduplication       cleaned_data          -> deduplicated_data
    3. language_detection  deduplicated_data     -> language_tagged_data
    4. chunking            language_tagged_data  -> chunked_data
    5. embedding           chunked_data          -> embedding_data
    6. vector_base         embedding_data + chunked_data -> vector_base
    7. keyword_index       embedding_data + chunked_data -> keyword_index

WHAT GETS REPROCESSED
---------------------
Cleaning (the slow, OCR-heavy stage) works file by file: a raw file is
cleaned again only when its bytes change, or when its filter fields change
(a new year in the source registry, a new fetch time for a web page). A
deleted raw file has its cleaned output deleted too.

Every later stage looks across the whole corpus (deduplication compares
documents with each other; the indexes hold everything), so each one runs
as a unit — but only when a fingerprint of its input files differs from
the last successful run. A cleaning change that leaves the deduplicated
text identical therefore stops there: nothing downstream reruns.

If a stage fails, the run stops, so later stages never build on stale or
half-written inputs, and the failed stage reruns next time.

STATE
-----
<repo>/data/ask/metadata/pipeline_state.json  (fingerprints, per-file
hashes, last results)

RUN
---
python3 pipeline_orchestrator.py                 (run what changed)
python3 pipeline_orchestrator.py --dry-run       (only show what would run)
python3 pipeline_orchestrator.py --from chunking (force chunking onward)
python3 pipeline_orchestrator.py --force         (rerun everything)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

SOURCE_FOLDER = Path(__file__).resolve().parent.parent
DATA_ASK_FOLDER = SOURCE_FOLDER.parent / "data" / "ask"

for folder in (
    SOURCE_FOLDER,
    SOURCE_FOLDER / "data_cleaning_program",
    SOURCE_FOLDER / "deduplication_program",
    SOURCE_FOLDER / "language_detection_program",
    SOURCE_FOLDER / "data_chunking_programs",
    SOURCE_FOLDER / "embedding_program",
    DATA_ASK_FOLDER / "vector_database",
    DATA_ASK_FOLDER / "keyword_index",
):
    sys.path.insert(0, str(folder))

from common import DATA_ROOT  # noqa: E402

STATE_FILE_NAME = "pipeline_state.json"

HASH_BLOCK_SIZE = 1 << 20


@dataclass(slots=True)
class Stage:
    """
    PURPOSE
    -------
    One pipeline stage.

    ATTRIBUTES
    ----------
    name      Stage name, used in the state file and on the command line.
    inputs    Folders whose files decide whether the stage must rerun.
    output    The folder the stage writes.
    run       Callable that runs the stage and returns its manifest dict.
    """

    name: str
    inputs: list[Path]
    output: Path
    run: Callable[[], dict[str, Any]] = field(repr=False)


def file_sha256(path: Path) -> str:
    r"""
    PURPOSE
    -------
    The sha256 of a file's bytes, read in blocks so large PDFs are fine.

    PARAMETERS
    ----------
    path   The file.

    RETURNS
    -------
    The hex digest.

    EXAMPLES
    --------
    >>> import tempfile
    >>> with tempfile.TemporaryDirectory() as folder:
    ...     path = Path(folder) / "a.txt"
    ...     _ = path.write_bytes(b"abc")
    ...     file_sha256(path)[:12]
    'ba7816bf8f01'
    """
    digest = hashlib.sha256()

    with path.open("rb") as input_file:
        for block in iter(lambda: input_file.read(HASH_BLOCK_SIZE), b""):
            digest.update(block)

    return digest.hexdigest()


def is_tracked(path: Path) -> bool:
    r"""
    PURPOSE
    -------
    Whether a file counts toward a folder's fingerprint. Manifests (which
    carry timestamps), temporary files, hidden files, caches and code are
    ignored, so they never trigger a rerun on their own.

    PARAMETERS
    ----------
    path   The file.

    RETURNS
    -------
    True for real data files.

    EXAMPLES
    --------
    >>> [is_tracked(Path(name)) for name in ("book_cleaned.json", "cleaning_manifest.json",
    ...                                        "x.json.tmp", ".DS_Store", "keyword_index.py")]
    [True, False, False, False, False]
    >>> is_tracked(Path("data/__pycache__/a.pyc"))
    False
    """
    name = path.name

    return not (
        name.startswith(".")
        or name.endswith(".tmp")
        or path.stem.endswith("_manifest")
        or path.suffix in {".py", ".pyc"}
        or "__pycache__" in path.parts
    )


def folder_fingerprint(folders: list[Path]) -> str:
    r"""
    PURPOSE
    -------
    One hash summarising every tracked file in some folders: their paths
    (relative to their folder) and contents. Any added, removed or changed
    data file changes it; a missing folder counts as empty.

    PARAMETERS
    ----------
    folders   The folders to fingerprint.

    RETURNS
    -------
    The hex digest.

    EXAMPLES
    --------
    >>> import tempfile
    >>> with tempfile.TemporaryDirectory() as folder:
    ...     root = Path(folder)
    ...     _ = (root / "a.json").write_text("1")
    ...     first = folder_fingerprint([root])
    ...     _ = (root / "run_manifest.json").write_text("timestamp")
    ...     same = folder_fingerprint([root])
    ...     _ = (root / "a.json").write_text("2")
    ...     changed = folder_fingerprint([root])
    >>> first == same, first == changed
    (True, False)
    >>> folder_fingerprint([Path("/no/such/folder")]) == folder_fingerprint([])
    False
    """
    digest = hashlib.sha256()

    for index, folder in enumerate(folders):
        digest.update(f"\x1e{index}".encode())

        if not folder.is_dir():
            continue

        for path in sorted(p for p in folder.rglob("*") if p.is_file() and is_tracked(p)):
            digest.update(str(path.relative_to(folder)).encode("utf-8"))
            digest.update(b"\x1f")
            digest.update(file_sha256(path).encode())

    return digest.hexdigest()


def load_state(path: Path) -> dict[str, Any]:
    r"""
    PURPOSE
    -------
    Read the orchestrator state; missing or unreadable means a fresh start.

    PARAMETERS
    ----------
    path   The pipeline_state.json file.

    RETURNS
    -------
    The state dict, with "stages" and "cleaning_files" keys.

    EXAMPLES
    --------
    >>> load_state(Path("/no/such/state.json"))
    {'stages': {}, 'cleaning_files': {}}
    """
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        state = {}

    state.setdefault("stages", {})
    state.setdefault("cleaning_files", {})

    return state


def save_state(path: Path, state: dict[str, Any]) -> None:
    r"""
    PURPOSE
    -------
    Write the state atomically, after every stage, so an interrupted run
    keeps the progress it made.

    PARAMETERS
    ----------
    path    The pipeline_state.json file.
    state   The state dict.

    EXAMPLES
    --------
    >>> import tempfile
    >>> with tempfile.TemporaryDirectory() as folder:
    ...     path = Path(folder) / "meta" / STATE_FILE_NAME
    ...     save_state(path, {"stages": {"x": {"status": "success"}}, "cleaning_files": {}})
    ...     load_state(path)["stages"]["x"]["status"]
    'success'
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(".json.tmp")
    temporary_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary_path.replace(path)


def has_output(folder: Path) -> bool:
    r"""
    PURPOSE
    -------
    Whether a stage's output folder holds any tracked file (a manifest
    alone does not count). A stage whose output was deleted must rerun even
    if its inputs did not change.

    PARAMETERS
    ----------
    folder   The output folder.

    RETURNS
    -------
    True when there is real output.

    EXAMPLES
    --------
    >>> has_output(Path("/no/such/folder"))
    False
    """
    return folder.is_dir() and any(p.is_file() and is_tracked(p) for p in folder.rglob("*"))


def clean_changed_files(
    raw_folder: Path,
    cleaned_folder: Path,
    state: dict[str, Any],
    force: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    r"""
    PURPOSE
    -------
    The incremental cleaning stage. A raw file is (re)cleaned when it is
    new, its bytes changed, its filter fields (tool / source_type / date)
    changed, or its cleaned output is missing. Cleaned output of raw files
    that no longer exist is deleted. Failures are recorded per file and do
    not stop the other files.

    PARAMETERS
    ----------
    raw_folder       The raw_data folder.
    cleaned_folder   The cleaned_data folder.
    state            Orchestrator state; its "cleaning_files" entry is
                      updated in place.
    force            Re-clean every file.
    dry_run          Report what would happen without doing it.

    RETURNS
    -------
    {"cleaned": [...], "unchanged": n, "failed": [...], "removed": [...]}.

    EXAMPLES
    --------
    >>> import tempfile
    >>> with tempfile.TemporaryDirectory() as folder:
    ...     root = Path(folder)
    ...     raw, cleaned = root / "raw_data", root / "cleaned_data"
    ...     raw.mkdir()
    ...     book = raw / "book.txt"
    ...     _ = book.write_text("Book\nChapter 1: Start\nA paragraph long enough to keep here.\n")
    ...     state = {"stages": {}, "cleaning_files": {}}
    ...     first = clean_changed_files(raw, cleaned, state)
    ...     second = clean_changed_files(raw, cleaned, state)
    ...     book.unlink()
    ...     third = clean_changed_files(raw, cleaned, state)
    ...     left = sorted(p.name for p in cleaned.iterdir())
    >>> first["cleaned"], second["unchanged"], third["removed"], left
    (['book.txt'], 1, ['book.txt'], [])
    """
    import data_cleaning as dc

    known: dict[str, dict[str, Any]] = state["cleaning_files"]
    report: dict[str, Any] = {"cleaned": [], "unchanged": 0, "failed": [], "removed": []}
    present: set[str] = set()

    books = dc.discover_books(raw_folder) if raw_folder.is_dir() else []

    for path in books:
        key = str(path.relative_to(raw_folder))
        present.add(key)

        signature = hashlib.sha256(
            (file_sha256(path) + json.dumps(dc.source_attributes(path), sort_keys=True)).encode()
        ).hexdigest()
        previous = known.get(key) or {}
        output = previous.get("output")

        if (
            not force
            and previous.get("signature") == signature
            and previous.get("status") == "success"
            and output
            and Path(output).exists()
        ):
            report["unchanged"] += 1
            continue

        if dry_run:
            report["cleaned"].append(key)
            continue

        result = dc.process_file(str(path), raw_folder, cleaned_folder)

        if result.get("status") == "success":
            known[key] = {"signature": signature, "status": "success", "output": result["output"]}
            report["cleaned"].append(key)
        else:
            known[key] = {"signature": signature, "status": "failed", "error": result.get("error")}
            report["failed"].append({"file": key, "error": result.get("error")})

    for key in sorted(set(known) - present):
        if dry_run:
            report["removed"].append(key)
            continue

        output = (known.pop(key) or {}).get("output")

        if output and Path(output).exists():
            Path(output).unlink()

        report["removed"].append(key)

    return report


def build_stages(
    data_root: Path = DATA_ROOT,
    language_model: Any = None,
    embedding_model: Any = None,
) -> list[Stage]:
    r"""
    PURPOSE
    -------
    The corpus-wide stages after cleaning, wired to the real modules with
    explicit folders under `data_root`. Each module is imported only when
    its stage actually runs, so planning a run never loads torch or a model.

    PARAMETERS
    ----------
    data_root         The tool's data folder (data/ask by default).
    language_model    GlotLID stand-in for tests; None loads GlotLID.
    embedding_model   bge-m3 stand-in for tests; None loads bge-m3.

    RETURNS
    -------
    The stages, in run order.

    EXAMPLES
    --------
    >>> [stage.name for stage in build_stages(Path("/tmp/x"))]
    ['deduplication', 'language_detection', 'chunking', 'embedding', 'vector_base', 'keyword_index']
    """
    cleaned = data_root / "cleaned_data"
    deduplicated = data_root / "deduplicated_data"
    tagged = data_root / "language_tagged_data"
    chunked = data_root / "chunked_data"
    embedded = data_root / "embedding_data"
    vector_base = data_root / "vector_base"
    keyword_index = data_root / "keyword_index"

    def deduplicate() -> dict[str, Any]:
        import deduplication

        return deduplication.deduplicate_corpus(cleaned, deduplicated)

    def tag_languages() -> dict[str, Any]:
        import language_detection

        return language_detection.tag_corpus(deduplicated, tagged, model=language_model)

    def chunk() -> dict[str, Any]:
        import data_chunking

        return data_chunking.run_pipeline(tagged, chunked)

    def embed() -> dict[str, Any]:
        import embedding

        return embedding.run_pipeline(chunked, embedded, model=embedding_model, verbose=False)

    def build_vectors() -> dict[str, Any]:
        import vector_base as vb

        return vb.build_vector_base(embedded, chunked, vector_base, verbose=False)

    def build_keywords() -> dict[str, Any]:
        import keyword_index as ki

        return ki.build_keyword_index(embedded, chunked, keyword_index, verbose=False)

    return [
        Stage("deduplication", [cleaned], deduplicated, deduplicate),
        Stage("language_detection", [deduplicated], tagged, tag_languages),
        Stage("chunking", [tagged], chunked, chunk),
        Stage("embedding", [chunked], embedded, embed),
        Stage("vector_base", [embedded, chunked], vector_base, build_vectors),
        Stage("keyword_index", [embedded, chunked], keyword_index, build_keywords),
    ]


def summarize_manifest(manifest: Any) -> dict[str, Any]:
    r"""
    PURPOSE
    -------
    Keep the small numeric/text fields of a stage manifest for the state
    file (big lists such as every removed paragraph are dropped).

    PARAMETERS
    ----------
    manifest   A stage's returned manifest.

    RETURNS
    -------
    The trimmed dict.

    EXAMPLES
    --------
    >>> summarize_manifest({"chunk_count": 4, "removed": [1, 2], "output_folder": "/x"})
    {'chunk_count': 4, 'output_folder': '/x'}
    """
    if not isinstance(manifest, dict):
        return {}

    return {
        key: value
        for key, value in manifest.items()
        if isinstance(value, (int, float, str, bool)) or value is None
    }


def run_pipeline(
    data_root: Path = DATA_ROOT,
    stages: list[Stage] | None = None,
    force: bool = False,
    force_from: str | None = None,
    dry_run: bool = False,
    state_file: Path | None = None,
    output_function: Callable[[str], None] = print,
) -> dict[str, Any]:
    r"""
    PURPOSE
    -------
    Run cleaning incrementally, then every later stage whose inputs
    changed, in order, stopping at the first failure.

    PARAMETERS
    ----------
    data_root         The tool's data folder.
    stages            The stages after cleaning; None uses build_stages.
    force             Rerun everything.
    force_from        Rerun this stage and every stage after it.
    dry_run           Only report what would run.
    state_file        State location; defaults to
                       data_root/metadata/pipeline_state.json.
    output_function   Where progress lines go.

    RETURNS
    -------
    {"cleaning": {...}, "stages": {name: {"status", ...}}, "ok": bool}.
    Stage statuses are "success", "skipped", "would_run" or "failed".

    RAISES
    ------
    ValueError   When force_from names no stage.

    EXAMPLES
    --------
    >>> import tempfile
    >>> calls = []
    >>> with tempfile.TemporaryDirectory() as folder:
    ...     root = Path(folder)
    ...     (root / "raw_data").mkdir()
    ...     def make(name, source, target):
    ...         def run():
    ...             calls.append(name)
    ...             (root / target).mkdir(exist_ok=True)
    ...             (root / target / "out.json").write_text(name)
    ...             return {"files": 1}
    ...         return Stage(name, [root / source], root / target, run)
    ...     stages = [make("a", "cleaned_data", "a_out"), make("b", "a_out", "b_out")]
    ...     first = run_pipeline(root, stages, output_function=lambda line: None)
    ...     second = run_pipeline(root, stages, output_function=lambda line: None)
    ...     third = run_pipeline(root, stages, force_from="b", output_function=lambda line: None)
    >>> calls
    ['a', 'b', 'b']
    >>> [second["stages"][name]["status"] for name in ("a", "b")]
    ['skipped', 'skipped']
    """
    stages = build_stages(data_root) if stages is None else stages
    names = [stage.name for stage in stages]

    if force_from is not None and force_from not in names + ["cleaning"]:
        raise ValueError(f"Unknown stage {force_from!r}; choose from {['cleaning', *names]}.")

    state_path = state_file or data_root / "metadata" / STATE_FILE_NAME
    state = load_state(state_path)
    forced = force or force_from == "cleaning"
    report: dict[str, Any] = {"stages": {}, "ok": True}

    started = time.perf_counter()
    cleaning = clean_changed_files(
        data_root / "raw_data", data_root / "cleaned_data", state, force=forced, dry_run=dry_run
    )
    report["cleaning"] = cleaning
    verb = "would clean" if dry_run else "cleaned"
    output_function(
        f"cleaning            {len(cleaning['cleaned'])} {verb}, {cleaning['unchanged']} unchanged, "
        f"{len(cleaning['removed'])} removed, {len(cleaning['failed'])} failed "
        f"({time.perf_counter() - started:.1f}s)"
    )

    if not dry_run:
        save_state(state_path, state)

    for stage in stages:
        forced = forced or stage.name == force_from
        fingerprint = folder_fingerprint(stage.inputs)
        previous = state["stages"].get(stage.name) or {}

        unchanged = (
            previous.get("status") == "success"
            and previous.get("input_fingerprint") == fingerprint
            and has_output(stage.output)
        )

        if unchanged and not forced:
            report["stages"][stage.name] = {"status": "skipped"}
            output_function(f"{stage.name:<20}skipped (inputs unchanged)")
            continue

        # Once one stage reruns, everything after it must rerun too, even
        # when this run happens to reproduce identical output.
        forced = True

        if dry_run:
            report["stages"][stage.name] = {"status": "would_run"}
            output_function(f"{stage.name:<20}would run")
            continue

        started = time.perf_counter()

        try:
            manifest = stage.run()
        except Exception as error:  # record it and stop: later stages need this output
            state["stages"][stage.name] = {
                "status": "failed",
                "error": f"{type(error).__name__}: {error}",
                "finished_at": datetime.now(timezone.utc).isoformat(),
            }
            save_state(state_path, state)
            report["stages"][stage.name] = state["stages"][stage.name]
            report["ok"] = False
            output_function(f"{stage.name:<20}FAILED: {type(error).__name__}: {error}")
            break

        seconds = round(time.perf_counter() - started, 2)
        state["stages"][stage.name] = {
            "status": "success",
            "input_fingerprint": fingerprint,
            "seconds": seconds,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "manifest": summarize_manifest(manifest),
        }
        save_state(state_path, state)
        report["stages"][stage.name] = {"status": "success", "seconds": seconds}
        output_function(f"{stage.name:<20}done ({seconds:.1f}s)")

    if cleaning["failed"]:
        report["ok"] = False

    return report


def parse_arguments(argument_list: list[str] | None = None) -> argparse.Namespace:
    r"""
    PURPOSE
    -------
    Parse the command line.

    PARAMETERS
    ----------
    argument_list   Arguments to parse; None reads from sys.argv.

    RETURNS
    -------
    The parsed arguments namespace.

    EXAMPLES
    --------
    >>> options = parse_arguments(["--dry-run", "--from", "chunking"])
    >>> options.dry_run, options.force_from, options.force
    (True, 'chunking', False)
    """
    parser = argparse.ArgumentParser(
        description="Run the data pipeline, reprocessing only what changed."
    )
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT, help="tool data folder")
    parser.add_argument("--force", action="store_true", help="rerun every stage")
    parser.add_argument("--from", dest="force_from", help="rerun this stage and all after it")
    parser.add_argument("--dry-run", action="store_true", help="show what would run")

    return parser.parse_args(argument_list)


def main() -> None:
    """
    PURPOSE
    -------
    Run the pipeline with the options given on the command line, and exit
    with an error status when anything failed.
    """
    options = parse_arguments()

    report = run_pipeline(
        data_root=options.data_root,
        force=options.force,
        force_from=options.force_from,
        dry_run=options.dry_run,
    )

    if not report["ok"]:
        raise SystemExit("The pipeline did not finish cleanly; see the lines above.")


if __name__ == "__main__":
    main()
