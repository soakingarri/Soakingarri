#!/usr/bin/env python3

"""
SoakinGarri Language Detection Program

PURPOSE
-------
Tag every paragraph of every cleaned document with its language, using
GlotLID (a fastText language identifier covering 2000+ languages,
including Yoruba, Igbo, Hausa, Swahili, Twi, Wolof, Amharic and Nigerian
Pidgin). This runs after deduplication.py and before data_chunking.py, so
the language label travels with each paragraph into the chunks, the vector
base and the keyword index, where retrieval.py can filter on it
(--language yor_Latn).

Labels are GlotLID's ISO 639-3 code plus script, for example "eng_Latn",
"yor_Latn", "hau_Latn", "amh_Ethi".

SHORT OR UNCERTAIN PARAGRAPHS
-----------------------------
Language identification is unreliable on a heading or a formula. A
paragraph shorter than MIN_DETECTION_CHARACTERS, or one GlotLID scores
below MIN_CONFIDENCE, inherits its document's language instead and is
marked "language_source": "document". Every other paragraph is marked
"language_source": "glotlid". A document's language is the label covering
the most characters among its confidently detected paragraphs.

Input:
    <repo>/data/ask/deduplicated_data

Output:
    <repo>/data/ask/language_tagged_data

Install dependencies:
    python3 -m pip install fasttext huggingface_hub

Run:
    python3 language_detection.py
    python3 language_detection.py --input /path/to/cleaned --output /path/to/out
    python3 language_detection.py --min-confidence 0.6

Notes:
    The GlotLID model (~1.7 GB) is downloaded from the Hugging Face hub
    (cis-lmu/glotlid) on first run and cached afterwards. Output files keep
    the input structure and file names; only "language" fields are added.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Protocol

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common import DATA_ROOT, load_documents, normalize_text  # noqa: E402

INPUT_FOLDER = DATA_ROOT / "deduplicated_data"
OUTPUT_FOLDER = DATA_ROOT / "language_tagged_data"

MANIFEST_FILE_NAME = "language_detection_manifest.json"

GLOTLID_REPOSITORY = "cis-lmu/glotlid"
GLOTLID_FILE_NAME = "model.bin"

MIN_DETECTION_CHARACTERS = 20
MIN_CONFIDENCE = 0.5

UNDETERMINED = "und"

LABEL_PREFIX = "__label__"

DEPENDENCY_MESSAGE = (
    "Language detection requires fasttext and huggingface_hub: "
    "python3 -m pip install fasttext huggingface_hub"
)


class LanguageModel(Protocol):
    """
    PURPOSE
    -------
    Define the interface a language identifier must implement. A loaded
    fastText model satisfies it; tests pass a small stub instead, so they
    never download GlotLID.
    """

    def predict(self, text: str, k: int = 1) -> tuple[Any, Any]:
        """
        PURPOSE
        -------
        Return the top-k (labels, probabilities) for one line of text, with
        labels written as "__label__<code>".
        """
        ...


def load_language_model() -> LanguageModel:
    r"""
    PURPOSE
    -------
    Download (first run only) and load the GlotLID fastText model.

    RETURNS
    -------
    The loaded fastText model.

    RAISES
    ------
    RuntimeError   When fasttext or huggingface_hub is not installed; the
                   message includes the pip install command.
    """
    try:
        import fasttext
        from huggingface_hub import hf_hub_download
    except ImportError as error:
        raise RuntimeError(DEPENDENCY_MESSAGE) from error

    model_path = hf_hub_download(
        repo_id=GLOTLID_REPOSITORY, filename=GLOTLID_FILE_NAME
    )

    return fasttext.load_model(model_path)


def detect_language(
    text: str,
    model: LanguageModel,
    min_characters: int = MIN_DETECTION_CHARACTERS,
    min_confidence: float = MIN_CONFIDENCE,
) -> tuple[str, float]:
    r"""
    PURPOSE
    -------
    Identify the language of one passage. fastText predicts one line at a
    time, so line breaks are collapsed to spaces first.

    PARAMETERS
    ----------
    text             The passage to identify.
    model            A loaded language model (see LanguageModel).
    min_characters   Passages shorter than this, after normalising, are
                      not sent to the model.
    min_confidence   Predictions scoring below this are not trusted.

    RETURNS
    -------
    A (label, confidence) tuple, for example ("yor_Latn", 0.97). The label
    is "und" when the passage is too short or the prediction is too
    uncertain; its confidence is then 0.0 or the low score respectively.

    EXAMPLES
    --------
    >>> class StubModel:
    ...     def predict(self, text, k=1):
    ...         if "\n" in text:
    ...             raise ValueError("predict processes one line at a time")
    ...         if "ni" in text.split():
    ...             return (("__label__yor_Latn",), (0.97,))
    ...         return (("__label__eng_Latn",), (0.42,))
    >>> detect_language("Bawo ni ara re\nE kaaro o.", StubModel())
    ('yor_Latn', 0.97)
    >>> detect_language("Too short", StubModel())
    ('und', 0.0)
    >>> detect_language("An uncertain English sentence here.", StubModel())
    ('und', 0.42)
    >>> detect_language("An uncertain English sentence here.", StubModel(), min_confidence=0.4)
    ('eng_Latn', 0.42)
    """
    line = normalize_text(text).replace("\n", " ")

    if len(line) < min_characters:
        return UNDETERMINED, 0.0

    labels, scores = model.predict(line, k=1)

    label = str(labels[0]).removeprefix(LABEL_PREFIX)
    confidence = round(float(scores[0]), 4)

    if confidence < min_confidence:
        return UNDETERMINED, confidence

    return label, confidence


def dominant_language(
    detections: list[tuple[str, int]],
) -> str:
    r"""
    PURPOSE
    -------
    Pick a document's language: the label covering the most characters
    among confidently detected paragraphs. Ties go to the label seen first.

    PARAMETERS
    ----------
    detections   (label, character count) pairs, one per paragraph.

    RETURNS
    -------
    The dominant label, or "und" when no paragraph was confidently
    detected.

    EXAMPLES
    --------
    >>> dominant_language([("eng_Latn", 400), ("yor_Latn", 120), ("eng_Latn", 50)])
    'eng_Latn'
    >>> dominant_language([("und", 900), ("yor_Latn", 30)])
    'yor_Latn'
    >>> dominant_language([("hau_Latn", 100), ("yor_Latn", 100)])
    'hau_Latn'
    >>> dominant_language([("und", 10)])
    'und'
    >>> dominant_language([])
    'und'
    """
    totals: Counter[str] = Counter()

    for label, characters in detections:
        if label != UNDETERMINED:
            totals[label] += characters

    if not totals:
        return UNDETERMINED

    return max(totals, key=lambda label: totals[label])


def paragraph_text(paragraph: Any) -> str:
    r"""
    PURPOSE
    -------
    Read a paragraph's text whether it is a {"text": ...} dict or a plain
    string, as data_chunking.py accepts both.

    PARAMETERS
    ----------
    paragraph   A paragraph entry from a cleaned document.

    RETURNS
    -------
    The paragraph's text, or '' when it has none.

    EXAMPLES
    --------
    >>> paragraph_text({"paragraph_id": "P-1", "text": "Hello."})
    'Hello.'
    >>> paragraph_text("Plain string paragraph.")
    'Plain string paragraph.'
    >>> paragraph_text({"paragraph_id": "P-2"})
    ''
    """
    if isinstance(paragraph, dict):
        return str(paragraph.get("text") or "")

    return str(paragraph or "")


def tag_document(
    document: dict[str, Any],
    model: LanguageModel,
    min_confidence: float = MIN_CONFIDENCE,
) -> dict[str, Any]:
    r"""
    PURPOSE
    -------
    Return a copy of a cleaned document with a language on every
    paragraph, every chapter and the document itself. Plain-string
    paragraphs are turned into {"text": ...} dicts so they can carry one.
    A flat {"text": ...} document (no chapters) is tagged as a whole.

    PARAMETERS
    ----------
    document         A cleaned document dict (data_cleaning.py output).
    model            A loaded language model.
    min_confidence   See detect_language.

    RETURNS
    -------
    A new document dict; the input is not modified. Paragraphs gain
    "language", "language_confidence" and "language_source"; chapters and
    the document gain "language".

    EXAMPLES
    --------
    >>> class StubModel:
    ...     def predict(self, text, k=1):
    ...         if "ni" in text.split():
    ...             return (("__label__yor_Latn",), (0.95,))
    ...         return (("__label__eng_Latn",), (0.9,))
    >>> document = {"document_id": "DOC-1", "chapters": [{"title": "Ori", "paragraphs": [
    ...     {"paragraph_id": "P-1", "text": "Ori ni ipin eniyan ninu asa Yoruba."},
    ...     {"paragraph_id": "P-2", "text": "Ori ni o ga julo."},
    ...     "A plain English gloss.",
    ... ]}]}
    >>> tagged = tag_document(document, StubModel())
    >>> [(p["language"], p["language_source"]) for p in tagged["chapters"][0]["paragraphs"]]
    [('yor_Latn', 'glotlid'), ('yor_Latn', 'document'), ('eng_Latn', 'glotlid')]
    >>> tagged["chapters"][0]["language"], tagged["language"]
    ('yor_Latn', 'yor_Latn')
    >>> "language" in document
    False
    >>> tag_document({"text": "Ori ni ipin eniyan ninu asa Yoruba."}, StubModel())["language"]
    'yor_Latn'
    """
    tagged = dict(document)

    if "chapters" not in document:
        label, confidence = detect_language(
            str(document.get("text") or ""), model, min_confidence=min_confidence
        )
        tagged["language"] = label
        tagged["language_confidence"] = confidence
        return tagged

    chapters: list[dict[str, Any]] = []
    document_detections: list[tuple[str, int]] = []

    for chapter in document.get("chapters") or []:
        paragraphs: list[dict[str, Any]] = []
        chapter_detections: list[tuple[str, int]] = []

        for paragraph in chapter.get("paragraphs") or []:
            text = paragraph_text(paragraph)
            new_paragraph = (
                dict(paragraph) if isinstance(paragraph, dict) else {"text": text}
            )
            label, confidence = detect_language(
                text, model, min_confidence=min_confidence
            )
            new_paragraph["language"] = label
            new_paragraph["language_confidence"] = confidence
            new_paragraph["language_source"] = "glotlid"
            paragraphs.append(new_paragraph)
            chapter_detections.append((label, len(text)))

        new_chapter = dict(chapter)
        new_chapter["paragraphs"] = paragraphs
        new_chapter["language"] = dominant_language(chapter_detections)
        chapters.append(new_chapter)
        document_detections.extend(chapter_detections)

    document_language = dominant_language(document_detections)

    # Second pass: undetermined paragraphs inherit their chapter's language,
    # falling back to the document's when the chapter has none either.
    for chapter in chapters:
        if chapter["language"] == UNDETERMINED:
            chapter["language"] = document_language

        for paragraph in chapter["paragraphs"]:
            if paragraph["language"] == UNDETERMINED:
                paragraph["language"] = chapter["language"]
                paragraph["language_source"] = "document"

    tagged["chapters"] = chapters
    tagged["language"] = document_language

    return tagged


def discover_input_files(input_folder: Path) -> list[Path]:
    r"""
    PURPOSE
    -------
    List cleaned (or deduplicated) document files in a folder, skipping
    manifests.

    PARAMETERS
    ----------
    input_folder   Root folder to search.

    RETURNS
    -------
    The sorted list of "*_cleaned.json" / "*_cleaned.jsonl" paths found.

    EXAMPLES
    --------
    >>> import tempfile
    >>> with tempfile.TemporaryDirectory() as folder:
    ...     root = Path(folder)
    ...     _ = (root / "book_cleaned.json").write_text("{}", encoding="utf-8")
    ...     _ = (root / "deduplication_manifest.json").write_text("{}", encoding="utf-8")
    ...     [p.name for p in discover_input_files(root)]
    ['book_cleaned.json']
    """
    return sorted(
        path
        for pattern in ("*_cleaned.json", "*_cleaned.jsonl")
        for path in input_folder.rglob(pattern)
        if path.is_file()
    )


def write_documents(output_path: Path, documents: list[dict[str, Any]]) -> None:
    r"""
    PURPOSE
    -------
    Atomically write documents in the same format as their input file: one
    JSON object per line for .jsonl, otherwise a single object (or a list
    when there are several).

    PARAMETERS
    ----------
    output_path   Destination file; its suffix picks the format.
    documents     The documents to write.

    EXAMPLES
    --------
    >>> import tempfile
    >>> with tempfile.TemporaryDirectory() as folder:
    ...     path = Path(folder) / "a_cleaned.jsonl"
    ...     write_documents(path, [{"a": 1}, {"b": 2}])
    ...     path.read_text(encoding="utf-8").splitlines()
    ['{"a": 1}', '{"b": 2}']
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")

    with temporary_path.open("w", encoding="utf-8") as output_file:
        if output_path.suffix.casefold() == ".jsonl":
            for document in documents:
                output_file.write(json.dumps(document, ensure_ascii=False) + "\n")
        else:
            payload = documents[0] if len(documents) == 1 else documents
            json.dump(payload, output_file, ensure_ascii=False, indent=2)

    temporary_path.replace(output_path)


def tag_corpus(
    input_folder: Path = INPUT_FOLDER,
    output_folder: Path = OUTPUT_FOLDER,
    model: LanguageModel | None = None,
    min_confidence: float = MIN_CONFIDENCE,
) -> dict[str, Any]:
    r"""
    PURPOSE
    -------
    Run language detection over every document in input_folder, write a
    tagged copy of each file to output_folder, and write a manifest with
    per-language paragraph and character counts.

    PARAMETERS
    ----------
    input_folder     Folder of "*_cleaned.json(l)" files.
    output_folder    Folder for tagged files and the manifest.
    model            A loaded language model; None loads GlotLID.
    min_confidence   See detect_language.

    RETURNS
    -------
    The manifest dict (also written to output_folder/
    "language_detection_manifest.json").

    EXAMPLES
    --------
    >>> import tempfile
    >>> class StubModel:
    ...     def predict(self, text, k=1):
    ...         if "ni" in text.split():
    ...             return (("__label__yor_Latn",), (0.95,))
    ...         return (("__label__eng_Latn",), (0.9,))
    >>> with tempfile.TemporaryDirectory() as folder:
    ...     root = Path(folder)
    ...     inp, out = root / "deduplicated", root / "tagged"
    ...     inp.mkdir()
    ...     book = {"document_id": "DOC-1", "chapters": [{"paragraphs": [
    ...         {"paragraph_id": "P-1", "text": "Ori ni ipin eniyan ninu asa Yoruba."},
    ...         {"paragraph_id": "P-2", "text": "An English gloss of the same idea."}]}]}
    ...     _ = (inp / "book_cleaned.json").write_text(json.dumps(book), encoding="utf-8")
    ...     manifest = tag_corpus(inp, out, model=StubModel())
    ...     written = json.loads((out / "book_cleaned.json").read_text(encoding="utf-8"))
    >>> manifest["document_count"], manifest["paragraph_count"], manifest["paragraphs_by_language"]
    (1, 2, {'eng_Latn': 1, 'yor_Latn': 1})
    >>> [p["language"] for p in written["chapters"][0]["paragraphs"]]
    ['yor_Latn', 'eng_Latn']
    """
    if model is None:
        model = load_language_model()

    files = discover_input_files(input_folder)

    paragraphs_by_language: Counter[str] = Counter()
    characters_by_language: Counter[str] = Counter()
    documents_by_language: Counter[str] = Counter()
    inherited = 0
    document_count = 0
    failed_files: list[dict[str, str]] = []

    for path in files:
        try:
            documents = load_documents(path)
        except (OSError, ValueError) as error:
            failed_files.append({"file": str(path), "error": str(error)})
            continue

        tagged_documents = [
            tag_document(document, model, min_confidence=min_confidence)
            for document in documents
        ]

        for document in tagged_documents:
            document_count += 1
            documents_by_language[document["language"]] += 1

            for chapter in document.get("chapters") or []:
                for paragraph in chapter["paragraphs"]:
                    paragraphs_by_language[paragraph["language"]] += 1
                    characters_by_language[paragraph["language"]] += len(
                        paragraph_text(paragraph)
                    )
                    inherited += paragraph["language_source"] == "document"

        if tagged_documents:
            write_documents(
                output_folder / path.relative_to(input_folder), tagged_documents
            )

    manifest = {
        "input_folder": str(input_folder),
        "output_folder": str(output_folder),
        "model": f"{GLOTLID_REPOSITORY}/{GLOTLID_FILE_NAME}",
        "min_confidence": min_confidence,
        "file_count": len(files),
        "failed_files": failed_files,
        "document_count": document_count,
        "paragraph_count": sum(paragraphs_by_language.values()),
        "paragraphs_inheriting_language": inherited,
        "documents_by_language": dict(sorted(documents_by_language.items())),
        "paragraphs_by_language": dict(sorted(paragraphs_by_language.items())),
        "characters_by_language": dict(sorted(characters_by_language.items())),
    }

    output_folder.mkdir(parents=True, exist_ok=True)
    manifest_path = output_folder / MANIFEST_FILE_NAME
    temporary_manifest_path = manifest_path.with_suffix(".json.tmp")

    with temporary_manifest_path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)

    temporary_manifest_path.replace(manifest_path)

    return manifest


def parse_arguments(
    argument_list: list[str] | None = None,
) -> argparse.Namespace:
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
    >>> options = parse_arguments([])
    >>> options.input == INPUT_FOLDER, options.output == OUTPUT_FOLDER, options.min_confidence
    (True, True, 0.5)
    >>> parse_arguments(["--min-confidence", "0.7"]).min_confidence
    0.7
    """
    parser = argparse.ArgumentParser(
        description="Tag every paragraph of the cleaned corpus with its language (GlotLID)."
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=INPUT_FOLDER,
        help="folder of cleaned or deduplicated JSON files",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_FOLDER,
        help="folder for language-tagged JSON files",
    )

    parser.add_argument(
        "--min-confidence",
        type=float,
        default=MIN_CONFIDENCE,
        help="predictions below this inherit the chapter/document language",
    )

    return parser.parse_args(argument_list)


def main() -> None:
    """
    PURPOSE
    -------
    Tag the configured folder using the options given on the command line,
    and print a summary.
    """
    options = parse_arguments()

    manifest = tag_corpus(
        input_folder=options.input,
        output_folder=options.output,
        min_confidence=options.min_confidence,
    )

    print(
        "Language detection completed.\n"
        f"Documents: {manifest['document_count']}\n"
        f"Paragraphs: {manifest['paragraph_count']} "
        f"({manifest['paragraphs_inheriting_language']} inherited)\n"
        f"Paragraphs by language: {manifest['paragraphs_by_language']}\n"
        f"Output folder: {manifest['output_folder']}"
    )


if __name__ == "__main__":
    main()
