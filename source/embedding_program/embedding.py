#!/usr/bin/env python3

"""
SoakinGarri Embedding Program

This program converts the text records produced by the chunking program into
dense numerical vectors that can be consumed by neural-network modules,
vector databases, classifiers, clustering algorithms, and semantic search.

Important:
Embeddings are numerical inputs to neural networks. They do not themselves
turn data into perceptrons. A perceptron or neural-network layer learns from
these vectors during training.

INPUT FOLDER
------------
<repo>/data/ask/chunked_data

EXPECTED INPUT FILES
--------------------
llm_training_data.jsonl
rag_data.jsonl

OUTPUT FOLDER
-------------
<repo>/data/ask/embedding_data

OUTPUT FILES
------------
llm_embeddings.npy
llm_embedding_metadata.jsonl
rag_embeddings.npy
rag_embedding_metadata.jsonl
embedding_manifest.json

INSTALL
-------
python3 -m pip install torch sentence-transformers numpy

RUN
---
python3 embedding_program.py
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterator, Protocol

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common import DATA_ROOT, normalize_text, stable_id  # noqa: E402

# torch and sentence-transformers are only needed to LOAD a model. Importing
# them lazily keeps this module importable (and testable with a stub model) on
# machines where they are not installed.
try:
    import torch
except ImportError:
    torch = None

DEPENDENCY_MESSAGE = (
    "Missing dependencies.\n\n"
    "Install them using:\n"
    "python3 -m pip install torch sentence-transformers numpy"
)


class EmbeddingModel(Protocol):
    """
    PURPOSE
    -------
    Define the interface any embedding backend must implement to be used by
    :func:`generate_file_embeddings`. Any object exposing these two methods
    works, which is what lets the tests run against a stub model instead of
    downloading sentence-transformers.
    """

    def get_sentence_embedding_dimension(self) -> int | None:
        """
        PURPOSE
        -------
        Report the dimensionality of the vectors this model produces.

        RETURNS
        -------
        The embedding vector size, or None if the model cannot report it.
        """
        ...

    def encode(self, sentences: list[str], **kwargs: Any):
        """
        PURPOSE
        -------
        Encode a batch of sentences into embedding vectors.

        PARAMETERS
        ----------
        sentences   The texts to embed.
        kwargs      Backend-specific encoding options (e.g. batch_size,
                    normalize_embeddings, convert_to_numpy).

        RETURNS
        -------
        The batch of embedding vectors, one per input sentence.
        """
        ...


INPUT_FOLDER = DATA_ROOT / "chunked_data"

OUTPUT_FOLDER = DATA_ROOT / "embedding_data"

LLM_INPUT_FILE = INPUT_FOLDER / "llm_training_data.jsonl"
RAG_INPUT_FILE = INPUT_FOLDER / "rag_data.jsonl"

LLM_VECTOR_FILE = OUTPUT_FOLDER / "llm_embeddings.npy"
LLM_METADATA_FILE = OUTPUT_FOLDER / "llm_embedding_metadata.jsonl"

RAG_VECTOR_FILE = OUTPUT_FOLDER / "rag_embeddings.npy"
RAG_METADATA_FILE = OUTPUT_FOLDER / "rag_embedding_metadata.jsonl"

MANIFEST_FILE = OUTPUT_FOLDER / "embedding_manifest.json"

SOURCE_LABEL = "Borrowed content and open source"

MODEL_NAME = "BAAI/bge-m3"

BATCH_SIZE_CPU = 128
BATCH_SIZE_MPS = 256
BATCH_SIZE_CUDA = 512

MAX_SEQUENCE_LENGTH = 8192
NORMALIZE_EMBEDDINGS = True

VECTOR_DATA_TYPE = np.float32

SKIP_EXISTING_RAG_EMBEDDINGS = False


def inference_context():
    r"""
    PURPOSE
    -------
    Provide an inference context manager: `torch.inference_mode()` when
    torch is available, otherwise a no-op context manager.

    RETURNS
    -------
    A context manager suitable for wrapping model inference calls.

    EXAMPLES
    --------
    >>> with inference_context():
    ...     1 + 1
    2
    """
    if torch is None:
        return contextlib.nullcontext()

    return torch.inference_mode()


def load_model(device: str, model_name: str = MODEL_NAME) -> EmbeddingModel:
    r"""
    PURPOSE
    -------
    Load the sentence-transformers model onto the given device, configuring
    its maximum sequence length and switching it to eval mode. Tests pass a
    stub model instead of calling this.

    PARAMETERS
    ----------
    device        The torch device to load the model onto (e.g. "cpu",
                  "mps", "cuda").
    model_name    The sentence-transformers model identifier to load.

    RETURNS
    -------
    The loaded embedding model.

    RAISES
    ------
    RuntimeError   When torch or sentence-transformers is not installed;
                   the message includes install instructions.
    """
    if torch is None:
        raise RuntimeError(DEPENDENCY_MESSAGE)

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as error:
        raise RuntimeError(DEPENDENCY_MESSAGE) from error

    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True

    model = SentenceTransformer(model_name, device=device)
    model.max_seq_length = MAX_SEQUENCE_LENGTH
    model.eval()

    return model


def select_device() -> str:
    r"""
    PURPOSE
    -------
    Pick the best available torch device, preferring CUDA, then MPS, then
    CPU. Returns "cpu" when torch is absent.

    RETURNS
    -------
    "cuda", "mps" or "cpu".

    EXAMPLES
    --------
    >>> select_device() in {"cuda", "mps", "cpu"}
    True
    """
    if torch is None:
        return "cpu"

    if torch.cuda.is_available():
        return "cuda"

    if (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        return "mps"

    return "cpu"


def select_batch_size(device: str) -> int:
    r"""
    PURPOSE
    -------
    Pick a batch size tuned for the given device.

    PARAMETERS
    ----------
    device   The torch device name ("cuda", "mps", "cpu", or anything else).

    RETURNS
    -------
    The batch size configured for that device; BATCH_SIZE_CPU for any
    unrecognized device.

    EXAMPLES
    --------
    >>> select_batch_size("cuda"), select_batch_size("mps"), select_batch_size("cpu")
    (512, 256, 128)
    >>> select_batch_size("anything-else")
    128
    """
    if device == "cuda":
        return BATCH_SIZE_CUDA

    if device == "mps":
        return BATCH_SIZE_MPS

    return BATCH_SIZE_CPU


def count_jsonl_records(path: Path) -> int:
    r"""
    PURPOSE
    -------
    Count non-blank lines in a JSONL file.

    PARAMETERS
    ----------
    path   Path to the JSONL file to count.

    RETURNS
    -------
    The number of non-blank lines.

    EXAMPLES
    --------
    >>> import tempfile
    >>> with tempfile.TemporaryDirectory() as folder:
    ...     path = Path(folder) / "a.jsonl"
    ...     _ = path.write_text('{"a": 1}\n\n{"b": 2}\n', encoding="utf-8")
    ...     count_jsonl_records(path)
    2
    """
    count = 0

    with path.open("r", encoding="utf-8") as input_file:
        for line in input_file:
            if line.strip():
                count += 1

    return count


def read_jsonl(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    r"""
    PURPOSE
    -------
    Read a JSONL file, yielding each non-blank line as a parsed JSON object
    paired with its 1-based line number.

    PARAMETERS
    ----------
    path   Path to the JSONL file to read.

    RETURNS
    -------
    An iterator of (line_number, record) tuples.

    RAISES
    ------
    ValueError   When a non-blank line is not valid JSON, or does not
                 decode to a JSON object.

    EXAMPLES
    --------
    >>> import tempfile
    >>> with tempfile.TemporaryDirectory() as folder:
    ...     path = Path(folder) / "a.jsonl"
    ...     _ = path.write_text('{"a": 1}\n\n{"b": 2}\n', encoding="utf-8")
    ...     list(read_jsonl(path))
    [(1, {'a': 1}), (3, {'b': 2})]
    >>> with tempfile.TemporaryDirectory() as folder:
    ...     path = Path(folder) / "bad.jsonl"
    ...     _ = path.write_text('{"a": 1}\noops\n', encoding="utf-8")
    ...     list(read_jsonl(path))
    Traceback (most recent call last):
        ...
    ValueError: Invalid JSON in bad.jsonl on line 2: Expecting value: line 1 column 1 (char 0)
    >>> with tempfile.TemporaryDirectory() as folder:
    ...     path = Path(folder) / "list.jsonl"
    ...     _ = path.write_text('[1, 2]\n', encoding="utf-8")
    ...     list(read_jsonl(path))
    Traceback (most recent call last):
        ...
    ValueError: list.jsonl, line 1, must contain a JSON object.
    """
    with path.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON in {path.name} on line "
                    f"{line_number}: {error}"
                ) from error

            if not isinstance(record, dict):
                raise ValueError(
                    f"{path.name}, line {line_number}, must contain "
                    "a JSON object."
                )

            yield line_number, record


def extract_message_text(record: dict[str, Any]) -> str:
    r"""
    PURPOSE
    -------
    Flatten a chat-style "messages" list into "role: content" lines, one
    per message with non-empty content. Non-dict entries and blank content
    are skipped.

    PARAMETERS
    ----------
    record   The record to extract chat messages from.

    RETURNS
    -------
    The flattened text, or "" when the record has no "messages" list.

    EXAMPLES
    --------
    >>> record = {"messages": [{"role": "user", "content": "Hi"},
    ...                        {"role": "assistant", "content": "Hello"},
    ...                        {"role": "system", "content": "  "},
    ...                        "junk"]}
    >>> extract_message_text(record)
    'user: Hi\nassistant: Hello'
    >>> extract_message_text({"messages": [{"content": "no role"}]})
    'no role'
    >>> extract_message_text({"question": "not a chat record"})
    ''
    """
    messages = record.get("messages")

    if not isinstance(messages, list):
        return ""

    components: list[str] = []

    for message in messages:
        if not isinstance(message, dict):
            continue

        role = normalize_text(message.get("role"))
        content = normalize_text(message.get("content"))

        if not content:
            continue

        if role:
            components.append(f"{role}: {content}")
        else:
            components.append(content)

    return "\n".join(components)


def extract_llm_text(record: dict[str, Any]) -> str:
    r"""
    PURPOSE
    -------
    Build the text to embed for one LLM training record. Chat-style records
    win; otherwise the record's type decides the layout. Both the
    "training_type" and "type" spellings are accepted.

    PARAMETERS
    ----------
    record   The LLM training record to extract embeddable text from.

    RETURNS
    -------
    The text to embed for this record.

    EXAMPLES
    --------
    >>> extract_llm_text({"type": "question_reasoning_answer", "question": "What is X?",
    ...                   "reasoning": "Because.", "answer": "X is Y."})
    'Question: What is X?\nReasoning: Because.\nAnswer: X is Y.'
    >>> extract_llm_text({"training_type": "concept_explanation",
    ...                   "concept": "Osmosis", "explanation": "Water moves."})
    'Concept: Osmosis\nExplanation: Water moves.'
    >>> extract_llm_text({"type": "formula_derivation_example", "formula": "F = ma",
    ...                   "derivation": "From Newton.", "example": "A cart."})
    'Formula: F = ma\nDerivation: From Newton.\nExample: A cart.'
    >>> extract_llm_text({"messages": [{"role": "user", "content": "Hi"}]})
    'user: Hi'
    >>> extract_llm_text({"content": "fallback body"})
    'fallback body'
    >>> extract_llm_text({})
    ''
    """
    messages_text = extract_message_text(record)

    if messages_text:
        return messages_text

    # The chunking program writes this key as "type"; accept both spellings.
    training_type = normalize_text(
        record.get("training_type") or record.get("type")
    )

    if training_type == "question_reasoning_answer":
        question = normalize_text(record.get("question"))
        reasoning = normalize_text(record.get("reasoning"))
        answer = normalize_text(record.get("answer"))

        return (
            f"Question: {question}\n"
            f"Reasoning: {reasoning}\n"
            f"Answer: {answer}"
        ).strip()

    if training_type == "concept_explanation":
        concept = normalize_text(record.get("concept"))
        explanation = normalize_text(record.get("explanation"))

        return (
            f"Concept: {concept}\n"
            f"Explanation: {explanation}"
        ).strip()

    if training_type == "formula_derivation_example":
        formula = normalize_text(record.get("formula"))
        derivation = normalize_text(record.get("derivation"))
        example = normalize_text(record.get("example"))

        return (
            f"Formula: {formula}\n"
            f"Derivation: {derivation}\n"
            f"Example: {example}"
        ).strip()

    return normalize_text(
        record.get("text")
        or record.get("content")
        or record.get("answer")
        or record.get("explanation")
    )


def extract_rag_text(record: dict[str, Any]) -> str:
    r"""
    PURPOSE
    -------
    Build the text to embed for one RAG record: the record's "text" field
    when present, otherwise a composed "Topic/Concept/Formula" summary.

    PARAMETERS
    ----------
    record   The RAG record to extract embeddable text from.

    RETURNS
    -------
    The text to embed for this record.

    EXAMPLES
    --------
    >>> extract_rag_text({"text": "  The chunk body.  ", "topic": "Heat"})
    'The chunk body.'
    >>> extract_rag_text({"topic": "Heat", "concept": "Entropy", "formula": "S = k log W"})
    'Topic: Heat\nConcept: Entropy\nFormula: S = k log W'
    >>> extract_rag_text({})
    ''
    """
    text = normalize_text(record.get("text"))

    if text:
        return text

    topic = normalize_text(record.get("topic"))
    concept = normalize_text(record.get("concept"))
    formula = normalize_text(record.get("formula"))

    components = []

    if topic:
        components.append(f"Topic: {topic}")

    if concept:
        components.append(f"Concept: {concept}")

    if formula:
        components.append(f"Formula: {formula}")

    return "\n".join(components)


def create_llm_metadata(
    record: dict[str, Any],
    source_line: int,
    vector_index: int,
    text: str,
    source_file: str = LLM_INPUT_FILE.name,
) -> dict[str, Any]:
    r"""
    PURPOSE
    -------
    Build the metadata line that pairs with one LLM vector. Reuses the
    record's own id when present, otherwise derives a stable one; never
    mutates the input record.

    PARAMETERS
    ----------
    record         The source LLM training record.
    source_line    1-based line number the record came from.
    vector_index   Row index of the paired vector in the output matrix.
    text           The text that was embedded for this record.
    source_file    Name of the input file the record came from.

    RETURNS
    -------
    The metadata dict for this record.

    EXAMPLES
    --------
    >>> record = {"record_id": "QA-1", "training_type": "concept_explanation"}
    >>> metadata = create_llm_metadata(record, 4, 3, "some text", "llm.jsonl")
    >>> metadata["vector_index"], metadata["record_id"], metadata["source_line"]
    (3, 'QA-1', 4)
    >>> metadata["source_metadata"]["source"] == SOURCE_LABEL
    True
    >>> record                                   # the input record is not modified
    {'record_id': 'QA-1', 'training_type': 'concept_explanation'}
    >>> create_llm_metadata({}, 1, 0, "x")["record_id"].startswith("LLM-")
    True
    >>> create_llm_metadata({}, 1, 0, "x")["text_sha256"][:8]
    '2d711642'
    """
    record_id = normalize_text(record.get("record_id"))

    if not record_id:
        record_id = stable_id(
            "LLM",
            str(source_line),
            text,
        )

    source_metadata = record.get("source_metadata")

    if isinstance(source_metadata, dict):
        source_metadata = dict(source_metadata)   # copy: never mutate the input
    else:
        source_metadata = {}

    source_metadata["source"] = SOURCE_LABEL

    return {
        "vector_index": vector_index,
        "record_id": record_id,
        "training_type": record.get("training_type") or record.get("type"),
        "source_line": source_line,
        "source_file": source_file,
        "source_metadata": source_metadata,
        "text_sha256": hashlib.sha256(
            text.encode("utf-8")
        ).hexdigest(),
    }


def create_rag_metadata(
    record: dict[str, Any],
    source_line: int,
    vector_index: int,
    text: str,
    source_file: str = RAG_INPUT_FILE.name,
) -> dict[str, Any]:
    r"""
    PURPOSE
    -------
    Build the metadata line that pairs with one RAG vector. Reuses the
    record's own document/chunk ids when present, otherwise derives stable
    ones; never mutates the input record.

    PARAMETERS
    ----------
    record         The source RAG record.
    source_line    1-based line number the record came from.
    vector_index   Row index of the paired vector in the output matrix.
    text           The text that was embedded for this record.
    source_file    Name of the input file the record came from.

    RETURNS
    -------
    The metadata dict for this record.

    EXAMPLES
    --------
    >>> record = {"document_id": "DOC-1", "chunk_id": "CHUNK-1", "topic": "Heat",
    ...           "source_metadata": {"chapter_title": "Ch 1"}}
    >>> metadata = create_rag_metadata(record, 2, 1, "some text", "rag.jsonl")
    >>> metadata["document_id"], metadata["chunk_id"], metadata["topic"]
    ('DOC-1', 'CHUNK-1', 'Heat')
    >>> metadata["source_metadata"]
    {'chapter_title': 'Ch 1', 'source': 'Borrowed content and open source'}
    >>> record["source_metadata"]                # the input record is not modified
    {'chapter_title': 'Ch 1'}
    >>> generated = create_rag_metadata({}, 1, 0, "x")
    >>> generated["document_id"].startswith("DOC-"), generated["chunk_id"].startswith("CHK-")
    (True, True)
    """
    document_id = normalize_text(record.get("document_id"))

    if not document_id:
        document_id = stable_id(
            "DOC",
            str(source_line),
            text,
        )

    chunk_id = normalize_text(record.get("chunk_id"))

    if not chunk_id:
        chunk_id = stable_id(
            "CHK",
            document_id,
            text,
        )

    source_metadata = record.get("source_metadata")

    if isinstance(source_metadata, dict):
        source_metadata = dict(source_metadata)   # copy: never mutate the input
    else:
        source_metadata = {}

    source_metadata["source"] = SOURCE_LABEL

    return {
        "vector_index": vector_index,
        "document_id": document_id,
        "chunk_id": chunk_id,
        "topic": record.get("topic"),
        "concept": record.get("concept"),
        "formula": record.get("formula"),
        "source_line": source_line,
        "source_file": source_file,
        "source_metadata": source_metadata,
        "text_sha256": hashlib.sha256(
            text.encode("utf-8")
        ).hexdigest(),
    }


def atomic_write_json(
    destination: Path,
    data: dict[str, Any],
) -> None:
    r"""
    PURPOSE
    -------
    Write JSON to destination atomically: serialize to a temp file in the
    same folder, fsync it, then rename over the target so readers never see
    a partially written file.

    PARAMETERS
    ----------
    destination   Path to write the JSON file to.
    data          The JSON-serializable data to write.

    EXAMPLES
    --------
    >>> import tempfile
    >>> with tempfile.TemporaryDirectory() as folder:
    ...     path = Path(folder) / "out" / "manifest.json"
    ...     atomic_write_json(path, {"records": 2, "label": "café"})
    ...     (json.loads(path.read_text("utf-8")), [p.name for p in path.parent.iterdir()])
    ({'records': 2, 'label': 'café'}, ['manifest.json'])
    """
    destination.parent.mkdir(parents=True, exist_ok=True)

    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )

    temporary_path = Path(temporary_name)

    try:
        with os.fdopen(
            file_descriptor,
            "w",
            encoding="utf-8",
        ) as output_file:
            json.dump(
                data,
                output_file,
                ensure_ascii=False,
                indent=2,
            )
            output_file.write("\n")
            output_file.flush()
            os.fsync(output_file.fileno())

        temporary_path.replace(destination)

    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def write_metadata_line(
    output_file,
    metadata: dict[str, Any],
) -> None:
    r"""
    PURPOSE
    -------
    Write one compact JSON object plus a trailing newline to an open text
    file.

    PARAMETERS
    ----------
    output_file   Open text file to write the JSON line to.
    metadata      The JSON-serializable metadata to write.

    EXAMPLES
    --------
    >>> import io
    >>> buffer = io.StringIO()
    >>> write_metadata_line(buffer, {"vector_index": 0, "record_id": "QA-1"})
    >>> buffer.getvalue()
    '{"vector_index":0,"record_id":"QA-1"}\n'
    """
    json.dump(
        metadata,
        output_file,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    output_file.write("\n")


def generate_file_embeddings(
    input_path: Path,
    vector_path: Path,
    metadata_path: Path,
    model: EmbeddingModel,
    batch_size: int,
    dataset_type: str,
    verbose: bool = True,
) -> dict[str, Any]:
    r"""
    PURPOSE
    -------
    Embed one JSONL file into a .npy matrix plus a metadata JSONL sidecar.
    Vectors are written into a memory-mapped array in batches, so memory use
    stays flat no matter how many records there are. Row i of the matrix
    corresponds to the metadata line with vector_index == i. Outputs are
    written to temporary files and renamed only on success.

    PARAMETERS
    ----------
    input_path      Path to the source JSONL file.
    vector_path     Path the output .npy vector matrix is written to.
    metadata_path   Path the output metadata JSONL sidecar is written to.
    model           The embedding model to encode text batches with.
    batch_size      Number of records to encode per model.encode() call.
    dataset_type    Either "llm" or "rag"; selects the text extractor and
                    metadata builder to use.
    verbose         Whether to print progress while embedding.

    RETURNS
    -------
    A result dict describing the outcome: at minimum a "status" of
    "success" or "skipped", plus details for each case.

    RAISES
    ------
    ValueError     When dataset_type is not "llm" or "rag".
    RuntimeError   When the model does not report its embedding dimension,
                   or the number of vectors written does not match the
                   record count.

    EXAMPLES
    --------
    >>> import tempfile
    >>> class StubModel:                      # stands in for SentenceTransformer
    ...     def get_sentence_embedding_dimension(self): return 4
    ...     def encode(self, sentences, **kwargs):
    ...         return np.array([[len(s), 1.0, 0.0, 0.0] for s in sentences], dtype="float32")
    >>> with tempfile.TemporaryDirectory() as folder:
    ...     root = Path(folder)
    ...     source = root / "rag_data.jsonl"
    ...     _ = source.write_text(
    ...         '{"chunk_id": "C1", "document_id": "D1", "text": "first chunk"}\n'
    ...         '{"chunk_id": "C2", "document_id": "D1", "text": "second chunk"}\n',
    ...         encoding="utf-8")
    ...     result = generate_file_embeddings(
    ...         source, root / "rag.npy", root / "rag_meta.jsonl",
    ...         StubModel(), batch_size=1, dataset_type="rag", verbose=False)
    ...     vectors = np.load(root / "rag.npy")
    ...     lines = (root / "rag_meta.jsonl").read_text("utf-8").splitlines()
    >>> result["status"], result["record_count"], result["vector_shape"]
    ('success', 2, [2, 4])
    >>> vectors.shape, vectors.dtype
    ((2, 4), dtype('float32'))
    >>> [json.loads(line)["chunk_id"] for line in lines]
    ['C1', 'C2']
    >>> generate_file_embeddings(Path("/no/such.jsonl"), Path("v.npy"), Path("m.jsonl"),
    ...                          StubModel(), 8, "rag", verbose=False)["status"]
    'skipped'
    """
    if not input_path.exists():
        return {
            "status": "skipped",
            "reason": "Input file does not exist.",
            "input_file": str(input_path),
        }

    total_records = count_jsonl_records(input_path)

    if total_records == 0:
        return {
            "status": "skipped",
            "reason": "Input file contains no records.",
            "input_file": str(input_path),
        }

    vector_path.parent.mkdir(parents=True, exist_ok=True)

    temporary_vector_path = vector_path.with_name(
        f".{vector_path.name}.tmp.npy"
    )

    temporary_metadata_path = metadata_path.with_name(
        f".{metadata_path.name}.tmp"
    )

    embedding_dimension = model.get_sentence_embedding_dimension()

    if embedding_dimension is None:
        raise RuntimeError(
            "The embedding model did not report its vector dimension."
        )

    vector_array = np.lib.format.open_memmap(
        temporary_vector_path,
        mode="w+",
        dtype=VECTOR_DATA_TYPE,
        shape=(total_records, embedding_dimension),
    )

    current_index = 0
    text_batch: list[str] = []
    metadata_batch: list[dict[str, Any]] = []

    if dataset_type == "llm":
        text_extractor = extract_llm_text
        metadata_creator = create_llm_metadata
    elif dataset_type == "rag":
        text_extractor = extract_rag_text
        metadata_creator = create_rag_metadata
    else:
        raise ValueError(
            f"Unsupported dataset type: {dataset_type}"
        )

    def encode_and_write_batch(metadata_output) -> None:
        """
        PURPOSE
        -------
        Encode the accumulated text_batch, write the resulting vectors into
        vector_array and the paired metadata lines to metadata_output, then
        advance current_index and clear both batches. Does nothing when
        text_batch is empty.

        PARAMETERS
        ----------
        metadata_output   Open text file the metadata lines are appended to.
        """
        nonlocal current_index

        if not text_batch:
            return

        with inference_context():
            batch_vectors = model.encode(
                text_batch,
                batch_size=len(text_batch),
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=NORMALIZE_EMBEDDINGS,
            )

        batch_vectors = np.asarray(
            batch_vectors,
            dtype=VECTOR_DATA_TYPE,
        )

        batch_length = len(batch_vectors)
        ending_index = current_index + batch_length

        vector_array[current_index:ending_index] = batch_vectors

        for metadata in metadata_batch:
            write_metadata_line(metadata_output, metadata)

        current_index = ending_index
        text_batch.clear()
        metadata_batch.clear()

        if verbose:
            print(
                f"\r{input_path.name}: "
                f"{current_index}/{total_records} records embedded",
                end="",
                flush=True,
            )

    try:
        with temporary_metadata_path.open(
            "w",
            encoding="utf-8",
        ) as metadata_output:
            for source_line, record in read_jsonl(input_path):
                text = text_extractor(record)

                if not text:
                    text = "[EMPTY RECORD]"

                metadata = metadata_creator(
                    record=record,
                    source_line=source_line,
                    vector_index=current_index + len(text_batch),
                    text=text,
                    source_file=input_path.name,
                )

                text_batch.append(text)
                metadata_batch.append(metadata)

                if len(text_batch) >= batch_size:
                    encode_and_write_batch(metadata_output)

            encode_and_write_batch(metadata_output)

            metadata_output.flush()
            os.fsync(metadata_output.fileno())

        vector_array.flush()
        vector_array = None

        if current_index != total_records:
            raise RuntimeError(
                f"Expected {total_records} vectors but generated "
                f"{current_index}."
            )

        temporary_vector_path.replace(vector_path)
        temporary_metadata_path.replace(metadata_path)

        if verbose:
            print()

        return {
            "status": "success",
            "dataset_type": dataset_type,
            "input_file": str(input_path),
            "vector_file": str(vector_path),
            "metadata_file": str(metadata_path),
            "record_count": current_index,
            "embedding_dimension": embedding_dimension,
            "vector_shape": [
                current_index,
                embedding_dimension,
            ],
            "vector_dtype": str(
                np.dtype(VECTOR_DATA_TYPE)
            ),
        }

    except Exception:
        vector_array = None
        temporary_vector_path.unlink(missing_ok=True)
        temporary_metadata_path.unlink(missing_ok=True)
        raise


def validate_vector_output(
    vector_path: Path,
    expected_records: int,
    expected_dimension: int,
) -> None:
    r"""
    PURPOSE
    -------
    Validate a written .npy vector file: correct shape, no NaN/infinite
    values, and (when normalization is enabled) unit-length rows on a
    sample of up to 1000 vectors.

    PARAMETERS
    ----------
    vector_path          Path to the .npy file to validate.
    expected_records     Expected number of rows.
    expected_dimension   Expected number of columns.

    RAISES
    ------
    RuntimeError   When the shape does not match, non-finite values are
                   found, or sampled rows are not unit length.

    EXAMPLES
    --------
    >>> import tempfile
    >>> with tempfile.TemporaryDirectory() as folder:
    ...     path = Path(folder) / "v.npy"
    ...     np.save(path, np.array([[0.6, 0.8], [1.0, 0.0]], dtype="float32"))
    ...     validate_vector_output(path, 2, 2)          # unit-length rows: passes
    ...     validate_vector_output(path, 5, 2)
    Traceback (most recent call last):
        ...
    RuntimeError: Vector validation failed for v.npy. Expected shape (5, 2), received (2, 2).
    >>> with tempfile.TemporaryDirectory() as folder:
    ...     path = Path(folder) / "bad.npy"
    ...     np.save(path, np.array([[np.nan, 0.0]], dtype="float32"))
    ...     validate_vector_output(path, 1, 2)
    Traceback (most recent call last):
        ...
    RuntimeError: Vector validation failed for bad.npy: NaN or infinite values were detected.
    >>> with tempfile.TemporaryDirectory() as folder:
    ...     path = Path(folder) / "long.npy"
    ...     np.save(path, np.array([[3.0, 4.0]], dtype="float32"))
    ...     validate_vector_output(path, 1, 2)
    Traceback (most recent call last):
        ...
    RuntimeError: Vector validation failed for long.npy: normalized embeddings do not have unit length.
    """
    vectors = np.load(
        vector_path,
        mmap_mode="r",
        allow_pickle=False,
    )

    expected_shape = (
        expected_records,
        expected_dimension,
    )

    if vectors.shape != expected_shape:
        raise RuntimeError(
            f"Vector validation failed for {vector_path.name}. "
            f"Expected shape {expected_shape}, received "
            f"{vectors.shape}."
        )

    if not np.isfinite(vectors).all():
        raise RuntimeError(
            f"Vector validation failed for {vector_path.name}: "
            "NaN or infinite values were detected."
        )

    if NORMALIZE_EMBEDDINGS and len(vectors) > 0:
        sample_size = min(len(vectors), 1000)
        sample_indexes = np.linspace(
            0,
            len(vectors) - 1,
            num=sample_size,
            dtype=int,
        )

        sample_norms = np.linalg.norm(
            vectors[sample_indexes],
            axis=1,
        )

        nonzero_norms = sample_norms[sample_norms > 0]

        if (
            len(nonzero_norms) > 0
            and not np.allclose(
                nonzero_norms,
                1.0,
                atol=1e-3,
            )
        ):
            raise RuntimeError(
                f"Vector validation failed for {vector_path.name}: "
                "normalized embeddings do not have unit length."
            )


def run_pipeline(
    input_folder: Path = INPUT_FOLDER,
    output_folder: Path = OUTPUT_FOLDER,
    model: EmbeddingModel | None = None,
    device: str | None = None,
    batch_size: int | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    r"""
    PURPOSE
    -------
    Embed both datasets (LLM training data and RAG data) found in
    input_folder, writing vector matrices, metadata sidecars and a manifest
    to output_folder.

    PARAMETERS
    ----------
    input_folder    Folder containing llm_training_data.jsonl and
                    rag_data.jsonl.
    output_folder   Folder the vectors, metadata and manifest are written
                    to.
    model           An already-loaded (or stub) embedding model; when None,
                    the real sentence-transformers model is loaded, which
                    needs torch installed.
    device          Torch device to use; auto-selected when None.
    batch_size      Records per encode() call; auto-selected from the
                    device when None.
    verbose         Whether to print progress and a summary.

    RETURNS
    -------
    The manifest dict describing both datasets' embedding results.

    RAISES
    ------
    FileNotFoundError    When input_folder does not exist.
    NotADirectoryError   When input_folder exists but is not a folder.

    EXAMPLES
    --------
    >>> import tempfile
    >>> class StubModel:
    ...     def get_sentence_embedding_dimension(self): return 3
    ...     def encode(self, sentences, **kwargs):
    ...         return np.array([[1.0, 0.0, 0.0] for _ in sentences], dtype="float32")
    >>> with tempfile.TemporaryDirectory() as folder:
    ...     inp, out = Path(folder) / "chunked", Path(folder) / "embedded"
    ...     inp.mkdir()
    ...     _ = (inp / "llm_training_data.jsonl").write_text(
    ...         '{"record_id": "QA-1", "type": "concept_explanation",'
    ...         ' "concept": "Heat", "explanation": "Energy in transit."}\n', encoding="utf-8")
    ...     _ = (inp / "rag_data.jsonl").write_text(
    ...         '{"chunk_id": "C1", "document_id": "D1", "text": "A chunk of text."}\n',
    ...         encoding="utf-8")
    ...     manifest = run_pipeline(inp, out, model=StubModel(), device="cpu",
    ...                             batch_size=8, verbose=False)
    ...     written = sorted(p.name for p in out.iterdir())
    >>> manifest["total_embedded_records"], manifest["device"]
    (2, 'cpu')
    >>> written
    ['embedding_manifest.json', 'llm_embedding_metadata.jsonl', 'llm_embeddings.npy', 'rag_embedding_metadata.jsonl', 'rag_embeddings.npy']
    >>> [dataset["status"] for dataset in manifest["datasets"]]
    ['success', 'success']
    >>> run_pipeline(Path("/definitely/not/here"), Path("/tmp/x"), model=StubModel())
    Traceback (most recent call last):
        ...
    FileNotFoundError: Input folder does not exist:
    /definitely/not/here
    """
    if not input_folder.exists():
        raise FileNotFoundError(
            f"Input folder does not exist:\n{input_folder}"
        )

    if not input_folder.is_dir():
        raise NotADirectoryError(
            f"Input path is not a folder:\n{input_folder}"
        )

    output_folder.mkdir(parents=True, exist_ok=True)

    if device is None:
        device = select_device()

    if batch_size is None:
        batch_size = select_batch_size(device)

    def announce(message: str) -> None:
        """
        PURPOSE
        -------
        Print message when verbose is enabled.
        """
        if verbose:
            print(message)

    announce(f"Device: {device}")
    announce(f"Embedding model: {MODEL_NAME}")
    announce(f"Batch size: {batch_size}")

    if model is None:
        announce("Loading embedding model...")
        model = load_model(device)

    datasets = (
        {
            "dataset_type": "llm",
            "input_path": input_folder / LLM_INPUT_FILE.name,
            "vector_path": output_folder / LLM_VECTOR_FILE.name,
            "metadata_path": output_folder / LLM_METADATA_FILE.name,
            "heading": "\nEmbedding LLM training records...",
        },
        {
            "dataset_type": "rag",
            "input_path": input_folder / RAG_INPUT_FILE.name,
            "vector_path": output_folder / RAG_VECTOR_FILE.name,
            "metadata_path": output_folder / RAG_METADATA_FILE.name,
            "heading": "\nEmbedding RAG records...",
        },
    )

    results: list[dict[str, Any]] = []

    for dataset in datasets:
        announce(dataset["heading"])

        result = generate_file_embeddings(
            input_path=dataset["input_path"],
            vector_path=dataset["vector_path"],
            metadata_path=dataset["metadata_path"],
            model=model,
            batch_size=batch_size,
            dataset_type=dataset["dataset_type"],
            verbose=verbose,
        )

        results.append(result)

        if result["status"] == "success":
            validate_vector_output(
                vector_path=dataset["vector_path"],
                expected_records=result["record_count"],
                expected_dimension=result["embedding_dimension"],
            )

    successful_results = [
        result
        for result in results
        if result["status"] == "success"
    ]

    manifest_path = output_folder / MANIFEST_FILE.name

    manifest = {
        "program": "SoakinGarri Embedding Program",
        "input_folder": str(input_folder),
        "output_folder": str(output_folder),
        "source_label": SOURCE_LABEL,
        "embedding_model": MODEL_NAME,
        "device": device,
        "batch_size": batch_size,
        "maximum_sequence_length": MAX_SEQUENCE_LENGTH,
        "normalized_embeddings": NORMALIZE_EMBEDDINGS,
        "vector_dtype": str(np.dtype(VECTOR_DATA_TYPE)),
        "total_embedded_records": sum(
            result["record_count"]
            for result in successful_results
        ),
        "datasets": results,
    }

    atomic_write_json(manifest_path, manifest)

    if verbose:
        print("\nEmbedding completed successfully.")

        for result in successful_results:
            print(
                f"{result['dataset_type'].upper()} vectors: "
                f"{result['vector_shape']}"
            )
            print(f"Vector file: {result['vector_file']}")
            print(f"Metadata file: {result['metadata_file']}")

        print(f"Manifest: {manifest_path}")

    return manifest


def main() -> None:
    """
    PURPOSE
    -------
    Embed the configured folders and print a summary.
    """
    run_pipeline()


if __name__ == "__main__":
    main()