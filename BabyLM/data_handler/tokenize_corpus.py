import re
from pathlib import Path
from typing import Iterator

import numpy as np
from datasets import load_dataset
from tokenizers import Tokenizer
from tqdm import tqdm

# Matches CHILDES-style speaker tags at the start of an utterance: *CHI:\t, *MOT:, etc.
_SPEAKER_TAG = re.compile(r"^\*[A-Z]+:\s*")
# Document boundary marker used in childes, simple_wiki, gutenberg raw files.
_DOC_HEADER = re.compile(r"^= = =.*= = =\s*$")


def _iter_docs_with_headers(path: Path, strip_speaker_tags: bool) -> Iterator[str]:
    """Yield one document per `= = = ... = = =` header block (childes / wiki / gutenberg).

    For childes the header marks .cha file boundaries (conversations).
    Lines starting with `[` (action annotations) or `%` (CHAT metadata) are dropped.
    Speaker tags `*XXX:\\t` are kept by default — they signal turn changes.
    """
    buf: list[str] = []
    with path.open() as f:
        for raw in f:
            line = raw.rstrip("\n")
            if _DOC_HEADER.match(line):
                if buf:
                    yield " ".join(buf)
                buf = []
                continue
            if not line or line.startswith("[") or line.startswith("%"):
                continue
            if strip_speaker_tags:
                line = _SPEAKER_TAG.sub("", line)
                if not line:
                    continue
            buf.append(line)
    if buf:
        yield " ".join(buf)


def _iter_docs_flat(path: Path) -> Iterator[str]:
    """No document structure available — yield the whole file as one document.

    Used for open_subtitles, bnc_spoken, switchboard — sources where the original
    document boundaries weren't preserved during preprocessing.
    """
    with path.open() as f:
        chunk = " ".join(line.strip() for line in f if line.strip())
    if chunk:
        yield chunk


# Per-source parser dispatch. Keys are filename stems under raw_dir.
_RAW_SOURCES: dict[str, str] = {
    "childes":        "headers",
    "simple_wiki":    "headers",
    "gutenberg":      "headers",
    "open_subtitles": "flat",
    "bnc_spoken":     "flat",
    "switchboard":    "flat",
}


def tokenize_from_raw(
    tokenizer_path: str | Path,
    output_path: str | Path,
    raw_dir: str | Path,
    strip_speaker_tags: bool = False,
) -> None:
    """Tokenize from the raw per-source txt files, preserving document granularity.

    Inserts [SEP] only at document boundaries (conversations / articles / books),
    not between utterances within a document. This gives the model a sparse,
    meaningful boundary signal instead of one SEP every ~10 tokens.
    """
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    if tokenizer.get_vocab_size() > 65535:
        raise ValueError("vocab too large for uint16 storage")
    sep_id = tokenizer.token_to_id("[SEP]")
    if sep_id is None:
        raise ValueError("tokenizer has no [SEP] token; raw mode always inserts SEP between docs")

    raw_dir = Path(raw_dir)
    tokens: list[int] = []
    n_docs_total = 0
    for stem, mode in _RAW_SOURCES.items():
        path = raw_dir / f"{stem}.train.txt"
        if not path.exists():
            raise FileNotFoundError(f"missing raw source: {path}")
        iterator = (
            _iter_docs_with_headers(path, strip_speaker_tags) if mode == "headers"
            else _iter_docs_flat(path)
        )
        n_docs = 0
        n_tokens_before = len(tokens)
        for doc in tqdm(iterator, desc=f"  {stem}"):
            tokens.extend(tokenizer.encode(doc).ids)
            tokens.append(sep_id)
            n_docs += 1
        n_docs_total += n_docs
        added = len(tokens) - n_tokens_before
        print(f"  {stem:16s}  mode={mode:7s}  docs={n_docs:>7,}  tokens={added:>11,}")

    arr = np.array(tokens, dtype=np.uint16)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    arr.tofile(output_path)
    print(f"\nwrote {len(arr):,} tokens to {output_path}")
    print(f"inserted [SEP] (id={sep_id}) between {n_docs_total:,} documents")


def tokenize_corpus(
    tokenizer_path: str | Path,
    output_path: str | Path,
    dataset_name: str = "BabyLM-community/BabyLM-2026-Strict-Small",
    split: str = "train",
    text_column: str = "text",
    strip_speaker_tags: bool = False,
    insert_sep: bool = False,
) -> None:
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    if tokenizer.get_vocab_size() > 65535:
        raise ValueError("vocab too large for uint16 storage")

    sep_id = tokenizer.token_to_id("[SEP]") if insert_sep else None
    if insert_sep and sep_id is None:
        raise ValueError("tokenizer has no [SEP] token; cannot --insert-sep")

    dataset = load_dataset(dataset_name, split=split)

    tokens: list[int] = []
    stripped = 0
    for item in tqdm(dataset, desc=f"tokenizing {split}"):
        text = item[text_column]
        if not text:
            continue
        if strip_speaker_tags:
            new = _SPEAKER_TAG.sub("", text)
            if new != text:
                stripped += 1
            text = new
            if not text:
                continue
        tokens.extend(tokenizer.encode(text).ids)
        if insert_sep:
            tokens.append(sep_id)

    arr = np.array(tokens, dtype=np.uint16)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    arr.tofile(output_path)
    print(f"wrote {len(arr):,} tokens to {output_path}")
    if strip_speaker_tags:
        print(f"stripped speaker tags from {stripped:,} docs")
    if insert_sep:
        print(f"inserted [SEP] (id={sep_id}) between {len(dataset):,} docs")
