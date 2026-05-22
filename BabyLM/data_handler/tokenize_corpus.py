import re
from pathlib import Path

import numpy as np
from datasets import load_dataset
from tokenizers import Tokenizer
from tqdm import tqdm

# Matches CHILDES-style speaker tags at the start of an utterance: *CHI:\t, *MOT:, etc.
_SPEAKER_TAG = re.compile(r"^\*[A-Z]+:\s*")


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
