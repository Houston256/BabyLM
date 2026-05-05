from pathlib import Path

import numpy as np
from datasets import load_dataset
from tokenizers import Tokenizer
from tqdm import tqdm


def tokenize_corpus(
    tokenizer_path: str | Path,
    output_path: str | Path,
    dataset_name: str = "BabyLM-community/BabyLM-2026-Strict-Small",
    split: str = "train",
    text_column: str = "text",
) -> None:
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    if tokenizer.get_vocab_size() > 65535:
        raise ValueError("vocab too large for uint16 storage")

    dataset = load_dataset(dataset_name, split=split)

    tokens: list[int] = []
    for item in tqdm(dataset, desc=f"tokenizing {split}"):
        text = item[text_column]
        if text:
            tokens.extend(tokenizer.encode(text).ids)

    arr = np.array(tokens, dtype=np.uint16)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    arr.tofile(output_path)
    print(f"wrote {len(arr):,} tokens to {output_path}")
