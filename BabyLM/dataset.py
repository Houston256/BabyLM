from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class PackedTokenDataset(Dataset):
    def __init__(self, bin_path: str | Path, seq_length: int):
        self.data = np.memmap(bin_path, dtype=np.uint16, mode="r")
        self.seq_length = seq_length
        self.n_chunks = len(self.data) // seq_length

    def __len__(self) -> int:
        return self.n_chunks

    def __getitem__(self, idx: int) -> torch.Tensor:
        start = idx * self.seq_length
        chunk = self.data[start : start + self.seq_length].astype(np.int64)
        return torch.from_numpy(chunk)


# Special tokens occupy ids 0..NUM_SPECIAL_TOKENS-1 (see BabyLM/tokenizer/bpe_tokenizer.py).
NUM_SPECIAL_TOKENS = 5  # [UNK], [CLS], [SEP], [PAD], [MASK]


def apply_mlm_mask(
    tokens: torch.Tensor,
    mask_token_id: int,
    vocab_size: int,
    mask_prob: float = 0.15,
) -> tuple[torch.Tensor, torch.Tensor]:
    # tokens: (batch, seq) -> inputs, labels both (batch, seq)
    B, T = tokens.shape
    device = tokens.device

    mask = torch.rand(B, T, device=device) < mask_prob                   # (batch, seq) bool
    labels = torch.where(mask, tokens, torch.full_like(tokens, -100))    # (batch, seq)

    rand = torch.rand(B, T, device=device)
    inputs = tokens.clone()
    inputs[mask & (rand < 0.8)] = mask_token_id
    rand_pick = mask & (rand >= 0.8) & (rand < 0.9)
    inputs[rand_pick] = torch.randint(
        NUM_SPECIAL_TOKENS, vocab_size, (int(rand_pick.sum()),), device=device
    )

    return inputs, labels


def apply_mntp_mask(
    tokens: torch.Tensor,
    mask_token_id: int,
    vocab_size: int,
    mask_prob: float = 0.15,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Masked Next-Token Prediction (GPT-BERT).

    Same 80/10/10 corruption as MLM, but labels are shifted left by one so that
    the prediction at position k targets the (possibly masked) token at k+1 —
    matching the alignment used by CLM, so both objectives share the same head.
    """
    B, T = tokens.shape
    device = tokens.device

    mask = torch.rand(B, T, device=device) < mask_prob
    mask[:, 0] = False  # no k=-1 output to read; masking pos 0 yields no supervision

    inputs = tokens.clone()
    rand = torch.rand(B, T, device=device)
    inputs[mask & (rand < 0.8)] = mask_token_id
    rand_pick = mask & (rand >= 0.8) & (rand < 0.9)
    inputs[rand_pick] = torch.randint(
        NUM_SPECIAL_TOKENS, vocab_size, (int(rand_pick.sum()),), device=device
    )

    # Shifted labels: label[:, k] = tokens[:, k+1] if mask[:, k+1] else -100
    labels = torch.full_like(tokens, -100)
    labels[:, :-1] = torch.where(mask[:, 1:], tokens[:, 1:], torch.full_like(tokens[:, 1:], -100))
    return inputs, labels


def make_clm_pair(tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # tokens: (batch, seq) -> inputs, labels both (batch, seq); model shifts internally
    return tokens, tokens