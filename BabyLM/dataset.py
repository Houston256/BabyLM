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
    inputs[rand_pick] = torch.randint(0, vocab_size, (int(rand_pick.sum()),), device=device)

    return inputs, labels


def make_clm_pair(tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # tokens: (batch, seq) -> inputs, labels both (batch, seq); model shifts internally
    return tokens, tokens