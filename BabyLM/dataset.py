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


class DocumentSegmentDataset(Dataset):
    """Block-diagonal document packing, like GPT-BERT's MaskedDataset/CausalDataset.

    Each item greedily packs several whole within-document segments into one seq_len window,
    each prefixed with [CLS]/<s>; the tail is padded. Instead of a (T,T) block mask we return a
    per-position segment id `seg_id` (documents 0,1,2,…; pad = -1); the trainer builds the
    block-diagonal attention mask on device as `seg_id[i] == seg_id[j]`, so tokens attend only
    within their own document. `valid` marks label-eligible positions (real tokens, not <s>/pad).
    The flat bin must contain documents separated by `sep_id` (`tokenize-corpus --source-mode raw`).
    """

    def __init__(self, bin_path: str | Path, seq_length: int, cls_id: int, sep_id: int, pad_id: int):
        self.data = np.memmap(bin_path, dtype=np.uint16, mode="r")
        self.seq_length = seq_length
        self.cls_id = cls_id
        self.pad_id = pad_id
        content = seq_length - 1  # each packed piece is cls + up to `content` tokens
        sep_pos = np.flatnonzero(np.asarray(self.data) == sep_id)
        bounds = np.concatenate(([0], sep_pos + 1))
        ends = np.concatenate((sep_pos, [len(self.data)]))
        # within-document pieces, each at most `content` tokens long
        self.pieces: list[tuple[int, int]] = []
        for doc_start, doc_end in zip(bounds.tolist(), ends.tolist()):
            for off in range(doc_start, doc_end, content):
                end = min(off + content, doc_end)
                if end > off:
                    self.pieces.append((off, end))

    def __len__(self) -> int:
        return len(self.pieces)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        T = self.seq_length
        input_ids = np.full(T, self.pad_id, dtype=np.int64)
        seg_id = np.full(T, -1, dtype=np.int64)   # -1 = padding (attends to nothing real)
        valid = np.zeros(T, dtype=np.int64)        # 1 = real content token (label-eligible)
        pos, doc = 0, 0
        i = idx
        while pos < T and i < len(self.pieces):
            off, end = self.pieces[i]
            n = end - off
            if pos + 1 + n > T:                    # cls + tokens wouldn't fit
                if pos > 0:
                    break                          # leave for the next item; pad the tail
                n = T - 1                           # first piece too long: truncate it
                end = off + n
            input_ids[pos] = self.cls_id            # <s> for this document
            input_ids[pos + 1 : pos + 1 + n] = self.data[off:end].astype(np.int64)
            seg_id[pos : pos + 1 + n] = doc         # cls + tokens share one block
            valid[pos + 1 : pos + 1 + n] = 1        # content tokens (not the <s>) are predicted
            pos += 1 + n
            doc += 1
            i += 1
        return torch.from_numpy(input_ids), torch.from_numpy(seg_id), torch.from_numpy(valid)


class BPEDropoutDataset(Dataset):
    def __init__(self, dataset_name: str, split: str, tokenizer_path: str | Path, seq_length: int, bpe_dropout: float):
        from datasets import load_dataset
        from tokenizers import Tokenizer
        
        self.dataset = load_dataset(dataset_name, split=split)
        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self.tokenizer.model.dropout = bpe_dropout
        self.seq_length = seq_length
        self.sep_id = self.tokenizer.token_to_id("[SEP]")
        
        self.data = np.array([], dtype=np.int64)
        self.refresh()

    def refresh(self):
        print("Re-tokenizing dataset with BPE dropout...")
        tokens: list[int] = []
        # Encoding sequentially is fast enough for 10M words (~10-20 seconds)
        for item in self.dataset:
            text = item.get("text", "")
            if not text:
                continue
            tokens.extend(self.tokenizer.encode(text).ids)
            if self.sep_id is not None:
                tokens.append(self.sep_id)
                
        self.data = np.array(tokens, dtype=np.uint16)
        self.n_chunks = len(self.data) // self.seq_length

    def __len__(self) -> int:
        return self.n_chunks

    def __getitem__(self, idx: int) -> torch.Tensor:
        start = idx * self.seq_length
        chunk = self.data[start : start + self.seq_length].astype(np.int64)
        return torch.from_numpy(chunk)


# Special tokens occupy ids 0..NUM_SPECIAL_TOKENS-1 (see BabyLM/tokenizer/bpe_tokenizer.py).
NUM_SPECIAL_TOKENS = 5  # [UNK], [CLS], [SEP], [PAD], [MASK]


def _get_span_mask(
    shape: tuple[int, int],
    mask_prob: float,
    device: torch.device,
    max_span_length: int = 5,
) -> torch.Tensor:
    B, T = shape
    avg_span_len = (1 + max_span_length) / 2.0
    # Slightly boost start_prob to account for overlaps
    start_prob = (mask_prob / avg_span_len) * 1.1 
    
    start_mask = torch.rand(B, T, device=device) < start_prob
    span_lens = torch.randint(1, max_span_length + 1, (B, T), device=device)
    
    mask = torch.zeros(B, T, dtype=torch.bool, device=device)
    for i in range(max_span_length):
        if i == 0:
            valid_starts = start_mask
        else:
            valid_starts = start_mask & (span_lens > i)
            valid_starts = torch.cat([
                torch.zeros(B, i, dtype=torch.bool, device=device),
                valid_starts[:, :-i]
            ], dim=1)
        mask = mask | valid_starts
        
    return mask


def apply_mlm_mask(
    tokens: torch.Tensor,
    mask_token_id: int,
    vocab_size: int,
    mask_prob: float = 0.15,
    span_masking: bool = False,
    n_special_tokens: int = NUM_SPECIAL_TOKENS,
    valid: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    # tokens: (batch, seq) -> inputs, labels both (batch, seq)
    # valid (batch, seq) bool: positions eligible to be masked/predicted (excludes pad). None = all.
    B, T = tokens.shape
    device = tokens.device

    if span_masking:
        mask = _get_span_mask((B, T), mask_prob, device)
    else:
        mask = torch.rand(B, T, device=device) < mask_prob                   # (batch, seq) bool

    mask[:, 0] = False  # never corrupt the leading [CLS]/<s>
    if valid is not None:
        mask &= valid.bool()
    
    labels = torch.where(mask, tokens, torch.full_like(tokens, -100))    # (batch, seq)

    rand = torch.rand(B, T, device=device)
    inputs = tokens.clone()
    inputs[mask & (rand < 0.8)] = mask_token_id
    rand_pick = mask & (rand >= 0.8) & (rand < 0.9)
    inputs[rand_pick] = torch.randint(
        n_special_tokens, vocab_size, (int(rand_pick.sum()),), device=device
    )

    return inputs, labels


def apply_mntp_mask(
    tokens: torch.Tensor,
    mask_token_id: int,
    vocab_size: int,
    mask_prob: float = 0.15,
    span_masking: bool = False,
    n_special_tokens: int = NUM_SPECIAL_TOKENS,
    valid: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Masked Next-Token Prediction (GPT-BERT).

    Same 80/10/10 corruption as MLM, but labels are shifted left by one so that
    the prediction at position k targets the (possibly masked) token at k+1 —
    matching the alignment used by CLM, so both objectives share the same head.
    """
    B, T = tokens.shape
    device = tokens.device

    if span_masking:
        mask = _get_span_mask((B, T), mask_prob, device)
    else:
        mask = torch.rand(B, T, device=device) < mask_prob
    
    mask[:, 0] = False  # no k=-1 output to read; masking pos 0 yields no supervision
    if valid is not None:
        mask &= valid.bool()

    inputs = tokens.clone()
    rand = torch.rand(B, T, device=device)
    inputs[mask & (rand < 0.8)] = mask_token_id
    rand_pick = mask & (rand >= 0.8) & (rand < 0.9)
    inputs[rand_pick] = torch.randint(
        n_special_tokens, vocab_size, (int(rand_pick.sum()),), device=device
    )

    # Shifted labels: label[:, k] = tokens[:, k+1] if mask[:, k+1] else -100
    labels = torch.full_like(tokens, -100)
    labels[:, :-1] = torch.where(mask[:, 1:], tokens[:, 1:], torch.full_like(tokens[:, 1:], -100))
    return inputs, labels


def make_clm_pair(tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # tokens: (batch, seq) -> inputs, labels both (batch, seq); model shifts internally
    return tokens, tokens