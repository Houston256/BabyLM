import argparse
import json
import math
import random
import time
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tokenizers import Tokenizer

from BabyLM.config import ModelConfig
from BabyLM.dataset import PackedTokenDataset, apply_mlm_mask, make_clm_pair
from BabyLM.logger import build_logger
from BabyLM.model import GPTBert


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def cosine_lr(step: int, max_steps: int, warmup: int, base_lr: float, min_lr: float) -> float:
    if step < warmup:
        return base_lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, max_steps - warmup)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))


def infinite(loader: DataLoader):
    while True:
        yield from loader


def add_pretrain_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", type=str, default="configs/small.json")
    p.add_argument("--tokenizer", type=str, default="models/tokenizer.json")
    p.add_argument("--train-bin", type=str, default="data/train.bin")
    p.add_argument("--output-dir", type=str, default="checkpoints/")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-steps", type=int, default=10_000)
    p.add_argument("--warmup-steps", type=int, default=500)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--min-lr-ratio", type=float, default=0.1)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--mask-prob", type=float, default=0.15)
    p.add_argument("--hybrid-numerator", type=int, default=15)
    p.add_argument("--hybrid-denominator", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--save-every", type=int, default=1_000)
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb-project", type=str, default="babylm")
    p.add_argument("--run-name", type=str, default=None)


def run_pretrain(args: argparse.Namespace) -> None:
    device = get_device()
    print(f"device: {device}")

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)

    with open(args.config) as f:
        cfg = ModelConfig(**json.load(f))

    tokenizer = Tokenizer.from_file(args.tokenizer)
    if tokenizer.get_vocab_size() != cfg.vocab_size:
        raise ValueError(
            f"tokenizer vocab ({tokenizer.get_vocab_size()}) != model vocab ({cfg.vocab_size})"
        )
    mask_id = tokenizer.token_to_id("[MASK]")
    if mask_id is None:
        raise ValueError("tokenizer must define a [MASK] token")

    dataset = PackedTokenDataset(args.train_bin, cfg.max_position_embeddings)
    print(f"chunks: {len(dataset):,} ({len(dataset) * cfg.max_position_embeddings:,} tokens)")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    batches = infinite(loader)

    model = GPTBert(cfg).to(device)
    print(f"model params: {model.num_parameters():,}")

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.98),
        weight_decay=args.weight_decay,
    )

    logger = build_logger(args.wandb, args.wandb_project, args.run_name)
    logger.update_config({**asdict(cfg), **vars(args)})

    use_amp = device.type == "cuda"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    mlm_p = args.hybrid_numerator / args.hybrid_denominator

    model.train()
    t_start = time.time()

    for step in range(args.max_steps):
        is_causal = rng.random() >= mlm_p
        chunks = next(batches).to(device, non_blocking=True)
        if is_causal:
            input_ids, labels = make_clm_pair(chunks)
        else:
            input_ids, labels = apply_mlm_mask(chunks, mask_id, cfg.vocab_size, args.mask_prob)

        lr = cosine_lr(step, args.max_steps, args.warmup_steps, args.lr, args.lr * args.min_lr_ratio)
        for g in opt.param_groups:
            g["lr"] = lr

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
            _, loss = model(input_ids, labels=labels, is_causal=is_causal)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()

        if step % args.log_every == 0:
            elapsed = time.time() - t_start
            tokens_seen = (step + 1) * args.batch_size * cfg.max_position_embeddings
            tag = "clm" if is_causal else "mlm"
            metrics = {
                "loss": loss.item(),
                f"loss/{tag}": loss.item(),
                "lr": lr,
                "grad_norm": float(grad_norm),
                "tokens_per_sec": tokens_seen / elapsed,
            }
            logger.log(metrics, step=step)
            print(f"step {step:6d} | {tag} | loss {loss.item():.4f} | lr {lr:.2e} | {metrics['tokens_per_sec']:,.0f} tok/s")

        if step > 0 and step % args.save_every == 0:
            ckpt = output_dir / f"step_{step}.pt"
            torch.save(
                {"model": model.state_dict(), "cfg": asdict(cfg), "step": step, "args": vars(args)},
                ckpt,
            )
            print(f"saved {ckpt}")

    final = output_dir / "final.pt"
    torch.save(
        {"model": model.state_dict(), "cfg": asdict(cfg), "step": args.max_steps, "args": vars(args)},
        final,
    )
    print(f"saved {final}")
    logger.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_pretrain_args(parser)
    run_pretrain(parser.parse_args())
