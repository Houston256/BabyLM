import argparse
import math
import os
import random
import subprocess
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerFast

from BabyLM.dataset import PackedTokenDataset, apply_mlm_mask, make_clm_pair
from BabyLM.eval_report import model_results_dir, parse_eval_results
from BabyLM.logger import build_logger
from BabyLM.modeling_gptbert import GPTBertConfig, GPTBertForCausalLM

EVAL_BACKENDS = ("causal", "mlm")


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def cosine_lr(step: int, max_steps: int, warmup_steps: int, base_lr: float, min_lr: float) -> float:
    if step < warmup_steps:
        return base_lr * (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))


def constant_lr(step: int, max_steps: int, warmup_steps: int, base_lr: float, min_lr: float) -> float:
    return base_lr


LR_SCHEDULES = {
    "cosine": cosine_lr,
    "constant": constant_lr,
}


def infinite(loader: DataLoader):
    while True:
        yield from loader


def build_param_groups(model: nn.Module, weight_decay: float) -> list[dict]:
    """Split params: decay for weights (>=2D), no decay for biases and norm scales."""
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim < 2 or name.endswith(".bias"):
            no_decay.append(p)
        else:
            decay.append(p)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


_IGNORE = {
    "save_every", "wandb", "wandb_project", "run_name",
    "num_workers", "log_every", "output_dir", "train_bin",
    "tokenizer", "config", "command", "seed", "eval",
}

_ABBREV = {
    "vocab_size": "v", "hidden_size": "h", "num_hidden_layers": "l",
    "num_attention_heads": "a", "intermediate_size": "ff",
    "max_position_embeddings": "pos", "dropout": "do",
    "batch_size": "bs", "warmup_steps": "wu", "lr_schedule": "sched",
    "lr": "lr", "min_lr_ratio": "mlr", "weight_decay": "wd",
    "grad_clip": "gc", "grad_accum": "ga", "mask_prob": "mp",
    "hybrid_numerator": "hn", "hybrid_denominator": "hd", "seed": "seed",
}


def _run_name(args: argparse.Namespace) -> str:
    return "_".join(
        f"{_ABBREV.get(k, k)}{v}"
        for k, v in vars(args).items()
        if k not in _IGNORE
    )


def add_pretrain_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", type=str, default="configs/small.json")
    p.add_argument("--tokenizer", type=str, default="models/tokenizer.json")
    p.add_argument("--train-bin", type=str, default="data/train.bin")
    p.add_argument("--output-dir", type=str, default="checkpoints/")
    p.add_argument("--batch-size", type=int, default=64)

    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--min-lr-ratio", type=float, default=0.1)
    p.add_argument("--lr-schedule", type=str, default="cosine", choices=sorted(LR_SCHEDULES))
    p.add_argument("--warmup-steps", type=int, default=300)

    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--mask-prob", type=float, default=0.15)
    # Default mix matches the GPT-BERT "causal-focus" baseline: ~6.25% MLM, 93.75% CLM.
    p.add_argument("--hybrid-numerator", type=int, default=1)
    p.add_argument("--hybrid-denominator", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--save-every", type=int, default=1_000)
    p.add_argument("--max-epochs", type=int, default=10)
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb-project", type=str, default="babylm")
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--eval", type=str, default="none", choices=("none", "fast", "full"),
                   help="run zero-shot eval on the final checkpoint (both backends) before finishing wandb")


def _build_fast_tokenizer(tokenizer_path: str) -> PreTrainedTokenizerFast:
    return PreTrainedTokenizerFast(
        tokenizer_file=tokenizer_path,
        unk_token="[UNK]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        mask_token="[MASK]",
        eos_token="[SEP]",
    )


def _save_checkpoint(
        model: GPTBertForCausalLM,
        tokenizer: PreTrainedTokenizerFast,
        out_dir: Path,
        wandb_run_id: str | None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir, safe_serialization=True)
    tokenizer.save_pretrained(out_dir)
    if wandb_run_id:
        (out_dir / "wandb_run_id.txt").write_text(wandb_run_id)


def run_pretrain(args: argparse.Namespace) -> None:
    device = get_device()
    print(f"device: {device}")

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)

    cfg = GPTBertConfig.from_json_file(args.config)
    # Expose both backends to eval pipelines that load via AutoModelForCausalLM
    # or AutoModelForMaskedLM. By default save_pretrained only records the class
    # used to save, so MLM-backend eval would fail otherwise.
    cfg.auto_map = {
        "AutoConfig": "modeling_gptbert.GPTBertConfig",
        "AutoModelForCausalLM": "modeling_gptbert.GPTBertForCausalLM",
        "AutoModelForMaskedLM": "modeling_gptbert.GPTBertForMaskedLM",
    }

    tokenizer = _build_fast_tokenizer(args.tokenizer)
    if len(tokenizer) != cfg.vocab_size:
        raise ValueError(
            f"tokenizer vocab ({len(tokenizer)}) != model vocab ({cfg.vocab_size})"
        )
    mask_id = tokenizer.mask_token_id
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

    model = GPTBertForCausalLM(cfg).to(device)
    print(f"model params: {model.num_parameters():,}")
    assert model.head.weight.data_ptr() == model.tok_emb.weight.data_ptr(), \
        "tok_emb / head weights are not tied"

    opt = torch.optim.AdamW(
        build_param_groups(model, args.weight_decay),
        lr=args.lr,
        betas=(0.9, 0.98),
    )

    logger = build_logger(args.wandb, args.wandb_project, args.run_name)
    logger.update_config({**cfg.to_dict(), **vars(args)})
    wandb_run_id = getattr(getattr(logger, "run", None), "id", None)

    use_amp = device.type == "cuda"
    output_dir = Path(args.output_dir) / _run_name(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    mlm_p = args.hybrid_numerator / args.hybrid_denominator

    # BabyLM 2026 competition rule: max 10 epochs.
    steps_per_epoch = len(dataset) // (args.batch_size * args.grad_accum)
    max_steps = args.max_epochs * steps_per_epoch
    print(f"training for {args.max_epochs} epochs ({max_steps:,} steps), schedule={args.lr_schedule}")

    schedule_fn = LR_SCHEDULES[args.lr_schedule]
    min_lr = args.lr * args.min_lr_ratio

    model.train()
    t_start = time.time()

    for step in range(max_steps):
        is_causal = rng.random() >= mlm_p
        epoch = step // steps_per_epoch if steps_per_epoch > 0 else 0

        lr = schedule_fn(step, max_steps, args.warmup_steps, args.lr, min_lr)

        for g in opt.param_groups:
            g["lr"] = lr

        opt.zero_grad(set_to_none=True)
        accum_loss = 0.0
        for _ in range(args.grad_accum):
            chunks = next(batches).to(device, non_blocking=True)
            if is_causal:
                input_ids, labels = make_clm_pair(chunks)
            else:
                input_ids, labels = apply_mlm_mask(chunks, mask_id, cfg.vocab_size, args.mask_prob)

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                out = model(input_ids, labels=labels, is_causal=is_causal)

            (out.loss / args.grad_accum).backward()
            accum_loss += out.loss.item() / args.grad_accum

        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()

        if step % args.log_every == 0:
            elapsed = time.time() - t_start
            tokens_seen = (step + 1) * args.batch_size * args.grad_accum * cfg.max_position_embeddings
            tag = "clm" if is_causal else "mlm"
            metrics = {
                f"loss/{tag}": accum_loss,
                "lr": lr,
                "grad_norm": float(grad_norm),
                "tokens_per_sec": tokens_seen / elapsed,
                "epoch": epoch,
            }
            logger.log(metrics, step=step)
            print(
                f"step {step:6d} | epoch {epoch} | {tag} | loss {accum_loss:.4f} | lr {lr:.2e} | {metrics['tokens_per_sec']:,.0f} tok/s")

        if step > 0 and step % args.save_every == 0:
            _save_checkpoint(model, tokenizer, output_dir, wandb_run_id)
            print(f"saved {output_dir}")

    _save_checkpoint(model, tokenizer, output_dir, wandb_run_id)
    print(f"saved {output_dir}")

    if args.eval != "none":
        _run_eval_and_log(args.eval, output_dir, logger, step=max_steps)

    logger.finish()


def _run_eval_and_log(mode: str, ckpt_dir: Path, logger, step: int) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    eval_sh = repo_root / "scripts" / "eval.sh"
    for backend in EVAL_BACKENDS:
        print(f"[eval] running {mode} {backend}")
        rc = subprocess.run([str(eval_sh), str(ckpt_dir), mode, backend]).returncode
        if rc != 0:
            print(f"[eval] {mode} {backend} exited with code {rc}")

    results_root = repo_root / "eval" / "strict" / "results"
    results_dir = model_results_dir(results_root, ckpt_dir.name)
    metrics = parse_eval_results(results_dir)
    if not metrics:
        print(f"[eval] no metrics parsed from {results_dir}")
        return
    print(f"[eval] logging {len(metrics)} metrics to wandb")
    logger.log(metrics, step=step)
    logger.update_summary(metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_pretrain_args(parser)
    run_pretrain(parser.parse_args())
