import argparse
import math
import random
import secrets
import subprocess
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerFast

from BabyLM.dataset import BPEDropoutDataset, PackedTokenDataset, apply_mlm_mask, apply_mntp_mask, make_clm_pair
from BabyLM.eval_report import model_results_dir, parse_eval_results
from BabyLM.logger import build_logger
from BabyLM.modeling_gptbert import GPTBertConfig, GPTBertForCausalLM

EVAL_BACKENDS = ("causal", "mlm", "mntp")


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
        if hasattr(loader.dataset, "refresh"):
            loader.dataset.refresh()
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


# Whitelist of fields that contribute to the checkpoint dir name.
_ABBREV = {
    # CLI args (training hyperparams)
    "batch_size": "bs", "warmup_steps": "wu", "lr_schedule": "sched",
    "lr": "lr", "min_lr_ratio": "mlr", "weight_decay": "wd",
    "grad_clip": "gc", "grad_accum": "ga", "mask_prob": "mp",
    "hybrid_numerator": "hn", "hybrid_denominator": "hd",
    "max_epochs": "max_epochs",
    "seed": "s",
    # architecture
    "vocab_size": "v", "hidden_size": "h", "num_hidden_layers": "l",
    "num_attention_heads": "a", "intermediate_size": "ff",
    "max_position_embeddings": "pos", "dropout": "do",
    "pos_emb": "pos_emb", "rope_base": "rb",
    "rope_partial_factor": "rp", "attn_dropout": "ad",
    "mlp_type": "mlp", "init_scheme": "init",
    "mlm_style": "mlms",
}


def _run_name(args: argparse.Namespace) -> str:
    pieces = [f"{_ABBREV[k]}{getattr(args, k)}" for k in _ABBREV if hasattr(args, k)]
    base = "_".join(pieces).replace(".", "p")
    return f"{base}__{secrets.token_hex(4)}"


def add_pretrain_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--tokenizer", type=str, default="models/tokenizer.json")
    p.add_argument("--train-bin", type=str, default="data/train.bin")
    p.add_argument("--output-dir", type=str, default="checkpoints/")
    p.add_argument("--batch-size", type=int, default=64)
    # learning schedule
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--min-lr-ratio", type=float, default=0.1)
    p.add_argument("--lr-schedule", type=str, default="cosine", choices=sorted(LR_SCHEDULES))
    p.add_argument("--warmup-steps", type=int, default=300)

    p.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "muon"], help="Optimizer to use")
    p.add_argument("--muon-lr", type=float, default=0.02, help="Base learning rate for Muon optimizer")
    p.add_argument("--label-smoothing", type=float, default=0.0, help="Label smoothing for cross entropy")
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--mask-prob", type=float, default=0.15)
    p.add_argument("--span-masking", action="store_true", help="Enable Span Masking for MLM/MNTP")
    p.add_argument("--bpe-dropout", type=float, default=0.0, help="Enable BPE Dropout (requires dynamic tokenization from raw text)")
    p.add_argument("--train-raw-dir", type=str, default="data/raw", help="Path to raw txt files (required if --bpe-dropout > 0)")
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
    # architecture
    p.add_argument("--vocab-size", type=int, default=8192)
    p.add_argument("--hidden-size", type=int, default=384)
    p.add_argument("--num-hidden-layers", type=int, default=12)
    p.add_argument("--num-attention-heads", type=int, default=6)
    p.add_argument("--intermediate-size", type=int, default=1024)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--attn-dropout", type=float, default=0.1)
    # positional embeddings
    p.add_argument("--pos-emb", type=str, default="rope", choices=["rope", "absolute", "none"])
    p.add_argument("--max-position-embeddings", type=int, default=512)
    # rope
    p.add_argument("--rope-base", type=float, default=10000.0)
    p.add_argument("--rope-partial-factor", type=float, default=1.0)
    # mlp / init
    p.add_argument("--mlp-type", type=str, default="gelu", choices=["gelu", "swiglu"])
    p.add_argument("--init-scheme", type=str, default="small",
                   choices=["small", "gpt2", "xavier", "kaiming"],
                   help="small=N(0,0.02); gpt2=small + 1/sqrt(2N) residual scaling; xavier/kaiming applied to Linear only")
    p.add_argument("--mlm-style", type=str, default="mntp", choices=["mlm", "mntp"],
                   help="mlm=predict masked tokens at their own position (standard BERT); "
                        "mntp=predict at position k-1 (GPT-BERT — same alignment as CLM, lets one head serve both)")


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

    cfg = GPTBertConfig(**vars(args))

    # Expose both backends for AutoModel compatibility
    cfg.auto_map = {
        "AutoConfig": "modeling_gptbert.GPTBertConfig",
        "AutoModelForCausalLM": "modeling_gptbert.GPTBertForCausalLM",
        "AutoModelForMaskedLM": "modeling_gptbert.GPTBertForMaskedLM",
    }

    tokenizer = _build_fast_tokenizer(args.tokenizer)
    if len(tokenizer) != args.vocab_size:
        raise ValueError(f"tokenizer vocab ({len(tokenizer)}) != args vocab ({args.vocab_size})")

    mask_id = tokenizer.mask_token_id
    if mask_id is None:
        raise ValueError("tokenizer must define a [MASK] token")

    if args.bpe_dropout > 0:
        print(f"Using dynamic BPEDropoutDataset (dropout={args.bpe_dropout})")
        dataset = BPEDropoutDataset(
            dataset_name="BabyLM-community/BabyLM-2026-Strict-Small", 
            split="train", 
            tokenizer_path=args.tokenizer, 
            seq_length=args.max_position_embeddings, 
            bpe_dropout=args.bpe_dropout
        )
    else:
        dataset = PackedTokenDataset(args.train_bin, args.max_position_embeddings)
    
    print(f"chunks: {len(dataset):,} ({len(dataset) * args.max_position_embeddings:,} tokens)")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=(args.num_workers > 0 and args.bpe_dropout == 0.0),
    )
    batches = infinite(loader)

    model = GPTBertForCausalLM(cfg).to(device)
    print(f"model params: {model.num_parameters():,}")
    assert model.head.weight.data_ptr() == model.tok_emb.weight.data_ptr(), \
        "tok_emb / head weights are not tied"

    opts = []
    if args.optimizer == "muon":
        muon_params, adamw_decay, adamw_no_decay = [], [], []
        for name, p in model.named_parameters():
            if not p.requires_grad: continue
            if p.ndim >= 2 and "tok_emb" not in name and "abs_pos_emb" not in name and "head" not in name:
                muon_params.append(p)
            elif p.ndim < 2 or name.endswith(".bias"):
                adamw_no_decay.append(p)
            else:
                adamw_decay.append(p)

        opts.append(torch.optim.Muon(muon_params, lr=args.muon_lr, momentum=0.95))
        opts.append(torch.optim.AdamW([
            {"params": adamw_decay, "weight_decay": args.weight_decay},
            {"params": adamw_no_decay, "weight_decay": 0.0}
        ], lr=args.lr, betas=(0.9, 0.95)))
    else:
        opts.append(torch.optim.AdamW(
            build_param_groups(model, args.weight_decay),
            lr=args.lr,
            betas=(0.9, 0.98),
        ))

    logger = build_logger(args.wandb, args.wandb_project, args.run_name)
    logger.update_config({**vars(args), "num_parameters": model.num_parameters()})
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

        lr_mult = schedule_fn(step, max_steps, args.warmup_steps, 1.0, args.min_lr_ratio)

        for opt in opts:
            base_lr = args.muon_lr if isinstance(opt, torch.optim.Muon) else args.lr
            for g in opt.param_groups:
                g["lr"] = base_lr * lr_mult
            opt.zero_grad(set_to_none=True)

        accum_loss = 0.0
        for _ in range(args.grad_accum):
            chunks = next(batches).to(device, non_blocking=True)
            if is_causal:
                input_ids, labels = make_clm_pair(chunks)
            elif args.mlm_style == "mntp":
                input_ids, labels = apply_mntp_mask(chunks, mask_id, args.vocab_size, args.mask_prob, args.span_masking)
            else:
                input_ids, labels = apply_mlm_mask(chunks, mask_id, args.vocab_size, args.mask_prob, args.span_masking)

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                out = model(input_ids, labels=labels, is_causal=is_causal)

            (out.loss / args.grad_accum).backward()
            accum_loss += out.loss.item() / args.grad_accum

        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        if math.isnan(accum_loss) or math.isnan(float(grad_norm)):
            print(f"[warn] NaN at step {step}, skipping update")
            for opt in opts: opt.zero_grad(set_to_none=True)
            continue
        for opt in opts: opt.step()

        if step % args.log_every == 0:
            elapsed = time.time() - t_start
            tokens_seen = (step + 1) * args.batch_size * args.grad_accum * args.max_position_embeddings
            tag = "clm" if is_causal else "mlm"
            reported_lr = opts[-1].param_groups[0]["lr"]
            metrics = {
                f"loss/{tag}": accum_loss,
                "lr": reported_lr,
                "grad_norm": float(grad_norm),
                "tokens_per_sec": tokens_seen / elapsed,
                "tokens_seen": tokens_seen,
                "epoch": epoch,
            }
            logger.log(metrics, step=step)
            print(
                f"step {step:6d} | epoch {epoch} | {tag} | loss {accum_loss:.4f} | lr {reported_lr:.2e} | {metrics['tokens_per_sec']:,.0f} tok/s")

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
