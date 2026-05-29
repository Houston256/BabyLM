import argparse
import copy
import json
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

from BabyLM.dataset import BPEDropoutDataset, DocumentSegmentDataset, PackedTokenDataset, apply_mlm_mask, apply_mntp_mask, make_clm_pair
from BabyLM.eval_report import model_results_dir, parse_eval_results
from BabyLM.logger import build_logger
from BabyLM.modeling_gptbert import GPTBertConfig, GPTBertForCausalLM
from BabyLM.optim.lamb import Lamb


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


def warmup_cosine_cooldown(step: int, max_steps: int, warmup_steps: int, base_lr: float, min_lr: float,
                           cooldown_steps: int = 0) -> float:
    """Official GPT-BERT schedule (utils.cosine_schedule_with_warmup_cooldown): linear warmup,
    cosine decay to min_lr, then a linear cooldown from min_lr to 0 over the final window."""
    if step < warmup_steps:
        return base_lr * step / max(1, warmup_steps)
    if cooldown_steps > 0 and step >= max_steps - cooldown_steps:
        return min_lr * (max_steps - step) / max(1, cooldown_steps)
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    return max(min_lr, min_lr + (base_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * progress)))


LR_SCHEDULES = {
    "cosine": cosine_lr,
    "constant": constant_lr,
    "warmup_cosine_cooldown": warmup_cosine_cooldown,
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


def z_loss_term(logits: torch.Tensor, labels: torch.Tensor, is_causal: bool) -> torch.Tensor:
    """Mean squared log-partition (logsumexp) over the supervised positions, matching the
    official GPT-BERT z-loss. Selects the same positions the model's cross-entropy uses:
    the CLM left-shift for causal, else the labels!=-100 mask."""
    if is_causal:
        logits = logits[:, :-1, :]
        labels = labels[:, 1:]
    logits = logits.reshape(-1, logits.size(-1))
    sel = labels.reshape(-1) != -100
    if not bool(sel.any()):
        return logits.new_zeros(())
    lse = torch.logsumexp(logits[sel].float(), dim=-1)
    return (lse ** 2).mean()


@torch.no_grad()
def ema_update(ema_model: nn.Module, model: nn.Module, decay: float) -> None:
    for ep, p in zip(ema_model.parameters(), model.parameters()):
        ep.mul_(decay).add_(p.detach(), alpha=1.0 - decay)
    # buffers (e.g. position_indices, rope caches) tracked as a straight copy
    for eb, b in zip(ema_model.buffers(), model.buffers()):
        eb.copy_(b)


def curriculum_seq_len(step: int, max_steps: int, base_seq_len: int, max_seq_len: int) -> int:
    """Official seq-length curriculum: base for the first 70% of training, 2x until 90%, 4x after."""
    progress = (step + 1) / max(1, max_steps)
    if progress >= 0.9:
        factor = 4
    elif progress >= 0.7:
        factor = 2
    else:
        factor = 1
    return min(base_seq_len * factor, max_seq_len)


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
    "head_type": "head", "norm_style": "norm", "attn_gate": "ag",
    "position_bucket_size": "pb", "layer_mixing": "mix",
    "hc_expansion_rate": "hcn",
}


def _run_name(args: argparse.Namespace) -> str:
    pieces = [f"{_ABBREV[k]}{getattr(args, k)}" for k in _ABBREV if hasattr(args, k)]
    base = "_".join(pieces).replace(".", "p")
    # Keep the whole dir name (base + "__" + 8-hex suffix) under the 255-byte filename limit;
    # the random suffix still guarantees uniqueness after truncation.
    base = base[:240]
    return f"{base}__{secrets.token_hex(4)}"


def add_pretrain_args(p: argparse.ArgumentParser) -> None:
    # Defaults reproduce our best baseline run ("repro-rawdata-flat": official recipe + official
    # tokenizer + raw data, flat packing). Override any field on the CLI.
    p.add_argument("--tokenizer", type=str, default="models/gpt-bert-official.json")
    p.add_argument("--train-bin", type=str, default="data/train.bin")
    p.add_argument("--output-dir", type=str, default="checkpoints/")
    p.add_argument("--batch-size", type=int, default=64)
    # learning schedule
    p.add_argument("--lr", type=float, default=7e-3)
    p.add_argument("--min-lr-ratio", type=float, default=0.1)
    p.add_argument("--lr-schedule", type=str, default="warmup_cosine_cooldown", choices=sorted(LR_SCHEDULES))
    p.add_argument("--warmup-steps", type=int, default=300)

    p.add_argument("--optimizer", type=str, default="lamb", choices=["adamw", "muon", "lamb"], help="Optimizer to use")
    p.add_argument("--muon-lr", type=float, default=0.02, help="Base learning rate for Muon optimizer")
    p.add_argument("--optimizer-eps", type=float, default=1e-8, help="Epsilon for adamw/lamb")
    p.add_argument("--cooldown-ratio", type=float, default=0.016,
                   help="for --lr-schedule warmup_cosine_cooldown: fraction of steps for the final linear cooldown to 0")
    p.add_argument("--warmup-ratio", type=float, default=0.016,
                   help="if >0, warmup steps = warmup_ratio * max_steps (overrides --warmup-steps)")
    p.add_argument("--label-smoothing", type=float, default=0.0, help="Label smoothing for cross entropy")
    p.add_argument("--z-loss-weight", type=float, default=1e-4,
                   help="weight on the logsumexp^2 z-loss (official GPT-BERT uses 1e-4)")
    p.add_argument("--ema-decay", type=float, default=0.999,
                   help="if >0, keep an EMA of the weights with this decay and save/eval it (official: 0.999)")
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=2.0)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--mask-prob", type=float, default=0.15)
    p.add_argument("--mask-p-start", type=float, default=0.3,
                   help="if >0, linearly anneal mask prob from mask-p-start to mask-p-end over training (official 0.3->0.15)")
    p.add_argument("--mask-p-end", type=float, default=0.15, help="final mask prob for the mask-prob curriculum")
    p.add_argument("--n-special-tokens", type=int, default=16,
                   help="number of reserved special-token ids; random-replacement draws from [n_special_tokens, vocab) (official tokenizer: 16)")
    # step-based training + sequence-length curriculum (official recipe; opt-in via --max-steps>0)
    p.add_argument("--max-steps", type=int, default=9914,
                   help="if >0, train for this many optimizer steps instead of --max-epochs")
    p.add_argument("--tokens-per-batch", type=int, default=16384,
                   help="if >0, hold the optimizer-step token budget constant and run the 128->256->512 seq-len curriculum")
    p.add_argument("--base-seq-len", type=int, default=128, help="starting sequence length for the curriculum")
    p.add_argument("--document-packing", action="store_true",
                   help="train on coherent within-document segments (leading <s> + padding mask), "
                        "equivalent to the official block-diagonal doc packing; needs a doc-separated bin "
                        "(tokenize-corpus --source-mode raw)")
    p.add_argument("--local-batch-size", type=int, default=32,
                   help="micro-batch size (sequences) for the curriculum loop; 0 = use --batch-size")
    p.add_argument("--batch-reduction", type=int, default=4,
                   help="linear batch warmup: start at tokens-per-batch/batch_reduction, grow to full over training")
    p.add_argument("--span-masking", action="store_true", help="Enable Span Masking for MLM/MNTP")
    p.add_argument("--bpe-dropout", type=float, default=0.0, help="Enable BPE Dropout (requires dynamic tokenization from raw text)")
    p.add_argument("--train-raw-dir", type=str, default="data/raw", help="Path to raw txt files (required if --bpe-dropout > 0)")
    # Default mix matches the official GPT-BERT "mixed" baseline: 50% MLM / 50% CLM.
    p.add_argument("--hybrid-numerator", type=int, default=1)
    p.add_argument("--hybrid-denominator", type=int, default=2)
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
    p.add_argument("--vocab-size", type=int, default=16384)
    p.add_argument("--hidden-size", type=int, default=384)
    p.add_argument("--num-hidden-layers", type=int, default=12)
    p.add_argument("--num-attention-heads", type=int, default=6)
    p.add_argument("--intermediate-size", type=int, default=1280)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--attn-dropout", type=float, default=0.1)
    p.add_argument("--layer-norm-eps", type=float, default=1e-7)
    # positional embeddings
    p.add_argument("--pos-emb", type=str, default="disentangled", choices=["rope", "absolute", "none", "disentangled"])
    p.add_argument("--max-position-embeddings", type=int, default=512)
    # rope
    p.add_argument("--rope-base", type=float, default=10000.0)
    p.add_argument("--rope-partial-factor", type=float, default=1.0)
    # disentangled (DeBERTa-style) relative attention
    p.add_argument("--position-bucket-size", type=int, default=32,
                   help="log-bucket size for --pos-emb disentangled")
    # attention / mlp / init
    p.add_argument("--attn-gate", action="store_true",
                   help="value gating on the SDPA attention path (intrinsic to disentangled)")
    p.add_argument("--mlp-type", type=str, default="geglu", choices=["gelu", "swiglu", "geglu"])
    p.add_argument("--norm-style", type=str, default="ltg", choices=["pre", "sandwich", "ltg"],
                   help="sandwich = pre+post LayerNorm around each sub-block; "
                        "ltg = official GPT-BERT scheme (embedding norm, non-affine pre-norms, "
                        "attention norm before out_proj, mid-FFN norm)")
    p.add_argument("--head-type", type=str, default="deep", choices=["linear", "deep"],
                   help="deep = LN->Linear->GELU->LN->Dropout->Linear (tied), like the GPT-BERT MaskClassifier")
    # layer mixing (residual topology)
    p.add_argument("--layer-mixing", type=str, default="dwa", choices=["none", "dwa", "mhc"],
                   help="dwa = DenseFormer depth-mixing; mhc = manifold-constrained hyper-connections")
    p.add_argument("--hc-expansion-rate", type=int, default=2, help="number of residual streams for --layer-mixing mhc")
    p.add_argument("--sinkhorn-iters", type=int, default=20, help="Sinkhorn-Knopp iterations for mhc res mapping")
    p.add_argument("--init-scheme", type=str, default="ltg",
                   choices=["small", "gpt2", "xavier", "kaiming", "ltg"],
                   help="small=N(0,0.02); gpt2=small + 1/sqrt(2N) residual scaling; "
                        "ltg=official GPT-BERT trunc_normal(sqrt(2/5H)) + depth-scaled FFN; "
                        "xavier/kaiming applied to Linear only")
    p.add_argument("--mlm-style", type=str, default="mntp", choices=["mlm", "mntp"],
                   help="mlm=predict masked tokens at their own position (standard BERT); "
                        "mntp=predict at position k-1 (GPT-BERT — same alignment as CLM, lets one head serve both)")


def _build_fast_tokenizer(tokenizer_path: str) -> PreTrainedTokenizerFast:
    # Two special-token conventions are supported: our own BPE ([UNK]/[CLS]/[SEP]/[PAD]/[MASK])
    # and the official GPT-BERT tokenizer (<unk>/<s>/</s>/<pad>/<mask>). Detect by vocab.
    vocab = json.loads(Path(tokenizer_path).read_text())["model"]["vocab"]
    if "<s>" in vocab:
        names = dict(unk_token="<unk>", cls_token="<s>", sep_token="</s>",
                     pad_token="<pad>", mask_token="<mask>", bos_token="<s>", eos_token="</s>")
    else:
        names = dict(unk_token="[UNK]", cls_token="[CLS]", sep_token="[SEP]",
                     pad_token="[PAD]", mask_token="[MASK]", eos_token="[SEP]")
    return PreTrainedTokenizerFast(tokenizer_file=tokenizer_path, **names)


def _save_checkpoint(
        model: GPTBertForCausalLM,
        tokenizer: PreTrainedTokenizerFast,
        out_dir: Path,
        wandb_run_id: str | None,
        raw_model: GPTBertForCausalLM | None = None,
) -> None:
    # `model` is what gets evaluated (the EMA weights when EMA is enabled). When an EMA model is
    # the primary checkpoint, the live ("raw") weights are saved as a sidecar for fallback eval.
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir, safe_serialization=True)
    tokenizer.save_pretrained(out_dir)
    if raw_model is not None:
        raw_dir = out_dir / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        raw_model.save_pretrained(raw_dir, safe_serialization=True)
        tokenizer.save_pretrained(raw_dir)
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

    bpe = args.bpe_dropout > 0
    if bpe:
        print(f"Using dynamic BPEDropoutDataset (dropout={args.bpe_dropout})")
    doc_packing = args.document_packing
    if doc_packing:
        cls_id, pad_id = tokenizer.cls_token_id, tokenizer.pad_token_id
        sep_id = tokenizer.sep_token_id
        if cls_id is None or sep_id is None or pad_id is None:
            raise ValueError("--document-packing needs cls/sep/pad tokens defined on the tokenizer")
        print(f"Document-segment packing: cls={cls_id} sep={sep_id} pad={pad_id}")

    def build_loader(seq_len: int, batch_size: int):
        if bpe:
            ds = BPEDropoutDataset(
                dataset_name="BabyLM-community/BabyLM-2026-Strict-Small",
                split="train",
                tokenizer_path=args.tokenizer,
                seq_length=seq_len,
                bpe_dropout=args.bpe_dropout,
            )
        elif doc_packing:
            ds = DocumentSegmentDataset(args.train_bin, seq_len, cls_id, sep_id, pad_id)
        else:
            ds = PackedTokenDataset(args.train_bin, seq_len)
        ld = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            persistent_workers=(args.num_workers > 0 and not bpe),
        )
        return ds, ld

    # Sequence-length curriculum (official recipe) is opt-in via --max-steps>0 and --tokens-per-batch>0.
    use_curriculum = args.max_steps > 0 and args.tokens_per_batch > 0
    base_seq = args.base_seq_len if use_curriculum else args.max_position_embeddings
    micro_bs = args.local_batch_size if (use_curriculum and args.local_batch_size > 0) else args.batch_size
    cur_seq = base_seq
    dataset, loader = build_loader(base_seq, micro_bs)
    print(f"chunks: {len(dataset):,} ({len(dataset) * base_seq:,} tokens) @ seq_len {base_seq}")
    batches = infinite(loader)

    model = GPTBertForCausalLM(cfg).to(device)
    print(f"model params: {model.num_parameters():,}")
    assert model.get_output_embeddings().weight.data_ptr() == model.tok_emb.weight.data_ptr(), \
        "tok_emb / head weights are not tied"

    ema_model = None
    if args.ema_decay > 0:
        ema_model = copy.deepcopy(model)
        for p in ema_model.parameters():
            p.requires_grad_(False)
        ema_model.eval()
        print(f"EMA enabled (decay={args.ema_decay}); EMA weights will be the evaluated checkpoint")

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
    elif args.optimizer == "lamb":
        opts.append(Lamb(
            build_param_groups(model, args.weight_decay),
            lr=args.lr,
            betas=(0.9, 0.98),
            eps=args.optimizer_eps,
        ))
    else:
        opts.append(torch.optim.AdamW(
            build_param_groups(model, args.weight_decay),
            lr=args.lr,
            betas=(0.9, 0.98),
            eps=args.optimizer_eps,
        ))

    logger = build_logger(args.wandb, args.wandb_project, args.run_name)
    logger.update_config({**vars(args), "num_parameters": model.num_parameters()})
    wandb_run_id = getattr(getattr(logger, "run", None), "id", None)

    use_amp = device.type == "cuda"
    output_dir = Path(args.output_dir) / _run_name(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    mlm_p = args.hybrid_numerator / args.hybrid_denominator

    if args.max_steps > 0:
        max_steps = args.max_steps
    else:
        # BabyLM 2026 competition rule: max 10 epochs.
        steps_per_epoch = len(dataset) // (args.batch_size * args.grad_accum)
        max_steps = args.max_epochs * steps_per_epoch
    steps_per_epoch = len(dataset) // (micro_bs * max(1, args.grad_accum))

    warmup_steps = int(max_steps * args.warmup_ratio) if args.warmup_ratio > 0 else args.warmup_steps
    cooldown_steps = int(max_steps * args.cooldown_ratio)
    schedule_fn = LR_SCHEDULES[args.lr_schedule]
    print(f"training for {max_steps:,} steps, schedule={args.lr_schedule}, warmup={warmup_steps}, "
          f"cooldown={cooldown_steps}, curriculum={use_curriculum}")

    def lr_mult_at(step: int) -> float:
        if args.lr_schedule == "warmup_cosine_cooldown":
            return warmup_cosine_cooldown(step, max_steps, warmup_steps, 1.0, args.min_lr_ratio, cooldown_steps)
        return schedule_fn(step, max_steps, warmup_steps, 1.0, args.min_lr_ratio)

    def mask_p_at(step: int) -> float:
        if args.mask_p_start > 0:
            return args.mask_p_start + (args.mask_p_end - args.mask_p_start) * step / max(1, max_steps)
        return args.mask_prob

    model.train()
    t_start = time.time()
    tokens_seen = 0

    for step in range(max_steps):
        # sequence-length curriculum: rebuild the loader at each phase boundary
        if use_curriculum:
            seq_len = curriculum_seq_len(step, max_steps, args.base_seq_len, args.max_position_embeddings)
            if seq_len != cur_seq:
                dataset, loader = build_loader(seq_len, micro_bs)
                batches = infinite(loader)
                cur_seq = seq_len
                print(f"[curriculum] step {step}: seq_len -> {seq_len}")
            # hold the token budget ~constant; optional linear batch warmup (--batch-reduction)
            warm = 1.0
            if args.batch_reduction > 1:
                warm = (1.0 / args.batch_reduction) * (1 - step / max_steps) + (step / max_steps)
            target_seqs = max(1, round(args.tokens_per_batch * warm / seq_len))
            grad_accum = max(1, round(target_seqs / micro_bs))
        else:
            seq_len = cur_seq
            grad_accum = args.grad_accum

        is_causal = rng.random() >= mlm_p
        epoch = step // steps_per_epoch if steps_per_epoch > 0 else 0
        mp = mask_p_at(step)

        lr_mult = lr_mult_at(step)
        for opt in opts:
            base_lr = args.muon_lr if isinstance(opt, torch.optim.Muon) else args.lr
            for g in opt.param_groups:
                g["lr"] = base_lr * lr_mult
            opt.zero_grad(set_to_none=True)

        accum_loss = 0.0
        for _ in range(grad_accum):
            batch = next(batches)
            if doc_packing:
                chunks, seg_id, valid = (t.to(device, non_blocking=True) for t in batch)
                # block-diagonal mask: position i attends j iff same document (pad seg_id=-1 -> isolated)
                attn_mask = seg_id.unsqueeze(2) == seg_id.unsqueeze(1)  # (B,T,T)
                valid = valid.bool()
            else:
                chunks = batch.to(device, non_blocking=True)
                attn_mask, valid = None, None
            tokens_seen += chunks.numel()
            if is_causal:
                input_ids, labels = make_clm_pair(chunks)
                if valid is not None:
                    labels = labels.masked_fill(~valid, -100)
            elif args.mlm_style == "mntp":
                input_ids, labels = apply_mntp_mask(chunks, mask_id, args.vocab_size, mp, args.span_masking, args.n_special_tokens, valid)
            else:
                input_ids, labels = apply_mlm_mask(chunks, mask_id, args.vocab_size, mp, args.span_masking, args.n_special_tokens, valid)

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                out = model(input_ids, labels=labels, attention_mask=attn_mask, is_causal=is_causal)
                loss = out.loss
                if args.z_loss_weight > 0:
                    loss = loss + args.z_loss_weight * z_loss_term(out.logits, labels, is_causal)

            (loss / grad_accum).backward()
            accum_loss += out.loss.item() / grad_accum

        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        if math.isnan(accum_loss) or math.isnan(float(grad_norm)):
            print(f"[warn] NaN at step {step}, skipping update")
            for opt in opts: opt.zero_grad(set_to_none=True)
            continue
        for opt in opts: opt.step()
        if ema_model is not None:
            ema_update(ema_model, model, args.ema_decay)

        if step % args.log_every == 0:
            elapsed = time.time() - t_start
            tag = "clm" if is_causal else "mlm"
            reported_lr = opts[-1].param_groups[0]["lr"]
            metrics = {
                f"loss/{tag}": accum_loss,
                "lr": reported_lr,
                "grad_norm": float(grad_norm),
                "tokens_per_sec": tokens_seen / elapsed,
                "tokens_seen": tokens_seen,
                "epoch": epoch,
                "seq_len": seq_len,
                "grad_accum": grad_accum,
                "mask_p": mp,
            }
            logger.log(metrics, step=step)
            print(
                f"step {step:6d} | epoch {epoch} | {tag} | loss {accum_loss:.4f} | lr {reported_lr:.2e} | "
                f"seq {seq_len} | ga {grad_accum} | {metrics['tokens_per_sec']:,.0f} tok/s")

        if step > 0 and step % args.save_every == 0:
            _save_checkpoint(model if ema_model is None else ema_model, tokenizer, output_dir,
                             wandb_run_id, raw_model=None if ema_model is None else model)
            print(f"saved {output_dir}")

    _save_checkpoint(model if ema_model is None else ema_model, tokenizer, output_dir,
                     wandb_run_id, raw_model=None if ema_model is None else model)
    print(f"saved {output_dir}")

    if args.eval != "none":
        # eval the CLM half + whichever masked objective we trained
        _run_eval_and_log(args.eval, output_dir, logger, step=max_steps, backends=("causal", args.mlm_style))

    logger.finish()


def _run_eval_and_log(mode: str, ckpt_dir: Path, logger, step: int, backends: tuple[str, ...]) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    eval_sh = repo_root / "scripts" / "eval.sh"
    for backend in backends:
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
