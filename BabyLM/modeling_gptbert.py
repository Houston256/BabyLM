import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutput, MaskedLMOutput


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        # persistent=True: these buffers must round-trip through save/load. With
        # persistent=False they were skipped by safetensors and ended up as raw
        # uninitialized memory after transformers v5's meta-device from_pretrained.
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.float32, device=device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=True)
        self._set_cos_sin_cache(seq_len=max_position_embeddings, device=self.inv_freq.device)

    # Cache is always kept in fp32; downcast happens lazily in forward.
    def _set_cos_sin_cache(self, seq_len, device):
        self.max_seq_len_cached = seq_len
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq.to(torch.float32))
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=True)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=True)

    def forward(self, x, seq_len=None):
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device)
        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def apply_partial_rotary_pos_emb(q, k, cos, sin, rot_dim: int):
    # rot_dim is the (even) number of channels rotated; rest pass through.
    q_rot, q_pass = q[..., :rot_dim], q[..., rot_dim:]
    k_rot, k_pass = k[..., :rot_dim], k[..., rot_dim:]
    q_rot = (q_rot * cos) + (rotate_half(q_rot) * sin)
    k_rot = (k_rot * cos) + (rotate_half(k_rot) * sin)
    return torch.cat([q_rot, q_pass], dim=-1), torch.cat([k_rot, k_pass], dim=-1)


class GPTBertConfig(PretrainedConfig):
    model_type = "gptbert"

    def __init__(self, **kwargs):
        # Tie word embeddings is standard for small models to save parameters
        kwargs.setdefault("tie_word_embeddings", True)
        super().__init__(**kwargs)


class SwiGLU(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate = nn.Linear(hidden_size, intermediate_size)
        self.up = nn.Linear(hidden_size, intermediate_size)
        self.down = nn.Linear(intermediate_size, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class GeGLU(nn.Module):
    """Gated-GELU FFN matching the official GPT-BERT FeedForward (bias-free). With
    mid_norm=True it inserts the LTG non-affine LayerNorm after the GeGLU activation."""

    def __init__(self, hidden_size: int, intermediate_size: int, mid_norm: bool = False, eps: float = 1e-5):
        super().__init__()
        self.up = nn.Linear(hidden_size, 2 * intermediate_size, bias=False)
        self.mid_norm = nn.LayerNorm(intermediate_size, eps=eps, elementwise_affine=False) if mid_norm else None
        self.down = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gate = self.up(x).chunk(2, dim=-1)
        h = x * F.gelu(gate, approximate="tanh")
        if self.mid_norm is not None:
            h = self.mid_norm(h)
        return self.down(h)


def _build_mlp(cfg: GPTBertConfig) -> nn.Module:
    mlp_type = cfg.mlp_type
    eps = cfg.layer_norm_eps
    if mlp_type == "swiglu":
        return SwiGLU(cfg.hidden_size, cfg.intermediate_size)
    if mlp_type == "geglu":
        return GeGLU(cfg.hidden_size, cfg.intermediate_size, mid_norm=cfg.norm_style == "ltg", eps=eps)
    if mlp_type == "gelu":
        return nn.Sequential(
            nn.Linear(cfg.hidden_size, cfg.intermediate_size),
            nn.GELU(),
            nn.Linear(cfg.intermediate_size, cfg.hidden_size),
        )
    raise ValueError(f"unknown mlp_type: {mlp_type}")


def _mlp_linears(mlp: nn.Module) -> tuple[nn.Linear, nn.Linear]:
    """(input_proj, output_proj) of an MLP, for residual-stream init scaling."""
    if isinstance(mlp, (SwiGLU, GeGLU)):
        return mlp.up, mlp.down
    return mlp[0], mlp[2]  # gelu Sequential


class SelfAttention(nn.Module):
    """Standard multi-head attention via SDPA. Supports RoPE / absolute / none
    positional strategies (rope vals passed in) and optional value gating. With
    norm_style=ltg, a non-affine LayerNorm is applied before the output projection."""

    def __init__(self, cfg: GPTBertConfig):
        super().__init__()
        self.n_heads = cfg.num_attention_heads
        self.head_dim = cfg.hidden_size // cfg.num_attention_heads
        self.attn_dropout_p = cfg.attn_dropout if cfg.attn_dropout is not None else cfg.dropout
        self.gate = cfg.attn_gate
        self.ltg = cfg.norm_style == "ltg"

        partial_factor = cfg.rope_partial_factor
        rot_dim = int(self.head_dim * partial_factor)
        self.rot_dim = rot_dim - (rot_dim % 2)  # must be even

        self.qkv = nn.Linear(cfg.hidden_size, 3 * cfg.hidden_size)
        if self.gate:
            self.g_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size)
        if self.ltg:
            self.post_norm = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps, elementwise_affine=False)
        self.proj = nn.Linear(cfg.hidden_size, cfg.hidden_size)

    def forward(self, h, attn_mask=None, is_causal=False, rotary_emb_vals=None, training=False):
        B, T, C = h.shape
        qkv = self.qkv(h).reshape(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(2)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if rotary_emb_vals is not None:
            cos, sin = rotary_emb_vals
            if self.rot_dim == self.head_dim:
                q, k = apply_rotary_pos_emb(q, k, cos, sin)
            else:
                q, k = apply_partial_rotary_pos_emb(q, k, cos, sin, self.rot_dim)

        p_attn = self.attn_dropout_p if training else 0.0
        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=p_attn, is_causal=is_causal)
        attn = attn.transpose(1, 2).reshape(B, T, C)
        if self.gate:
            attn = attn * F.gelu(self.g_proj(h))
        if self.ltg:
            attn = self.post_norm(attn)
        return self.proj(attn)


def _make_log_bucket_position(relative_pos: torch.Tensor, bucket_size: int, max_position: int) -> torch.Tensor:
    sign = torch.sign(relative_pos)
    mid = bucket_size // 2
    abs_pos = torch.where(
        (relative_pos < mid) & (relative_pos > -mid),
        torch.full_like(relative_pos, mid - 1),
        torch.abs(relative_pos).clamp(max=max_position - 1),
    )
    log_pos = torch.ceil(torch.log(abs_pos / mid) / math.log((max_position - 1) / mid) * (mid - 1)).int() + mid
    return torch.where(abs_pos <= mid, relative_pos, log_pos * sign).long()


class DisentangledAttention(nn.Module):
    """DeBERTa-style disentangled relative attention with log-bucket positions,
    ported from the official GPT-BERT. The content<->position terms are computed
    against the (2*bucket-1)-entry table and gathered to (B,H,T,T) — avoiding the
    O(T,T,2C) operand — then folded into F.scaled_dot_product_attention as an
    additive bias so the fused kernel handles softmax+V. Value gating is intrinsic."""

    def __init__(self, cfg: GPTBertConfig):
        super().__init__()
        self.n_heads = cfg.num_attention_heads
        self.head_dim = cfg.hidden_size // cfg.num_attention_heads
        self.hidden_size = cfg.hidden_size
        self.bucket_size = cfg.position_bucket_size
        self.max_pos = cfg.max_position_embeddings
        self.scale = 1.0 / math.sqrt(3 * self.head_dim)
        self.attn_dropout_p = cfg.attn_dropout if cfg.attn_dropout is not None else cfg.dropout
        self.ltg = cfg.norm_style == "ltg"

        self.in_proj_qk = nn.Linear(cfg.hidden_size, 2 * cfg.hidden_size)
        self.in_proj_vg = nn.Linear(cfg.hidden_size, 2 * cfg.hidden_size)
        if self.ltg:
            self.post_norm = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps, elementwise_affine=False)
        self.out_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size)

        pos = torch.arange(self.max_pos, dtype=torch.long).unsqueeze(1) - torch.arange(self.max_pos, dtype=torch.long).unsqueeze(0)
        pos = _make_log_bucket_position(pos, self.bucket_size, self.max_pos)
        pos = self.bucket_size - 1 + pos
        self.register_buffer("position_indices", pos, persistent=True)

    def _position_bias(self, qh, kh, rel_emb, T):
        # rel_emb: (2*bucket-1, C). Project once (shared with the content in_proj_qk),
        # split into per-head query/key position tables, score against q/k, then gather.
        H, d, B = self.n_heads, self.head_dim, qh.shape[0]
        pos = self.in_proj_qk(rel_emb).view(-1, 2, H, d)  # (P2, 2, H, d)
        table_q, table_k = pos[:, 0], pos[:, 1]           # (P2, H, d) each
        idx = self.position_indices[:T, :T]               # (T, T) -> bucket id

        c2p = torch.einsum("bhqd,phd->bhqp", qh, table_k)              # (B,H,T,P2)
        c2p = torch.gather(c2p, 3, idx.view(1, 1, T, T).expand(B, H, T, T))
        p2c = torch.einsum("bhkd,phd->bhkp", kh, table_q)              # (B,H,T,P2)
        p2c = torch.gather(p2c, 3, idx.t().reshape(1, 1, T, T).expand(B, H, T, T))
        return (c2p + p2c.transpose(-1, -2)) * self.scale             # (B,H,T,T)

    def forward(self, h, rel_emb, attn_mask=None, is_causal=False, training=False):
        B, T, C = h.shape
        q, k = self.in_proj_qk(h).chunk(2, dim=-1)
        v, g = self.in_proj_vg(h).chunk(2, dim=-1)
        g = F.gelu(g)

        qh = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # (B,H,T,d)
        kh = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        vh = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        bias = self._position_bias(qh, kh, rel_emb, T)  # (B,H,T,T), already scaled
        if attn_mask is not None:
            bias = bias.masked_fill(~attn_mask.bool(), float("-inf"))
        if is_causal:
            causal = torch.ones(T, T, device=h.device, dtype=torch.bool).triu(1).view(1, 1, T, T)
            bias = bias.masked_fill(causal, float("-inf"))

        p_attn = self.attn_dropout_p if training else 0.0
        # SDPA computes softmax(scale * qhᵀkh + bias) vh; pass our DeBERTa scale and the
        # position+mask bias so content-content and position terms share the same scale.
        out = F.scaled_dot_product_attention(qh, kh, vh, attn_mask=bias, scale=self.scale, dropout_p=p_attn)
        out = out.transpose(1, 2).reshape(B, T, C)
        out = out * g
        if self.ltg:
            out = self.post_norm(out)
        return self.out_proj(out)


class _Block(nn.Module):
    def __init__(self, cfg: GPTBertConfig):
        super().__init__()
        self.dropout_p = cfg.dropout
        self.disentangled = cfg.pos_emb == "disentangled"
        self.norm_style = cfg.norm_style
        self.ltg = self.norm_style == "ltg"
        self.sandwich = self.norm_style == "sandwich"
        eps = cfg.layer_norm_eps
        # LTG pre-norms are non-affine (the official model's pre_layer_norm).
        affine = not self.ltg

        self.norm1 = nn.LayerNorm(cfg.hidden_size, eps=eps, elementwise_affine=affine)
        self.attn = DisentangledAttention(cfg) if self.disentangled else SelfAttention(cfg)
        self.norm2 = nn.LayerNorm(cfg.hidden_size, eps=eps, elementwise_affine=affine)
        self.mlp = _build_mlp(cfg)

        if self.sandwich:
            self.post_norm1 = nn.LayerNorm(cfg.hidden_size, eps=eps, elementwise_affine=False)
            self.post_norm2 = nn.LayerNorm(cfg.hidden_size, eps=eps, elementwise_affine=False)

    def output_projections(self) -> list[nn.Linear]:
        """Residual-stream output projections, for the gpt2 1/sqrt(2N) init scale."""
        proj = self.attn.out_proj if self.disentangled else self.attn.proj
        return [proj, _mlp_linears(self.mlp)[1]]

    def attn_sublayer(self, x, attn_mask=None, is_causal=False, rotary_emb_vals=None, rel_emb=None):
        h = self.norm1(x)
        if self.disentangled:
            out = self.attn(h, rel_emb, attn_mask=attn_mask, is_causal=is_causal, training=self.training)
        else:
            out = self.attn(h, attn_mask=attn_mask, is_causal=is_causal, rotary_emb_vals=rotary_emb_vals, training=self.training)
        if self.sandwich:
            out = self.post_norm1(out)
        return F.dropout(out, p=self.dropout_p, training=self.training)

    def mlp_sublayer(self, x):
        out = self.mlp(self.norm2(x))
        if self.sandwich:
            out = self.post_norm2(out)
        return F.dropout(out, p=self.dropout_p, training=self.training)


class _InPlaceSetSlice(torch.autograd.Function):
    """Write x_val into full_tensor[x_idx] without growing the autograd graph with a
    fresh stack each step. From DenseFormer (github.com/epfml/DenseFormer); used by the
    official GPT-BERT so the DWA accumulator stays a single reused buffer."""

    @staticmethod
    def forward(ctx, full_tensor, last_slice, x_idx, x_val):
        full_tensor[x_idx] = x_val
        ctx.x_idx = x_idx
        ret = torch.Tensor().to(full_tensor.device)
        ret.set_(full_tensor[: x_idx + 1])
        return ret

    @staticmethod
    def backward(ctx, grad_out):
        if ctx.x_idx == 0:
            return None, None, None, grad_out[ctx.x_idx]
        return None, grad_out[: ctx.x_idx], None, grad_out[ctx.x_idx]


def _apply_inplace_set(x_acc, x_idx, x_val):
    full_tensor, last_slice = x_acc
    new_slice = _InPlaceSetSlice.apply(full_tensor, last_slice, x_idx, x_val)
    return full_tensor, new_slice


class DWA(nn.Module):
    """DenseFormer Dynamic Weighted Average: each sub-block output is replaced by a
    learned weighted sum over the embedding and all prior sub-block outputs. Uses one
    preallocated accumulator buffer (not a growing stack) to keep memory bounded.
    Initialized to the identity (weight 1.0 on the current output)."""

    def __init__(self, n_mixers: int):
        super().__init__()
        self.n_mixers = n_mixers
        self.alphas = nn.ParameterList([nn.Parameter(torch.zeros(i + 2)) for i in range(n_mixers)])
        for a in self.alphas:
            a.data[-1] = 1.0
        self._acc = None

    def init_accumulator(self, x: torch.Tensor) -> None:
        self._acc = (torch.zeros((self.n_mixers + 1, *x.shape), device=x.device, dtype=x.dtype), None)
        self._acc = _apply_inplace_set(self._acc, 0, x)

    def forward(self, idx: int, x: torch.Tensor) -> torch.Tensor:
        self._acc = _apply_inplace_set(self._acc, idx + 1, x)
        return torch.tensordot(self.alphas[idx], self._acc[1], dims=1)


def _sinkhorn_knopp(logits: torch.Tensor, iters: int) -> torch.Tensor:
    """Project onto the doubly-stochastic (Birkhoff) polytope. logits: (..., n, n)."""
    m = torch.exp(logits - logits.amax(dim=(-2, -1), keepdim=True))
    for _ in range(iters):
        m = m / m.sum(dim=-1, keepdim=True)
        m = m / m.sum(dim=-2, keepdim=True)
    return m


class HyperConnection(nn.Module):
    """Manifold-Constrained Hyper-Connection (mHC, DeepSeek arXiv:2512.24880) for a
    single residual sub-block F. Maintains n parallel streams; the res mapping is
    projected onto the doubly-stochastic manifold via Sinkhorn-Knopp so signal mean
    is conserved and norm is non-expansive. n=1 recovers the standard residual."""

    def __init__(self, hidden_size: int, n: int, sinkhorn_iters: int):
        super().__init__()
        self.n = n
        self.iters = sinkhorn_iters
        self.norm = nn.RMSNorm(n * hidden_size)
        # dynamic (input-dependent) projections; bias folded into the static terms below
        self.phi_pre = nn.Linear(n * hidden_size, n, bias=False)
        self.phi_post = nn.Linear(n * hidden_size, n, bias=False)
        self.phi_res = nn.Linear(n * hidden_size, n * n, bias=False)
        # dynamic gates, init 0 -> at start only the static (identity) mapping is active
        self.alpha_pre = nn.Parameter(torch.zeros(1))
        self.alpha_post = nn.Parameter(torch.zeros(1))
        self.alpha_res = nn.Parameter(torch.zeros(1))
        # static biases: read = mean of streams, write = 1 to every stream, res = identity
        self.b_pre = nn.Parameter(torch.full((n,), math.log(1.0 / max(n - 1, 1)) if n > 1 else 0.0))
        self.b_post = nn.Parameter(torch.zeros(n))
        self.b_res = nn.Parameter(4.0 * torch.eye(n))

    def _coeffs(self, X: torch.Tensor):
        # X: (B, T, n, C)
        xt = self.norm(X.flatten(-2))  # (B, T, nC)
        h_pre = torch.sigmoid(self.alpha_pre * self.phi_pre(xt) + self.b_pre)          # (B,T,n)
        h_post = 2.0 * torch.sigmoid(self.alpha_post * self.phi_post(xt) + self.b_post)  # (B,T,n)
        res_logits = self.alpha_res * self.phi_res(xt).view(*xt.shape[:-1], self.n, self.n) + self.b_res
        h_res = _sinkhorn_knopp(res_logits, self.iters)                                # (B,T,n,n)
        return h_pre, h_post, h_res

    def read(self, X, h_pre):
        return torch.einsum("btn,btnc->btc", h_pre, X)

    def write(self, X, h_res, h_post, delta):
        mixed = torch.einsum("btij,btjc->btic", h_res, X)
        wrote = torch.einsum("btn,btc->btnc", h_post, delta)
        return mixed + wrote


class _GPTBertBase(PreTrainedModel):
    config_class = GPTBertConfig
    base_model_prefix = "gptbert"
    _tied_weights_keys = {"head.weight": "tok_emb.weight"}

    def __init__(self, config: GPTBertConfig):
        super().__init__(config)
        self.pos_strategy = config.pos_emb
        self.layer_mixing = config.layer_mixing
        self.ltg = config.norm_style == "ltg"
        eps = config.layer_norm_eps

        self.tok_emb = nn.Embedding(config.vocab_size, config.hidden_size)
        # LTG normalizes the token embeddings before layer 0 (official GPT-BERT).
        self.emb_norm = nn.LayerNorm(config.hidden_size, eps=eps, elementwise_affine=False) if self.ltg else None
        self.drop = nn.Dropout(config.dropout)

        if self.pos_strategy == "absolute":
            self.abs_pos_emb = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        elif self.pos_strategy == "rope":
            head_dim = config.hidden_size // config.num_attention_heads
            partial_factor = config.rope_partial_factor
            rot_dim = int(head_dim * partial_factor)
            rot_dim -= rot_dim % 2  # must be even for the half-split rotation
            self.rotary_emb = RotaryEmbedding(
                rot_dim,
                max_position_embeddings=config.max_position_embeddings,
                base=config.rope_base,
            )
        elif self.pos_strategy == "disentangled":
            # Relative embedding shared across layers (each block projects it itself).
            bucket = config.position_bucket_size
            self.rel_emb = nn.Parameter(torch.empty(2 * bucket - 1, config.hidden_size))
            nn.init.normal_(self.rel_emb, std=0.02)
            self.rel_norm = nn.LayerNorm(config.hidden_size, eps=eps)

        self.blocks = nn.ModuleList([_Block(config) for _ in range(config.num_hidden_layers)])

        if self.layer_mixing == "dwa":
            self.dwa = DWA(2 * config.num_hidden_layers)
        elif self.layer_mixing == "mhc":
            self.hc_n = config.hc_expansion_rate
            iters = config.sinkhorn_iters
            self.hc = nn.ModuleList(
                [HyperConnection(config.hidden_size, self.hc_n, iters) for _ in range(2 * config.num_hidden_layers)]
            )

        self.norm = nn.LayerNorm(config.hidden_size, eps=eps)
        self._build_head(config, eps)
        self.post_init()
        self._apply_residual_scaling(config)

    def _build_head(self, config: GPTBertConfig, eps: float) -> None:
        if config.head_type == "deep":
            self.head = nn.Sequential(
                nn.LayerNorm(config.hidden_size, eps=eps, elementwise_affine=False),
                nn.Linear(config.hidden_size, config.hidden_size),
                nn.GELU(),
                nn.LayerNorm(config.hidden_size, eps=eps, elementwise_affine=False),
                nn.Dropout(config.dropout),
                # bias matches the official GPT-BERT MaskClassifier (zero-init, trainable per-token bias)
                nn.Linear(config.hidden_size, config.vocab_size, bias=True),
            )
            self._tied_weights_keys = {"head.5.weight": "tok_emb.weight"}
        else:
            self.head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
            self._tied_weights_keys = {"head.weight": "tok_emb.weight"}

    def _apply_residual_scaling(self, config: GPTBertConfig) -> None:
        """Depth scaling of residual-stream output projections. Runs after post_init so
        it multiplies freshly initialized weights; from_pretrained then overwrites."""
        scheme = config.init_scheme
        if scheme == "gpt2":
            # constant 1/sqrt(2N) on attention + mlp output projections (GPT-2 style)
            scale = (2 * config.num_hidden_layers) ** -0.5
            for block in self.blocks:
                for proj in block.output_projections():
                    proj.weight.data.mul_(scale)
        elif scheme == "ltg":
            # depth-dependent 1/sqrt(2(1+l)) on the FFN input AND output projections
            # (official GPT-BERT). Attention projections are left unscaled.
            for i, block in enumerate(self.blocks):
                s = math.sqrt(1.0 / (2.0 * (1 + i)))
                fin, fout = _mlp_linears(block.mlp)
                fin.weight.data.mul_(s)
                fout.weight.data.mul_(s)

    def _init_weights(self, module: nn.Module) -> None:
        scheme = self.config.init_scheme
        if scheme == "ltg":
            std = math.sqrt(2.0 / (5.0 * self.config.hidden_size))
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.trunc_normal_(module.weight, mean=0.0, std=std, a=-2 * std, b=2 * std)
                if isinstance(module, nn.Linear) and module.bias is not None:
                    nn.init.zeros_(module.bias)
            return
        if isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)
        elif isinstance(module, nn.Linear):
            if scheme in ("small", "gpt2"):
                nn.init.normal_(module.weight, std=0.02)
            elif scheme == "xavier":
                nn.init.xavier_normal_(module.weight)
            elif scheme == "kaiming":
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            else:
                raise ValueError(f"unknown init_scheme: {scheme}")
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    # These two are required for HF's tie_weights() to actually tie head <-> tok_emb
    def get_input_embeddings(self) -> nn.Embedding:
        return self.tok_emb

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.tok_emb = value

    def get_output_embeddings(self) -> nn.Linear:
        return self.head[-1] if isinstance(self.head, nn.Sequential) else self.head

    def set_output_embeddings(self, value: nn.Linear) -> None:
        if isinstance(self.head, nn.Sequential):
            self.head[-1] = value
        else:
            self.head = value

    def _attn_mask(self, attention_mask, is_causal, B, T, device):
        # Convention: True = "can attend". Returns (mask_or_None, sdpa_is_causal).
        # attention_mask may be (B,T) padding or (B,T,T) block-diagonal (document packing).
        if attention_mask is None:
            return None, is_causal
        if attention_mask.dim() == 3:
            final_mask = attention_mask.bool().view(B, 1, T, T)
        else:
            final_mask = attention_mask.bool().view(B, 1, 1, T)
        if is_causal:
            causal = torch.tril(torch.ones(T, T, device=device, dtype=torch.bool))
            final_mask = final_mask & causal
        return final_mask, False

    def _backbone(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None, is_causal: bool) -> torch.Tensor:
        B, T = input_ids.shape

        x = self.tok_emb(input_ids)
        if self.emb_norm is not None:
            x = self.emb_norm(x)
        rotary_emb_vals = None
        rel_emb = None
        if self.pos_strategy == "rope":
            rotary_emb_vals = self.rotary_emb(x, seq_len=T)
        elif self.pos_strategy == "absolute":
            pos = torch.arange(T, device=input_ids.device)
            x = x + self.abs_pos_emb(pos)
        elif self.pos_strategy == "disentangled":
            rel_emb = self.rel_norm(self.rel_emb)
        x = self.drop(x)

        final_mask, sdpa_is_causal = self._attn_mask(attention_mask, is_causal, B, T, input_ids.device)
        kw = dict(attn_mask=final_mask, is_causal=sdpa_is_causal, rotary_emb_vals=rotary_emb_vals, rel_emb=rel_emb)

        if self.layer_mixing == "mhc":
            x = self._forward_mhc(x, kw)
        elif self.layer_mixing == "dwa":
            x = self._forward_dwa(x, kw)
        else:
            for block in self.blocks:
                x = x + block.attn_sublayer(x, **kw)
                x = x + block.mlp_sublayer(x)

        return self.head(self.norm(x))

    def _forward_dwa(self, x, kw):
        self.dwa.init_accumulator(x)
        for i, block in enumerate(self.blocks):
            x = self.dwa(2 * i, x + block.attn_sublayer(x, **kw))
            x = self.dwa(2 * i + 1, x + block.mlp_sublayer(x))
        return x

    def _forward_mhc(self, x, kw):
        X = x.unsqueeze(2).expand(-1, -1, self.hc_n, -1).contiguous()  # (B,T,n,C)
        for i, block in enumerate(self.blocks):
            hc = self.hc[2 * i]
            h_pre, h_post, h_res = hc._coeffs(X)
            delta = block.attn_sublayer(hc.read(X, h_pre), **kw)
            X = hc.write(X, h_res, h_post, delta)

            hc = self.hc[2 * i + 1]
            h_pre, h_post, h_res = hc._coeffs(X)
            delta = block.mlp_sublayer(hc.read(X, h_pre))
            X = hc.write(X, h_res, h_post, delta)
        return X.mean(dim=2)


class GPTBertForCausalLM(_GPTBertBase):
    def forward(self, input_ids, labels=None, attention_mask=None, is_causal=True, **kwargs):
        logits = self._backbone(input_ids, attention_mask, is_causal=is_causal)
        loss = None
        if labels is not None:
            if is_causal:
                # CLM: token at position i predicts token at i+1
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()
                loss = F.cross_entropy(
                    shift_logits.flatten(0, 1),
                    shift_labels.flatten(),
                    ignore_index=-100,
                    label_smoothing=self.config.label_smoothing
                )
            else:
                # MLM: predict at masked positions (labels are -100 elsewhere), no shift
                loss = F.cross_entropy(
                    logits.flatten(0, 1),
                    labels.flatten(),
                    ignore_index=-100,
                    label_smoothing=self.config.label_smoothing
                )
        cls = CausalLMOutput if is_causal else MaskedLMOutput
        return cls(loss=loss, logits=logits)

class GPTBertForMaskedLM(_GPTBertBase):
    def forward(self, input_ids, labels=None, attention_mask=None, **kwargs):
        logits = self._backbone(input_ids, attention_mask, is_causal=False)
        loss = None
        if labels is not None:
            # MLM objective (predict the [MASK] tokens). Labels are -100 for non-masked positions.
            loss = F.cross_entropy(
                logits.flatten(0, 1),
                labels.flatten(),
                ignore_index=-100,
                label_smoothing=self.config.label_smoothing
            )
        return MaskedLMOutput(loss=loss, logits=logits)


# Register for Hugging Face compatibility
GPTBertConfig.register_for_auto_class()
GPTBertForCausalLM.register_for_auto_class("AutoModelForCausalLM")
GPTBertForMaskedLM.register_for_auto_class("AutoModelForMaskedLM")
