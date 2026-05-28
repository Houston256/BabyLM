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


class _Block(nn.Module):
    def __init__(self, cfg: GPTBertConfig):
        super().__init__()
        self.n_heads = cfg.num_attention_heads
        self.head_dim = cfg.hidden_size // cfg.num_attention_heads
        self.dropout_p = cfg.dropout
        # attn_dropout is the SDPA dropout; cfg.dropout is for residual/embedding path.
        self.attn_dropout_p = cfg.attn_dropout if cfg.attn_dropout is not None else cfg.dropout

        # Partial-rotary: only rotate the first rot_dim channels of each head.
        # cos/sin tensors are sized to rot_dim by the parent module's RotaryEmbedding,
        # so just slice q/k to match.
        partial_factor = getattr(cfg, "rope_partial_factor", 1.0)
        rot_dim = int(self.head_dim * partial_factor)
        self.rot_dim = rot_dim - (rot_dim % 2)  # must be even

        self.norm1 = nn.LayerNorm(cfg.hidden_size)
        self.qkv = nn.Linear(cfg.hidden_size, 3 * cfg.hidden_size)
        self.proj = nn.Linear(cfg.hidden_size, cfg.hidden_size)

        self.norm2 = nn.LayerNorm(cfg.hidden_size)
        mlp_type = getattr(cfg, "mlp_type", "gelu")
        if mlp_type == "swiglu":
            self.mlp = SwiGLU(cfg.hidden_size, cfg.intermediate_size)
        elif mlp_type == "gelu":
            self.mlp = nn.Sequential(
                nn.Linear(cfg.hidden_size, cfg.intermediate_size),
                nn.GELU(),
                nn.Linear(cfg.intermediate_size, cfg.hidden_size),
            )
        else:
            raise ValueError(f"unknown mlp_type: {mlp_type}")

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None, is_causal: bool = False, rotary_emb_vals: tuple[torch.Tensor, torch.Tensor] | None = None) -> torch.Tensor:
        # x: (batch, seq, hidden)
        B, T, C = x.shape
        h = self.norm1(x)

        # 1. Generate Q, K, V
        qkv = self.qkv(h).reshape(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(2)  # split into 3 tensors of (B, T, H, D)

        # 2. Transpose for PyTorch's Scaled Dot Product Attention (SDPA)
        # SDPA expects (B, H, T, D)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        # Apply RoPE if provided. cos/sin are sized to self.rot_dim by the parent module.
        if rotary_emb_vals is not None:
            cos, sin = rotary_emb_vals
            if self.rot_dim == self.head_dim:
                q, k = apply_rotary_pos_emb(q, k, cos, sin)
            else:
                q, k = apply_partial_rotary_pos_emb(q, k, cos, sin, self.rot_dim)

        # 3. Compute Attention. When attn_mask is None and is_causal=True, SDPA
        # dispatches to the FlashAttention kernel; passing a materialized mask forces math kernel.
        p_res = self.dropout_p if self.training else 0.0
        p_attn = self.attn_dropout_p if self.training else 0.0
        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=p_attn, is_causal=is_causal)

        # 4. Reshape back to (B, T, C)
        attn = attn.transpose(1, 2).reshape(B, T, C)

        # 5. Residual connections and MLP
        x = x + F.dropout(self.proj(attn), p=p_res, training=self.training)
        x = x + F.dropout(self.mlp(self.norm2(x)), p=p_res, training=self.training)
        return x


class _GPTBertBase(PreTrainedModel):
    config_class = GPTBertConfig
    base_model_prefix = "gptbert"
    # transformers v5: dict of {target_param: source_param}. Consumed by
    # PreTrainedModel.get_expanded_tied_weights_keys -> tie_weights().
    _tied_weights_keys = {"head.weight": "tok_emb.weight"}

    def __init__(self, config: GPTBertConfig):
        super().__init__(config)
        self.pos_strategy = config.pos_emb

        self.tok_emb = nn.Embedding(config.vocab_size, config.hidden_size)
        self.drop = nn.Dropout(config.dropout)

        if self.pos_strategy == "absolute":
            self.abs_pos_emb = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        elif self.pos_strategy == "rope":
            head_dim = config.hidden_size // config.num_attention_heads
            partial_factor = getattr(config, "rope_partial_factor", 1.0)
            rot_dim = int(head_dim * partial_factor)
            rot_dim -= rot_dim % 2  # must be even for the half-split rotation
            self.rotary_emb = RotaryEmbedding(
                rot_dim,
                max_position_embeddings=config.max_position_embeddings,
                base=getattr(config, "rope_base", 10000.0)
            )

        self.blocks = nn.ModuleList([_Block(config) for _ in range(config.num_hidden_layers)])
        self.norm = nn.LayerNorm(config.hidden_size)
        self.head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

        # gpt2 scheme also scales residual-stream output projections by 1/sqrt(2N)
        # so residual variance stays bounded with depth. Runs after post_init so it
        # multiplies the freshly initialized weights; from_pretrained then overwrites.
        if getattr(config, "init_scheme", "small") == "gpt2":
            scale = (2 * config.num_hidden_layers) ** -0.5
            for block in self.blocks:
                block.proj.weight.data.mul_(scale)
                mlp_out = block.mlp.down if isinstance(block.mlp, SwiGLU) else block.mlp[2]
                mlp_out.weight.data.mul_(scale)

    def _init_weights(self, module: nn.Module) -> None:
        scheme = getattr(self.config, "init_scheme", "small")
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
        return self.head

    def set_output_embeddings(self, value: nn.Linear) -> None:
        self.head = value

    def _backbone(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None, is_causal: bool) -> torch.Tensor:
        B, T = input_ids.shape

        x = self.tok_emb(input_ids)
        rotary_emb_vals = None
        if self.pos_strategy == "rope":
            rotary_emb_vals = self.rotary_emb(x, seq_len=T)
        elif self.pos_strategy == "absolute":
            pos = torch.arange(T, device=input_ids.device)
            x = x + self.abs_pos_emb(pos)
        x = self.drop(x)

        # --- MASK LOGIC ---
        # Fast path: pure causal with no padding -> hand off to SDPA's `is_causal=True`,
        # which can dispatch to the FlashAttention kernel. Materializing the triangular
        # mask would force the slower math kernel.
        # Convention for the explicit mask: True = "can attend", False = "ignore".
        final_mask = None
        sdpa_is_causal = False
        if attention_mask is None:
            sdpa_is_causal = is_causal
        else:
            final_mask = attention_mask.bool().view(B, 1, 1, T)
            if is_causal:
                causal_mask = torch.tril(torch.ones(T, T, device=input_ids.device, dtype=torch.bool))
                final_mask = final_mask & causal_mask  # non-pad AND in the past

        # Run through transformer blocks
        for block in self.blocks:
            x = block(x, attn_mask=final_mask, is_causal=sdpa_is_causal, rotary_emb_vals=rotary_emb_vals)

        # Final LayerNorm and Head
        return self.head(self.norm(x))


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
                    label_smoothing=getattr(self.config, "label_smoothing", 0.0)
                )
            else:
                # MLM: predict at masked positions (labels are -100 elsewhere), no shift
                loss = F.cross_entropy(
                    logits.flatten(0, 1),
                    labels.flatten(),
                    ignore_index=-100,
                    label_smoothing=getattr(self.config, "label_smoothing", 0.0)
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
                label_smoothing=getattr(self.config, "label_smoothing", 0.0)
            )
        return MaskedLMOutput(loss=loss, logits=logits)


# Register for Hugging Face compatibility
GPTBertConfig.register_for_auto_class()
GPTBertForCausalLM.register_for_auto_class("AutoModelForCausalLM")
GPTBertForMaskedLM.register_for_auto_class("AutoModelForMaskedLM")