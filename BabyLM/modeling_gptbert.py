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
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.float32, device=device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._set_cos_sin_cache(seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.float32)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)
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


class GPTBertConfig(PretrainedConfig):
    model_type = "gptbert"

    def __init__(
        self,
        vocab_size: int = 8192,
        hidden_size: int = 384,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 6,
        intermediate_size: int = 1024,
        max_position_embeddings: int = 512,
        dropout: float = 0.1,
        use_rope: bool = True,
        rope_base: float = 10000.0,
        **kwargs,
    ):
        # Tie word embeddings is standard for small models to save parameters
        kwargs.setdefault("tie_word_embeddings", True)
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.dropout = dropout
        self.use_rope = use_rope
        self.rope_base = rope_base


class _Block(nn.Module):
    def __init__(self, cfg: GPTBertConfig):
        super().__init__()
        self.n_heads = cfg.num_attention_heads
        self.head_dim = cfg.hidden_size // cfg.num_attention_heads
        self.dropout_p = cfg.dropout

        self.norm1 = nn.LayerNorm(cfg.hidden_size)
        self.qkv = nn.Linear(cfg.hidden_size, 3 * cfg.hidden_size)
        self.proj = nn.Linear(cfg.hidden_size, cfg.hidden_size)

        self.norm2 = nn.LayerNorm(cfg.hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.hidden_size, cfg.intermediate_size),
            nn.GELU(),
            nn.Linear(cfg.intermediate_size, cfg.hidden_size),
        )

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

        # Apply RoPE if provided
        if rotary_emb_vals is not None:
            cos, sin = rotary_emb_vals
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # 3. Compute Attention. When attn_mask is None and is_causal=True, SDPA
        # dispatches to the FlashAttention kernel; passing a materialized mask forces math kernel.
        p = self.dropout_p if self.training else 0.0
        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=p, is_causal=is_causal)

        # 4. Reshape back to (B, T, C)
        attn = attn.transpose(1, 2).reshape(B, T, C)

        # 5. Residual connections and MLP
        x = x + F.dropout(self.proj(attn), p=p, training=self.training)
        x = x + F.dropout(self.mlp(self.norm2(x)), p=p, training=self.training)
        return x


class _GPTBertBase(PreTrainedModel):
    config_class = GPTBertConfig
    base_model_prefix = "gptbert"
    # transformers v5: dict of {target_param: source_param}. Consumed by
    # PreTrainedModel.get_expanded_tied_weights_keys -> tie_weights().
    _tied_weights_keys = {"head.weight": "tok_emb.weight"}

    def __init__(self, config: GPTBertConfig):
        super().__init__(config)
        self.tok_emb = nn.Embedding(config.vocab_size, config.hidden_size)
        self.pos_emb = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.drop = nn.Dropout(config.dropout)

        self.use_rope = getattr(config, "use_rope", True)
        if self.use_rope:
            head_dim = config.hidden_size // config.num_attention_heads
            self.rotary_emb = RotaryEmbedding(
                head_dim,
                max_position_embeddings=config.max_position_embeddings,
                base=getattr(config, "rope_base", 10000.0)
            )

        self.blocks = nn.ModuleList([_Block(config) for _ in range(config.num_hidden_layers)])
        self.norm = nn.LayerNorm(config.hidden_size)
        self.head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)
            if hasattr(module, "bias") and module.bias is not None:
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
        pos = torch.arange(T, device=input_ids.device)

        x = self.tok_emb(input_ids)
        if not self.use_rope:
            x = x + self.pos_emb(pos)
        x = self.drop(x)

        rotary_emb_vals = None
        if self.use_rope:
            rotary_emb_vals = self.rotary_emb(x, seq_len=T)

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
                )
            else:
                # MLM: predict at masked positions (labels are -100 elsewhere), no shift
                loss = F.cross_entropy(
                    logits.flatten(0, 1),
                    labels.flatten(),
                    ignore_index=-100,
                )
        cls = CausalLMOutput if is_causal else MaskedLMOutput
        return cls(loss=loss, logits=logits)

class GPTBertForMaskedLM(_GPTBertBase):
    def forward(self, input_ids, labels=None, attention_mask=None, **kwargs):
        logits = self._backbone(input_ids, attention_mask, is_causal=False)
        loss = None
        if labels is not None:
            # MLM objective (predict the [MASK] tokens). Labels are -100 for non-masked positions.
            loss = F.cross_entropy(logits.flatten(0, 1), labels.flatten(), ignore_index=-100)
        return MaskedLMOutput(loss=loss, logits=logits)


# Register for Hugging Face compatibility
GPTBertConfig.register_for_auto_class()
GPTBertForCausalLM.register_for_auto_class("AutoModelForCausalLM")
GPTBertForMaskedLM.register_for_auto_class("AutoModelForMaskedLM")