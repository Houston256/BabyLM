import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutput, MaskedLMOutput


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

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        # x: (batch, seq, hidden)
        B, T, C = x.shape
        h = self.norm1(x)

        # 1. Generate Q, K, V
        qkv = self.qkv(h).reshape(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(2)  # split into 3 tensors of (B, T, H, D)

        # 2. Transpose for PyTorch's Scaled Dot Product Attention (SDPA)
        # SDPA expects (B, H, T, D)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        # 3. Compute Attention
        # attn_mask is a boolean mask where True means "can attend" and False means "ignore"
        p = self.dropout_p if self.training else 0.0
        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=p)

        # 4. Reshape back to (B, T, C)
        attn = attn.transpose(1, 2).reshape(B, T, C)

        # 5. Residual connections and MLP
        x = x + F.dropout(self.proj(attn), p=p, training=self.training)
        x = x + F.dropout(self.mlp(self.norm2(x)), p=p, training=self.training)
        return x


class _GPTBertBase(PreTrainedModel):
    config_class = GPTBertConfig
    base_model_prefix = "gptbert"
    _tied_weights_keys = {"head.weight": "tok_emb.weight"}

    def __init__(self, config: GPTBertConfig):
        super().__init__(config)
        self.tok_emb = nn.Embedding(config.vocab_size, config.hidden_size)
        self.pos_emb = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.drop = nn.Dropout(config.dropout)
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
        x = self.drop(self.tok_emb(input_ids) + self.pos_emb(pos))

        # --- MASK LOGIC ---
        # 1. Padding Mask: HF provides attention_mask as (B, T) with 1 for valid, 0 for pad.
        # We convert it to a boolean mask (B, 1, 1, T) so it broadcasts over heads and queries.
        # Convention: True = "can attend", False = "ignore" (matches SDPA's bool mask semantics).
        final_mask = None
        if attention_mask is not None:
            final_mask = attention_mask.bool().view(B, 1, 1, T)

        # 2. Causal Mask: If in GPT mode, we add a triangular mask (T, T)
        if is_causal:
            causal_mask = torch.tril(torch.ones(T, T, device=input_ids.device, dtype=torch.bool))
            if final_mask is not None:
                final_mask = final_mask & causal_mask  # Token must be both non-pad AND in the past
            else:
                final_mask = causal_mask

        # Run through transformer blocks
        for block in self.blocks:
            x = block(x, attn_mask=final_mask)

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