import torch
import torch.nn as nn
import torch.nn.functional as F

from BabyLM.config import ModelConfig


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
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

    def forward(self, x: torch.Tensor, is_causal: bool) -> torch.Tensor:
        # x: (batch, seq, hidden)
        B, T, C = x.shape
        h = self.norm1(x)
        q, k, v = (
            self.qkv(h)                                    # (batch, seq, 3*hidden)
            .reshape(B, T, 3, self.n_heads, self.head_dim) # (batch, seq, 3, heads, head_dim)
            .permute(2, 0, 3, 1, 4)                        # (3, batch, heads, seq, head_dim)
            .unbind(0)                                     # 3x (batch, heads, seq, head_dim)
        )
        p = self.dropout_p if self.training else 0.0
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal, dropout_p=p)  # (batch, heads, seq, head_dim)
        attn = attn.transpose(1, 2).reshape(B, T, C)       # (batch, seq, hidden)

        x = x + F.dropout(self.proj(attn), p, self.training)
        x = x + F.dropout(self.mlp(self.norm2(x)), p, self.training)
        return x


class GPTBert(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.pos_emb = nn.Embedding(cfg.max_position_embeddings, cfg.hidden_size)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.num_hidden_layers)])
        self.norm = nn.LayerNorm(cfg.hidden_size)
        self.head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        is_causal: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # input_ids, labels: (batch, seq)
        B, T = input_ids.shape
        assert T <= self.cfg.max_position_embeddings

        pos = torch.arange(T, device=input_ids.device)             # (seq,)
        x = self.drop(self.tok_emb(input_ids) + self.pos_emb(pos)) # (batch, seq, hidden)
        for block in self.blocks:
            x = block(x, is_causal=is_causal)
        x = self.norm(x)
        logits = self.head(x)                                      # (batch, seq, vocab)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.flatten(0, 1),  # (batch*seq, vocab)
                labels.flatten(),      # (batch*seq,)
                ignore_index=-100,
            )
        return logits, loss
