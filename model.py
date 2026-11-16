"""A small decoder-only transformer and a character tokenizer, in plain PyTorch."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

CHARS = list("0123456789+-=")
PAD, BOS, EOS = 0, 1, 2
STOI = {c: i + 3 for i, c in enumerate(CHARS)}
ITOS = {i: c for c, i in STOI.items()}
VOCAB_SIZE = len(CHARS) + 3


def encode(text: str) -> list[int]:
    return [STOI[c] for c in text]


def decode(ids: list[int]) -> str:
    """Decode up to (not including) the first EOS; PAD and BOS are dropped."""
    out = []
    for i in ids:
        if i == EOS:
            break
        if i in ITOS:
            out.append(ITOS[i])
    return "".join(out)


class Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.ln1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model)
        )

    def attention(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        # (B, heads, T, head_dim)
        q, k, v = (t.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2) for t in (q, k, v))
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(k.size(-1))
        causal = torch.ones(T, T, dtype=torch.bool, device=x.device).tril()
        scores = scores.masked_fill(~causal, float("-inf"))
        out = F.softmax(scores, dim=-1) @ v
        return self.proj(out.transpose(1, 2).contiguous().view(B, T, C))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class TinyTransformer(nn.Module):
    def __init__(self, d_model: int = 96, n_layers: int = 3, n_heads: int = 4, max_len: int = 16):
        super().__init__()
        self.max_len = max_len
        self.tok = nn.Embedding(VOCAB_SIZE, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList([Block(d_model, n_heads) for _ in range(n_layers)])
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, VOCAB_SIZE, bias=False)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """ids: (B, T) -> logits (B, T, vocab)."""
        T = ids.size(1)
        x = self.tok(ids) + self.pos(torch.arange(T, device=ids.device))
        for block in self.blocks:
            x = block(x)
        return self.head(self.ln(x))
