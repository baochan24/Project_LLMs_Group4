"""Causal multi-head attention viết tường minh để học cơ chế (TV3)."""
import math
import torch
from torch import nn


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, context_length, dropout):
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=True)
        self.proj = nn.Linear(d_model, d_model, bias=True)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.register_buffer("mask", torch.ones(context_length, context_length, dtype=torch.bool).tril(), persistent=False)

    def forward(self, x, return_weights=False):
        b, t, d = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = [z.view(b, t, self.n_heads, self.head_dim).transpose(1, 2) for z in (q, k, v)]
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        weights = scores.masked_fill(~self.mask[:t, :t], float("-inf")).softmax(dim=-1)
        y = self.attn_dropout(weights) @ v
        y = y.transpose(1, 2).contiguous().view(b, t, d)
        y = self.resid_dropout(self.proj(y))
        return (y, weights) if return_weights else y
