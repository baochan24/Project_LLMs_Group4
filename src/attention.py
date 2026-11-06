
"""Causal multi-head attention written explicitly for learning."""

import math

import torch
from torch import nn


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, context_length, dropout):
        super().__init__()

        if type(d_model) is not int or d_model <= 0:
            raise ValueError("d_model must be a positive integer")

        if type(n_heads) is not int or n_heads <= 0:
            raise ValueError("n_heads must be a positive integer")

        if type(context_length) is not int or context_length <= 0:
            raise ValueError("context_length must be a positive integer")

        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")

        if not isinstance(dropout, (int, float)) or isinstance(dropout, bool):
            raise ValueError("dropout must be a number")

        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")

        self.d_model = d_model
        self.n_heads = n_heads
        self.context_length = context_length
        self.head_dim = d_model // n_heads

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=True)
        self.proj = nn.Linear(d_model, d_model, bias=True)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        self.register_buffer(
            "mask",
            torch.ones(
                context_length,
                context_length,
                dtype=torch.bool
            ).tril(),
            persistent=False
        )

    def forward(self, x, return_weights=False):
        if not isinstance(x, torch.Tensor):
            raise TypeError("x must be a torch.Tensor")

        if x.ndim != 3:
            raise ValueError("Expected input shape [B, T, D]")

        b, t, d = x.shape

        if b <= 0 or t <= 0:
            raise ValueError("Batch and sequence dimensions must be positive")

        if d != self.d_model:
            raise ValueError(
                f"Expected embedding dimension {self.d_model}, got {d}"
            )

        if t > self.context_length:
            raise ValueError("Sequence exceeds context_length")

        if not x.is_floating_point():
            raise TypeError("Attention input must have a floating-point dtype")

        if not isinstance(return_weights, bool):
            raise TypeError("return_weights must be a bool")

        # Project input into Query, Key and Value.
        q, k, v = self.qkv(x).chunk(3, dim=-1)

        # [B, T, D] -> [B, H, T, head_dim]
        q, k, v = [
            z.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
            for z in (q, k, v)
        ]

        # Scaled dot-product attention scores.
        scores = (
            q @ k.transpose(-2, -1)
        ) / math.sqrt(self.head_dim)

        # Prevent each position from attending to future positions.
        weights = scores.masked_fill(
            ~self.mask[:t, :t],
            float("-inf")
        ).softmax(dim=-1)

        # Weighted sum of Value vectors.
        y = self.attn_dropout(weights) @ v

        # Concatenate all heads.
        y = (
            y.transpose(1, 2)
            .contiguous()
            .view(b, t, d)
        )

        # Output projection and residual dropout.
        y = self.resid_dropout(self.proj(y))

        return (y, weights) if return_weights else y
