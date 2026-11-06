
"""Pre-LN decoder-only Transformer with learned positions and tied head."""

import torch
from torch import nn

from .attention import CausalSelfAttention


def validate_model_config(vocab_size, cfg):
    """Validate model configuration before creating modules."""

    if type(vocab_size) is not int or vocab_size <= 0:
        raise ValueError("vocab_size must be a positive integer")

    required = (
        "d_model",
        "n_heads",
        "n_layers",
        "context_length",
        "d_ff",
        "dropout",
    )

    missing = [key for key in required if key not in cfg]

    if missing:
        raise ValueError(f"Missing model configuration: {missing}")

    for key in (
        "d_model",
        "n_heads",
        "n_layers",
        "context_length",
        "d_ff",
    ):
        if type(cfg[key]) is not int or cfg[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")

    if cfg["d_model"] % cfg["n_heads"] != 0:
        raise ValueError("d_model must be divisible by n_heads")

    dropout = cfg["dropout"]

    if (
        not isinstance(dropout, (int, float))
        or isinstance(dropout, bool)
        or not 0 <= dropout < 1
    ):
        raise ValueError("dropout must be in [0, 1)")


class DecoderBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        d = cfg["d_model"]

        self.ln1 = nn.LayerNorm(d, eps=1e-5)

        self.attn = CausalSelfAttention(
            d,
            cfg["n_heads"],
            cfg["context_length"],
            cfg["dropout"]
        )

        self.ln2 = nn.LayerNorm(d, eps=1e-5)

        self.ffn = nn.Sequential(
            nn.Linear(d, cfg["d_ff"]),
            nn.GELU(approximate="none"),
            nn.Linear(cfg["d_ff"], d),
            nn.Dropout(cfg["dropout"])
        )

    def forward(self, x):
        # Pre-LN Attention + Residual Connection.
        x = x + self.attn(self.ln1(x))

        # Pre-LN FFN + Residual Connection.
        return x + self.ffn(self.ln2(x))


class LanguageModel(nn.Module):
    def __init__(self, vocab_size, cfg):
        super().__init__()

        validate_model_config(vocab_size, cfg)

        self.cfg = dict(cfg)
        self.vocab_size = vocab_size

        d = cfg["d_model"]

        self.token_embedding = nn.Embedding(vocab_size, d)

        self.position_embedding = nn.Embedding(
            cfg["context_length"],
            d
        )

        self.dropout = nn.Dropout(cfg["dropout"])

        self.blocks = nn.ModuleList([
            DecoderBlock(cfg)
            for _ in range(cfg["n_layers"])
        ])

        self.final_norm = nn.LayerNorm(d, eps=1e-5)

        self.apply(self._init_weights)

        self.head = nn.Linear(
            d,
            vocab_size,
            bias=False
        )

        # Share the embedding parameter with the LM head.
        self.head.weight = self.token_embedding.weight

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)

            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, ids):
        if not isinstance(ids, torch.Tensor):
            raise TypeError("ids must be a torch.Tensor")

        if ids.ndim != 2:
            raise ValueError("Expected IDs with shape [B, T]")

        if ids.dtype != torch.long:
            raise TypeError("Token IDs must have dtype torch.long")

        b, t = ids.shape

        if b <= 0 or not 1 <= t <= self.cfg["context_length"]:
            raise ValueError(
                "Expected IDs [B,T] with B > 0 and "
                "1 <= T <= context_length"
            )

        if torch.any(ids < 0) or torch.any(ids >= self.vocab_size):
            raise ValueError("Token IDs are outside vocabulary range")

        positions = torch.arange(t, device=ids.device)

        x = self.dropout(
            self.token_embedding(ids)
            + self.position_embedding(positions)
        )

        for block in self.blocks:
            x = block(x)

        return self.head(self.final_norm(x))

    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())

    def expected_parameter_count(self):
        d = self.cfg["d_model"]
        m = self.cfg["d_ff"]

        block = 4 * d * d + 2 * d * m + 9 * d + m

        return (
            self.vocab_size * d
            + self.cfg["context_length"] * d
            + self.cfg["n_layers"] * block
            + 2 * d
        )
