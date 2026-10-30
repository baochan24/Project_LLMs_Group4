"""Pre-LN decoder-only Transformer, learned position và tied head (TV4)."""
import torch
from torch import nn
from .attention import CausalSelfAttention


class DecoderBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg["d_model"]
        self.ln1 = nn.LayerNorm(d, eps=1e-5)
        self.attn = CausalSelfAttention(d, cfg["n_heads"], cfg["context_length"], cfg["dropout"])
        self.ln2 = nn.LayerNorm(d, eps=1e-5)
        self.ffn = nn.Sequential(nn.Linear(d, cfg["d_ff"]), nn.GELU(approximate="none"),
                                 nn.Linear(cfg["d_ff"], d), nn.Dropout(cfg["dropout"]))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.ffn(self.ln2(x))


class LanguageModel(nn.Module):
    def __init__(self, vocab_size, cfg):
        super().__init__()
        self.cfg = dict(cfg)
        self.vocab_size = vocab_size
        d = cfg["d_model"]
        self.token_embedding = nn.Embedding(vocab_size, d)
        self.position_embedding = nn.Embedding(cfg["context_length"], d)
        self.dropout = nn.Dropout(cfg["dropout"])
        self.blocks = nn.ModuleList([DecoderBlock(cfg) for _ in range(cfg["n_layers"])])
        self.final_norm = nn.LayerNorm(d, eps=1e-5)
        self.apply(self._init_weights)
        self.head = nn.Linear(d, vocab_size, bias=False)
        self.head.weight = self.token_embedding.weight

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, ids):
        if ids.ndim != 2 or not 1 <= ids.shape[1] <= self.cfg["context_length"]:
            raise ValueError("Expected IDs [B,T] with 1 <= T <= context_length")
        positions = torch.arange(ids.shape[1], device=ids.device)
        x = self.dropout(self.token_embedding(ids) + self.position_embedding(positions))
        for block in self.blocks:
            x = block(x)
        return self.head(self.final_norm(x))

    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())

    def expected_parameter_count(self):
        d, m = self.cfg["d_model"], self.cfg["d_ff"]
        block = 4*d*d + 2*d*m + 9*d + m
        return self.vocab_size*d + self.cfg["context_length"]*d + self.cfg["n_layers"]*block + 2*d
