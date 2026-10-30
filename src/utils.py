"""Tiện ích chung của TV6."""
import json
import random
from pathlib import Path
import torch


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(name="cpu"):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def load_config(path):
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    m, t = cfg["model"], cfg["training"]
    for key in ("d_model", "n_heads", "n_layers", "context_length", "d_ff"):
        if type(m[key]) is not int or m[key] <= 0:
            raise ValueError(f"model.{key} must be a positive integer")
    if m["d_model"] % m["n_heads"] or not 0 <= m["dropout"] < 1:
        raise ValueError("Invalid heads or dropout")
    for key in ("batch_size", "steps", "eval_interval", "eval_tokens", "num_threads"):
        if type(t[key]) is not int or t[key] <= 0:
            raise ValueError(f"training.{key} must be positive")
    if t["learning_rate"] <= 0 or t["weight_decay"] < 0 or t["grad_clip"] <= 0:
        raise ValueError("Invalid optimizer settings")
    if not 0 < t["betas"][0] < 1 or not 0 < t["betas"][1] < 1 or t["eps"] <= 0:
        raise ValueError("Invalid AdamW betas/epsilon")
    return cfg
