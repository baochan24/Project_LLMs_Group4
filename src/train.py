"""Training loop và checkpoint/resume (TV4). Chỉ load checkpoint tin cậy."""
import csv
import json
import platform
from pathlib import Path
import torch
from torch.nn import functional as F
from .data import load_data, random_batch
from .model import LanguageModel
from .tokenizer import CharacterTokenizer
from .evaluate import evaluate
from .utils import get_device, set_seed, save_json


def make_optimizer(model, cfg):
    # parameters() loại trùng tied weights; mỗi Parameter chỉ thuộc một group.
    decay, no_decay = [], []
    for p in model.parameters():
        (decay if p.ndim >= 2 else no_decay).append(p)
    return torch.optim.AdamW([{"params": decay, "weight_decay": cfg["weight_decay"]},
                             {"params": no_decay, "weight_decay": 0.0}],
                            lr=cfg["learning_rate"], betas=tuple(cfg["betas"]), eps=cfg["eps"])


def save_checkpoint(path, model, optimizer, tokenizer, cfg, step, best, data_sha256):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "tokenizer": tokenizer.to_dict(), "config": cfg, "step": step,
                "best_validation_loss": best, "data_sha256": data_sha256,
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}, temp)
    temp.replace(path)


def load_checkpoint(path, device="cpu"):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    torch.set_num_threads(payload["config"].get("training", {}).get("num_threads", 1))
    tokenizer = CharacterTokenizer.from_dict(payload["tokenizer"])
    model = LanguageModel(tokenizer.vocab_size, payload["config"]["model"]).to(device)
    model.load_state_dict(payload["model"])
    return model, tokenizer, payload


def train(cfg, resume=None, steps=None):
    t = cfg["training"]
    torch.set_num_threads(t["num_threads"])
    set_seed(cfg["seed"])
    device = get_device(cfg["device"])
    data, tokenizer = load_data(cfg["data_dir"])
    meta = json.loads((Path(cfg["data_dir"]) / "metadata.json").read_text())
    model = LanguageModel(tokenizer.vocab_size, cfg["model"]).to(device)
    optimizer = make_optimizer(model, t)
    start, best = 0, float("inf")
    if resume:
        model, saved_tok, payload = load_checkpoint(resume, device)
        if payload["config"] != cfg or saved_tok.to_dict() != tokenizer.to_dict() or payload["data_sha256"] != meta["sha256"]:
            raise ValueError("Resume config/tokenizer/dataset mismatch")
        optimizer = make_optimizer(model, t)
        optimizer.load_state_dict(payload["optimizer"])
        start, best = payload["step"], payload["best_validation_loss"]
        torch.set_rng_state(payload["torch_rng"])
        if payload["cuda_rng"] and device.type == "cuda":
            torch.cuda.set_rng_state_all(payload["cuda_rng"])
    target = t["steps"] if steps is None else steps
    if target <= start:
        raise ValueError("Target total steps must exceed checkpoint step")
    out = Path(cfg["output_dir"])
    checkpoints = Path(cfg["checkpoint_dir"])
    out.mkdir(parents=True, exist_ok=True)
    save_json(out / "environment.json", {"python": platform.python_version(), "torch": str(torch.__version__),
                                          "device": str(device), "platform": platform.platform()})
    from .generate import generate
    sample_path = out / "samples" / "before_train.json"
    if start == 0:
        save_json(sample_path, generate(model, tokenizer, cfg["generation"]["prompt"], strategy="greedy", max_new_tokens=100))
    metrics_path = out / "metrics" / "training.csv"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    # Resume từ checkpoint cũ: loại bỏ metrics thuộc phần chạy sau checkpoint đó.
    rows = []
    if resume and metrics_path.exists():
        with metrics_path.open(newline="") as f:
            rows = [r for r in csv.DictReader(f) if int(r["step"]) <= start]
    fields = ["step", "batch_train_loss", "train_eval_loss", "validation_loss", "validation_ppl", "validation_tokens"]
    with metrics_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        def record(step, batch_loss):
            nonlocal best
            tr = evaluate(model, data["train"], device, t["eval_tokens"])
            va = evaluate(model, data["validation"], device, t["eval_tokens"])
            writer.writerow(dict(zip(fields, [step, batch_loss, tr["loss"], va["loss"], va["ppl"], va["tokens"]])))
            f.flush()
            if va["loss"] < best:
                best = va["loss"]
                save_checkpoint(checkpoints / "best.pt", model, optimizer, tokenizer, cfg, step, best, meta["sha256"])
            save_checkpoint(checkpoints / "latest.pt", model, optimizer, tokenizer, cfg, step, best, meta["sha256"])
            print(f"step={step} train_eval={tr['loss']:.4f} val={va['loss']:.4f} ppl={va['ppl']:.2f}", flush=True)
        if start == 0:
            record(0, "")
        for step in range(start + 1, target + 1):
            model.train()
            x, y = random_batch(data["train"], t["batch_size"], cfg["model"]["context_length"], device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x).flatten(0, 1), y.flatten())
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), t["grad_clip"], error_if_nonfinite=True)
            optimizer.step()
            if step % t["eval_interval"] == 0 or step == target:
                record(step, loss.item())
    return str(checkpoints / "best.pt")
