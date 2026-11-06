
"""Training loop and reliable checkpoint/resume (TV4)."""

import copy
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
    """Create AdamW with separate decay and no-decay groups."""
    decay, no_decay = [], []

    for p in model.parameters():
        (decay if p.ndim >= 2 else no_decay).append(p)

    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": cfg["weight_decay"]},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=cfg["learning_rate"],
        betas=tuple(cfg["betas"]),
        eps=cfg["eps"],
    )


def _cpu_state_dict(state_dict):
    """Create an independent CPU copy of a model state."""
    return {
        key: value.detach().cpu().clone()
        for key, value in state_dict.items()
    }


def _atomic_save(payload, path):
    """Write checkpoint atomically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temp)
    temp.replace(path)


def save_checkpoint(
    path,
    model,
    optimizer,
    tokenizer,
    cfg,
    step,
    best,
    data_sha256,
    best_checkpoint=None,
):
    """Save a complete training checkpoint."""

    payload = {
        "model": _cpu_state_dict(model.state_dict()),
        "optimizer": copy.deepcopy(optimizer.state_dict()),
        "tokenizer": tokenizer.to_dict(),
        "config": copy.deepcopy(cfg),
        "step": step,
        "best_validation_loss": best,
        "data_sha256": data_sha256,
        "torch_rng": torch.get_rng_state().clone(),
        "cuda_rng": (
            torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else []
        ),
        # Snapshot of the best complete checkpoint so far.
        "best_checkpoint": copy.deepcopy(best_checkpoint),
    }

    _atomic_save(payload, path)
    return payload


def load_checkpoint(path, device="cpu"):
    """Load a trusted checkpoint and restore its model."""
    payload = torch.load(
        path,
        map_location="cpu",
        weights_only=True,
    )

    torch.set_num_threads(
        payload["config"].get("training", {}).get("num_threads", 1)
    )

    tokenizer = CharacterTokenizer.from_dict(payload["tokenizer"])

    model = LanguageModel(
        tokenizer.vocab_size,
        payload["config"]["model"],
    ).to(device)

    model.load_state_dict(payload["model"])

    return model, tokenizer, payload


def _validate_best_snapshot(snapshot, cfg, tokenizer, data_sha256, current_step):
    """Validate that a best snapshot belongs to the resumed history."""

    if not isinstance(snapshot, dict):
        raise ValueError("Invalid best checkpoint snapshot")

    if snapshot.get("config") != cfg:
        raise ValueError("Best checkpoint config mismatch")

    if snapshot.get("tokenizer") != tokenizer.to_dict():
        raise ValueError("Best checkpoint tokenizer mismatch")

    if snapshot.get("data_sha256") != data_sha256:
        raise ValueError("Best checkpoint dataset mismatch")

    step = snapshot.get("step")

    if type(step) is not int or not 0 <= step <= current_step:
        raise ValueError("Best checkpoint step is outside resumed history")

    loss = snapshot.get("best_validation_loss")

    if not isinstance(loss, (int, float)) or not torch.isfinite(
        torch.tensor(loss)
    ).item():
        raise ValueError("Invalid best checkpoint validation loss")

    return True


def train(cfg, resume=None, steps=None):
    """Train the language model, optionally resuming from latest.pt."""

    t = cfg["training"]

    torch.set_num_threads(t["num_threads"])
    set_seed(cfg["seed"])

    device = get_device(cfg["device"])

    data, tokenizer = load_data(cfg["data_dir"])

    meta = json.loads(
        (Path(cfg["data_dir"]) / "metadata.json").read_text(
            encoding="utf-8"
        )
    )

    model = LanguageModel(
        tokenizer.vocab_size,
        cfg["model"],
    ).to(device)

    optimizer = make_optimizer(model, t)

    start = 0
    best = float("inf")
    best_checkpoint = None

    if resume:
        model, saved_tok, payload = load_checkpoint(resume, device)

        if (
            payload["config"] != cfg
            or saved_tok.to_dict() != tokenizer.to_dict()
            or payload["data_sha256"] != meta["sha256"]
        ):
            raise ValueError(
                "Resume config/tokenizer/dataset mismatch"
            )

        optimizer = make_optimizer(model, t)
        optimizer.load_state_dict(payload["optimizer"])

        start = payload["step"]
        best = payload["best_validation_loss"]

        if type(start) is not int or start < 0:
            raise ValueError("Invalid checkpoint step")

        best_checkpoint = payload.get("best_checkpoint")

        if best_checkpoint is not None:
            _validate_best_snapshot(
                best_checkpoint,
                cfg,
                tokenizer,
                meta["sha256"],
                start,
            )

            # Restore best.pt from the checkpoint's own history.
            _atomic_save(
                best_checkpoint,
                Path(cfg["checkpoint_dir"]) / "best.pt",
            )

        else:
            # Legacy checkpoint recovery:
            # Do not trust a possibly newer best.pt on disk.
            # Re-establish a valid best snapshot from the resumed model.
            recovered = evaluate(
                model,
                data["validation"],
                device,
                t["eval_tokens"],
            )

            best = recovered["loss"]

            best_checkpoint = save_checkpoint(
                Path(cfg["checkpoint_dir"]) / "best.pt",
                model,
                optimizer,
                tokenizer,
                cfg,
                start,
                best,
                meta["sha256"],
            )

        # Restore RNG only after model and optimizer initialization.
        torch.set_rng_state(payload["torch_rng"])

        if payload["cuda_rng"] and device.type == "cuda":
            torch.cuda.set_rng_state_all(payload["cuda_rng"])

    target = t["steps"] if steps is None else steps

    if type(target) is not int or target <= start:
        raise ValueError(
            "Target total steps must be an integer greater than checkpoint step"
        )

    out = Path(cfg["output_dir"])
    checkpoints = Path(cfg["checkpoint_dir"])

    out.mkdir(parents=True, exist_ok=True)

    save_json(
        out / "environment.json",
        {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "device": str(device),
            "platform": platform.platform(),
        },
    )

    from .generate import generate

    sample_path = out / "samples" / "before_train.json"

    if start == 0:
        save_json(
            sample_path,
            generate(
                model,
                tokenizer,
                cfg["generation"]["prompt"],
                strategy="greedy",
                max_new_tokens=cfg["generation"]["max_new_tokens"],
            ),
        )

    metrics_path = out / "metrics" / "training.csv"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []

    if resume and metrics_path.exists():
        with metrics_path.open(
            newline="",
            encoding="utf-8",
        ) as f:
            rows = [
                row
                for row in csv.DictReader(f)
                if int(row["step"]) <= start
            ]

    fields = [
        "step",
        "batch_train_loss",
        "train_eval_loss",
        "validation_loss",
        "validation_ppl",
        "validation_tokens",
    ]

    with metrics_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(f, fieldnames=fields)

        writer.writeheader()
        writer.writerows(rows)

        def record(step, batch_loss):
            nonlocal best, best_checkpoint

            tr = evaluate(
                model,
                data["train"],
                device,
                t["eval_tokens"],
            )

            va = evaluate(
                model,
                data["validation"],
                device,
                t["eval_tokens"],
            )

            if not torch.isfinite(torch.tensor(va["loss"])).item():
                raise FloatingPointError(
                    "Non-finite validation loss"
                )

            writer.writerow(
                dict(
                    zip(
                        fields,
                        [
                            step,
                            batch_loss,
                            tr["loss"],
                            va["loss"],
                            va["ppl"],
                            va["tokens"],
                        ],
                    )
                )
            )

            f.flush()

            if va["loss"] < best:
                best = va["loss"]

                best_checkpoint = save_checkpoint(
                    checkpoints / "best.pt",
                    model,
                    optimizer,
                    tokenizer,
                    cfg,
                    step,
                    best,
                    meta["sha256"],
                )

            save_checkpoint(
                checkpoints / "latest.pt",
                model,
                optimizer,
                tokenizer,
                cfg,
                step,
                best,
                meta["sha256"],
                best_checkpoint=best_checkpoint,
            )

            print(
                f"step={step} "
                f"train_eval={tr['loss']:.4f} "
                f"val={va['loss']:.4f} "
                f"ppl={va['ppl']:.2f}",
                flush=True,
            )

        if start == 0:
            record(0, "")

        for step in range(start + 1, target + 1):
            model.train()

            x, y = random_batch(
                data["train"],
                t["batch_size"],
                cfg["model"]["context_length"],
                device,
            )

            optimizer.zero_grad(set_to_none=True)

            logits = model(x)

            loss = F.cross_entropy(
                logits.flatten(0, 1),
                y.flatten(),
            )

            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss")

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                t["grad_clip"],
                error_if_nonfinite=True,
            )

            optimizer.step()

            if step % t["eval_interval"] == 0 or step == target:
                record(step, loss.item())

    return str(checkpoints / "best.pt")
