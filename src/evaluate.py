"""Token-weighted NLL và perplexity trên fixed chunks (TV5)."""

import math

import torch
from torch.nn import functional as F

from .data import evaluation_chunks


@torch.no_grad()
def evaluate(model, ids, device, max_tokens=None):
    was_training = model.training
    model.eval()

    total_nll = 0.0
    count = 0

    try:
        for x, y in evaluation_chunks(
            ids,
            model.cfg["context_length"],
            max_tokens,
        ):
            logits = model(x.to(device))
            targets = y.to(device)

            batch_nll = F.cross_entropy(
                logits.flatten(0, 1),
                targets.flatten(),
                reduction="sum",
            ).item()

            if not math.isfinite(batch_nll):
                raise FloatingPointError(
                    "Non-finite evaluation NLL"
                )

            total_nll += batch_nll
            count += targets.numel()

        if count == 0:
            raise ValueError(
                "Evaluation split must have at least two tokens"
            )

        # Trung bình NLL theo số target token.
        loss = total_nll / count

        if not math.isfinite(loss):
            raise FloatingPointError(
                "Non-finite evaluation loss"
            )

        try:
            ppl = math.exp(loss)
        except OverflowError as exc:
            raise FloatingPointError(
                "Evaluation perplexity overflow"
            ) from exc

        return {
            "loss": loss,
            "ppl": ppl,
            "tokens": count,
        }

    finally:
        # Khôi phục mode ban đầu, kể cả khi đánh giá gặp lỗi.
        model.train(was_training)