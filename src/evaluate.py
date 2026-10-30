"""Token-weighted NLL và perplexity trên fixed chunks (TV5)."""
import math
import torch
from torch.nn import functional as F
from .data import evaluation_chunks


@torch.no_grad()
def evaluate(model, ids, device, max_tokens=None):
    was_training = model.training
    model.eval()
    total_nll, count = 0.0, 0
    try:
        for x, y in evaluation_chunks(ids, model.cfg["context_length"], max_tokens):
            logits = model(x.to(device))
            total_nll += F.cross_entropy(logits.flatten(0, 1), y.to(device).flatten(), reduction="sum").item()
            count += y.numel()
        if not count:
            raise ValueError("Evaluation split must have at least two tokens")
        loss = total_nll / count
        return {"loss": loss, "ppl": math.exp(loss), "tokens": count}
    finally:
        model.train(was_training)
