"""Greedy và sampling; không có KV cache trong custom model (TV5)."""
import torch


def filter_logits(logits, top_k=0, top_p=1.0):
    if top_k < 0 or not 0 < top_p <= 1:
        raise ValueError("Require top_k >= 0 and 0 < top_p <= 1")
    logits = logits.clone()
    if top_k:
        threshold = logits.topk(min(top_k, logits.shape[-1])).values[..., -1:]
        logits = logits.masked_fill(logits < threshold, float("-inf"))
    if top_p < 1:
        ordered, indices = logits.sort(descending=True, dim=-1)
        remove = ordered.softmax(-1).cumsum(-1) > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        ordered = ordered.masked_fill(remove, float("-inf"))
        logits = torch.empty_like(logits).scatter(-1, indices, ordered)
    return logits


@torch.no_grad()
def generate(model, tokenizer, prompt, max_new_tokens=100, strategy="sample",
             temperature=1.0, top_k=0, top_p=1.0, seed=42):
    if not prompt or max_new_tokens < 0 or strategy not in ("greedy", "sample"):
        raise ValueError("Nonempty prompt, nonnegative token count and valid strategy required")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    device = next(model.parameters()).device
    generator = torch.Generator(device=device).manual_seed(seed)
    ids = torch.tensor([tokenizer.encode(prompt)], device=device, dtype=torch.long)
    was_training = model.training
    model.eval()
    try:
        for _ in range(max_new_tokens):
            logits = model(ids[:, -model.cfg["context_length"]:])[:, -1, :]
            if strategy == "greedy":
                nxt = logits.argmax(-1, keepdim=True)
            else:
                probs = filter_logits(logits / temperature, top_k, top_p).softmax(-1)
                nxt = torch.multinomial(probs, 1, generator=generator)
            ids = torch.cat((ids, nxt), dim=1)
        return {"prompt": prompt, "text": tokenizer.decode(ids[0].tolist()),
                "generated_ids": ids[0, len(tokenizer.encode(prompt)):].tolist(),
                "strategy": strategy, "temperature": temperature, "top_k": top_k,
                "top_p": top_p, "seed": seed, "max_new_tokens": max_new_tokens}
    finally:
        model.train(was_training)
