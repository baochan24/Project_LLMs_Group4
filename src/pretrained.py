
"""GPT-2 pretrained inference riêng biệt với custom character model."""

import math
import torch


def load_pretrained(device="cpu", local_files_only=False):
    """Load pretrained GPT-2 model và tokenizer."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    name = "openai-community/gpt2"

    tokenizer = AutoTokenizer.from_pretrained(
        name,
        local_files_only=local_files_only
    )

    model = AutoModelForCausalLM.from_pretrained(
        name,
        local_files_only=local_files_only
    ).to(device)

    model.eval()

    return model, tokenizer


@torch.no_grad()
def generate_pretrained(
    model,
    tokenizer,
    prompt,
    seed=42,
    strategy="sample",
    max_new_tokens=100,
    temperature=1.0,
    top_k=0,
    top_p=1.0
):
    """Generate text using pretrained GPT-2."""

    # 1. Validate input
    if not isinstance(prompt, str) or not prompt:
        raise ValueError("Prompt must be a non-empty string")

    if type(max_new_tokens) is not int or max_new_tokens < 1:
        raise ValueError("max_new_tokens must be a positive integer")

    if strategy not in ("greedy", "sample"):
        raise ValueError("Unsupported generation strategy")

    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be positive and finite")

    if type(top_k) is not int or top_k < 0:
        raise ValueError("top_k must be a non-negative integer")

    if not math.isfinite(top_p) or not 0 < top_p <= 1:
        raise ValueError("top_p must be in (0, 1]")

    # 2. Set random seed
    from .utils import set_seed
    set_seed(seed)

    model.eval()

    # 3. Tokenize prompt
    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    ).to(next(model.parameters()).device)

    # 4. Check context length
    if inputs.input_ids.shape[1] + max_new_tokens > model.config.n_positions:
        raise ValueError("Prompt + generation exceeds GPT-2 position limit")

    # 5. Configure decoding
    kwargs = {
        "do_sample": strategy == "sample",
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.eos_token_id
    }

    if strategy == "sample":
        kwargs.update(
            temperature=temperature,
            top_k=top_k,
            top_p=top_p
        )

    # 6. Generate
    ids = model.generate(
        **inputs,
        **kwargs
    )

    # 7. Decode output
    generated_text = tokenizer.decode(
        ids[0],
        skip_special_tokens=True
    )

    return {
        "model": "openai-community/gpt2",
        "prompt": prompt,
        "seed": seed,
        "strategy": strategy,
        "temperature": temperature,
        "top_k": top_k,
        "top_p": top_p,
        "max_new_tokens": max_new_tokens,
        "text": generated_text
    }
