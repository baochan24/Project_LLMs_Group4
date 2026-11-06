"""GPT-2 inference riêng, không dùng character IDs (TV1)."""
import torch


def load_pretrained(device="cpu", local_files_only=False):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    name = "openai-community/gpt2"
    tok = AutoTokenizer.from_pretrained(name, local_files_only=local_files_only)
    model = AutoModelForCausalLM.from_pretrained(name, local_files_only=local_files_only).to(device).eval()
    return model, tok


@torch.no_grad()
def generate_pretrained(model, tokenizer, prompt, seed=42, max_new_tokens=100,
                        strategy="sample", temperature=1.0, top_k=0, top_p=1.0):
    if not prompt or max_new_tokens < 1 or strategy not in ("greedy", "sample"):
        raise ValueError("Invalid prompt, token count or strategy")
    if temperature <= 0 or top_k < 0 or not 0 < top_p <= 1:
        raise ValueError("Invalid sampling settings")
    from .utils import set_seed
    set_seed(seed)
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt").to(next(model.parameters()).device)
    if inputs.input_ids.shape[1] + max_new_tokens > model.config.n_positions:
        raise ValueError("Prompt + generation exceeds GPT-2 position limit")
    kwargs = {"do_sample": strategy == "sample", "max_new_tokens": max_new_tokens,
              "pad_token_id": tokenizer.eos_token_id}
    if strategy == "sample":
        kwargs.update(temperature=temperature, top_k=top_k, top_p=top_p)
    ids = model.generate(**inputs, **kwargs)
    return {"model": "openai-community/gpt2", "prompt": prompt, "seed": seed,
            "strategy": strategy, "temperature": temperature, "top_k": top_k, "top_p": top_p,
            "max_new_tokens": max_new_tokens, "text": tokenizer.decode(ids[0], skip_special_tokens=True)}
def load_pretrained(device="cpu", local_files_only=False):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    name = "openai-community/gpt2"

    tok = AutoTokenizer.from_pretrained(
        name,
        local_files_only=local_files_only
    )

    model = AutoModelForCausalLM.from_pretrained(
        name,
        local_files_only=local_files_only
    ).to(device).eval()

    return model, tok


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
    if not prompt or max_new_tokens < 1 or strategy not in ("greedy", "sample"):
        raise ValueError("Invalid prompt, token count or strategy")

    if temperature <= 0 or top_k < 0 or not 0 < top_p <= 1:
        raise ValueError("Invalid sampling settings")

    from .utils import set_seed

    set_seed(seed)
    model.eval()

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    ).to(next(model.parameters()).device)

    if inputs.input_ids.shape[1] + max_new_tokens > model.config.n_positions:
        raise ValueError("Prompt + generation exceeds GPT-2 position limit")

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

    ids = model.generate(**inputs, **kwargs)

    return {
        "model": "openai-community/gpt2",
        "prompt": prompt,
        "seed": seed,
        "strategy": strategy,
        "temperature": temperature,
        "top_k": top_k,
        "top_p": top_p,
        "max_new_tokens": max_new_tokens,
        "text": tokenizer.decode(
            ids[0],
            skip_special_tokens=True
        )
    }

