
"""Prepare Tiny Shakespeare data and create next-character targets."""

import hashlib
from pathlib import Path
from urllib.request import urlopen

import torch

from .tokenizer import CharacterTokenizer
from .utils import save_json


DATA_URL = (
    "https://raw.githubusercontent.com/"
    "karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)


def _validate_positive_int(value, name):
    """Validate a positive integer parameter."""
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def prepare_data(
    raw_path="data/raw/input.txt",
    output_dir="data/processed"
):
    """Download/read corpus, split it, fit tokenizer on train, and save IDs."""
    raw_path = Path(raw_path)
    output_dir = Path(output_dir)

    # Download only when the raw file does not exist.
    if not raw_path.is_file():
        raw_path.parent.mkdir(parents=True, exist_ok=True)

        with urlopen(DATA_URL, timeout=60) as response:
            raw = response.read()

        raw_path.write_bytes(raw)

    # Preserve original text: no strip or newline normalization.
    raw = raw_path.read_bytes()
    text = raw.decode("utf-8")

    if not text:
        raise ValueError("Corpus cannot be empty")

    # Sequential 90/10 split.
    split_index = int(0.9 * len(text))
    train_text = text[:split_index]
    validation_text = text[split_index:]

    if not train_text or not validation_text:
        raise ValueError("Corpus too short for 90/10 split")

    # Vocabulary is fitted exclusively on training text.
    tokenizer = CharacterTokenizer.fit(train_text)

    train_ids = torch.tensor(
        tokenizer.encode(train_text),
        dtype=torch.long
    )

    validation_ids = torch.tensor(
        tokenizer.encode(validation_text),
        dtype=torch.long
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "train": train_ids,
            "validation": validation_ids
        },
        output_dir / "ids.pt"
    )

    tokenizer.save(output_dir / "tokenizer.json")

    metadata = {
        "source": DATA_URL,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "split_index": split_index,
        "train_tokens": len(train_ids),
        "validation_tokens": len(validation_ids),
        "vocab_size": tokenizer.vocab_size
    }

    save_json(output_dir / "metadata.json", metadata)

    return metadata


def load_data(directory):
    """Load processed token IDs and their tokenizer."""
    directory = Path(directory)

    ids_path = directory / "ids.pt"
    tokenizer_path = directory / "tokenizer.json"

    if not ids_path.is_file():
        raise FileNotFoundError(ids_path)

    if not tokenizer_path.is_file():
        raise FileNotFoundError(tokenizer_path)

    data = torch.load(
        ids_path,
        map_location="cpu",
        weights_only=True
    )

    tokenizer = CharacterTokenizer.load(tokenizer_path)

    if not isinstance(data, dict):
        raise TypeError("Processed data must be a dictionary")

    if "train" not in data or "validation" not in data:
        raise ValueError("Missing train or validation tensor")

    for name in ("train", "validation"):
        ids = data[name]

        if not isinstance(ids, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")

        if ids.ndim != 1 or ids.dtype != torch.long:
            raise ValueError(
                f"{name} must be a 1D torch.long tensor"
            )

        if ids.numel() == 0:
            raise ValueError(f"{name} cannot be empty")

        if torch.any(ids < 0) or torch.any(ids >= tokenizer.vocab_size):
            raise ValueError(f"{name} contains invalid token IDs")

    return data, tokenizer


def random_batch(ids, batch_size, context_length, device):
    """Sample random input-target batches for next-token training."""
    _validate_positive_int(batch_size, "batch_size")
    _validate_positive_int(context_length, "context_length")

    if not isinstance(ids, torch.Tensor):
        raise TypeError("ids must be a torch.Tensor")

    if ids.ndim != 1 or ids.dtype != torch.long:
        raise ValueError("ids must be a 1D torch.long tensor")

    if len(ids) <= context_length:
        raise ValueError(
            "Training split needs at least context_length + 1 tokens"
        )

    # Valid starting positions: 0 through len(ids)-context_length-1.
    starts = torch.randint(
        len(ids) - context_length,
        (batch_size,)
    )

    chunks = torch.stack([
        ids[start:start + context_length + 1]
        for start in starts.tolist()
    ])

    x = chunks[:, :-1].to(device)
    y = chunks[:, 1:].to(device)

    return x, y


def evaluation_chunks(ids, context_length, max_tokens=None):
    """Yield sequential non-overlapping evaluation chunks.

    Every target after the first validation token is evaluated once.
    Context is reset at the beginning of each chunk.
    """
    _validate_positive_int(context_length, "context_length")

    if not isinstance(ids, torch.Tensor):
        raise TypeError("ids must be a torch.Tensor")

    if ids.ndim != 1 or ids.dtype != torch.long:
        raise ValueError("ids must be a 1D torch.long tensor")

    if len(ids) < 2:
        raise ValueError(
            "Evaluation split must contain at least two tokens"
        )

    if max_tokens is not None:
        if type(max_tokens) is not int or max_tokens <= 0:
            raise ValueError("max_tokens must be a positive integer")

    # max_tokens counts targets, not input tokens.
    target_count = len(ids) - 1

    if max_tokens is not None:
        target_count = min(target_count, max_tokens)

    # The final target is at index target_count.
    end = target_count + 1

    for start in range(0, target_count, context_length):
        chunk = ids[
            start:min(start + context_length + 1, end)
        ]

        x = chunk[:-1].unsqueeze(0)
        y = chunk[1:].unsqueeze(0)

        yield x, y
