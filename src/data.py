"""Chuẩn bị dữ liệu và tạo next-character targets (TV2)."""
import hashlib
from pathlib import Path
from urllib.request import urlopen
import torch
from .tokenizer import CharacterTokenizer
from .utils import save_json

DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def prepare_data(raw_path="data/raw/input.txt", output_dir="data/processed"):
    raw_path, out = Path(raw_path), Path(output_dir)
    if not raw_path.exists():
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        with urlopen(DATA_URL, timeout=60) as response:
            raw_path.write_bytes(response.read())
    raw = raw_path.read_bytes()
    text = raw.decode("utf-8")  # Không normalize newline hoặc strip.
    cut = int(0.9 * len(text))
    train_text, val_text = text[:cut], text[cut:]
    if not train_text or not val_text:
        raise ValueError("Corpus too short")
    tok = CharacterTokenizer.fit(train_text)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"train": torch.tensor(tok.encode(train_text), dtype=torch.long),
                "validation": torch.tensor(tok.encode(val_text), dtype=torch.long)}, out / "ids.pt")
    tok.save(out / "tokenizer.json")
    metadata = {"source": DATA_URL, "sha256": hashlib.sha256(raw).hexdigest(),
                "split_index": cut, "train_tokens": len(train_text),
                "validation_tokens": len(val_text), "vocab_size": tok.vocab_size}
    save_json(out / "metadata.json", metadata)
    return metadata


def load_data(directory):
    path = Path(directory)
    return torch.load(path / "ids.pt", map_location="cpu", weights_only=True), CharacterTokenizer.load(path / "tokenizer.json")


def random_batch(ids, batch_size, context_length, device):
    if len(ids) <= context_length:
        raise ValueError("Training split needs at least context_length + 1 tokens")
    starts = torch.randint(len(ids) - context_length, (batch_size,))
    chunks = torch.stack([ids[i:i + context_length + 1] for i in starts.tolist()])
    return chunks[:, :-1].to(device), chunks[:, 1:].to(device)


def evaluation_chunks(ids, context_length, max_tokens=None):
    """Mỗi target sau token đầu được tính một lần; giữ cả phần đuôi."""
    end = len(ids) if max_tokens is None else min(len(ids), max_tokens + 1)
    for start in range(0, end - 1, context_length):
        chunk = ids[start:min(start + context_length + 1, end)]
        yield chunk[:-1].unsqueeze(0), chunk[1:].unsqueeze(0)
