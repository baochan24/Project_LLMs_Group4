"""Character tokenizer; vocabulary chỉ được xây từ train (TV2)."""
import json
from pathlib import Path


class CharacterTokenizer:
    def __init__(self, chars):
        if any(len(c) != 1 for c in chars) or len(set(chars)) != len(chars):
            raise ValueError("chars must contain unique individual characters")
        self.chars = list(chars)
        self.stoi = {c: i + 1 for i, c in enumerate(chars)}
        self.unk_id = 0

    @classmethod
    def fit(cls, text):
        if not text:
            raise ValueError("Empty training text")
        return cls(sorted(set(text)))

    @property
    def vocab_size(self):
        return len(self.chars) + 1

    def encode(self, text):
        return [self.stoi.get(c, self.unk_id) for c in text]

    def decode(self, ids):
        return "".join("<UNK>" if i == 0 else self.chars[i - 1] for i in ids)

    def to_dict(self):
        return {"type": "character", "chars": self.chars, "unk_id": 0}

    @classmethod
    def from_dict(cls, obj):
        if obj["type"] != "character" or obj["unk_id"] != 0:
            raise ValueError("Unsupported tokenizer mapping")
        return cls(obj["chars"])

    def save(self, path):
        from .utils import save_json
        save_json(path, self.to_dict())

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
