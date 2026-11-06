
"""Character tokenizer; vocabulary is built from training text only."""

import json
from numbers import Integral
from pathlib import Path


class CharacterTokenizer:
    """Character-level tokenizer with ID 0 reserved for UNK."""

    def __init__(self, chars):
        try:
            chars = list(chars)
        except TypeError as exc:
            raise TypeError("chars must be an iterable of characters") from exc

        if not chars:
            raise ValueError("Vocabulary cannot be empty")

        if any(not isinstance(c, str) or len(c) != 1 for c in chars):
            raise ValueError("Each vocabulary entry must be one character")

        if len(set(chars)) != len(chars):
            raise ValueError("chars must contain unique characters")

        self.chars = chars
        self.stoi = {c: i + 1 for i, c in enumerate(chars)}
        self.unk_id = 0

    @classmethod
    def fit(cls, text):
        """Build vocabulary from training text only."""
        if not isinstance(text, str):
            raise TypeError("Training text must be a string")

        if not text:
            raise ValueError("Empty training text")

        return cls(sorted(set(text)))

    @property
    def vocab_size(self):
        """Vocabulary size, including UNK."""
        return len(self.chars) + 1

    def encode(self, text):
        """Convert characters into token IDs."""
        if not isinstance(text, str):
            raise TypeError("Input text must be a string")

        return [
            self.stoi.get(char, self.unk_id)
            for char in text
        ]

    def decode(self, ids):
        """Convert token IDs back into characters."""
        result = []

        for token_id in ids:
            if isinstance(token_id, bool) or not isinstance(token_id, Integral):
                raise TypeError("Token IDs must be integers")

            token_id = int(token_id)

            if not 0 <= token_id < self.vocab_size:
                raise ValueError(f"Invalid token ID: {token_id}")

            if token_id == self.unk_id:
                result.append("<UNK>")
            else:
                result.append(self.chars[token_id - 1])

        return "".join(result)

    def to_dict(self):
        """Export tokenizer configuration."""
        return {
            "type": "character",
            "chars": self.chars,
            "unk_id": self.unk_id
        }

    @classmethod
    def from_dict(cls, obj):
        """Restore tokenizer from saved configuration."""
        if not isinstance(obj, dict):
            raise TypeError("Tokenizer configuration must be a dictionary")

        if obj.get("type") != "character":
            raise ValueError("Unsupported tokenizer type")

        if type(obj.get("unk_id")) is not int or obj["unk_id"] != 0:
            raise ValueError("Unsupported UNK ID")

        if "chars" not in obj:
            raise ValueError("Missing tokenizer vocabulary")

        return cls(obj["chars"])

    def save(self, path):
        """Save tokenizer configuration as JSON."""
        from .utils import save_json

        save_json(path, self.to_dict())

    @classmethod
    def load(cls, path):
        """Load tokenizer configuration from JSON."""
        path = Path(path)

        if not path.is_file():
            raise FileNotFoundError(path)

        with path.open("r", encoding="utf-8") as file:
            config = json.load(file)

        return cls.from_dict(config)
