import argparse
import json
from pathlib import Path
from src.train import load_checkpoint
from src.data import load_data
from src.evaluate import evaluate
from src.utils import save_json

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/best.pt")
    p.add_argument("--output", default="outputs/metrics/validation.json")
    a = p.parse_args()
    model, tok, payload = load_checkpoint(a.checkpoint)
    data, data_tok = load_data(payload["config"]["data_dir"])
    metadata = json.loads((Path(payload["config"]["data_dir"]) / "metadata.json").read_text())
    if metadata["sha256"] != payload["data_sha256"]:
        raise ValueError("Dataset checksum mismatch")
    if tok.to_dict() != data_tok.to_dict():
        raise ValueError("Tokenizer mismatch")
    result = evaluate(model, data["validation"], "cpu")
    result.update(checkpoint=a.checkpoint, split="validation", policy="nonoverlapping targets; full tail; first token excluded")
    save_json(a.output, result)
    print(result)
