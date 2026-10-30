import argparse
from src.train import load_checkpoint
from src.generate import generate
from src.utils import save_json

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/best.pt")
    p.add_argument("--prompt", default="ROMEO:\n")
    p.add_argument("--strategy", choices=["greedy", "sample"], default="sample")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-new-tokens", type=int, default=100)
    p.add_argument("--output", default="outputs/samples/custom.json")
    a = p.parse_args()
    model, tok, _ = load_checkpoint(a.checkpoint)
    result = generate(model, tok, a.prompt, a.max_new_tokens, a.strategy, a.temperature, a.top_k, a.top_p, a.seed)
    result["checkpoint"] = a.checkpoint
    save_json(a.output, result)
    print(result["text"])
