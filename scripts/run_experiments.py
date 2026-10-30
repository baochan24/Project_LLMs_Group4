"""Q1 thu từ training metrics; Q2 decoding; Q3 bật rõ bằng --with-pretrained."""
import argparse
from pathlib import Path
from src.train import load_checkpoint
from src.generate import generate
from src.visualize import plot_loss
from src.utils import save_json


def run(checkpoint, with_pretrained=False, local_files_only=False):
    model, tok, payload = load_checkpoint(checkpoint)
    cfg = payload["config"]
    out = Path(cfg["output_dir"])
    prompt = cfg["generation"]["prompt"]
    n = cfg["generation"]["max_new_tokens"]
    # Mỗi series chỉ thay một yếu tố so với sampling không lọc, temperature=1.
    settings = [("greedy", {"strategy": "greedy"}),
                ("sample_base", {}),
                ("temperature_0.7", {"temperature": 0.7}),
                ("temperature_1.3", {"temperature": 1.3}),
                ("top_k_10", {"top_k": 10}),
                ("top_p_0.9", {"top_p": 0.9})]
    samples = []
    for seed in cfg["experiments"]["seeds"]:
        for name, options in settings:
            result = generate(model, tok, prompt, max_new_tokens=n, seed=seed, **options)
            result.update(setting=name, checkpoint=str(checkpoint))
            samples.append(result)
    save_json(out / "samples" / "q2_custom.json", samples)
    metrics = out / "metrics" / "training.csv"
    if metrics.exists():
        plot_loss(metrics, out / "figures" / "loss.png")
    if with_pretrained:
        from src.pretrained import load_pretrained, generate_pretrained
        pretrained, tokenizer = load_pretrained(local_files_only=local_files_only)
        paired = []
        for seed in cfg["experiments"]["seeds"]:
            paired.append({"custom": generate(model, tok, prompt, max_new_tokens=n, seed=seed),
                           "gpt2": generate_pretrained(pretrained, tokenizer, prompt, seed=seed, max_new_tokens=n)})
        save_json(out / "samples" / "q3_comparison.json", paired)
    print("Saved Q2 samples; Q3 run:", with_pretrained)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/best.pt")
    p.add_argument("--with-pretrained", action="store_true")
    p.add_argument("--local-files-only", action="store_true")
    a = p.parse_args()
    run(a.checkpoint, a.with_pretrained, a.local_files_only)
