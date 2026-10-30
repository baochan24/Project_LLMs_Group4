"""Hình phục vụ slide; không tự tạo số liệu (TV5)."""
import csv
from pathlib import Path


def plot_loss(csv_path, output):
    import matplotlib.pyplot as plt
    with open(csv_path, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    fig, ax = plt.subplots()
    steps = [int(r["step"]) for r in rows]
    for key, label in [("train_eval_loss", "Train (eval mode)"), ("validation_loss", "Validation")]:
        ax.plot(steps, [float(r[key]) for r in rows], label=label)
    ax.set(xlabel="Training step", ylabel="Mean token NLL")
    ax.legend()
    fig.tight_layout()
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)


def plot_attention(weights, labels, output):
    import matplotlib.pyplot as plt
    a = weights.detach().cpu().numpy()
    fig, ax = plt.subplots()
    image = ax.imshow(a, vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), labels=labels, rotation=90)
    ax.set_yticks(range(len(labels)), labels=labels)
    ax.set(xlabel="Key position", ylabel="Query position")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)
