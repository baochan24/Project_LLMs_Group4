import argparse
from src.train import train
from src.utils import load_config

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/baseline.json")
    p.add_argument("--resume")
    p.add_argument("--steps", type=int, help="Target total optimizer steps, not additional steps")
    a = p.parse_args()
    print(train(load_config(a.config), a.resume, a.steps))
