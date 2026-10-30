import argparse
from src.data import prepare_data

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--raw", default="data/raw/input.txt")
    p.add_argument("--output", default="data/processed")
    args = p.parse_args()
    print(prepare_data(args.raw, args.output))
