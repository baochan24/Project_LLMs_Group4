
"""Q3 Extended: GPT-2 prompt and decoding comparison."""

from pathlib import Path

from src.pretrained import load_pretrained, generate_pretrained
from src.utils import save_json


def run_q3_extended():
    device = "cpu"

    prompt = (
        "JULIET:\n"
        "What light through yonder window breaks?"
    )

    seeds = [42, 43, 44]

    experiments = [
        {
            "name": "greedy",
            "strategy": "greedy",
            "temperature": 1.0
        },
        {
            "name": "temperature_0.7",
            "strategy": "sample",
            "temperature": 0.7
        },
        {
            "name": "temperature_1.0",
            "strategy": "sample",
            "temperature": 1.0
        }
    ]

    print("Loading pretrained GPT-2...")
    model, tokenizer = load_pretrained(device=device)

    results = []

    for experiment in experiments:
        for seed in seeds:
            result = generate_pretrained(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                seed=seed,
                strategy=experiment["strategy"],
                max_new_tokens=100,
                temperature=experiment["temperature"],
                top_k=0,
                top_p=1.0
            )

            result["experiment"] = experiment["name"]
            results.append(result)

            print(
                f"\nExperiment: {experiment['name']}"
                f"\nSeed: {seed}"
                f"\n{result['text']}"
                "\n" + "=" * 60
            )

    output_path = Path("outputs/samples/q3_extended.json")

    save_json(output_path, results)

    print(f"\nQ3 EXTENDED COMPLETED: {output_path}")
    print(f"Total samples: {len(results)}")


if __name__ == "__main__":
    run_q3_extended()
