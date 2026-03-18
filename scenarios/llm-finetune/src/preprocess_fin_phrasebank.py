"""
TDP 1 — Market Data Provider: prepare financial sentiment dataset.

Downloads FinanceMTEB/financial_phrasebank from HuggingFace (Parquet format),
converts to Alpaca JSONL, and writes to /mnt/output/.

Run by the Market Data Provider on their own machine before encrypting.
"""

import json
import os


def main():
    from datasets import load_dataset

    output_dir = os.environ.get("OUTPUT_DIR", "/mnt/output")
    max_samples = int(os.environ.get("MAX_SAMPLES", "0"))

    os.makedirs(output_dir, exist_ok=True)

    print("Downloading FinanceMTEB/financial_phrasebank...")
    ds = load_dataset("FinanceMTEB/financial_phrasebank", split="train+test")

    records = []
    for row in ds:
        records.append({
            "instruction": "Classify the sentiment of this financial statement.",
            "input": row["text"],
            "output": row.get("label_text", "neutral"),
        })

    if max_samples > 0:
        records = records[:max_samples]

    outpath = os.path.join(output_dir, "financial_phrasebank.jsonl")
    with open(outpath, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    print(f"Market data: {len(records)} samples -> {outpath}")


if __name__ == "__main__":
    main()
