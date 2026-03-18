"""
TDP 2 — FinTech Provider: prepare financial instruction-response dataset.

Downloads sujet-ai/Sujet-Finance-Instruct-177k from HuggingFace,
converts to Alpaca JSONL, and writes to /mnt/output/.

Run by the FinTech Provider on their own machine before encrypting.
"""

import json
import os


def main():
    from datasets import load_dataset

    output_dir = os.environ.get("OUTPUT_DIR", "/mnt/output")
    max_samples = int(os.environ.get("MAX_SAMPLES", "0"))

    os.makedirs(output_dir, exist_ok=True)

    print("Downloading sujet-ai/Sujet-Finance-Instruct-177k...")
    ds = load_dataset("sujet-ai/Sujet-Finance-Instruct-177k", split="train")

    records = []
    for row in ds:
        instruction = row.get("system_prompt", "") or ""
        user_input = row.get("user_prompt", "") or row.get("inputs", "")
        answer = row.get("answer", "")

        if not user_input and not instruction:
            continue

        records.append({
            "instruction": instruction if instruction else "Answer the following financial question.",
            "input": user_input,
            "output": answer,
        })

    if max_samples > 0:
        records = records[:max_samples]

    outpath = os.path.join(output_dir, "finance_instruct.jsonl")
    with open(outpath, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    print(f"Fintech data: {len(records)} samples -> {outpath}")


if __name__ == "__main__":
    main()
