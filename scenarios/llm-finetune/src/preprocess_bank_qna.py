import json
import os

# mimics bank's internal compliance Q&A dataset
QA_PAIRS = [
    ("What is KYC?",
     "Know Your Customer (KYC) is a process where banks verify the identity of their clients to prevent fraud, money laundering, and terrorism financing."),
    ("What are the AML reporting thresholds?",
     "Banks must file a Currency Transaction Report (CTR) for transactions exceeding $10,000 and a Suspicious Activity Report (SAR) for suspicious transactions of any amount."),
    ("What is the Basel III capital requirement?",
     "Basel III requires banks to maintain a minimum Common Equity Tier 1 ratio of 4.5%, a Tier 1 capital ratio of 6%, and a total capital ratio of 8%."),
    ("Explain the Volcker Rule.",
     "The Volcker Rule prohibits banks from engaging in proprietary trading and limits their investments in hedge funds and private equity funds."),
    ("What is a stress test?",
     "A stress test evaluates a bank's ability to withstand adverse economic scenarios by modeling the impact on capital, liquidity, and profitability."),
    ("What is PSD2?",
     "The Payment Services Directive 2 (PSD2) is an EU regulation that requires banks to open their payment infrastructure to third-party providers and mandates strong customer authentication."),
    ("Define operational risk.",
     "Operational risk is the risk of loss from inadequate or failed internal processes, people, systems, or external events, including fraud and cyber attacks."),
    ("What is GDPR compliance for banks?",
     "Banks must ensure data minimization, purpose limitation, lawful processing, right to erasure, data portability, and appointment of a Data Protection Officer under GDPR."),
]


def main():
    output_dir = os.environ.get("OUTPUT_DIR", "/mnt/output")
    max_samples = int(os.environ.get("MAX_SAMPLES", "0"))

    os.makedirs(output_dir, exist_ok=True)

    records = []
    for q, a in QA_PAIRS:
        records.append({"instruction": q, "input": "", "output": a})

    records = records * 20
    if max_samples > 0:
        records = records[:max_samples]

    outpath = os.path.join(output_dir, "compliance_qa.jsonl")
    with open(outpath, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    print(f"Bank data: {len(records)} samples -> {outpath}")


if __name__ == "__main__":
    main()
