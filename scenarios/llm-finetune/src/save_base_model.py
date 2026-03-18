"""
Download a HuggingFace model and tokenizer to a local directory.

Run by the MODELLER on their own machine (which has internet access),
before encrypting and uploading model files for CCR deployment.
The Confidential Clean Room has NO network access at runtime.

Usage:
    python3 download_model.py [--model microsoft/phi-2] [--token HF_TOKEN]

Output is written to /mnt/model/ (mount-point set by docker-compose).
"""

import argparse
import os


def download_model(model_id: str, output_dir: str, token: str | None = None):
    from huggingface_hub import snapshot_download

    os.makedirs(output_dir, exist_ok=True)

    if not token:
        token = None
        os.environ.pop("HF_TOKEN", None)
        os.environ.pop("HUGGING_FACE_HUB_TOKEN", None)

    print(f"Downloading model: {model_id}")
    print(f"Output directory:  {output_dir}")

    snapshot_download(
        repo_id=model_id,
        local_dir=output_dir,
        token=token,
        ignore_patterns=["*.bin", "*.pt", "*.pth", "*.pkl", "*.gguf",
                         "original/**", "consolidated.*"],
    )

    blocked = {".bin", ".pt", ".pth", ".pkl", ".pickle", ".gguf"}
    for root, _dirs, files in os.walk(output_dir):
        for f in files:
            if os.path.splitext(f)[1].lower() in blocked:
                path = os.path.join(root, f)
                os.remove(path)
                print(f"  Removed blocked format: {f}")

    print(f"\nModel downloaded to {output_dir}")
    print("Files:")
    for root, _dirs, files in os.walk(output_dir):
        for f in sorted(files):
            rel = os.path.relpath(os.path.join(root, f), output_dir)
            size_mb = os.path.getsize(os.path.join(root, f)) / (1024 * 1024)
            print(f"  {rel} ({size_mb:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(description="Download HuggingFace model for offline CCR use")
    parser.add_argument("--model", default=os.environ.get("HF_MODEL_ID", "microsoft/phi-2"),
                        help="HuggingFace model ID")
    parser.add_argument("--output", default="/mnt/model",
                        help="Output directory (default: /mnt/model)")
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN"),
                        help="HuggingFace token for gated models")
    args = parser.parse_args()
    download_model(args.model, args.output, args.token)


if __name__ == "__main__":
    main()
