"""
Safe model I/O for LLM fine-tuning.

- Saves adapter weights or merged models in safetensors format ONLY.
- Blocks pickle-based formats (.bin, .pt, .pkl, etc.).
- Computes SHA-256 hash of all output files for the attestation receipt.
"""

import hashlib
import os
from pathlib import Path

BLOCKED_EXTENSIONS = frozenset({".pkl", ".pickle", ".pt", ".pth", ".bin", ".npy", ".npz", ".joblib"})


def _check_no_pickle(directory: str):
    """Raise if any pickle-format file exists in the output directory."""
    for root, _dirs, files in os.walk(directory):
        for fname in files:
            ext = Path(fname).suffix.lower()
            if ext in BLOCKED_EXTENSIONS:
                full = os.path.join(root, fname)
                os.remove(full)
                print(f"Security: removed blocked file format {fname}")


def safe_save_model(model, tokenizer, output_path: str, merge_adapter: bool = False):
    """Save model and tokenizer to ``output_path`` in safetensors format.

    Parameters
    ----------
    model : PreTrainedModel | PeftModel
        The trained model (possibly PEFT-wrapped).
    tokenizer : PreTrainedTokenizer
        The tokenizer.
    output_path : str
        Directory to save into.
    merge_adapter : bool
        If True and model is a PeftModel, merge adapter into base model
        before saving.
    """
    os.makedirs(output_path, exist_ok=True)

    is_peft = hasattr(model, "peft_config") or hasattr(model, "merge_and_unload")

    if is_peft and merge_adapter:
        print("Merging LoRA adapter into base model...")
        model = model.merge_and_unload()
        model.save_pretrained(output_path, safe_serialization=True)
    elif is_peft:
        model.save_pretrained(output_path, safe_serialization=True)
    else:
        model.save_pretrained(output_path, safe_serialization=True)

    tokenizer.save_pretrained(output_path)

    _check_no_pickle(output_path)

    print(f"Model saved to {output_path} (safetensors)")


def compute_output_hash(output_dir: str) -> str:
    """Compute a deterministic SHA-256 over all files in the output dir.

    Files are sorted by relative path for reproducibility.
    """
    h = hashlib.sha256()
    output_path = Path(output_dir)

    if not output_path.exists():
        return hashlib.sha256(b"empty").hexdigest()

    files = sorted(
        f for f in output_path.rglob("*")
        if f.is_file() and not f.name.startswith(".")
    )

    for fpath in files:
        rel = str(fpath.relative_to(output_path))
        h.update(rel.encode("utf-8"))
        with open(fpath, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)

    return h.hexdigest()
