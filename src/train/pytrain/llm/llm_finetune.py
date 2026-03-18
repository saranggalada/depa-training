"""
LLM_Finetune -- top-level pipeline task for LLM fine-tuning inside a
DEPA Confidential Clean Room.

Extends ``TaskBase`` so it plugs into the existing pytrain pipeline
executor via the standard ``{"name": "LLM_Finetune", "config": {...}}``
pipeline config entry.
"""

import hashlib
import time

from ..task_base import TaskBase
from .config_validator import validate, config_hash
from .config_translators import get_translator
from .data.dataset_loader import load_and_merge_datasets
from .data.tokenizer_pipeline import TokenizerPipeline
from .runners import create_runner
from .model_io import compute_output_hash
from .privacy.dp_lora import maybe_wrap_dp
from .observability.mlflow_tracker import MLflowTracker
from .observability.audit_logger import AuditLogger


def _dataset_hashes(cfg: dict) -> dict[str, str]:
    """Compute SHA-256 of each source data path for the audit record."""
    hashes = {}
    for src in cfg["dataset"]["sources"]:
        path = src["path"]
        try:
            import os
            if os.path.isfile(path):
                h = hashlib.sha256()
                with open(path, "rb") as f:
                    for chunk in iter(lambda: f.read(1 << 20), b""):
                        h.update(chunk)
                hashes[path] = h.hexdigest()
            else:
                hashes[path] = "directory"
        except Exception:
            hashes[path] = "unavailable"
    return hashes


class LLM_Finetune(TaskBase):
    """Config-driven LLM fine-tuning task.

    Orchestrates: validate config -> load data -> tokenize ->
    select framework runner -> optionally apply DP -> train ->
    save model -> log audit.
    """

    def execute(self, config):
        print("=" * 60)
        print("  DEPA CCR — LLM Fine-Tuning")
        print("=" * 60)

        # ── 1. Validate & normalize config ───────────────────────
        cfg = validate(config)
        framework = cfg["framework"]
        cfg_hash = cfg["_config_hash"]
        print(f"Config validated | framework={framework} | hash={cfg_hash[:16]}...")

        # ── 2. Initialize observability ──────────────────────────
        ds_hashes = _dataset_hashes(cfg)
        audit = AuditLogger(cfg)
        tracker = MLflowTracker(cfg)

        audit.log_event("run_start", {
            "config_hash": cfg_hash,
            "dataset_hashes": ds_hashes,
            "framework": framework,
        })
        tracker.start_run()

        try:
            # ── 3. Load & merge multi-TDP datasets ──────────────
            t0 = time.time()
            raw_dataset = load_and_merge_datasets(cfg)
            print(f"Loaded {len(raw_dataset)} samples from {len(cfg['dataset']['sources'])} source(s) "
                  f"in {time.time() - t0:.1f}s")

            # ── 4. Tokenize ─────────────────────────────────────
            needs_pre_tokenization = framework in ("huggingface", "pytorch")
            tok_pipeline = TokenizerPipeline(cfg)

            if needs_pre_tokenization:
                t0 = time.time()
                tokenized = tok_pipeline.tokenize_dataset(raw_dataset)
                train_ds, val_ds = tok_pipeline.split_train_val(
                    tokenized,
                    cfg["dataset"].get("val_split_ratio", 0.05),
                )
                print(f"Tokenized | train={len(train_ds)} "
                      f"| val={len(val_ds) if val_ds else 0} "
                      f"| {time.time() - t0:.1f}s")
                dataset_payload = {"train": train_ds, "val": val_ds, "collator": tok_pipeline.get_data_collator()}
            else:
                dataset_payload = {"raw": raw_dataset, "tokenizer_pipeline": tok_pipeline}

            # ── 5. Translate config for the chosen framework ─────
            translator = get_translator(framework)
            fw_config = translator.translate(cfg)

            # ── 6. Create runner & inject DP if requested ────────
            runner = create_runner(framework)
            callbacks = [tracker, audit]

            # ── 7. Train ─────────────────────────────────────────
            print(f"\nStarting {framework} fine-tuning...")
            t0 = time.time()
            runner.run(fw_config, dataset_payload, callbacks=callbacks)
            train_time = time.time() - t0
            print(f"Training complete in {train_time:.1f}s")

            # ── 8. Save model ────────────────────────────────────
            runner.save_model(cfg)
            model_hash = compute_output_hash(cfg["output"].get("path", "/mnt/remote/output"))
            print(f"Model saved | hash={model_hash[:16]}...")

            # ── 9. Final audit ───────────────────────────────────
            privacy_cfg = cfg.get("privacy", {})
            audit.log_event("run_complete", {
                "model_hash": model_hash,
                "train_time_seconds": round(train_time, 2),
                "epsilon_spent": getattr(runner, "epsilon_spent", None) if privacy_cfg.get("enabled") else None,
            })
            tracker.end_run()

            print("\n" + "=" * 60)
            print("  CCR LLM Fine-Tuning complete!")
            print("=" * 60)

        except Exception as e:
            audit.log_event("run_failed", {"error": str(e)})
            tracker.end_run(status="FAILED")
            print(f"LLM Fine-Tuning failed: {e}")
            raise
