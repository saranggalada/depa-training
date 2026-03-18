"""
Immutable, tamper-evident audit logger for LLM fine-tuning runs.

Produces an append-only JSONL file where each entry:
  - Is timestamped
  - Includes the SHA-256 hash of the previous entry (hash chain)
  - Is HMAC-signed with an enclave-derived key (if available) or a
    run-local secret for local/dev runs
"""

import hashlib
import hmac
import json
import os
import time
import secrets


def _canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


class AuditLogger:
    """Hash-chained, HMAC-signed JSONL audit log."""

    def __init__(self, cfg: dict):
        output_dir = cfg.get("output", {}).get("path", "/mnt/remote/output")
        os.makedirs(output_dir, exist_ok=True)
        self.log_path = os.path.join(output_dir, "audit.jsonl")

        self._signing_key = self._load_signing_key()
        self._prev_hash = hashlib.sha256(b"genesis").hexdigest()
        self._seq = 0

    @staticmethod
    def _load_signing_key() -> bytes:
        """Load enclave-derived signing key, or generate an ephemeral one
        for local / dev runs."""
        key_path = os.environ.get("ENCLAVE_SIGNING_KEY_PATH")
        if key_path and os.path.isfile(key_path):
            with open(key_path, "rb") as f:
                return f.read()
        return secrets.token_bytes(32)

    def _sign(self, payload_bytes: bytes) -> str:
        return hmac.new(self._signing_key, payload_bytes, hashlib.sha256).hexdigest()

    def log_event(self, event_type: str, data: dict | None = None):
        """Append a signed, hash-chained event to the audit log.

        Parameters
        ----------
        event_type : str
            E.g. ``run_start``, ``epoch_complete``, ``run_complete``,
            ``run_failed``.
        data : dict, optional
            Arbitrary event payload (must be JSON-serializable).
        """
        entry = {
            "seq": self._seq,
            "timestamp": time.time(),
            "event": event_type,
            "data": data or {},
            "prev_hash": self._prev_hash,
        }

        payload = _canonical(entry)
        entry["hmac"] = self._sign(payload.encode("utf-8"))

        self._prev_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        self._seq += 1

        with open(self.log_path, "a") as f:
            f.write(json.dumps(entry, sort_keys=True) + "\n")

    def on_log(self, metrics: dict):
        """Callback interface for the training loop."""
        pass

    def on_epoch_end(self, metrics: dict):
        self.log_event("epoch_complete", metrics)

    def as_hf_callback(self):
        """Return a HuggingFace TrainerCallback that logs epoch events."""
        logger = self

        from transformers import TrainerCallback

        class _AuditHFCallback(TrainerCallback):
            def on_epoch_end(self, args, state, control, **kwargs):
                logger.log_event("epoch_complete", {
                    "epoch": state.epoch,
                    "step": state.global_step,
                })

            def on_train_end(self, args, state, control, **kwargs):
                logger.log_event("training_ended", {
                    "total_steps": state.global_step,
                    "epoch": state.epoch,
                })

        return _AuditHFCallback()
