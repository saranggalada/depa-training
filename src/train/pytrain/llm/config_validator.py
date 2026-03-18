"""
Configuration validator and normalizer for LLM fine-tuning.

Accepts YAML or JSON input, validates against the JSON Schema,
injects hardcoded security overrides, and produces a normalized
config dict with a SHA-256 content hash for the attestation receipt.
"""

import copy
import hashlib
import json
import os
from pathlib import Path

import yaml
from jsonschema import Draft202012Validator, ValidationError

_SCHEMA_PATH = Path(__file__).parent / "schemas" / "llm_finetune_schema.json"

SECURITY_OVERRIDES = {
    "trust_remote_code": False,
    "push_to_hub": False,
    "report_to": "none",
}

BLOCKED_FILE_EXTENSIONS = frozenset({
    ".pkl", ".pickle", ".pt", ".pth", ".bin", ".npy", ".npz", ".joblib",
})

_validator_instance = None


def _get_validator():
    global _validator_instance
    if _validator_instance is None:
        with open(_SCHEMA_PATH, "r") as f:
            schema = json.load(f)
        _validator_instance = Draft202012Validator(schema)
    return _validator_instance


def _load_raw(config_input):
    """Parse YAML/JSON string, file path, or pass-through dict."""
    if isinstance(config_input, dict):
        return copy.deepcopy(config_input)

    text = config_input
    if os.path.isfile(config_input):
        with open(config_input, "r") as f:
            text = f.read()

    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        pass

    return yaml.safe_load(text)


_NUMERIC_TRAINING_KEYS = frozenset({
    "learning_rate", "warmup_ratio", "weight_decay", "max_grad_norm",
    "lora_dropout", "lora_alpha",
})


def _coerce_numeric_strings(cfg):
    """Convert string representations of numbers to actual floats/ints
    (e.g. YAML may serialize '2e-4' as a string)."""
    for section_key in ("training", "method", "privacy"):
        section = cfg.get(section_key, {})
        if not isinstance(section, dict):
            continue
        for k, v in section.items():
            if isinstance(v, str) and k in _NUMERIC_TRAINING_KEYS:
                try:
                    section[k] = float(v)
                except ValueError:
                    pass
    return cfg


def _apply_defaults(cfg):
    """Fill in essential defaults that the schema declares but that
    callers frequently omit."""
    cfg.setdefault("tokenizer", {})
    if not cfg["tokenizer"].get("name_or_path"):
        cfg["tokenizer"]["name_or_path"] = cfg["base_model"]["name_or_path"]

    cfg.setdefault("distributed", {"strategy": "none", "num_gpus": 1})
    cfg.setdefault("privacy", {"enabled": False})
    cfg.setdefault("observability", {})
    cfg["observability"].setdefault("log_steps", 10)
    cfg["observability"].setdefault("eval_steps", 50)
    cfg["observability"].setdefault("eval_strategy", "steps")
    cfg["observability"].setdefault("metrics", ["loss", "eval_loss", "perplexity"])
    cfg["observability"].setdefault("mlflow", {"enabled": True, "experiment_name": "depa-llm-finetune"})

    cfg.setdefault("output", {})
    cfg["output"].setdefault("path", "/mnt/remote/output")
    cfg["output"].setdefault("merge_adapter", False)

    method = cfg.get("method", {})
    if method.get("type") in ("lora", "qlora"):
        method.setdefault("r", 16)
        method.setdefault("lora_alpha", 32)
        method.setdefault("lora_dropout", 0.05)
        method.setdefault("target_modules", ["q_proj", "v_proj"])
        method.setdefault("bias", "none")
        method.setdefault("task_type", "CAUSAL_LM")

    training = cfg.setdefault("training", {})
    training.setdefault("epochs", 3)
    training.setdefault("per_device_batch_size", 4)
    training.setdefault("gradient_accumulation_steps", 4)
    training.setdefault("learning_rate", 2e-4)
    training.setdefault("lr_scheduler_type", "cosine")
    training.setdefault("warmup_ratio", 0.03)
    training.setdefault("weight_decay", 0.01)
    training.setdefault("max_grad_norm", 1.0)
    training.setdefault("bf16", True)
    training.setdefault("fp16", False)
    training.setdefault("gradient_checkpointing", True)
    training.setdefault("optim", "adamw_torch")
    training.setdefault("seed", 42)

    return cfg


def _apply_security_overrides(cfg):
    """Inject non-negotiable security settings."""
    cfg["_security"] = dict(SECURITY_OVERRIDES)

    output_path = cfg.get("output", {}).get("path", "")
    for ext in BLOCKED_FILE_EXTENSIONS:
        if output_path.endswith(ext):
            raise ValidationError(
                f"Output path uses blocked file extension '{ext}'. "
                "Only safetensors format is allowed."
            )

    if cfg.get("privacy", {}).get("enabled") and cfg["method"]["type"] == "full":
        raise ValidationError(
            "DP-LoRA privacy requires method.type to be 'lora' or 'qlora', not 'full'."
        )

    return cfg


def canonical_json(obj):
    """Deterministic JSON serialization for hashing."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def config_hash(cfg):
    """SHA-256 of the canonical JSON representation."""
    return hashlib.sha256(canonical_json(cfg).encode("utf-8")).hexdigest()


def validate(config_input):
    """
    Validate, normalize, and hash an LLM fine-tuning configuration.

    Parameters
    ----------
    config_input : dict | str
        A config dict, a JSON/YAML string, or a file path.

    Returns
    -------
    dict
        Normalized config with an added ``_config_hash`` field.

    Raises
    ------
    jsonschema.ValidationError
        If the config fails schema validation.
    """
    raw = _load_raw(config_input)
    raw = _coerce_numeric_strings(raw)
    cfg = _apply_defaults(raw)

    validator = _get_validator()
    validator.validate(cfg)

    cfg = _apply_security_overrides(cfg)
    cfg["_config_hash"] = config_hash(cfg)
    return cfg
