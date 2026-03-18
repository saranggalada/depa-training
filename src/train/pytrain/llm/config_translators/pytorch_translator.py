"""
Translate normalized DEPA LLM config into a flat dict consumed by the
raw-PyTorch training loop runner.

The PyTorch runner uses HuggingFace for model/tokenizer loading and PEFT
for adapter application, but runs a hand-written training loop rather
than relying on Trainer.
"""


class PytorchTranslator:

    _DTYPE_MAP = {
        "bf16": "bfloat16",
        "fp16": "float16",
        "fp32": "float32",
    }

    def translate(self, cfg: dict) -> dict:
        base = cfg["base_model"]
        tok = cfg.get("tokenizer", {})
        ds = cfg["dataset"]
        method = cfg["method"]
        training = cfg["training"]
        output = cfg.get("output", {})
        obs = cfg.get("observability", {})

        pt = {
            "model_name_or_path": base["name_or_path"],
            "tokenizer_name_or_path": tok.get("name_or_path", base["name_or_path"]),
            "torch_dtype": self._DTYPE_MAP.get(base.get("dtype", "bf16"), "bfloat16"),
            "trust_remote_code": False,
            "max_seq_length": tok.get("max_seq_length", 2048),
        }

        attn = base.get("attn_implementation")
        if attn and attn != "eager":
            pt["attn_implementation"] = attn

        pt["method"] = method["type"]
        if method["type"] in ("lora", "qlora"):
            pt["lora_r"] = method.get("r", 16)
            pt["lora_alpha"] = method.get("lora_alpha", 32)
            pt["lora_dropout"] = method.get("lora_dropout", 0.05)
            pt["lora_target_modules"] = method.get("target_modules", ["q_proj", "v_proj"])
            pt["lora_bias"] = method.get("bias", "none")
            pt["lora_task_type"] = method.get("task_type", "CAUSAL_LM")
            if method.get("modules_to_save"):
                pt["lora_modules_to_save"] = method["modules_to_save"]

        if method["type"] == "qlora":
            pt["load_in_4bit"] = True
            pt["bnb_4bit_quant_type"] = method.get("bnb_4bit_quant_type", "nf4")
            pt["bnb_4bit_compute_dtype"] = self._DTYPE_MAP.get(
                method.get("bnb_4bit_compute_dtype", "bf16"), "bfloat16"
            )

        pt["epochs"] = training.get("epochs", 3)
        pt["max_steps"] = training.get("max_steps", -1)
        pt["batch_size"] = training.get("per_device_batch_size", 4)
        pt["gradient_accumulation_steps"] = training.get("gradient_accumulation_steps", 4)
        pt["learning_rate"] = training.get("learning_rate", 2e-4)
        pt["lr_scheduler_type"] = training.get("lr_scheduler_type", "cosine")
        pt["warmup_ratio"] = training.get("warmup_ratio", 0.03)
        pt["weight_decay"] = training.get("weight_decay", 0.01)
        pt["max_grad_norm"] = training.get("max_grad_norm", 1.0)
        pt["gradient_checkpointing"] = training.get("gradient_checkpointing", True)
        pt["seed"] = training.get("seed", 42)

        pt["use_amp"] = training.get("bf16", True) or training.get("fp16", False)
        pt["amp_dtype"] = "bfloat16" if training.get("bf16", True) else "float16"

        pt["output_dir"] = output.get("path", "/mnt/remote/output")
        pt["merge_adapter"] = output.get("merge_adapter", False)

        pt["log_steps"] = obs.get("log_steps", 10)
        pt["eval_steps"] = obs.get("eval_steps", 50)

        pt["dataset_sources"] = ds["sources"]
        pt["dataset_format"] = ds.get("format", "alpaca")
        pt["val_split_ratio"] = ds.get("val_split_ratio", 0.05)

        return pt
