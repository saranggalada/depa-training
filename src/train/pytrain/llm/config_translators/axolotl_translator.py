"""
Translate normalized DEPA LLM config into an Axolotl-compatible YAML dict.

Axolotl expects a flat YAML with specific key names. This translator maps
from the schema-validated DEPA config to that format, writing the result
to a temporary YAML file that the Axolotl runner can consume.
"""


class AxolotlTranslator:

    _DTYPE_MAP = {"bf16": "bfloat16", "fp16": "float16", "fp32": "float32"}

    _FORMAT_MAP = {
        "alpaca": "alpaca",
        "sharegpt": "sharegpt",
        "completion": "completion",
        "instruction": "alpaca",
    }

    def translate(self, cfg: dict) -> dict:
        base = cfg["base_model"]
        tok = cfg.get("tokenizer", {})
        ds = cfg["dataset"]
        method = cfg["method"]
        training = cfg["training"]
        dist = cfg.get("distributed", {})
        obs = cfg.get("observability", {})

        ax = {
            "base_model": base["name_or_path"],
            "model_type": "AutoModelForCausalLM",
            "tokenizer_type": "AutoTokenizer",
            "trust_remote_code": False,
            "push_to_hub": False,
        }

        ax["load_in_4bit"] = method["type"] == "qlora"
        ax["load_in_8bit"] = False

        dtype = self._DTYPE_MAP.get(base.get("dtype", "bf16"), "bfloat16")
        ax["torch_dtype"] = dtype
        ax["bf16"] = dtype == "bfloat16" or training.get("bf16", False)
        ax["fp16"] = dtype == "float16" or training.get("fp16", False)

        ax["sequence_len"] = tok.get("max_seq_length", 2048)
        ax["pad_to_sequence_len"] = True

        ax["datasets"] = []
        fmt = self._FORMAT_MAP.get(ds.get("format", "alpaca"), "alpaca")
        for src in ds["sources"]:
            entry = {
                "path": src["path"],
                "type": fmt,
            }
            if src.get("columns"):
                entry["field_mapping"] = src["columns"]
            ax["datasets"].append(entry)

        ax["val_set_size"] = ds.get("val_split_ratio", 0.05)
        ax["dataset_prepared_path"] = "/tmp/axolotl_prepared"

        if method["type"] in ("lora", "qlora"):
            ax["adapter"] = "lora" if method["type"] == "lora" else "qlora"
            ax["lora_r"] = method.get("r", 16)
            ax["lora_alpha"] = method.get("lora_alpha", 32)
            ax["lora_dropout"] = method.get("lora_dropout", 0.05)
            ax["lora_target_modules"] = method.get("target_modules", ["q_proj", "v_proj"])
            if method.get("modules_to_save"):
                ax["lora_modules_to_save"] = method["modules_to_save"]

        ax["num_epochs"] = training.get("epochs", 3)
        ax["max_steps"] = training.get("max_steps", -1)
        ax["micro_batch_size"] = training.get("per_device_batch_size", 4)
        ax["gradient_accumulation_steps"] = training.get("gradient_accumulation_steps", 4)
        ax["learning_rate"] = training.get("learning_rate", 2e-4)
        ax["lr_scheduler"] = training.get("lr_scheduler_type", "cosine")
        ax["warmup_ratio"] = training.get("warmup_ratio", 0.03)
        ax["weight_decay"] = training.get("weight_decay", 0.01)
        ax["max_grad_norm"] = training.get("max_grad_norm", 1.0)
        ax["optimizer"] = training.get("optim", "adamw_torch")
        ax["gradient_checkpointing"] = training.get("gradient_checkpointing", True)
        ax["seed"] = training.get("seed", 42)

        if training.get("packing"):
            ax["sample_packing"] = True

        if training.get("neftune_noise_alpha") is not None:
            ax["neftune_noise_alpha"] = training["neftune_noise_alpha"]

        if dist.get("strategy", "none") == "deepspeed_zero2":
            ax["deepspeed"] = "deepspeed_configs/zero2.json"
        elif dist.get("strategy") == "deepspeed_zero3":
            ax["deepspeed"] = "deepspeed_configs/zero3.json"
        elif dist.get("strategy") == "fsdp":
            ax["fsdp"] = ["full_shard", "auto_wrap"]
            ax["fsdp_config"] = {"fsdp_transformer_layer_cls_to_wrap": "LlamaDecoderLayer"}

        output = cfg.get("output", {})
        ax["output_dir"] = output.get("path", "/mnt/remote/output")
        ax["save_safetensors"] = True

        if output.get("merge_adapter"):
            ax["merge_lora_after_training"] = True

        ax["logging_steps"] = obs.get("log_steps", 10)
        ax["eval_steps"] = obs.get("eval_steps", 50)
        ax["save_steps"] = output.get("save_steps", 0) or ax["eval_steps"]

        ax["flash_attention"] = base.get("attn_implementation") == "flash_attention_2"

        return ax
