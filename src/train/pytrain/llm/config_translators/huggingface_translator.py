"""
Translate normalized DEPA LLM config into HuggingFace TrainingArguments
plus SFTTrainer / PeftConfig kwargs.

Returns a dict with three top-level keys:
  - ``training_args``: kwargs for ``transformers.TrainingArguments``
  - ``peft_config``: kwargs for ``peft.LoraConfig`` (or None for full fine-tune)
  - ``model_config``: kwargs for ``AutoModelForCausalLM.from_pretrained``
  - ``sft_config``: extra kwargs for ``trl.SFTTrainer``
"""


class HuggingFaceTranslator:

    _DTYPE_MAP = {
        "bf16": "bfloat16",
        "fp16": "float16",
        "fp32": "float32",
    }

    _OPTIM_MAP = {
        "adamw_torch": "adamw_torch",
        "adamw_hf": "adamw_hf",
        "adamw_torch_fused": "adamw_torch_fused",
        "adamw_8bit": "adamw_bnb_8bit",
        "paged_adamw_8bit": "paged_adamw_8bit",
        "paged_adamw_32bit": "paged_adamw_32bit",
        "sgd": "sgd",
        "adafactor": "adafactor",
        "lion_8bit": "lion_8bit",
        "lion_32bit": "lion_32bit",
    }

    def translate(self, cfg: dict) -> dict:
        base = cfg["base_model"]
        tok = cfg.get("tokenizer", {})
        ds = cfg["dataset"]
        method = cfg["method"]
        training = cfg["training"]
        dist = cfg.get("distributed", {})
        obs = cfg.get("observability", {})
        output = cfg.get("output", {})

        model_kwargs = {
            "pretrained_model_name_or_path": base["name_or_path"],
            "trust_remote_code": False,
            "torch_dtype": self._DTYPE_MAP.get(base.get("dtype", "bf16"), "bfloat16"),
        }

        attn = base.get("attn_implementation")
        if attn and attn != "eager":
            model_kwargs["attn_implementation"] = attn

        if method["type"] == "qlora":
            model_kwargs["quantization_config"] = {
                "load_in_4bit": True,
                "bnb_4bit_quant_type": method.get("bnb_4bit_quant_type", "nf4"),
                "bnb_4bit_compute_dtype": self._DTYPE_MAP.get(
                    method.get("bnb_4bit_compute_dtype", "bf16"), "bfloat16"
                ),
                "bnb_4bit_use_double_quant": True,
            }

        if base.get("max_memory"):
            model_kwargs["max_memory"] = base["max_memory"]
            model_kwargs["device_map"] = "auto"

        peft_config = None
        if method["type"] in ("lora", "qlora"):
            target_modules = method.get("target_modules", ["q_proj", "v_proj"])
            if target_modules == ["all-linear"]:
                target_modules = "all-linear"
            peft_config = {
                "r": method.get("r", 16),
                "lora_alpha": method.get("lora_alpha", 32),
                "lora_dropout": method.get("lora_dropout", 0.05),
                "target_modules": target_modules,
                "bias": method.get("bias", "none"),
                "task_type": method.get("task_type", "CAUSAL_LM"),
            }
            if method.get("modules_to_save"):
                peft_config["modules_to_save"] = method["modules_to_save"]

        training_args = {
            "output_dir": output.get("path", "/mnt/remote/output"),
            "num_train_epochs": training.get("epochs", 3),
            "max_steps": training.get("max_steps", -1),
            "per_device_train_batch_size": training.get("per_device_batch_size", 4),
            "per_device_eval_batch_size": training.get("per_device_batch_size", 4),
            "gradient_accumulation_steps": training.get("gradient_accumulation_steps", 4),
            "learning_rate": training.get("learning_rate", 2e-4),
            "lr_scheduler_type": training.get("lr_scheduler_type", "cosine"),
            "warmup_ratio": training.get("warmup_ratio", 0.03),
            "warmup_steps": training.get("warmup_steps", 0),
            "weight_decay": training.get("weight_decay", 0.01),
            "max_grad_norm": training.get("max_grad_norm", 1.0),
            "bf16": training.get("bf16", True),
            "fp16": training.get("fp16", False),
            "gradient_checkpointing": training.get("gradient_checkpointing", True),
            "optim": self._OPTIM_MAP.get(training.get("optim", "adamw_torch"), "adamw_torch"),
            "seed": training.get("seed", 42),
            "logging_steps": obs.get("log_steps", 10),
            "eval_strategy": obs.get("eval_strategy", "steps"),
            "eval_steps": obs.get("eval_steps", 50),
            "save_strategy": "steps",
            "save_steps": output.get("save_steps", 0) or obs.get("eval_steps", 50),
            "save_total_limit": output.get("save_total_limit", 2),
            "push_to_hub": False,
            "report_to": "none",
            "remove_unused_columns": False,
            "dataloader_pin_memory": True,
        }

        if training.get("gradient_checkpointing"):
            training_args["gradient_checkpointing_kwargs"] = {"use_reentrant": False}

        strategy = dist.get("strategy", "none")
        if strategy == "deepspeed_zero2":
            training_args["deepspeed"] = self._deepspeed_config(2, dist)
        elif strategy == "deepspeed_zero3":
            training_args["deepspeed"] = self._deepspeed_config(3, dist)
        elif strategy == "fsdp":
            training_args["fsdp"] = "full_shard auto_wrap"
            training_args["fsdp_config"] = {
                "fsdp_transformer_layer_cls_to_wrap": "LlamaDecoderLayer"
            }

        sft_config = {
            "max_seq_length": tok.get("max_seq_length", 2048),
            "packing": training.get("packing", False),
        }

        if training.get("neftune_noise_alpha") is not None:
            sft_config["neftune_noise_alpha"] = training["neftune_noise_alpha"]

        dataset_format = ds.get("format", "alpaca")
        sft_config["dataset_text_field"] = "text" if dataset_format == "completion" else None
        sft_config["dataset_kwargs"] = {"add_special_tokens": False}

        return {
            "model_config": model_kwargs,
            "peft_config": peft_config,
            "training_args": training_args,
            "sft_config": sft_config,
        }

    @staticmethod
    def _deepspeed_config(stage, dist):
        ds_cfg = {
            "zero_optimization": {
                "stage": stage,
                "offload_optimizer": {"device": "cpu", "pin_memory": True},
                "overlap_comm": True,
                "contiguous_gradients": True,
            },
            "bf16": {"enabled": True},
            "train_micro_batch_size_per_gpu": "auto",
            "gradient_accumulation_steps": "auto",
            "gradient_clipping": "auto",
        }
        if stage == 3:
            ds_cfg["zero_optimization"]["offload_param"] = {"device": "cpu", "pin_memory": True}
            ds_cfg["zero_optimization"]["stage3_gather_16bit_weights_on_model_save"] = True
        user_overrides = dist.get("deepspeed_config", {})
        ds_cfg.update(user_overrides)
        return ds_cfg
