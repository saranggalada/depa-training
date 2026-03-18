"""
Translate normalized DEPA LLM config into a torchtune recipe config dict.

Torchtune uses YAML configs keyed by component (model, tokenizer, dataset,
optimizer, etc.) and a recipe name. This translator maps the DEPA schema
into that format.
"""


class TorchtuneTranslator:

    _DTYPE_MAP = {"bf16": "bf16", "fp16": "fp16", "fp32": "fp32"}

    _RECIPE_MAP = {
        "full": "full_finetune_single_device",
        "lora": "lora_finetune_single_device",
        "qlora": "lora_finetune_single_device",
    }

    _RECIPE_MAP_MULTI = {
        "full": "full_finetune_distributed",
        "lora": "lora_finetune_distributed",
        "qlora": "lora_finetune_distributed",
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
        multi_gpu = dist.get("num_gpus", 1) > 1

        recipe_map = self._RECIPE_MAP_MULTI if multi_gpu else self._RECIPE_MAP
        recipe = recipe_map.get(method["type"], "lora_finetune_single_device")

        tt = {"_recipe": recipe}

        tt["model"] = {
            "_component_": self._model_component(base, method),
            "model_id": base["name_or_path"],
        }

        if method["type"] in ("lora", "qlora"):
            tt["model"]["lora_attn_modules"] = self._translate_target_modules(
                method.get("target_modules", ["q_proj", "v_proj"])
            )
            tt["model"]["lora_rank"] = method.get("r", 16)
            tt["model"]["lora_alpha"] = method.get("lora_alpha", 32)
            tt["model"]["lora_dropout"] = method.get("lora_dropout", 0.05)
            tt["model"]["apply_lora_to_mlp"] = any(
                m in method.get("target_modules", [])
                for m in ("gate_proj", "up_proj", "down_proj")
            )
            if method["type"] == "qlora":
                tt["model"]["quantize_base"] = True

        tt["tokenizer"] = {
            "_component_": "torchtune.models.llama3.llama3_tokenizer",
            "path": tok.get("name_or_path", base["name_or_path"]),
            "max_seq_len": tok.get("max_seq_length", 2048),
        }

        tt["dataset"] = {
            "_component_": self._dataset_component(ds.get("format", "alpaca")),
            "source": ds["sources"][0]["path"] if ds["sources"] else "",
            "split": "train",
        }
        if ds.get("val_split_ratio", 0) > 0:
            tt["dataset"]["val_split"] = ds["val_split_ratio"]

        tt["seed"] = training.get("seed", 42)
        tt["shuffle"] = ds.get("shuffle", True)

        tt["epochs"] = training.get("epochs", 3)
        tt["max_steps_per_epoch"] = training.get("max_steps", -1) if training.get("max_steps", -1) > 0 else None
        tt["batch_size"] = training.get("per_device_batch_size", 4)
        tt["gradient_accumulation_steps"] = training.get("gradient_accumulation_steps", 4)

        tt["optimizer"] = {
            "_component_": "torch.optim.AdamW",
            "lr": training.get("learning_rate", 2e-4),
            "weight_decay": training.get("weight_decay", 0.01),
        }

        tt["lr_scheduler"] = {
            "_component_": "torchtune.training.lr_schedulers.get_cosine_schedule_with_warmup",
            "num_warmup_steps": max(1, int(training.get("warmup_ratio", 0.03) * training.get("epochs", 3) * 100)),
        }

        tt["loss"] = {"_component_": "torchtune.modules.loss.CEWithChunkedOutputLoss"}

        tt["dtype"] = self._DTYPE_MAP.get(base.get("dtype", "bf16"), "bf16")
        tt["enable_activation_checkpointing"] = training.get("gradient_checkpointing", True)

        tt["output_dir"] = output.get("path", "/mnt/remote/output")
        tt["log_every_n_steps"] = obs.get("log_steps", 10)

        tt["metric_logger"] = {
            "_component_": "torchtune.training.metric_logging.StdoutLogger",
        }

        return tt

    @staticmethod
    def _model_component(base, method):
        model_name = base["name_or_path"].lower()
        if "llama-3" in model_name or "llama3" in model_name:
            family = "llama3"
        elif "llama-2" in model_name or "llama2" in model_name:
            family = "llama2"
        elif "mistral" in model_name:
            family = "mistral"
        elif "phi" in model_name:
            family = "phi3"
        elif "gemma" in model_name:
            family = "gemma"
        else:
            family = "llama3"

        if method["type"] in ("lora", "qlora"):
            return f"torchtune.models.{family}.lora_{family}"
        return f"torchtune.models.{family}.{family}"

    @staticmethod
    def _translate_target_modules(modules):
        mapping = {
            "q_proj": "q_proj", "k_proj": "k_proj",
            "v_proj": "v_proj", "o_proj": "output_proj",
        }
        return [mapping.get(m, m) for m in modules if m in mapping]

    @staticmethod
    def _dataset_component(fmt):
        components = {
            "alpaca": "torchtune.datasets.alpaca_dataset",
            "sharegpt": "torchtune.datasets.chat_dataset",
            "completion": "torchtune.datasets.text_completion_dataset",
            "instruction": "torchtune.datasets.alpaca_dataset",
        }
        return components.get(fmt, "torchtune.datasets.alpaca_dataset")
