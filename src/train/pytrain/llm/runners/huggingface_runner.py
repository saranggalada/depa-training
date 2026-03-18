"""
HuggingFace / TRL runner -- the primary reference implementation.

Uses ``SFTTrainer`` from the ``trl`` library with optional PEFT (LoRA / QLoRA)
and Opacus DP-LoRA integration.
"""

import os

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

try:
    from trl import SFTConfig, SFTTrainer
    _USE_SFT_CONFIG = True
except ImportError:
    from transformers import TrainingArguments
    from trl import SFTTrainer
    _USE_SFT_CONFIG = False

from .base_runner import FrameworkRunner
from ..model_io import safe_save_model
from ..privacy.dp_lora import maybe_wrap_dp

_DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


class HuggingFaceRunner(FrameworkRunner):

    def __init__(self):
        self.model = None
        self.trainer = None
        self.tokenizer = None
        self.epsilon_spent = None

    def run(self, config: dict, dataset, callbacks: list | None = None):
        mc = config["model_config"]
        pc = config["peft_config"]
        ta = config["training_args"]
        sc = config["sft_config"]

        torch_dtype = _DTYPE_MAP.get(mc.pop("torch_dtype", "bfloat16"), torch.bfloat16)

        quant_config = None
        raw_qc = mc.pop("quantization_config", None)
        if raw_qc:
            compute_dtype = _DTYPE_MAP.get(raw_qc.pop("bnb_4bit_compute_dtype", "bfloat16"), torch.bfloat16)
            quant_config = BitsAndBytesConfig(
                **raw_qc,
                bnb_4bit_compute_dtype=compute_dtype,
            )

        model_path = mc.pop("pretrained_model_name_or_path")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=False, use_fast=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            quantization_config=quant_config,
            trust_remote_code=False,
            **{k: v for k, v in mc.items() if k not in ("trust_remote_code",)},
        )

        peft_config_obj = None
        if pc is not None:
            from peft import LoraConfig, TaskType
            task_map = {"CAUSAL_LM": TaskType.CAUSAL_LM, "SEQ_2_SEQ_LM": TaskType.SEQ_2_SEQ_LM}
            peft_config_obj = LoraConfig(
                r=pc["r"],
                lora_alpha=pc["lora_alpha"],
                lora_dropout=pc["lora_dropout"],
                target_modules=pc["target_modules"],
                bias=pc["bias"],
                task_type=task_map.get(pc.get("task_type", "CAUSAL_LM"), TaskType.CAUSAL_LM),
                modules_to_save=pc.get("modules_to_save"),
            )

        train_ds = dataset.get("train")
        val_ds = dataset.get("val")
        collator = dataset.get("collator")

        hf_callbacks = []
        if callbacks:
            for cb in callbacks:
                hf_cb = getattr(cb, "as_hf_callback", None)
                if hf_cb:
                    hf_callbacks.append(hf_cb())

        if _USE_SFT_CONFIG:
            if sc.get("max_seq_length"):
                ta["max_length"] = sc["max_seq_length"]
            if sc.get("packing"):
                ta["packing"] = sc["packing"]
            training_args = SFTConfig(**ta)
            sft_kwargs = {}
        else:
            training_args = TrainingArguments(**ta)
            sft_kwargs = {}
            if sc.get("max_seq_length"):
                sft_kwargs["max_seq_length"] = sc["max_seq_length"]
            if sc.get("packing"):
                sft_kwargs["packing"] = sc["packing"]

        self.trainer = SFTTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            data_collator=collator,
            processing_class=self.tokenizer,
            peft_config=peft_config_obj,
            callbacks=hf_callbacks or None,
            **sft_kwargs,
        )

        self.trainer.train()

        if val_ds is not None:
            metrics = self.trainer.evaluate()
            print(f"Eval metrics: {metrics}")

    def save_model(self, config: dict):
        output_path = config.get("output", {}).get("path", "/mnt/remote/output")
        merge = config.get("output", {}).get("merge_adapter", False)
        safe_save_model(self.trainer.model, self.tokenizer, output_path, merge_adapter=merge)
