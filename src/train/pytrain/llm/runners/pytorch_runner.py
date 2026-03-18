"""
Raw PyTorch runner -- a hand-written training loop that uses HuggingFace
for model/tokenizer loading and PEFT for adapter application, but does
NOT use Trainer.  This gives full control over the training loop, which
is required for clean Opacus DP-LoRA integration.
"""

import math
import os
import time

import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    get_scheduler,
)

from .base_runner import FrameworkRunner
from ..model_io import safe_save_model
from ..privacy.dp_lora import maybe_wrap_dp

_DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


class PytorchRunner(FrameworkRunner):

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.epsilon_spent = None

    def run(self, config: dict, dataset, callbacks: list | None = None):
        pt = config
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch_dtype = _DTYPE_MAP.get(pt.get("torch_dtype", "bfloat16"), torch.bfloat16)

        self.tokenizer = AutoTokenizer.from_pretrained(
            pt["tokenizer_name_or_path"], trust_remote_code=False, use_fast=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs = {"torch_dtype": torch_dtype, "trust_remote_code": False}
        if pt.get("attn_implementation"):
            model_kwargs["attn_implementation"] = pt["attn_implementation"]

        if pt.get("load_in_4bit"):
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=pt.get("bnb_4bit_quant_type", "nf4"),
                bnb_4bit_compute_dtype=_DTYPE_MAP.get(pt.get("bnb_4bit_compute_dtype", "bfloat16"), torch.bfloat16),
                bnb_4bit_use_double_quant=True,
            )

        self.model = AutoModelForCausalLM.from_pretrained(
            pt["model_name_or_path"], **model_kwargs,
        )

        if pt.get("method") in ("lora", "qlora"):
            from peft import LoraConfig, get_peft_model, TaskType
            task_map = {"CAUSAL_LM": TaskType.CAUSAL_LM, "SEQ_2_SEQ_LM": TaskType.SEQ_2_SEQ_LM}
            lora_config = LoraConfig(
                r=pt.get("lora_r", 16),
                lora_alpha=pt.get("lora_alpha", 32),
                lora_dropout=pt.get("lora_dropout", 0.05),
                target_modules=pt.get("lora_target_modules", ["q_proj", "v_proj"]),
                bias=pt.get("lora_bias", "none"),
                task_type=task_map.get(pt.get("lora_task_type", "CAUSAL_LM"), TaskType.CAUSAL_LM),
                modules_to_save=pt.get("lora_modules_to_save"),
            )
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()

        if pt.get("gradient_checkpointing"):
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

        self.model.to(device)

        train_ds = dataset.get("train")
        val_ds = dataset.get("val")
        collator = dataset.get("collator")

        train_loader = DataLoader(
            train_ds, batch_size=pt.get("batch_size", 4),
            shuffle=True, collate_fn=collator, pin_memory=True,
        )
        val_loader = None
        if val_ds is not None:
            val_loader = DataLoader(
                val_ds, batch_size=pt.get("batch_size", 4),
                shuffle=False, collate_fn=collator, pin_memory=True,
            )

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=pt.get("learning_rate", 2e-4),
            weight_decay=pt.get("weight_decay", 0.01),
        )

        grad_accum = pt.get("gradient_accumulation_steps", 4)
        epochs = pt.get("epochs", 3)
        max_steps = pt.get("max_steps", -1)
        total_steps = len(train_loader) * int(math.ceil(epochs)) // grad_accum if max_steps <= 0 else max_steps
        warmup_steps = max(1, int(pt.get("warmup_ratio", 0.03) * total_steps))

        scheduler = get_scheduler(
            pt.get("lr_scheduler_type", "cosine"),
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        use_amp = pt.get("use_amp", True) and device.type == "cuda"
        amp_dtype = _DTYPE_MAP.get(pt.get("amp_dtype", "bfloat16"), torch.bfloat16)
        scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and amp_dtype == torch.float16))

        privacy_engine = maybe_wrap_dp(self.model, optimizer, train_loader, pt)
        if privacy_engine:
            self.model, optimizer, train_loader = privacy_engine

        log_steps = pt.get("log_steps", 10)
        eval_steps = pt.get("eval_steps", 50)
        global_step = 0
        self.model.train()

        for epoch in range(int(math.ceil(epochs))):
            epoch_loss = 0.0
            for step, batch in enumerate(train_loader):
                batch = {k: v.to(device) for k, v in batch.items()}

                with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                    outputs = self.model(**batch)
                    loss = outputs.loss / grad_accum

                scaler.scale(loss).backward()
                epoch_loss += loss.item() * grad_accum

                if (step + 1) % grad_accum == 0:
                    if pt.get("max_grad_norm", 1.0) > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), pt["max_grad_norm"]
                        )
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

                    if global_step % log_steps == 0:
                        avg_loss = epoch_loss / (step + 1)
                        lr = scheduler.get_last_lr()[0]
                        print(f"  step {global_step} | loss={avg_loss:.4f} | lr={lr:.2e}")
                        if callbacks:
                            for cb in callbacks:
                                log_fn = getattr(cb, "on_log", None)
                                if log_fn:
                                    log_fn({"loss": avg_loss, "learning_rate": lr, "step": global_step})

                    if val_loader and global_step % eval_steps == 0:
                        val_loss = self._evaluate(val_loader, device, use_amp, amp_dtype)
                        print(f"  step {global_step} | eval_loss={val_loss:.4f}")
                        self.model.train()

                    if max_steps > 0 and global_step >= max_steps:
                        break

            print(f"Epoch {epoch + 1} complete | avg_loss={epoch_loss / max(len(train_loader), 1):.4f}")
            if callbacks:
                for cb in callbacks:
                    epoch_fn = getattr(cb, "on_epoch_end", None)
                    if epoch_fn:
                        epoch_fn({"epoch": epoch + 1, "loss": epoch_loss / max(len(train_loader), 1)})

            if max_steps > 0 and global_step >= max_steps:
                break

    def _evaluate(self, val_loader, device, use_amp, amp_dtype):
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                    outputs = self.model(**batch)
                total_loss += outputs.loss.item()
        return total_loss / max(len(val_loader), 1)

    def save_model(self, config: dict):
        output_path = config.get("output", {}).get("path", "/mnt/remote/output")
        merge = config.get("output", {}).get("merge_adapter", False)
        safe_save_model(self.model, self.tokenizer, output_path, merge_adapter=merge)
