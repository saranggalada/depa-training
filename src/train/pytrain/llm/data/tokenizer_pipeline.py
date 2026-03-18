"""
In-enclave tokenization pipeline for LLM fine-tuning.

Loads a tokenizer from a local path (never from the hub at runtime),
formats examples according to the declared dataset format (alpaca,
sharegpt, completion, instruction), tokenizes with truncation/padding,
and returns a ready-to-train ``datasets.Dataset`` with
``input_ids``, ``attention_mask``, and ``labels``.
"""

from functools import partial

from datasets import Dataset
from transformers import AutoTokenizer, DataCollatorForSeq2Seq

# ── Chat template presets ────────────────────────────────────────────
# These are safe, static templates.  Arbitrary Jinja is NOT permitted.

ALPACA_TEMPLATE = (
    "Below is an instruction that describes a task, paired with an input that provides "
    "further context. Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{input}\n\n"
    "### Response:\n{output}"
)

ALPACA_NO_INPUT_TEMPLATE = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Response:\n{output}"
)

IGNORE_INDEX = -100


class TokenizerPipeline:
    """Stateful wrapper around tokenizer + formatting logic."""

    def __init__(self, cfg: dict):
        tok_cfg = cfg.get("tokenizer", {})
        self.tokenizer = AutoTokenizer.from_pretrained(
            tok_cfg["name_or_path"],
            trust_remote_code=False,
            use_fast=True,
        )
        self.max_seq_length = tok_cfg.get("max_seq_length", 2048)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.tokenizer.padding_side = tok_cfg.get("padding_side", "right")
        self.dataset_format = cfg["dataset"].get("format", "alpaca")
        self.ds_cfg = cfg["dataset"]

    # ── Format functions ─────────────────────────────────────────────

    def _format_alpaca(self, example: dict) -> str:
        f_instr = self.ds_cfg.get("field_instruction", "instruction")
        f_input = self.ds_cfg.get("field_input", "input")
        f_output = self.ds_cfg.get("field_output", "output")

        instruction = example.get(f_instr, "")
        inp = example.get(f_input, "")
        output = example.get(f_output, "")

        if inp and inp.strip():
            return ALPACA_TEMPLATE.format(instruction=instruction, input=inp, output=output)
        return ALPACA_NO_INPUT_TEMPLATE.format(instruction=instruction, output=output)

    def _format_sharegpt(self, example: dict) -> str:
        f_conv = self.ds_cfg.get("field_conversations", "conversations")
        conversations = example.get(f_conv, [])
        parts = []
        for turn in conversations:
            role = turn.get("from", turn.get("role", "user"))
            content = turn.get("value", turn.get("content", ""))
            if role in ("system", "human", "user"):
                parts.append(f"<|user|>\n{content}")
            elif role in ("gpt", "assistant"):
                parts.append(f"<|assistant|>\n{content}")
        return "\n".join(parts)

    def _format_completion(self, example: dict) -> str:
        f_text = self.ds_cfg.get("field_text", "text")
        return example.get(f_text, "")

    def _format_instruction(self, example: dict) -> str:
        return self._format_alpaca(example)

    def _get_formatter(self):
        formatters = {
            "alpaca": self._format_alpaca,
            "sharegpt": self._format_sharegpt,
            "completion": self._format_completion,
            "instruction": self._format_instruction,
        }
        return formatters[self.dataset_format]

    # ── Tokenization ─────────────────────────────────────────────────

    def _tokenize_batch(self, examples: dict, formatter) -> dict:
        """Tokenize a batch of examples (called via ``dataset.map``)."""
        texts = []
        n = len(next(iter(examples.values())))
        for i in range(n):
            row = {k: v[i] for k, v in examples.items()}
            texts.append(formatter(row))

        tokenized = self.tokenizer(
            texts,
            truncation=True,
            max_length=self.max_seq_length,
            padding=False,
        )

        tokenized["labels"] = [ids[:] for ids in tokenized["input_ids"]]

        return tokenized

    def tokenize_dataset(self, dataset: Dataset) -> Dataset:
        """Apply formatting + tokenization to the full dataset.

        Uses batched ``dataset.map`` for Arrow-backed lazy efficiency.
        """
        formatter = self._get_formatter()
        tokenized = dataset.map(
            partial(self._tokenize_batch, formatter=formatter),
            batched=True,
            remove_columns=dataset.column_names,
            desc="Tokenizing",
        )
        return tokenized

    def get_data_collator(self):
        """Return a DataCollator appropriate for the task."""
        return DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            padding=True,
            max_length=self.max_seq_length,
            label_pad_token_id=IGNORE_INDEX,
        )

    def split_train_val(self, dataset: Dataset, val_ratio: float = 0.05):
        """Split a tokenized dataset into train and validation."""
        if val_ratio <= 0:
            return dataset, None
        split = dataset.train_test_split(test_size=val_ratio, seed=42)
        return split["train"], split["test"]
