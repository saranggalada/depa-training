"""
Axolotl runner -- wraps axolotl's Python API.

Writes the translated YAML config to a temporary file and invokes
axolotl's training entry point programmatically.
"""

import os
import tempfile

import yaml

from .base_runner import FrameworkRunner
from ..model_io import safe_save_model


class AxolotlRunner(FrameworkRunner):

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.epsilon_spent = None
        self._output_dir = None

    def run(self, config: dict, dataset, callbacks: list | None = None):
        self._output_dir = config.get("output_dir", "/mnt/remote/output")

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yml", prefix="axolotl_", delete=False, dir="/tmp"
        ) as f:
            yaml.dump(config, f, default_flow_style=False)
            cfg_path = f.name

        print(f"Axolotl config written to {cfg_path}")

        try:
            from axolotl.cli.train import do_train
            from axolotl.utils.config import normalize_config, validate_config
            from axolotl.utils.dict import DictDefault

            ax_cfg = DictDefault(config)
            normalize_config(ax_cfg)
            validate_config(ax_cfg)

            self.model, self.tokenizer = do_train(ax_cfg)

        except ImportError:
            print(
                "WARNING: axolotl is not installed. Falling back to HuggingFace runner. "
                "Install axolotl to use the axolotl framework."
            )
            from .huggingface_runner import HuggingFaceRunner
            from ..config_translators.huggingface_translator import HuggingFaceTranslator

            raw_cfg = dataset.get("_original_config", config)
            hf_translator = HuggingFaceTranslator()
            hf_config = hf_translator.translate(raw_cfg) if isinstance(raw_cfg, dict) and "base_model" in raw_cfg else config
            fallback = HuggingFaceRunner()
            fallback.run(hf_config, dataset, callbacks=callbacks)
            self.model = fallback.model
            self.tokenizer = fallback.tokenizer
        finally:
            if os.path.exists(cfg_path):
                os.unlink(cfg_path)

    def save_model(self, config: dict):
        output_path = config.get("output", {}).get("path", self._output_dir or "/mnt/remote/output")
        merge = config.get("output", {}).get("merge_adapter", False)
        if self.model is not None and self.tokenizer is not None:
            safe_save_model(self.model, self.tokenizer, output_path, merge_adapter=merge)
        else:
            print("Model save skipped — axolotl handles saving internally.")
