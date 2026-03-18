"""
Torchtune runner -- wraps torchtune's recipe system.

Writes the translated config to a temporary YAML and invokes
the appropriate torchtune recipe programmatically.
"""

import os
import subprocess
import sys
import tempfile

import yaml

from .base_runner import FrameworkRunner
from ..model_io import safe_save_model


class TorchtuneRunner(FrameworkRunner):

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.epsilon_spent = None
        self._output_dir = None

    def run(self, config: dict, dataset, callbacks: list | None = None):
        recipe = config.pop("_recipe", "lora_finetune_single_device")
        self._output_dir = config.get("output_dir", "/mnt/remote/output")

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", prefix="torchtune_", delete=False, dir="/tmp"
        ) as f:
            yaml.dump(config, f, default_flow_style=False)
            cfg_path = f.name

        print(f"Torchtune config written to {cfg_path}")
        print(f"Running torchtune recipe: {recipe}")

        try:
            cmd = [
                sys.executable, "-m", "torchtune._cli.tune",
                "run", recipe, "--config", cfg_path,
            ]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=86400,
            )

            if result.stdout:
                print(result.stdout)
            if result.returncode != 0:
                print(f"Torchtune stderr:\n{result.stderr}")
                raise RuntimeError(
                    f"Torchtune recipe '{recipe}' failed with exit code {result.returncode}"
                )
        except FileNotFoundError:
            print(
                "WARNING: torchtune is not installed. Falling back to HuggingFace runner. "
                "Install torchtune to use the torchtune framework."
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
        if self.model is not None and self.tokenizer is not None:
            merge = config.get("output", {}).get("merge_adapter", False)
            safe_save_model(self.model, self.tokenizer, output_path, merge_adapter=merge)
        else:
            print(f"Torchtune saves output to {output_path} internally.")
