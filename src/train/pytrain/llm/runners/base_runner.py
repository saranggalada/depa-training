"""
Abstract base class for all LLM fine-tuning framework runners.

Every runner must implement ``run()`` and ``save_model()``.
The contract is:
  1. ``run()``  receives the validated config, a tokenized dataset (or raw
     dataset for frameworks that handle tokenization internally), and a
     list of callback objects (MLflow tracker, audit logger).
  2. ``save_model()`` persists the trained model/adapter to the output path
     in safetensors format.
"""

from abc import ABC, abstractmethod


class FrameworkRunner(ABC):

    @abstractmethod
    def run(self, config: dict, dataset, callbacks: list | None = None):
        """Execute the fine-tuning run.

        Parameters
        ----------
        config : dict
            Framework-translated config (output of the appropriate translator).
        dataset : datasets.Dataset | dict
            The prepared dataset.  May be tokenized (HuggingFace, PyTorch) or
            raw (Axolotl, Torchtune handle tokenization internally).
        callbacks : list, optional
            Callback objects that expose ``on_log``, ``on_epoch_end``, etc.
        """

    @abstractmethod
    def save_model(self, config: dict):
        """Save the trained model / adapter weights.

        Parameters
        ----------
        config : dict
            The original validated DEPA config (not the translated one) so
            that ``output.path`` and ``output.merge_adapter`` are available.
        """
