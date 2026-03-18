from .base_runner import FrameworkRunner
from .huggingface_runner import HuggingFaceRunner
from .axolotl_runner import AxolotlRunner
from .torchtune_runner import TorchtuneRunner
from .pytorch_runner import PytorchRunner

RUNNERS = {
    "huggingface": HuggingFaceRunner,
    "axolotl": AxolotlRunner,
    "torchtune": TorchtuneRunner,
    "pytorch": PytorchRunner,
}


def create_runner(framework: str) -> FrameworkRunner:
    cls = RUNNERS.get(framework)
    if cls is None:
        raise ValueError(f"Unknown framework '{framework}'. Choose from: {list(RUNNERS)}")
    return cls()
