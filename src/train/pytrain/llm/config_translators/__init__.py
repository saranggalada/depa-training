from .axolotl_translator import AxolotlTranslator
from .torchtune_translator import TorchtuneTranslator
from .huggingface_translator import HuggingFaceTranslator
from .pytorch_translator import PytorchTranslator

TRANSLATORS = {
    "axolotl": AxolotlTranslator,
    "torchtune": TorchtuneTranslator,
    "huggingface": HuggingFaceTranslator,
    "pytorch": PytorchTranslator,
}


def get_translator(framework: str):
    cls = TRANSLATORS.get(framework)
    if cls is None:
        raise ValueError(f"Unknown framework '{framework}'. Choose from: {list(TRANSLATORS)}")
    return cls()
