# ... existing code ...
from .skill import (
    TranslatorSkill,
    TranslatorInput,
    TranslatorOutput,
    DictionaryBackend,
    GoogleTranslateBackend,
    _HAS_DEEP_TRANSLATOR,
    _HAS_LANGDETECT,
)

__all__ = [
    "TranslatorSkill",
    "TranslatorInput",
    "TranslatorOutput",
    "DictionaryBackend",
    "GoogleTranslateBackend",
    "_HAS_DEEP_TRANSLATOR",
    "_HAS_LANGDETECT",
]
