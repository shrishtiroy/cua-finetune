"""Backend registry for the eval driver."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base_vllm_backend import BaseVLLMBackend  # noqa: F401


_BACKEND_REGISTRY = {
    "qwen_vl_cua": ("eval.backends.qwen_vl_cua", "QwenVLCuaBackend"),
    "llama_vision_cua": ("eval.backends.llama_vision_cua", "LlamaVisionCuaBackend"),
    "kimi_vl_cua": ("eval.backends.kimi_vl_cua", "KimiVLCuaBackend"),
    "deepseek_vl_cua": ("eval.backends.deepseek_vl_cua", "DeepSeekVLCuaBackend"),
}


def get_backend_class(name: str):
    if name not in _BACKEND_REGISTRY:
        raise ValueError(f"Unknown backend: {name}. Known: {sorted(_BACKEND_REGISTRY)}")
    module_name, class_name = _BACKEND_REGISTRY[name]
    import importlib
    mod = importlib.import_module(module_name)
    return getattr(mod, class_name)
