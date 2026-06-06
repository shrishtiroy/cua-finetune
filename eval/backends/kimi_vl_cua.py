"""Kimi-VL-A3B CUA backend.

Run vLLM as::

    vllm serve moonshotai/Kimi-VL-A3B-Instruct \\
        --enable-lora --lora-modules cua=checkpoints/kimi_vl_a3b_cua/best \\
        --max-model-len 16384 --tensor-parallel-size 1 --port 8000 \\
        --trust-remote-code

Notes:
  * Kimi-VL needs ``--trust-remote-code``.
  * Pre-SFT, the base Kimi often emits prose around the JSON; the parent class's
    ``_extract_first_json`` handles that. Post-SFT it should return clean JSON.
"""

from __future__ import annotations

from .base_vllm_backend import BaseVLLMBackend


class KimiVLCuaBackend(BaseVLLMBackend):
    name = "kimi_vl_cua"
    model_name = "moonshotai/Kimi-VL-A3B-Instruct"
    # Slightly stricter prompt wording — Kimi tends to add trailing commentary.
    system_prompt = (
        "You are a computer-use agent. You see a browser screenshot and a task. "
        "Your reply MUST be exactly one JSON object and nothing else (no prose, no "
        "markdown fences). Required keys: \"reasoning\" (short string) and "
        "\"action\" with subkeys \"function_name\" and \"arguments\"."
    )
