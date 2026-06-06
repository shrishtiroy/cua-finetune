"""Qwen3-VL-8B CUA backend.

Run vLLM as::

    vllm serve Qwen/Qwen3-VL-8B-Instruct \\
        --enable-lora --lora-modules cua=checkpoints/qwen3_vl_8b_cua/best \\
        --max-model-len 16384 --tensor-parallel-size 1 --port 8000

Then::

    CUA_LORA_ADAPTER=baseline python eval/run_eval.py --backend qwen_vl_cua --pass-k 5
    CUA_LORA_ADAPTER=cua      python eval/run_eval.py --backend qwen_vl_cua --pass-k 5
"""

from __future__ import annotations

from .base_vllm_backend import BaseVLLMBackend


class QwenVLCuaBackend(BaseVLLMBackend):
    name = "qwen_vl_cua"
    model_name = "Qwen/Qwen3-VL-8B-Instruct"
    # Qwen3-VL handles JSON cleanly out of the box; default system prompt is fine.
