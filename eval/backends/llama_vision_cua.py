"""Llama-3.2-11B-Vision CUA backend.

Run vLLM as::

    vllm serve meta-llama/Llama-3.2-11B-Vision-Instruct \\
        --enable-lora --lora-modules cua=checkpoints/llama3_2_vision_11b_cua/best \\
        --max-model-len 16384 --tensor-parallel-size 1 --port 8000

Notes:
  * Llama-3.2-Vision in vLLM only supports a single image per turn — already true
    of our payload (we only send the *current* screenshot).
  * If you see "max_num_images=1" errors, ensure no follow-up turn injects more.
"""

from __future__ import annotations

from .base_vllm_backend import BaseVLLMBackend


class LlamaVisionCuaBackend(BaseVLLMBackend):
    name = "llama_vision_cua"
    model_name = "meta-llama/Llama-3.2-11B-Vision-Instruct"
