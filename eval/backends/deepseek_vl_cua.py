"""DeepSeek-VL2-Small CUA backend.

Run vLLM as::

    vllm serve deepseek-ai/deepseek-vl2-small \\
        --enable-lora --lora-modules cua=checkpoints/deepseek_vl2_small_cua/best \\
        --max-model-len 16384 --tensor-parallel-size 1 --port 8000 \\
        --trust-remote-code

Notes:
  * DeepSeek-VL2 needs ``--trust-remote-code``. As of vLLM 0.7.x the model is
    supported but watch the release notes for vision-encoder updates.
  * Same JSON-discipline caveat as Kimi-VL.
"""

from __future__ import annotations

from .base_vllm_backend import BaseVLLMBackend


class DeepSeekVLCuaBackend(BaseVLLMBackend):
    name = "deepseek_vl_cua"
    model_name = "deepseek-ai/deepseek-vl2-small"
    system_prompt = (
        "You are a computer-use agent. You see a browser screenshot and a task. "
        "Your reply MUST be exactly one JSON object and nothing else. "
        "Required keys: \"reasoning\" (short string) and \"action\" with subkeys "
        "\"function_name\" and \"arguments\"."
    )
