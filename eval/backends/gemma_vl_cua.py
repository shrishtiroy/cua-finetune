"""Gemma-3 VL CUA backend.

Targets ``google/gemma-3-12b-it`` (primary) and the rest of the Gemma-3
instruct family (``google/gemma-3-4b-it``, ``google/gemma-3-27b-it``). All
three are dense multimodal models with a SigLIP vision encoder, 128K
context, and native vLLM support (>= 0.8.x). They use the standard chat
template, so vLLM's ``/v1/chat/completions`` accepts the same OpenAI-style
``image_url`` payload we already build in :class:`BaseVLLMBackend`.

Run vLLM as::

    vllm serve google/gemma-3-12b-it \\
        --max-model-len 16384 --tensor-parallel-size 1 --port 8000 \\
        --dtype bfloat16

LoRA-loaded variant::

    vllm serve google/gemma-3-12b-it \\
        --enable-lora --lora-modules cua=checkpoints/gemma3_12b_cua/best \\
        --max-model-len 16384 --tensor-parallel-size 1 --port 8000 \\
        --dtype bfloat16

Then::

    CUA_LORA_ADAPTER=baseline python eval/run_eval.py --backend gemma_vl_cua --pass-k 1
    CUA_LORA_ADAPTER=cua      python eval/run_eval.py --backend gemma_vl_cua --pass-k 5

Notes:
  * Gemma 3 landed in ``transformers`` natively, so ``--trust-remote-code``
    is NOT required (do not set it; vLLM will use its built-in modeling).
  * Gemma 3's full 128K context is overkill for CUA turns. ``--max-model-len
    16384`` keeps KV-cache memory in check on a single H100 while still
    leaving headroom for screenshot tokens + reasoning.
  * Gemma 3 tends to wrap JSON in ```json fences and emit short prose
    preambles. The parent class's ``_extract_first_json`` already handles
    both (strips fenced blocks and walks balanced braces), so we just
    inherit that.
  * We override ``max_completion_tokens`` down to 512: enough room for
    reasoning + a single JSON action, but tight enough that the model
    can't burn the whole turn on a monologue.

# SMOKE TEST
# ------------------------------------------------------------------
# Verify the backend end-to-end without committing to a full baseline:
#
#   1. Start vLLM (in a separate tmux on the GPU box):
#        bash scripts/serve_vllm.sh google/gemma-3-12b-it
#
#   2. Wait for the log line:
#        INFO ... Uvicorn running on http://0.0.0.0:8000
#      and confirm /v1/models is reachable:
#        curl -s http://127.0.0.1:8000/v1/models | python3 -m json.tool
#
#   3. Run a single-task live-web eval against one of the held-out YAMLs:
#        cd ~/Dillinger
#        PYTHONPATH=~/cua-finetune \
#        CUA_LORA_ADAPTER=baseline \
#        VLLM_BASE_URL=http://127.0.0.1:8000/v1 \
#          uv run python ~/cua-finetune/eval/run_eval.py \
#            --backend gemma_vl_cua \
#            --task pydocs-sorted-vs-list-sort \
#            --pass-k 1 \
#            --live-web
#
#   4. Expected: the agent takes 1+ steps, JSON parses on every turn (no
#      "failed to parse JSON, falling back to wait" warnings dominating
#      the log), and there is no httpx.ConnectError. A non-zero score is
#      a bonus; what we're checking here is plumbing, not capability.
# ------------------------------------------------------------------
"""

from __future__ import annotations

from .base_vllm_backend import BaseVLLMBackend


class GemmaVLCuaBackend(BaseVLLMBackend):
    name = "gemma_vl_cua"
    model_name = "google/gemma-3-12b-it"
    # Gemma 3 is verbose by default — be explicit about "JSON only" the same
    # way the Kimi/DeepSeek backends are. The parent's brace-walking parser
    # still cleans up any stray fences, but tighter wording cuts the failure
    # rate on the first turn.
    system_prompt = (
        "You are a computer-use agent. You see a browser screenshot and a task. "
        "Your reply MUST be exactly one JSON object and nothing else (no prose, "
        "no markdown fences, no commentary before or after). Required keys: "
        "\"reasoning\" (a short string) and \"action\" with subkeys "
        "\"function_name\" and \"arguments\". "
        "Available action function_names: click, double_click, triple_click, "
        "right_click, scroll, type, keypress, drag, mouse_move, mouse_down, "
        "mouse_up, hold_key, wait, zoom, terminate, answer."
    )
    # Cap action turns at 512 tokens: comfortably fits reasoning + a JSON
    # action, but stops Gemma from filibustering on the rare confused turn.
    max_completion_tokens: int = 512
