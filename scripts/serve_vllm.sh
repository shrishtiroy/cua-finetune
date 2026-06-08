#!/usr/bin/env bash
# Lambda 1xH100: serve a base VLM, optionally with a trained LoRA adapter loaded.
#
# Usage:
#   scripts/serve_vllm.sh <MODEL> [ADAPTER_PATH] [PORT]
#
# When ADAPTER_PATH is empty/omitted, serves the base model only (use this for
# baseline browser-eval runs where we want untuned behaviour). When provided,
# loads the LoRA under name "cua" so /v1/chat/completions can hit either the
# base (model="<base name>") or the adapter (model="cua").
#
# Examples (from cua-finetune root):
#   # Baseline (no adapter):
#   scripts/serve_vllm.sh Qwen/Qwen3-VL-8B-Instruct
#   scripts/serve_vllm.sh moonshotai/Kimi-VL-A3B-Instruct '' 8000
#   scripts/serve_vllm.sh deepseek-ai/deepseek-vl2-small '' 8000
#
#   # Baseline + LoRA loaded:
#   scripts/serve_vllm.sh Qwen/Qwen3-VL-8B-Instruct           checkpoints/qwen3_vl_8b_cua/best
#   scripts/serve_vllm.sh meta-llama/Llama-3.2-11B-Vision-Instruct checkpoints/llama3_2_vision_11b_cua/best
set -euo pipefail

MODEL="${1:?MODEL required}"
ADAPTER_PATH="${2:-}"
PORT="${3:-8000}"

# Preflight: confirm vllm is on PATH inside the active venv. ms-swift's [llm]
# extras *may or may not* pull in vllm depending on version, so we don't rely
# on it. Install on-demand instead of failing later inside the model load.
# Note: vllm has its own torch pinning. Recent vllm (>= 0.7) wants torch 2.6,
# matching what lambda_setup.sh installs. If the install changes torch, the
# warning is loud — we don't try to be clever about reverting it.
if ! command -v vllm >/dev/null 2>&1; then
  echo "vllm CLI not found in PATH. Installing into the active venv..."
  pip install --no-cache-dir "vllm>=0.7.0" || {
    echo "ERROR: vllm install failed. Possible causes:" >&2
    echo "       1. torch version mismatch (vllm wants torch 2.6, lambda_setup pins 2.6 — should be fine)" >&2
    echo "       2. CUDA version mismatch (vllm wheels are cuda 12.4 — Lambda Stack is 12.4 — fine)" >&2
    echo "       3. Out of disk in /tmp during build (re-run after: df -h)" >&2
    echo "       Try: pip install vllm  (no version pin)" >&2
    exit 2
  }
fi

# Some Lambda images set HF_HOME under a small partition. Be explicit and put
# the 17-GB Qwen3-VL-8B (and friends) under the user home where there's space.
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
mkdir -p "${HF_HOME}"

EXTRA_ARGS=()
case "$MODEL" in
  *Kimi-VL*|*deepseek-vl2*)
    EXTRA_ARGS+=(--trust-remote-code)
    ;;
esac

if [[ -n "$ADAPTER_PATH" ]]; then
  EXTRA_ARGS+=(--enable-lora --lora-modules "cua=$ADAPTER_PATH" --max-loras 1)
  echo "Serving $MODEL with LoRA adapter at $ADAPTER_PATH (use model=\"cua\" or model=\"$MODEL\")."
else
  echo "Serving $MODEL baseline (no LoRA). Use model=\"$MODEL\" in /v1/chat/completions."
fi

vllm serve "$MODEL" \
  --max-model-len 16384 \
  --tensor-parallel-size 1 \
  --port "$PORT" \
  --gpu-memory-utilization 0.92 \
  "${EXTRA_ARGS[@]}"
