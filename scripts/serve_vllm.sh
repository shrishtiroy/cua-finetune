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
#   scripts/serve_vllm.sh google/gemma-3-12b-it '' 8000
#
#   # Baseline + LoRA loaded:
#   scripts/serve_vllm.sh Qwen/Qwen3-VL-8B-Instruct           checkpoints/qwen3_vl_8b_cua/best
#   scripts/serve_vllm.sh meta-llama/Llama-3.2-11B-Vision-Instruct checkpoints/llama3_2_vision_11b_cua/best
#   scripts/serve_vllm.sh google/gemma-3-12b-it               checkpoints/gemma3_12b_cua/best
set -euo pipefail

MODEL="${1:?MODEL required}"
ADAPTER_PATH="${2:-}"
PORT="${3:-8000}"

# Auto-activate the cua-finetune venv if not already inside one. Without this,
# `pip install --user vllm` ends up co-installed with system scipy/sklearn that
# were apt-compiled against numpy 1.x — and we hit numpy 2.x ABI breaks like
# "ImportError: cannot import name 'Inf' from 'numpy'" or
# "ValueError: numpy.dtype size changed". The venv is isolated from
# /usr/lib/python3/dist-packages so it dodges those collisions entirely.
VENV_DIR="${HOME}/cua-finetune/.venv"
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  if [[ -f "${VENV_DIR}/bin/activate" ]]; then
    echo "Activating ${VENV_DIR}"
    # shellcheck disable=SC1091
    source "${VENV_DIR}/bin/activate"
  else
    echo "WARN: ${VENV_DIR} not found. Running with system Python — expect" >&2
    echo "      numpy/scipy/sklearn ABI errors. Run lambda_setup.sh first." >&2
  fi
fi

# Preflight: confirm vllm is on PATH inside the active venv. ms-swift's [llm]
# extras *may or may not* pull in vllm depending on version, so we don't rely
# on it. Install on-demand instead of failing later inside the model load.
if ! command -v vllm >/dev/null 2>&1; then
  echo "vllm CLI not found in venv. Installing..."
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

# Per-model overrides:
#  * --trust-remote-code for models that ship custom modeling code on HF.
#  * --max-model-len capped to whatever the model's config.json actually
#    supports. DeepSeek-VL2-Small declares max_position_embeddings=4096,
#    so vLLM rejects 16384 with a hard validation error. Other models
#    we use default to a 16384 ceiling, which is well within their
#    supported context.
MAX_MODEL_LEN=16384
case "$MODEL" in
  *Kimi-VL*)
    EXTRA_ARGS+=(--trust-remote-code)
    ;;
  *deepseek-vl2*)
    EXTRA_ARGS+=(--trust-remote-code)
    MAX_MODEL_LEN=4096
    ;;
  google/gemma-3-*)
    # Gemma 3 is supported natively in transformers + vLLM (>= 0.8.x), so we
    # do NOT pass --trust-remote-code. The model's native context is 128K;
    # 16384 is plenty for CUA turns and keeps the KV cache off the GPU
    # memory cliff on a single H100. bfloat16 matches Gemma 3's training
    # dtype and avoids the fp16 accuracy hit on the SigLIP vision tower.
    EXTRA_ARGS+=(--dtype bfloat16)
    ;;
esac

if [[ -n "$ADAPTER_PATH" ]]; then
  # Auto-detect LoRA rank from adapter_config.json so we don't have to keep this
  # in sync with the training config. ms-swift's default in our configs is r=64,
  # but vLLM's default --max-lora-rank is 16, so we'd otherwise get
  # "LoRA rank 64 is greater than max_lora_rank 16" at adapter-load time.
  ADAPTER_RANK=64
  if [[ -f "$ADAPTER_PATH/adapter_config.json" ]]; then
    ADAPTER_RANK=$(python3 -c "import json; print(json.load(open('$ADAPTER_PATH/adapter_config.json')).get('r', 64))" 2>/dev/null || echo 64)
  fi
  EXTRA_ARGS+=(--enable-lora --lora-modules "cua=$ADAPTER_PATH" --max-loras 1 --max-lora-rank "$ADAPTER_RANK")
  echo "Serving $MODEL with LoRA adapter at $ADAPTER_PATH (rank=$ADAPTER_RANK, use model=\"cua\" or model=\"$MODEL\")."
else
  echo "Serving $MODEL baseline (no LoRA). Use model=\"$MODEL\" in /v1/chat/completions."
fi
echo "  max_model_len=${MAX_MODEL_LEN}"

vllm serve "$MODEL" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --tensor-parallel-size 1 \
  --port "$PORT" \
  --gpu-memory-utilization 0.92 \
  "${EXTRA_ARGS[@]}"
