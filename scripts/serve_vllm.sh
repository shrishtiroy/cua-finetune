#!/usr/bin/env bash
# Lambda 1xH100: serve a base CUA model with the trained LoRA adapter loaded.
#
# Usage:
#   scripts/serve_vllm.sh <MODEL> <ADAPTER_PATH> [PORT]
#
# Examples (from cua-finetune root):
#   scripts/serve_vllm.sh Qwen/Qwen3-VL-8B-Instruct           checkpoints/qwen3_vl_8b_cua/best
#   scripts/serve_vllm.sh meta-llama/Llama-3.2-11B-Vision-Instruct checkpoints/llama3_2_vision_11b_cua/best
#   scripts/serve_vllm.sh moonshotai/Kimi-VL-A3B-Instruct     checkpoints/kimi_vl_a3b_cua/best
#   scripts/serve_vllm.sh deepseek-ai/deepseek-vl2-small      checkpoints/deepseek_vl2_small_cua/best
#
# After it's up, hit `/v1/chat/completions` with model="<base name>" for baseline
# or model="cua" for the LoRA-merged forward pass.
set -euo pipefail

MODEL="${1:?MODEL required}"
ADAPTER_PATH="${2:?ADAPTER_PATH required}"
PORT="${3:-8000}"

EXTRA=""
case "$MODEL" in
  *Kimi-VL*|*deepseek-vl2*)
    EXTRA="--trust-remote-code"
    ;;
esac

vllm serve "$MODEL" \
  --enable-lora \
  --lora-modules "cua=$ADAPTER_PATH" \
  --max-model-len 16384 \
  --tensor-parallel-size 1 \
  --port "$PORT" \
  --max-loras 1 \
  --gpu-memory-utilization 0.92 \
  $EXTRA
