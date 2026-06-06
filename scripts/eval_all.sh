#!/usr/bin/env bash
# Lambda 1xH100: full eval sweep — for each model, serve with --enable-lora,
# run baseline pass@5 and finetuned pass@5 against the held-out tasks, kill
# vLLM, move on.
set -euo pipefail

cd "$(dirname "$0")/.."

declare -A BASE_MODEL=(
  [qwen_vl_cua]="Qwen/Qwen3-VL-8B-Instruct"
  [llama_vision_cua]="meta-llama/Llama-3.2-11B-Vision-Instruct"
  [kimi_vl_cua]="moonshotai/Kimi-VL-A3B-Instruct"
  [deepseek_vl_cua]="deepseek-ai/deepseek-vl2-small"
)

declare -A ADAPTER=(
  [qwen_vl_cua]="checkpoints/qwen3_vl_8b_cua/best"
  [llama_vision_cua]="checkpoints/llama3_2_vision_11b_cua/best"
  [kimi_vl_cua]="checkpoints/kimi_vl_a3b_cua/best"
  [deepseek_vl_cua]="checkpoints/deepseek_vl2_small_cua/best"
)

PASS_K="${PASS_K:-5}"
PORT="${PORT:-8000}"

wait_for_health() {
  for _ in {1..120}; do
    if curl -sf "http://localhost:${PORT}/v1/models" >/dev/null 2>&1; then
      return 0
    fi
    sleep 5
  done
  echo "vLLM did not come up in time"; return 1
}

for KEY in qwen_vl_cua llama_vision_cua kimi_vl_cua deepseek_vl_cua; do
  echo "=== ${KEY}: serving + eval ==="
  scripts/serve_vllm.sh "${BASE_MODEL[$KEY]}" "${ADAPTER[$KEY]}" "$PORT" \
    > "logs/vllm_${KEY}.log" 2>&1 &
  VLLM_PID=$!
  trap "kill $VLLM_PID 2>/dev/null || true" EXIT
  wait_for_health

  CUA_LORA_ADAPTER=baseline python eval/run_eval.py --backend "$KEY" --pass-k "$PASS_K"
  CUA_LORA_ADAPTER=cua      python eval/run_eval.py --backend "$KEY" --pass-k "$PASS_K"

  kill "$VLLM_PID" 2>/dev/null || true
  wait "$VLLM_PID" 2>/dev/null || true
  trap - EXIT
done

echo "Aggregating headline table..."
python eval/aggregate_results.py
