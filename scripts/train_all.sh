#!/usr/bin/env bash
# Lambda 1xH100: train all four models in sequence on the same instance.
#
# Each `swift sft` call writes its checkpoints to ./checkpoints/<model>/.
# Pick the best LoRA adapter via offline action accuracy, then symlink as
# checkpoints/<model>/best.
set -euo pipefail

cd "$(dirname "$0")/.."

models=(qwen3_vl_8b llama3_2_vision_11b kimi_vl_a3b deepseek_vl2_small)

for m in "${models[@]}"; do
  echo "=== Training $m ==="
  swift sft --config "configs/${m}.yaml"
done

echo
echo "All four trainings finished. Pick best checkpoints with eval/action_accuracy.py."
echo "Example:"
echo "  CUA_LORA_ADAPTER=cua python eval/action_accuracy.py --backend qwen_vl_cua --max-samples 200"
