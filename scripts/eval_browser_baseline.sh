#!/usr/bin/env bash
# Phase 3e/3f/3g: browser-eval one of our candidate VLMs on the 24 held-out
# tasks. Assumes vLLM is already serving the model on port 8000 (run
# `scripts/serve_vllm.sh <MODEL>` in a separate tmux first).
#
# This script:
#   1. Starts conduit-runtime Docker container on port 7777.
#   2. Verifies vLLM /v1/models is reachable on 8000.
#   3. Runs eval/run_eval.py against held_out_tasks.yaml with the chosen backend.
#   4. Tears down runtime container.
#
# Usage:
#   scripts/eval_browser_baseline.sh <BACKEND> [ADAPTER_NAME] [PASS_K]
#
# Args:
#   BACKEND       - qwen_vl_cua | kimi_vl_cua | deepseek_vl_cua | llama_vision_cua
#   ADAPTER_NAME  - "baseline" (default; uses bare base model) or "cua" (uses LoRA)
#   PASS_K        - number of trials per task (default 1)
#
# Examples:
#   scripts/eval_browser_baseline.sh qwen_vl_cua                     # 24 tasks x 1 trial
#   scripts/eval_browser_baseline.sh qwen_vl_cua cua 5               # 24 tasks x 5 trials, with LoRA
#   scripts/eval_browser_baseline.sh kimi_vl_cua baseline 1
#
# Output:
#   ~/cua-finetune/results/<BACKEND>/<ADAPTER_NAME>/<task>/<ts>/...
#   ~/cua-finetune/results/<BACKEND>/<ADAPTER_NAME>/_summary.jsonl

set -euo pipefail

BACKEND="${1:?BACKEND required (qwen_vl_cua | kimi_vl_cua | deepseek_vl_cua | llama_vision_cua)}"
ADAPTER="${2:-baseline}"
PASS_K="${3:-1}"

# Set CUA_LIVE_WEB=1 (default) to bypass pywb archive replay and have the agent
# hit live URLs. Set to 0 to use archive replay (requires .wacz files in
# ~/Dillinger/archives/). Live-web is the simpler path: no archive transfer
# needed, but verifiers written against frozen archive state may drift.
LIVE_WEB="${CUA_LIVE_WEB:-1}"
LIVE_WEB_FLAG=""
if [[ "${LIVE_WEB}" == "1" ]]; then
  LIVE_WEB_FLAG="--live-web"
  echo "Live-web mode ON (--live-web). Set CUA_LIVE_WEB=0 to use archive replay."
fi

CUA_REPO="${HOME}/cua-finetune"
RUNTIME_PORT=7777
VLLM_PORT=8000
CONTAINER_NAME="conduit-runtime-eval"

DOCKER_CMD="docker"
if ! docker info >/dev/null 2>&1; then
  DOCKER_CMD="sudo docker"
fi

# ---- 0. preflight -----------------------------------------------------------
echo "==> [0/4] Preflight"
if ! ${DOCKER_CMD} image inspect conduit-runtime >/dev/null 2>&1; then
  echo "ERROR: conduit-runtime Docker image not built. Run lambda_browser_setup.sh." >&2
  exit 2
fi
if ! curl -fsS "http://127.0.0.1:${VLLM_PORT}/v1/models" >/dev/null 2>&1; then
  echo "ERROR: vLLM not reachable at http://127.0.0.1:${VLLM_PORT}/v1/models" >&2
  echo "Start vLLM first in a separate tmux:" >&2
  case "$BACKEND" in
    qwen_vl_cua)     echo "  scripts/serve_vllm.sh Qwen/Qwen3-VL-8B-Instruct" >&2 ;;
    kimi_vl_cua)     echo "  scripts/serve_vllm.sh moonshotai/Kimi-VL-A3B-Instruct" >&2 ;;
    deepseek_vl_cua) echo "  scripts/serve_vllm.sh deepseek-ai/deepseek-vl2-small" >&2 ;;
    llama_vision_cua) echo "  scripts/serve_vllm.sh meta-llama/Llama-3.2-11B-Vision-Instruct" >&2 ;;
  esac
  exit 2
fi
echo "  vLLM reachable. Models served:"
curl -fsS "http://127.0.0.1:${VLLM_PORT}/v1/models" | python3 -c "import json,sys; print('   ', [m['id'] for m in json.load(sys.stdin)['data']])"

# ---- 1. start conduit-runtime ----------------------------------------------
echo "==> [1/4] Booting conduit-runtime on :${RUNTIME_PORT}"
${DOCKER_CMD} rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
${DOCKER_CMD} run -d --rm \
  --name "${CONTAINER_NAME}" \
  -p ${RUNTIME_PORT}:8000 \
  -p 5900:5900 \
  -p 8080:8080 \
  conduit-runtime >/dev/null

cleanup() {
  echo
  echo "==> Tearing down ${CONTAINER_NAME}"
  ${DOCKER_CMD} rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

echo -n "  waiting for /health"
for i in $(seq 1 120); do
  if curl -fsS "http://127.0.0.1:${RUNTIME_PORT}/health" >/dev/null 2>&1; then
    echo " -> healthy"
    break
  fi
  echo -n "."
  sleep 2
  if [[ $i -eq 120 ]]; then
    echo
    echo "ERROR: runtime did not become healthy in 4 minutes. Logs:"
    ${DOCKER_CMD} logs "${CONTAINER_NAME}"
    exit 3
  fi
done

# ---- 2. dry-run plan --------------------------------------------------------
cd "${CUA_REPO}"
source .venv/bin/activate
echo "==> [2/4] Dry-run plan (resolves task YAMLs without running anything)"
CUA_LORA_ADAPTER="${ADAPTER}" \
CONDUIT_RUNTIME_URL="http://127.0.0.1:${RUNTIME_PORT}" \
  python eval/run_eval.py \
    --backend "${BACKEND}" \
    --pass-k "${PASS_K}" \
    --runtime-container "${CONTAINER_NAME}" \
    --runtime-url "http://127.0.0.1:${RUNTIME_PORT}" \
    ${LIVE_WEB_FLAG} \
    --dry-run

# ---- 3. real run ------------------------------------------------------------
echo "==> [3/4] Real run: ${BACKEND} adapter=${ADAPTER} pass_k=${PASS_K}"
START=$(date +%s)
CUA_LORA_ADAPTER="${ADAPTER}" \
CONDUIT_RUNTIME_URL="http://127.0.0.1:${RUNTIME_PORT}" \
VLLM_BASE_URL="http://127.0.0.1:${VLLM_PORT}/v1" \
  python eval/run_eval.py \
    --backend "${BACKEND}" \
    --pass-k "${PASS_K}" \
    --runtime-container "${CONTAINER_NAME}" \
    --runtime-url "http://127.0.0.1:${RUNTIME_PORT}" \
    ${LIVE_WEB_FLAG} \
    -v 2>&1 | tee "logs/eval_${BACKEND}_${ADAPTER}_$(date +%Y%m%d_%H%M%S).log"
END=$(date +%s)
echo "  Wall-clock: $((END - START))s"

# ---- 4. quick aggregate -----------------------------------------------------
echo "==> [4/4] Quick aggregate"
SUMMARY="${CUA_REPO}/results/${BACKEND}/${ADAPTER}/_summary.jsonl"
if [[ -f "${SUMMARY}" ]]; then
  python3 - <<PY
import json
from collections import defaultdict
rows = [json.loads(l) for l in open("${SUMMARY}")]
n = len(rows)
n_passed = sum(1 for r in rows if r.get("score", 0) >= 1.0)
n_partial = sum(1 for r in rows if 0 < r.get("score", 0) < 1.0)
n_zero = sum(1 for r in rows if r.get("score", 0) == 0)
print(f"  ${BACKEND}/${ADAPTER}: {n} attempts, pass={n_passed} partial={n_partial} zero={n_zero}")

per_task = defaultdict(list)
for r in rows: per_task[r["task"]].append(r.get("score", 0))
mean_per_task = {t: max(s) for t, s in per_task.items()}   # pass@k = max over trials
n_tasks = len(mean_per_task)
n_tasks_pass = sum(1 for s in mean_per_task.values() if s >= 1.0)
print(f"  Tasks: {n_tasks_pass}/{n_tasks} passed pass@${PASS_K} ({100*n_tasks_pass/max(n_tasks,1):.1f}%)")
PY
else
  echo "  (no summary file at ${SUMMARY})"
fi

echo
echo "============================================================"
echo "  Done. Results in ${CUA_REPO}/results/${BACKEND}/${ADAPTER}/"
echo "============================================================"
