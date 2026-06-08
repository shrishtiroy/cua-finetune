#!/usr/bin/env bash
# Phase 3b smoke test: Anthropic native computer-use via AWS Bedrock,
# bypassing run_conduit.sh entirely.
#
# Why bypass run_conduit.sh: it has `require_cmd npm` BEFORE the
# CONDUIT_SKIP_SETUP gate, and it always boots the viewer process.
# `conduit run` itself only needs npm/viewer when --watch is passed,
# so we call it directly and skip the whole UI machinery.
#
# What this validates (everything BUT our model):
#   - conduit-runtime container boots and serves /health
#   - pywb archive replay loads docs-python-org.wacz
#   - the agent loop drives the browser turn-by-turn until terminate
#   - the rubric grader returns a score
#
# Cost: ~$0.50 per run (1 task, ~10 turns of Bedrock Opus + 1 grading call).
#
# Pre-reqs:
#   - lambda_browser_setup.sh has succeeded (built conduit-runtime image,
#     ran `uv sync` in ~/Dillinger so the conduit CLI is available)
#   - ~/Dillinger/.env contains AWS_* creds, ANTHROPIC_API_KEY, LITELLM_API_KEY
#   - AWS account has Bedrock model access enabled for Claude Opus in your
#     region (one-time grant in AWS console -> Bedrock -> Model access).
#
# Usage:
#   bash scripts/eval_browser_smoke_bedrock.sh
#   bash scripts/eval_browser_smoke_bedrock.sh pydocs-os-urandom-pep
#   CONDUIT_BEDROCK_MODEL=us.anthropic.claude-opus-4-5-20251101-v1:0 \
#     bash scripts/eval_browser_smoke_bedrock.sh

set -euo pipefail

DILLINGER_DIR="${HOME}/Dillinger"
TASK_NAME="${1:-pydocs-os-sched-policy}"
N_RUNS="${2:-1}"
RUNTIME_PORT="${RUNTIME_PORT:-7777}"
CONTAINER_NAME="conduit-runtime-smoke-bedrock"

# All bedrock config (CONDUIT_BEDROCK_MODEL, CONDUIT_AGENT_BACKEND,
# AWS_*, etc.) is read from ~/Dillinger/.env — same setup the user has
# locally. We don't override anything here.

DOCKER_CMD="docker"
if ! docker info >/dev/null 2>&1; then
  DOCKER_CMD="sudo docker"
fi

# ---- 0. preflight -----------------------------------------------------------
echo "==> [0/4] Preflight"
if [[ ! -d "${DILLINGER_DIR}" ]]; then
  echo "ERROR: ${DILLINGER_DIR} not found. Run lambda_browser_setup.sh first." >&2
  exit 2
fi
if [[ ! -f "${DILLINGER_DIR}/.env" ]]; then
  echo "ERROR: ${DILLINGER_DIR}/.env missing" >&2
  exit 2
fi
if ! ${DOCKER_CMD} image inspect conduit-runtime >/dev/null 2>&1; then
  echo "ERROR: conduit-runtime Docker image not built. Run lambda_browser_setup.sh." >&2
  exit 2
fi

# Trust whatever's in the user's .env — no opinionated key checks.
# If credentials are wrong/missing, the run will fail fast with a clear
# AWS / Anthropic error message anyway.

cd "${DILLINGER_DIR}"

TASK_YAML="$(grep -rl "name: ${TASK_NAME}\b" tasks/ 2>/dev/null | head -1 || true)"
if [[ -z "${TASK_YAML}" ]]; then
  echo "ERROR: no YAML in ${DILLINGER_DIR}/tasks/ defines task '${TASK_NAME}'." >&2
  echo "Available pydocs tasks:" >&2
  grep -rh "^- name: pydocs-" tasks/ 2>/dev/null | head -8 >&2
  exit 2
fi
TASK_YAML_REL="${TASK_YAML#${DILLINGER_DIR}/}"

echo "    task          : ${TASK_NAME}"
echo "    yaml          : ${TASK_YAML_REL}"
echo "    n_runs        : ${N_RUNS}"
echo "    backend/model : (from ~/Dillinger/.env)"
echo

# ---- 1. start runtime container --------------------------------------------
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

echo -n "    waiting for /health"
for i in $(seq 1 60); do
  if curl -fsS "http://127.0.0.1:${RUNTIME_PORT}/health" >/dev/null 2>&1; then
    echo " -> healthy"
    break
  fi
  echo -n "."
  sleep 2
  if [[ $i -eq 60 ]]; then
    echo
    echo "ERROR: runtime did not become healthy in 2 minutes. Logs:"
    ${DOCKER_CMD} logs "${CONTAINER_NAME}" | tail -40
    exit 3
  fi
done

# ---- 2. export .env so conduit picks it up ----------------------------------
echo "==> [2/4] Loading ~/Dillinger/.env"
set -a
source "${DILLINGER_DIR}/.env"
set +a

# ---- 3. run the task --------------------------------------------------------
echo "==> [3/4] Running task: ${TASK_NAME}"
START=$(date +%s)
uv run conduit run \
  --tasks-file "${TASK_YAML_REL}" \
  --task "${TASK_NAME}" \
  --runs "${N_RUNS}" \
  --runtime-url "http://127.0.0.1:${RUNTIME_PORT}" \
  --runtime-container "${CONTAINER_NAME}"
END=$(date +%s)
echo "    Wall-clock: $((END - START))s"

# ---- 4. inspect result ------------------------------------------------------
echo "==> [4/4] Result"
LATEST_RUN_DIR="$(ls -td "${DILLINGER_DIR}/runs/${TASK_NAME}"/* 2>/dev/null | head -1 || true)"
if [[ -n "${LATEST_RUN_DIR}" ]]; then
  echo "    ${LATEST_RUN_DIR}"
  if [[ -f "${LATEST_RUN_DIR}/result.json" ]]; then
    python3 -c "
import json
r = json.load(open('${LATEST_RUN_DIR}/result.json'))
print(f\"    score = {r.get('score', 'N/A')}\")
for s in (r.get('subscores') or []):
    print(f\"      {s.get('weight', 0):3}pt  {s.get('met', '?')!s:5}  {(s.get('requirement') or '')[:60]}\")
"
  else
    echo "    (no result.json — task may have crashed before grading)"
  fi
else
  echo "    (no run output found)"
fi

echo
echo "============================================================"
echo "  Smoke test complete."
echo "============================================================"
echo "  Pass criterion: any score appears at all (even 0.0 with non-empty"
echo "  subscores). That proves runtime + replay + agent + grading fired"
echo "  end-to-end."
