#!/usr/bin/env bash
# Phase 3b smoke test using Anthropic's NATIVE computer-use API via AWS
# Bedrock. This is the cleanest "borrow-someone-else's-CUA" path because:
#
#   1. Dillinger's bedrock_cua backend already implements the official
#      computer_20250124 / computer_20251124 tool spec (the actual API
#      Anthropic CUAs are trained against — not generic JSON tool calling).
#   2. AWS_* creds + CONDUIT_AGENT_BACKEND=bedrock_cua are already present
#      in ~/Dillinger/.env, so we don't have to patch anything.
#   3. We're not paying Tzafon margin or LiteLLM wrapper overhead.
#
# What this script validates (everything BUT our model):
#   - conduit-runtime container boots and serves /health
#   - pywb archive replay loads docs-python-org.wacz
#   - the agent loop drives the browser turn-by-turn until terminate
#   - the rubric grader (Claude Opus via LiteLLM judge) returns a score
#
# If this passes, the only remaining unknown is whether our vLLM-served
# Qwen/Kimi/DeepSeek can produce parseable actions. If this fails, we
# have a Dillinger / Docker / archive / grading problem, not a model
# problem.
#
# Cost: ~$0.50 per run (1 task, ~10 turns of Bedrock Opus + 1 grading call).
#
# Pre-reqs (verify before running):
#   - lambda_browser_setup.sh has succeeded
#   - ~/Dillinger/.env contains:
#       AWS_ACCESS_KEY_ID=...
#       AWS_SECRET_ACCESS_KEY=...
#       AWS_REGION_NAME=us-east-1            (or your preferred region)
#       CONDUIT_AGENT_BACKEND=bedrock_cua    (default in your .env already)
#       LITELLM_API_KEY=...                  (used by the rubric grader)
#       ANTHROPIC_API_KEY=...                (used by anthropic judge backend)
#   - Your AWS account has Bedrock model access enabled for Claude Opus
#     in your selected region (one-time grant in the AWS console).
#
# Usage:
#   bash scripts/eval_browser_smoke_bedrock.sh
#   bash scripts/eval_browser_smoke_bedrock.sh pydocs-os-urandom-pep
#   CONDUIT_BEDROCK_MODEL=global.anthropic.claude-opus-4-7 \
#     bash scripts/eval_browser_smoke_bedrock.sh

set -euo pipefail

DILLINGER_DIR="${HOME}/Dillinger"
TASK_NAME="${1:-pydocs-os-sched-policy}"
N_RUNS="${2:-1}"

# Default to Claude Opus 4.5 on Bedrock (one of the models bedrock_cua's
# _NEW_BETA_PATTERNS list recognises and routes through the
# computer-use-2025-11-24 beta header). User can override via env.
export CONDUIT_BEDROCK_MODEL="${CONDUIT_BEDROCK_MODEL:-global.anthropic.claude-opus-4-5}"
export CONDUIT_AGENT_BACKEND="bedrock_cua"

# CONDUIT_SKIP_SETUP=1 short-circuits run_conduit.sh's heavy first-run
# steps (uv sync / playwright install / npm install / npm run build /
# docker build). Note: it does NOT skip the `require_cmd npm` gate
# itself (that runs before the env check), so npm must already be on
# PATH — `lambda_browser_setup.sh` installs it.
export CONDUIT_SKIP_SETUP=1

# Sanity-check the env first so we fail before spinning up Docker.
if [[ ! -d "${DILLINGER_DIR}" ]]; then
  echo "ERROR: ${DILLINGER_DIR} not found. Run lambda_browser_setup.sh first." >&2
  exit 2
fi
if [[ ! -f "${DILLINGER_DIR}/.env" ]]; then
  echo "ERROR: ${DILLINGER_DIR}/.env missing" >&2
  exit 2
fi

missing_keys=()
for key in AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_REGION_NAME LITELLM_API_KEY; do
  if ! grep -qE "^${key}=.+" "${DILLINGER_DIR}/.env"; then
    missing_keys+=("${key}")
  fi
done
if (( ${#missing_keys[@]} > 0 )); then
  echo "ERROR: ${DILLINGER_DIR}/.env is missing: ${missing_keys[*]}" >&2
  echo "       Bedrock smoke test requires these. Add them and re-run." >&2
  exit 2
fi

cd "${DILLINGER_DIR}"

TASK_YAML="$(grep -rl "name: ${TASK_NAME}\b" tasks/ 2>/dev/null | head -1 || true)"
if [[ -z "${TASK_YAML}" ]]; then
  echo "ERROR: no YAML in ${DILLINGER_DIR}/tasks/ defines task '${TASK_NAME}'." >&2
  echo "Available pydocs tasks:" >&2
  grep -rh "^- name: pydocs-" tasks/ 2>/dev/null | head -8 >&2
  exit 2
fi

echo "==> Bedrock smoke test"
echo "    backend       : bedrock_cua"
echo "    model         : ${CONDUIT_BEDROCK_MODEL}"
echo "    region        : (from .env AWS_REGION_NAME)"
echo "    task          : ${TASK_NAME}"
echo "    yaml          : ${TASK_YAML}"
echo "    n_runs        : ${N_RUNS}"
echo "    skip_setup    : ${CONDUIT_SKIP_SETUP}"
echo

bash ./run_conduit.sh -k "${N_RUNS}" "${TASK_YAML}" --task "${TASK_NAME}"

echo
echo "============================================================"
echo "  Smoke test complete."
echo "============================================================"
echo "  Latest run:"
LATEST_RUN_DIR="$(ls -td "${DILLINGER_DIR}/runs/${TASK_NAME}"/* 2>/dev/null | head -1 || true)"
if [[ -n "${LATEST_RUN_DIR}" ]]; then
  echo "    ${LATEST_RUN_DIR}"
  if [[ -f "${LATEST_RUN_DIR}/result.json" ]]; then
    echo "  Score (from result.json):"
    python3 -c "
import json, sys
try:
    r = json.load(open('${LATEST_RUN_DIR}/result.json'))
    print(f\"    score={r.get('score', 'N/A')}\")
    if 'subscores' in r:
        for s in r['subscores']:
            print(f\"      {s.get('weight', 0)}pt - {s.get('met', '?'):5} - {(s.get('requirement') or '')[:60]}\")
except Exception as e:
    print(f'    (could not parse result.json: {e})')
"
  else
    echo "  (no result.json yet — run may have failed before grading)"
  fi
else
  echo "    (no run output found — the loop may not have started)"
fi
echo
echo "  Pass criterion: any score appears at all (even 0.0 with non-empty"
echo "  subscores). That proves runtime + replay + agent + grading all"
echo "  fired end-to-end."
