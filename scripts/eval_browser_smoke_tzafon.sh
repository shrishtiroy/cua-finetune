#!/usr/bin/env bash
# Phase 3b: smoke test the conduit/runtime/grading pipeline end-to-end with
# Tzafon's CUA-as-a-service backend (NOT our model). The point is to validate
# every moving part EXCEPT model serving:
#
#   - conduit-runtime container boots and serves /health
#   - pywb archive replay loads the task's start URL
#   - the agent loop drives the browser, stops at terminate/done
#   - the rubric grader (Claude Opus via LiteLLM) returns a score
#
# If this passes, the only remaining unknown is whether vLLM-served Qwen/Kimi/
# DeepSeek can produce parseable actions. If this fails, we have a Dillinger /
# infra problem, not a model problem.
#
# Cost: ~$0.50 (1 task, ~10 turns of Tzafon Opus + 1 grading call).
#
# Pre-reqs:
#   - lambda_browser_setup.sh has been run
#   - ~/Dillinger/.env has at minimum LITELLM_API_KEY + ANTHROPIC_API_KEY (grader)
#   - For Tzafon backend: TZAFON_API_KEY
#   - For Claude backend (fallback): just ANTHROPIC_API_KEY
#
# Usage:
#   scripts/eval_browser_smoke_tzafon.sh                                    # tzafon default
#   scripts/eval_browser_smoke_tzafon.sh pydocs-os-sched-policy 1           # explicit task
#   SMOKE_BACKEND=litellm_chat scripts/eval_browser_smoke_tzafon.sh         # use Claude

set -euo pipefail

DILLINGER_DIR="${HOME}/Dillinger"
TASK_NAME="${1:-pydocs-os-sched-policy}"   # pick a held-out task, default is a pydocs task
N_RUNS="${2:-1}"
SMOKE_BACKEND="${SMOKE_BACKEND:-tzafon_responses}"  # or litellm_chat (Claude via LiteLLM)

if [[ ! -d "${DILLINGER_DIR}" ]]; then
  echo "ERROR: ${DILLINGER_DIR} not found. Run lambda_browser_setup.sh first." >&2
  exit 2
fi
if [[ ! -f "${DILLINGER_DIR}/.env" ]]; then
  echo "ERROR: ${DILLINGER_DIR}/.env missing" >&2
  exit 2
fi

# Verify the needed API key is present for the chosen smoke backend
case "${SMOKE_BACKEND}" in
  tzafon_responses)
    if ! grep -qE "^TZAFON_API_KEY=.+" "${DILLINGER_DIR}/.env"; then
      echo "ERROR: TZAFON_API_KEY missing from ${DILLINGER_DIR}/.env" >&2
      echo "Either add it, or rerun with: SMOKE_BACKEND=litellm_chat $0 $@" >&2
      exit 2
    fi
    ;;
  litellm_chat)
    if ! grep -qE "^ANTHROPIC_API_KEY=.+" "${DILLINGER_DIR}/.env"; then
      echo "ERROR: ANTHROPIC_API_KEY missing from ${DILLINGER_DIR}/.env" >&2
      exit 2
    fi
    ;;
esac

cd "${DILLINGER_DIR}"

# Find the YAML that defines this task name
TASK_YAML="$(grep -rl "name: ${TASK_NAME}" tasks/ 2>/dev/null | head -1 || true)"
if [[ -z "${TASK_YAML}" ]]; then
  echo "ERROR: no YAML in ${DILLINGER_DIR}/tasks/ defines task '${TASK_NAME}'." >&2
  echo "Try one of:" >&2
  grep -rh "^- name: " tasks/ 2>/dev/null | head -10 >&2
  exit 2
fi
echo "Task '${TASK_NAME}' found in ${TASK_YAML}"

# Use the existing run_conduit.sh orchestrator. It:
#   - builds conduit-runtime if needed
#   - spins up the runtime container
#   - runs `conduit run --tasks-file ... --task <name> --backend <SMOKE_BACKEND>`
#   - tears down the container
#
# tzafon_responses uses the Tzafon Responses API as the CUA brain.
# litellm_chat uses Claude (or whatever conduit's litellm_model config points at)
# via LiteLLM. Either way, this is just a sanity check of the rest of the pipeline.
export CONDUIT_AGENT_BACKEND="${SMOKE_BACKEND}"
echo "Smoke backend: ${SMOKE_BACKEND}"

bash ./run_conduit.sh -k "${N_RUNS}" "${TASK_YAML}" --task "${TASK_NAME}"

echo
echo "============================================================"
echo "  Smoke test complete."
echo "============================================================"
echo "  Look in ${DILLINGER_DIR}/runs/ for the latest run output."
echo "  Open ${DILLINGER_DIR}/runs/<latest>/result.json to see the rubric score."
echo "  If score > 0 the pipeline works end-to-end."
