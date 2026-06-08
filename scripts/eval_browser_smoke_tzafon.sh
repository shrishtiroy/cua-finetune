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
#   - ~/Dillinger/.env has TZAFON_API_KEY + LITELLM_API_KEY + ANTHROPIC_API_KEY

set -euo pipefail

DILLINGER_DIR="${HOME}/Dillinger"
TASK_NAME="${1:-pydocs-os-sched-policy}"   # pick a held-out task, default is a pydocs task
N_RUNS="${2:-1}"

if [[ ! -d "${DILLINGER_DIR}" ]]; then
  echo "ERROR: ${DILLINGER_DIR} not found. Run lambda_browser_setup.sh first." >&2
  exit 2
fi
if [[ ! -f "${DILLINGER_DIR}/.env" ]]; then
  echo "ERROR: ${DILLINGER_DIR}/.env missing" >&2
  exit 2
fi

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
#   - runs `conduit run --tasks-file ... --task <name> --backend tzafon_responses`
#   - tears down the container
#
# CONDUIT_AGENT_BACKEND=tzafon_responses uses the Tzafon Responses API as the
# CUA brain (not us; this is just a sanity check of the rest of the pipeline).
export CONDUIT_AGENT_BACKEND="tzafon_responses"

bash ./run_conduit.sh -k "${N_RUNS}" "${TASK_YAML}" --task "${TASK_NAME}"

echo
echo "============================================================"
echo "  Smoke test complete."
echo "============================================================"
echo "  Look in ${DILLINGER_DIR}/runs/ for the latest run output."
echo "  Open ${DILLINGER_DIR}/runs/<latest>/result.json to see the rubric score."
echo "  If score > 0 the pipeline works end-to-end."
