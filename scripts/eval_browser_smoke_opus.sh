#!/usr/bin/env bash
# Phase 3b: smoke-test the conduit/runtime/grading pipeline end-to-end with
# Claude Opus DIRECTLY via Anthropic's API (NOT Tzafon, NOT our model).
#
# Goal: validate every moving part EXCEPT model serving:
#   - conduit-runtime container boots and serves /health
#   - pywb archive replay (or live web) loads the task's start URL
#   - the agent loop drives the browser, stops at terminate/done
#   - the rubric grader (Claude Opus via Anthropic SDK) returns a score
#
# If this passes, the only remaining unknown is whether vLLM-served Qwen/Kimi/
# DeepSeek can produce parseable actions. If this fails, we have a Dillinger /
# infra problem, not a model problem.
#
# Why Claude Opus direct (not Tzafon, not Bedrock)?
#   - Tzafon adds a third-party hop we don't need for a smoke test.
#   - Bedrock works (`run_opus.sh` uses it) but requires AWS creds; we already
#     have ANTHROPIC_API_KEY configured for the rubric grader, so reusing it is
#     the path of least friction.
#
# How it routes:
#   - CONDUIT_AGENT_BACKEND=litellm_chat   (Dillinger has no native anthropic
#     CUA backend; bedrock_cua is the only direct-CUA backend and it goes via
#     AWS. litellm_chat with model="anthropic/<model>" hits Anthropic directly.)
#   - LITELLM_MODEL=anthropic/claude-opus-4-5-20251101  (recent Opus, matches
#     the version already used by Dillinger's default Anthropic judge)
#   - LITELLM_BASE_URL and LITELLM_API_KEY are temporarily blanked in .env so
#     LiteLLM doesn't try to go through Tzafon's proxy with the wrong key.
#     LiteLLM then reads ANTHROPIC_API_KEY from the env directly.
#   - CONDUIT_JUDGE_BACKEND=anthropic so the rubric grader also uses
#     ANTHROPIC_API_KEY (via Anthropic SDK) instead of the LiteLLM proxy.
#
# Cost: ~$0.50 (1 task, ~10 turns of Opus + 1 grading call).
#
# Pre-reqs:
#   - lambda_browser_setup.sh has been run (installs docker, nodejs/npm, uv,
#     conduit-runtime image, etc.)
#   - ~/Dillinger/.env has ANTHROPIC_API_KEY + LITELLM_API_KEY
#     (LITELLM_API_KEY is only required by lambda_browser_setup's sanity
#      check; this script blanks it for the smoke run.)
#
# Usage:
#   bash scripts/eval_browser_smoke_opus.sh [TASK_NAME] [N_RUNS]
# Defaults: TASK_NAME=pydocs-os-sched-policy, N_RUNS=1

set -euo pipefail

DILLINGER_DIR="${HOME}/Dillinger"
TASK_NAME="${1:-pydocs-os-sched-policy}"
N_RUNS="${2:-1}"

# ---- 0. preflight -----------------------------------------------------------
if [[ ! -d "${DILLINGER_DIR}" ]]; then
  echo "ERROR: ${DILLINGER_DIR} not found. Run lambda_browser_setup.sh first." >&2
  exit 2
fi
if [[ ! -f "${DILLINGER_DIR}/.env" ]]; then
  echo "ERROR: ${DILLINGER_DIR}/.env missing" >&2
  exit 2
fi

# Only ANTHROPIC_API_KEY is strictly required for this script. LITELLM_API_KEY
# stays a soft requirement so the rest of the eval harness (which can run
# against the Tzafon LiteLLM proxy) still works on this box.
for key in ANTHROPIC_API_KEY; do
  if ! grep -qE "^${key}=.+" "${DILLINGER_DIR}/.env"; then
    echo "ERROR: ${DILLINGER_DIR}/.env is missing ${key} (required for Opus smoke)" >&2
    exit 2
  fi
done

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

# ---- 1. temporarily patch .env to point LiteLLM at Anthropic directly -------
#
# Why: run_conduit.sh does `set -a; source .env; set +a` which would clobber
# any shell-level overrides we export. So we patch .env in place and restore
# it on exit. The original is backed up to .env.smoke_opus.bak; if this script
# is SIGKILLed mid-run, restore manually with:
#   mv ~/Dillinger/.env.smoke_opus.bak ~/Dillinger/.env
ENV_BACKUP="${DILLINGER_DIR}/.env.smoke_opus.bak"
cp "${DILLINGER_DIR}/.env" "${ENV_BACKUP}"
trap 'mv "${ENV_BACKUP}" "${DILLINGER_DIR}/.env" 2>/dev/null || true' EXIT INT TERM

python3 - <<'PY'
import os, re, pathlib
p = pathlib.Path(os.path.expanduser("~/Dillinger/.env"))
text = p.read_text()

def upsert(key: str, value: str) -> None:
    global text
    pattern = rf"^{re.escape(key)}=.*$"
    if re.search(pattern, text, re.M):
        text = re.sub(pattern, f"{key}={value}", text, flags=re.M)
    else:
        text = text.rstrip() + f"\n{key}={value}\n"

# Force LiteLLM to hit Anthropic directly (no proxy, no proxy key).
upsert("LITELLM_BASE_URL", "")
upsert("LITELLM_API_KEY", "")
upsert("LITELLM_MODEL", "anthropic/claude-opus-4-5-20251101")

# Drive the browser loop with Claude Opus via Anthropic-direct.
upsert("CONDUIT_AGENT_BACKEND", "litellm_chat")

# Grade with Anthropic SDK directly (also uses ANTHROPIC_API_KEY) so we
# don't depend on a LiteLLM proxy for grading either.
upsert("CONDUIT_JUDGE_BACKEND", "anthropic")
upsert("CONDUIT_JUDGE_ANTHROPIC_MODEL", "claude-opus-4-5-20251101")

p.write_text(text)
print("patched .env -> litellm_chat / anthropic/claude-opus-4-5-20251101 / anthropic judge")
PY

# ---- 2. run -----------------------------------------------------------------
# CONDUIT_SKIP_SETUP=1 tells run_conduit.sh to skip `uv sync`, playwright
# install, the viewer npm install/build, and the conduit-runtime docker build.
# All of those were handled by lambda_browser_setup.sh already.
#
# NOTE: run_conduit.sh still has `require_cmd npm` at the top (line 241) which
# runs BEFORE the SKIP_SETUP check, so npm must be installed even with this
# var set. lambda_browser_setup.sh installs nodejs+npm to satisfy that gate.
export CONDUIT_SKIP_SETUP=1

bash ./run_conduit.sh -k "${N_RUNS}" "${TASK_YAML}" --task "${TASK_NAME}" --backend litellm_chat

# ---- 3. report --------------------------------------------------------------
echo
echo "============================================================"
echo "  Opus smoke test complete."
echo "============================================================"
echo "  Look in ${DILLINGER_DIR}/runs/${TASK_NAME}/ for the latest run output."
echo "  Open ${DILLINGER_DIR}/runs/${TASK_NAME}/<latest>/result.json to see the rubric score."
echo "  If score > 0 the pipeline works end-to-end."
