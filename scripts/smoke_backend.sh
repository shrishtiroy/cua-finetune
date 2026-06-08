#!/usr/bin/env bash
# Convenience wrapper: run smoke_backend.py from Dillinger's uv-managed venv
# (which has conduit + httpx + pydantic) with cua-finetune on PYTHONPATH (so
# eval.backends.* are importable).
#
# Usage:
#   bash scripts/smoke_backend.sh <BACKEND>           # baseline adapter
#   bash scripts/smoke_backend.sh <BACKEND> cua       # cua LoRA adapter

set -euo pipefail

BACKEND="${1:?BACKEND required (qwen_vl_cua | kimi_vl_cua | deepseek_vl_cua | llama_vision_cua)}"
ADAPTER="${2:-baseline}"

CUA_REPO="${HOME}/cua-finetune"
DILLINGER_DIR="${HOME}/Dillinger"

if [[ ! -d "${DILLINGER_DIR}/.venv" ]]; then
  echo "ERROR: ${DILLINGER_DIR}/.venv missing. Run lambda_browser_setup.sh first." >&2
  exit 2
fi

cd "${DILLINGER_DIR}"
PYTHONPATH="${CUA_REPO}" \
CUA_LORA_ADAPTER="${ADAPTER}" \
  uv run python "${CUA_REPO}/scripts/smoke_backend.py" \
    --backend "${BACKEND}" \
    --adapter "${ADAPTER}"
