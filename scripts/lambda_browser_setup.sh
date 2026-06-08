#!/usr/bin/env bash
# One-shot Lambda bootstrap for browser-in-the-loop eval (Phase 3a).
#
# Run AFTER lambda_setup.sh has succeeded and the box can already train. This
# script gets the box ready to *evaluate*: stand up Dillinger/conduit, build
# the conduit-runtime Docker image, and smoke-import the Python entry points.
#
# What it does (idempotent):
#   1. Verifies Docker is installed (Lambda Stack ships with it on most images).
#   2. Installs Python 3.11 (Dillinger requires >=3.11; Lambda's default is 3.10).
#   3. Installs `uv` (Dillinger uses uv for env management).
#   4. Installs nodejs + npm (Dillinger's run_conduit.sh has `require_cmd npm`
#      at the top, which fires even when CONDUIT_SKIP_SETUP=1 is set — so we
#      need npm on PATH even if we never actually build the viewer).
#   5. Clones github.com/refreshdotdev/Dillinger to ~/Dillinger if absent, else pulls.
#   6. uv sync inside Dillinger (installs conduit + rubric + tzafon + playwright).
#   7. uv run python -m playwright install chromium (host-side; the runtime
#      container has its own copy baked in).
#   8. docker build -t conduit-runtime ~/Dillinger  (~5-10 min, ~3GB).
#   9. Verifies ~/Dillinger/.env exists and has at least LITELLM_API_KEY +
#      ANTHROPIC_API_KEY (needed for grading judge AND Anthropic-direct Opus smoke test).
#  10. Smoke-imports conduit + verifies the runtime container can boot to /health.
#
# Pre-requisites you must do BEFORE running this:
#   - rsync ~/refresh/repos/Dillinger/.env from your Mac to Lambda:~/Dillinger/.env
#     (or paste API keys manually after step 4 fails the check).
#
# Usage on the Lambda box:
#   bash ~/cua-finetune/scripts/lambda_browser_setup.sh

set -euo pipefail

# ---- 0. sanity --------------------------------------------------------------
HOME_DIR="${HOME}"
DILLINGER_DIR="${HOME_DIR}/Dillinger"
CUA_REPO="${HOME_DIR}/cua-finetune"

echo "==> [0/10] Sanity checks"
if ! command -v nvidia-smi >/dev/null; then
  echo "ERROR: nvidia-smi not found. Run this on the Lambda GPU box." >&2
  exit 2
fi
if [[ ! -d "${CUA_REPO}" ]]; then
  echo "ERROR: ${CUA_REPO} not found. Run lambda_setup.sh first." >&2
  exit 2
fi

# ---- 1. docker --------------------------------------------------------------
echo "==> [1/10] Docker"
if ! command -v docker >/dev/null; then
  echo "Installing Docker..."
  sudo apt-get update -qq
  sudo apt-get install -y -qq docker.io
  sudo systemctl enable --now docker
  sudo usermod -aG docker "${USER}"
  echo "Docker installed. You may need to log out and back in for the group change."
  echo "For now we'll use sudo docker."
  DOCKER_CMD="sudo docker"
else
  if docker info >/dev/null 2>&1; then
    DOCKER_CMD="docker"
  else
    DOCKER_CMD="sudo docker"
  fi
fi
echo "  using: ${DOCKER_CMD}"
${DOCKER_CMD} version --format '{{.Server.Version}}' || { echo "ERROR: docker not working"; exit 2; }

# ---- 2. python 3.11 ---------------------------------------------------------
echo "==> [2/10] Python 3.11"
if ! command -v python3.11 >/dev/null; then
  echo "Installing python3.11 from deadsnakes..."
  sudo add-apt-repository -y ppa:deadsnakes/ppa
  sudo apt-get update -qq
  sudo apt-get install -y -qq python3.11 python3.11-venv python3.11-dev
fi
python3.11 --version

# ---- 3. uv ------------------------------------------------------------------
echo "==> [3/10] uv (Python pkg manager)"
if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  # uv installer puts it in ~/.local/bin
  export PATH="${HOME_DIR}/.local/bin:${PATH}"
fi
uv --version

# ---- 4. nodejs + npm --------------------------------------------------------
# Dillinger's run_conduit.sh does `require_cmd npm` at the top, before checking
# CONDUIT_SKIP_SETUP. So npm must be on PATH even if the smoke script sets
# CONDUIT_SKIP_SETUP=1 to skip the actual viewer build. The viewer build itself
# is skipped during eval runs — we only need the binary to satisfy the gate.
echo "==> [4/10] nodejs + npm (satisfies run_conduit.sh's require_cmd npm)"
if ! command -v npm >/dev/null; then
  sudo apt-get update -qq
  sudo apt-get install -y -qq nodejs npm
fi
node --version
npm --version

# ---- 5. Dillinger checkout --------------------------------------------------
echo "==> [5/10] Dillinger checkout"
if [[ ! -d "${DILLINGER_DIR}" ]]; then
  echo "  ${DILLINGER_DIR} not found. Either:"
  echo "    a) rsync your local Dillinger from your Mac, OR"
  echo "    b) git clone (needs PAT/SSH; refreshdotdev/Dillinger is private)."
  echo "  Aborting until ${DILLINGER_DIR} exists."
  exit 2
else
  echo "  Dillinger present at ${DILLINGER_DIR}"
  # Try a courtesy fetch but don't fail the whole setup if auth isn't configured.
  (cd "${DILLINGER_DIR}" && git fetch origin && git status -sb) || \
    echo "  (git fetch skipped — auth not set up; rsync'd state is fine for eval)"
fi

# ---- 6. uv sync -------------------------------------------------------------
echo "==> [6/10] uv sync (installs conduit + deps)"
cd "${DILLINGER_DIR}"
uv sync

# ---- 7. playwright (host-side, optional) ------------------------------------
echo "==> [7/10] Playwright host-side (optional; runtime container has its own)"
uv run python -m playwright install chromium || \
  echo "  (playwright host install failed — non-fatal; runtime container has chromium baked in)"

# ---- 8. docker build conduit-runtime ----------------------------------------
echo "==> [8/10] docker build conduit-runtime (~5-10 min, ~3 GB)"
if ${DOCKER_CMD} image inspect conduit-runtime >/dev/null 2>&1; then
  echo "  conduit-runtime image already built. Use 'docker rmi conduit-runtime' to force rebuild."
else
  ${DOCKER_CMD} build -t conduit-runtime "${DILLINGER_DIR}"
fi

# ---- 9. .env check ----------------------------------------------------------
echo "==> [9/10] API keys"
if [[ ! -f "${DILLINGER_DIR}/.env" ]]; then
  echo "  ${DILLINGER_DIR}/.env is MISSING."
  echo "  Either rsync it from your Mac:"
  echo "    rsync -avz ~/refresh/repos/Dillinger/.env  ubuntu@<lambda-ip>:~/Dillinger/.env"
  echo "  ...or copy ${DILLINGER_DIR}/.env.example to .env and paste keys manually:"
  echo "    cp ${DILLINGER_DIR}/.env.example ${DILLINGER_DIR}/.env && nano ${DILLINGER_DIR}/.env"
  echo "  Required at minimum:"
  echo "    ANTHROPIC_API_KEY      (rubric judge backend + Anthropic-direct Opus smoke)"
  echo "    LITELLM_API_KEY        (LiteLLM proxy key; used by eval baselines through the proxy)"
  exit 1
fi
miss=()
for key in LITELLM_API_KEY ANTHROPIC_API_KEY; do
  if ! grep -qE "^${key}=.+" "${DILLINGER_DIR}/.env"; then
    miss+=("${key}")
  fi
done
if (( ${#miss[@]} > 0 )); then
  echo "  WARN: ${DILLINGER_DIR}/.env is missing: ${miss[*]}"
  echo "  Grading will fail until these are set. Continuing anyway."
fi
echo "  ${DILLINGER_DIR}/.env present."

# ---- 10. smoke-import conduit + boot runtime --------------------------------
echo "==> [10/10] Smoke-import conduit + boot runtime container"
cd "${DILLINGER_DIR}"
uv run python -c "
from conduit.config import load_settings
from conduit.runtime.computer_loop import run_task
from conduit.runtime.runtime_client import RuntimeClient
from conduit.task_loader import load_tasks
print('conduit imports OK')
"

echo "Booting conduit-runtime container on port 7777 (will tear down after /health probe)..."
${DOCKER_CMD} rm -f conduit-runtime-smoke >/dev/null 2>&1 || true
${DOCKER_CMD} run -d --rm \
  --name conduit-runtime-smoke \
  -p 7777:8000 \
  -p 5900:5900 \
  -p 8080:8080 \
  conduit-runtime >/dev/null

echo -n "  waiting for /health"
for i in $(seq 1 60); do
  if curl -fsS http://127.0.0.1:7777/health >/dev/null 2>&1; then
    echo " -> healthy"
    break
  fi
  echo -n "."
  sleep 2
  if [[ $i -eq 60 ]]; then
    echo
    echo "ERROR: runtime did not become healthy. Logs:"
    ${DOCKER_CMD} logs conduit-runtime-smoke
    ${DOCKER_CMD} rm -f conduit-runtime-smoke
    exit 3
  fi
done

echo "  noVNC for live debug: http://<lambda-ip>:8080/vnc.html  (only if your firewall allows)"
${DOCKER_CMD} rm -f conduit-runtime-smoke >/dev/null

echo
echo "============================================================"
echo "  Phase 3a setup complete."
echo "============================================================"
echo "  Next:"
echo "    1) Smoke test the full eval loop with Claude Opus directly (no Tzafon, no us):"
echo "       bash ~/cua-finetune/scripts/eval_browser_smoke_opus.sh"
echo "       (alternative — Tzafon-backed smoke: scripts/eval_browser_smoke_tzafon.sh)"
echo "    2) Then start vLLM and run baselines for our 3 candidate models:"
echo "       bash ~/cua-finetune/scripts/serve_vllm.sh Qwen/Qwen3-VL-8B-Instruct '' 8000  # in tmux"
echo "       bash ~/cua-finetune/scripts/eval_browser_baseline.sh qwen_vl_cua             # in another tmux"
