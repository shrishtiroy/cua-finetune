#!/usr/bin/env bash
# One-shot Lambda H100 bootstrap for cua-finetune.
#
# Run AFTER you have:
#   1. Provisioned a Lambda 1xH100 80GB instance (Lambda Stack Ubuntu 22.04 image)
#   2. SSH'd in as `ubuntu`
#   3. rsync'd this repo to ~/cua-finetune on the box
#   4. Copied .env (with SUPABASE_URL + SUPABASE_SERVICE_ROLE_KEY) to ~/cua-finetune/.env
#
# What this does:
#   1. Creates a python 3.11 venv at ~/cua-finetune/.venv
#   2. Installs torch 2.5 + flash-attn (matching the Lambda Stack CUDA 12.4 base)
#   3. Installs ms-swift 4.x with VLM extras
#   4. Installs cua-finetune project + supabase client
#   5. Re-pulls trajectories from Supabase to ~/cua-finetune/data/raw/supabase/
#      (~4.3 GB, 10-30 min depending on network)
#   6. Re-builds train.jsonl/test.jsonl with Lambda-local image paths
#   7. Smoke-tests one swift sft batch (no actual training) to confirm GPU + template work
#
# Idempotent: re-running won't redownload existing files.

set -euo pipefail

cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"

# ---- 0. sanity --------------------------------------------------------------
if [[ ! -f .env ]]; then
  echo "ERROR: $REPO_ROOT/.env missing. Copy it from your Mac before running this." >&2
  exit 2
fi
if ! command -v nvidia-smi >/dev/null; then
  echo "ERROR: nvidia-smi not found. Are you on a Lambda GPU instance?" >&2
  exit 2
fi
echo "GPU detected:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

# ---- 1. python venv ---------------------------------------------------------
if [[ ! -d .venv ]]; then
  echo "Creating .venv (python 3.11)"
  python3.11 -m venv .venv 2>/dev/null || python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --upgrade pip wheel setuptools

# ---- 2. torch + flash-attn (CUDA 12.4 base) ---------------------------------
# Lambda Stack ships with CUDA 12.4 + python 3.11. Pin torch to a known-good
# combo for ms-swift v4.x.
pip install --no-cache-dir \
  "torch==2.5.1" "torchvision==0.20.1" "torchaudio==2.5.1" \
  --index-url https://download.pytorch.org/whl/cu124

# flash-attn — pre-built wheel for cu124/torch 2.5
pip install --no-cache-dir flash-attn==2.7.0.post2 --no-build-isolation || {
  echo "flash-attn install failed; ms-swift will fall back to sdpa (slower but works)."
}

# ---- 3. ms-swift + project deps --------------------------------------------
pip install --no-cache-dir "ms-swift[llm]==4.0.*" deepspeed==0.16.* tensorboard
pip install --no-cache-dir -e .

# ---- 4. re-pull trajectories from Supabase ---------------------------------
echo
echo "Re-pulling trajectories from Supabase (cached, idempotent)..."
python data/pull_trajectories.py 2>&1 | tee /tmp/lambda-pull.log

# ---- 5. categorize + build dataset ------------------------------------------
echo
echo "Building SFT dataset with Lambda-local paths..."
python data/categorize_tasks.py 2>&1 | tee /tmp/lambda-cat.log
python data/atif_to_swift.py --pass-threshold 1.0 --max-per-task 3 \
  2>&1 | tee /tmp/lambda-build.log

# ---- 6. quick path-resolution sanity check ---------------------------------
echo
echo "Verifying first 3 training rows resolve to local images:"
python3 - <<'PY'
import json
from pathlib import Path
ok = 0
bad = 0
with open("data/cua_sft/train.jsonl") as h:
    for i, line in enumerate(h):
        if i >= 3: break
        r = json.loads(line)
        for img in r.get("images", []):
            if Path(img).exists():
                ok += 1
            else:
                bad += 1
                print(f"  MISSING: {img}")
print(f"  resolved: {ok}, missing: {bad}")
assert bad == 0, "image paths not resolving — manifest/raw cache mismatch"
PY

# ---- 7. swift sft dry-run (1 forward pass) ---------------------------------
echo
echo "Smoke-testing swift sft (1 forward pass, no optimization)..."
swift sft --config configs/qwen3_vl_8b.yaml --max_steps 1 --output_dir ./checkpoints/_smoke \
  2>&1 | tee /tmp/lambda-smoke.log | tail -40

echo
echo "================================================================"
echo " Bootstrap complete."
echo " GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1)"
echo " Train rows: $(wc -l < data/cua_sft/train.jsonl)"
echo " Test rows:  $(wc -l < data/cua_sft/test.jsonl)"
echo
echo " Next: kick off the real run in a tmux/screen session:"
echo "   tmux new -s qwen"
echo "   source .venv/bin/activate"
echo "   swift sft --config configs/qwen3_vl_8b.yaml 2>&1 | tee logs/qwen_v1.log"
echo "   # detach with C-b d; reattach with: tmux attach -t qwen"
echo "================================================================"
