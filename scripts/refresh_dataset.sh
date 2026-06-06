#!/usr/bin/env bash
# Refresh the SFT dataset end-to-end after new Opus trajectories are uploaded
# to Supabase (or after new local Dillinger runs are produced). Idempotent —
# safe to re-run anytime; downloads only NEW trajectories.
#
# What it does:
#   1. Pull from all sources (Supabase + local Dillinger + trajectories.sh).
#      data/raw/supabase/ acts as a content-addressable cache; previously-pulled
#      trials are skipped instantly. Only NEW trial_ids hit the network.
#   2. Re-inventory (score distribution, per-source totals).
#   3. Re-categorize using the latest categorize_tasks.py rules.
#   4. Re-build the SFT JSONL with rejection sampling (default --pass-threshold 1.0).
#   5. Re-validate (sample 5 rows, check images decode + assistant JSON parses).
#   6. Print a diff: how many new trajectories, new tasks, change in train/test row counts.
#
# Usage:
#   scripts/refresh_dataset.sh                   # full refresh
#   scripts/refresh_dataset.sh --pass-threshold 0.8   # relax everywhere
#   scripts/refresh_dataset.sh --dry-run         # don't write, just show what'd change
#
# After this, retrain with scripts/train_all.sh (Phase 2). Eval is independent
# (Phase 3) and only needs to be rerun if the held-out task split changed.
set -euo pipefail

cd "$(dirname "$0")/.."

# ---- arg passthrough -----------------------------------------------------
PASS_THRESHOLD=1.0
MAX_PER_TASK=3
DRY_RUN=""
EXTRA_BUILD_ARGS=()
EXTRA_PULL_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --pass-threshold) PASS_THRESHOLD="$2"; shift 2 ;;
    --max-per-task)   MAX_PER_TASK="$2";   shift 2 ;;
    --dry-run)        DRY_RUN="--dry-run"; shift ;;
    --skip-pull)      SKIP_PULL=1;         shift ;;
    --supabase-only)  EXTRA_PULL_ARGS+=(--sources supabase); shift ;;
    --local-only)     EXTRA_PULL_ARGS+=(--sources local);    shift ;;
    *) EXTRA_BUILD_ARGS+=("$1"); shift ;;
  esac
done

# ---- env -----------------------------------------------------------------
if [[ ! -f .env ]]; then
  echo "ERROR: .env not found at $(pwd)/.env" >&2
  echo "Set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY before running." >&2
  exit 2
fi

if [[ -z "${VIRTUAL_ENV:-}" && -d .venv ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

# ---- snapshot BEFORE state ----------------------------------------------
mkdir -p data/manifests
manifest=data/manifests/pulled_trajectories.jsonl
train_jsonl=data/cua_sft/train.jsonl
test_jsonl=data/cua_sft/test.jsonl

count_lines() { [[ -f "$1" ]] && wc -l < "$1" | tr -d ' ' || echo 0; }

before_manifest=$(count_lines "$manifest")
before_train=$(count_lines "$train_jsonl")
before_test=$(count_lines "$test_jsonl")
before_unique_tasks=0
if [[ -f "$manifest" ]]; then
  before_unique_tasks=$(python3 -c "
import json,sys
seen=set()
for line in open('$manifest'):
    try: r=json.loads(line)
    except: continue
    n=r.get('task_external_id') or r.get('task_name')
    if n: seen.add(n)
print(len(seen))" 2>/dev/null || echo 0)
fi

echo "================================================================"
echo " BEFORE: manifest=$before_manifest rows, train=$before_train, test=$before_test, unique_tasks=$before_unique_tasks"
echo "================================================================"

# ---- 1. PULL -------------------------------------------------------------
if [[ -z "${SKIP_PULL:-}" ]]; then
  echo
  echo "[1/5] PULL — Supabase + local Dillinger + trajectories.sh (idempotent, appending)"
  python data/pull_trajectories.py ${EXTRA_PULL_ARGS[@]+"${EXTRA_PULL_ARGS[@]}"} $DRY_RUN \
    2>&1 | tee /tmp/cua-refresh-pull.log
else
  echo "[1/5] PULL skipped (--skip-pull)"
fi

# ---- 2. INVENTORY --------------------------------------------------------
echo
echo "[2/5] INVENTORY"
python data/inventory.py 2>&1 | tee /tmp/cua-refresh-inventory.log

# ---- 3. CATEGORIZE -------------------------------------------------------
echo
echo "[3/5] CATEGORIZE"
python data/categorize_tasks.py 2>&1 | tee /tmp/cua-refresh-cat.log

# ---- 4. BUILD ------------------------------------------------------------
echo
echo "[4/5] BUILD — rejection sampling, pass_threshold=$PASS_THRESHOLD max_per_task=$MAX_PER_TASK"
if [[ -n "$DRY_RUN" ]]; then
  echo "  (dry-run; skipping JSONL write)"
else
  python data/atif_to_swift.py \
    --pass-threshold "$PASS_THRESHOLD" \
    --max-per-task "$MAX_PER_TASK" \
    ${EXTRA_BUILD_ARGS[@]+"${EXTRA_BUILD_ARGS[@]}"} \
    2>&1 | tee /tmp/cua-refresh-build.log
fi

# ---- 5. VALIDATE ---------------------------------------------------------
echo
echo "[5/5] VALIDATE"
if [[ -f data/validate_dataset.py ]]; then
  python data/validate_dataset.py 2>&1 | tee /tmp/cua-refresh-validate.log
else
  echo "  (data/validate_dataset.py not present, skipping)"
fi

# ---- snapshot AFTER state + diff ----------------------------------------
after_manifest=$(count_lines "$manifest")
after_train=$(count_lines "$train_jsonl")
after_test=$(count_lines "$test_jsonl")
after_unique_tasks=0
if [[ -f "$manifest" ]]; then
  after_unique_tasks=$(python3 -c "
import json
seen=set()
for line in open('$manifest'):
    try: r=json.loads(line)
    except: continue
    n=r.get('task_external_id') or r.get('task_name')
    if n: seen.add(n)
print(len(seen))" 2>/dev/null || echo 0)
fi

dm=$(( after_manifest - before_manifest ))
dt=$(( after_train - before_train ))
de=$(( after_test - before_test ))
du=$(( after_unique_tasks - before_unique_tasks ))

echo
echo "================================================================"
echo " AFTER:  manifest=$after_manifest rows (Δ$dm), train=$after_train (Δ$dt), test=$after_test (Δ$de), unique_tasks=$after_unique_tasks (Δ$du)"
echo "================================================================"

if [[ -f data/manifests/rejection_sampling_report.yaml ]]; then
  echo
  echo "Per-category rejection-sampling report:"
  python3 -c "
import yaml
r = yaml.safe_load(open('data/manifests/rejection_sampling_report.yaml'))
pc = r.get('per_category', {}) or {}
overrides = r.get('overrides', {}) or {}
print(f\"  pass_threshold (default): {r.get('pass_threshold_default')}\")
print(f\"  max_per_task:             {r.get('max_per_task')}\")
print(f\"  overrides (relaxed cats): {overrides if overrides else '(none)'}\")
print()
print(f\"  {'category':<20} {'thr':>5} {'cand':>6} {'sel':>6} {'trajs':>7} {'avg_steps':>10}\")
for cat, info in pc.items():
    thr = info.get('pass_threshold_used','?')
    print(f\"  {cat:<20} {str(thr):>5} {info.get('candidate_tasks',0):>6} {info.get('selected_tasks',0):>6} {info.get('selected_trajectories',0):>7} {info.get('avg_n_steps',0):>10}\")
"
fi

echo
echo "Refresh complete."
echo "  Logs: /tmp/cua-refresh-{pull,inventory,cat,build,validate}.log"
echo "  Next: scripts/train_all.sh   (only re-run training if dataset row count changed meaningfully)"
