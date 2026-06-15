"""Split eval scores by task behavioral type (recall vs navigation).

Hypothesis: the Qwen LoRA learned a more action-oriented policy. It hurts on
tasks where the answer was already visible in the initial screen (passive
baseline accidentally wins) but helps on tasks that genuinely required
navigation.

This script auto-classifies each task as ``recall`` or ``navigation`` based on
the *baseline* model's action distribution (NOT the LoRA's — we want to
classify by ground-truth task difficulty, not by what the LoRA happened to
do). Tasks where the baseline emitted mostly ``wait`` actions are tasks the
baseline could "solve" by sitting still — i.e., recall tasks.

Then it aggregates baseline vs LoRA mean score per category, so the blog can
honestly report "LoRA -X on recall, +Y on navigation" instead of a misleading
flat regression.

Usage::

    python eval/split_by_behavior.py qwen_vl_cua
    python eval/split_by_behavior.py qwen_vl_cua --wait-threshold 0.4
    python eval/split_by_behavior.py qwen_vl_cua --print-classification
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_ROOT = REPO_ROOT / "results"

# Functions that count as "passive" — the model emits these without changing
# state. wait dominates; sleep / look might appear in some action vocabularies.
PASSIVE_ACTIONS = {"wait", "sleep", "look", "noop"}


def _action_counter(trajectory_path: Path) -> Counter[str]:
    """Return a Counter of action function_name across all agent steps."""
    out: Counter[str] = Counter()
    if not trajectory_path.exists():
        return out
    try:
        data = json.loads(trajectory_path.read_text())
    except Exception:
        return out
    for step in data.get("steps", []) or []:
        for tc in step.get("tool_calls", []) or []:
            fn = tc.get("function_name")
            if isinstance(fn, str):
                out[fn] += 1
    return out


def _latest_trajectory(task_dir: Path) -> Path | None:
    if not task_dir.is_dir():
        return None
    runs = sorted(d for d in task_dir.iterdir() if d.is_dir())
    if not runs:
        return None
    return runs[-1] / "trajectory.json"


def _load_summary(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    out: dict[str, dict[str, Any]] = {}
    for line in path.open():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        task = row.get("task") or row.get("task_id") or row.get("name")
        if isinstance(task, str):
            out[task] = row
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Split baseline vs LoRA scores by behavioral category",
    )
    parser.add_argument("backend", help="e.g. qwen_vl_cua")
    parser.add_argument(
        "--results-root",
        type=Path,
        default=DEFAULT_RESULTS_ROOT,
    )
    parser.add_argument(
        "--baseline-adapter",
        default="baseline",
        help="adapter dir name for the un-finetuned run (default: baseline)",
    )
    parser.add_argument(
        "--lora-adapter",
        default="cua",
        help="adapter dir name for the finetuned run (default: cua)",
    )
    parser.add_argument(
        "--wait-threshold",
        type=float,
        default=0.5,
        help="If baseline emits >= this fraction of passive actions, classify "
             "the task as 'recall' (default: 0.5)",
    )
    parser.add_argument(
        "--print-classification",
        action="store_true",
        help="Print every task's classification + action distribution",
    )
    args = parser.parse_args()

    base_root = args.results_root / args.backend / args.baseline_adapter
    lora_root = args.results_root / args.backend / args.lora_adapter

    if not base_root.exists():
        print(f"ERROR: baseline results dir missing: {base_root}", file=sys.stderr)
        return 1
    if not lora_root.exists():
        print(f"ERROR: lora results dir missing: {lora_root}", file=sys.stderr)
        return 1

    base_summary = _load_summary(base_root / "_summary.jsonl")
    lora_summary = _load_summary(lora_root / "_summary.jsonl")

    common_tasks = sorted(set(base_summary) & set(lora_summary))
    if not common_tasks:
        print("ERROR: no tasks in common between baseline and lora summaries", file=sys.stderr)
        return 1

    # Classify each task using the BASELINE trajectory's action distribution.
    classification: dict[str, str] = {}
    action_dists: dict[str, Counter[str]] = {}
    for task in common_tasks:
        traj = _latest_trajectory(base_root / task)
        if traj is None:
            classification[task] = "unknown"
            action_dists[task] = Counter()
            continue
        counts = _action_counter(traj)
        action_dists[task] = counts
        total = sum(counts.values())
        if total == 0:
            classification[task] = "unknown"
            continue
        passive = sum(counts.get(p, 0) for p in PASSIVE_ACTIONS)
        ratio = passive / total
        classification[task] = "recall" if ratio >= args.wait_threshold else "navigation"

    if args.print_classification:
        print(f"{'task':<55} {'class':<11} {'wait%':>6} {'top actions':<40}")
        print("-" * 115)
        for task in common_tasks:
            counts = action_dists[task]
            total = sum(counts.values()) or 1
            wait_pct = 100 * sum(counts.get(p, 0) for p in PASSIVE_ACTIONS) / total
            top = ", ".join(f"{fn}:{n}" for fn, n in counts.most_common(3))
            print(f"{task[:55]:<55} {classification[task]:<11} {wait_pct:>5.0f}% {top:<40}")
        print()

    # Aggregate per category.
    by_cat: dict[str, dict[str, list[float]]] = defaultdict(lambda: {"base": [], "lora": []})
    for task in common_tasks:
        cat = classification[task]
        by_cat[cat]["base"].append(base_summary[task]["score"])
        by_cat[cat]["lora"].append(lora_summary[task]["score"])

    # Overall.
    all_base = [base_summary[t]["score"] for t in common_tasks]
    all_lora = [lora_summary[t]["score"] for t in common_tasks]
    by_cat["__OVERALL__"] = {"base": all_base, "lora": all_lora}

    print(f"{'category':<14} {'n':>4}   {'base mean':>10}   {'lora mean':>10}   {'delta':>9}   {'rel':>9}")
    print("-" * 70)
    for cat in ("recall", "navigation", "unknown", "__OVERALL__"):
        if cat not in by_cat:
            continue
        b = by_cat[cat]["base"]
        c = by_cat[cat]["lora"]
        if not b:
            continue
        bm, cm = statistics.mean(b), statistics.mean(c)
        delta = cm - bm
        rel = (cm / bm - 1) * 100 if bm > 0 else float("inf") if cm > 0 else 0.0
        rel_s = f"{rel:+.0f}%" if rel != float("inf") else "  +inf"
        label = "OVERALL" if cat == "__OVERALL__" else cat
        print(f"{label:<14} {len(b):>4}   {bm:>10.3f}   {cm:>10.3f}   {delta:>+9.3f}   {rel_s:>9}")

    print()
    print("Per-category task lists:")
    for cat in ("recall", "navigation", "unknown"):
        tasks_in_cat = [t for t in common_tasks if classification[t] == cat]
        if not tasks_in_cat:
            continue
        print(f"  {cat} ({len(tasks_in_cat)}):")
        for t in tasks_in_cat:
            b_score = base_summary[t]["score"]
            c_score = lora_summary[t]["score"]
            print(f"    {t:<55} base={b_score:.2f}  lora={c_score:.2f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
