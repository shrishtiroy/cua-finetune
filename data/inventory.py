"""Phase 0c: inventory + integrity report over ``data/manifests/pulled_trajectories.jsonl``.

This file is self-contained on purpose — earlier revisions imported helpers from
``pull_trajectories.py``, which broke whenever the row-schema there drifted.
Everything we need is right here and reads the manifest at face value
(``source``, ``kind``, ``task_name``, ``score``, ``backend``, ``model_used``,
``n_steps``, ``extra``).

Prints:

  * total rows, by source / by kind / by backend
  * score distribution (=1.0, ≥0.8, ≥0.5, >0, =0) and unique-task counts at each
  * trajectories per task (avg / max)
  * screenshot integrity for a small random sample
  * per-category breakdown (if ``categories.yaml`` exists)

Exits non-zero (rc=2) only if the trainable trajectory pool falls below
``--min-trainable`` (default 100) so downstream CI can gate on it.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "data" / "manifests" / "pulled_trajectories.jsonl"
DEFAULT_CATEGORIES = REPO_ROOT / "data" / "manifests" / "categories.yaml"


def _load_manifest(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _load_categories(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if isinstance(data, dict):
        if all(isinstance(v, str) for v in data.values()):
            return dict(data)
        out: dict[str, str] = {}
        for cat, tasks in data.items():
            if isinstance(tasks, list):
                for t in tasks:
                    out[str(t)] = str(cat)
        return out
    return {}


def _check_screenshots(row: dict[str, Any]) -> dict[str, Any]:
    extra = row.get("extra") or {}
    sd = extra.get("screenshots_dir")
    if not sd:
        return {"present": False, "count_png": 0, "count_jpg": 0, "count_webp": 0}
    p = Path(sd)
    if not p.exists():
        return {"present": False, "count_png": 0, "count_jpg": 0, "count_webp": 0}
    return {
        "present": True,
        "count_png": len(list(p.glob("*.png"))),
        "count_jpg": len(list(p.glob("*.jpg"))) + len(list(p.glob("*.jpeg"))),
        "count_webp": len(list(p.glob("*.webp"))),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Inventory the trajectory pool")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--categories", type=Path, default=DEFAULT_CATEGORIES)
    parser.add_argument("--min-trainable", type=int, default=100)
    parser.add_argument("--score-threshold", type=float, default=80.0)
    parser.add_argument("--sample-size", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.manifest.exists():
        print(f"ERROR: manifest not found at {args.manifest}; run pull_trajectories.py first",
              file=sys.stderr)
        return 1

    rows = _load_manifest(args.manifest)
    categories = _load_categories(args.categories)

    by_source: Counter[str] = Counter()
    by_kind: Counter[str] = Counter()
    by_backend: Counter[str] = Counter()
    by_model: Counter[str] = Counter()
    by_task_traj: dict[str, list[dict[str, Any]]] = defaultdict(list)

    trajectory_rows = [r for r in rows if r.get("kind") == "trajectory"]
    task_def_rows = [r for r in rows if r.get("kind") == "task_definition"]
    eval_log_rows = [r for r in rows if r.get("kind") == "eval_log"]

    # Score buckets on the unified 0..100 scale.
    score_buckets = {
        "=100":  0,
        ">=80":  0,
        ">=50":  0,
        ">0":    0,
        "=0":    0,
        "missing": 0,
    }
    tasks_in_bucket: dict[str, set[str]] = defaultdict(set)

    trainable = 0
    screenshot_health: list[dict[str, Any]] = []

    for row in rows:
        by_source[row.get("source", "?")] += 1
        by_kind[row.get("kind", "?")] += 1
        if row.get("backend"):
            by_backend[row["backend"]] += 1
        if row.get("model_used"):
            by_model[row["model_used"]] += 1

        if row.get("kind") != "trajectory":
            continue
        task_name = row.get("task_name") or "?"
        by_task_traj[task_name].append(row)
        s = row.get("score")
        if s is None:
            score_buckets["missing"] += 1
            continue
        # Bucket boundaries (cumulative thresholds).
        if s >= 100:
            score_buckets["=100"] += 1
            tasks_in_bucket["=100"].add(task_name)
        if s >= 80:
            score_buckets[">=80"] += 1
            tasks_in_bucket[">=80"].add(task_name)
        if s >= 50:
            score_buckets[">=50"] += 1
            tasks_in_bucket[">=50"].add(task_name)
        if s > 0:
            score_buckets[">0"] += 1
            tasks_in_bucket[">0"].add(task_name)
        if s == 0:
            score_buckets["=0"] += 1
            tasks_in_bucket["=0"].add(task_name)

        if s >= args.score_threshold:
            trainable += 1

        check = _check_screenshots(row)
        check["task_name"] = task_name
        check["score"] = s
        screenshot_health.append(check)

    print("=" * 72)
    print("TRAJECTORY POOL INVENTORY")
    print("=" * 72)
    print(f"Manifest:        {args.manifest}")
    print(f"Total rows:      {len(rows)}")
    print(f"  trajectories:   {len(trajectory_rows)}")
    print(f"  task_defs:      {len(task_def_rows)}")
    print(f"  eval_logs:      {len(eval_log_rows)}")
    print()
    print("By source:")
    for src, n in by_source.most_common():
        print(f"  {src:30s} {n}")
    print()
    print("By backend (trajectories only):")
    for be, n in by_backend.most_common():
        print(f"  {be:30s} {n}")
    print()
    print("By model_used (top 10):")
    for m, n in by_model.most_common(10):
        print(f"  {m:55s} {n}")
    print()
    print("Score distribution (0..100 scale; cumulative buckets):")
    print(f"  =100               {score_buckets['=100']:>5d}    unique tasks: {len(tasks_in_bucket['=100'])}")
    print(f"  >=80               {score_buckets['>=80']:>5d}    unique tasks: {len(tasks_in_bucket['>=80'])}")
    print(f"  >=50               {score_buckets['>=50']:>5d}    unique tasks: {len(tasks_in_bucket['>=50'])}")
    print(f"  >0                 {score_buckets['>0']:>5d}    unique tasks: {len(tasks_in_bucket['>0'])}")
    print(f"  =0                 {score_buckets['=0']:>5d}    unique tasks: {len(tasks_in_bucket['=0'])}")
    print(f"  missing            {score_buckets['missing']:>5d}")
    print()
    print(f"TRAINABLE (score >= {args.score_threshold}): {trainable}")
    print(f"Unique task names with at least one trajectory: {len(by_task_traj)}")
    if by_task_traj:
        max_traj = max(len(v) for v in by_task_traj.values())
        avg_traj = sum(len(v) for v in by_task_traj.values()) / len(by_task_traj)
        print(f"Trajectories per task: avg={avg_traj:.2f}  max={max_traj}")
    print()

    # Screenshot integrity over a random sample
    rng = random.Random(args.seed)
    sample = rng.sample(screenshot_health, min(args.sample_size, len(screenshot_health)))
    sh_with = sum(1 for s in sample if s["present"])
    sh_without = sum(1 for s in sample if not s["present"])
    png_total = sum(s["count_png"] for s in sample)
    jpg_total = sum(s["count_jpg"] for s in sample)
    webp_total = sum(s["count_webp"] for s in sample)
    print(f"Screenshot integrity sample (n={len(sample)} trajectories):")
    print(f"  with screenshots dir:    {sh_with}")
    print(f"  without screenshots dir: {sh_without}")
    print(f"  total .png in sample:    {png_total}")
    print(f"  total .jpg in sample:    {jpg_total}")
    print(f"  total .webp in sample:   {webp_total}")
    print()

    if categories:
        per_cat_traj_total: Counter[str] = Counter()
        per_cat_traj_passing: Counter[str] = Counter()
        per_cat_unique_tasks: dict[str, set[str]] = defaultdict(set)
        per_cat_unique_tasks_passing: dict[str, set[str]] = defaultdict(set)
        uncategorized = 0
        for row in trajectory_rows:
            tn = row.get("task_name", "")
            cat = categories.get(tn)
            if cat is None:
                uncategorized += 1
                continue
            per_cat_traj_total[cat] += 1
            per_cat_unique_tasks[cat].add(tn)
            if (row.get("score") or 0) >= args.score_threshold:
                per_cat_traj_passing[cat] += 1
                per_cat_unique_tasks_passing[cat].add(tn)
        print("Per-category trajectory coverage:")
        print(f"  {'category':12s} {'total':>8s} {'passing':>8s} {'tasks':>8s} {'tasks_p':>8s}")
        for cat in sorted(set(categories.values())):
            print(
                f"  {cat:12s} {per_cat_traj_total.get(cat, 0):>8d} "
                f"{per_cat_traj_passing.get(cat, 0):>8d} "
                f"{len(per_cat_unique_tasks.get(cat, set())):>8d} "
                f"{len(per_cat_unique_tasks_passing.get(cat, set())):>8d}"
            )
        if uncategorized:
            print(f"  uncategorized trajectories: {uncategorized}")
        print()
    else:
        print("(no categories.yaml yet — run data/categorize_tasks.py to enable per-category breakdown)")
        print()

    print("=" * 72)
    if trainable < args.min_trainable:
        print(
            f"GATE WARNING: trainable count {trainable} is below threshold {args.min_trainable}."
        )
        return 2
    print(f"GATE PASSED: trainable count {trainable} >= threshold {args.min_trainable}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
