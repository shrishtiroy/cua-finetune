"""Rebuild ``_summary.jsonl`` from per-task ``trajectory.json`` files.

The browser-eval pipeline writes one ``trajectory.json`` per task per attempt
under ``results/<backend>/<adapter>/<task>/<timestamp>/trajectory.json``, plus a
roll-up ``results/<backend>/<adapter>/_summary.jsonl``. The roll-up is
*overwritten* on every run, which means a partial backfill or a re-run of a
subset of tasks loses the rest of the summary.

This script reconstructs the summary by walking the per-task directories and
picking one trajectory per task (latest by default; pass ``--best`` for
pass@k-style "best of all attempts").

Usage::

    python eval/rebuild_summary.py qwen_vl_cua baseline
    python eval/rebuild_summary.py qwen_vl_cua cua --best
    python eval/rebuild_summary.py gemma_vl_cua baseline --results-root /custom/path
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_ROOT = REPO_ROOT / "results"


def _extract_score(data: dict[str, Any]) -> float | None:
    """Defensively pull a numeric score out of a trajectory record.

    Different revisions of conduit / our backends have nested the score in
    different places. Try them all in priority order.
    """
    for key in ("score", "final_score"):
        v = data.get(key)
        if isinstance(v, (int, float)):
            return float(v)

    for parent_key in ("grading", "rubric", "result", "summary"):
        parent = data.get(parent_key)
        if isinstance(parent, dict):
            for child_key in ("score", "final_score", "value"):
                v = parent.get(child_key)
                if isinstance(v, (int, float)):
                    return float(v)

    return None


def _extract_category(data: dict[str, Any]) -> str | None:
    for key in ("category", "task_category"):
        v = data.get(key)
        if isinstance(v, str):
            return v
    task = data.get("task") or data.get("task_spec")
    if isinstance(task, dict):
        for key in ("category", "task_category"):
            v = task.get(key)
            if isinstance(v, str):
                return v
    return None


def rebuild(
    backend: str,
    adapter: str,
    *,
    results_root: Path = DEFAULT_RESULTS_ROOT,
    use_best: bool = False,
    dry_run: bool = False,
) -> int:
    base = results_root / backend / adapter
    if not base.exists():
        print(f"ERROR: {base} does not exist", file=sys.stderr)
        return 1

    rows: list[dict[str, Any]] = []
    skipped: list[str] = []

    for task_dir in sorted(base.iterdir()):
        if not task_dir.is_dir() or task_dir.name.startswith("_"):
            continue

        run_dirs = sorted(d for d in task_dir.iterdir() if d.is_dir())
        if not run_dirs:
            skipped.append(f"{task_dir.name}: no run subdirectories")
            continue

        # Collect (run_dir, score) for every run that has a parseable trajectory.
        candidates: list[tuple[Path, float, dict[str, Any]]] = []
        for run_dir in run_dirs:
            traj = run_dir / "trajectory.json"
            if not traj.exists():
                continue
            try:
                data = json.loads(traj.read_text())
            except Exception as exc:
                skipped.append(f"{task_dir.name}/{run_dir.name}: parse error: {exc}")
                continue
            score = _extract_score(data)
            if score is None:
                skipped.append(f"{task_dir.name}/{run_dir.name}: no score field")
                continue
            candidates.append((run_dir, score, data))

        if not candidates:
            skipped.append(f"{task_dir.name}: no usable trajectory.json across {len(run_dirs)} runs")
            continue

        if use_best:
            chosen = max(candidates, key=lambda c: c[1])
            selection = "best"
        else:
            # Latest by timestamp directory name (timestamps sort lexicographically).
            chosen = candidates[-1]
            selection = "latest"

        run_dir, score, data = chosen
        rows.append({
            "task": task_dir.name,
            "score": score,
            "category": _extract_category(data),
            "trajectory_path": str(run_dir / "trajectory.json"),
            "run_timestamp": run_dir.name,
            "selection": selection,
            "n_attempts_on_disk": len(candidates),
        })

    summary_path = base / "_summary.jsonl"
    if not dry_run:
        with summary_path.open("w") as h:
            for row in rows:
                h.write(json.dumps(row) + "\n")
        print(f"Wrote {summary_path} ({len(rows)} rows)")
    else:
        print(f"DRY RUN — would write {summary_path} ({len(rows)} rows)")

    if rows:
        scores = [r["score"] for r in rows]
        full = sum(1 for s in scores if s >= 0.999)
        zero = sum(1 for s in scores if s <= 0.001)
        partial = len(scores) - full - zero
        print()
        print(f"  mean    = {statistics.mean(scores):.3f}")
        print(f"  median  = {statistics.median(scores):.3f}")
        print(f"  n       = {len(scores)}")
        print(f"  full    = {full}")
        print(f"  partial = {partial}")
        print(f"  zero    = {zero}")

    if skipped:
        print()
        print(f"SKIPPED {len(skipped)} entries:")
        for line in skipped[:20]:
            print(f"  - {line}")
        if len(skipped) > 20:
            print(f"  ... and {len(skipped) - 20} more")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Rebuild _summary.jsonl from per-task trajectory.json files",
    )
    parser.add_argument("backend", help="e.g. qwen_vl_cua, gemma_vl_cua, kimi_vl_cua")
    parser.add_argument("adapter", help="e.g. baseline, cua")
    parser.add_argument(
        "--results-root",
        type=Path,
        default=DEFAULT_RESULTS_ROOT,
        help=f"results/ directory (default: {DEFAULT_RESULTS_ROOT})",
    )
    parser.add_argument(
        "--best",
        action="store_true",
        help="Pick the best-scoring trajectory per task (pass@k semantics) "
             "instead of the latest run",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be written without overwriting _summary.jsonl",
    )
    args = parser.parse_args()
    return rebuild(
        args.backend,
        args.adapter,
        results_root=args.results_root,
        use_best=args.best,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    sys.exit(main())
