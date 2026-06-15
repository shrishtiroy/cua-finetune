"""Side-by-side comparison of original vs strict rubric scores.

Reads ``results/<backend>/<adapter>/_summary.jsonl`` (original rubric) and
``results/<backend>/<adapter>/_summary_strict.jsonl`` (strict rubric, produced
by ``eval/regrade_strict.py``) and prints per-task deltas + aggregate stats.

Usage::

    python eval/strict_compare.py qwen_vl_cua baseline
    python eval/strict_compare.py qwen_vl_cua cua
    python eval/strict_compare.py qwen_vl_cua baseline --only-changed
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_ROOT = REPO_ROOT / "results"


def _load_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return out
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


def _truncate(text: str | None, width: int) -> str:
    if not text:
        return ""
    text = text.replace("\n", " ").replace("\r", " ")
    if len(text) <= width:
        return text
    return text[: width - 1] + "\u2026"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare original-rubric vs strict-rubric scores per task.",
    )
    parser.add_argument("backend", help="e.g. qwen_vl_cua")
    parser.add_argument("adapter", help="e.g. baseline, cua")
    parser.add_argument(
        "--results-root",
        type=Path,
        default=DEFAULT_RESULTS_ROOT,
    )
    parser.add_argument(
        "--only-changed",
        action="store_true",
        help="Only print tasks where strict != orig",
    )
    parser.add_argument(
        "--answer-width",
        type=int,
        default=60,
        help="Max characters of the declared_answer to show inline (default 60)",
    )
    args = parser.parse_args()

    base = args.results_root / args.backend / args.adapter
    orig_path = base / "_summary.jsonl"
    strict_path = base / "_summary_strict.jsonl"

    if not orig_path.exists():
        print(f"ERROR: missing {orig_path}", file=sys.stderr)
        return 1
    if not strict_path.exists():
        print(
            f"ERROR: missing {strict_path}\n"
            f"Run: python eval/regrade_strict.py {args.backend} {args.adapter}",
            file=sys.stderr,
        )
        return 1

    orig = _load_jsonl(orig_path)
    strict = _load_jsonl(strict_path)
    tasks = sorted(set(orig) | set(strict))

    print(f"{'task':<55} {'orig':>5}  {'strict':>6}  {'delta':>6}  notes")
    print("-" * 120)

    deltas: list[float] = []
    orig_scores: list[float] = []
    strict_scores: list[float] = []
    failure_modes: Counter[str] = Counter()
    n_no_data = 0

    for task in tasks:
        o = orig.get(task) or {}
        s = strict.get(task) or {}
        o_score = o.get("score")
        s_score = s.get("score_strict")
        mode = s.get("failure_mode", "missing")
        failure_modes[mode] += 1

        if not isinstance(o_score, (int, float)) or not isinstance(s_score, (int, float)):
            n_no_data += 1
            if args.only_changed:
                continue
            o_str = f"{o_score:.2f}" if isinstance(o_score, (int, float)) else "  ? "
            s_str = f"{s_score:.2f}" if isinstance(s_score, (int, float)) else "  ?  "
            print(f"{task[:55]:<55} {o_str:>5}  {s_str:>6}  {'   ?':>6}  ({mode})")
            continue

        delta = s_score - o_score
        deltas.append(delta)
        orig_scores.append(o_score)
        strict_scores.append(s_score)

        if args.only_changed and abs(delta) < 1e-6:
            continue

        ans = s.get("declared_answer")
        ans_show = _truncate(ans, args.answer_width)
        ans_part = f'declared_answer: "{ans_show}"' if ans else "declared_answer: null"
        note = f"({ans_part}, {mode})"
        print(
            f"{task[:55]:<55} {o_score:>5.2f}  {s_score:>6.2f}  {delta:>+6.2f}  {note}"
        )

    print()
    if orig_scores and strict_scores:
        o_mean = statistics.mean(orig_scores)
        s_mean = statistics.mean(strict_scores)
        print(f"aggregate over {len(orig_scores)} tasks:")
        print(f"  orig mean   = {o_mean:.3f}")
        print(f"  strict mean = {s_mean:.3f}")
        print(f"  delta       = {s_mean - o_mean:+.3f}")
        if o_mean > 0:
            print(f"  relative    = {(s_mean / o_mean - 1) * 100:+.0f}%")
        n_up = sum(1 for d in deltas if d > 1e-6)
        n_dn = sum(1 for d in deltas if d < -1e-6)
        n_eq = len(deltas) - n_up - n_dn
        print(f"  per-task    : {n_up} up, {n_dn} down, {n_eq} unchanged")

    print()
    print("failure modes (strict):")
    for mode, n in failure_modes.most_common():
        print(f"  {mode:<24} {n:>4}")
    if n_no_data:
        print(f"  (tasks without scores on both sides: {n_no_data})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
