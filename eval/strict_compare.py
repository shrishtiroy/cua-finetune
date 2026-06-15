"""Side-by-side comparison of original vs strict (and optionally reasoning-trace)
rubric scores.

Reads:

- ``results/<backend>/<adapter>/_summary.jsonl``           (original, C0)
- ``results/<backend>/<adapter>/_summary_strict.jsonl``    (strict declared
  answer, C1; produced by ``eval/regrade_strict.py``)
- ``results/<backend>/<adapter>/_summary_reasoning.jsonl`` (reasoning trace,
  C2; produced by ``eval/regrade_reasoning.py``) — only when ``--reasoning``
  is passed.

Default behavior (2-way): prints per-task ``orig vs strict`` deltas + aggregate.
With ``--reasoning``, switches to a 3-way layout that also shows the reasoning
score and a breakdown of how often the reasoning grader credited tasks the
original rubric missed.

Usage::

    python eval/strict_compare.py qwen_vl_cua baseline
    python eval/strict_compare.py qwen_vl_cua cua
    python eval/strict_compare.py qwen_vl_cua baseline --only-changed
    python eval/strict_compare.py qwen_vl_cua baseline --reasoning
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


def _two_way(
    orig: dict[str, dict[str, Any]],
    strict: dict[str, dict[str, Any]],
    *,
    only_changed: bool,
    answer_width: int,
) -> int:
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
            if only_changed:
                continue
            o_str = f"{o_score:.2f}" if isinstance(o_score, (int, float)) else "  ? "
            s_str = f"{s_score:.2f}" if isinstance(s_score, (int, float)) else "  ?  "
            print(f"{task[:55]:<55} {o_str:>5}  {s_str:>6}  {'   ?':>6}  ({mode})")
            continue

        delta = s_score - o_score
        deltas.append(delta)
        orig_scores.append(o_score)
        strict_scores.append(s_score)

        if only_changed and abs(delta) < 1e-6:
            continue

        ans = s.get("declared_answer")
        ans_show = _truncate(ans, answer_width)
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


def _reasoning_note(reasoning_row: dict[str, Any], width: int) -> str:
    """Inline note for a reasoning row.

    Try to surface the explanation of the first MET criterion so the user can
    see *why* the reasoning grader credited the task. Falls back to the
    failure_mode tag.
    """
    reports = reasoning_row.get("rubric_report") or []
    if isinstance(reports, list):
        for r in reports:
            if isinstance(r, dict) and r.get("verdict") == "MET":
                reason = r.get("reason")
                if isinstance(reason, str) and reason.strip():
                    return f'reasoning: "{_truncate(reason, width)}"'
    mode = reasoning_row.get("failure_mode") or "missing"
    return f"({mode})"


def _three_way(
    orig: dict[str, dict[str, Any]],
    strict: dict[str, dict[str, Any]],
    reasoning: dict[str, dict[str, Any]],
    *,
    only_changed: bool,
    answer_width: int,
) -> int:
    tasks = sorted(set(orig) | set(strict) | set(reasoning))

    print(
        f"{'task':<45} {'orig':>5}  {'strict':>6}  {'reasoning':>9}  notes"
    )
    print("-" * 140)

    orig_scores: list[float] = []
    strict_scores: list[float] = []
    reasoning_scores: list[float] = []
    failure_modes_reasoning: Counter[str] = Counter()
    n_no_data = 0

    r_vs_o_up = 0
    r_vs_o_eq = 0
    r_vs_o_dn = 0

    for task in tasks:
        o = orig.get(task) or {}
        s = strict.get(task) or {}
        r = reasoning.get(task) or {}
        o_score = o.get("score")
        s_score = s.get("score_strict")
        r_score = r.get("score_reasoning")
        r_mode = r.get("failure_mode", "missing")
        failure_modes_reasoning[r_mode] += 1

        all_numeric = (
            isinstance(o_score, (int, float))
            and isinstance(s_score, (int, float))
            and isinstance(r_score, (int, float))
        )
        if not all_numeric:
            n_no_data += 1
            if only_changed:
                continue
            o_str = f"{o_score:.2f}" if isinstance(o_score, (int, float)) else "  ?  "
            s_str = f"{s_score:.2f}" if isinstance(s_score, (int, float)) else "  ?  "
            r_str = f"{r_score:.2f}" if isinstance(r_score, (int, float)) else "  ?  "
            print(
                f"{task[:45]:<45} {o_str:>5}  {s_str:>6}  {r_str:>9}  ({r_mode})"
            )
            continue

        orig_scores.append(o_score)
        strict_scores.append(s_score)
        reasoning_scores.append(r_score)

        if r_score > o_score + 1e-6:
            r_vs_o_up += 1
        elif r_score < o_score - 1e-6:
            r_vs_o_dn += 1
        else:
            r_vs_o_eq += 1

        if only_changed and abs(r_score - o_score) < 1e-6 and abs(s_score - o_score) < 1e-6:
            continue

        note = _reasoning_note(r, answer_width)
        print(
            f"{task[:45]:<45} {o_score:>5.2f}  {s_score:>6.2f}  {r_score:>9.2f}  {note}"
        )

    print()
    if orig_scores and strict_scores and reasoning_scores:
        o_mean = statistics.mean(orig_scores)
        s_mean = statistics.mean(strict_scores)
        r_mean = statistics.mean(reasoning_scores)
        print(f"aggregate over {len(orig_scores)} tasks:")
        print(f"  orig mean      = {o_mean:.3f}")
        print(f"  strict mean    = {s_mean:.3f}")
        print(f"  reasoning mean = {r_mean:.3f}")
        print()
        print(f"  per-task breakdown vs original:")
        print(
            f"    reasoning > orig:  {r_vs_o_up:>3} tasks "
            f"(model knew answer but rubric missed it)"
        )
        print(
            f"    reasoning = orig:  {r_vs_o_eq:>3} tasks "
            f"(rubric was right)"
        )
        print(
            f"    reasoning < orig:  {r_vs_o_dn:>3} tasks "
            f"(rare; would mean rubric over-credited)"
        )

    print()
    print("failure modes (reasoning):")
    for mode, n in failure_modes_reasoning.most_common():
        print(f"  {mode:<24} {n:>4}")
    if n_no_data:
        print(f"  (tasks without scores on all sides: {n_no_data})")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare original-rubric vs strict-rubric scores per task. "
                    "With --reasoning, also compares the reasoning-trace grader.",
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
        help="Only print tasks where at least one mode differs from the others",
    )
    parser.add_argument(
        "--answer-width",
        type=int,
        default=60,
        help="Max characters of inline notes (declared_answer or reasoning "
             "snippet) to show (default 60)",
    )
    parser.add_argument(
        "--reasoning",
        action="store_true",
        help="Include the reasoning-trace grader (_summary_reasoning.jsonl) as "
             "a third column. Requires `python eval/regrade_reasoning.py ...` "
             "to have been run first.",
    )
    args = parser.parse_args()

    base = args.results_root / args.backend / args.adapter
    orig_path = base / "_summary.jsonl"
    strict_path = base / "_summary_strict.jsonl"
    reasoning_path = base / "_summary_reasoning.jsonl"

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

    if args.reasoning:
        if not reasoning_path.exists():
            print(
                f"ERROR: missing {reasoning_path}\n"
                f"Run: python eval/regrade_reasoning.py {args.backend} {args.adapter}",
                file=sys.stderr,
            )
            return 1
        reasoning = _load_jsonl(reasoning_path)
        return _three_way(
            orig,
            strict,
            reasoning,
            only_changed=args.only_changed,
            answer_width=args.answer_width,
        )

    return _two_way(
        orig,
        strict,
        only_changed=args.only_changed,
        answer_width=args.answer_width,
    )


if __name__ == "__main__":
    sys.exit(main())
