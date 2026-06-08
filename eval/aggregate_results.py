"""Cross-model results aggregator.

Reads ``results/<backend>/<adapter>/_summary.jsonl`` for every (backend, adapter)
pair present on disk and produces:

  1. A pass@k table per (model × category) printed to stdout.
  2. A flat JSON dump at ``results/_aggregate.json`` with all rows.
  3. A CSV at ``results/_aggregate.csv`` suitable for the blog table.

pass@k semantics:
  For each task we have ``k`` independent attempts (rows with attempt=0..k-1).
  A task is considered "passed" if any attempt's score ≥ ``--pass-threshold``
  (default 1.0). pass@k = (# tasks passed) / (# tasks attempted).

Usage::

    python eval/aggregate_results.py                       # default: pass@1, score>=1.0
    python eval/aggregate_results.py --pass-threshold 0.5  # partial credit counts
    python eval/aggregate_results.py --adapter baseline    # only the baseline runs
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

logger = logging.getLogger("aggregate_results")

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = REPO_ROOT / "results"

KNOWN_BACKENDS = ["qwen_vl_cua", "kimi_vl_cua", "deepseek_vl_cua", "llama_vision_cua"]
CATEGORY_ORDER = ["C1_ui_nav", "C2_structured", "C3_docs", "C4_shopping", "C5_government", "C99_other"]


def _read_summaries(results_dir: Path, adapter_filter: str | None) -> dict[tuple[str, str], list[dict[str, Any]]]:
    """Return ``{(backend, adapter): [row, ...]}`` from every ``_summary.jsonl`` found."""
    out: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for backend_dir in sorted(results_dir.glob("*")):
        if not backend_dir.is_dir() or backend_dir.name.startswith("_"):
            continue
        backend = backend_dir.name
        for adapter_dir in sorted(backend_dir.glob("*")):
            if not adapter_dir.is_dir():
                continue
            adapter = adapter_dir.name
            if adapter_filter and adapter != adapter_filter:
                continue
            summary_path = adapter_dir / "_summary.jsonl"
            if not summary_path.exists():
                continue
            rows = []
            with summary_path.open("r", encoding="utf-8") as h:
                for line in h:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        logger.warning("could not parse line in %s", summary_path)
            out[(backend, adapter)] = rows
    return out


def _pass_at_k(rows: list[dict[str, Any]], threshold: float) -> tuple[float, dict[str, float]]:
    """Compute pass@k across all attempts in *rows*. Returns (overall, per_category)."""
    per_task_best: dict[str, float] = defaultdict(float)
    per_task_cat: dict[str, str] = {}
    for r in rows:
        task = r.get("task", "?")
        cat = r.get("category", "?")
        score = float(r.get("score", 0) or 0)
        per_task_best[task] = max(per_task_best[task], score)
        per_task_cat[task] = cat

    if not per_task_best:
        return 0.0, {}

    n_total = len(per_task_best)
    n_pass = sum(1 for s in per_task_best.values() if s >= threshold)
    overall = n_pass / n_total

    by_cat_total: dict[str, int] = defaultdict(int)
    by_cat_pass: dict[str, int] = defaultdict(int)
    for task, score in per_task_best.items():
        cat = per_task_cat[task]
        by_cat_total[cat] += 1
        if score >= threshold:
            by_cat_pass[cat] += 1
    per_cat = {c: by_cat_pass[c] / by_cat_total[c] if by_cat_total[c] else 0.0
               for c in by_cat_total}
    return overall, per_cat


def _format_row(name: str, overall: float, per_cat: dict[str, float], cats: list[str]) -> str:
    cells = [f"{name:<28s}", f"{overall*100:6.1f}%"]
    for c in cats:
        if c in per_cat:
            cells.append(f"{per_cat[c]*100:5.1f}%")
        else:
            cells.append("   —  ")
    return "  ".join(cells)


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate browser-eval results across models")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--pass-threshold", type=float, default=1.0,
                        help="A task is 'passed' if best score across attempts ≥ this. Default 1.0.")
    parser.add_argument("--adapter", default=None,
                        help="Only aggregate this adapter (e.g. 'baseline' or 'cua'). Default all.")
    parser.add_argument("--csv", type=Path, default=None,
                        help="Where to write CSV. Default: <results-dir>/_aggregate.csv")
    parser.add_argument("--json", dest="json_out", type=Path, default=None,
                        help="Where to write JSON. Default: <results-dir>/_aggregate.json")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        level=logging.DEBUG if args.verbose else logging.INFO,
    )

    if not args.results_dir.exists():
        print(f"ERROR: results-dir not found: {args.results_dir}", file=sys.stderr)
        return 1

    summaries = _read_summaries(args.results_dir, args.adapter)
    if not summaries:
        print(f"ERROR: no _summary.jsonl files found under {args.results_dir}", file=sys.stderr)
        return 1

    seen_cats: set[str] = set()
    aggregate: dict[str, Any] = {
        "pass_threshold": args.pass_threshold,
        "results": [],
    }
    for (backend, adapter), rows in summaries.items():
        for r in rows:
            if r.get("category"):
                seen_cats.add(r["category"])

    cats_in_order = [c for c in CATEGORY_ORDER if c in seen_cats]
    cats_extras = sorted(c for c in seen_cats if c not in CATEGORY_ORDER)
    cats_all = cats_in_order + cats_extras

    header = f"{'Model / Adapter':<28s}  {'Overall':>7s}  " + "  ".join(f"{c:>6s}" for c in cats_all)
    sep = "-" * len(header)
    print()
    print(f"pass@k @ threshold ≥ {args.pass_threshold}")
    print(header)
    print(sep)

    for (backend, adapter), rows in sorted(summaries.items()):
        overall, per_cat = _pass_at_k(rows, args.pass_threshold)
        label = f"{backend}/{adapter}"
        print(_format_row(label, overall, per_cat, cats_all))

        n_tasks = len({r.get("task") for r in rows if r.get("task")})
        n_attempts = len(rows)
        parse_fail = sum(1 for r in rows if isinstance(r.get("error"), str)
                         and "parse" in str(r["error"]).lower())
        errored = sum(1 for r in rows if r.get("error"))
        aggregate["results"].append({
            "backend": backend,
            "adapter": adapter,
            "overall_pass_at_k": overall,
            "per_category": per_cat,
            "n_tasks": n_tasks,
            "n_attempts": n_attempts,
            "n_errored_attempts": errored,
            "n_parse_failures": parse_fail,
        })

    print(sep)
    print("(— = no tasks for that category in this run)")

    csv_path = args.csv or (args.results_dir / "_aggregate.csv")
    with csv_path.open("w", newline="", encoding="utf-8") as h:
        writer = csv.writer(h)
        writer.writerow(["backend", "adapter", "overall_pass_at_k"] + cats_all
                        + ["n_tasks", "n_attempts", "n_errored", "n_parse_failures"])
        for row in aggregate["results"]:
            writer.writerow([
                row["backend"],
                row["adapter"],
                f"{row['overall_pass_at_k']:.4f}",
                *[f"{row['per_category'].get(c, 0.0):.4f}" if c in row["per_category"] else ""
                  for c in cats_all],
                row["n_tasks"],
                row["n_attempts"],
                row["n_errored_attempts"],
                row["n_parse_failures"],
            ])

    json_path = args.json_out or (args.results_dir / "_aggregate.json")
    with json_path.open("w", encoding="utf-8") as h:
        json.dump(aggregate, h, indent=2)

    print()
    print(f"Wrote CSV : {csv_path}")
    print(f"Wrote JSON: {json_path}")

    print()
    print("Per-task failure modes (only attempts that errored or scored 0):")
    for (backend, adapter), rows in sorted(summaries.items()):
        bad = [r for r in rows if float(r.get("score", 0) or 0) == 0]
        if not bad:
            continue
        print(f"  {backend}/{adapter}: {len(bad)} zero-score attempts")
        per_task_zero: dict[str, int] = defaultdict(int)
        for r in bad:
            per_task_zero[r.get("task", "?")] += 1
        for t, n in sorted(per_task_zero.items(), key=lambda kv: -kv[1])[:8]:
            print(f"    {n}× {t}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
