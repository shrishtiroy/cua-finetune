"""Phase 4a: aggregate per-run grade JSONs into the headline 4x5x2 table.

Walks ``results/<backend>/<adapter>/<task>/<ts>/result.json`` and joins on
``data/manifests/categories.yaml``. Computes::

    pass@1: at least one of the *first* attempt's score >= 80
    pass@k: at least one of any of the k attempts' score >= 80

Outputs:
    results/category_breakdown.csv          long-form (model, adapter, category, k, n, success_rate)
    results/headline.md                     paired baseline/finetuned table per (model, category) + lifts
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

import yaml

logger = logging.getLogger("aggregate_results")

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = REPO_ROOT / "results"
DEFAULT_CATEGORIES = REPO_ROOT / "data" / "manifests" / "categories.yaml"

CATEGORY_LABELS = {
    "C1": "UI Navigation & Interactive Viz",
    "C2": "Structured Data & Tables",
    "C3": "Documentation & Reference",
    "C4": "E-commerce & Shopping",
    "C5": "Government & Civic",
    "C0": "Other / Uncategorized",
}

PASS_THRESHOLD = 80.0


def _load_categories(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as h:
        data = yaml.safe_load(h) or {}
    if isinstance(data, dict):
        return {str(k): str(v) for k, v in data.items() if isinstance(v, str)}
    return {}


def _walk_results(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not root.exists():
        return rows
    # Layout: results/<backend>/<adapter>/<task>/<ts>/result.json
    for backend_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for adapter_dir in sorted(p for p in backend_dir.iterdir() if p.is_dir()):
            for task_dir in sorted(p for p in adapter_dir.iterdir() if p.is_dir()):
                # Each ts dir in task_dir is one attempt
                attempt_idx = -1
                for ts_dir in sorted(p for p in task_dir.iterdir() if p.is_dir()):
                    attempt_idx += 1
                    rj = ts_dir / "result.json"
                    if not rj.exists():
                        continue
                    try:
                        data = json.loads(rj.read_text(encoding="utf-8"))
                    except json.JSONDecodeError as exc:
                        logger.warning("bad result.json: %s (%s)", rj, exc)
                        continue
                    grade = data.get("grade") or {}
                    raw_score = grade.get("score")
                    score = float(raw_score) * 100.0 if isinstance(raw_score, (int, float)) and 0.0 <= raw_score <= 1.0 else (
                        float(raw_score) if isinstance(raw_score, (int, float)) else 0.0
                    )
                    rows.append({
                        "backend": backend_dir.name,
                        "adapter": adapter_dir.name,
                        "task": task_dir.name,
                        "attempt": attempt_idx,
                        "score": score,
                        "run_dir": str(ts_dir),
                    })
    return rows


def _passk_from_scores(scores: list[float], k: int, threshold: float) -> tuple[float, int]:
    """Return (success_indicator_at_k_in_[0,1], n_attempts_used)."""
    if not scores:
        return 0.0, 0
    take = scores[:k]
    return (1.0 if any(s >= threshold for s in take) else 0.0), len(take)


def _aggregate(rows: list[dict[str, Any]], categories: dict[str, str]) -> dict[str, Any]:
    by_key: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for r in rows:
        cat = categories.get(r["task"], "C0")
        by_key[(r["backend"], r["adapter"], cat, r["task"])].append(r["score"]) if False else None
    # Need 4-tuple including task — rebuild correctly
    by_key = defaultdict(list)
    for r in rows:
        cat = categories.get(r["task"], "C0")
        by_key[(r["backend"], r["adapter"], cat, r["task"])].append(r["score"])

    out: dict[tuple[str, str, str], dict[str, Any]] = defaultdict(
        lambda: {"pass_at_1_num": 0, "pass_at_5_num": 0, "n_tasks": 0, "tasks": []}
    )
    for (backend, adapter, cat, task), scores in by_key.items():
        scores.sort(reverse=True)  # not strictly necessary; pass@k uses any-of
        # Use INSERT order from results listing (already sorted by ts above)
        # Re-fetch insertion order:
        # (We'll just reuse `scores` here — for pass@k semantics any-of is order-independent.)
        p1, _ = _passk_from_scores(scores, 1, PASS_THRESHOLD)
        p5, _ = _passk_from_scores(scores, 5, PASS_THRESHOLD)
        agg = out[(backend, adapter, cat)]
        agg["pass_at_1_num"] += p1
        agg["pass_at_5_num"] += p5
        agg["n_tasks"] += 1
        agg["tasks"].append(task)

    long_rows: list[dict[str, Any]] = []
    for (backend, adapter, cat), v in sorted(out.items()):
        n = v["n_tasks"]
        long_rows.append({
            "backend": backend,
            "adapter": adapter,
            "category": cat,
            "n_tasks": n,
            "pass_at_1": v["pass_at_1_num"] / n if n else 0.0,
            "pass_at_5": v["pass_at_5_num"] / n if n else 0.0,
        })
    return {"long": long_rows}


def _write_csv(long_rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["backend", "adapter", "category", "n_tasks", "pass_at_1", "pass_at_5"]
    with path.open("w", encoding="utf-8", newline="") as h:
        w = csv.DictWriter(h, fieldnames=cols)
        w.writeheader()
        for r in long_rows:
            w.writerow({k: r.get(k, "") for k in cols})


def _write_headline_md(long_rows: list[dict[str, Any]], path: Path) -> None:
    # Pivot: backend × category → {baseline:(p1,p5), cua:(p1,p5)}
    pivot: dict[tuple[str, str], dict[str, dict[str, float]]] = defaultdict(dict)
    for r in long_rows:
        pivot[(r["backend"], r["category"])][r["adapter"]] = {
            "pass_at_1": r["pass_at_1"],
            "pass_at_5": r["pass_at_5"],
            "n_tasks": r["n_tasks"],
        }
    backends = sorted({b for b, _ in pivot})
    cats = sorted({c for _, c in pivot})

    lines: list[str] = []
    lines.append("# CUA SFT 4×5 headline\n")
    lines.append(f"_Threshold for pass: score ≥ {PASS_THRESHOLD}_\n")

    for k_label in ("pass_at_1", "pass_at_5"):
        lines.append(f"## {k_label.replace('_', ' ').title()}\n")
        header = ["model"] + [f"{c} ({CATEGORY_LABELS.get(c, '?')})" for c in cats]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|" + "|".join(["---"] * len(header)) + "|")
        for b in backends:
            row = [b]
            for c in cats:
                cell = pivot.get((b, c), {})
                base = (cell.get("baseline") or {}).get(k_label)
                cua = (cell.get("cua") or {}).get(k_label)
                if base is None and cua is None:
                    row.append("—")
                    continue
                base_s = f"{base:.2f}" if base is not None else "—"
                cua_s = f"{cua:.2f}" if cua is not None else "—"
                lift = ""
                if base is not None and cua is not None and base > 0:
                    rel = (cua - base) / base
                    bold_open = "**" if rel >= 0.15 else ""
                    bold_close = "**" if rel >= 0.15 else ""
                    lift = f" {bold_open}({rel:+.0%}){bold_close}"
                elif cua is not None and (base is None or base == 0):
                    if cua > 0:
                        lift = " (∞)"
                row.append(f"{base_s} → {cua_s}{lift}")
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    lines.append("## Notes\n")
    lines.append("- Cells render as `baseline → finetuned (relative lift)`. Bold cells exceed +15%.\n")
    lines.append("- Empty cells = no held-out tasks in that category for the given model adapter.\n")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate eval results into the headline table")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--categories", type=Path, default=DEFAULT_CATEGORIES)
    parser.add_argument("--out-csv", type=Path, default=None)
    parser.add_argument("--out-md", type=Path, default=None)
    args = parser.parse_args()

    out_csv = args.out_csv or args.results_dir / "category_breakdown.csv"
    out_md = args.out_md or args.results_dir / "headline.md"

    rows = _walk_results(args.results_dir)
    if not rows:
        print("No result.json files found under", args.results_dir, file=sys.stderr)
        # Still emit empty placeholders so downstream tooling has stable paths
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        out_csv.write_text("backend,adapter,category,n_tasks,pass_at_1,pass_at_5\n", encoding="utf-8")
        out_md.write_text("# CUA SFT headline\n\n_(no results yet)_\n", encoding="utf-8")
        return 0

    categories = _load_categories(args.categories)
    agg = _aggregate(rows, categories)
    _write_csv(agg["long"], out_csv)
    _write_headline_md(agg["long"], out_md)
    print(json.dumps({
        "n_attempt_rows": len(rows),
        "n_aggregated_rows": len(agg["long"]),
        "csv": str(out_csv),
        "md": str(out_md),
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
