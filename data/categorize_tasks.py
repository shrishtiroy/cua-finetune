"""Phase 1a: tag every task with one of five categories (plus a C99_other fallback).

Categories::

    C1_ui_nav     — UI navigation & interactive viz
    C2_structured — structured data / tables / scraping
    C3_docs       — documentation & reference
    C4_shopping   — e-commerce
    C5_government — government & civic
    C99_other     — fallback

Matching rules are case-insensitive substring matches on ``task_external_id``.
The first matching keyword wins. Per-task overrides live in
``data/manifests/category_overrides.yaml`` (optional)::

    overrides:
      some-task: C1_ui_nav

Inputs:
    data/manifests/pulled_trajectories.jsonl       (from pull_trajectories.py)
    ../project-dojo/staging/**/*.yaml              (read-only)
    ../liveweb/tasks/*.yaml                        (read-only)
    ../Dillinger/tasks/**/*.yaml                   (read-only)
    data/manifests/category_overrides.yaml         (optional)

Output:
    data/manifests/categories.yaml                 {counts: {...}, tasks: {task: cat}}
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SIBLINGS = REPO_ROOT.parent
DEFAULT_MANIFEST = REPO_ROOT / "data" / "manifests" / "pulled_trajectories.jsonl"
DEFAULT_OVERRIDES = REPO_ROOT / "data" / "manifests" / "category_overrides.yaml"
DEFAULT_OUT = REPO_ROOT / "data" / "manifests" / "categories.yaml"


CATEGORY_KEYWORDS: list[tuple[str, list[str]]] = [
    ("C1_ui_nav", [
        "karthikeya", "happy_map", "happy-map", "happymap",
        "worldshapin", "peoplemovin",
        "histography", "thatsedu", "nasa", "rhythm-of-food", "rhythm_of_food",
        "ncase",
        "gapminder",
        "fleet-map", "fleet_map",
        "desmos",
        "canvas-diagram", "canvas_diagram",
        "histo-",
    ]),
    ("C2_structured", [
        "scrapethissite", "sts-", "sts_", "githut", "csrankings",
        "400freestyle", "lolpros", "reuters",
        "fed-reserve", "fed_reserve",
        "cross-ref-", "cross_ref_",
        "cmu-",
        "cambridge-",
        "charleston-",
        "edg-",
        "difference-between-", "difference_between_",
        "cua-bench/structured",
    ]),
    ("C3_docs", [
        "pydocs", "python_docs", "python-docs", "iana", "sonos",
    ]),
    ("C4_shopping", [
        "stripe", "allbirds", "shopify", "portillos",
        "carls_jr", "carls-jr", "carlsjr", "carls-",
        "eatwellatx", "eatwell",
        "doordash",
        "easy-steak", "easy_steak",
    ]),
    ("C5_government", [
        "uk-food", "uk_food", "holyrood", "city-of", "city_of",
        "pilot_government", "irs", "sec", "noaa",
        "gb-", "gb_",
        "gov-", "gov_",
        "fema",
        "medicare",
        "-county-", "_county_", "-county/", "county-permits",
        "planning-zoning", "planning_zoning",
        "permits-inspections", "permits_inspections",
        "workflow-requirements",
        "beudo",
    ]),
]


DROP_PATTERNS: list[str] = [
    "bert-fill-in-blank", "bert_fill_in_blank",
    "click-calibration", "click_calibration",
    "counter-app-",
    "filebrowser-",
    "flatnotes-",
    "form-validation-",
    "cua-verifier/",
    "computer-1/click-calibration",
    "computer-1/canvas-diagram-llm-judge",
]


DROP_CATEGORY = "C_drop_non_browser"
DEFAULT_CATEGORY = "C99_other"
ALL_CATEGORIES = [k for k, _ in CATEGORY_KEYWORDS] + [DEFAULT_CATEGORY, DROP_CATEGORY]


def categorize_one(name: str, overrides: dict[str, str]) -> str:
    if name in overrides:
        return overrides[name]
    low = name.lower()
    for pat in DROP_PATTERNS:
        if pat in low:
            return DROP_CATEGORY
    for cat, keywords in CATEGORY_KEYWORDS:
        for kw in keywords:
            if kw in low:
                return cat
    return DEFAULT_CATEGORY


def _iter_yaml_tasks(data: Any) -> Iterable[dict[str, Any]]:
    if isinstance(data, list):
        for t in data:
            if isinstance(t, dict):
                yield t
    elif isinstance(data, dict):
        tasks = data.get("tasks")
        if isinstance(tasks, list):
            for t in tasks:
                if isinstance(t, dict):
                    yield t
        elif "name" in data:
            yield data


def collect_task_names(manifest: Path, task_yaml_roots: Iterable[Path]) -> tuple[set[str], set[str]]:
    """Return (task_names_seen_anywhere, task_names_only_in_manifest)."""
    yaml_names: set[str] = set()
    for root in task_yaml_roots:
        if not root.exists():
            continue
        for path in list(root.rglob("*.yaml")) + list(root.rglob("*.yml")):
            if path.name.endswith(".bak"):
                continue
            try:
                with path.open("r", encoding="utf-8") as h:
                    data = yaml.safe_load(h)
            except (yaml.YAMLError, OSError):
                continue
            for t in _iter_yaml_tasks(data):
                name = str(t.get("name", "")).strip()
                if name:
                    yaml_names.add(name)

    manifest_names: set[str] = set()
    if manifest.exists():
        with manifest.open("r", encoding="utf-8") as h:
            for line in h:
                line = line.strip()
                if line:
                    row = json.loads(line)
                    tn = row.get("task_external_id") or row.get("task_name")
                    if tn:
                        manifest_names.add(tn)

    all_names = yaml_names | manifest_names
    trajectory_only = manifest_names - yaml_names
    return all_names, trajectory_only


def load_overrides(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as h:
        data = yaml.safe_load(h) or {}
    out: dict[str, str] = {}
    if isinstance(data, dict):
        section = data.get("overrides") if "overrides" in data else data
        if isinstance(section, dict):
            for k, v in section.items():
                if isinstance(k, str) and isinstance(v, str):
                    out[k] = v
    return out


def load_manifest_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as h:
        for line in h:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Categorize tasks into C1-C5 / C99_other")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--overrides", type=Path, default=DEFAULT_OVERRIDES)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--task-yaml-root", type=Path, action="append",
                        default=[
                            SIBLINGS / "project-dojo" / "staging",
                            SIBLINGS / "liveweb" / "tasks",
                            SIBLINGS / "Dillinger" / "tasks",
                        ])
    args = parser.parse_args()

    overrides = load_overrides(args.overrides)
    all_names, traj_only = collect_task_names(args.manifest, args.task_yaml_root)
    if not all_names:
        print("ERROR: no task names found in manifest or yaml roots", file=sys.stderr)
        return 1

    out_tasks: dict[str, str] = {}
    counts: Counter[str] = Counter({c: 0 for c in ALL_CATEGORIES})
    for name in sorted(all_names):
        cat = categorize_one(name, overrides)
        out_tasks[name] = cat
        counts[cat] += 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "counts": {c: int(counts[c]) for c in ALL_CATEGORIES if counts[c] > 0 or c == DEFAULT_CATEGORY},
        "tasks": out_tasks,
    }
    with args.out.open("w", encoding="utf-8") as h:
        yaml.safe_dump(payload, h, sort_keys=True, default_flow_style=False)

    manifest_rows = load_manifest_rows(args.manifest)
    by_cat_trial: Counter[str] = Counter()
    by_cat_tasks_any: dict[str, set[str]] = defaultdict(set)
    by_cat_tasks_perfect: dict[str, set[str]] = defaultdict(set)
    by_cat_steps: dict[str, list[int]] = defaultdict(list)
    for r in manifest_rows:
        tn = r.get("task_external_id") or r.get("task_name") or ""
        cat = out_tasks.get(tn, DEFAULT_CATEGORY)
        by_cat_trial[cat] += 1
        by_cat_tasks_any[cat].add(tn)
        if r.get("reward") == 1.0:
            by_cat_tasks_perfect[cat].add(tn)
        if isinstance(r.get("n_steps"), int):
            by_cat_steps[cat].append(int(r["n_steps"]))

    print(f"Wrote {args.out} with {len(out_tasks)} tasks ({sum(1 for v in out_tasks.values() if v == DEFAULT_CATEGORY)} in {DEFAULT_CATEGORY})")
    print()
    print("Category coverage (cross-referenced with pulled_trajectories.jsonl):")
    print(f"  {'category':14s} {'tasks':>7s} {'trials':>7s} {'tasks_w_traj':>12s} {'tasks_w_perfect':>16s} {'mean_steps':>10s}")
    for cat in ALL_CATEGORIES:
        steps = by_cat_steps.get(cat, [])
        mean_steps = (sum(steps) / len(steps)) if steps else 0.0
        print(
            f"  {cat:14s} "
            f"{counts[cat]:>7d} "
            f"{by_cat_trial[cat]:>7d} "
            f"{len(by_cat_tasks_any.get(cat, set())):>12d} "
            f"{len(by_cat_tasks_perfect.get(cat, set())):>16d} "
            f"{mean_steps:>10.1f}"
        )
    print()
    if traj_only:
        print(f"Trajectory-only tasks (no YAML found): {len(traj_only)}")
        sample = sorted(traj_only)[:15]
        for name in sample:
            print(f"  {out_tasks.get(name, DEFAULT_CATEGORY):14s} {name}")
        if len(traj_only) > 15:
            print(f"  ... and {len(traj_only) - 15} more")
    other_names = [n for n, c in out_tasks.items() if c == DEFAULT_CATEGORY]
    if other_names:
        print()
        print(f"{DEFAULT_CATEGORY} sample (first 20 of {len(other_names)}):")
        for n in sorted(other_names)[:20]:
            print(f"  {n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
