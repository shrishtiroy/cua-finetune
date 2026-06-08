"""Phase 1d: synthesize conduit-compatible task YAMLs from the CortexBench /
Liveweb CSV exports for tasks that exist in our trajectory pool but have NO
YAML definition anywhere across the sibling repos.

Why: ``atif_to_swift.py --require-yaml-for-test`` forces test eligibility on
the presence of a task YAML (so every test task is browser-evaluable). Some
tasks (notably most government tasks) only exist as Supabase trajectories +
CSV-tracked specs from the QA team. This script bridges that gap by emitting
a YAML per such task at ``data/manifests/synthesized_tasks/<name>.yaml``,
which the splitter and ``eval/run_eval.py`` then pick up automatically.

Inputs:
    data/manifests/raw_csv/*.csv          # CortexBench / Liveweb tabs
    data/manifests/pulled_trajectories.jsonl  # determines which tasks have
                                              # at least one passing trajectory
    (the YAML scan inside) — to skip tasks that already have a hand-written YAML

Outputs:
    data/manifests/synthesized_tasks/<task_name>.yaml
    data/manifests/synthesized_tasks/_index.yaml   # summary + provenance

Verifier-parsing rules:
    - Lines (or paragraphs) of the form ``<weight>: <requirement>`` are split
      into separate rubric items with that weight.
    - If no weighted markers are present, the entire verifier text becomes one
      rubric item with weight=100.
    - Empty / whitespace-only sections are dropped.
    - Multiline requirement text is preserved verbatim (conduit accepts it).

Usage::

    python -m data.csv_to_yaml -v
    python -m data.csv_to_yaml --dry-run

Then re-run ``python -m data.atif_to_swift --train-frac 0.70`` to refresh the
split with the synthesized tasks now eligible for test.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger("csv_to_yaml")

REPO_ROOT = Path(__file__).resolve().parents[1]
SIBLINGS = REPO_ROOT.parent
DEFAULT_CSV_DIR = REPO_ROOT / "data" / "manifests" / "raw_csv"
DEFAULT_OUT_DIR = REPO_ROOT / "data" / "manifests" / "synthesized_tasks"
DEFAULT_MANIFEST = REPO_ROOT / "data" / "manifests" / "pulled_trajectories.jsonl"

# Mirror atif_to_swift.DEFAULT_TASK_YAML_ROOTS so we know which tasks already
# have hand-written YAMLs and skip them.
EXISTING_YAML_ROOTS = [
    SIBLINGS / "Dillinger" / "tasks",
    SIBLINGS / "Dillinger" / "QA",
    SIBLINGS / "Dillinger" / "environments",
    SIBLINGS / "Dillinger" / "environments-realworld",
    SIBLINGS / "project-dojo" / "staging",
    SIBLINGS / "project-dojo" / "accepted",
    SIBLINGS / "liveweb" / "tasks",
    SIBLINGS / "liveweb" / "example_tasks",
    SIBLINGS / "liveweb" / "dylan",
    SIBLINGS / "liveweb" / "healthcheck",
    SIBLINGS / "qa-cua-bench" / "tasks",
]

# Verifier rubric-item marker: optional whitespace + integer + ':' at start of line.
# Captures groups: (weight, body-on-same-line). The body continues until the next
# marker (or end of text).
_WEIGHT_MARKER = re.compile(r"(?m)^\s*(\d{1,3})\s*[:.]\s*")


def _find_header_row(rows: list[list[str]]) -> int:
    for i, r in enumerate(rows[:8]):
        if any(cell.strip() == "Task Name" for cell in r):
            return i
    return -1


def _existing_yaml_names(roots: list[Path]) -> set[str]:
    out: set[str] = set()
    for root in roots:
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
            if isinstance(data, dict):
                if isinstance(data.get("name"), str):
                    out.add(data["name"].strip())
                tasks = data.get("tasks")
                if isinstance(tasks, list):
                    for t in tasks:
                        if isinstance(t, dict) and isinstance(t.get("name"), str):
                            out.add(t["name"].strip())
            elif isinstance(data, list):
                for t in data:
                    if isinstance(t, dict) and isinstance(t.get("name"), str):
                        out.add(t["name"].strip())
    return out


def _passing_trajectory_tasks(manifest: Path) -> set[str]:
    out: set[str] = set()
    if not manifest.exists():
        return out
    with manifest.open("r", encoding="utf-8") as h:
        for line in h:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            name = row.get("task_external_id") or row.get("task_name")
            if not name:
                continue
            if (row.get("reward") == 1.0) or (row.get("score") == 100.0):
                out.add(str(name))
    return out


def parse_verifier(text: str) -> list[dict[str, Any]]:
    """Parse a free-form verifier blob into a list of rubric items.

    Returns a list of dicts with keys ``r`` (requirement) and ``w`` (weight).
    """
    text = (text or "").strip()
    if not text:
        return []

    matches = list(_WEIGHT_MARKER.finditer(text))
    if not matches:
        # No weighted markers: treat entire blob as one rubric item.
        return [{"r": text, "w": 100}]

    items: list[dict[str, Any]] = []
    for i, m in enumerate(matches):
        weight = int(m.group(1))
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if not body:
            continue
        items.append({"r": body, "w": weight})

    if not items:
        # Fall back to single item if all markers had empty bodies (shouldn't happen).
        return [{"r": text, "w": 100}]
    return items


def _load_csv_records(csv_dir: Path) -> dict[str, dict[str, Any]]:
    """Return ``{task_name: best_record}`` across all CSV tabs.

    "Best" = first record encountered with both Verifier and Starting URL
    populated. Falls back to first record if none are fully populated.
    """
    by_name: dict[str, dict[str, Any]] = {}
    for path in sorted(csv_dir.glob("*.csv")):
        try:
            with path.open("r", encoding="utf-8") as h:
                rows = list(csv.reader(h))
        except OSError as exc:
            logger.warning("could not read %s: %s", path, exc)
            continue
        hi = _find_header_row(rows)
        if hi < 0:
            continue
        headers = [c.strip() for c in rows[hi]]
        try:
            n_idx = headers.index("Task Name")
        except ValueError:
            continue
        for r in rows[hi + 1:]:
            if len(r) <= n_idx:
                continue
            if len(r) < len(headers):
                r = r + [""] * (len(headers) - len(r))
            rec = dict(zip(headers, r))
            rec["__source_csv__"] = path.name
            name = (rec.get("Task Name") or "").strip()
            if not name:
                continue
            existing = by_name.get(name)
            this_full = bool(rec.get("Verifier", "").strip()) and bool(rec.get("Starting URL", "").strip())
            if existing is None:
                by_name[name] = rec
                continue
            existing_full = bool(existing.get("Verifier", "").strip()) and bool(existing.get("Starting URL", "").strip())
            # Prefer the most-complete record we've seen.
            if this_full and not existing_full:
                by_name[name] = rec
    return by_name


def synthesize_one(rec: dict[str, Any]) -> dict[str, Any]:
    """Build a conduit-compatible task dict from a CSV row."""
    name = rec["Task Name"].strip()
    url = rec.get("Starting URL", "").strip() or "about:blank"
    prompt = rec.get("Task Text", "").strip()
    verifier_text = rec.get("Verifier", "").strip()
    archive = rec.get("Archive", "").strip()
    rubric = parse_verifier(verifier_text)

    out: dict[str, Any] = {
        "name": name,
        "url": url,
        "prompt": prompt,
        "rubric": rubric,
    }
    if archive:
        # Conduit accepts a list of archive names. The CSV "Archive" cell is
        # often a path or filename; strip extension and dirs to get the bare
        # archive id (matches the convention used in hand-written YAMLs like
        # python_docs_tasks.yaml -> archives: [docs-python-org]).
        bare = Path(archive).stem
        out["archives"] = [bare] if bare else []
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Synthesize conduit task YAMLs from CortexBench/Liveweb CSVs")
    parser.add_argument("--csv-dir", type=Path, default=DEFAULT_CSV_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST,
                        help="pulled_trajectories.jsonl (used to gate on ≥1 passing trajectory)")
    parser.add_argument("--require-passing-trajectory", action="store_true", default=True,
                        help="Only emit YAMLs for tasks with ≥1 passing trajectory in the manifest. Default ON.")
    parser.add_argument("--no-require-passing-trajectory",
                        dest="require_passing_trajectory", action="store_false")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print plan but don't write any files.")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        level=logging.DEBUG if args.verbose else logging.INFO,
    )

    csv_records = _load_csv_records(args.csv_dir)
    logger.info("Loaded %d unique task records from %d CSVs",
                len(csv_records), len(list(args.csv_dir.glob("*.csv"))))

    have_yaml = _existing_yaml_names(EXISTING_YAML_ROOTS)
    logger.info("Existing hand-written YAML universe: %d tasks", len(have_yaml))

    have_passing_traj = _passing_trajectory_tasks(args.manifest) if args.require_passing_trajectory else None
    if have_passing_traj is not None:
        logger.info("Trajectory pool with ≥1 pass: %d tasks", len(have_passing_traj))

    candidates: list[tuple[str, dict[str, Any]]] = []
    skip_have_yaml = 0
    skip_no_traj = 0
    skip_incomplete = 0
    for name, rec in sorted(csv_records.items()):
        if name in have_yaml:
            skip_have_yaml += 1
            continue
        if not rec.get("Verifier", "").strip() or not rec.get("Starting URL", "").strip():
            skip_incomplete += 1
            continue
        if have_passing_traj is not None and name not in have_passing_traj:
            skip_no_traj += 1
            continue
        candidates.append((name, rec))

    logger.info("Synthesis candidates: %d", len(candidates))
    logger.info("  skipped: %d already have YAML, %d incomplete CSV row, %d no passing trajectory",
                skip_have_yaml, skip_incomplete, skip_no_traj)

    if args.dry_run:
        for name, rec in candidates:
            spec = synthesize_one(rec)
            print(f"--- {name} ---")
            print(yaml.safe_dump(spec, sort_keys=False, allow_unicode=True))
        return 0

    args.out_dir.mkdir(parents=True, exist_ok=True)
    written: list[dict[str, Any]] = []
    for name, rec in candidates:
        spec = synthesize_one(rec)
        out_path = args.out_dir / f"{name}.yaml"
        with out_path.open("w", encoding="utf-8") as h:
            yaml.safe_dump(spec, h, sort_keys=False, allow_unicode=True)
        n_rubric_items = len(spec["rubric"])
        total_weight = sum(int(it["w"]) for it in spec["rubric"])
        written.append({
            "name": name,
            "source_csv": rec["__source_csv__"],
            "out_path": str(out_path.relative_to(REPO_ROOT)),
            "url": spec["url"],
            "n_rubric_items": n_rubric_items,
            "total_rubric_weight": total_weight,
            "has_archive": bool(spec.get("archives")),
        })

    index_path = args.out_dir / "_index.yaml"
    with index_path.open("w", encoding="utf-8") as h:
        yaml.safe_dump({
            "synthesized_count": len(written),
            "csv_dir": str(args.csv_dir.relative_to(REPO_ROOT)),
            "tasks": written,
        }, h, sort_keys=False, allow_unicode=True)
    logger.info("Wrote %d YAMLs to %s", len(written), args.out_dir)
    logger.info("Index: %s", index_path)

    print()
    print(f"Synthesized {len(written)} task YAMLs:")
    for w in written:
        print(f"  {w['out_path']}  ({w['n_rubric_items']} rubric items, weight={w['total_rubric_weight']})")
    print()
    print("Next: re-run the splitter to pick up the new tasks:")
    print("  python -m data.atif_to_swift --train-frac 0.70 --pass-threshold 1.0 -v")
    return 0


if __name__ == "__main__":
    sys.exit(main())
