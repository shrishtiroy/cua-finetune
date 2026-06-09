"""Phase 3a helper: copy the YAML definition of every held-out test task into
``data/manifests/test_tasks_yaml/<name>.yaml`` so the eval driver can run on
machines that don't have all sibling repos cloned (e.g. Lambda only has
Dillinger, not liveweb / project-dojo / qa-cua-bench).

Each output YAML contains a single task spec (``name``, ``url``, ``prompt``,
``rubric``, optionally ``archives`` / ``environment`` / ``verifier``) — the
same shape conduit's task_loader expects. We extract individual task entries
from multi-task YAMLs (which is the most common shape under
``Dillinger/tasks/`` and ``liveweb/tasks/``) so each file is independently
loadable.

The output dir is committed to git, so once Lambda pulls cua-finetune it has
everything required to resolve the 26 test tasks without rsyncing the other
sibling repos.

Re-run after the test split changes.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Iterable

import yaml

logger = logging.getLogger("vendor_test_yamls")

REPO_ROOT = Path(__file__).resolve().parents[1]
SIBLINGS = REPO_ROOT.parent
DEFAULT_HELDOUT = REPO_ROOT / "data" / "manifests" / "held_out_tasks.yaml"
DEFAULT_OUT_DIR = REPO_ROOT / "data" / "manifests" / "test_tasks_yaml"

SEARCH_ROOTS = [
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
    REPO_ROOT / "data" / "manifests" / "synthesized_tasks",
]


def _iter_yamls(roots: Iterable[Path]) -> Iterable[Path]:
    for root in roots:
        if not root.exists():
            continue
        for p in list(root.rglob("*.yaml")) + list(root.rglob("*.yml")):
            if p.name.endswith(".bak") or p.name.startswith("_"):
                continue
            yield p


def _flatten(payload: Any) -> list[dict[str, Any]]:
    """Return list of task-dicts contained in a YAML file."""
    out: list[dict[str, Any]] = []
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict) and isinstance(item.get("name"), str):
                out.append(item)
    elif isinstance(payload, dict):
        if isinstance(payload.get("name"), str):
            out.append(payload)
        tasks = payload.get("tasks")
        if isinstance(tasks, list):
            for item in tasks:
                if isinstance(item, dict) and isinstance(item.get("name"), str):
                    out.append(item)
    return out


def find_task_def(name: str, roots: Iterable[Path]) -> tuple[dict[str, Any], Path] | None:
    for path in _iter_yamls(roots):
        try:
            with path.open("r", encoding="utf-8") as h:
                data = yaml.safe_load(h)
        except (yaml.YAMLError, OSError):
            continue
        for task in _flatten(data):
            if str(task.get("name", "")).strip() == name:
                return task, path
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Vendor test-task YAMLs into the repo")
    parser.add_argument("--held-out", type=Path, default=DEFAULT_HELDOUT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        level=logging.DEBUG if args.verbose else logging.INFO,
    )

    held = yaml.safe_load(args.held_out.read_text(encoding="utf-8")) or {}
    per_cat = held.get("held_out_per_category") or held.get("held_out") or {}
    if isinstance(per_cat, list):
        per_cat = {"C0": per_cat}
    test_tasks: list[tuple[str, str]] = []
    for cat, names in per_cat.items():
        for n in names or []:
            test_tasks.append((str(cat), str(n)))
    logger.info("Held-out test tasks: %d", len(test_tasks))

    if not args.dry_run:
        args.out_dir.mkdir(parents=True, exist_ok=True)

    found_summary: list[dict[str, Any]] = []
    missing: list[tuple[str, str]] = []
    for cat, name in test_tasks:
        result = find_task_def(name, SEARCH_ROOTS)
        if result is None:
            missing.append((cat, name))
            logger.warning("MISSING %s (cat=%s)", name, cat)
            continue
        task_def, source = result
        out_path = args.out_dir / f"{name}.yaml"
        rel_source = str(source.relative_to(SIBLINGS) if source.is_relative_to(SIBLINGS) else source)
        if not args.dry_run:
            with out_path.open("w", encoding="utf-8") as h:
                # conduit's task_loader.load_tasks() expects either a top-level
                # list of task dicts OR a dict with a ``tasks:`` key. Writing a
                # bare dict makes it return [] silently. Wrap as a 1-element
                # list so load_tasks works.
                yaml.safe_dump([task_def], h, sort_keys=False, allow_unicode=True)
        found_summary.append({
            "name": name,
            "category": cat,
            "source": rel_source,
            "out": str(out_path.relative_to(REPO_ROOT)),
        })
        logger.debug("vendored %s from %s", name, rel_source)

    if not args.dry_run:
        index_path = args.out_dir / "_index.yaml"
        with index_path.open("w", encoding="utf-8") as h:
            yaml.safe_dump({
                "vendored_count": len(found_summary),
                "missing_count": len(missing),
                "missing": [{"name": n, "category": c} for c, n in missing],
                "tasks": found_summary,
            }, h, sort_keys=False, allow_unicode=True)
        logger.info("Wrote %d YAMLs to %s", len(found_summary), args.out_dir)
        logger.info("Index: %s", index_path)

    print()
    print(f"Vendored: {len(found_summary)}/{len(test_tasks)} test-task YAMLs")
    if missing:
        print(f"Missing : {len(missing)}")
        for cat, name in missing:
            print(f"   {cat}  {name}")
    print()
    return 0 if not missing else 1


if __name__ == "__main__":
    sys.exit(main())
