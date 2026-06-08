"""Phase 3b: browser-in-the-loop eval driver.

Imports Dillinger as a library (the ``conduit`` package) and runs each held-out
task against the chosen backend, ``--pass-k`` times. Per-run output goes into
``results/<backend>/<adapter>/<task>/<timestamp>/`` with the same on-disk shape
as ``Dillinger/runs/`` so the existing Dillinger viewer can open it.

Run on Lambda (after ``vllm serve --enable-lora`` is up)::

    CUA_LORA_ADAPTER=baseline python eval/run_eval.py --backend qwen_vl_cua --pass-k 5
    CUA_LORA_ADAPTER=cua      python eval/run_eval.py --backend qwen_vl_cua --pass-k 5

The script writes ``results/<backend>/<adapter>/<task>/<ts>/result.json`` for each
attempt; ``aggregate_results.py`` reads from there.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger("run_eval")

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HELDOUT = REPO_ROOT / "data" / "manifests" / "held_out_tasks.yaml"
DEFAULT_RESULTS = REPO_ROOT / "results"

# Standard task search paths. Kept in sync with
# data/atif_to_swift.py:DEFAULT_TASK_YAML_ROOTS so the splitter's
# yaml-eligibility check and the eval driver's task resolver see
# the same universe of task definitions.
TASK_DIRS = [
    REPO_ROOT.parent / "Dillinger" / "tasks",
    REPO_ROOT.parent / "Dillinger" / "QA",
    REPO_ROOT.parent / "Dillinger" / "environments",
    REPO_ROOT.parent / "Dillinger" / "environments-realworld",
    REPO_ROOT.parent / "project-dojo" / "staging",
    REPO_ROOT.parent / "project-dojo" / "accepted",
    REPO_ROOT.parent / "liveweb" / "tasks",
    REPO_ROOT.parent / "liveweb" / "example_tasks",
    REPO_ROOT.parent / "liveweb" / "dylan",
    REPO_ROOT.parent / "liveweb" / "healthcheck",
    REPO_ROOT.parent / "qa-cua-bench" / "tasks",
    # YAMLs synthesized from CortexBench/Liveweb CSVs (see data/csv_to_yaml.py).
    REPO_ROOT / "data" / "manifests" / "synthesized_tasks",
    # Vendored copies of test-task YAMLs (see data/vendor_test_yamls.py).
    # Lets the eval driver work on machines that don't have all sibling repos
    # cloned (e.g. Lambda only has Dillinger).
    REPO_ROOT / "data" / "manifests" / "test_tasks_yaml",
]


def _flatten_held_out(payload: dict[str, Any]) -> list[tuple[str, str]]:
    """Return [(category, task_name), ...].

    Accepts either ``held_out_per_category: {cat: [tasks]}`` (the format that
    ``atif_to_swift.py`` writes) or ``held_out: {cat: [tasks]}`` /
    ``held_out: [tasks]`` (legacy flat formats).
    """
    out: list[tuple[str, str]] = []
    if not isinstance(payload, dict):
        return out
    held = payload.get("held_out_per_category") or payload.get("held_out")
    if isinstance(held, dict):
        for cat, tasks in held.items():
            for t in tasks or []:
                out.append((str(cat), str(t)))
    elif isinstance(held, list):
        for t in held:
            out.append(("C0", str(t)))
    return out


def _resolve_task(name: str):
    """Search the standard YAML dirs for a task with ``name``. Returns a TaskSpec or None."""
    from conduit.task_loader import load_tasks
    for d in TASK_DIRS:
        if not d.exists():
            continue
        for path in list(d.rglob("*.yaml")) + list(d.rglob("*.yml")):
            if path.name.endswith(".bak"):
                continue
            try:
                tasks = load_tasks(path)
            except Exception as exc:  # noqa: BLE001
                logger.debug("skip %s: %s", path, exc)
                continue
            for t in tasks:
                if t.name == name:
                    return t, path
    return None, None


def main() -> int:
    parser = argparse.ArgumentParser(description="Run held-out tasks against a CUA backend")
    parser.add_argument("--backend", required=True,
                        choices=["qwen_vl_cua", "llama_vision_cua", "kimi_vl_cua", "deepseek_vl_cua"])
    parser.add_argument("--held-out", type=Path, default=DEFAULT_HELDOUT)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--pass-k", type=int, default=5)
    parser.add_argument("--task", action="append", default=[], help="Limit to specific task name(s)")
    parser.add_argument("--category", action="append", default=[], help="Limit to specific category code(s)")
    parser.add_argument("--runtime-url", default=None)
    parser.add_argument("--runtime-container", default="conduit-runtime")
    parser.add_argument("--dry-run", action="store_true",
                        help="Resolve tasks and print plan, but don't run anything")
    parser.add_argument("--live-web", action="store_true",
                        help="Force every task to run against the live internet "
                             "instead of pywb-archived .wacz replays. Strips the "
                             "TaskSpec.archives field before handing the task to "
                             "conduit. Faster setup (no archive transfer), but "
                             "verifiers written against frozen archive state may "
                             "fail on time-sensitive sites.")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        level=logging.DEBUG if args.verbose else logging.INFO,
    )

    if not args.held_out.exists():
        print(f"ERROR: held_out_tasks.yaml not found at {args.held_out}", file=sys.stderr)
        return 1
    held = yaml.safe_load(args.held_out.read_text(encoding="utf-8")) or {}
    tasks_raw = _flatten_held_out(held)
    if args.task:
        wanted = set(args.task)
        tasks_raw = [(c, t) for c, t in tasks_raw if t in wanted]
    if args.category:
        wanted = set(args.category)
        tasks_raw = [(c, t) for c, t in tasks_raw if c in wanted]
    logger.info("Held-out tasks to run: %d", len(tasks_raw))

    # Resolve every task to a TaskSpec up front so failures surface before we spend
    # any wall-clock time on the runtime.
    plan: list[tuple[str, str, Any, Path]] = []  # (cat, name, TaskSpec, source_yaml)
    missing: list[str] = []
    n_archives_stripped = 0
    for cat, name in tasks_raw:
        task, source_yaml = _resolve_task(name)
        if task is None:
            missing.append(name)
            continue
        if args.live_web and getattr(task, "archives", None):
            # Empty archives list -> conduit's runtime takes the live-web path
            # (browser_server.RuntimeState.activate_archives short-circuits when
            # the list is empty).
            n_archives_stripped += 1
            try:
                task.archives = []
            except (AttributeError, TypeError):
                # TaskSpec is a dataclass with slots; tolerate frozen variants by
                # rebuilding via dataclasses.replace if direct assignment fails.
                import dataclasses
                task = dataclasses.replace(task, archives=[])
        plan.append((cat, name, task, source_yaml))

    if args.live_web:
        logger.info("--live-web: stripped archives from %d task(s); all tasks will hit live URLs",
                    n_archives_stripped)

    if missing:
        logger.warning("Could not resolve %d task(s): %s", len(missing), missing)

    adapter = os.environ.get("CUA_LORA_ADAPTER", "baseline").strip() or "baseline"
    out_root = args.results_dir / args.backend / adapter
    out_root.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        print(json.dumps({
            "backend": args.backend,
            "adapter": adapter,
            "pass_k": args.pass_k,
            "n_tasks": len(plan),
            "missing": missing,
            "plan": [{"category": c, "task": n, "source_yaml": str(p)} for c, n, _, p in plan],
            "results_root": str(out_root),
        }, indent=2))
        return 0

    # Real run — Dillinger imports go here so --dry-run works without conduit's heavy deps.
    from conduit.config import load_settings
    from conduit.runtime.computer_loop import run_task
    from conduit.runtime.runtime_client import RuntimeClient
    from eval.backends import get_backend_class

    settings = load_settings()
    if args.runtime_url:
        settings.runtime_url = args.runtime_url
    runtime = RuntimeClient(settings.runtime_url)
    backend_cls = get_backend_class(args.backend)
    backend = backend_cls(settings)
    logger.info("Backend %s ready (adapter=%s)", args.backend, backend.adapter)

    summary_rows: list[dict[str, Any]] = []
    for cat, name, task, source_yaml in plan:
        for k in range(args.pass_k):
            ts = datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S-%f")
            run_dir = out_root / name / ts
            try:
                artifact = run_task(
                    settings=settings,
                    runtime=runtime,
                    backend=backend,
                    task=task,
                    run_dir=run_dir,
                    project_root=REPO_ROOT.parent / "Dillinger",
                    runtime_container=args.runtime_container,
                )
                grade = artifact.grade
                score = float(grade.score) if grade else 0.0
                summary_rows.append({
                    "category": cat,
                    "task": name,
                    "attempt": k,
                    "score": score,
                    "run_dir": str(run_dir),
                    "source_yaml": str(source_yaml),
                })
                logger.info("[%s/%s] attempt %d/%d → score=%.3f", cat, name, k + 1, args.pass_k, score)
            except Exception as exc:  # noqa: BLE001
                logger.exception("Task %s attempt %d failed: %s", name, k, exc)
                summary_rows.append({
                    "category": cat,
                    "task": name,
                    "attempt": k,
                    "score": 0.0,
                    "error": str(exc),
                    "run_dir": str(run_dir),
                    "source_yaml": str(source_yaml),
                })

    summary_path = out_root / "_summary.jsonl"
    with summary_path.open("w", encoding="utf-8") as h:
        for r in summary_rows:
            h.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {summary_path} ({len(summary_rows)} rows)")
    print(f"Backend parse failures: {backend.parse_failures}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
