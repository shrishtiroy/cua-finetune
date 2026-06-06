"""Phase 1b: ATIF trajectories -> ms-swift sharegpt-multimodal JSONL via rejection sampling.

CLI::

    python -m data.atif_to_swift \
        --pass-threshold 1.0 \
        --max-per-task 3 \
        --train-frac 0.8 \
        --thin-fallback 0.8 \
        --seed 42

Algorithm:
    1. Load ``pulled_trajectories.jsonl`` and ``categories.yaml``.
    2. Per task_external_id:
         - Filter to ``reward >= pass_threshold``.
         - If 0 passing: try ``reward >= thin_fallback`` IF this task's category
           still has < 5 unique tasks with any selected trajectories.
         - Keep the ``max_per_task`` shortest passing trajectories; agent
           tie-break order: ``conduit > dillinger > taiga-nibbles > computer-1 > refresh-editor``.
    3. Task-level 80/20 split stratified by category, fixed seed.
    4. For each kept trajectory's agent step, emit one sharegpt example with
       only the *current* step's screenshot in ``images``.

Outputs:
    data/cua_sft/train.jsonl
    data/cua_sft/test.jsonl
    data/manifests/splits.yaml
    data/manifests/held_out_tasks.yaml      (~10 per category, balanced)
    data/manifests/dataset_stats.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import yaml

logger = logging.getLogger("atif_to_swift")

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "data" / "manifests" / "pulled_trajectories.jsonl"
DEFAULT_CATEGORIES = REPO_ROOT / "data" / "manifests" / "categories.yaml"
DEFAULT_OUT_DIR = REPO_ROOT / "data" / "cua_sft"
DEFAULT_SPLITS = REPO_ROOT / "data" / "manifests" / "splits.yaml"
DEFAULT_HELDOUT = REPO_ROOT / "data" / "manifests" / "held_out_tasks.yaml"
DEFAULT_STATS = REPO_ROOT / "data" / "manifests" / "dataset_stats.yaml"

DEFAULT_CATEGORY = "C99_other"
DROP_CATEGORY = "C_drop_non_browser"
ALL_CATEGORIES = ["C1_ui_nav", "C2_structured", "C3_docs", "C4_shopping", "C5_government", DEFAULT_CATEGORY]

AGENT_PRIORITY = ("conduit", "dillinger", "taiga-nibbles", "computer-1", "refresh-editor")

SYSTEM_PROMPT = "You are a computer-use agent. Output JSON with reasoning and action."


@dataclass
class Trajectory:
    row: dict[str, Any]
    task: str
    agent: str
    reward: float
    n_steps: int
    category: str
    trajectory_path: Path
    screenshots_dir: Path


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as h:
        for line in h:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _load_categories(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as h:
        data = yaml.safe_load(h) or {}
    if isinstance(data, dict):
        tasks = data.get("tasks") if "tasks" in data else data
        if isinstance(tasks, dict):
            return {str(k): str(v) for k, v in tasks.items() if isinstance(v, str)}
    return {}


def _agent_rank(agent: str) -> int:
    try:
        return AGENT_PRIORITY.index(agent)
    except ValueError:
        return len(AGENT_PRIORITY) + 1


def _to_trajectory(row: dict[str, Any], categories: dict[str, str]) -> Trajectory | None:
    task = row.get("task_external_id") or row.get("task_name") or ""
    if not task:
        return None
    traj_path = Path(row.get("local_trajectory_path") or "")
    if not traj_path.exists():
        return None
    ss_dir = Path(row.get("local_screenshots_dir") or "")
    return Trajectory(
        row=row,
        task=task,
        agent=row.get("agent_name") or "unknown",
        reward=float(row.get("reward") or 0.0),
        n_steps=int(row.get("n_steps") or 0),
        category=categories.get(task, DEFAULT_CATEGORY),
        trajectory_path=traj_path,
        screenshots_dir=ss_dir,
    )


def select_trajectories(
    rows: list[dict[str, Any]],
    categories: dict[str, str],
    pass_threshold: float,
    thin_fallback: float,
    max_per_task: int,
) -> tuple[list[Trajectory], dict[str, Any]]:
    grouped: dict[str, list[Trajectory]] = defaultdict(list)
    n_dropped_non_browser_rows = 0
    dropped_non_browser_tasks: set[str] = set()
    for row in rows:
        t = _to_trajectory(row, categories)
        if t is None:
            continue
        if t.category == DROP_CATEGORY:
            n_dropped_non_browser_rows += 1
            dropped_non_browser_tasks.add(t.task)
            continue
        grouped[t.task].append(t)

    by_cat_selected_tasks: dict[str, set[str]] = defaultdict(set)
    dropped_tasks: list[str] = []
    thin_fallback_categories: set[str] = set()
    selected_by_task: dict[str, list[Trajectory]] = {}

    pending_for_fallback: list[str] = []
    for task, candidates in grouped.items():
        passing = [c for c in candidates if c.reward >= pass_threshold]
        if not passing:
            pending_for_fallback.append(task)
            continue
        passing.sort(key=lambda t: (t.n_steps, _agent_rank(t.agent), -t.reward))
        kept = passing[:max_per_task]
        selected_by_task[task] = kept
        by_cat_selected_tasks[kept[0].category].add(task)

    for task in pending_for_fallback:
        candidates = grouped[task]
        cat = categories.get(task, DEFAULT_CATEGORY)
        if len(by_cat_selected_tasks[cat]) >= 5:
            dropped_tasks.append(task)
            continue
        relaxed = [c for c in candidates if c.reward >= thin_fallback]
        if not relaxed:
            dropped_tasks.append(task)
            continue
        relaxed.sort(key=lambda t: (t.n_steps, _agent_rank(t.agent), -t.reward))
        kept = relaxed[:max_per_task]
        selected_by_task[task] = kept
        by_cat_selected_tasks[cat].add(task)
        thin_fallback_categories.add(cat)

    selected: list[Trajectory] = []
    for kept in selected_by_task.values():
        selected.extend(kept)

    info = {
        "n_tasks_input": len(grouped) + len(dropped_non_browser_tasks),
        "n_tasks_selected": len(selected_by_task),
        "n_trajectories_selected": len(selected),
        "dropped_tasks": dropped_tasks,
        "thin_fallback_categories": sorted(thin_fallback_categories),
        "n_dropped_non_browser_rows": n_dropped_non_browser_rows,
        "n_dropped_non_browser_tasks": len(dropped_non_browser_tasks),
        "dropped_non_browser_tasks_sample": sorted(dropped_non_browser_tasks)[:20],
    }
    return selected, info


def stratified_split(
    trajectories: list[Trajectory],
    train_frac: float,
    rng: random.Random,
) -> tuple[set[str], set[str]]:
    by_cat: dict[str, list[str]] = defaultdict(list)
    seen: set[str] = set()
    for t in trajectories:
        if t.task in seen:
            continue
        seen.add(t.task)
        by_cat[t.category].append(t.task)

    train_tasks: set[str] = set()
    test_tasks: set[str] = set()
    for _cat, tasks in by_cat.items():
        rng.shuffle(tasks)
        if len(tasks) == 1:
            train_tasks.update(tasks)
            continue
        n_test = max(1, int(round(len(tasks) * (1 - train_frac))))
        test_tasks.update(tasks[:n_test])
        train_tasks.update(tasks[n_test:])
    return train_tasks, test_tasks


def _resolve_screenshot(traj_dir: Path, ref: str | None) -> Path | None:
    if not ref:
        return None
    p = Path(ref)
    candidates: list[Path] = []
    if p.is_absolute():
        candidates.append(p)
    rel = ref.lstrip("/")
    if rel.startswith("agent/"):
        rel = rel[len("agent/"):]
    candidates.extend([
        traj_dir / rel,
        traj_dir / ref,
        traj_dir / Path(ref).name,
        traj_dir / "screenshots" / Path(ref).name,
    ])
    for c in candidates:
        if c.exists() and c.stat().st_size > 0:
            return c.resolve()
    return None


def _extract_screenshot_from_message(msg: Any) -> str | None:
    if isinstance(msg, list):
        for part in msg:
            if isinstance(part, dict) and part.get("type") == "image":
                src = part.get("source") or {}
                p = src.get("path") or src.get("url")
                if isinstance(p, str):
                    return p
    elif isinstance(msg, dict):
        if msg.get("type") == "image":
            src = msg.get("source") or {}
            p = src.get("path") or src.get("url")
            if isinstance(p, str):
                return p
    return None


def _extract_observation_screenshot(step: dict[str, Any]) -> str | None:
    obs = step.get("observation") or {}
    for r in obs.get("results", []) or []:
        content = r.get("content")
        ref = _extract_screenshot_from_message(content)
        if ref:
            return ref
    return None


def _format_history(fn: str, args: dict[str, Any]) -> str:
    if fn in {"click", "left_click", "double_click", "triple_click", "right_click"}:
        return f"{fn}({args.get('x')}, {args.get('y')})"
    if fn == "type":
        text = (args.get("text") or "").replace("\n", "\\n")
        if len(text) > 40:
            text = text[:37] + "..."
        return f"type({text!r})"
    if fn == "keypress":
        keys = args.get("keys") or []
        return f"keypress({'+'.join(keys) if keys else ''})"
    if fn == "scroll":
        return f"scroll(dx={args.get('scroll_x', 0)}, dy={args.get('scroll_y', 0)})"
    if fn == "drag":
        return f"drag(({args.get('x')},{args.get('y')})->({args.get('end_x')},{args.get('end_y')}))"
    if fn in {"terminate", "answer", "done"}:
        return f"{fn}({args.get('result') or args.get('text') or ''})"
    if fn == "wait":
        return f"wait({args.get('duration', '')})"
    return f"{fn}(...)"


_VALID_FUNCTIONS = {
    "click", "left_click", "double_click", "right_click", "triple_click",
    "type", "keypress", "scroll", "wait", "screenshot",
    "drag", "mouse_move", "mouse_down", "mouse_up", "hold_key",
    "terminate", "answer", "done", "zoom",
}


def emit_examples(traj: dict[str, Any], traj_dir: Path) -> Iterable[dict[str, Any]]:
    steps = traj.get("steps") or []
    if not steps:
        return

    first = steps[0]
    if first.get("source") != "user":
        return
    instruction = ""
    msg = first.get("message")
    if isinstance(msg, list):
        for part in msg:
            if isinstance(part, dict) and part.get("type") == "text":
                instruction = (part.get("text") or "").strip()
                break
    elif isinstance(msg, str):
        instruction = msg.strip()
    if not instruction:
        return

    last_screenshot_ref = _extract_screenshot_from_message(msg)
    history: list[str] = []

    for step in steps:
        if step.get("source") != "agent":
            ref = _extract_screenshot_from_message(step.get("message")) or _extract_observation_screenshot(step)
            if ref:
                last_screenshot_ref = ref
            continue

        tool_calls = step.get("tool_calls") or []
        if not tool_calls:
            ref = _extract_observation_screenshot(step)
            if ref:
                last_screenshot_ref = ref
            continue

        tc = tool_calls[0]
        fn = (tc.get("function_name") or "").strip()
        if fn not in _VALID_FUNCTIONS:
            ref = _extract_observation_screenshot(step)
            if ref:
                last_screenshot_ref = ref
            continue
        args = tc.get("arguments") or {}
        clean_args = {k: v for k, v in args.items() if k not in {"model_x", "model_y"}}

        ss = _resolve_screenshot(traj_dir, last_screenshot_ref)
        if ss is None:
            ref = _extract_observation_screenshot(step)
            if ref:
                last_screenshot_ref = ref
            history.append(_format_history(fn, clean_args))
            continue

        reasoning = step.get("reasoning_content") or step.get("message") or ""
        if isinstance(reasoning, list):
            chunks = [p.get("text", "") for p in reasoning if isinstance(p, dict) and p.get("type") == "text"]
            reasoning = " ".join(c for c in chunks if c).strip()
        reasoning = (reasoning or "").strip()
        if len(reasoning) > 800:
            reasoning = reasoning[:797] + "..."

        history_block = ""
        if history:
            shown = history[-12:]
            history_block = "Step history:\n" + "\n".join(f"{i+1}. {h}" for i, h in enumerate(shown))

        user_text_parts = [f"<image>Task: {instruction}"]
        if history_block:
            user_text_parts.append(history_block)
        user_text_parts.append("What is the next action?")
        user_text = "\n\n".join(user_text_parts)

        assistant_obj = {
            "reasoning": reasoning,
            "action": {"function_name": fn, "arguments": clean_args},
        }
        yield {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": json.dumps(assistant_obj, ensure_ascii=False)},
            ],
            "images": [str(ss)],
        }
        history.append(_format_history(fn, clean_args))
        ref = _extract_observation_screenshot(step)
        if ref:
            last_screenshot_ref = ref


def balanced_heldout(
    test_tasks: set[str],
    categories: dict[str, str],
    n_per_category: int,
    rng: random.Random,
) -> dict[str, list[str]]:
    by_cat: dict[str, list[str]] = defaultdict(list)
    for t in test_tasks:
        by_cat[categories.get(t, DEFAULT_CATEGORY)].append(t)
    out: dict[str, list[str]] = {}
    for cat in ALL_CATEGORIES:
        tasks = sorted(by_cat.get(cat, []))
        rng.shuffle(tasks)
        out[cat] = tasks[:n_per_category]
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="ATIF -> ms-swift sharegpt JSONL with rejection sampling")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--categories", type=Path, default=DEFAULT_CATEGORIES)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--splits-out", type=Path, default=DEFAULT_SPLITS)
    parser.add_argument("--heldout-out", type=Path, default=DEFAULT_HELDOUT)
    parser.add_argument("--stats-out", type=Path, default=DEFAULT_STATS)
    parser.add_argument("--pass-threshold", type=float, default=1.0)
    parser.add_argument("--thin-fallback", type=float, default=0.8)
    parser.add_argument("--max-per-task", type=int, default=3)
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--heldout-per-category", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        level=logging.DEBUG if args.verbose else logging.INFO,
    )

    if not args.manifest.exists():
        print(f"ERROR: manifest not found at {args.manifest}", file=sys.stderr)
        return 1
    categories = _load_categories(args.categories)
    rows = _load_jsonl(args.manifest)
    logger.info("Loaded %d manifest rows, %d task categories", len(rows), len(categories))

    selected, info = select_trajectories(
        rows=rows,
        categories=categories,
        pass_threshold=args.pass_threshold,
        thin_fallback=args.thin_fallback,
        max_per_task=args.max_per_task,
    )
    logger.info("Selected %d trajectories across %d tasks (%d dropped, thin-fallback in %s)",
                info["n_trajectories_selected"], info["n_tasks_selected"],
                len(info["dropped_tasks"]), info["thin_fallback_categories"])

    rng = random.Random(args.seed)
    train_tasks, test_tasks = stratified_split(selected, args.train_frac, rng)
    logger.info("Task-level split: train=%d test=%d", len(train_tasks), len(test_tasks))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.out_dir / "train.jsonl"
    test_path = args.out_dir / "test.jsonl"

    examples_by_cat: dict[str, dict[str, int]] = defaultdict(lambda: {"train": 0, "test": 0})
    trajectory_steps: list[int] = []
    skipped_no_image = 0
    train_n = test_n = 0

    with train_path.open("w", encoding="utf-8") as train_f, test_path.open("w", encoding="utf-8") as test_f:
        for t in selected:
            try:
                with t.trajectory_path.open("r", encoding="utf-8") as h:
                    traj = json.load(h)
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("skipping %s: %s", t.trajectory_path, exc)
                continue
            target = test_f if t.task in test_tasks else train_f
            label = "test" if t.task in test_tasks else "train"
            n_emitted = 0
            for ex in emit_examples(traj, t.trajectory_path.parent):
                if not ex.get("images"):
                    skipped_no_image += 1
                    continue
                target.write(json.dumps(ex, ensure_ascii=False) + "\n")
                n_emitted += 1
                examples_by_cat[t.category][label] += 1
                if label == "test":
                    test_n += 1
                else:
                    train_n += 1
            trajectory_steps.append(n_emitted)

    args.splits_out.parent.mkdir(parents=True, exist_ok=True)
    splits_payload = {
        "pass_threshold": args.pass_threshold,
        "thin_fallback": args.thin_fallback,
        "max_per_task": args.max_per_task,
        "train_frac": args.train_frac,
        "seed": args.seed,
        "n_train_tasks": len(train_tasks),
        "n_test_tasks": len(test_tasks),
        "train_tasks": sorted(train_tasks),
        "test_tasks": sorted(test_tasks),
        "dropped_tasks": sorted(info["dropped_tasks"]),
        "thin_fallback_categories": list(info["thin_fallback_categories"]),
    }
    with args.splits_out.open("w", encoding="utf-8") as h:
        yaml.safe_dump(splits_payload, h, sort_keys=False)

    held_out = balanced_heldout(test_tasks, categories, args.heldout_per_category, random.Random(args.seed + 1))
    args.heldout_out.parent.mkdir(parents=True, exist_ok=True)
    with args.heldout_out.open("w", encoding="utf-8") as h:
        yaml.safe_dump({"held_out_per_category": held_out}, h, sort_keys=False)

    mean_steps = (sum(trajectory_steps) / len(trajectory_steps)) if trajectory_steps else 0.0
    stats_payload = {
        "n_train_examples": train_n,
        "n_test_examples": test_n,
        "n_train_tasks": len(train_tasks),
        "n_test_tasks": len(test_tasks),
        "examples_per_category": {
            cat: dict(examples_by_cat.get(cat, {"train": 0, "test": 0}))
            for cat in ALL_CATEGORIES
        },
        "mean_steps_per_trajectory": round(mean_steps, 2),
        "dropped_task_count": len(info["dropped_tasks"]),
        "thin_fallback_categories": info["thin_fallback_categories"],
        "skipped_examples_no_image": skipped_no_image,
        "pass_threshold": args.pass_threshold,
        "thin_fallback": args.thin_fallback,
        "max_per_task": args.max_per_task,
        "train_frac": args.train_frac,
        "seed": args.seed,
    }
    args.stats_out.parent.mkdir(parents=True, exist_ok=True)
    with args.stats_out.open("w", encoding="utf-8") as h:
        yaml.safe_dump(stats_payload, h, sort_keys=False)

    print(json.dumps({
        **stats_payload,
        "train_path": str(train_path),
        "test_path": str(test_path),
        "splits_path": str(args.splits_out),
        "heldout_path": str(args.heldout_out),
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
