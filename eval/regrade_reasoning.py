"""Re-grade existing browser-eval trajectories against the agent's full
reasoning trace (C1: reasoning-based grading).

# USAGE

This is the third grading mode alongside:

- **C0 / original** (``conduit.grading.rubric`` → ``_summary.jsonl``): judges
  trajectory + screenshots via Claude Opus. Tends to reward whatever happens to
  be on the final screen, which biases toward passive ``wait``-spam policies.
- **C1 / strict** (``eval/regrade_strict.py`` → ``_summary_strict.jsonl``):
  requires an explicit ``answer`` / ``terminate`` / ``done`` tool call. Scores
  the declared answer only. Tends to score 0.0 across the board for agents that
  never learned to terminate cleanly (which is most of our LoRA runs — the
  training data has only ~2 ``done`` actions out of 7,200+).

This script implements **C2 / reasoning-based**:

- The agent's full "stream of consciousness" across all steps is concatenated:
  ``reasoning_content`` (Qwen3-VL exposes this as a separate field), the
  flattened ``message`` text, plus a one-line summary of each tool call so the
  judge has concrete action context.
- The trace is graded by Claude Opus against the task's rubric criteria, with
  a modified judge instruction: **mark MET if the criterion is satisfied at ANY
  point in the trace, not just at the end.**

## Policy

**Criterion is MET if satisfied at ANY point in the reasoning trace.** Mid-task
contradictions or subsequent distractions do not unset a MET verdict. UNMET
requires the criterion never to be satisfied anywhere in the trace.

## Why use it

Tests the hypothesis that the LoRA *discovered* the correct answer at some
intermediate step (visible in the agent's ``reasoning_content`` field) but then
kept scrolling and polluted its final state. The original rubric weighted toward
end-of-trajectory state and missed those mid-trajectory insights; strict gave
the LoRA no credit because it never emitted an ``answer`` tool call. The
reasoning-based grader reads the agent's stream-of-consciousness across the
whole trajectory and credits the model for getting the right answer at any
point.

If C2 ≫ C0 on the LoRA but C2 ≈ C0 on the baseline, the LoRA learned the task
but lost the answer in the action stream. If C2 ≈ C0 on both, the LoRA really
did regress. If C2 < C0 on the LoRA, the original rubric was over-crediting
on-screen state.

## Cost estimate

~26 tasks × N_criteria (median 2) × 1 Bedrock Opus call each ≈ 50 calls ×
~$0.025/call ≈ **~$1.30 per (backend, adapter) pair**. Similar to strict.
Longer prompts (the full trace) inflate input tokens but the judge response
is still ~1 short JSON object, so the dominant cost stays in input tokens —
hence the ``--max-trace-chars`` cap to keep cost predictable.

## Commands

Smoke-test (no Bedrock calls)::

    python eval/regrade_reasoning.py qwen_vl_cua baseline --dry-run --max-tasks 3

Full re-grade (requires AWS_* env vars or ``~/Dillinger/.env`` sourced)::

    set -a; source ~/Dillinger/.env; set +a
    python eval/regrade_reasoning.py qwen_vl_cua baseline
    python eval/regrade_reasoning.py qwen_vl_cua cua

Three-way compare::

    python eval/strict_compare.py qwen_vl_cua baseline --reasoning
    python eval/strict_compare.py qwen_vl_cua cua --reasoning

## Output

Writes ``results/<backend>/<adapter>/_summary_reasoning.jsonl`` with one JSON
row per task::

    {
      "task": "pydocs-async-context-dunder",
      "score_reasoning": 0.50,
      "score": 1.00,                         # original rubric score (for parity)
      "n_criteria": 2,
      "n_met": 1,
      "failure_mode": "graded",              # one of: graded, empty_trajectory,
                                             # task_yaml_missing, bedrock_error
      "task_yaml": "data/manifests/test_tasks_yaml/pydocs-async-context-dunder.yaml",
      "trajectory_path": "results/.../trajectory.json",
      "run_timestamp": "2026-06-09_01-02-35-947449",
      "reasoning_chars": 18452,              # length of extracted trace
      "rubric_report": [
        {"requirement": "...", "weight": 50, "verdict": "MET", "reason": "..."},
        ...
      ]
    }
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path
from typing import Any

from regrade_strict import (
    DEFAULT_BEDROCK_MODEL,
    DEFAULT_RESULTS_ROOT,
    DEFAULT_TASK_YAML_ROOTS,
    JUDGE_SYSTEM_PROMPT,
    _aggregate_score,
    _load_orig_scores,
    _parse_judge_verdict,
    _step_message_text,
    grade_criterion_bedrock,
    load_task_rubric,
)

DEFAULT_MAX_TRACE_CHARS = 30_000
# When truncating, keep this much of the head and the rest from the tail. The
# tail is where the agent typically lands on a final answer, so it's weighted
# heavier; the head preserves the initial reasoning about what the task is.
TRACE_HEAD_CHARS = 5_000


# ---------------------------------------------------------------------------
# Trajectory → stream-of-consciousness extraction
# ---------------------------------------------------------------------------


def _format_tool_call(tc: dict[str, Any]) -> str:
    """One-line ``function_name(k=v, ...)`` summary for a single tool call.

    Long string args are truncated so a single ``type(text="...")`` call can't
    eat the whole budget. Non-scalar args are JSON-serialized (with separators
    that keep them compact).
    """
    fn = tc.get("function_name") or tc.get("name") or "?"
    args = tc.get("arguments")
    if not isinstance(args, dict) or not args:
        return f"{fn}()"
    parts: list[str] = []
    for k, v in args.items():
        if k.startswith("model_") and k in ("model_x", "model_y"):
            # Redundant with x/y, just clutter for the judge.
            continue
        if isinstance(v, str):
            shown = v if len(v) <= 200 else v[:197] + "..."
            parts.append(f"{k}={shown!r}")
        elif isinstance(v, (int, float, bool)) or v is None:
            parts.append(f"{k}={v}")
        else:
            try:
                blob = json.dumps(v, ensure_ascii=False, separators=(",", ":"))
            except (TypeError, ValueError):
                blob = repr(v)
            if len(blob) > 200:
                blob = blob[:197] + "..."
            parts.append(f"{k}={blob}")
    return f"{fn}({', '.join(parts)})"


def _step_reasoning_text(step: dict[str, Any]) -> str:
    """Pull ``reasoning_content`` out of a step. May be string, list-of-parts,
    or missing/None. Mirrors ATIF-v1.6 conventions also used by
    ``data/atif_to_swift.py``.
    """
    r = step.get("reasoning_content")
    if r is None:
        return ""
    if isinstance(r, str):
        return r.strip()
    if isinstance(r, list):
        chunks: list[str] = []
        for p in r:
            if isinstance(p, str):
                chunks.append(p)
            elif isinstance(p, dict):
                t = p.get("text") or p.get("content")
                if isinstance(t, str):
                    chunks.append(t)
        return "\n".join(c for c in chunks if c).strip()
    if isinstance(r, dict):
        t = r.get("text") or r.get("content")
        return str(t).strip() if isinstance(t, str) else ""
    return ""


def extract_reasoning_trace(
    trajectory: dict[str, Any],
    *,
    max_chars: int = DEFAULT_MAX_TRACE_CHARS,
) -> tuple[str, int]:
    """Build the full stream-of-consciousness trace from agent steps.

    Returns ``(trace_text, raw_chars)`` where ``raw_chars`` is the length
    *before* truncation (useful for diagnostics — long tails are where
    scroll-spam blows up token budgets).

    Format per step::

        Step N: [reasoning: ...] [message: ...] [action: click(x=412, y=318)]

    Steps with no useful content (no reasoning, no message, no tool calls) are
    skipped to keep the trace compact.
    """
    steps = trajectory.get("steps") or []
    if not isinstance(steps, list):
        return "", 0

    lines: list[str] = []
    step_counter = 0
    for step in steps:
        if not isinstance(step, dict):
            continue
        src = step.get("source") or step.get("role")
        if src not in ("assistant", "agent", "model"):
            continue
        step_counter += 1
        step_id = step.get("step_id", step_counter)

        reasoning = _step_reasoning_text(step)
        msg = _step_message_text(step.get("message"))

        tool_calls = step.get("tool_calls") or []
        tc_strs: list[str] = []
        if isinstance(tool_calls, list):
            for tc in tool_calls:
                if isinstance(tc, dict):
                    tc_strs.append(_format_tool_call(tc))

        parts: list[str] = []
        if reasoning:
            parts.append(f"[reasoning: {reasoning}]")
        if msg:
            parts.append(f"[message: {msg}]")
        if tc_strs:
            parts.append(f"[action: {'; '.join(tc_strs)}]")

        if not parts:
            continue

        lines.append(f"Step {step_id}: " + " ".join(parts))

    full = "\n".join(lines)
    raw_chars = len(full)
    if raw_chars <= max_chars:
        return full, raw_chars

    # Truncate middle. The tail is where final answers tend to surface, so we
    # keep more of it; head preserves "what task is this" context. Match on
    # newlines so we don't slice through a step mid-sentence.
    head_budget = TRACE_HEAD_CHARS
    tail_budget = max_chars - head_budget
    head = full[:head_budget]
    # Snap head to the last full step line so the elision marker is clean.
    nl = head.rfind("\n")
    if nl > 0:
        head = head[:nl]
    tail = full[-tail_budget:]
    nl = tail.find("\n")
    if nl >= 0:
        tail = tail[nl + 1 :]
    elided = raw_chars - len(head) - len(tail)
    truncated = (
        f"{head}\n\n... [TRUNCATED {elided} chars in the middle of the trace] ...\n\n{tail}"
    )
    return truncated, raw_chars


# ---------------------------------------------------------------------------
# Judge prompt
# ---------------------------------------------------------------------------


def _build_reasoning_user_prompt(
    requirement: str, weight: float, query: str | None, trace: str
) -> str:
    """User prompt for the C2 (reasoning-trace) judge call.

    Reuses the same MET/UNMET system prompt (``JUDGE_SYSTEM_PROMPT``) as
    strict/original, but reframes the ``<response>`` field as a full reasoning
    trace and explicitly tells the judge to mark MET on first satisfaction
    anywhere in the trace.
    """
    criterion_type = "negative" if weight < 0 else "positive"
    query_text = f"<query>{query}</query>" if query else ""
    return f"""<criterion_type>
{criterion_type}
</criterion_type>

<criterion>
{requirement}
</criterion>

{query_text}

<response_type>full_trajectory_reasoning</response_type>

<instructions>
The response below is the complete reasoning trace of an agent attempting the task across multiple steps.
Each step is marked "Step N:" and may contain the agent's [reasoning: ...], its narrated [message: ...],
and the [action: ...] it executed.

For POSITIVE criteria: mark MET if the criterion is satisfied or stated correctly AT ANY POINT in the trace,
even if the agent later contradicts itself, gets distracted, or fails to terminate cleanly. Mark UNMET only
if the criterion is never satisfied anywhere in the trace.

For NEGATIVE criteria: mark MET if the agent advocates/states/recommends the problematic thing at any point.
Mark UNMET if the agent never makes the error, or only mentions the thing to warn against it.

The trace may be long. Read it fully before deciding.
</instructions>

<response>
{trace}
</response>"""


def grade_criterion_reasoning_bedrock(
    bedrock_client: Any,
    model_id: str,
    requirement: str,
    weight: float,
    query: str | None,
    trace: str,
    *,
    max_retries: int = 3,
) -> tuple[str, str]:
    """Like ``grade_criterion_bedrock`` but with the reasoning-trace prompt.

    Kept as a separate function (rather than passing a prompt-builder into the
    strict helper) so the call site is greppable and so any future per-mode
    tweaks — token budget, system prompt, etc. — stay localized.
    """
    import json as _json
    import time as _time

    user_prompt = _build_reasoning_user_prompt(requirement, weight, query, trace)
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1024,
        "system": JUDGE_SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": user_prompt}],
    }
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            response = bedrock_client.invoke_model(
                modelId=model_id,
                body=_json.dumps(body).encode("utf-8"),
                contentType="application/json",
                accept="application/json",
            )
            payload = _json.loads(response["body"].read())
            text_parts: list[str] = []
            for part in payload.get("content", []) or []:
                if isinstance(part, dict) and part.get("type") == "text":
                    t = part.get("text")
                    if isinstance(t, str):
                        text_parts.append(t)
            raw = "\n".join(text_parts).strip() or _json.dumps(payload)
            verdict, _ = _parse_judge_verdict(raw)
            return verdict, raw
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt < max_retries - 1:
                _time.sleep(2 ** attempt)
    raise RuntimeError(
        f"Bedrock grading (reasoning) failed after {max_retries} attempts: {last_exc}"
    )


# ---------------------------------------------------------------------------
# Main re-grade loop
# ---------------------------------------------------------------------------


def regrade(
    backend: str,
    adapter: str,
    *,
    results_root: Path = DEFAULT_RESULTS_ROOT,
    task_yaml_roots: list[Path] | None = None,
    bedrock_model: str = DEFAULT_BEDROCK_MODEL,
    dry_run: bool = False,
    max_tasks: int | None = None,
    max_trace_chars: int = DEFAULT_MAX_TRACE_CHARS,
) -> int:
    task_yaml_roots = task_yaml_roots or DEFAULT_TASK_YAML_ROOTS
    base = results_root / backend / adapter
    if not base.exists():
        print(f"ERROR: {base} does not exist", file=sys.stderr)
        return 1

    orig_scores = _load_orig_scores(base / "_summary.jsonl")

    bedrock_client: Any = None
    if not dry_run:
        try:
            import boto3  # type: ignore[import-not-found]
        except ImportError:
            print(
                "ERROR: boto3 is required for live grading. "
                "Install it (`pip install boto3`) or pass --dry-run.",
                file=sys.stderr,
            )
            return 2
        region = (
            os.environ.get("AWS_REGION_NAME")
            or os.environ.get("AWS_REGION")
            or os.environ.get("AWS_DEFAULT_REGION")
        )
        kwargs: dict[str, Any] = {}
        if region:
            kwargs["region_name"] = region
        bedrock_client = boto3.client("bedrock-runtime", **kwargs)

    rows: list[dict[str, Any]] = []
    task_dirs = sorted(d for d in base.iterdir() if d.is_dir() and not d.name.startswith("_"))
    if max_tasks is not None:
        task_dirs = task_dirs[:max_tasks]

    n_empty = 0
    n_skipped_no_yaml = 0
    n_graded = 0
    n_errors = 0

    for task_dir in task_dirs:
        task_name = task_dir.name
        run_dirs = sorted(d for d in task_dir.iterdir() if d.is_dir())
        if not run_dirs:
            continue
        latest = run_dirs[-1]
        traj_path = latest / "trajectory.json"
        if not traj_path.exists():
            continue
        try:
            trajectory = json.loads(traj_path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            print(f"  {task_name}: trajectory unreadable ({exc})", file=sys.stderr)
            continue

        trace, raw_chars = extract_reasoning_trace(trajectory, max_chars=max_trace_chars)
        yaml_path, prompt, rubric_items = load_task_rubric(task_name, task_yaml_roots)
        orig = orig_scores.get(task_name)

        row: dict[str, Any] = {
            "task": task_name,
            "score_reasoning": 0.0,
            "score": orig,
            "n_criteria": len(rubric_items),
            "n_met": 0,
            "failure_mode": "graded",
            "task_yaml": str(yaml_path) if yaml_path else None,
            "trajectory_path": str(traj_path),
            "run_timestamp": latest.name,
            "reasoning_chars": raw_chars,
            "rubric_report": [],
        }

        if not trace:
            row["failure_mode"] = "empty_trajectory"
            row["score_reasoning"] = 0.0
            n_empty += 1
            print(f"  {task_name}: no reasoning text extractable -> 0.00")
        elif not rubric_items or yaml_path is None:
            row["failure_mode"] = "task_yaml_missing"
            row["score_reasoning"] = 0.0
            n_skipped_no_yaml += 1
            print(f"  {task_name}: task YAML / rubric missing -> 0.00")
        else:
            reports: list[dict[str, Any]] = []
            try:
                for item in rubric_items:
                    if dry_run:
                        verdict = "MET"
                        raw = "(dry-run: assumed MET)"
                    else:
                        verdict, raw = grade_criterion_reasoning_bedrock(
                            bedrock_client,
                            bedrock_model,
                            item["requirement"],
                            item["weight"],
                            prompt,
                            trace,
                        )
                    reports.append({
                        "requirement": item["requirement"],
                        "weight": item["weight"],
                        "verdict": verdict,
                        "reason": raw[:1000],
                    })
                score, n_met = _aggregate_score(reports)
                row["score_reasoning"] = round(score, 3)
                row["n_met"] = n_met
                row["rubric_report"] = reports
                n_graded += 1
                print(
                    f"  {task_name}: graded {n_met}/{len(reports)} MET -> "
                    f"{score:.2f} (orig={orig if orig is not None else '?'}, "
                    f"trace={raw_chars} chars)"
                )
            except Exception as exc:  # noqa: BLE001
                row["failure_mode"] = "bedrock_error"
                row["score_reasoning"] = 0.0
                row["error"] = str(exc)
                n_errors += 1
                print(f"  {task_name}: bedrock error: {exc}", file=sys.stderr)

        rows.append(row)

    out_path = base / "_summary_reasoning.jsonl"
    with out_path.open("w", encoding="utf-8") as h:
        for row in rows:
            h.write(json.dumps(row, ensure_ascii=False) + "\n")

    print()
    print(f"Wrote {out_path} ({len(rows)} rows)")
    if rows:
        reasoning = [r["score_reasoning"] for r in rows]
        orig = [r["score"] for r in rows if isinstance(r["score"], (int, float))]
        print(f"  reasoning mean = {statistics.mean(reasoning):.3f}  (n={len(reasoning)})")
        if orig:
            print(f"  orig mean      = {statistics.mean(orig):.3f}  (n={len(orig)})")
            print(f"  delta          = {statistics.mean(reasoning) - statistics.mean(orig):+.3f}")
    print(f"  graded            : {n_graded}")
    print(f"  empty_trajectory  : {n_empty}")
    print(f"  task_yaml_missing : {n_skipped_no_yaml}")
    print(f"  bedrock_error     : {n_errors}")
    if dry_run:
        print()
        print("DRY RUN: scores above are dummy (assumed MET on every criterion).")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Reasoning-trace re-grade of existing eval trajectories "
                    "(criterion MET if satisfied at ANY point in the agent's "
                    "stream-of-consciousness).",
    )
    parser.add_argument("backend", help="e.g. qwen_vl_cua, gemma_vl_cua")
    parser.add_argument("adapter", help="e.g. baseline, cua")
    parser.add_argument(
        "--results-root",
        type=Path,
        default=DEFAULT_RESULTS_ROOT,
        help=f"results/ directory (default: {DEFAULT_RESULTS_ROOT})",
    )
    parser.add_argument(
        "--task-yaml-root",
        type=Path,
        action="append",
        default=None,
        help="Directory of task YAMLs to search; pass multiple times. "
             f"Default: {[str(p) for p in DEFAULT_TASK_YAML_ROOTS]}",
    )
    parser.add_argument(
        "--bedrock-model",
        default=DEFAULT_BEDROCK_MODEL,
        help=f"Bedrock model id for the judge (default: {DEFAULT_BEDROCK_MODEL})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip Bedrock calls (assume MET on every criterion). Use to validate "
             "trajectory parsing + task-YAML loading before burning API credits.",
    )
    parser.add_argument(
        "--max-tasks",
        type=int,
        default=None,
        help="Process only the first N tasks (alphabetical). For smoke tests.",
    )
    parser.add_argument(
        "--max-trace-chars",
        type=int,
        default=DEFAULT_MAX_TRACE_CHARS,
        help=f"Truncation cap for the reasoning trace (default {DEFAULT_MAX_TRACE_CHARS}). "
             f"When exceeded, keep the first {TRACE_HEAD_CHARS} chars plus the tail.",
    )
    args = parser.parse_args()
    return regrade(
        args.backend,
        args.adapter,
        results_root=args.results_root,
        task_yaml_roots=args.task_yaml_root,
        bedrock_model=args.bedrock_model,
        dry_run=args.dry_run,
        max_tasks=args.max_tasks,
        max_trace_chars=args.max_trace_chars,
    )


if __name__ == "__main__":
    sys.exit(main())
