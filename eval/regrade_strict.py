"""Re-grade existing browser-eval trajectories under a STRICT answer-only rubric.

# USAGE

The existing rubric (``conduit.grading.rubric``) appears to grade tasks by what
is visible on the final screen — which rewards the base Qwen3-VL-8B model for
emitting long sequences of ``wait`` actions on knowledge-recall tasks where the
answer happens to be in the initial viewport. The LoRA we trained moved Qwen
towards a more action-oriented policy (scroll-spam instead of wait-spam), which
moves the answer off-screen and gets the rubric to score 0.0.

This script re-scores existing trajectories under a stricter policy:

- The agent must *explicitly declare* an answer via an ``answer`` / ``terminate``
  / ``done`` / ``final_answer`` / ``submit`` / ``finish`` / ``stop`` tool call
  (or, as a fallback, by emitting a non-empty assistant ``message`` on the
  final step).
- If no answer was declared, the task is scored ``0.0`` *without* an LLM call.
- Otherwise the declared answer (NOT the final screen state) is graded against
  the task YAML's ``rubric`` criteria by Claude Opus on Bedrock, using the same
  per-criterion MET/UNMET prompt that ``conduit.grading.rubric`` uses.

Use this to test whether the apparent LoRA regression is partly an artifact of
the original rubric rewarding passive "answer-stays-on-screen" behavior. Under
a non-passivity-rewarding rubric the LoRA may net positive.

## Cost estimate

~26 tasks × N_criteria (median 2) × 1 Bedrock Opus call each ≈ 50 calls ×
~$0.025/call ≈ **~$1.30 per (backend, adapter) pair**. Compare to ~50 min and
~$2 to actually re-run the agent against the browser runtime. The script does
NOT touch the browser runtime; it only re-grades artifacts already on disk.

## Commands

Smoke-test (no Bedrock calls)::

    python eval/regrade_strict.py qwen_vl_cua baseline --dry-run --max-tasks 3

Full re-grade (requires AWS_* env vars or ``~/Dillinger/.env`` sourced)::

    set -a; source ~/Dillinger/.env; set +a
    python eval/regrade_strict.py qwen_vl_cua baseline
    python eval/regrade_strict.py qwen_vl_cua cua

Compare::

    python eval/strict_compare.py qwen_vl_cua baseline
    python eval/strict_compare.py qwen_vl_cua cua

## Output

Writes ``results/<backend>/<adapter>/_summary_strict.jsonl`` with one JSON row
per task::

    {
      "task": "pydocs-async-context-dunder",
      "score_strict": 0.50,
      "score": 1.00,                         # original rubric score (for parity)
      "declared_answer": "__aenter__ and __aexit__ ...",
      "n_criteria": 2,
      "n_met": 1,
      "failure_mode": "graded",              # one of: graded, no_answer_declared,
                                             # task_yaml_missing, bedrock_error
      "task_yaml": "data/manifests/test_tasks_yaml/pydocs-async-context-dunder.yaml",
      "trajectory_path": "results/.../trajectory.json",
      "run_timestamp": "2026-06-09_01-02-35-947449",
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
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover - yaml is in pyproject
    print("ERROR: PyYAML is required (pip install pyyaml)", file=sys.stderr)
    raise

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_ROOT = REPO_ROOT / "results"
DEFAULT_TASK_YAML_ROOTS = [
    REPO_ROOT / "data" / "manifests" / "test_tasks_yaml",
    REPO_ROOT / "data" / "manifests" / "synthesized_tasks",
]

# Bedrock model id for the judge. Matches ``conduit.config.Settings.qa_model``
# default (the conduit-side "QA evaluator" uses Opus 4.5 on Bedrock). The
# actual rubric judge is configured via LITELLM_MODEL / CONDUIT_JUDGE_BACKEND
# in ~/Dillinger/.env; if it differs in production, override with
# ``--bedrock-model``.
DEFAULT_BEDROCK_MODEL = "global.anthropic.claude-opus-4-5-20251101-v1:0"

# Tool calls that count as "the agent declared an answer". Drawn from the
# action vocab observed in our trajectories (e.g. peoplemovin-... uses
# ``terminate``, athena-... uses ``done``). ``stop`` / ``finish`` / ``submit``
# are speculative but cheap to keep — they catch alternative vocabs without
# false positives, since these names aren't used for any other purpose.
ANSWER_FUNCTION_NAMES = {
    "answer",
    "terminate",
    "done",
    "finish",
    "stop",
    "submit",
    "final_answer",
    "return",
}

# Argument keys to try (in order) when extracting the answer string from a
# terminal tool call. Different agent backends use different conventions.
ANSWER_ARG_KEYS = (
    "answer",
    "text",
    "final_answer",
    "result",
    "response",
    "message",
    "content",
    "summary",
    "output",
)

# Verbatim copy of ``rubric.autograders.per_criterion_grader.DEFAULT_SYSTEM_PROMPT``
# from the installed ``rubric`` package, so the strict re-grade uses the same
# judge instructions that the original eval uses. If the upstream prompt
# changes, this drifts — see ``--bedrock-model`` caveat in the commit message.
JUDGE_SYSTEM_PROMPT = """You are evaluating a response for a given query against a single \
criterion.

You will receive the response to evaluate, a single criterion to check, and a \
<criterion_type> field indicating if the criterion is positive or negative.

CRITERION TYPES:
The <criterion_type> field tells you whether this criterion describes something desirable \
(positive) or undesirable (negative). Your job is THE SAME for both types: determine if the thing \
described in the criterion is actually present in the response.

POSITIVE CRITERIA:
Positive criteria describe desired traits, requirements, or content that should be present.
- MET (criterion_status: "MET"): The response contains/satisfies the requirement
- UNMET (criterion_status: "UNMET"): The response does not contain/satisfy the requirement

NEGATIVE CRITERIA:
Negative criteria describe active errors or mistakes that the response is making.
- MET (criterion_status: "MET"): The response advocates, states, or recommends the problematic thing
- UNMET (criterion_status: "UNMET"): The response does NOT make this error, OR it mentions \
the thing only to warn against it or mention why it's wrong

Examples of what does NOT count as MET for negative criteria:
- "This is often misdiagnosed as X, but it's actually Y" \u2192 NOT stating it's X (UNMET)
- "Avoid doing X because..." \u2192 NOT recommending X (UNMET)
- "Unlike X, the correct approach is Y" \u2192 NOT advocating for X (UNMET)
- "A common mistake is thinking X" \u2192 NOT claiming X is correct (UNMET)

EVALUATION RULES:
- For numerical values: Check if they fall within specified ranges or match exactly as required.
- For factual claims: Verify the information is present and accurate, regardless of exact phrasing.
- For required elements: Confirm presence, counting precisely when numbers are specified.
- For exclusion requirements: Confirm that restricted content is absent.
- For length requirements: Carefully measure the number of words, characters, items, etc.
- Be strict about factual accuracy but flexible about wording.
- Accept semantically equivalent statements or implications where appropriate.
- Pay careful attention to negation, warnings, and contrasts.

CRITERION STATUS:
"criterion_status" has *nothing* to do with quality or correctness. It only means:
- "MET": The thing described in the criterion IS present/occurring in the response
- "UNMET": The thing described in the criterion IS NOT present/occurring in the response

Your response must be valid JSON with this exact format:

{
"criterion_status": "MET",
"explanation": "Brief explanation of why the criterion is or isn't present."
}

Return only raw JSON starting with {, no back-ticks, no 'json' prefix."""


# ---------------------------------------------------------------------------
# Trajectory \u2192 declared answer extraction
# ---------------------------------------------------------------------------


def _step_message_text(message: Any) -> str:
    """Extract a flat text string from a step's ``message`` field.

    ATIF-v1.6 messages can be a plain string, a list of content parts
    (``[{type: text, text: ...}, {type: image, ...}]``), or missing entirely.
    """
    if message is None:
        return ""
    if isinstance(message, str):
        return message.strip()
    if isinstance(message, list):
        parts: list[str] = []
        for item in message:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                t = item.get("text") or item.get("content")
                if isinstance(t, str):
                    parts.append(t)
        return "\n".join(p for p in parts if p).strip()
    if isinstance(message, dict):
        t = message.get("text") or message.get("content")
        return str(t).strip() if isinstance(t, str) else ""
    return ""


def _extract_answer_from_args(args: Any) -> str | None:
    if not isinstance(args, dict):
        return None
    for key in ANSWER_ARG_KEYS:
        v = args.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()
        if isinstance(v, (int, float, bool)):
            return str(v)
    # Last resort: serialize the whole dict if it's non-trivial. Avoids the
    # silent-empty case where the agent emitted ``{"foo": "bar"}`` without any
    # of the expected keys; the judge can still read the JSON blob.
    if args:
        try:
            return json.dumps(args, ensure_ascii=False)
        except (TypeError, ValueError):
            return None
    return None


def extract_declared_answer(trajectory: dict[str, Any]) -> tuple[str | None, str]:
    """Return ``(answer, source)`` where ``source`` describes how it was found.

    Priority:
      1. Last step containing a tool_call with function_name in
         ANSWER_FUNCTION_NAMES \u2192 extract from arguments.
      2. Last agent step's ``message`` field (string or list-of-parts), if
         non-empty.
      3. ``(None, "none")`` \u2192 caller should score 0.0 without an LLM call.
    """
    steps = trajectory.get("steps") or []
    if not isinstance(steps, list):
        return None, "none"

    for step in reversed(steps):
        if not isinstance(step, dict):
            continue
        tool_calls = step.get("tool_calls") or []
        if not isinstance(tool_calls, list):
            continue
        for tc in tool_calls:
            if not isinstance(tc, dict):
                continue
            fn = tc.get("function_name")
            if not isinstance(fn, str) or fn not in ANSWER_FUNCTION_NAMES:
                continue
            ans = _extract_answer_from_args(tc.get("arguments"))
            if ans:
                return ans, f"tool_call:{fn}"
            # Function called with no usable args \u2192 fall back to message.
            msg = _step_message_text(step.get("message"))
            if msg:
                return msg, f"tool_call:{fn}+message"
            return None, f"tool_call:{fn}+empty"

    for step in reversed(steps):
        if not isinstance(step, dict):
            continue
        src = step.get("source") or step.get("role")
        if src not in ("assistant", "agent", "model", None):
            continue
        msg = _step_message_text(step.get("message"))
        if msg:
            return msg, "assistant_message"

    return None, "none"


# ---------------------------------------------------------------------------
# Task YAML \u2192 rubric criteria loading
# ---------------------------------------------------------------------------


def _rubric_items_from_raw(rubric: Any) -> list[dict[str, Any]]:
    """Normalize a YAML ``rubric`` value into ``[{requirement, weight}]``.

    Mirrors ``conduit.task_loader._convert_rubric_to_items`` but stays
    dependency-free so this script runs on machines that don't have the
    ``conduit`` package importable.
    """
    out: list[dict[str, Any]] = []
    if rubric is None:
        return out
    if isinstance(rubric, str):
        return [{"requirement": rubric, "weight": 1.0}]
    if not isinstance(rubric, list):
        return out
    for item in rubric:
        if isinstance(item, str):
            out.append({"requirement": item, "weight": 1.0})
        elif isinstance(item, dict):
            req = item.get("requirement") or item.get("r") or item.get("criterion")
            if not isinstance(req, str) or not req.strip():
                continue
            weight = item.get("weight", item.get("w", 1))
            try:
                w = float(weight)
            except (TypeError, ValueError):
                w = 1.0
            out.append({"requirement": req.strip(), "weight": w})
    return out


def _extract_rubric_from_task(task_dict: dict[str, Any]) -> list[dict[str, Any]]:
    """Find rubric criteria under any of the field names we've seen in the wild.

    The held-out YAMLs use ``rubric:``; some upstream specs use ``verifiers:``
    or ``criteria:``. Try them in order.
    """
    for key in ("rubric", "verifiers", "criteria"):
        items = _rubric_items_from_raw(task_dict.get(key))
        if items:
            return items
    return []


def load_task_rubric(
    task_name: str, task_yaml_roots: list[Path]
) -> tuple[Path | None, str | None, list[dict[str, Any]]]:
    """Return ``(yaml_path, prompt, rubric_items)`` for ``task_name``.

    Walks every YAML under the given roots, parses with ``yaml.safe_load``,
    and returns the first task dict whose ``name`` matches.
    """
    for root in task_yaml_roots:
        if not root.exists():
            continue
        for path in sorted(list(root.rglob("*.yaml")) + list(root.rglob("*.yml"))):
            if path.name.startswith("_"):
                continue
            try:
                data = yaml.safe_load(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            tasks_raw: list[Any]
            if isinstance(data, list):
                tasks_raw = data
            elif isinstance(data, dict) and isinstance(data.get("tasks"), list):
                tasks_raw = data["tasks"]
            else:
                continue
            for raw in tasks_raw:
                if not isinstance(raw, dict):
                    continue
                if str(raw.get("name", "")).strip() != task_name:
                    continue
                rubric_items = _extract_rubric_from_task(raw)
                prompt = str(raw.get("prompt", "")).strip() or None
                return path, prompt, rubric_items
    return None, None, []


# ---------------------------------------------------------------------------
# Bedrock judge
# ---------------------------------------------------------------------------


def _parse_judge_verdict(text: str) -> tuple[str, str]:
    """Pull ``(verdict, explanation)`` out of a judge response.

    Mirrors ``conduit.grading.judge_backends.extract_verdict_from_response``
    \u2014 the upstream is the source of truth; this is a self-contained copy so
    we don't import the conduit package.
    """
    text_stripped = text.strip()
    m = re.search(r'\{[^{}]*"criterion_status"[^{}]*\}', text_stripped)
    if m:
        try:
            data = json.loads(m.group())
            status = str(data.get("criterion_status", "")).upper()
            if status in ("MET", "UNMET"):
                return status, str(data.get("explanation") or data.get("reason", ""))
        except (json.JSONDecodeError, TypeError):
            pass
    sm = re.search(
        r'"criterion_status"\s*:\s*"?\s*(MET|UNMET)\b', text_stripped, re.IGNORECASE,
    )
    if sm:
        verdict = "MET" if sm.group(1).upper() == "MET" else "UNMET"
        em = re.search(
            r'"(?:explanation|reason|reasoning)"\s*:\s*"(.*?)"\s*[},]',
            text_stripped,
            re.DOTALL,
        )
        return verdict, (em.group(1) if em else text_stripped)
    try:
        data = json.loads(text_stripped)
        if isinstance(data, dict):
            status = str(
                data.get("criterion_status") or data.get("status") or data.get("verdict") or ""
            ).upper()
            verdict = "MET" if status == "MET" else "UNMET"
            explanation = data.get("explanation") or data.get("reason") or data.get("reasoning", "")
            return verdict, str(explanation)
    except (json.JSONDecodeError, TypeError):
        pass
    verdict = "UNMET"
    expl: list[str] = []
    for line in text_stripped.splitlines():
        u = line.strip().upper()
        if u.startswith("VERDICT:"):
            verdict = "MET" if ("MET" in u and "UNMET" not in u) else "UNMET"
        else:
            expl.append(line)
    return verdict, "\n".join(expl).strip()


def _build_user_prompt(
    requirement: str, weight: float, query: str | None, answer: str
) -> str:
    """Reproduce the user prompt that ``PerCriterionGrader`` sends.

    Negative weights flip the criterion_type tag, exactly matching the upstream
    template.
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

<response>
{answer}
</response>"""


def grade_criterion_bedrock(
    bedrock_client: Any,
    model_id: str,
    requirement: str,
    weight: float,
    query: str | None,
    answer: str,
    *,
    max_retries: int = 3,
) -> tuple[str, str]:
    """Call Bedrock once for a single criterion. Returns ``(verdict, raw_text)``.

    Raises ``RuntimeError`` after ``max_retries`` if Bedrock keeps erroring \u2014
    callers wrap the whole task in a try/except and tag it ``bedrock_error``.
    """
    user_prompt = _build_user_prompt(requirement, weight, query, answer)
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
                body=json.dumps(body).encode("utf-8"),
                contentType="application/json",
                accept="application/json",
            )
            payload = json.loads(response["body"].read())
            # Anthropic-on-Bedrock returns ``content: [{type: text, text: ...}, ...]``.
            text_parts: list[str] = []
            for part in payload.get("content", []) or []:
                if isinstance(part, dict) and part.get("type") == "text":
                    t = part.get("text")
                    if isinstance(t, str):
                        text_parts.append(t)
            raw = "\n".join(text_parts).strip() or json.dumps(payload)
            verdict, _ = _parse_judge_verdict(raw)
            return verdict, raw
        except Exception as exc:  # noqa: BLE001 \u2014 boto exceptions are not enumerable
            last_exc = exc
            if attempt < max_retries - 1:
                # Linear backoff; Bedrock throttling is rarely sustained for
                # this workload (~50 calls per re-grade).
                time.sleep(2 ** attempt)
    raise RuntimeError(f"Bedrock grading failed after {max_retries} attempts: {last_exc}")


def _aggregate_score(reports: list[dict[str, Any]]) -> tuple[float, int]:
    """Weighted-mean over positive criteria, matching conduit's normalization.

    Returns ``(score, n_met)``. If all weights are <= 0, returns ``(0.0, n_met)``
    (degenerate; the held-out tasks all use positive weights).
    """
    pos_weight = sum(r["weight"] for r in reports if r["weight"] > 0)
    n_met = sum(1 for r in reports if r["verdict"] == "MET")
    if pos_weight <= 0:
        return 0.0, n_met
    weighted = sum(
        r["weight"] for r in reports if r["weight"] > 0 and r["verdict"] == "MET"
    )
    return max(0.0, min(1.0, weighted / pos_weight)), n_met


# ---------------------------------------------------------------------------
# Trajectory walking
# ---------------------------------------------------------------------------


def _load_orig_scores(summary_path: Path) -> dict[str, float]:
    out: dict[str, float] = {}
    if not summary_path.exists():
        return out
    for line in summary_path.open():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        task = row.get("task")
        score = row.get("score")
        if isinstance(task, str) and isinstance(score, (int, float)):
            out[task] = float(score)
    return out


def regrade(
    backend: str,
    adapter: str,
    *,
    results_root: Path = DEFAULT_RESULTS_ROOT,
    task_yaml_roots: list[Path] | None = None,
    bedrock_model: str = DEFAULT_BEDROCK_MODEL,
    dry_run: bool = False,
    max_tasks: int | None = None,
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

    n_done = 0
    n_skipped_no_answer = 0
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

        declared, source = extract_declared_answer(trajectory)
        yaml_path, prompt, rubric_items = load_task_rubric(task_name, task_yaml_roots)
        orig = orig_scores.get(task_name)

        row: dict[str, Any] = {
            "task": task_name,
            "score_strict": 0.0,
            "score": orig,
            "declared_answer": declared,
            "declared_answer_source": source,
            "n_criteria": len(rubric_items),
            "n_met": 0,
            "failure_mode": "graded",
            "task_yaml": str(yaml_path) if yaml_path else None,
            "trajectory_path": str(traj_path),
            "run_timestamp": latest.name,
            "rubric_report": [],
        }

        if declared is None:
            row["failure_mode"] = "no_answer_declared"
            row["score_strict"] = 0.0
            n_skipped_no_answer += 1
            print(f"  {task_name}: no answer declared -> 0.00")
        elif not rubric_items or yaml_path is None:
            row["failure_mode"] = "task_yaml_missing"
            row["score_strict"] = 0.0
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
                        verdict, raw = grade_criterion_bedrock(
                            bedrock_client,
                            bedrock_model,
                            item["requirement"],
                            item["weight"],
                            prompt,
                            declared,
                        )
                    reports.append({
                        "requirement": item["requirement"],
                        "weight": item["weight"],
                        "verdict": verdict,
                        "reason": raw[:1000],
                    })
                score, n_met = _aggregate_score(reports)
                row["score_strict"] = round(score, 3)
                row["n_met"] = n_met
                row["rubric_report"] = reports
                n_graded += 1
                print(
                    f"  {task_name}: graded {n_met}/{len(reports)} MET -> "
                    f"{score:.2f} (orig={orig if orig is not None else '?'})"
                )
            except Exception as exc:  # noqa: BLE001
                row["failure_mode"] = "bedrock_error"
                row["score_strict"] = 0.0
                row["error"] = str(exc)
                n_errors += 1
                print(f"  {task_name}: bedrock error: {exc}", file=sys.stderr)

        rows.append(row)
        n_done += 1

    out_path = base / "_summary_strict.jsonl"
    with out_path.open("w", encoding="utf-8") as h:
        for row in rows:
            h.write(json.dumps(row, ensure_ascii=False) + "\n")

    print()
    print(f"Wrote {out_path} ({len(rows)} rows)")
    if rows:
        strict = [r["score_strict"] for r in rows]
        orig = [r["score"] for r in rows if isinstance(r["score"], (int, float))]
        print(f"  strict mean = {statistics.mean(strict):.3f}  (n={len(strict)})")
        if orig:
            print(f"  orig mean   = {statistics.mean(orig):.3f}  (n={len(orig)})")
            print(f"  delta       = {statistics.mean(strict) - statistics.mean(orig):+.3f}")
    print(f"  graded            : {n_graded}")
    print(f"  no_answer_declared: {n_skipped_no_answer}")
    print(f"  task_yaml_missing : {n_skipped_no_yaml}")
    print(f"  bedrock_error     : {n_errors}")
    if dry_run:
        print()
        print("DRY RUN: scores above are dummy (assumed MET on every criterion).")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Strict re-grade of existing eval trajectories (answer-only, "
                    "no screen-state credit).",
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
    args = parser.parse_args()
    return regrade(
        args.backend,
        args.adapter,
        results_root=args.results_root,
        task_yaml_roots=args.task_yaml_root,
        bedrock_model=args.bedrock_model,
        dry_run=args.dry_run,
        max_tasks=args.max_tasks,
    )


if __name__ == "__main__":
    sys.exit(main())
