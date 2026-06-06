"""Phase 2 / 3a: offline action-accuracy eval.

Reads the same ``data/cua_sft/test.jsonl`` that ms-swift consumes and asks the
*served* model (via vLLM ``/v1/chat/completions``) to predict each step's
action. Compares to the ground-truth assistant payload and reports::

  * function_name exact-match rate (per type and overall)
  * for click/double_click/right_click/triple_click: ``model click within 50px``
  * for type: text exact-match (after .strip())
  * for keypress: key-set exact match (order-insensitive)
  * for scroll: sign of scroll_x and scroll_y (left/right/up/down)

This is the cheap signal we use during training (every ``eval_steps`` steps) to
pick the best LoRA checkpoint. Loss alone is misleading for action prediction
because models can learn to mimic our JSON envelope while picking the wrong
coordinates.

Usage::

    # On Lambda, after vLLM is serving the trained adapter:
    CUA_LORA_ADAPTER=cua python eval/action_accuracy.py --backend qwen_vl_cua --max-samples 200
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from eval.backends import get_backend_class

logger = logging.getLogger("action_accuracy")

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEST = REPO_ROOT / "data" / "cua_sft" / "test.jsonl"

CLICK_FNS = {"click", "left_click", "double_click", "triple_click", "right_click", "middle_click"}


def _load_examples(path: Path, max_samples: int | None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as h:
        for line in h:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
            if max_samples and len(out) >= max_samples:
                break
    return out


def _gold_action(example: dict[str, Any]) -> dict[str, Any] | None:
    for m in example.get("messages", []):
        if m.get("role") == "assistant":
            try:
                obj = json.loads(m.get("content") or "")
            except json.JSONDecodeError:
                return None
            return obj.get("action") if isinstance(obj, dict) else None
    return None


def _click_within(gold: dict[str, Any], pred: dict[str, Any], tolerance: int) -> bool:
    try:
        gx, gy = int(gold.get("x")), int(gold.get("y"))
        px, py = int(pred.get("x")), int(pred.get("y"))
    except (TypeError, ValueError):
        return False
    return ((gx - px) ** 2 + (gy - py) ** 2) <= tolerance ** 2


def _scroll_dir_match(gold: dict[str, Any], pred: dict[str, Any]) -> bool:
    def _sign(v: Any) -> int:
        try:
            iv = int(v or 0)
        except (TypeError, ValueError):
            return 0
        return (iv > 0) - (iv < 0)
    return (
        _sign(gold.get("scroll_x")) == _sign(pred.get("scroll_x"))
        and _sign(gold.get("scroll_y")) == _sign(pred.get("scroll_y"))
    )


def _b64_screenshot(image_path: str) -> str:
    import base64
    with open(image_path, "rb") as h:
        return base64.b64encode(h.read()).decode("ascii")


def main() -> int:
    parser = argparse.ArgumentParser(description="Offline action-accuracy eval over test.jsonl")
    parser.add_argument("--backend", required=True,
                        choices=["qwen_vl_cua", "llama_vision_cua", "kimi_vl_cua", "deepseek_vl_cua"])
    parser.add_argument("--test", type=Path, default=DEFAULT_TEST)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--click-tolerance-px", type=int, default=50)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        level=logging.DEBUG if args.verbose else logging.INFO,
    )
    if not args.test.exists():
        print(f"ERROR: test set missing: {args.test}", file=sys.stderr)
        return 1

    # Settings + backend (we don't need a full Dillinger Settings object for this offline eval;
    # we instantiate just enough to reuse the request-construction code in the backend).
    from conduit.config import Settings
    settings = Settings(_env_file=None)  # picks up env vars / defaults
    backend_cls = get_backend_class(args.backend)
    backend = backend_cls(settings)

    examples = _load_examples(args.test, args.max_samples)
    logger.info("Loaded %d test examples", len(examples))

    fn_counts: dict[str, int] = defaultdict(int)
    fn_correct: dict[str, int] = defaultdict(int)
    click_within: dict[int, int] = {args.click_tolerance_px: 0, 100: 0, 200: 0}
    type_match = 0
    keypress_match = 0
    scroll_match = 0
    parse_failures = 0
    total = 0

    for i, ex in enumerate(examples):
        gold = _gold_action(ex)
        if gold is None:
            continue
        gold_fn = str(gold.get("function_name", ""))
        images = ex.get("images") or []
        if not images:
            continue
        try:
            screenshot_ref = _b64_screenshot(images[0])
        except OSError as exc:
            logger.warning("skip %s: %s", images[0], exc)
            continue

        # Find the original task text from the user message
        task_text = ""
        for m in ex.get("messages", []):
            if m.get("role") == "user":
                content = m.get("content") or ""
                if isinstance(content, str):
                    # Strip the leading "<image>\n\n" we baked in
                    task_text = content.replace("<image>", "").strip()
                break

        # Reset per-example state on the backend
        backend.history = []
        try:
            step = backend.create_initial_step(task_text or "Perform the next action.", screenshot_ref)
        except Exception as exc:  # noqa: BLE001
            logger.warning("backend call failed: %s", exc)
            continue

        action = step.action
        total += 1
        fn_counts[gold_fn] += 1
        if action is None or action.type == "wait" and step.message:
            parse_failures += 1
            continue

        pred_fn = action.type
        pred_args = {
            "x": action.x, "y": action.y, "end_x": action.end_x, "end_y": action.end_y,
            "text": action.text, "keys": action.keys, "scroll_x": action.scroll_x,
            "scroll_y": action.scroll_y, "button": action.button,
        }
        gold_args = gold.get("arguments", {}) or {}

        if pred_fn == gold_fn or (gold_fn in CLICK_FNS and pred_fn in CLICK_FNS):
            fn_correct[gold_fn] += 1
            if gold_fn in CLICK_FNS:
                for tol in click_within:
                    if _click_within(gold_args, pred_args, tol):
                        click_within[tol] += 1
                        break
            elif gold_fn == "type":
                if (gold_args.get("text") or "").strip() == (pred_args.get("text") or "").strip():
                    type_match += 1
            elif gold_fn == "keypress":
                if set(gold_args.get("keys") or []) == set(pred_args.get("keys") or []):
                    keypress_match += 1
            elif gold_fn == "scroll":
                if _scroll_dir_match(gold_args, pred_args):
                    scroll_match += 1
        if (i + 1) % 25 == 0:
            logger.info("Processed %d/%d examples", i + 1, len(examples))

    overall_correct = sum(fn_correct.values())
    summary = {
        "backend": args.backend,
        "adapter": backend.adapter,
        "total_evaluated": total,
        "function_name_accuracy": (overall_correct / total) if total else 0.0,
        "per_fn_total": dict(fn_counts),
        "per_fn_correct": dict(fn_correct),
        "click_within_50px": click_within[50],
        "click_within_100px": click_within[100],
        "click_within_200px": click_within[200],
        "type_text_exact_match": type_match,
        "keypress_keyset_match": keypress_match,
        "scroll_direction_match": scroll_match,
        "parse_failures": parse_failures + backend.parse_failures,
    }
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
