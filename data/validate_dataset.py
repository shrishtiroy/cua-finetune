"""Phase 1c: validate the cua_sft dataset and smoke-test template rendering.

Spot-checks performed:

  1. Load 5 random train + 5 random test examples.
  2. Confirm all ``images`` paths exist and decode as JPEG/PNG via Pillow.
  3. Confirm every assistant message parses as JSON with a recognized
     ``action.function_name``.
  4. For each of the four template families we'll train on
     (``qwen3_vl``, ``mllama``, ``kimi_vl``, ``deepseek_vl2``) print what the
     prompt would look like after the chat-template + image-token substitution.
     We don't need ms-swift installed; we just emit the conceptual layout based
     on each model's HuggingFace card.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN = REPO_ROOT / "data" / "cua_sft" / "train.jsonl"
DEFAULT_TEST = REPO_ROOT / "data" / "cua_sft" / "test.jsonl"

RECOGNIZED_FUNCTIONS = {
    "click", "left_click", "double_click", "right_click", "triple_click",
    "type", "keypress", "scroll", "wait", "screenshot",
    "drag", "mouse_move", "mouse_down", "mouse_up", "hold_key",
    "terminate", "answer", "done", "zoom",
}


def render_qwen3_vl(messages: list[dict[str, Any]]) -> str:
    """Qwen3-VL uses ChatML-style with <|vision_start|><|image_pad|><|vision_end|>."""
    parts: list[str] = []
    for m in messages:
        content = m["content"].replace("<image>", "<|vision_start|><|image_pad|><|vision_end|>") \
            if m["role"] == "user" else m["content"]
        parts.append(f"<|im_start|>{m['role']}\n{content}<|im_end|>")
    parts.append("<|im_start|>assistant\n")
    return "\n".join(parts)


def render_mllama(messages: list[dict[str, Any]]) -> str:
    """Llama-3.2-Vision (mllama) — Llama-3.x header tokens + <|image|>."""
    parts: list[str] = ["<|begin_of_text|>"]
    for m in messages:
        content = m["content"].replace("<image>", "<|image|>") if m["role"] == "user" else m["content"]
        parts.append(f"<|start_header_id|>{m['role']}<|end_header_id|>\n\n{content}<|eot_id|>")
    parts.append("<|start_header_id|>assistant<|end_header_id|>\n\n")
    return "".join(parts)


def render_kimi_vl(messages: list[dict[str, Any]]) -> str:
    """Kimi-VL-A3B — ChatML wrapper with <|media_*|> markers around the image."""
    parts: list[str] = []
    for m in messages:
        content = (
            m["content"].replace("<image>", "<|media_start|>image<|media_content|><|media_pad|><|media_end|>")
            if m["role"] == "user" else m["content"]
        )
        parts.append(f"<|im_start|>{m['role']}\n{content}<|im_end|>")
    parts.append("<|im_start|>assistant\n")
    return "\n".join(parts)


def render_deepseek_vl2(messages: list[dict[str, Any]]) -> str:
    """DeepSeek-VL2-Small — DeepSeek prompt format with role labels + <image>."""
    parts: list[str] = ["<｜begin▁of▁sentence｜>"]
    role_label = {"system": "System", "user": "User", "assistant": "Assistant"}
    for m in messages:
        content = m["content"].replace("<image>", "<image>\n") if m["role"] == "user" else m["content"]
        parts.append(f"{role_label.get(m['role'], m['role'])}: {content}\n\n")
    parts.append("Assistant:")
    return "".join(parts)


TEMPLATES = {
    "qwen3_vl": render_qwen3_vl,
    "mllama": render_mllama,
    "kimi_vl": render_kimi_vl,
    "deepseek_vl2": render_deepseek_vl2,
}


def check_image(path: str) -> tuple[bool, str | None]:
    p = Path(path)
    if not p.exists():
        return False, "missing"
    try:
        with Image.open(p) as img:
            img.verify()
    except Exception as exc:  # noqa: BLE001
        return False, f"PIL_verify: {exc}"
    return True, None


def check_assistant(content: str) -> tuple[bool, str | None]:
    try:
        obj = json.loads(content)
    except json.JSONDecodeError as exc:
        return False, f"json: {exc}"
    if not isinstance(obj, dict):
        return False, "assistant content is not an object"
    action = obj.get("action")
    if not isinstance(action, dict):
        return False, "missing 'action' object"
    fn = action.get("function_name")
    if fn not in RECOGNIZED_FUNCTIONS:
        return False, f"unrecognized function_name: {fn!r}"
    if "arguments" not in action:
        return False, "missing action.arguments"
    return True, None


def _load_some(path: Path, n: int, rng: random.Random) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        return []
    sample_idx = rng.sample(range(len(lines)), min(n, len(lines)))
    return [json.loads(lines[i]) for i in sample_idx]


def validate_examples(examples: list[dict[str, Any]], label: str) -> tuple[int, int, int]:
    bad_image = bad_assistant = ok = 0
    for ex in examples:
        local_ok = True
        for img in ex.get("images") or []:
            ok_img, err = check_image(img)
            if not ok_img:
                bad_image += 1
                local_ok = False
                print(f"  [{label} bad image] {img}: {err}")
                break
        assistant = next((m for m in ex["messages"] if m["role"] == "assistant"), None)
        if assistant is not None:
            ok_a, err = check_assistant(assistant["content"])
            if not ok_a:
                bad_assistant += 1
                local_ok = False
                print(f"  [{label} bad assistant] {err}")
        if local_ok:
            ok += 1
    return ok, bad_image, bad_assistant


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate cua_sft dataset + template smoke-test")
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--test", type=Path, default=DEFAULT_TEST)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    train_ex = _load_some(args.train, args.samples, rng)
    test_ex = _load_some(args.test, args.samples, rng)

    print("=" * 72)
    print("DATASET VALIDATION")
    print("=" * 72)
    if not train_ex:
        print(f"ERROR: no train examples in {args.train}")
        return 1
    if not test_ex:
        print(f"WARNING: no test examples in {args.test}")

    train_ok, train_bad_img, train_bad_a = validate_examples(train_ex, "train")
    test_ok, test_bad_img, test_bad_a = validate_examples(test_ex, "test")
    print()
    print(f"Train sample: ok={train_ok}/{len(train_ex)} bad_image={train_bad_img} bad_assistant={train_bad_a}")
    print(f"Test  sample: ok={test_ok}/{len(test_ex)} bad_image={test_bad_img} bad_assistant={test_bad_a}")

    print()
    print("=" * 72)
    print("TEMPLATE RENDERING SMOKE-TEST (first train example)")
    print("=" * 72)
    first = train_ex[0]
    for tname, fn in TEMPLATES.items():
        rendered = fn(first["messages"])
        preview = rendered if len(rendered) <= 800 else rendered[:400] + "\n... [TRUNCATED] ...\n" + rendered[-200:]
        print(f"\n--- template: {tname} ({len(rendered)} chars) ---")
        print(preview)

    print()
    bad_total = train_bad_img + train_bad_a + test_bad_img + test_bad_a
    if bad_total == 0:
        print("VALIDATED — all sampled examples passed image + JSON checks.")
        return 0
    print(f"VALIDATION FAILED — {bad_total} issues across sampled examples", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
