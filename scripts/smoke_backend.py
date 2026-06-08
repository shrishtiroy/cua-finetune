"""Phase 3d: cheap end-to-end backend smoke.

Before kicking off a $22 / 3-hour baseline run, prove that:
  1. vLLM is reachable at $VLLM_BASE_URL.
  2. The backend can build a request payload, POST it, and parse the
     response into a valid BrowserAction (no JSON parse failure).
  3. Token usage looks sane (no runaway prompt or empty output).

Costs roughly $0 (10-20 tokens of inference, no grading judge).

Usage::

    # In a tmux: scripts/serve_vllm.sh Qwen/Qwen3-VL-8B-Instruct  (port 8000)
    python scripts/smoke_backend.py --backend qwen_vl_cua
    python scripts/smoke_backend.py --backend kimi_vl_cua
    python scripts/smoke_backend.py --backend deepseek_vl_cua

Exit code 0 = parse OK; 1 = parse failure; 2 = transport failure.
"""

from __future__ import annotations

import argparse
import base64
import io
import logging
import os
import sys
import time
from pathlib import Path

logger = logging.getLogger("smoke_backend")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def _make_dummy_screenshot() -> str:
    """Return a base64 PNG of a tiny synthetic screenshot.

    Uses Pillow if available; falls back to a hand-crafted 1x1 transparent
    PNG so the smoke works even before Pillow is installed.
    """
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        # Hand-crafted 1x1 transparent PNG (smallest valid PNG).
        return ("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYP"
                "hfDwAChwGA60e6kgAAAABJRU5ErkJggg==")
    img = Image.new("RGB", (320, 240), color=(245, 245, 245))
    draw = ImageDraw.Draw(img)
    draw.rectangle([(40, 60), (280, 100)], fill=(80, 100, 200), outline=(0, 0, 0))
    draw.text((60, 75), "Click me", fill=(255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def main() -> int:
    parser = argparse.ArgumentParser(description="End-to-end backend JSON-parse smoke")
    parser.add_argument("--backend", required=True,
                        choices=["qwen_vl_cua", "kimi_vl_cua", "deepseek_vl_cua", "llama_vision_cua"])
    parser.add_argument("--instruction", default=("Find the blue button labelled 'Click me' "
                                                  "and click it. Output JSON only."))
    parser.add_argument("--adapter", default=os.environ.get("CUA_LORA_ADAPTER", "baseline"),
                        help="LoRA adapter to use. Default $CUA_LORA_ADAPTER or 'baseline'.")
    parser.add_argument("--vllm-url", default=os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8000/v1"))
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        level=logging.DEBUG if args.verbose else logging.INFO,
    )

    os.environ["CUA_LORA_ADAPTER"] = args.adapter
    os.environ["VLLM_BASE_URL"] = args.vllm_url

    # Import lazily so --help works without conduit installed.
    try:
        from eval.backends import get_backend_class
    except ImportError as exc:
        print(f"ERROR: cannot import eval.backends ({exc}). Run from cua-finetune root with venv active.",
              file=sys.stderr)
        return 2

    # Surface vLLM connection error early with a clean message.
    import httpx
    try:
        models_resp = httpx.get(f"{args.vllm_url.rstrip('/')}/models", timeout=5.0)
        models_resp.raise_for_status()
        served = [m["id"] for m in models_resp.json().get("data", [])]
        print(f"vLLM at {args.vllm_url}: {len(served)} model(s) served → {served}")
    except (httpx.HTTPError, httpx.RequestError, KeyError) as exc:
        print(f"ERROR: cannot reach vLLM at {args.vllm_url}: {exc}", file=sys.stderr)
        print(f"Start it in a tmux first:", file=sys.stderr)
        case_to_model = {
            "qwen_vl_cua": "Qwen/Qwen3-VL-8B-Instruct",
            "kimi_vl_cua": "moonshotai/Kimi-VL-A3B-Instruct",
            "deepseek_vl_cua": "deepseek-ai/deepseek-vl2-small",
            "llama_vision_cua": "meta-llama/Llama-3.2-11B-Vision-Instruct",
        }
        print(f"  bash scripts/serve_vllm.sh {case_to_model[args.backend]}", file=sys.stderr)
        return 2

    try:
        from conduit.config import load_settings
    except ImportError as exc:
        print(f"ERROR: cannot import conduit ({exc}). Make sure ~/Dillinger is uv-synced.", file=sys.stderr)
        return 2

    settings = load_settings()
    backend_cls = get_backend_class(args.backend)
    backend = backend_cls(settings)
    print(f"Backend {args.backend}: served_model_name={backend.served_model_name} adapter={backend.adapter}")

    screenshot_b64 = _make_dummy_screenshot()

    t0 = time.time()
    try:
        step = backend.create_initial_step(args.instruction, screenshot_b64)
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: backend raised on initial step: {exc}", file=sys.stderr)
        return 2
    elapsed_ms = (time.time() - t0) * 1000

    action = step.action
    parsed_ok = (action is not None and action.type != "wait" and backend.parse_failures == 0)
    print()
    print(f"=== {args.backend} / {args.adapter} ===")
    print(f"Latency:        {elapsed_ms:.0f} ms")
    print(f"Action type:    {action.type if action else '(none)'}")
    if action and action.x is not None:
        print(f"Action coords:  ({action.x}, {action.y})")
    if step.reasoning_content:
        rc = step.reasoning_content[:120].replace("\n", " ")
        print(f"Reasoning:      {rc}{'...' if len(step.reasoning_content) > 120 else ''}")
    if step.metrics:
        print(f"Tokens:         prompt={step.metrics.prompt_tokens} completion={step.metrics.completion_tokens}")
    print(f"Parse failures: {backend.parse_failures}")

    if parsed_ok:
        print()
        print(f"OK: {args.backend} produced a parseable action.")
        print(f"   Cleared to run: CUA_LIVE_WEB=1 bash scripts/eval_browser_baseline.sh {args.backend} {args.adapter} 1")
        return 0
    else:
        print()
        print(f"FAILURE: backend returned action.type='wait' / parse_failures={backend.parse_failures}")
        print(f"   Raw output (first 300 chars):")
        try:
            content = step.raw["choices"][0]["message"]["content"][:300]
            print(f"   {content!r}")
        except (KeyError, IndexError, TypeError):
            print(f"   (could not read raw content from step.raw)")
        print()
        print("Common fixes:")
        print("  - System prompt may need model-specific tweaking. Edit eval/backends/<model>_cua.py")
        print("  - For Kimi-VL / DeepSeek-VL2: verify --trust-remote-code was passed to vllm serve")
        print("  - Check token budget: max_completion_tokens=1024; if truncated, raise it")
        return 1


if __name__ == "__main__":
    sys.exit(main())
