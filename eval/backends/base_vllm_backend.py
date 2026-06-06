"""Shared parent class for all four CUA backends.

Each subclass only customises the model name and (optionally) the system
prompt. The protocol matches Dillinger's :class:`conduit.models.base.AgentBackend`
exactly, so the existing :func:`conduit.runtime.computer_loop.run_task` can drive
us without any modification to Dillinger itself.

Wire format: OpenAI-compatible ``POST /v1/chat/completions`` against a
``vllm serve --enable-lora`` instance. The ``model`` field selects the LoRA:

    * ``CUA_LORA_ADAPTER=baseline``  → use the bare base model (no adapter)
    * ``CUA_LORA_ADAPTER=cua``       → use the loaded ``cua`` adapter

The output text is expected to be a JSON object::

    {
      "reasoning": "...",
      "action": {"function_name": "click", "arguments": {"x": 412, "y": 88}}
    }

Defensive parsing: if the model emits prose or markdown around the JSON, we
extract the first balanced ``{...}`` block. If extraction still fails we return
a ``wait`` no-op and increment ``self.parse_failures`` for eval reporting.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

import httpx

from conduit.config import Settings
from conduit.models.base import AgentBackend, AgentStep, extract_usage_metrics, to_trace_payload
from conduit.runtime.actions import BrowserAction

logger = logging.getLogger(__name__)


def _extract_first_json(text: str) -> dict[str, Any] | None:
    """Return the first JSON object found in *text*, or None.

    Strategy:
      1. Try ``json.loads(text.strip())`` for the easy case.
      2. Find the first ``{`` and walk a brace counter to find a balanced match.
    """
    s = text.strip()
    if not s:
        return None
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = re.sub(r"\s*```\s*$", "", s)
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    start = s.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(s)):
            ch = s[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = s[start : i + 1]
                    try:
                        obj = json.loads(candidate)
                        if isinstance(obj, dict):
                            return obj
                    except json.JSONDecodeError:
                        break
        start = s.find("{", start + 1)
    return None


def _data_url_from_screenshot_ref(screenshot_ref: str) -> str:
    """Normalize Dillinger's screenshot ref to a base64 data URL.

    Dillinger passes either:
      * a ``data:image/...;base64,...`` URL, or
      * a raw base64 string (PNG bytes).
    Both end up as a data URL we can stuff into an OpenAI vision message.
    """
    if screenshot_ref.startswith("data:"):
        return screenshot_ref
    return f"data:image/png;base64,{screenshot_ref}"


def _action_from_payload(payload: dict[str, Any], call_id: str) -> BrowserAction | None:
    """Build a :class:`BrowserAction` from the model's ``action`` dict."""
    fn = payload.get("function_name") or payload.get("type") or ""
    args = payload.get("arguments") or {}
    if not fn:
        return None
    # Extra defensive: tolerate keys that arrived flattened at the top level
    if not args and any(k in payload for k in ("x", "y", "text", "keys", "scroll_x", "scroll_y")):
        args = {k: v for k, v in payload.items() if k not in ("function_name", "type")}

    def _to_int(v: Any) -> int | None:
        if isinstance(v, (int, float)):
            return int(v)
        try:
            return int(str(v))
        except (ValueError, TypeError):
            return None

    return BrowserAction(
        type=str(fn),
        x=_to_int(args.get("x")),
        y=_to_int(args.get("y")),
        end_x=_to_int(args.get("end_x")),
        end_y=_to_int(args.get("end_y")),
        text=args.get("text"),
        keys=args.get("keys"),
        url=args.get("url"),
        scroll_x=_to_int(args.get("scroll_x")),
        scroll_y=_to_int(args.get("scroll_y")),
        button=args.get("button"),
        status=args.get("status"),
        result=args.get("result"),
        modifier=args.get("modifier"),
        duration=args.get("duration"),
        source="normalized_completion",
        metadata={"call_id": call_id},
    )


class BaseVLLMBackend(AgentBackend):
    """Parent class — set ``model_name`` and (optionally) ``system_prompt`` in subclasses."""

    name: str = "base_vllm_cua"
    model_name: str = "base"            # subclass override
    base_url_env: str = "VLLM_BASE_URL"
    default_base_url: str = "http://localhost:8000/v1"
    system_prompt: str = (
        "You are a computer-use agent. You see a browser screenshot and a task. "
        "Output a single JSON object with two keys: \"reasoning\" (a short string) "
        "and \"action\" (a tool call with \"function_name\" and \"arguments\"). "
        "Available action function_names: click, double_click, triple_click, right_click, "
        "scroll, type, keypress, drag, mouse_move, mouse_down, mouse_up, hold_key, wait, "
        "zoom, terminate, answer."
    )
    # Cap multimodal context so we don't blow vLLM's max_model_len
    max_history_items: int = 12
    request_timeout_seconds: float = 90.0
    max_completion_tokens: int = 1024

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.base_url = os.environ.get(self.base_url_env, self.default_base_url).rstrip("/")
        # CUA_LORA_ADAPTER selects which LoRA adapter the vLLM server uses.
        # vLLM's --enable-lora maps adapter names → model field in chat completions.
        adapter = os.environ.get("CUA_LORA_ADAPTER", "baseline").strip()
        self.adapter = adapter
        if adapter and adapter != "baseline":
            self.served_model_name = adapter  # e.g. "cua"
        else:
            self.served_model_name = self.model_name
        self.client = httpx.Client(timeout=self.request_timeout_seconds)
        self.history: list[dict[str, Any]] = []
        self.parse_failures = 0
        self._call_counter = 0
        logger.info(
            "%s init: base_url=%s served_model=%s adapter=%s",
            self.__class__.__name__, self.base_url, self.served_model_name, self.adapter,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _new_call_id(self) -> str:
        self._call_counter += 1
        return f"call_{self._call_counter:04d}"

    def _summarize_history(self) -> str:
        if not self.history:
            return "(no prior actions)"
        lines: list[str] = []
        for i, a in enumerate(self.history[-self.max_history_items :], start=1):
            fn = a.get("function_name", "?")
            args = a.get("arguments") or {}
            if fn in {"click", "double_click", "triple_click", "right_click"}:
                lines.append(f"{i}. {fn}({args.get('x')}, {args.get('y')})")
            elif fn == "type":
                t = (args.get("text") or "").replace("\n", "\\n")
                if len(t) > 40:
                    t = t[:37] + "..."
                lines.append(f"{i}. type({t!r})")
            elif fn == "scroll":
                lines.append(f"{i}. scroll(dx={args.get('scroll_x', 0)}, dy={args.get('scroll_y', 0)})")
            elif fn == "keypress":
                keys = args.get("keys") or []
                lines.append(f"{i}. keypress({'+'.join(keys) if keys else ''})")
            else:
                lines.append(f"{i}. {fn}({args})")
        return "\n".join(lines)

    def _build_payload(self, instruction: str, screenshot_ref: str) -> dict[str, Any]:
        history_text = self._summarize_history()
        user_text = (
            f"Task: {instruction}\n\n"
            f"Step history (most recent first):\n{history_text}\n\n"
            "What is the next action? Respond with the JSON object."
        )
        return {
            "model": self.served_model_name,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": _data_url_from_screenshot_ref(screenshot_ref)}},
                        {"type": "text", "text": user_text},
                    ],
                },
            ],
            "max_tokens": self.max_completion_tokens,
            "temperature": 0.0,
        }

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.base_url}/chat/completions"
        response = self.client.post(url, json=payload)
        response.raise_for_status()
        return response.json()

    def _build_step(self, response: dict[str, Any], request_payload: dict[str, Any]) -> AgentStep:
        content = ""
        try:
            content = response["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError):
            content = ""
        parsed = _extract_first_json(content)
        action: BrowserAction | None = None
        reasoning_content: str | None = None
        message_text: str | None = None
        call_id = self._new_call_id()
        if parsed is None:
            self.parse_failures += 1
            logger.warning("%s: failed to parse JSON, falling back to wait. raw=%r",
                           self.__class__.__name__, content[:300])
            action = BrowserAction(type="wait", source="normalized_completion", metadata={"call_id": call_id})
            message_text = content
        else:
            reasoning_content = (parsed.get("reasoning") or None) if isinstance(parsed.get("reasoning"), str) else None
            action_payload = parsed.get("action") if isinstance(parsed.get("action"), dict) else parsed
            action = _action_from_payload(action_payload, call_id) or BrowserAction(
                type="wait", source="normalized_completion", metadata={"call_id": call_id}
            )
            self.history.append({
                "function_name": action.type,
                "arguments": {
                    "x": action.x, "y": action.y, "end_x": action.end_x, "end_y": action.end_y,
                    "text": action.text, "keys": action.keys, "scroll_x": action.scroll_x,
                    "scroll_y": action.scroll_y, "button": action.button,
                },
            })

        usage = response.get("usage") if isinstance(response.get("usage"), dict) else None
        return AgentStep(
            action=action,
            message=message_text,
            response_id=response.get("id") if isinstance(response, dict) else None,
            model_name=self.model_name,
            reasoning_content=reasoning_content,
            request_payload=request_payload,
            response_payload=to_trace_payload(response),
            metrics=extract_usage_metrics(usage),
            extra={"adapter": self.adapter, "served_model_name": self.served_model_name,
                   "parse_failures_so_far": self.parse_failures},
            raw=response,
        )

    # ------------------------------------------------------------------
    # AgentBackend interface
    # ------------------------------------------------------------------

    def create_initial_step(self, instruction: str, screenshot_ref: str) -> AgentStep:
        self.history = []
        self.parse_failures = 0
        request_payload = self._build_payload(instruction, screenshot_ref)
        response = self._post(request_payload)
        return self._build_step(response, request_payload)

    def create_follow_up_step(
        self,
        previous_step: AgentStep,
        screenshot_ref: str,
        extra_message: str | None = None,
    ) -> AgentStep:
        # We re-encode the *current* screenshot only; history is text. (See plan: cap to 1
        # screenshot per turn so we don't blow context length on small VLMs.)
        instruction = (previous_step.request_payload or {}).get("messages", [{}])[1].get("content", [{}])
        # Recover the original instruction by looking at the very first text part we sent.
        original_instruction = ""
        for m in (previous_step.request_payload or {}).get("messages", []):
            if m.get("role") != "user":
                continue
            content = m.get("content")
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        text = part.get("text", "")
                        first = text.split("\n", 1)[0]
                        if first.startswith("Task:"):
                            original_instruction = first[len("Task:") :].strip()
                            break
            if original_instruction:
                break
        if extra_message:
            original_instruction = f"{original_instruction}\n\n{extra_message}".strip()

        request_payload = self._build_payload(original_instruction, screenshot_ref)
        response = self._post(request_payload)
        return self._build_step(response, request_payload)

    def extract_text(self, prompt: str, screenshot_ref: str) -> str:
        # Plain VQA-style call: no system prompt nudging toward JSON, so models give
        # a free-text answer that the rubric grader can read.
        payload = {
            "model": self.served_model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url",
                         "image_url": {"url": _data_url_from_screenshot_ref(screenshot_ref)}},
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            "max_tokens": self.max_completion_tokens,
            "temperature": 0.0,
        }
        try:
            response = self._post(payload)
            return response["choices"][0]["message"]["content"] or ""
        except (httpx.HTTPError, KeyError, IndexError) as exc:
            logger.warning("%s.extract_text failed: %s", self.__class__.__name__, exc)
            return ""

    def __del__(self) -> None:
        try:
            self.client.close()
        except Exception:  # noqa: BLE001
            pass
