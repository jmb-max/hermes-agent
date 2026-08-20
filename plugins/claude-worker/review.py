"""Terra advisory review for claude_worker (requirement 10).

Reuses the existing auxiliary-task system
(``agent.auxiliary_client.call_llm``) rather than building a bespoke review
path or registering a new provider. ``claude_worker_review`` is declared via
``ctx.register_auxiliary_task`` in ``__init__.py`` with defaults pointing at
Terra (``openai-codex`` / ``gpt-5.6-terra``); a user can still override
``auxiliary.claude_worker_review.*`` in config.yaml, same as any other
auxiliary task.

Advisory only: a review failure (auxiliary model down, timeout, bad
response shape) is captured and returned as ``{"reviewed": False, "error":
...}`` — it never raises, and the caller (``runner.py``) never lets it
affect the claude_worker tool result's own success/failure.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

AUX_TASK_KEY = "claude_worker_review"
AUX_TASK_DEFAULTS: Dict[str, Any] = {
    "provider": "openai-codex",
    "model": "gpt-5.6-terra",
    "timeout": 60,
}

_REVIEW_SYSTEM_PROMPT = (
    "You are Terra, an advisory secondary reviewer for changes made by an "
    "autonomous coding worker. Be concise (a few sentences or a short "
    "bullet list). Flag real risks (correctness, security, scope creep); "
    "do not restate the diff."
)

_FALLBACK_SYSTEM_PROMPT = (
    "You are Terra, standing in for a coding worker that is currently "
    "unavailable. Give concrete, self-contained implementation guidance the "
    "caller can apply by hand: what to change, where, and what to verify. "
    "Be specific and concise; do not ask clarifying questions."
)


def should_review(files_touched: Optional[List[str]], min_changed_files: int) -> bool:
    """Return True when a change is substantial enough to warrant review."""
    count = len(files_touched or [])
    if count == 0:
        return False
    threshold = max(1, min_changed_files)
    return count >= threshold


def _call_llm(task: str = None, *, messages: list, **kwargs: Any) -> Any:
    from agent.auxiliary_client import call_llm

    return call_llm(task=task, messages=messages, **kwargs)


def run_review(task: str, files_touched: List[str], summary: str) -> Dict[str, Any]:
    """Run the Terra advisory review. Never raises."""
    try:
        messages = [
            {"role": "system", "content": _REVIEW_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Task: {task}\n"
                    f"Files touched ({len(files_touched)}): {', '.join(files_touched)}\n"
                    f"Worker summary: {summary}"
                ),
            },
        ]
        response = _call_llm(task=AUX_TASK_KEY, messages=messages, max_tokens=800, timeout=45)
        content = response.choices[0].message.content
        return {"reviewed": True, "notes": content}
    except Exception as exc:
        logger.warning("claude_worker review failed (non-fatal, advisory only): %s", exc)
        return {"reviewed": False, "error": str(exc)}


def run_fallback(task: str, reason: str = "") -> Dict[str, Any]:
    """Ask Terra to stand in for an unavailable worker. Never raises.

    Returns ``{"ok": True, "notes": <non-empty text>}`` ONLY when Terra
    actually came back with content. An exception, a malformed response, or
    an empty/whitespace answer returns ``{"ok": False, "error": ...}`` —
    the gate treats anything but a real result as a continued HOLD, so a
    silent Terra failure can never become an unlock.
    """
    try:
        messages = [
            {"role": "system", "content": _FALLBACK_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"The coding worker is unavailable ({reason or 'unspecified'}).\n"
                    f"Task: {task}"
                ),
            },
        ]
        response = _call_llm(task=AUX_TASK_KEY, messages=messages, max_tokens=1200, timeout=60)
        content = response.choices[0].message.content
        if not isinstance(content, str) or not content.strip():
            return {"ok": False, "error": "terra returned an empty result"}
        return {"ok": True, "notes": content}
    except Exception as exc:
        logger.warning("claude_worker Terra fallback failed: %s", exc)
        return {"ok": False, "error": str(exc)}
