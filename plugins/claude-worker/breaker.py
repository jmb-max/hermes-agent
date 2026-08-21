"""Circuit breaker for claude_worker (requirement 6).

Classifies a failed spawn's exit code/stderr into ``auth`` / ``rate`` /
``extra_usage`` / ``other``. The first three open the breaker for that class
with a per-class cooldown; ``other`` (a plain task failure) never opens it —
those are handled by the one-shot escalation in ``routing.py`` instead.

State is a small JSON file under ``get_hermes_home()/claude-worker/
breaker.json`` so it survives process restarts and is shared across
concurrent gateway sessions. Writes are atomic (temp file + rename) and
guarded by an in-process lock; cross-process races are tolerated (worst
case: one extra spawn slips through right at the boundary, which is
acceptable — the breaker is a cooldown, not a distributed lock).
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from hermes_constants import get_hermes_home

BREAKER_CLASSES = ("auth", "rate", "extra_usage")

_AUTH_MARKERS = (
    "invalid api key", "not logged in", "claude login", "oauth token",
    "authentication_error", "401", "please run `claude login`",
    "please run /login",
)
_RATE_MARKERS = ("rate limit", "429", "too many requests")
_EXTRA_USAGE_MARKERS = (
    "usage limit", "quota exceeded", "credit balance", "plan limit",
    "usage cap", "exceeded your usage",
)

_lock = threading.Lock()


def _state_path() -> Path:
    return Path(get_hermes_home()) / "claude-worker" / "breaker.json"


def classify_failure(exit_code: Optional[int], stderr: str, stdout: str = "") -> str:
    """Classify a failed spawn into one of the breaker classes or ``other``.

    ``stdout`` is best-effort: ``claude --output-format json`` can exit
    non-zero with an empty ``stderr`` and the real reason in a structured
    stdout body (e.g. ``{"is_error": true, "api_error_status": 401}``) —
    see :func:`_classify_structured_stdout`. That check runs first since it
    is unambiguous when present; malformed/non-JSON stdout is treated as no
    signal and falls through to the existing stderr-marker classification.
    """
    structured = _classify_structured_stdout(stdout)
    if structured is not None:
        return structured
    text = (stderr or "").lower()
    if any(marker in text for marker in _AUTH_MARKERS):
        return "auth"
    if any(marker in text for marker in _RATE_MARKERS):
        return "rate"
    if any(marker in text for marker in _EXTRA_USAGE_MARKERS):
        return "extra_usage"
    return "other"


def _classify_structured_stdout(stdout: str) -> Optional[str]:
    """Best-effort auth classification from ``--output-format json`` stdout.

    Only classifies the unambiguous case — ``is_error`` true with an HTTP
    401 ``api_error_status`` — and never raises on malformed/non-JSON
    input; anything else (bad JSON, missing fields, a different status
    code) returns ``None`` so the caller falls back to stderr markers.
    """
    if not stdout:
        return None
    try:
        parsed = json.loads(stdout)
    except (ValueError, TypeError):
        return None
    if not isinstance(parsed, dict):
        return None
    if parsed.get("is_error") is not True:
        return None
    if parsed.get("api_error_status") == 401:
        return "auth"
    return None


def _read_state() -> Dict[str, Any]:
    path = _state_path()
    try:
        raw = path.read_text(encoding="utf-8")
    except (FileNotFoundError, OSError):
        return {}
    try:
        data = json.loads(raw)
    except ValueError:
        return {}
    return data if isinstance(data, dict) else {}


def _write_state(state: Dict[str, Any]) -> None:
    path = _state_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state), encoding="utf-8")
        tmp.replace(path)
    except OSError:
        pass


def record_failure(
    failure_class: str,
    cooldown_seconds: Dict[str, Any],
    now: Optional[float] = None,
) -> None:
    """Open the breaker for *failure_class* (a no-op for ``other``/unknown)."""
    if failure_class not in BREAKER_CLASSES:
        return
    now = now if now is not None else time.time()
    cooldown = cooldown_seconds.get(failure_class, 3600)
    try:
        cooldown = float(cooldown)
    except (TypeError, ValueError):
        cooldown = 3600.0
    with _lock:
        state = _read_state()
        state[failure_class] = {"opened_at": now, "cooldown_seconds": cooldown}
        _write_state(state)


def open_classes(now: Optional[float] = None) -> List[str]:
    """Return the breaker classes currently open, in class order."""
    now = now if now is not None else time.time()
    with _lock:
        state = _read_state()
    out: List[str] = []
    for cls in BREAKER_CLASSES:
        entry = state.get(cls)
        if not isinstance(entry, dict):
            continue
        opened_at = entry.get("opened_at")
        cooldown = entry.get("cooldown_seconds")
        if not isinstance(opened_at, (int, float)) or not isinstance(cooldown, (int, float)):
            continue
        if now < opened_at + cooldown:
            out.append(cls)
    return out


def is_open(now: Optional[float] = None) -> bool:
    """Return True if ANY breaker class is currently open."""
    return bool(open_classes(now=now))
