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

#: Deliberately NOT a bare "401": a task's own output can legitimately
#: contain that numeral (a line number, a test file name, an HTTP status
#: quoted in unrelated log output) with no authentication problem at all.
#: A real numeric HTTP 401 is recognized structurally instead, from a typed
#: status FIELD in parsed JSON — see :func:`_numeric_status` — never from a
#: substring match over free text.
_AUTH_MARKERS = (
    "invalid api key", "not logged in", "claude login", "oauth token",
    "authentication_error", "please run `claude login`",
    "please run /login",
)
#: Deliberately NOT a bare "429": a task's own output can legitimately
#: contain that numeral (a line number, a port, a byte count) with no rate
#: limit involved at all. A real HTTP 429 response is always paired with the
#: phrase "too many requests" in Claude CLI's own diagnostics, so that phrase
#: is the marker, never the bare digits.
_RATE_MARKERS = ("rate limit", "too many requests")
_EXTRA_USAGE_MARKERS = (
    "usage limit", "quota exceeded", "credit balance", "plan limit",
    "usage cap", "exceeded your usage",
)

#: Alternate key names real ``claude --output-format json`` error bodies (and
#: nested ``error`` objects within them) have been observed to carry an HTTP
#: status under. ``api_error_status`` is the shape already handled; ``status``/
#: ``status_code``/``code`` are the same signal under a different name, never
#: a new kind of evidence.
_STATUS_FIELD_NAMES = ("api_error_status", "status", "status_code", "code")

#: Typed-field spelling of "this is an authentication error" — distinct from
#: a free-text marker search: an exact match on a ``type``/``subtype`` field
#: (top-level or nested under ``error``) is unambiguous evidence, not a
#: substring that could coincidentally appear in unrelated output.
_AUTH_ERROR_TYPE_VALUE = "authentication_error"

_lock = threading.Lock()


def _state_path() -> Path:
    return Path(get_hermes_home()) / "claude-worker" / "breaker.json"


#: Post-spawn signal that the OAuth SESSION itself (not a plain task
#: failure) is stale/revoked/unusable — the shape ``oauth.py``'s pre-spawn
#: preflight cannot see, because the local credential file can look fresh
#: (unexpired timestamp) while the server has already rejected it. This is
#: deliberately NOT one of ``BREAKER_CLASSES``: it does not open the hour-
#: long ``auth`` cooldown, because waiting does not fix an invalid session —
#: only re-authenticating does. ``runner.py`` reports it to the caller as
#: ``failure_class: auth_preflight`` with reauthentication guidance, the
#: same vocabulary as the pre-spawn HOLD.
AUTH_PREFLIGHT_FAILURE_CLASS = "auth_preflight"


def classify_failure(exit_code: Optional[int], stderr: str, stdout: str = "") -> str:
    """Classify a failed spawn into a breaker class, ``auth_preflight``
    (session-invalid, no breaker mutation), or ``other``.

    ``stdout`` is best-effort: ``claude --output-format json`` can exit
    non-zero with an empty ``stderr`` and the real reason in a structured
    stdout body — see :func:`_classify_structured_stdout`. That check runs
    first since it is unambiguous when present; malformed/non-JSON stdout is
    treated as no signal and falls through to the existing stderr-marker
    classification.
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


def _structured_error_text(parsed: Dict[str, Any]) -> str:
    """Best-effort free text out of a parsed ``--output-format json`` error
    body: the top-level ``result``/``error``/``message`` string fields, plus
    a nested ``error.message`` — the shapes real Claude CLI diagnostics use.
    Never raises on an unexpected shape."""
    parts = []
    for key in ("result", "error", "message"):
        value = parsed.get(key)
        if isinstance(value, str):
            parts.append(value)
    nested_error = parsed.get("error")
    if isinstance(nested_error, dict):
        nested_message = nested_error.get("message")
        if isinstance(nested_message, str):
            parts.append(nested_message)
    return " ".join(parts).lower()


def _numeric_status(parsed: Dict[str, Any]) -> Optional[int]:
    """The first well-typed (non-bool) integer found under any of
    :data:`_STATUS_FIELD_NAMES`, checked at the top level and inside a
    nested ``error`` object. Never a substring match over free text — this
    is a real JSON field, not a numeral that happened to appear in output."""
    for source in (parsed, parsed.get("error")):
        if not isinstance(source, dict):
            continue
        for key in _STATUS_FIELD_NAMES:
            value = source.get(key)
            if isinstance(value, int) and not isinstance(value, bool):
                return value
    return None


def _has_authentication_error_type(parsed: Dict[str, Any]) -> bool:
    """``True`` if a ``type``/``subtype`` field — top-level or nested under
    ``error`` — is exactly ``"authentication_error"``. An exact field-value
    match, not a substring search, so it cannot be triggered by unrelated
    text that happens to mention the phrase."""
    for source in (parsed, parsed.get("error")):
        if not isinstance(source, dict):
            continue
        for key in ("type", "subtype"):
            if source.get(key) == _AUTH_ERROR_TYPE_VALUE:
                return True
    return False


def _classify_structured_stdout(stdout: str) -> Optional[str]:
    """Best-effort auth classification from ``--output-format json`` stdout.

    Two cases, both requiring ``is_error: true`` — never raises on
    malformed/non-JSON input, and anything short of these two returns
    ``None`` so the caller falls back to stderr markers:

    * an unambiguous HTTP 401 under any of :data:`_STATUS_FIELD_NAMES`
      (``api_error_status``, or the equivalent ``status``/``status_code``/
      ``code`` spellings real CLI error bodies have also been observed to
      use) classifies as the real breaker class ``"auth"`` (an observed API
      rejection; the hour-long cooldown is the correct response);
    * no numeric 401, but either a typed ``type``/``subtype`` field reading
      exactly ``"authentication_error"`` (:func:`_has_authentication_error_type`)
      or the structured error text (``result``/``error``/``message``)
      carrying one of the same reauthentication phrases (``_AUTH_MARKERS``)
      classifies as ``AUTH_PREFLIGHT_FAILURE_CLASS``: a stale/revoked OAuth
      SESSION signal that a cooldown would not fix, so it deliberately does
      not open the breaker. Every one of these is a typed field or a fixed
      phrase list, never a bare numeral or an arbitrary "auth"-adjacent
      word, so a generic task failure that happens to mention a status code
      or the word "authentication" in unrelated output is never
      misclassified.
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
    if _numeric_status(parsed) == 401:
        return "auth"
    if _has_authentication_error_type(parsed) or any(
        marker in _structured_error_text(parsed) for marker in _AUTH_MARKERS
    ):
        return AUTH_PREFLIGHT_FAILURE_CLASS
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
