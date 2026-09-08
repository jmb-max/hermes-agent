"""Telemetry for claude_worker (requirement 9).

Appends exactly one JSON line per invocation to a FIXED host path —
``get_hermes_home()/claude-worker/telemetry.jsonl`` — never a caller- or
config-supplied one: ``append_record``'s ``configured_path`` parameter is
deprecated and ignored, so nothing task-, tool-arg-, or config.yaml-derived
can redirect telemetry at an attacker-chosen file. Safety over completeness:
the write path is a strict field ALLOWLIST (``SAFE_FIELDS``), not a
blocklist — any field not explicitly named here is dropped, so a new caller
accidentally passing something sensitive (a prompt, raw stdout, a token) can
never reach disk just because nobody thought to blocklist it.

Terra's review pointed out that an allowlist of *field names* still lets
sensitive **values** through, because paths and filenames are attacker- and
user-influenced: a repo path can name a customer, and a generated filename
can carry a token. So:

* ``cwd`` is never written verbatim — it is stored as a stable
  ``sha256:<16 hex>`` fingerprint, which still supports "same repo?"
  correlation across records without naming the repo;
* ``files_touched`` entries are normalized to repo-relative paths (an
  absolute path is hashed rather than recorded), truncated, and capped in
  number;
* every remaining string value is scanned for secret-shaped content, not
  just secret-shaped key names.

The destination itself is opened trusted, not merely path-checked: a
symlink, a FIFO, a device node, or a directory at the fixed path is refused
by an ``os.open`` with ``O_NOFOLLOW`` (never follows a symlink) and
``O_NONBLOCK`` (never blocks waiting for a FIFO reader), followed by an
``fstat`` that requires exactly a regular file owned by the current user or
root, never group/world-accessible. A primary-write failure of any kind
falls back to exactly one minimal, redacted record at a second FIXED path
(``telemetry-fallback.jsonl``) carrying ``telemetry_error`` and whatever
already-safe correlation fields are available — logged as an operational
error, never raised. ``append_record`` guarantees the "exactly one record
per invocation, never raises, never blocks" property on its own: a malformed
value, an unexpected sanitization error, or a primary AND fallback write
failure all degrade rather than propagating.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import stat
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

_lock = threading.Lock()

# The full, closed set of fields ever written to telemetry. Deliberately
# excludes anything that could carry free-form text: no prompt, no raw
# stdout/stderr, no command strings, no tokens.
SAFE_FIELDS: tuple = (
    "ts", "session_id", "chat_id", "model", "route_reason", "attempt",
    "escalated", "duration_ms", "exit_code", "failure_class",
    "breaker_state", "cwd", "files_touched", "success", "telemetry_error",
    "diagnostic",
)

#: Fields safe enough to also carry into a fallback record for
#: correlation — always run through ``_safe_scalar`` first, same as any
#: other scalar field.
_CORRELATION_FIELDS: tuple = ("ts", "session_id", "chat_id", "model")

_SECRET_KEY_PATTERN = re.compile(
    r"(key|token|secret|password|credential|authorization|prompt|stdout|stderr|command)",
    re.IGNORECASE,
)

# Secret-SHAPED values, wherever they appear — including inside a path or a
# filename, which is exactly where the field-name allowlist has no opinion.
_SECRET_VALUE_PATTERN = re.compile(
    r"(sk-[A-Za-z0-9_\-]{8,}"
    r"|gh[pousr]_[A-Za-z0-9]{8,}"
    r"|AKIA[0-9A-Z]{12,}"
    r"|xox[abprs]-[A-Za-z0-9\-]{8,}"
    r"|eyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}"
    r"|(?:api[_\-]?key|token|secret|password|passwd|credential)[=:][^\s]{4,}"
    r"|[0-9a-fA-F]{32,}"
    r"|[A-Za-z0-9+/]{40,}={0,2})",
    re.IGNORECASE,
)

_MAX_FILES = 50
_MAX_VALUE_CHARS = 256

#: Independent bound on a diagnostic excerpt, applied here regardless of
#: whatever bound the caller already applied (``runner._MAX_ERROR_EXCERPT_CHARS``)
#: — defense in depth must not trust the caller to have bounded it correctly.
_MAX_DIAGNOSTIC_EXCERPT_CHARS = 256

# ---------------------------------------------------------------------------
# Fixed destinations — never derived from caller/config input
# ---------------------------------------------------------------------------

_TRUSTED_DIR_MODE = 0o700
_TRUSTED_FILE_MODE = 0o600


def _primary_path() -> Path:
    """The one fixed primary destination. Never derived from a caller- or
    config-supplied value — always under ``get_hermes_home()``."""
    return Path(get_hermes_home()) / "claude-worker" / "telemetry.jsonl"


def _fallback_path() -> Path:
    """The one fixed fallback destination, used only when a write to
    ``_primary_path`` fails."""
    return Path(get_hermes_home()) / "claude-worker" / "telemetry-fallback.jsonl"


class _UntrustedTelemetryDestination(OSError):
    """The fixed telemetry path exists but is not a plain, trusted regular
    file (symlink, FIFO, device, directory, or wrong owner/permissions)."""


def _ensure_trusted_dir(path: Path) -> None:
    """Create *path* (parents included) and force it to exactly
    ``_TRUSTED_DIR_MODE`` — never relying on umask, which the caller's
    environment controls."""
    path.mkdir(parents=True, exist_ok=True)
    os.chmod(path, _TRUSTED_DIR_MODE)


def _open_trusted_for_append(path: Path) -> int:
    """Open *path* for a trusted, bounded append and return the fd.

    ``O_NOFOLLOW`` refuses a symlink at the final path component outright
    (raises ``OSError``/``ELOOP``) rather than following it to whatever it
    points at. ``O_NONBLOCK`` means opening a FIFO with no reader on the
    other end fails immediately (``ENXIO``) instead of hanging the caller.
    The ``fstat`` on the held-open fd — race-free relative to the open
    itself — then requires exactly a regular file, owned by the current uid
    or root, with no group/world permission bits at all; anything else
    (a device node, a FIFO that *did* have a reader, a directory that
    somehow passed ``O_CREAT``) raises and closes the fd before use.
    """
    flags = (
        os.O_WRONLY | os.O_APPEND | os.O_CREAT
        | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK
    )
    fd = os.open(str(path), flags, _TRUSTED_FILE_MODE)
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            raise _UntrustedTelemetryDestination(
                f"telemetry destination is not a regular file: {path!r}"
            )
        if st.st_uid not in (os.getuid(), 0):
            raise _UntrustedTelemetryDestination(
                f"telemetry destination is not owned by the current user or root: {path!r}"
            )
        if st.st_mode & (stat.S_IRWXG | stat.S_IRWXO):
            raise _UntrustedTelemetryDestination(
                f"telemetry destination has group/world permissions: {path!r}"
            )
    except Exception:
        os.close(fd)
        raise
    return fd


def _write_trusted_line(path: Path, line: str) -> None:
    """Create *path*'s parent directory trusted, open *path* trusted, and
    append one bounded line. Raises on any failure — callers decide how to
    degrade."""
    _ensure_trusted_dir(path.parent)
    fd = _open_trusted_for_append(path)
    try:
        os.write(fd, (line + "\n").encode("utf-8"))
    finally:
        os.close(fd)


# ---------------------------------------------------------------------------
# Sanitization
# ---------------------------------------------------------------------------


def redact_secret_shapes(text: Any) -> str:
    """Replace every secret-SHAPED substring in *text* with ``[REDACTED]``,
    wherever it appears — not an all-or-nothing drop like ``_safe_scalar``'s
    field scan, a targeted substitution so a bounded diagnostic excerpt stays
    readable around the part that had to be redacted.

    Shared by ``runner.py``'s bounded failure-diagnostic builder (so an
    excerpt is redacted before it is ever handed to telemetry) and by this
    module's own defense-in-depth re-scan of a diagnostic object — the same
    pattern, applied twice, never trusted from only one side.
    """
    value = text if isinstance(text, str) else ("" if text is None else str(text))
    if not value:
        return value
    return _SECRET_VALUE_PATTERN.sub("[REDACTED]", value)


def hash_value(value: Any) -> str:
    """Stable, unsalted fingerprint — correlatable, not reversible to a name."""
    try:
        text = value if isinstance(value, str) else str(value)
    except Exception:
        return "sha256:unhashable"
    return "sha256:" + hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()[:16]


def hash_path(path: Any) -> str:
    """Fingerprint a filesystem path by its canonical form, so the same repo
    reached through a symlink or a trailing slash fingerprints identically."""
    if not isinstance(path, str) or not path:
        return ""
    try:
        canonical = os.path.realpath(path)
    except (OSError, ValueError):
        canonical = path
    return hash_value(canonical)


def _repo_relative(entry: str, cwd: Any) -> str:
    """Normalize a touched-file entry to a repo-relative path.

    An absolute path (or one that escapes the repo) is fingerprinted instead
    — telemetry never records where on the host a file lives.
    """
    if not os.path.isabs(entry):
        return os.path.normpath(entry)
    if not isinstance(cwd, str) or not cwd:
        return hash_value(entry)
    try:
        relative = os.path.relpath(os.path.realpath(entry), os.path.realpath(cwd))
    except (OSError, ValueError):
        return hash_value(entry)
    if relative.startswith(".."):
        return hash_value(entry)
    return relative


def _safe_files(value: Any, cwd: Any) -> List[str]:
    if not isinstance(value, (list, tuple)):
        return []
    out: List[str] = []
    for entry in list(value)[:_MAX_FILES]:
        if not isinstance(entry, str) or not entry:
            continue
        relative = _repo_relative(entry, cwd)
        if _SECRET_VALUE_PATTERN.search(relative):
            out.append(hash_value(relative))
        else:
            out.append(relative[:_MAX_VALUE_CHARS])
    return out


def _safe_scalar(key: str, value: Any) -> Any:
    if isinstance(value, str):
        if _SECRET_KEY_PATTERN.search(key) or _SECRET_VALUE_PATTERN.search(value):
            return "[REDACTED]"
        return value[:_MAX_VALUE_CHARS]
    if value is None or isinstance(value, (bool, int, float)):
        return value
    # A malformed value (dict/list/object in a scalar field) is recorded as a
    # type marker rather than serialized — it could carry anything.
    return f"[UNSUPPORTED:{type(value).__name__}]"


def build_record(
    *,
    ts: str = "",
    session_id: str = "",
    chat_id: str = "",
    model: str = "",
    route_reason: str = "",
    attempt: int = 0,
    escalated: bool = False,
    duration_ms: int = 0,
    exit_code: Optional[int] = None,
    failure_class: Optional[str] = None,
    breaker_state: str = "closed",
    cwd: str = "",
    files_touched: Optional[List[str]] = None,
    success: bool = False,
    diagnostic: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a telemetry record from named, already-safe fields.

    ``diagnostic`` is ``None`` on every success, preflight-HOLD, and
    breaker-open-refusal path — it is only ever populated by a caller that
    actually ran a failing spawn, never fabricated here.
    """
    return {
        "ts": ts,
        "session_id": session_id,
        "chat_id": chat_id,
        "model": model,
        "route_reason": route_reason,
        "attempt": attempt,
        "escalated": escalated,
        "duration_ms": duration_ms,
        "exit_code": exit_code,
        "failure_class": failure_class,
        "breaker_state": breaker_state,
        "cwd": cwd,
        "files_touched": list(files_touched or []),
        "success": success,
        "diagnostic": diagnostic,
    }


def _safe_diagnostic(value: Any) -> Optional[Dict[str, Any]]:
    """Sanitize a bounded failure-diagnostic object down to its exact,
    already-small, already-redacted surface.

    Independent defense in depth, not trust in the caller: the excerpt is
    re-redacted for secret shapes and re-truncated to this module's own
    (possibly tighter) bound, and every field is re-typed rather than
    passed through. ``None`` (no diagnostic — success, preflight, or
    refusal) and anything not shaped like a diagnostic both sanitize to
    ``None``, never a fabricated placeholder.
    """
    if not isinstance(value, dict):
        return None
    source = value.get("error_source")
    excerpt = value.get("error_excerpt")
    fingerprint = value.get("error_fingerprint")
    status = value.get("api_error_status")

    excerpt_text = excerpt if isinstance(excerpt, str) else ""
    excerpt_text = redact_secret_shapes(excerpt_text)[:_MAX_DIAGNOSTIC_EXCERPT_CHARS]

    return {
        "error_source": source[:64] if isinstance(source, str) else "",
        "error_excerpt": excerpt_text,
        "error_fingerprint": fingerprint if isinstance(fingerprint, str) else "",
        "api_error_status": (
            status if isinstance(status, int) and not isinstance(status, bool) else None
        ),
    }


def _sanitize(record: Dict[str, Any]) -> Dict[str, Any]:
    raw_cwd = record.get("cwd")
    safe: Dict[str, Any] = {}
    for key in SAFE_FIELDS:
        if key not in record:
            continue
        value = record[key]
        if key == "cwd":
            safe[key] = hash_path(value)
        elif key == "files_touched":
            safe[key] = _safe_files(value, raw_cwd)
        elif key == "diagnostic":
            safe[key] = _safe_diagnostic(value)
        else:
            safe[key] = _safe_scalar(key, value)
    return safe


def _safe_correlation_fields(record: Any) -> Dict[str, Any]:
    """A small, already-safe subset of *record* to carry into a fallback
    marker record, so a fallback line is still correlatable to the run that
    produced it — never anything beyond what ``_safe_scalar`` allows
    through."""
    if not isinstance(record, dict):
        return {}
    out: Dict[str, Any] = {}
    for key in _CORRELATION_FIELDS:
        if key not in record:
            continue
        try:
            out[key] = _safe_scalar(key, record[key])
        except Exception:
            continue
    return out


# ---------------------------------------------------------------------------
# Append
# ---------------------------------------------------------------------------


def _write_primary_or_fallback(line: str, correlation: Dict[str, Any]) -> None:
    """Write *line* to the fixed primary destination; on ANY failure, write
    exactly one minimal, redacted fallback record instead. Never raises."""
    try:
        _write_trusted_line(_primary_path(), line)
        return
    except Exception as exc:
        logger.error(
            "claude_worker telemetry: primary write failed (%s); writing fallback record", exc
        )

    try:
        fallback_record = {"telemetry_error": "primary_write_failed", **correlation}
        fallback_line = json.dumps(fallback_record, ensure_ascii=False, sort_keys=True, default=str)
        _write_trusted_line(_fallback_path(), fallback_line)
    except Exception:
        logger.error("claude_worker telemetry: fallback write also failed", exc_info=True)


def append_record(record: Dict[str, Any], configured_path: str = "") -> None:
    """Append exactly one sanitized JSON line to the fixed telemetry path.

    ``configured_path`` is DEPRECATED and ignored: the write destination is
    always ``get_hermes_home()/claude-worker/telemetry.jsonl`` — no
    caller-, tool-arg-, or config.yaml-supplied path can ever redirect where
    telemetry lands. Never raises and never blocks the caller down with it:
    a malformed record, a sanitization error, or a primary+fallback write
    failure all degrade to a logged operational error rather than an
    exception or a hang.
    """
    if configured_path:
        logger.error(
            "claude_worker telemetry: configured_path %r is deprecated and ignored; "
            "telemetry always writes to the fixed hermes-home destination",
            configured_path,
        )

    try:
        correlation = _safe_correlation_fields(record)
        try:
            safe = _sanitize(record if isinstance(record, dict) else {})
            line = json.dumps(safe, ensure_ascii=False, sort_keys=True, default=str)
        except Exception:
            line = json.dumps(
                {"telemetry_error": "record_sanitization_failed", **correlation}, sort_keys=True,
            )
        with _lock:
            _write_primary_or_fallback(line, correlation)
    except Exception:
        logger.error("claude_worker telemetry: append_record failed unexpectedly", exc_info=True)
