"""The write-gate for claude_worker (requirement 8).

A ``pre_tool_call`` callback that blocks ``patch`` / ``write_file`` / a
mutating ``skill_manage`` action when every one of these holds:

  1. the calling session is canary-eligible (see ``canary.py``), AND
  2. the target path resolves inside a canonical configured repo root — the
     SAME resolution the runner uses (``policy.canonical_repo_roots``), so
     the gate can never block a repository the worker isn't allowed to run
     in, AND
  3. this (session, repo_root) pair has not been unlocked yet.

Everything else returns ``None`` and the call proceeds untouched: another
tool, another repo, a scratch directory, a non-canary session.

Three corrections from Terra's review are load-bearing here:

* **The tool set and the gate itself are immutable.** ``policy.GATED_TOOLS``
  is not read from config; there is no ``gate.enabled`` / ``gate.tools``
  escape hatch. Config can only choose which roots are in scope.
* **An open breaker does not unlock anything.** It used to disengage the
  gate entirely, so opening the breaker (invoke the worker until an
  auth/rate/quota error) was a one-step bypass. An open breaker now HOLDS:
  the block message explains the worker is unavailable.
* **Only two things unlock a repo for a session:** a successful
  ``claude_worker`` run there, or an explicit Terra fallback that actually
  came back with a result (``fallback_ready`` plus a non-empty Terra
  payload, from the ``claude_worker`` tool itself). A Terra failure stays
  HOLD.

Fail-closed, not fail-dark: ``canary.current_session_eligibility_state()``
is tri-state (``True`` / ``False`` / ``None``). ``False`` (confirmed not
canary) allows. ``True`` (confirmed canary) applies the normal gate below.
``None`` (UNKNOWN — a Discord thread whose ``pre_gateway_dispatch`` cache
entry is missing, evicted, reset, or never recorded because the gateway's
own session-key resolution failed; a missing session key; any internal
error reading identity, the cache, or config) BLOCKS a gated mutation that
resolves inside a configured root, with an actionable message — the
previous design silently allowed here, which is exactly how a cache-loss
event dropped the write gate for an active canary thread.

To keep that fail-closed posture from over-reaching, the gate first
determines whether this call is even IN SCOPE — a gated tool, a resolvable
mutation target, inside a configured repo root — before it ever consults
eligibility. An irrelevant tool, a non-mutating ``skill_manage`` action, or
a path outside every configured root is allowed regardless of eligibility,
UNKNOWN included: those calls could never be blocked no matter what the
session turns out to be, so UNKNOWN must not manufacture a block for them.

Once a session is CONFIRMED canary-eligible (or its eligibility is
UNKNOWN), everything downstream (config load, path resolution, unlock
lookup) is wrapped so any unexpected error blocks instead of silently
allowing — including a mutating ``skill_manage`` whose target cannot be
resolved.
"""

from __future__ import annotations

import json
import logging
import threading
from typing import Any, Dict, Optional, Set, Tuple

from gateway.session_context import get_session_env

from . import canary as _canary
from . import config as _config
from . import breaker as _breaker
from . import policy as _policy

logger = logging.getLogger(__name__)

_unlock_lock = threading.Lock()

# (session_key, repo_root) pairs with a confirmed successful claude_worker
# run, and pairs released by an explicit Terra fallback. In-process only —
# deliberately not persisted: a gateway restart resetting these to "locked"
# is the SAFE direction (worst case, one extra claude_worker call).
_successful_runs: Set[Tuple[str, str]] = set()
_fallback_unlocks: Set[Tuple[str, str]] = set()

_BLOCK_MESSAGE = (
    "claude_worker gate: direct {tool} edits are disabled for this canary "
    "session inside {repo_root} until a claude_worker run has completed "
    "here at least once. Call the claude_worker tool with this task first; "
    "once it succeeds, direct edits in this repo are unblocked for the "
    "rest of the session."
)

_BREAKER_MESSAGE = (
    "claude_worker gate: the worker is currently unavailable ({classes}) and "
    "this canary session stays on HOLD inside {repo_root} — an open breaker "
    "does not unlock direct {tool} edits. Wait for the cooldown, or call "
    "claude_worker to request the explicit Terra fallback; direct edits "
    "unlock only once claude_worker returns a successful run or a Terra "
    "fallback result."
)

_FAIL_CLOSED_MESSAGE = (
    "claude_worker gate: this edit could not be evaluated in a canary "
    "session (missing or unresolvable target, or an internal error). "
    "Failing closed — call claude_worker instead of retrying the direct edit."
)

_UNKNOWN_ELIGIBILITY_MESSAGE = (
    "claude_worker gate: canary eligibility for this session could not be "
    "confirmed (a Discord thread with no recorded dispatch decision — an "
    "evicted, reset, or not-yet-populated cache — or an internal error), "
    "and this direct {tool} edit resolves inside a configured repo root "
    "({repo_root}). Failing closed: call claude_worker instead of retrying "
    "the direct edit. If this is not actually a canary session, wait for "
    "the gateway to (re)record this session's eligibility and retry."
)


def reset_unlocks() -> None:
    """Drop every recorded unlock. For tests and for an operator-initiated
    re-arm of the gate; the safe direction is always "locked"."""
    with _unlock_lock:
        _successful_runs.clear()
        _fallback_unlocks.clear()


def _session_key(session_id: str = "") -> str:
    """The explicit dispatch ``session_id`` wins over the ambient contextvar.

    ``post_tool_call`` can be observed from a context bound to a *different*
    session than the call it reports; trusting the ambient var there would
    let one session's successful run unlock another's.
    """
    if session_id:
        return session_id
    return get_session_env("HERMES_SESSION_KEY") or ""


def _resolve_skill_manage_path(args: Dict[str, Any]) -> Optional[str]:
    """Absolute path a mutating ``skill_manage`` action would write to.

    Returns ``None`` when the target cannot be determined — which, for an
    action already known to mutate, the caller treats as fail-closed (block),
    not as "out of scope".
    """
    name = args.get("name")
    if not name or not isinstance(name, str):
        return None
    from tools.skill_manager_tool import _resolve_skill_dir

    category = args.get("category")
    skill_dir = _resolve_skill_dir(name, category if isinstance(category, str) else None)
    file_path = args.get("file_path")
    if isinstance(file_path, str) and file_path:
        return str(skill_dir / file_path)
    return str(skill_dir / "SKILL.md")


def _target_path(tool_name: str, args: Dict[str, Any]) -> Tuple[Optional[str], bool]:
    """Return ``(path, in_scope_tool)`` for a gated tool call.

    ``in_scope_tool`` False means this call is not a repo mutation at all
    (e.g. a read-only ``skill_manage`` action) and must be allowed. A True
    with a ``None`` path is a mutation whose target we could not resolve —
    the caller blocks.
    """
    if tool_name in ("patch", "write_file"):
        path = args.get("path")
        return (path if isinstance(path, str) and path else None), True
    if tool_name == "skill_manage":
        action = str(args.get("action") or "").strip().lower()
        if action not in _policy.SKILL_MANAGE_MUTATING_ACTIONS:
            return None, False
        return _resolve_skill_manage_path(args), True
    return None, False


def on_pre_tool_call(
    tool_name: str = "",
    args: Optional[Dict[str, Any]] = None,
    session_id: str = "",
    **_: Any,
) -> Optional[Dict[str, str]]:
    args = args if isinstance(args, dict) else {}

    # Determine scope FIRST, independent of eligibility: an irrelevant tool
    # is never gated no matter what the session turns out to be, so it must
    # never pay for (or be blocked by) an eligibility check at all.
    if tool_name not in _policy.GATED_TOOLS:
        return None

    # Eligibility is tri-state. False allows outright. True or None (UNKNOWN)
    # both continue below — an internal error resolving it is treated the
    # same as an explicit UNKNOWN, never as "not canary".
    try:
        eligible = _canary.current_session_eligibility_state()
    except Exception:
        eligible = None
    if eligible is False:
        return None

    # Not confirmed "not canary" — fail closed from here.
    try:
        path, in_scope_tool = _target_path(tool_name, args)
        if not in_scope_tool:
            return None
        if not path:
            return {"action": "block", "message": _FAIL_CLOSED_MESSAGE}

        cfg = _config.load_plugin_config()
        roots = _policy.canonical_repo_roots(cfg)
        repo_root = _policy.resolve_within_roots(path, roots)
        if repo_root is None:
            # Outside every root the worker may run in — never in scope,
            # regardless of eligibility. Gating it would deadlock the
            # session: nothing could ever satisfy the gate.
            return None

        if eligible is None:
            return {
                "action": "block",
                "message": _UNKNOWN_ELIGIBILITY_MESSAGE.format(repo_root=repo_root, tool=tool_name),
            }

        # eligible is True (confirmed canary) from here on.
        key = (_session_key(session_id), repo_root)
        with _unlock_lock:
            unlocked = key in _successful_runs or key in _fallback_unlocks
        if unlocked:
            return None

        open_classes = _breaker.open_classes()
        if open_classes:
            return {
                "action": "block",
                "message": _BREAKER_MESSAGE.format(
                    classes=", ".join(open_classes), repo_root=repo_root, tool=tool_name,
                ),
            }

        return {
            "action": "block",
            "message": _BLOCK_MESSAGE.format(tool=tool_name, repo_root=repo_root),
        }
    except Exception:
        logger.warning("claude-worker gate: internal error, failing closed", exc_info=True)
        return {"action": "block", "message": _FAIL_CLOSED_MESSAGE}


def _parse_result(result: Any) -> Optional[Dict[str, Any]]:
    parsed = result
    if isinstance(result, str):
        try:
            parsed = json.loads(result)
        except ValueError:
            return None
    return parsed if isinstance(parsed, dict) else None


def _fallback_delivered(payload: Dict[str, Any]) -> bool:
    """True only for an explicit Terra fallback that actually returned text
    AND carries its full provenance — every check below is required
    independently, so forging or dropping any single field leaves the
    session gated.

    ``fallback_ready`` alone is not enough: a HOLD result carries the marker
    False, and a Terra failure carries ``fallback.ok`` False. Either way the
    session stays gated. Beyond that: ``fallback_requested`` must be the
    literal ``True`` (the caller actually opted in), ``fallback_provenance``
    must be the exact ``"terra_auxiliary"`` literal (not merely present),
    and the nested ``fallback`` dict must itself carry ``ready`` as the
    literal ``True`` in addition to ``ok`` True and non-empty ``notes`` —
    a look-alike payload missing any one of these cannot unlock.
    """
    if payload.get("fallback_ready") is not True:
        return False
    if payload.get("fallback_requested") is not True:
        return False
    if payload.get("fallback_provenance") != "terra_auxiliary":
        return False
    fallback = payload.get("fallback")
    if not isinstance(fallback, dict):
        return False
    if fallback.get("ok") is not True or fallback.get("ready") is not True:
        return False
    notes = fallback.get("notes")
    return isinstance(notes, str) and bool(notes.strip())


def on_post_tool_call(
    tool_name: str = "",
    args: Optional[Dict[str, Any]] = None,
    result: Any = None,
    session_id: str = "",
    status: Optional[str] = None,
    **_: Any,
) -> None:
    # Only the claude_worker tool's own result can unlock anything — another
    # tool returning a look-alike payload is ignored.
    if tool_name != "claude_worker":
        return None
    try:
        payload = _parse_result(result)
        if payload is None:
            return None

        succeeded = payload.get("success") is True and (status is None or status == "ok")
        fallback = _fallback_delivered(payload)
        if not succeeded and not fallback:
            return None

        cwd = payload.get("cwd")
        cfg = _config.load_plugin_config()
        repo_root = _policy.resolve_within_roots(cwd, _policy.canonical_repo_roots(cfg))
        if repo_root is None:
            # A run outside every configured root can't unlock a configured one.
            return None

        key_session = _session_key(session_id)
        if not key_session:
            return None
        with _unlock_lock:
            if succeeded:
                _successful_runs.add((key_session, repo_root))
            else:
                _fallback_unlocks.add((key_session, repo_root))
    except Exception:
        logger.debug("claude-worker gate: post_tool_call recording failed", exc_info=True)
    return None
