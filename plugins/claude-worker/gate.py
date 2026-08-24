"""The write gate for claude_worker.

A ``pre_tool_call`` callback with two jobs in one Discord-origin session.

1. **The write gate.** Block ``patch`` / ``write_file`` / a mutating
   ``skill_manage`` action when all of these hold:

     a. the calling session is Discord-origin (see ``canary.py``), AND
     b. the target path resolves inside a real Git worktree — the SAME
        dynamic resolution the runner uses to decide what to mount
        (``project.resolve_project_root*``), so the gate can never block a
        repository the worker isn't allowed to run in, AND
     c. this (session, project_root) pair has not been released by an
        explicit Terra fallback.

   A successful ``claude_worker`` run is NOT a release. It used to be, which
   turned the worker into a one-time toll gate — call it once, then edit the
   repository directly for the rest of the session — and that is exactly what
   was observed happening: one Opus run, then a stream of direct ``patch`` /
   ``write_file`` calls. In a Discord session the worker is the coder, every
   time; a successful run proves the worker works and grants nothing.

2. **The terminal bypass.** Block a ``terminal`` command that directly
   invokes the Claude CLI on the host (``terminal_guard``). That path
   sidestepped the whole sandbox — image pin, non-root uid, empty MCP config,
   tool allowlist, resource caps — so it is refused outright for a
   Discord-origin session, unlock state or not: ``claude_worker`` is the
   canonical way to reach Claude from here. Ordinary build/test/git commands
   are untouched.

Everything else returns ``None`` and the call proceeds: another tool, a path
outside any Git worktree, a scratch directory, a non-Discord session.

Load-bearing corrections carried forward from Terra's review:

* **The tool set and the gate itself are immutable.** ``policy.GATED_TOOLS``
  is not read from config; there is no ``gate.enabled`` / ``gate.tools``
  escape hatch, and there is no longer a ``gate.repo_roots`` either — scope
  is derived from the request, not configured. A stale ``repo_roots`` key in
  an old config is accepted by the loader and then simply never read.
* **An open breaker does not release anything.** It used to disengage the gate
  entirely, so opening the breaker (invoke the worker until an auth/rate/quota
  error) was a one-step bypass. An open breaker now HOLDS.
* **Exactly one thing releases a project for a session:** an explicit Terra
  fallback that actually came back with a result (``fallback_ready`` plus full
  provenance, from the ``claude_worker`` tool itself). A Terra failure stays
  HOLD. That contingency exists because a session whose worker cannot run at
  all would otherwise have no way forward; a session whose worker CAN run has
  a way forward already — call it.

Fail-closed, not fail-dark: ``canary.current_session_eligibility_state()`` is
tri-state. ``False`` (confirmed not Discord) allows. ``True`` applies the
gate. ``None`` (UNKNOWN — a real chat session whose platform could not be
confirmed, a missing session key, any internal error) BLOCKS a gated mutation
that resolves inside a Git worktree, and blocks a direct Claude CLI
invocation, with an actionable message.

Eligibility is resolved from the EXPLICIT ``session_id`` this callback is
handed, not from the ambient contextvars alone. Both halves of this file
depend on that. A continuation after context compression arrives with every
ambient session var cleared to ``""`` — no chat id, no platform, no session
key — which the resolver used to read as "not a chat session at all" and
answer ``False``, waving through both the direct edits and a host ``claude
-p``. The Discord key was in the arguments the whole time; it is now what the
resolver is asked about first (see ``canary.py``).

Which explicit argument carries that key matters. ``session_id`` is the
TRANSCRIPT id in production (``20260823_101500_ab12cd``) — it names no
platform and context compression re-mints it mid-conversation — so the stable
per-chat key the gateway built travels alongside it as ``gateway_session_key``
and is preferred wherever a CHAT identity is needed: eligibility resolution,
the identity-cache lookup behind it, and the key a Terra fallback release is
recorded under. A caller with no gateway key resolves exactly as before.

To keep that fail-closed posture from over-reaching, the gate determines
whether a call is even IN SCOPE — a gated tool, a resolvable mutation target,
inside a resolvable project root — before it lets UNKNOWN block anything. An
irrelevant tool, a non-mutating ``skill_manage`` action, or a path outside
every Git worktree is allowed regardless of eligibility, UNKNOWN included:
those calls could never be blocked no matter what the session turns out to
be, so UNKNOWN must not manufacture a block for them. Equally, a path outside
any Git worktree is never gated even for a CONFIRMED Discord session — the
worker could not run there either, and gating it would deadlock the session
with nothing able to satisfy the gate.
"""

from __future__ import annotations

import json
import logging
import threading
from typing import Any, Dict, Optional, Set, Tuple

from gateway.session_context import get_session_env

from . import canary as _canary
from . import breaker as _breaker
from . import policy as _policy
from . import project as _project
from . import terminal_guard as _terminal_guard

logger = logging.getLogger(__name__)

_unlock_lock = threading.Lock()

# (session_key, project_root) pairs released by an explicit Terra fallback —
# the ONLY release there is. In-process only — deliberately not persisted: a
# gateway restart resetting these to "locked" is the SAFE direction (worst
# case, the session calls claude_worker again).
#
# There is deliberately no companion set for successful runs. One used to
# exist, and a successful run added to it, which made the worker a ONE-TIME
# toll gate: call it once, then edit the repository directly for the rest of
# the session. That is not the policy — in a Discord session the worker is
# the coder, every time — and it was observed happening in production.
_fallback_unlocks: Set[Tuple[str, str]] = set()

_BLOCK_MESSAGE = (
    "claude_worker gate: direct {tool} edits are disabled for this Discord "
    "session inside {repo_root}. Code changes in this repository go through "
    "the claude_worker tool — call it with this task. A successful run does "
    "NOT hand back direct edit access: the worker stays the coder for the "
    "rest of the session, so send the next change to it too."
)

_BREAKER_MESSAGE = (
    "claude_worker gate: the worker is currently unavailable ({classes}) and "
    "this Discord session stays on HOLD inside {repo_root} — an open breaker "
    "does not enable direct {tool} edits. Wait for the cooldown, or call "
    "claude_worker to request the explicit Terra fallback, which is the only "
    "thing that releases direct edits here and only when Terra actually "
    "returns a result."
)

_FAIL_CLOSED_MESSAGE = (
    "claude_worker gate: this edit could not be evaluated in a Discord "
    "session (missing or unresolvable target, or an internal error). "
    "Failing closed — call claude_worker instead of retrying the direct edit."
)

_UNKNOWN_ELIGIBILITY_MESSAGE = (
    "claude_worker gate: Discord origin for this session could not be "
    "confirmed (no recorded dispatch identity — an evicted, reset, or "
    "not-yet-populated cache — and no confirmable platform on the session "
    "itself), and this direct {tool} edit resolves inside the Git repository "
    "{repo_root}. Failing closed: call claude_worker instead of retrying the "
    "direct edit. If this is not actually a Discord session, wait for the "
    "gateway to (re)record this session's identity and retry."
)

_TERMINAL_BLOCK_MESSAGE = (
    "claude_worker gate: this terminal command invokes the Claude CLI "
    "directly on the host ({reason}), which bypasses the claude_worker "
    "sandbox entirely — no pinned image, no non-root user, no empty MCP "
    "config, no tool allowlist, no resource limits. Use the claude_worker "
    "tool instead; it is the only supported way to run Claude from a Discord "
    "session. Ordinary terminal commands (builds, tests, git) are unaffected."
)

_TERMINAL_FAIL_CLOSED_MESSAGE = (
    "claude_worker gate: this terminal command could not be evaluated for a "
    "direct Claude CLI invocation in a Discord session. Failing closed — use "
    "the claude_worker tool, or re-issue the command in a well-formed shape."
)


def reset_unlocks() -> None:
    """Drop every recorded release. For tests and for an operator-initiated
    re-arm of the gate; the safe direction is always "locked"."""
    with _unlock_lock:
        _fallback_unlocks.clear()


def _session_key(session_id: str = "", gateway_session_key: str = "") -> str:
    """The identity a release is recorded against, most stable source first.

    ``gateway_session_key`` — the per-CHAT key the gateway built — wins,
    because it is the only one of the three that names the chat itself. The
    dispatch sites pass ``agent.session_id``, which is the TRANSCRIPT id: it is
    re-minted by context compression, so a release keyed on it is scoped to a
    conversation slice and is silently revoked by the next compression.

    Failing that, the explicit dispatch ``session_id`` wins over the ambient
    contextvar. ``post_tool_call`` can be observed from a context bound to a
    *different* session than the call it reports; trusting the ambient var
    there would let one session's Terra fallback release another's repository.
    On the ``pre_tool_call`` side the explicit key matters for a second reason:
    after a context compression/continuation the ambient var is cleared
    outright, so it is the only session identity left.
    """
    if gateway_session_key:
        return gateway_session_key
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


def _terminal_directive(args: Dict[str, Any]) -> Optional[Dict[str, str]]:
    """Block a direct host Claude CLI invocation from a Discord session.

    Deliberately unconditional on unlock state: unlocking a repository for
    direct ``patch`` edits is a statement about that repository, never a
    licence to run an unsandboxed Claude on the host.
    """
    try:
        verdict = _terminal_guard.evaluate(args.get("command"))
    except Exception:
        logger.warning("claude-worker gate: terminal guard errored, failing closed", exc_info=True)
        return {"action": "block", "message": _TERMINAL_FAIL_CLOSED_MESSAGE}
    if not verdict.get("blocked"):
        return None
    return {
        "action": "block",
        "message": _TERMINAL_BLOCK_MESSAGE.format(reason=verdict.get("reason") or "detected"),
    }


def on_pre_tool_call(
    tool_name: str = "",
    args: Optional[Dict[str, Any]] = None,
    session_id: str = "",
    gateway_session_key: str = "",
    **_: Any,
) -> Optional[Dict[str, str]]:
    args = args if isinstance(args, dict) else {}

    # Determine scope FIRST, independent of eligibility: an irrelevant tool
    # is never gated no matter what the session turns out to be, so it must
    # never pay for (or be blocked by) an eligibility check at all.
    is_gated_tool = tool_name in _policy.GATED_TOOLS
    is_terminal_tool = tool_name in _policy.TERMINAL_TOOLS
    if not is_gated_tool and not is_terminal_tool:
        return None

    # Eligibility is tri-state. False allows outright. True or None (UNKNOWN)
    # both continue below — an internal error resolving it is treated the
    # same as an explicit UNKNOWN, never as "not Discord".
    #
    # The explicit chat identity is handed to the resolver, not just used for
    # the unlock key: a continuation whose ambient contextvars were cleared has
    # NO ambient identity left, and reading that as "not a chat session" is
    # exactly how a Discord session bypassed both this gate and the terminal
    # guard.
    #
    # The GATEWAY key is preferred over the dispatch ``session_id`` because
    # that argument carries the TRANSCRIPT id in production — it names no
    # platform and it rotates on every context compression, so it can neither
    # confirm Discord nor hit the identity cache the gateway recorded under the
    # chat's own key. When no gateway key is offered (the CLI, and every
    # pre-existing call site) resolution is exactly what it was.
    resolved_key = gateway_session_key or session_id
    try:
        eligible = _canary.current_session_eligibility_state(resolved_key)
    except Exception:
        eligible = None
    if eligible is False:
        return None

    if is_terminal_tool:
        return _terminal_directive(args)

    # Not confirmed "not Discord" — fail closed from here.
    try:
        path, in_scope_tool = _target_path(tool_name, args)
        if not in_scope_tool:
            return None
        if not path:
            return {"action": "block", "message": _FAIL_CLOSED_MESSAGE}

        repo_root = _project.resolve_project_root_for_target(path)
        if repo_root is None:
            # Not inside any Git worktree the worker could run in — never in
            # scope, regardless of eligibility. Gating it would deadlock the
            # session: nothing could ever satisfy the gate.
            return None

        if eligible is None:
            return {
                "action": "block",
                "message": _UNKNOWN_ELIGIBILITY_MESSAGE.format(repo_root=repo_root, tool=tool_name),
            }

        # eligible is True (confirmed Discord-origin) from here on.
        key = (_session_key(session_id, gateway_session_key), repo_root)
        with _unlock_lock:
            released = key in _fallback_unlocks
        if released:
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
    gateway_session_key: str = "",
    status: Optional[str] = None,
    **_: Any,
) -> None:
    """Record the ONE thing that releases a repository: a delivered Terra
    fallback.

    A successful ``claude_worker`` run is deliberately not recorded — it
    releases nothing. ``status`` and ``success`` are therefore no longer read
    here at all, which is the point: there is no branch left that a
    success-shaped payload could take.

    The release is keyed by the CHAT (``gateway_session_key``) when one is
    offered, so it survives the transcript-id rotation a context compression
    performs and can be matched by the identity ``on_pre_tool_call`` resolves
    eligibility from.
    """
    # Only the claude_worker tool's own result can release anything — another
    # tool returning a look-alike payload is ignored.
    if tool_name != "claude_worker":
        return None
    try:
        payload = _parse_result(result)
        if payload is None:
            return None
        if not _fallback_delivered(payload):
            return None

        # The SAME dynamic resolution ``on_pre_tool_call`` uses, so a release
        # is recorded against exactly the root a later gated edit resolves to.
        repo_root = _project.resolve_project_root(payload.get("cwd"))
        if repo_root is None:
            # A fallback outside any resolvable Git worktree releases nothing.
            return None

        key_session = _session_key(session_id, gateway_session_key)
        if not key_session:
            return None
        with _unlock_lock:
            _fallback_unlocks.add((key_session, repo_root))
    except Exception:
        logger.debug("claude-worker gate: post_tool_call recording failed", exc_info=True)
    return None
