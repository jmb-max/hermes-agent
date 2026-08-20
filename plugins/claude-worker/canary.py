"""Canary channel eligibility for claude_worker (requirement 7).

``pre_tool_call`` only has ``HERMES_SESSION_CHAT_ID`` / ``HERMES_SESSION_KEY``
via ``gateway.session_context`` (task-local ContextVars), and — critically —
``parent_chat_id`` is NOT one of those tracked vars. Inside a Discord thread,
``chat_id`` is the *thread* id, not the parent channel id, so a naive
"is chat_id one of the canary channels" check silently misses every thread
under a canary channel.

The fix (no core change): register ``pre_gateway_dispatch`` — a pure
observer hook fired once per inbound message, BEFORE the session context is
even bound, with the full ``MessageEvent`` (including ``source.parent_chat_id``
and ``source.thread_id``) and the live ``GatewayRunner``. It records this
source's IDENTITY — ``platform``/``chat_id``/``parent_chat_id``/``thread_id``,
all as strings — keyed by the session key the gateway will use for this
source (``gateway._session_key_for_source(source)`` — the same resolution the
gateway itself uses to bind ``HERMES_SESSION_KEY`` a few steps later).
``pre_tool_call`` (via ``gate.py``) then consults THAT cache by the now-bound
``HERMES_SESSION_KEY``.

The cache stores IDENTITY, never a derived eligibility bool. This is
deliberate, and is the fix for a second, subtler bug a boolean cache has: a
bool baked in at dispatch time goes stale the instant configuration changes
— a thread recorded while the canary was disabled stays "ineligible"
forever even after an operator re-enables it, and a thread recorded while in
scope stays "eligible" forever even after config narrows it out. Recording
identity instead and re-applying ``policy.enabled_canary_channel_ids`` to it
on every :func:`current_session_eligibility_state` lookup — loading config
FRESH each time — means a config change takes effect on the very next
lookup, with no new dispatch event required. Because identity recording no
longer depends on config at all, :func:`compute_and_cache_eligibility` also
can no longer fail to record on a transient config-load error, or record a
decision that was only correct for the config in force at that instant.

Tri-state, not boolean — this is the cache-loss fix. The cache is the ONLY
authority for a Discord THREAD (``chat_id`` there is the thread id, not the
parent channel, so a thread cannot self-determine membership from session
vars alone). :func:`current_session_eligibility_state` returns
``Optional[bool]``: ``True``/``False`` only when actually confirmed one way
or the other, and ``None`` (UNKNOWN) whenever it cannot be — which
``gate.py`` must treat as fail-closed, never as "not canary". A cache miss
(evicted entry, process restart, a ``pre_gateway_dispatch`` session-key
resolution failure, a lookup before the hook ran, an error loading config
for a cached Discord identity) all return ``None`` for a thread, never
``False``.

A DIRECT (non-thread) Discord channel needs no cache at all: its own
``HERMES_SESSION_CHAT_ID`` is authoritative, checked straight against
:func:`policy.enabled_canary_channel_ids`, so it stays correctly eligible
even with an empty/reset cache. CLI and every non-Discord platform remain
confidently ``False`` — nothing here weakens that.

Scope is decided entirely by ``policy.py`` — the two fixed Discord channel
ids and the literal platform ``discord``. Configuration is consulted only
through :func:`policy.enabled_canary_channel_ids`, which can narrow the
policy set (disable, or select a subset) but can never add to it.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Dict, Optional

from gateway.session_context import get_session_env

from . import policy as _policy

logger = logging.getLogger(__name__)

_lock = threading.Lock()

#: session_key -> {"platform", "chat_id", "parent_chat_id", "thread_id"},
#: all strings. Immutable source identity, never a derived eligibility bool
#: — see the module docstring for why a bool cache goes stale on a config
#: change and an identity cache cannot.
_identity_cache: Dict[str, Dict[str, str]] = {}

# Deliberately unbounded: a claude_worker canary channel sees far fewer
# distinct sessions than would ever matter for memory, and the previous
# bounded-eviction policy was itself the cache-loss vulnerability — an
# active Discord thread's recorded identity could be silently evicted by
# unrelated sessions' traffic, then read back as a confident "not canary"
# and drop the write gate. There is no cap and no eviction; only an
# explicit :func:`reset_cache` (tests, or an operator-initiated restart)
# clears it, and a cleared/never-recorded thread reads as UNKNOWN, not
# ineligible.


def _record(session_key: str, identity: Dict[str, str]) -> None:
    if not session_key:
        return
    with _lock:
        _identity_cache[session_key] = dict(identity)


def reset_cache() -> None:
    """Drop every recorded ``pre_gateway_dispatch`` identity.

    For tests, and to model a process restart / cache loss: every Discord
    thread session immediately reads back as UNKNOWN (never ``False``) until
    ``pre_gateway_dispatch`` repopulates it — the safe direction, since
    ``gate.py`` fails closed on UNKNOWN.
    """
    with _lock:
        _identity_cache.clear()


def compute_and_cache_eligibility(
    event: Any,
    gateway: Any,
    session_store: Any = None,
    cfg: Optional[Dict[str, Any]] = None,
    **_: Any,
) -> None:
    """``pre_gateway_dispatch`` hook entry point.

    Pure observer — ALWAYS returns ``None`` (never ``skip``/``rewrite``), so
    it can never alter message dispatch. Records this source's IDENTITY
    (platform/chat_id/parent_chat_id/thread_id) keyed by the resolved
    session key — never a derived eligibility decision, and therefore never
    dependent on config at all: this always records, regardless of whether
    the canary is currently disabled, narrowed, or its config fails to load.
    Any error here is swallowed; a resolution failure simply means nothing
    is recorded for this dispatch, which :func:`current_session_eligibility_state`
    already treats as UNKNOWN (for a thread) rather than a confirmed decision.
    """
    try:
        source = getattr(event, "source", None)
        if source is None:
            return None

        platform_obj = getattr(source, "platform", None)
        platform = getattr(platform_obj, "value", None) or (
            platform_obj if isinstance(platform_obj, str) else ""
        )

        identity = {
            "platform": platform if isinstance(platform, str) else "",
            "chat_id": str(getattr(source, "chat_id", "") or ""),
            "parent_chat_id": str(getattr(source, "parent_chat_id", "") or ""),
            "thread_id": str(getattr(source, "thread_id", "") or ""),
        }

        session_key = None
        try:
            session_key = gateway._session_key_for_source(source)
        except Exception:
            session_key = None

        if session_key:
            _record(str(session_key), identity)
    except Exception:
        logger.debug("claude-worker canary: identity recording failed", exc_info=True)
    return None


def _eligibility_from_identity(identity: Dict[str, str]) -> Optional[bool]:
    """Apply the CURRENT config to a cached source identity.

    Loads config fresh on every call — never memoized — so a config change
    (enable/disable/narrow) takes effect on the very next lookup for an
    already-cached thread, with no new dispatch event required. Returns
    ``None`` (UNKNOWN) if config cannot be loaded for this lookup, never a
    silent ``False``.
    """
    platform = identity.get("platform", "")
    if not _policy.is_canary_platform(platform):
        return False
    try:
        from . import config as _cfg_mod

        cfg = _cfg_mod.load_plugin_config()
        in_scope = _policy.enabled_canary_channel_ids(cfg)
    except Exception:
        return None
    chat_id = identity.get("chat_id", "")
    parent_chat_id = identity.get("parent_chat_id", "")
    return bool(in_scope) and (chat_id in in_scope or parent_chat_id in in_scope)


def current_session_eligibility_state() -> Optional[bool]:
    """``pre_tool_call`` entry point — tri-state canary eligibility.

    Returns:
      ``True``  — confirmed canary: a cached ``pre_gateway_dispatch`` source
                  identity that is currently in scope under the CURRENT
                  config, OR (needing no cache at all) a direct, non-thread
                  Discord channel whose own ``HERMES_SESSION_CHAT_ID`` is
                  itself one of the enabled canary ids.
      ``False`` — confirmed NOT canary: a cached identity that is currently
                  out of scope (non-Discord platform, or a channel/parent
                  outside the enabled ids), or a direct Discord channel
                  outside the enabled canary ids.
      ``None``  — UNKNOWN, cannot be confirmed either way. Always a Discord
                  THREAD (``HERMES_SESSION_THREAD_ID`` set — its ``chat_id``
                  is the thread id, not the parent channel, so it cannot
                  self-determine) with no cached identity for this session
                  key (an evicted/reset/restarted cache, a
                  ``pre_gateway_dispatch`` session-key resolution failure, a
                  lookup that ran before the hook did, a missing/unbound
                  session key), or a cached Discord identity whose config
                  could not be loaded for this lookup, or any unexpected
                  error reading session identity, the cache, or config.
                  Callers MUST fail closed on ``None`` — it is never coerced
                  to ``False``.
    """
    try:
        session_key = get_session_env("HERMES_SESSION_KEY")
    except Exception:
        session_key = ""

    if session_key:
        try:
            with _lock:
                identity = _identity_cache.get(session_key)
        except Exception:
            return None
        if identity is not None:
            try:
                return _eligibility_from_identity(identity)
            except Exception:
                return None

    try:
        platform = get_session_env("HERMES_SESSION_PLATFORM")
    except Exception:
        return None

    if not _policy.is_canary_platform(platform):
        return False

    try:
        thread_id = str(get_session_env("HERMES_SESSION_THREAD_ID") or "")
        if thread_id:
            # Cannot self-determine — the only authoritative source for a
            # thread is the pre_gateway_dispatch cache entry we already
            # looked for (by session key) above and did not find.
            return None

        chat_id = str(get_session_env("HERMES_SESSION_CHAT_ID") or "")
        from . import config as _cfg_mod

        cfg = _cfg_mod.load_plugin_config()
        in_scope = _policy.enabled_canary_channel_ids(cfg)
        return chat_id in in_scope
    except Exception:
        return None
