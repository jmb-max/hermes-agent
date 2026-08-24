"""Discord-origin session eligibility for claude_worker.

Scope is now the PLATFORM, not a channel list. Every Discord-origin session
— guild channel, DM, and every thread under either — is in scope; no other
platform ever is. The two hardcoded canary channel ids this module used to
consult are gone, and configuration can no longer narrow scope to selected
channel ids (see ``policy.discord_scope_enabled``): the only thing config may
still do is turn the whole feature off globally, which is safe because it
removes the restriction from every session at once rather than quietly
exempting one.

That change simplifies the hard part but does not remove it. ``pre_tool_call``
has ``HERMES_SESSION_CHAT_ID`` / ``HERMES_SESSION_KEY`` /
``HERMES_SESSION_PLATFORM`` via ``gateway.session_context`` (task-local
ContextVars), plus the session key the dispatcher passes it EXPLICITLY as an
argument. ``parent_chat_id`` is NOT one of those tracked vars — which used
to matter enormously, because inside a Discord thread ``chat_id`` is the
*thread* id and a channel-list check silently missed every thread. With
platform-wide scope a thread no longer needs its parent at all: if the
platform is Discord, the thread is in scope, full stop.

The ambient vars are not sufficient on their own, and assuming they were is
what produced a live bypass. ``clear_session_vars`` resets every one of them
to ``""``, and an empty contextvar deliberately suppresses the ``os.environ``
fallback — so after a context compression / continuation a perfectly live
Discord session reaches :func:`current_session_eligibility_state` with no
ambient chat id, no platform, and no session key at all. That matched the
"no chat identity → this must be the CLI" branch and returned a confident
``False``, which dropped both the write gate and the terminal guard for a
session whose Discord key was in the call arguments the whole time. The
explicit key is therefore treated as first-hand provenance: it is the first
thing the identity cache is looked up by, and its own platform segment is
read directly (see :func:`_platform_from_session_key`).

The ``pre_gateway_dispatch`` observer is still registered and still records
source IDENTITY (platform/chat_id/parent_chat_id/thread_id, all strings)
keyed by the session key the gateway will use for that source
(``gateway._session_key_for_source(source)`` — the same resolution that binds
``HERMES_SESSION_KEY`` a few steps later). It is retained because it is the
one place the platform is known first-hand, straight off the ``MessageEvent``,
which makes it authoritative for any session whose contextvars later turn out
to be unset or unreadable.

The cache stores IDENTITY, never a derived eligibility bool. A bool baked in
at dispatch time goes stale the instant configuration changes — a session
recorded while the feature was disabled would stay "ineligible" forever after
an operator re-enabled it. Recording identity instead, and re-applying
``policy.discord_scope_enabled`` to it on every
:func:`current_session_eligibility_state` lookup — loading config FRESH each
time — means a config change takes effect on the very next lookup, with no
new dispatch event required. Because identity recording no longer depends on
config at all, :func:`compute_and_cache_eligibility` also can never fail to
record on a transient config-load error.

Tri-state, not boolean. :func:`current_session_eligibility_state` returns
``Optional[bool]``: ``True``/``False`` only when actually confirmed one way or
the other, and ``None`` (UNKNOWN) whenever it cannot be — which ``gate.py``
must treat as fail-closed, never as "not Discord". UNKNOWN is narrower than
it used to be (a Discord thread can now self-determine from its platform
alone) but it is emphatically still needed: a session that is plainly a real
chat — it has a chat id, or a thread id — whose PLATFORM cannot be confirmed
from the contextvars, the session key, or the dispatch cache is exactly the
"Discord identity should be known but the cache is missing" case, and it
fails closed rather than being waved through as non-Discord. So does any
error reading identity, the cache, or config.

Confidently ``False``, by contrast: a confirmed non-Discord platform, and a
session with no chat identity at all (the CLI, which never fires
``pre_gateway_dispatch`` and has no chat/thread id to be unsure about).
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

# Deliberately unbounded: the gateway sees far fewer distinct sessions than
# would ever matter for memory, and the previous bounded-eviction policy was
# itself a vulnerability — an active Discord thread's recorded identity could
# be silently evicted by unrelated sessions' traffic, then read back as a
# confident "not in scope" and drop the write gate. There is no cap and no
# eviction; only an explicit :func:`reset_cache` (tests, or an
# operator-initiated restart) clears it, and a cleared/never-recorded session
# falls back to the contextvar/session-key path below rather than to a
# confident allow.


def _record(session_key: str, identity: Dict[str, str]) -> None:
    if not session_key:
        return
    with _lock:
        _identity_cache[session_key] = dict(identity)


def reset_cache() -> None:
    """Drop every recorded ``pre_gateway_dispatch`` identity.

    For tests, and to model a process restart / cache loss: eligibility then
    has to be re-derived from the session contextvars and the session key,
    and anything that cannot be re-derived reads as UNKNOWN (never ``False``)
    — the safe direction, since ``gate.py`` fails closed on UNKNOWN.
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
    (platform/chat_id/parent_chat_id/thread_id) keyed by the resolved session
    key — never a derived eligibility decision, and therefore never dependent
    on config at all: this always records, regardless of whether the feature
    is currently disabled or its config fails to load. Any error here is
    swallowed; a resolution failure simply means nothing is recorded for this
    dispatch, which :func:`current_session_eligibility_state` handles by
    falling back to the session contextvars (and, failing those, UNKNOWN).
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
        logger.debug("claude-worker: Discord identity recording failed", exc_info=True)
    return None


def _scope_enabled() -> Optional[bool]:
    """Whether the Discord gate is globally on, loading config FRESH.

    ``None`` (UNKNOWN) if config cannot be loaded for this lookup — never a
    silent ``False``, which would drop the gate on a transient config error.
    """
    try:
        from . import config as _cfg_mod

        return bool(_policy.discord_scope_enabled(_cfg_mod.load_plugin_config()))
    except Exception:
        return None


def _eligibility_from_identity(identity: Dict[str, str]) -> Optional[bool]:
    """Apply the CURRENT config to a cached source identity.

    Loads config fresh on every call — never memoized — so enabling or
    disabling the feature takes effect on the very next lookup for an
    already-cached session, with no new dispatch event required.
    """
    if not _policy.is_discord_platform(identity.get("platform", "")):
        return False
    return _scope_enabled()


def _is_gateway_session_key(session_key: str) -> bool:
    """Whether *session_key* is a key the GATEWAY built for a real chat.

    ``build_session_key`` always emits ``agent:<profile>:<platform>:…``, so the
    shape alone says "this is a chat session, not the CLI" even when the
    platform segment itself turns out to be unreadable. That distinction is
    what keeps a malformed gateway key at UNKNOWN (fail closed) instead of
    dropping into the "no chat identity at all" branch and answering ``False``.
    """
    if not isinstance(session_key, str) or not session_key:
        return False
    parts = session_key.split(":")
    return len(parts) >= 4 and parts[0] == "agent"


def _platform_from_session_key(session_key: str) -> str:
    """The platform encoded in a gateway session key, or ``""``.

    ``gateway.session.build_session_key`` always emits
    ``<namespace>:<platform>:...`` with a two-segment ``agent:<profile>``
    namespace, so the key itself carries first-hand platform provenance. This
    is a genuine second source, not a guess: it is what lets a Discord session
    whose contextvars were never bound still be recognised as Discord instead
    of falling into UNKNOWN, and equally lets a confirmed non-Discord session
    be answered ``False`` rather than blocked.
    """
    if not isinstance(session_key, str) or not session_key:
        return ""
    parts = session_key.split(":")
    if len(parts) >= 4 and parts[0] == "agent":
        return parts[2]
    return ""


def _cached_identity(session_key: str) -> Optional[Dict[str, str]]:
    """Recorded dispatch identity for *session_key*, or ``None``.

    Propagates any error from the cache itself — the caller turns that into
    UNKNOWN, never into a confident answer.
    """
    if not session_key:
        return None
    with _lock:
        return _identity_cache.get(session_key)


def current_session_eligibility_state(session_id: str = "") -> Optional[bool]:
    """``pre_tool_call`` entry point — tri-state Discord scope eligibility.

    *session_id* is the session key the dispatcher passed with THIS tool call
    (``pre_tool_call``'s own argument). It is a first-hand source in exactly
    the way the contextvars are, and it is the one that survives a context
    compression / continuation: :func:`gateway.session_context.clear_session_vars`
    resets every ambient var to ``""``, which deliberately suppresses the
    ``os.environ`` fallback, so a live Discord session can arrive here with no
    ambient identity whatsoever. Reading that state as "no chat identity at
    all" answered ``False`` — the CLI's answer — and dropped the gate for a
    session whose Discord key was sitting right there in the arguments. So
    the explicit key is consulted first, both as a cache lookup key and as
    platform provenance in its own right.

    Omitting *session_id* preserves the previous ambient-only resolution
    exactly, for callers that have no explicit key to offer.

    Returns:
      ``True``  — confirmed Discord-origin and the feature is enabled. Any
                  Discord chat qualifies: a guild channel, a DM, or a thread
                  under either (a thread needs no parent lookup now that
                  scope is platform-wide).
      ``False`` — confirmed out of scope: a cached or contextvar-confirmed
                  non-Discord platform, an explicit or ambient session key
                  that names a non-Discord platform, a session with no chat
                  identity at all (CLI), or Discord with the feature globally
                  disabled.
      ``None``  — UNKNOWN, cannot be confirmed either way: a session that is
                  plainly a real chat (it has a chat id and/or a thread id,
                  or a gateway-shaped session key) whose platform is not
                  confirmable from the dispatch cache, the contextvars, or
                  either session key; a config load failure for a
                  Discord-origin session; or any unexpected error reading
                  identity, the cache, or config. Callers MUST fail closed on
                  ``None`` — it is never coerced to ``False``.
    """
    explicit_key = session_id if isinstance(session_id, str) else ""

    try:
        ambient_key = get_session_env("HERMES_SESSION_KEY") or ""
        platform = get_session_env("HERMES_SESSION_PLATFORM") or ""
    except Exception:
        return None

    try:
        explicit_identity = _cached_identity(explicit_key)
        ambient_identity = _cached_identity(ambient_key)
    except Exception:
        return None

    explicit_key_platform = _platform_from_session_key(explicit_key)
    ambient_key_platform = _platform_from_session_key(ambient_key)

    # Any first-hand source naming Discord decides it. Sources are consulted
    # together rather than in a first-hit-wins chain because they can only
    # ever disagree in the direction of one of them having gone missing (a
    # cleared contextvar, an unrecorded dispatch, a stale ambient key
    # inherited by a continuation), and the safe resolution of a disagreement
    # is always "this is Discord, apply the gate".
    for candidate in (
        explicit_identity.get("platform", "") if explicit_identity else "",
        explicit_key_platform,
        ambient_identity.get("platform", "") if ambient_identity else "",
        ambient_key_platform,
        platform,
    ):
        if _policy.is_discord_platform(candidate):
            return _scope_enabled()

    # Nothing names Discord. A recorded dispatch identity is the most
    # specific confirmation available — the explicit key's first, since it
    # describes the call actually being made.
    for identity in (explicit_identity, ambient_identity):
        if identity is not None:
            try:
                return _eligibility_from_identity(identity)
            except Exception:
                return None

    # A non-empty platform from any first-hand source is a confirmed answer,
    # and the answer is "not Discord".
    if explicit_key_platform or platform or ambient_key_platform:
        return False

    try:
        thread_id = str(get_session_env("HERMES_SESSION_THREAD_ID") or "")
        chat_id = str(get_session_env("HERMES_SESSION_CHAT_ID") or "")
    except Exception:
        return None

    if thread_id or chat_id or _is_gateway_session_key(explicit_key):
        # A real chat session whose platform nothing could confirm. This is
        # the "Discord identity should be known but the cache is missing"
        # case — fail closed rather than assume it is some other platform.
        return None

    # No chat identity at all (the CLI never fires pre_gateway_dispatch and
    # has no chat/thread to be uncertain about) — confidently out of scope.
    return False
