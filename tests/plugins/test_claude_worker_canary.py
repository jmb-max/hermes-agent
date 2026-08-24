"""Tests for ``plugins/claude-worker/canary.py``.

Scope is the PLATFORM, not a channel list: EVERY Discord-origin session —
any guild channel, any DM, and every thread under either — is eligible, and
no other platform ever is. The two hardcoded canary channel ids are gone, and
configuration can no longer narrow scope to selected channel ids; the only
thing it may still do is turn the whole feature off globally.

Eligibility stays tri-state and fail-closed. ``True``/``False`` are only
returned when actually confirmed; ``None`` (UNKNOWN) covers a session that is
plainly a real chat whose platform nothing could confirm, and every error
reading identity, the cache, or config. ``gate.py`` must block on ``None``,
so nothing here may coerce it to ``False``.
"""

from __future__ import annotations

import asyncio
import re
from contextvars import copy_context
from dataclasses import dataclass
from typing import Optional
from unittest.mock import patch

import pytest

from gateway.session_context import _UNSET, _VAR_MAP

from tests.plugins._claude_worker_helpers import load_submodule

canary = load_submodule("canary")
policy = load_submodule("policy")

# Deliberately arbitrary ids — none of these is special to the plugin
# anymore, which is the whole point of this change.
CHANNEL_ARBITRARY = "1527706694665113670"
CHANNEL_OTHER = "999999999999999999"
CHANNEL_THIRD = "42"
THREAD_ID = "111222333444555666"


@pytest.fixture(autouse=True)
def _isolated_state(monkeypatch):
    """Clean contextvars + identity cache before/after every test, and pin
    config loading to the plugin's own defaults.

    Tests that read eligibility straight from the contextvars (rather than
    through ``_eligible``) would otherwise load the host's real config.yaml
    off disk, which would make "the feature is on" an environmental accident
    instead of an assertion.
    """
    saved_ctx = {name: var.get() for name, var in _VAR_MAP.items()}
    for var in _VAR_MAP.values():
        var.set(_UNSET)
    canary._identity_cache.clear()
    monkeypatch.setattr(load_submodule("config"), "_load_raw_config", lambda: {})
    yield
    for var, val in zip(_VAR_MAP.values(), saved_ctx.values()):
        var.set(val)
    canary._identity_cache.clear()


@dataclass
class _FakeSource:
    platform: str
    chat_id: str
    parent_chat_id: Optional[str] = None
    thread_id: Optional[str] = None

    class _P:
        def __init__(self, value):
            self.value = value

    def __post_init__(self):
        self.platform = self._P(self.platform)


@dataclass
class _FakeEvent:
    source: _FakeSource


class _FakeGateway:
    def __init__(self, session_key: str):
        self._session_key = session_key

    def _session_key_for_source(self, source):
        return self._session_key


def _default_cfg():
    """The plugin's real default config — the feature is globally on."""
    return {"discord": {"enabled": True}, "canary": {"enabled": True, "channel_ids": []}}


def _eligible(event, session_key, cfg=None):
    """Record identity at dispatch, then look up eligibility with *cfg* as
    the CURRENT config a lookup would load.

    The cache stores identity, never a derived bool, so the config a test
    wants in force must be patched for the LOOKUP, not merely handed to the
    dispatch call.
    """
    effective_cfg = cfg if cfg is not None else _default_cfg()
    canary.compute_and_cache_eligibility(event, _FakeGateway(session_key), cfg=effective_cfg)
    _VAR_MAP["HERMES_SESSION_KEY"].set(session_key)
    config_mod = load_submodule("config")
    with patch.object(config_mod, "load_plugin_config", lambda: effective_cfg):
        return canary.current_session_eligibility_state()


class TestPreGatewayDispatchIsPureObserver:
    def test_returns_none(self):
        event = _FakeEvent(source=_FakeSource(platform="discord", chat_id=CHANNEL_ARBITRARY))
        result = canary.compute_and_cache_eligibility(
            event, _FakeGateway("sess:1"), cfg=_default_cfg(),
        )
        assert result is None

    def test_records_full_source_identity_not_a_bool(self):
        event = _FakeEvent(
            source=_FakeSource(
                platform="discord", chat_id=THREAD_ID,
                parent_chat_id=CHANNEL_ARBITRARY, thread_id=THREAD_ID,
            )
        )
        canary.compute_and_cache_eligibility(event, _FakeGateway("sess:identity"), cfg=_default_cfg())
        assert canary._identity_cache["sess:identity"] == {
            "platform": "discord",
            "chat_id": THREAD_ID,
            "parent_chat_id": CHANNEL_ARBITRARY,
            "thread_id": THREAD_ID,
        }


class TestEveryDiscordChannelIsInScope:
    """The core policy change: an ARBITRARY Discord channel is eligible.
    There is no allowlist to be on."""

    @pytest.mark.parametrize(
        "channel",
        [
            CHANNEL_ARBITRARY,
            CHANNEL_OTHER,
            CHANNEL_THIRD,
            "1501268569697026140",
            "0",
            "dm-with-a-single-user",
        ],
    )
    def test_arbitrary_discord_channel_is_eligible(self, channel):
        event = _FakeEvent(source=_FakeSource(platform="discord", chat_id=channel))
        assert _eligible(event, f"sess:{channel}") is True

    def test_discord_dm_is_eligible(self):
        event = _FakeEvent(source=_FakeSource(platform="discord", chat_id="dm:824719"))
        assert _eligible(event, "sess:dm") is True

    def test_previously_hardcoded_ids_are_no_longer_special(self):
        """Both old canary ids and a never-enrolled id must now behave
        IDENTICALLY — same answer, for the same reason."""
        results = []
        for channel in ("1527706694665113670", "1501268569697026140", CHANNEL_OTHER):
            event = _FakeEvent(source=_FakeSource(platform="discord", chat_id=channel))
            results.append(_eligible(event, f"sess:uniform:{channel}"))
        assert results == [True, True, True]


class TestEveryDiscordThreadIsInScope:
    """A thread's ``chat_id`` is the THREAD id, and ``parent_chat_id`` is not
    a tracked session contextvar — which is exactly why a channel-list check
    silently missed every thread. Platform-wide scope removes the need for a
    parent lookup entirely."""

    @pytest.mark.parametrize("parent", [CHANNEL_ARBITRARY, CHANNEL_OTHER, CHANNEL_THIRD, ""])
    def test_thread_under_any_parent_is_eligible(self, parent):
        event = _FakeEvent(
            source=_FakeSource(
                platform="discord", chat_id=THREAD_ID,
                parent_chat_id=parent or None, thread_id=THREAD_ID,
            )
        )
        assert _eligible(event, f"sess:thread:{parent or 'noparent'}") is True

    def test_thread_needs_no_parent_and_no_cache_when_platform_is_bound(self):
        """The cache-loss case for a thread: identity was never recorded (or
        was reset), but the session's own platform contextvar confirms
        Discord — which is now sufficient on its own."""
        canary._identity_cache.clear()
        _VAR_MAP["HERMES_SESSION_PLATFORM"].set("discord")
        _VAR_MAP["HERMES_SESSION_CHAT_ID"].set(THREAD_ID)
        _VAR_MAP["HERMES_SESSION_THREAD_ID"].set(THREAD_ID)
        _VAR_MAP["HERMES_SESSION_KEY"].set("sess:thread-self-determined")
        assert canary.current_session_eligibility_state() is True

    def test_thread_is_eligible_from_the_session_key_platform_alone(self):
        """No cache, no platform contextvar — but the gateway session key
        itself carries first-hand platform provenance."""
        canary._identity_cache.clear()
        _VAR_MAP["HERMES_SESSION_CHAT_ID"].set(THREAD_ID)
        _VAR_MAP["HERMES_SESSION_THREAD_ID"].set(THREAD_ID)
        _VAR_MAP["HERMES_SESSION_KEY"].set(f"agent:main:discord:{THREAD_ID}")
        assert canary.current_session_eligibility_state() is True


class TestNonDiscordIsNeverInScope:
    @pytest.mark.parametrize(
        "platform", ["telegram", "slack", "whatsapp", "cli", "matrix", "signal", "imessage"],
    )
    def test_other_platforms_are_confidently_false(self, platform):
        event = _FakeEvent(source=_FakeSource(platform=platform, chat_id=CHANNEL_ARBITRARY))
        assert _eligible(event, f"sess:{platform}") is False

    @pytest.mark.parametrize("platform", ["Discord", "DISCORD", "discord ", " discord", "discordapp"])
    def test_near_miss_platform_strings_are_not_discord(self, platform):
        event = _FakeEvent(source=_FakeSource(platform=platform, chat_id=CHANNEL_ARBITRARY))
        assert _eligible(event, f"sess:nearmiss:{platform}") is False

    def test_non_discord_session_key_platform_is_false_without_cache(self):
        canary._identity_cache.clear()
        _VAR_MAP["HERMES_SESSION_CHAT_ID"].set("55555")
        _VAR_MAP["HERMES_SESSION_KEY"].set("agent:main:telegram:55555")
        assert canary.current_session_eligibility_state() is False

    def test_config_cannot_add_a_platform(self):
        cfg = {"discord": {"enabled": True}, "canary": {"platforms": ["telegram"]}}
        event = _FakeEvent(source=_FakeSource(platform="telegram", chat_id=CHANNEL_ARBITRARY))
        assert _eligible(event, "sess:tg-added", cfg=cfg) is False

    def test_cli_with_no_chat_identity_is_confidently_false(self):
        # The CLI never fires pre_gateway_dispatch and has no chat/thread id
        # to be uncertain about — confidently out of scope, not UNKNOWN.
        _VAR_MAP["HERMES_SESSION_KEY"].set("agent:main:local:cli")
        assert canary.current_session_eligibility_state() is False

    def test_no_session_key_and_no_chat_identity_is_false(self):
        assert canary.current_session_eligibility_state() is False


class TestGlobalKillSwitchOnly:
    def test_discord_enabled_false_disables_every_session(self):
        cfg = {"discord": {"enabled": False}}
        event = _FakeEvent(source=_FakeSource(platform="discord", chat_id=CHANNEL_ARBITRARY))
        assert _eligible(event, "sess:killed", cfg=cfg) is False

    def test_deprecated_canary_enabled_false_still_disables(self):
        cfg = {"canary": {"enabled": False, "channel_ids": []}}
        event = _FakeEvent(source=_FakeSource(platform="discord", chat_id=CHANNEL_ARBITRARY))
        assert _eligible(event, "sess:killed-legacy", cfg=cfg) is False

    @pytest.mark.parametrize(
        "channel_ids",
        [[], [CHANNEL_ARBITRARY], [CHANNEL_OTHER], ["1527706694665113670"], ["nonsense"]],
    )
    def test_stale_channel_ids_are_inert_for_every_channel(self, channel_ids):
        """A stale ``canary.channel_ids`` may still be PARSED, but it can no
        longer select a subset: every Discord channel stays in scope whether
        or not it appears in the list."""
        cfg = {"discord": {"enabled": True}, "canary": {"enabled": True, "channel_ids": channel_ids}}
        in_list = _FakeEvent(source=_FakeSource(platform="discord", chat_id=CHANNEL_ARBITRARY))
        not_in_list = _FakeEvent(source=_FakeSource(platform="discord", chat_id=CHANNEL_OTHER))
        assert _eligible(in_list, f"sess:inert-a:{channel_ids}", cfg=cfg) is True
        assert _eligible(not_in_list, f"sess:inert-b:{channel_ids}", cfg=cfg) is True

    def test_missing_config_leaves_the_feature_on(self):
        event = _FakeEvent(source=_FakeSource(platform="discord", chat_id=CHANNEL_ARBITRARY))
        assert _eligible(event, "sess:nocfg", cfg={}) is True

    @pytest.mark.parametrize("bad", [{"discord": None}, {"canary": None}, {"discord": {"enabled": "no"}}])
    def test_malformed_kill_switch_leaves_the_feature_on(self, bad):
        """Only the literal ``False`` disables. A truthy-looking string like
        ``"no"`` must not read as "off" for a safety toggle."""
        event = _FakeEvent(source=_FakeSource(platform="discord", chat_id=CHANNEL_ARBITRARY))
        assert _eligible(event, f"sess:malformed:{bad}", cfg=bad) is True


class TestTriStateFailClosed:
    """UNKNOWN is narrower than it used to be — a Discord thread can now
    self-determine from its platform alone — but it is emphatically still
    needed, and must never be coerced to ``False``."""

    def test_real_chat_with_no_confirmable_platform_is_unknown(self):
        canary._identity_cache.clear()
        _VAR_MAP["HERMES_SESSION_CHAT_ID"].set(THREAD_ID)
        _VAR_MAP["HERMES_SESSION_THREAD_ID"].set(THREAD_ID)
        _VAR_MAP["HERMES_SESSION_KEY"].set("sess:unconfirmable")
        assert canary.current_session_eligibility_state() is None

    def test_chat_id_only_with_no_platform_is_unknown(self):
        canary._identity_cache.clear()
        _VAR_MAP["HERMES_SESSION_CHAT_ID"].set(CHANNEL_ARBITRARY)
        _VAR_MAP["HERMES_SESSION_KEY"].set("sess:chat-only")
        assert canary.current_session_eligibility_state() is None

    def test_missing_session_key_on_a_real_chat_is_unknown(self):
        _VAR_MAP["HERMES_SESSION_CHAT_ID"].set(CHANNEL_ARBITRARY)
        _VAR_MAP["HERMES_SESSION_THREAD_ID"].set(THREAD_ID)
        assert canary.current_session_eligibility_state() is None

    def test_cache_reset_restart_on_an_unconfirmable_chat_is_unknown(self):
        """Simulates a process restart: identity was recorded correctly at
        dispatch time, but the cache backing it is gone by the time
        ``pre_tool_call`` runs, and nothing else confirms the platform."""
        event = _FakeEvent(
            source=_FakeSource(platform="discord", chat_id=THREAD_ID, thread_id=THREAD_ID)
        )
        canary.compute_and_cache_eligibility(event, _FakeGateway("sess:restart"), cfg=_default_cfg())
        assert canary._identity_cache.get("sess:restart") is not None

        canary.reset_cache()
        _VAR_MAP["HERMES_SESSION_CHAT_ID"].set(THREAD_ID)
        _VAR_MAP["HERMES_SESSION_THREAD_ID"].set(THREAD_ID)
        _VAR_MAP["HERMES_SESSION_KEY"].set("sess:restart")
        assert canary.current_session_eligibility_state() is None

    def test_gateway_session_key_failure_leaves_the_chat_unknown(self):
        class _BrokenGateway:
            def _session_key_for_source(self, source):
                raise RuntimeError("gateway session-key resolution exploded")

        event = _FakeEvent(source=_FakeSource(platform="discord", chat_id=THREAD_ID))
        assert canary.compute_and_cache_eligibility(event, _BrokenGateway(), cfg=_default_cfg()) is None

        _VAR_MAP["HERMES_SESSION_CHAT_ID"].set(THREAD_ID)
        _VAR_MAP["HERMES_SESSION_THREAD_ID"].set(THREAD_ID)
        _VAR_MAP["HERMES_SESSION_KEY"].set("sess:key-failure")
        assert canary.current_session_eligibility_state() is None

    def test_config_failure_at_lookup_is_unknown_not_false(self, monkeypatch):
        event = _FakeEvent(source=_FakeSource(platform="discord", chat_id=CHANNEL_ARBITRARY))
        canary.compute_and_cache_eligibility(event, _FakeGateway("sess:cfg-boom"), cfg=_default_cfg())

        config_mod = load_submodule("config")

        def _boom():
            raise RuntimeError("config backend exploded")

        monkeypatch.setattr(config_mod, "load_plugin_config", _boom)
        _VAR_MAP["HERMES_SESSION_KEY"].set("sess:cfg-boom")
        assert canary.current_session_eligibility_state() is None

    def test_config_failure_for_an_uncached_discord_session_is_unknown(self, monkeypatch):
        config_mod = load_submodule("config")

        def _boom():
            raise RuntimeError("config backend exploded")

        monkeypatch.setattr(config_mod, "load_plugin_config", _boom)
        _VAR_MAP["HERMES_SESSION_PLATFORM"].set("discord")
        _VAR_MAP["HERMES_SESSION_CHAT_ID"].set(CHANNEL_ARBITRARY)
        _VAR_MAP["HERMES_SESSION_KEY"].set("sess:cfg-boom-uncached")
        assert canary.current_session_eligibility_state() is None

    def test_cache_lookup_exception_is_unknown_not_false(self, monkeypatch):
        _VAR_MAP["HERMES_SESSION_PLATFORM"].set("discord")
        _VAR_MAP["HERMES_SESSION_CHAT_ID"].set(CHANNEL_ARBITRARY)
        _VAR_MAP["HERMES_SESSION_KEY"].set("sess:cache-boom")

        class _BoomDict(dict):
            def get(self, key, default=None):
                raise RuntimeError("cache backing store exploded")

        monkeypatch.setattr(canary, "_identity_cache", _BoomDict())
        assert canary.current_session_eligibility_state() is None

    def test_session_key_read_failure_is_unknown(self, monkeypatch):
        def _boom(name, default=""):
            raise RuntimeError("session backend exploded")

        monkeypatch.setattr(canary, "get_session_env", _boom)
        assert canary.current_session_eligibility_state() is None

    def test_confirmed_non_discord_is_false_not_unknown(self):
        """Fail-closed must not over-reach: a CONFIRMED other platform is a
        real answer and must allow, or every non-Discord session would be
        gated by a resolver that is merely being careful."""
        _VAR_MAP["HERMES_SESSION_PLATFORM"].set("telegram")
        _VAR_MAP["HERMES_SESSION_CHAT_ID"].set(CHANNEL_ARBITRARY)
        _VAR_MAP["HERMES_SESSION_KEY"].set("sess:tg-confirmed")
        assert canary.current_session_eligibility_state() is False


class TestIdentityCacheRecomputesAgainstCurrentConfig:
    """The cache stores immutable SOURCE IDENTITY, never a derived bool baked
    in against whatever config happened to be loaded at dispatch time. Every
    lookup loads config FRESH, so enabling/disabling takes effect on the very
    next lookup with no new dispatch event required."""

    def _bind(self, session_key: str, chat_id: str) -> None:
        _VAR_MAP["HERMES_SESSION_KEY"].set(session_key)
        _VAR_MAP["HERMES_SESSION_PLATFORM"].set("discord")
        _VAR_MAP["HERMES_SESSION_CHAT_ID"].set(chat_id)

    def test_recorded_while_disabled_then_enabled_becomes_true(self, monkeypatch):
        disabled = {"discord": {"enabled": False}}
        event = _FakeEvent(source=_FakeSource(platform="discord", chat_id=CHANNEL_ARBITRARY))
        canary.compute_and_cache_eligibility(event, _FakeGateway("sess:flip-on"), cfg=disabled)

        config_mod = load_submodule("config")
        monkeypatch.setattr(config_mod, "load_plugin_config", lambda: _default_cfg())
        self._bind("sess:flip-on", CHANNEL_ARBITRARY)
        assert canary.current_session_eligibility_state() is True

    def test_recorded_while_enabled_then_disabled_becomes_false(self, monkeypatch):
        event = _FakeEvent(source=_FakeSource(platform="discord", chat_id=CHANNEL_ARBITRARY))
        canary.compute_and_cache_eligibility(event, _FakeGateway("sess:flip-off"), cfg=_default_cfg())

        config_mod = load_submodule("config")
        monkeypatch.setattr(config_mod, "load_plugin_config", lambda: {"discord": {"enabled": False}})
        self._bind("sess:flip-off", CHANNEL_ARBITRARY)
        assert canary.current_session_eligibility_state() is False

    def test_dispatch_records_even_when_config_is_unavailable(self, monkeypatch):
        """Identity recording never touches config at all, so a transient
        config failure at dispatch time cannot prevent capture."""
        event = _FakeEvent(source=_FakeSource(platform="discord", chat_id=CHANNEL_OTHER))
        canary.compute_and_cache_eligibility(event, _FakeGateway("sess:no-cfg-dispatch"), cfg=None)
        assert canary._identity_cache["sess:no-cfg-dispatch"]["chat_id"] == CHANNEL_OTHER

        config_mod = load_submodule("config")
        monkeypatch.setattr(config_mod, "load_plugin_config", lambda: _default_cfg())
        self._bind("sess:no-cfg-dispatch", CHANNEL_OTHER)
        assert canary.current_session_eligibility_state() is True

    def test_cached_non_discord_identity_is_false_regardless_of_config(self):
        event = _FakeEvent(source=_FakeSource(platform="telegram", chat_id=CHANNEL_ARBITRARY))
        canary.compute_and_cache_eligibility(
            event, _FakeGateway("sess:cached-tg"), cfg=_default_cfg(),
        )
        _VAR_MAP["HERMES_SESSION_KEY"].set("sess:cached-tg")
        assert canary.current_session_eligibility_state() is False


class TestConcurrentSessionsDoNotLeak:
    def test_concurrent_sessions_resolve_independently(self):
        """Mirrors tests/gateway/test_session_context_inheritance.py: two
        concurrent 'sessions' resolve independently even though asyncio tasks
        snapshot (copy_context) the spawning context."""
        canary.compute_and_cache_eligibility(
            _FakeEvent(source=_FakeSource(platform="discord", chat_id=CHANNEL_ARBITRARY)),
            _FakeGateway("sess:concurrent-discord"), cfg=_default_cfg(),
        )
        canary.compute_and_cache_eligibility(
            _FakeEvent(source=_FakeSource(platform="telegram", chat_id=CHANNEL_ARBITRARY)),
            _FakeGateway("sess:concurrent-telegram"), cfg=_default_cfg(),
        )

        results = {}

        def _bind_and_read(session_key: str, out_key: str):
            _VAR_MAP["HERMES_SESSION_KEY"].set(session_key)
            results[out_key] = canary.current_session_eligibility_state()

        async def _run():
            ctx_a = copy_context()
            ctx_b = copy_context()
            loop = asyncio.get_event_loop()
            await asyncio.gather(
                loop.run_in_executor(None, ctx_a.run, _bind_and_read, "sess:concurrent-discord", "a"),
                loop.run_in_executor(None, ctx_b.run, _bind_and_read, "sess:concurrent-telegram", "b"),
            )

        asyncio.run(_run())

        assert results["a"] is True
        assert results["b"] is False


class TestIdentityCacheConcurrency:
    def test_concurrent_record_and_lookup_across_real_threads(self):
        """Many OS threads recording and reading distinct session identities
        at once must never raise, deadlock, or cross-contaminate — the cache
        is guarded by ``canary._lock`` for every read/write."""
        import threading

        errors = []

        def _worker(i: int) -> None:
            try:
                session_key = f"sess:thread-worker:{i}"
                channel = CHANNEL_ARBITRARY if i % 2 == 0 else CHANNEL_OTHER
                event = _FakeEvent(source=_FakeSource(platform="discord", chat_id=channel))
                canary.compute_and_cache_eligibility(
                    event, _FakeGateway(session_key), cfg=_default_cfg(),
                )
                identity = canary._identity_cache.get(session_key)
                if identity is None or identity.get("chat_id") != channel:
                    errors.append(f"identity mismatch for {session_key}: {identity}")
            except Exception as exc:  # pragma: no cover - failure path only
                errors.append(f"{i}: {exc!r}")

        threads = [threading.Thread(target=_worker, args=(i,)) for i in range(200)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        assert canary._identity_cache["sess:thread-worker:0"]["chat_id"] == CHANNEL_ARBITRARY
        assert canary._identity_cache["sess:thread-worker:1"]["chat_id"] == CHANNEL_OTHER

    def test_no_bounded_eviction_past_the_old_4096_cap(self):
        """The old cache evicted the oldest entry at 4096 distinct session
        keys — an active Discord thread could silently lose its recorded
        identity purely from OTHER sessions' traffic, then read back as a
        confident "not in scope". Eviction is gone entirely."""
        canary._identity_cache.clear()
        identity = {
            "platform": "discord", "chat_id": CHANNEL_ARBITRARY,
            "parent_chat_id": "", "thread_id": "",
        }
        for i in range(4200):
            canary._record(f"sess:bulk:{i}", identity)
        assert len(canary._identity_cache) == 4200
        assert canary._identity_cache["sess:bulk:0"] == identity
        assert canary._identity_cache["sess:bulk:4199"] == identity


class TestMalformedEventFailsSafe:
    def test_missing_source_does_not_raise(self):
        class _Empty:
            pass

        assert canary.compute_and_cache_eligibility(
            _Empty(), _FakeGateway("sess:x"), cfg=_default_cfg(),
        ) is None

    def test_plain_string_platform_is_accepted(self):
        """``source.platform`` is normally an enum with ``.value``, but a
        plain string must be handled too rather than recorded as ``""``."""

        class _Source:
            platform = "discord"
            chat_id = CHANNEL_ARBITRARY
            parent_chat_id = None
            thread_id = None

        class _Event:
            source = _Source()

        canary.compute_and_cache_eligibility(_Event(), _FakeGateway("sess:strplat"), cfg=_default_cfg())
        assert canary._identity_cache["sess:strplat"]["platform"] == "discord"

    def test_empty_session_key_records_nothing(self):
        event = _FakeEvent(source=_FakeSource(platform="discord", chat_id=CHANNEL_ARBITRARY))
        canary.compute_and_cache_eligibility(event, _FakeGateway(""), cfg=_default_cfg())
        assert canary._identity_cache == {}


def _clear_every_session_contextvar() -> None:
    """Model what ``gateway.session_context.clear_session_vars`` leaves behind.

    NOT the ``_UNSET`` sentinel the fixture installs: ``clear_session_vars``
    sets every var to ``""``, which deliberately suppresses the ``os.environ``
    fallback. That is the exact state a compressed/continued turn was observed
    in — the session is very much alive, but every ambient contextvar reads
    empty.
    """
    for var in _VAR_MAP.values():
        var.set("")


class TestExplicitSessionIdIsFirstHandProvenance:
    """The live regression.

    After context compression/continuation the ambient session ContextVars
    were cleared to ``""``, so a real Discord session matched the "no chat
    identity at all" branch — the CLI answer — and eligibility came back a
    confident ``False``. Every ``pre_tool_call`` is handed the Discord session
    key EXPLICITLY, so that key is a first-hand source in its own right and
    must be consulted before the contextvars are believed to mean "CLI".
    """

    DISCORD_SESSION_ID = "agent:main:discord:group:1527706694665113670:970735341680082944"
    TELEGRAM_SESSION_ID = "agent:main:telegram:group:55555"
    CLI_SESSION_ID = "agent:main:local:cli"

    def test_a_discord_session_id_is_eligible_with_every_contextvar_cleared(self):
        canary._identity_cache.clear()
        _clear_every_session_contextvar()
        assert canary.current_session_eligibility_state(self.DISCORD_SESSION_ID) is True

    def test_a_recorded_identity_is_found_by_the_explicit_session_id(self):
        event = _FakeEvent(source=_FakeSource(platform="discord", chat_id=THREAD_ID))
        canary.compute_and_cache_eligibility(
            event, _FakeGateway(self.DISCORD_SESSION_ID), cfg=_default_cfg(),
        )
        _clear_every_session_contextvar()
        assert canary.current_session_eligibility_state(self.DISCORD_SESSION_ID) is True

    def test_a_non_discord_session_id_stays_false_with_every_contextvar_cleared(self):
        """Fail-closed must not become fail-dark: the explicit key confirms a
        NON-Discord platform just as first-hand as it confirms Discord."""
        canary._identity_cache.clear()
        _clear_every_session_contextvar()
        assert canary.current_session_eligibility_state(self.TELEGRAM_SESSION_ID) is False

    def test_a_cli_session_id_stays_false_with_every_contextvar_cleared(self):
        canary._identity_cache.clear()
        _clear_every_session_contextvar()
        assert canary.current_session_eligibility_state(self.CLI_SESSION_ID) is False

    def test_a_gateway_shaped_key_with_no_platform_segment_is_unknown(self):
        """A key the gateway plainly built, whose platform segment says
        nothing, is a real chat session that cannot be confirmed — UNKNOWN,
        never the CLI's confident ``False``."""
        canary._identity_cache.clear()
        _clear_every_session_contextvar()
        assert canary.current_session_eligibility_state("agent:main::group:9") is None

    def test_the_explicit_session_id_wins_over_a_stale_ambient_key(self):
        """A continuation can carry a foreign/stale ambient key. The key the
        dispatcher passed with THIS call is the one that decides."""
        canary.compute_and_cache_eligibility(
            _FakeEvent(source=_FakeSource(platform="telegram", chat_id=CHANNEL_OTHER)),
            _FakeGateway("sess:stale-ambient"), cfg=_default_cfg(),
        )
        _clear_every_session_contextvar()
        _VAR_MAP["HERMES_SESSION_KEY"].set("sess:stale-ambient")
        assert canary.current_session_eligibility_state(self.DISCORD_SESSION_ID) is True

    def test_omitting_the_session_id_preserves_the_ambient_behavior(self):
        """The parameter is additive: every existing ambient-only caller keeps
        resolving exactly as it did."""
        canary._identity_cache.clear()
        _VAR_MAP["HERMES_SESSION_PLATFORM"].set("discord")
        _VAR_MAP["HERMES_SESSION_CHAT_ID"].set(CHANNEL_ARBITRARY)
        _VAR_MAP["HERMES_SESSION_KEY"].set("sess:ambient-only")
        assert canary.current_session_eligibility_state() is True

    def test_a_non_string_session_id_is_ignored_not_fatal(self):
        canary._identity_cache.clear()
        _clear_every_session_contextvar()
        assert canary.current_session_eligibility_state(None) is False


class TestNoChannelAllowlistRemainsInTheModule:
    def test_module_never_reads_a_channel_selection_config_key(self):
        source = open(canary.__file__, encoding="utf-8").read()
        for config_key in ('"canary_channel_ids"', "'canary_channel_ids'",
                           '"canary_platforms"', "'canary_platforms'",
                           '"channel_ids"', "'channel_ids'",
                           '"platforms"', "'platforms'"):
            assert config_key not in source
        # The removed narrowing helper must be gone, not merely unused.
        assert "enabled_canary_channel_ids" not in source
        assert "_policy.is_discord_platform(" in source
        assert "_policy.discord_scope_enabled(" in source

    def test_module_contains_no_channel_id_literals(self):
        """The old two ids are gone; a second copy anywhere could drift back
        into being a de-facto allowlist."""
        source = open(canary.__file__, encoding="utf-8").read()
        assert re.search(r"\d{17,20}", source) is None
