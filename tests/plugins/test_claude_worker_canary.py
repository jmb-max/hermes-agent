"""Tests for ``plugins/claude-worker/canary.py``.

Canary eligibility is decided by the immutable policy layer: platform
``discord`` and exactly the two fixed channel ids, including threads (whose
``chat_id`` is the thread id, so ``parent_chat_id`` must be consulted —
captured at ``pre_gateway_dispatch``, since ``pre_tool_call`` never sees it).
Configuration may disable the feature or select a subset; it can never add a
channel or a platform. Everything else fails dark (ineligible).
"""

from __future__ import annotations

import asyncio
from contextvars import copy_context
from dataclasses import dataclass
from typing import Optional
from unittest.mock import patch

import pytest

from gateway.session_context import _UNSET, _VAR_MAP

from tests.plugins._claude_worker_helpers import load_submodule

canary = load_submodule("canary")
policy = load_submodule("policy")

CHANNEL_A = "1527706694665113670"
CHANNEL_B = "1501268569697026140"


@pytest.fixture(autouse=True)
def _isolated_state(monkeypatch):
    """Clean contextvars + canary cache before/after every test."""
    saved_ctx = {name: var.get() for name, var in _VAR_MAP.items()}
    for var in _VAR_MAP.values():
        var.set(_UNSET)
    canary._identity_cache.clear()
    yield
    for var, val in zip(_VAR_MAP.values(), saved_ctx.values()):
        var.set(val)
    canary._identity_cache.clear()


@dataclass
class _FakeSource:
    platform: str
    chat_id: str
    parent_chat_id: Optional[str] = None

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
    """The plugin's real default config — no canary keys set at all."""
    return {"canary": {"enabled": True, "channel_ids": []}}


def _eligible(event, session_key, cfg=None):
    """Record identity at dispatch, then look up eligibility with *cfg*
    (default: the plugin's real default config) as the CURRENT config a
    lookup would load — the identity cache no longer bakes config into the
    recorded decision, so the config a test wants in force must be patched
    for the lookup itself, not merely handed to the dispatch call."""
    effective_cfg = cfg if cfg is not None else _default_cfg()
    canary.compute_and_cache_eligibility(event, _FakeGateway(session_key), cfg=effective_cfg)
    _VAR_MAP["HERMES_SESSION_KEY"].set(session_key)
    config_mod = load_submodule("config")
    with patch.object(config_mod, "load_plugin_config", lambda: effective_cfg):
        return canary.current_session_eligibility_state()


class TestPreGatewayDispatchIsPureObserver:
    def test_returns_none(self):
        event = _FakeEvent(source=_FakeSource(platform="discord", chat_id=CHANNEL_A))
        result = canary.compute_and_cache_eligibility(event, _FakeGateway("sess:1"), cfg=_default_cfg())
        assert result is None


class TestFixedChannelPolicy:
    @pytest.mark.parametrize("channel", [CHANNEL_A, CHANNEL_B])
    def test_both_required_channels_are_eligible_on_defaults(self, channel):
        event = _FakeEvent(source=_FakeSource(platform="discord", chat_id=channel))
        assert _eligible(event, f"sess:{channel}") is True

    @pytest.mark.parametrize(
        "channel",
        ["999999999999999999", "1527706694665113671", "152770669466511367", "", "0"],
    )
    def test_every_other_channel_is_ineligible(self, channel):
        event = _FakeEvent(source=_FakeSource(platform="discord", chat_id=channel))
        assert _eligible(event, f"sess:other:{channel}") is False

    def test_config_cannot_add_a_channel(self):
        cfg = {"canary": {"enabled": True, "channel_ids": ["999999999999999999"]}}
        event = _FakeEvent(source=_FakeSource(platform="discord", chat_id="999999999999999999"))
        assert _eligible(event, "sess:added", cfg=cfg) is False

    def test_config_may_select_a_subset(self):
        cfg = {"canary": {"enabled": True, "channel_ids": [CHANNEL_A]}}
        event_a = _FakeEvent(source=_FakeSource(platform="discord", chat_id=CHANNEL_A))
        event_b = _FakeEvent(source=_FakeSource(platform="discord", chat_id=CHANNEL_B))
        assert _eligible(event_a, "sess:subset-a", cfg=cfg) is True
        assert _eligible(event_b, "sess:subset-b", cfg=cfg) is False

    def test_config_may_disable_the_feature(self):
        cfg = {"canary": {"enabled": False, "channel_ids": []}}
        event = _FakeEvent(source=_FakeSource(platform="discord", chat_id=CHANNEL_A))
        assert _eligible(event, "sess:disabled", cfg=cfg) is False

    def test_missing_config_still_enforces_policy_channels(self):
        event = _FakeEvent(source=_FakeSource(platform="discord", chat_id=CHANNEL_A))
        assert _eligible(event, "sess:nocfg", cfg={}) is True


class TestThreadEligibility:
    @pytest.mark.parametrize("parent", [CHANNEL_A, CHANNEL_B])
    def test_thread_under_each_canary_channel_is_eligible(self, parent):
        event = _FakeEvent(
            source=_FakeSource(platform="discord", chat_id="999888777666555444", parent_chat_id=parent)
        )
        assert _eligible(event, f"sess:thread:{parent}") is True

    def test_thread_under_non_canary_parent_is_ineligible(self):
        event = _FakeEvent(
            source=_FakeSource(platform="discord", chat_id="1111", parent_chat_id="2222")
        )
        assert _eligible(event, "sess:thread2") is False

    def test_config_cannot_add_a_parent_channel(self):
        cfg = {"canary": {"enabled": True, "channel_ids": ["2222"]}}
        event = _FakeEvent(source=_FakeSource(platform="discord", chat_id="1111", parent_chat_id="2222"))
        assert _eligible(event, "sess:thread3", cfg=cfg) is False


class TestPlatformPolicy:
    @pytest.mark.parametrize("platform", ["telegram", "slack", "whatsapp", "cli", "matrix", ""])
    def test_non_discord_platform_never_eligible_even_with_matching_chat_id(self, platform):
        event = _FakeEvent(source=_FakeSource(platform=platform, chat_id=CHANNEL_A))
        assert _eligible(event, f"sess:{platform or 'blank'}") is False

    def test_config_cannot_add_a_platform(self):
        cfg = {"canary": {"enabled": True, "channel_ids": [], "platforms": ["telegram"]}}
        event = _FakeEvent(source=_FakeSource(platform="telegram", chat_id=CHANNEL_A))
        assert _eligible(event, "sess:tg", cfg=cfg) is False

    def test_cli_never_eligible(self):
        # CLI never fires pre_gateway_dispatch, so nothing populates the
        # cache for its session key — deny by default.
        _VAR_MAP["HERMES_SESSION_KEY"].set("agent:main:local:cli")
        assert canary.current_session_eligibility_state() is False

    def test_no_session_key_bound_is_ineligible(self):
        assert canary.current_session_eligibility_state() is False


class TestConcurrentSessionsDoNotLeak:
    def test_concurrent_sessions_do_not_leak_eligibility(self):
        """Mirrors tests/gateway/test_session_context_inheritance.py: two
        concurrent 'sessions' resolve independently even though asyncio
        tasks snapshot (copy_context) the spawning context."""

        eligible_event = _FakeEvent(source=_FakeSource(platform="discord", chat_id=CHANNEL_A))
        canary.compute_and_cache_eligibility(
            eligible_event, _FakeGateway("sess:concurrent-eligible"), cfg=_default_cfg()
        )
        ineligible_event = _FakeEvent(source=_FakeSource(platform="discord", chat_id="000"))
        canary.compute_and_cache_eligibility(
            ineligible_event, _FakeGateway("sess:concurrent-ineligible"), cfg=_default_cfg()
        )

        results = {}

        def _bind_and_read(session_key: str, out_key: str):
            _VAR_MAP["HERMES_SESSION_KEY"].set(session_key)
            results[out_key] = canary.current_session_eligibility_state()

        async def _run():
            ctx_a = copy_context()
            ctx_b = copy_context()
            task_a = asyncio.get_event_loop().run_in_executor(
                None, ctx_a.run, _bind_and_read, "sess:concurrent-eligible", "a"
            )
            task_b = asyncio.get_event_loop().run_in_executor(
                None, ctx_b.run, _bind_and_read, "sess:concurrent-ineligible", "b"
            )
            await asyncio.gather(task_a, task_b)

        asyncio.run(_run())

        assert results["a"] is True
        assert results["b"] is False


class TestMalformedEventFailsDark:
    def test_missing_source_does_not_raise(self):
        class _Empty:
            pass

        assert canary.compute_and_cache_eligibility(
            _Empty(), _FakeGateway("sess:x"), cfg=_default_cfg()
        ) is None

    def test_gateway_key_resolution_error_does_not_raise(self):
        class _BrokenGateway:
            def _session_key_for_source(self, source):
                raise RuntimeError("boom")

        event = _FakeEvent(source=_FakeSource(platform="discord", chat_id=CHANNEL_A))
        assert canary.compute_and_cache_eligibility(event, _BrokenGateway(), cfg=_default_cfg()) is None

    def test_broken_config_still_enforces_policy(self, monkeypatch):
        event = _FakeEvent(source=_FakeSource(platform="discord", chat_id=CHANNEL_A))
        assert _eligible(event, "sess:brokencfg", cfg={"canary": None}) is True


class TestTriStateEligibilityCacheLoss:
    """The cache-loss finding: a direct (non-thread) Discord channel must
    resolve its OWN eligibility from ``HERMES_SESSION_PLATFORM`` /
    ``HERMES_SESSION_CHAT_ID`` — never solely from the ``pre_gateway_dispatch``
    cache. A Discord THREAD cannot self-determine (its ``chat_id`` is the
    thread id, not the parent channel) and so must return ``None`` (UNKNOWN)
    on any cache miss — never silently downgrade to ``False`` (which the old
    deny-by-default cache did, letting a genuine canary thread session fall
    through with the gate wide open the instant the cache lost the entry).
    """

    def test_direct_channel_still_eligible_after_cache_reset(self):
        _VAR_MAP["HERMES_SESSION_PLATFORM"].set("discord")
        _VAR_MAP["HERMES_SESSION_CHAT_ID"].set(CHANNEL_A)
        _VAR_MAP["HERMES_SESSION_THREAD_ID"].set("")
        _VAR_MAP["HERMES_SESSION_KEY"].set("sess:direct-no-cache")
        canary._identity_cache.clear()
        assert canary.current_session_eligibility_state() is True

    def test_direct_channel_non_canary_is_false_without_cache(self):
        _VAR_MAP["HERMES_SESSION_PLATFORM"].set("discord")
        _VAR_MAP["HERMES_SESSION_CHAT_ID"].set("000000000000000000")
        _VAR_MAP["HERMES_SESSION_THREAD_ID"].set("")
        _VAR_MAP["HERMES_SESSION_KEY"].set("sess:direct-non-canary-no-cache")
        canary._identity_cache.clear()
        assert canary.current_session_eligibility_state() is False

    def test_non_discord_platform_is_false_without_cache(self):
        _VAR_MAP["HERMES_SESSION_PLATFORM"].set("telegram")
        _VAR_MAP["HERMES_SESSION_CHAT_ID"].set(CHANNEL_A)
        _VAR_MAP["HERMES_SESSION_KEY"].set("sess:telegram-no-cache")
        canary._identity_cache.clear()
        assert canary.current_session_eligibility_state() is False

    def test_thread_with_cache_miss_is_unknown_not_false(self):
        _VAR_MAP["HERMES_SESSION_PLATFORM"].set("discord")
        _VAR_MAP["HERMES_SESSION_CHAT_ID"].set("999888777666555444")
        _VAR_MAP["HERMES_SESSION_THREAD_ID"].set("999888777666555444")
        _VAR_MAP["HERMES_SESSION_KEY"].set("sess:thread-cache-miss")
        canary._identity_cache.clear()
        assert canary.current_session_eligibility_state() is None

    def test_thread_cache_reset_after_dispatch_is_unknown(self):
        """Simulates a process restart / cache eviction: the thread's source
        identity was correctly recorded at pre_gateway_dispatch time, but the
        cache backing it is gone by the time pre_tool_call runs."""
        event = _FakeEvent(
            source=_FakeSource(platform="discord", chat_id="111222333444555666", parent_chat_id=CHANNEL_A)
        )
        canary.compute_and_cache_eligibility(event, _FakeGateway("sess:thread-restart"), cfg=_default_cfg())
        recorded = canary._identity_cache.get("sess:thread-restart")
        assert recorded == {
            "platform": "discord",
            "chat_id": "111222333444555666",
            "parent_chat_id": CHANNEL_A,
            "thread_id": "",
        }

        canary.reset_cache()

        _VAR_MAP["HERMES_SESSION_PLATFORM"].set("discord")
        _VAR_MAP["HERMES_SESSION_CHAT_ID"].set("111222333444555666")
        _VAR_MAP["HERMES_SESSION_THREAD_ID"].set("111222333444555666")
        _VAR_MAP["HERMES_SESSION_KEY"].set("sess:thread-restart")
        assert canary.current_session_eligibility_state() is None

    def test_gateway_session_key_failure_leaves_thread_unknown(self):
        """If pre_gateway_dispatch could not resolve the session key (the
        gateway's own resolution raised), nothing is ever recorded for the
        key the thread is eventually bound to — must read as UNKNOWN, not
        as a confirmed non-canary session."""

        class _BrokenGateway:
            def _session_key_for_source(self, source):
                raise RuntimeError("gateway session-key resolution exploded")

        event = _FakeEvent(
            source=_FakeSource(platform="discord", chat_id="777666555444333222", parent_chat_id=CHANNEL_B)
        )
        canary.compute_and_cache_eligibility(event, _BrokenGateway(), cfg=_default_cfg())

        _VAR_MAP["HERMES_SESSION_PLATFORM"].set("discord")
        _VAR_MAP["HERMES_SESSION_CHAT_ID"].set("777666555444333222")
        _VAR_MAP["HERMES_SESSION_THREAD_ID"].set("777666555444333222")
        _VAR_MAP["HERMES_SESSION_KEY"].set("sess:thread-key-failure")
        assert canary.current_session_eligibility_state() is None

    def test_missing_session_key_on_a_thread_is_unknown(self):
        _VAR_MAP["HERMES_SESSION_PLATFORM"].set("discord")
        _VAR_MAP["HERMES_SESSION_CHAT_ID"].set("333222111000999888")
        _VAR_MAP["HERMES_SESSION_THREAD_ID"].set("333222111000999888")
        assert canary.current_session_eligibility_state() is None

    def test_cache_lookup_exception_is_unknown_not_false(self, monkeypatch):
        _VAR_MAP["HERMES_SESSION_PLATFORM"].set("discord")
        _VAR_MAP["HERMES_SESSION_CHAT_ID"].set("444555666777888999")
        _VAR_MAP["HERMES_SESSION_THREAD_ID"].set("444555666777888999")
        _VAR_MAP["HERMES_SESSION_KEY"].set("sess:cache-lookup-boom")

        class _BoomDict(dict):
            def get(self, key, default=None):
                raise RuntimeError("cache backing store exploded")

        monkeypatch.setattr(canary, "_identity_cache", _BoomDict())
        assert canary.current_session_eligibility_state() is None

    def test_no_bounded_eviction_past_the_old_4096_cap(self):
        """The old cache evicted the oldest entry once it hit 4096 distinct
        session keys — a long-lived canary thread could silently lose its
        recorded decision (and read back as ineligible) purely from OTHER
        sessions' traffic. Eviction is gone entirely."""
        canary._identity_cache.clear()
        identity = {"platform": "discord", "chat_id": CHANNEL_A, "parent_chat_id": "", "thread_id": ""}
        for i in range(4200):
            canary._record(f"sess:bulk:{i}", identity)
        assert len(canary._identity_cache) == 4200
        assert canary._identity_cache.get("sess:bulk:0") == identity
        assert canary._identity_cache.get("sess:bulk:4199") == identity


class TestIdentityCacheRecomputesAgainstCurrentConfig:
    """The cache stores immutable SOURCE IDENTITY (platform/chat_id/
    parent_chat_id/thread_id) recorded at ``pre_gateway_dispatch`` time —
    never a derived eligibility bool baked in against whatever config
    happened to be loaded at THAT moment. Every
    ``current_session_eligibility_state`` lookup loads the CURRENT config
    fresh and re-applies ``policy.enabled_canary_channel_ids`` to the cached
    identity, so a stale decision from an earlier config can never linger:
    disabling/narrowing/re-enabling the canary takes effect on the very next
    lookup, with no new dispatch event required.
    """

    def _thread_event(self, thread_chat_id: str, parent: str) -> _FakeEvent:
        return _FakeEvent(
            source=_FakeSource(platform="discord", chat_id=thread_chat_id, parent_chat_id=parent)
        )

    def _bind_thread_session(self, session_key: str, thread_chat_id: str) -> None:
        _VAR_MAP["HERMES_SESSION_KEY"].set(session_key)
        _VAR_MAP["HERMES_SESSION_PLATFORM"].set("discord")
        _VAR_MAP["HERMES_SESSION_CHAT_ID"].set(thread_chat_id)
        _VAR_MAP["HERMES_SESSION_THREAD_ID"].set(thread_chat_id)

    def test_thread_recorded_while_disabled_then_enabled_becomes_true(self, monkeypatch):
        disabled_cfg = {"canary": {"enabled": False, "channel_ids": []}}
        event = self._thread_event("thread:disable-then-enable", CHANNEL_A)
        canary.compute_and_cache_eligibility(
            event, _FakeGateway("sess:disable-then-enable"), cfg=disabled_cfg,
        )

        config_mod = load_submodule("config")
        monkeypatch.setattr(config_mod, "load_plugin_config", lambda: _default_cfg())

        self._bind_thread_session("sess:disable-then-enable", "thread:disable-then-enable")
        assert canary.current_session_eligibility_state() is True

    def test_thread_recorded_in_scope_then_config_narrows_becomes_false(self, monkeypatch):
        wide_cfg = {"canary": {"enabled": True, "channel_ids": [CHANNEL_A]}}
        event = self._thread_event("thread:narrowed", CHANNEL_A)
        canary.compute_and_cache_eligibility(
            event, _FakeGateway("sess:narrowed"), cfg=wide_cfg,
        )

        config_mod = load_submodule("config")
        narrow_cfg = {"canary": {"enabled": True, "channel_ids": [CHANNEL_B]}}
        monkeypatch.setattr(config_mod, "load_plugin_config", lambda: narrow_cfg)

        self._bind_thread_session("sess:narrowed", "thread:narrowed")
        assert canary.current_session_eligibility_state() is False

    def test_transient_config_failure_at_dispatch_then_valid_lookup_computes_true(self, monkeypatch):
        """``pre_gateway_dispatch`` never touches config at all now — the
        identity is recorded unconditionally — so even a caller that would
        have failed to load config for this dispatch cannot prevent the
        identity from being captured; the next lookup, with a working
        config, computes eligibility fresh."""
        event = self._thread_event("thread:transient-cfg-failure", CHANNEL_B)
        canary.compute_and_cache_eligibility(
            event, _FakeGateway("sess:transient-cfg-failure"), cfg=None,
        )
        assert canary._identity_cache.get("sess:transient-cfg-failure") == {
            "platform": "discord",
            "chat_id": "thread:transient-cfg-failure",
            "parent_chat_id": CHANNEL_B,
            "thread_id": "",
        }

        config_mod = load_submodule("config")
        monkeypatch.setattr(config_mod, "load_plugin_config", lambda: _default_cfg())

        self._bind_thread_session("sess:transient-cfg-failure", "thread:transient-cfg-failure")
        assert canary.current_session_eligibility_state() is True

    def test_config_failure_at_lookup_is_unknown_not_false(self, monkeypatch):
        event = self._thread_event("thread:lookup-cfg-failure", CHANNEL_A)
        canary.compute_and_cache_eligibility(
            event, _FakeGateway("sess:lookup-cfg-failure"), cfg=_default_cfg(),
        )

        config_mod = load_submodule("config")

        def _boom():
            raise RuntimeError("config backend exploded")

        monkeypatch.setattr(config_mod, "load_plugin_config", _boom)

        self._bind_thread_session("sess:lookup-cfg-failure", "thread:lookup-cfg-failure")
        assert canary.current_session_eligibility_state() is None

    def test_cached_non_discord_identity_is_false_regardless_of_config(self, monkeypatch):
        event = _FakeEvent(source=_FakeSource(platform="telegram", chat_id=CHANNEL_A))
        canary.compute_and_cache_eligibility(
            event, _FakeGateway("sess:cached-non-discord"), cfg=_default_cfg(),
        )
        _VAR_MAP["HERMES_SESSION_KEY"].set("sess:cached-non-discord")
        assert canary.current_session_eligibility_state() is False


class TestIdentityCacheConcurrency:
    def test_concurrent_record_and_lookup_across_real_threads_do_not_corrupt_state(self):
        """Many OS threads recording and reading distinct session identities
        at once must never raise, deadlock, or cross-contaminate results —
        the identity cache is guarded by ``canary._lock`` for every
        read/write, not just the asyncio-contextvar scenario covered by
        ``TestConcurrentSessionsDoNotLeak``."""
        import threading

        errors = []

        def _worker(i: int) -> None:
            try:
                session_key = f"sess:thread-worker:{i}"
                channel = CHANNEL_A if i % 2 == 0 else "000000000000000000"
                event = _FakeEvent(source=_FakeSource(platform="discord", chat_id=channel))
                canary.compute_and_cache_eligibility(
                    event, _FakeGateway(session_key), cfg=_default_cfg(),
                )
                identity = canary._identity_cache.get(session_key)
                expected_chat_id = channel
                if identity is None or identity.get("chat_id") != expected_chat_id:
                    errors.append(f"identity mismatch for {session_key}: {identity}")
            except Exception as exc:  # pragma: no cover - failure path only
                errors.append(f"{i}: {exc!r}")

        threads = [threading.Thread(target=_worker, args=(i,)) for i in range(200)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        assert canary._identity_cache.get("sess:thread-worker:0", {}).get("chat_id") == CHANNEL_A
        assert canary._identity_cache.get("sess:thread-worker:1", {}).get("chat_id") == "000000000000000000"


class TestNoConfigurableChannelSourceInModule:
    def test_module_never_reads_channel_ids_from_a_config_key(self):
        """Scope may only come from the policy layer — never from a key this
        module pulls out of the config mapping itself."""
        source = open(canary.__file__, encoding="utf-8").read()
        for config_key in ('"canary_channel_ids"', "'canary_channel_ids'",
                           '"canary_platforms"', "'canary_platforms'",
                           '"channel_ids"', "'channel_ids'",
                           '"platforms"', "'platforms'"):
            assert config_key not in source
        assert "_policy.enabled_canary_channel_ids(cfg)" in source
        assert "_policy.is_canary_platform(" in source

    def test_module_contains_no_channel_id_literals(self):
        """The ids live in policy.py only; a second copy here could drift."""
        import re

        source = open(canary.__file__, encoding="utf-8").read()
        assert re.search(r"\d{17,20}", source) is None
