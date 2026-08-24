"""The ``pre_tool_call`` / ``post_tool_call`` dispatch identity contract.

A gateway chat session has TWO identities and they are not interchangeable:

* ``agent.session_id`` — the TRANSCRIPT id (``20260823_101500_ab12cd``). It is
  per-conversation-slice, not per-chat: context compression mints a brand new
  one mid-conversation (``agent/conversation_compression.py`` rotates it and
  re-binds the contextvar), and it carries no platform provenance at all.
* ``agent._gateway_session_key`` — the STABLE per-chat key the gateway built
  (``agent:main:discord:group:…``). It survives compression, and its own
  ``<namespace>:<platform>:…`` shape is first-hand platform provenance.

Every ``pre_tool_call`` dispatch site used to hand the plugin layer only the
first of those, under the parameter name ``session_id``. A plugin that has to
answer "which CHAT is this call from?" — ``plugins/claude-worker`` — therefore
received an id that can never name a platform, and after a compression /
continuation (every ambient session ContextVar cleared to ``""``) it had no
other source left either. That is the Discord continuation bypass: the write
gate and the terminal guard both dropped for a live Discord session.

These tests pin the runtime contract that closes it, from the REAL agent
tool-execution paths (sequential, concurrent, ``_invoke_tool``) rather than
from a hand-built hook payload:

1. the transcript ``session_id`` is passed through UNCHANGED — nothing is
   substituted for it, so every existing consumer (observability, session
   correlation, middleware) keeps seeing the id it already sees; and
2. the gateway key is passed ALONGSIDE it as a SEPARATE ``gateway_session_key``
   argument, sourced from ``agent._gateway_session_key``; and
3. ``post_tool_call`` carries the same pair, so a plugin that authorizes on a
   tool RESULT keys that authorization by chat rather than by transcript
   slice.

Compatibility is part of the contract: a CLI agent has no gateway key, and the
dispatch must still fire with an empty one rather than inventing a value or
falling over.
"""

from __future__ import annotations

import json
import uuid
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import hermes_cli.plugins as hermes_plugins
from hermes_cli.plugins import (
    PluginContext,
    PluginManager,
    PluginManifest,
    resolve_pre_tool_block,
)
from run_agent import AIAgent

#: A transcript session id, in the exact shape ``agent_init`` mints
#: (``<timestamp>_<short uuid>``). Deliberately NOT gateway-key shaped.
TRANSCRIPT_SESSION_ID = "20260823_101500_ab12cd"

#: The stable per-chat gateway key for the same session — the Discord key from
#: the incident, verbatim in shape.
DISCORD_GATEWAY_KEY = "agent:main:discord:group:1527706694665113670:970735341680082944"


def _make_tool_defs(*names: str) -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": f"{name} tool",
                "parameters": {"type": "object", "properties": {}},
            },
        }
        for name in names
    ]


def _make_agent(*, gateway_session_key: str | None = DISCORD_GATEWAY_KEY) -> AIAgent:
    """A real ``AIAgent`` carrying both identities, as the gateway builds it."""
    with (
        patch("run_agent.get_tool_definitions", return_value=_make_tool_defs("patch", "terminal", "web_search")),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("hermes_cli.config.load_config", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            session_id=TRANSCRIPT_SESSION_ID,
            platform="discord" if gateway_session_key else "cli",
            gateway_session_key=gateway_session_key,
        )
    agent.client = MagicMock()
    agent.tool_delay = 0
    agent.compression_enabled = False
    agent.save_trajectories = False
    return agent


def _tool_call(name: str, args: dict, call_id: str | None = None):
    return SimpleNamespace(
        id=call_id or f"call_{uuid.uuid4().hex[:8]}",
        type="function",
        function=SimpleNamespace(name=name, arguments=json.dumps(args)),
    )


def _recording_pre_tool_block(records: list):
    """Stand in for ``resolve_pre_tool_block`` and record what it was handed.

    Returns a block message so no real tool executes — the dispatch identity is
    the whole subject here, not the tool's behavior. ``**kwargs`` is deliberate:
    a missing ``gateway_session_key`` shows up as an absent key (a failed
    assertion) rather than as a TypeError swallowed by a hook wrapper.
    """

    def _resolve(tool_name, args=None, **kwargs):
        records.append({"tool_name": tool_name, "args": args, **kwargs})
        return "blocked by the recording pre_tool_call stub"

    return _resolve


@pytest.fixture()
def clean_plugin_manager():
    """A PluginManager with nothing but the hooks a test registers itself."""
    saved = hermes_plugins._plugin_manager
    manager = PluginManager()
    manager._discovered = True
    hermes_plugins._plugin_manager = manager
    ctx = PluginContext(
        PluginManifest(name="identity-observer", key="identity-observer", source="bundled"),
        manager,
    )
    try:
        yield ctx
    finally:
        hermes_plugins._plugin_manager = saved


# ---------------------------------------------------------------------------
# A. pre_tool_call, from the real agent tool-execution paths
# ---------------------------------------------------------------------------


class TestPreToolCallCarriesBothIdentities:
    """Each dispatch site passes the transcript id UNCHANGED and the gateway
    key as its own argument. All three sites are asserted because the bypass
    only needs ONE of them to keep handing over a transcript id alone."""

    def test_the_sequential_path_passes_both(self):
        agent = _make_agent()
        records: list = []
        message = SimpleNamespace(
            content="", tool_calls=[_tool_call("patch", {"path": "/repo/app.py"}, "c-seq")],
        )

        with patch("hermes_cli.plugins.resolve_pre_tool_block", _recording_pre_tool_block(records)):
            agent._execute_tool_calls_sequential(message, [], "task-seq")

        assert len(records) == 1
        assert records[0]["session_id"] == TRANSCRIPT_SESSION_ID
        assert records[0]["gateway_session_key"] == DISCORD_GATEWAY_KEY

    def test_the_concurrent_path_passes_both_for_every_call_in_the_batch(self):
        agent = _make_agent()
        records: list = []
        message = SimpleNamespace(
            content="",
            tool_calls=[
                _tool_call("patch", {"path": "/repo/app.py"}, "c-con-1"),
                _tool_call("terminal", {"command": "claude -p 'fix it'"}, "c-con-2"),
            ],
        )

        with patch("hermes_cli.plugins.resolve_pre_tool_block", _recording_pre_tool_block(records)):
            agent._execute_tool_calls_concurrent(message, [], "task-con")

        assert [r["tool_name"] for r in records] == ["patch", "terminal"]
        for record in records:
            assert record["session_id"] == TRANSCRIPT_SESSION_ID
            assert record["gateway_session_key"] == DISCORD_GATEWAY_KEY

    def test_the_invoke_tool_path_passes_both(self):
        agent = _make_agent()
        records: list = []

        with patch("hermes_cli.plugins.resolve_pre_tool_block", _recording_pre_tool_block(records)):
            agent._invoke_tool("patch", {"path": "/repo/app.py"}, "task-inv", tool_call_id="c-inv")

        assert len(records) == 1
        assert records[0]["session_id"] == TRANSCRIPT_SESSION_ID
        assert records[0]["gateway_session_key"] == DISCORD_GATEWAY_KEY

    def test_the_gateway_key_is_added_and_never_substituted_for_the_transcript_id(self):
        """The two identities travel together. Passing the gateway key AS
        ``session_id`` would look like a fix and would silently break every
        consumer that correlates on the transcript id."""
        agent = _make_agent()
        records: list = []
        message = SimpleNamespace(
            content="", tool_calls=[_tool_call("patch", {"path": "/repo/app.py"}, "c-both")],
        )

        with patch("hermes_cli.plugins.resolve_pre_tool_block", _recording_pre_tool_block(records)):
            agent._execute_tool_calls_sequential(message, [], "task-both")

        assert records[0]["session_id"] == agent.session_id
        assert records[0]["session_id"] != DISCORD_GATEWAY_KEY
        assert records[0]["gateway_session_key"] != records[0]["session_id"]

    def test_a_cli_agent_with_no_gateway_key_still_dispatches(self):
        """Compatibility: the CLI has no gateway session key at all. The
        dispatch must still fire, with the transcript id untouched and an empty
        gateway key — not a fabricated one, and not an exception."""
        agent = _make_agent(gateway_session_key=None)
        records: list = []
        message = SimpleNamespace(
            content="", tool_calls=[_tool_call("patch", {"path": "/repo/app.py"}, "c-cli")],
        )

        with patch("hermes_cli.plugins.resolve_pre_tool_block", _recording_pre_tool_block(records)):
            agent._execute_tool_calls_sequential(message, [], "task-cli")

        assert len(records) == 1
        assert records[0]["session_id"] == TRANSCRIPT_SESSION_ID
        assert records[0].get("gateway_session_key") in ("", None)


class TestPluginsSeeBothIdentities:
    """The plugin boundary itself: ``resolve_pre_tool_block`` must forward the
    gateway key into the ``pre_tool_call`` hook payload. A dispatch site that
    passes it to a function which then drops it fixes nothing."""

    def test_the_hook_payload_carries_both(self, clean_plugin_manager):
        payloads: list = []
        clean_plugin_manager.register_hook(
            "pre_tool_call", lambda **kwargs: payloads.append(kwargs),
        )

        resolve_pre_tool_block(
            "patch",
            {"path": "/repo/app.py"},
            session_id=TRANSCRIPT_SESSION_ID,
            gateway_session_key=DISCORD_GATEWAY_KEY,
        )

        assert len(payloads) == 1
        assert payloads[0]["session_id"] == TRANSCRIPT_SESSION_ID
        assert payloads[0]["gateway_session_key"] == DISCORD_GATEWAY_KEY

    def test_a_caller_that_omits_the_gateway_key_still_reaches_the_hook(self, clean_plugin_manager):
        """Compatibility: every existing call site that has no gateway key to
        offer keeps working, and observer plugins keep receiving the hook."""
        payloads: list = []
        clean_plugin_manager.register_hook(
            "pre_tool_call", lambda **kwargs: payloads.append(kwargs),
        )

        resolve_pre_tool_block(
            "patch", {"path": "/repo/app.py"}, session_id=TRANSCRIPT_SESSION_ID,
        )

        assert len(payloads) == 1
        assert payloads[0]["session_id"] == TRANSCRIPT_SESSION_ID
        assert payloads[0].get("gateway_session_key") in ("", None)


# ---------------------------------------------------------------------------
# B. post_tool_call — result-based authorization keys on the CHAT
# ---------------------------------------------------------------------------


class TestPostToolCallCarriesBothIdentities:
    """``post_tool_call`` is where a plugin authorizes on a tool RESULT (the
    claude-worker Terra fallback release). Keyed by transcript id, such a
    release is scoped to a conversation SLICE — the next compression mints a
    new id and silently revokes it — and can never be matched against the
    gateway-key identity ``pre_tool_call`` now resolves eligibility from."""

    def test_the_hook_payload_carries_both(self, clean_plugin_manager):
        from model_tools import handle_function_call

        payloads: list = []
        clean_plugin_manager.register_hook(
            "post_tool_call", lambda **kwargs: payloads.append(kwargs),
        )

        registry = MagicMock()
        registry.dispatch.return_value = json.dumps({"success": True})
        with patch("model_tools.registry", registry):
            handle_function_call(
                "claude_worker",
                {"task": "fix it"},
                task_id="task-post",
                session_id=TRANSCRIPT_SESSION_ID,
                gateway_session_key=DISCORD_GATEWAY_KEY,
                skip_pre_tool_call_hook=True,
            )

        assert len(payloads) == 1
        assert payloads[0]["session_id"] == TRANSCRIPT_SESSION_ID
        assert payloads[0]["gateway_session_key"] == DISCORD_GATEWAY_KEY

    def test_a_caller_that_omits_the_gateway_key_still_emits_the_hook(self, clean_plugin_manager):
        from model_tools import handle_function_call

        payloads: list = []
        clean_plugin_manager.register_hook(
            "post_tool_call", lambda **kwargs: payloads.append(kwargs),
        )

        registry = MagicMock()
        registry.dispatch.return_value = json.dumps({"success": True})
        with patch("model_tools.registry", registry):
            handle_function_call(
                "web_search",
                {"query": "test"},
                task_id="task-post-cli",
                session_id=TRANSCRIPT_SESSION_ID,
                skip_pre_tool_call_hook=True,
            )

        assert len(payloads) == 1
        assert payloads[0]["session_id"] == TRANSCRIPT_SESSION_ID
        assert payloads[0].get("gateway_session_key") in ("", None)

    def test_the_agent_path_forwards_the_gateway_key_into_tool_dispatch(self):
        """The real agent path reaches ``handle_function_call`` through
        ``_invoke_tool``; the gateway key has to survive that hop or the
        ``post_tool_call`` it emits is back to transcript-only."""
        agent = _make_agent()
        captured: dict = {}

        def _fake_handle_function_call(*args, **kwargs):
            captured.update(kwargs)
            return json.dumps({"success": True})

        with patch("run_agent.handle_function_call", _fake_handle_function_call):
            agent._invoke_tool(
                "web_search", {"query": "test"}, "task-fwd",
                tool_call_id="c-fwd", pre_tool_block_checked=True,
            )

        assert captured.get("session_id") == TRANSCRIPT_SESSION_ID
        assert captured.get("gateway_session_key") == DISCORD_GATEWAY_KEY
