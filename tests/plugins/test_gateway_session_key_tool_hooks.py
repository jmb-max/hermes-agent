"""RED: generic additive plugin-hook contract threading ``gateway_session_key``
from the agent through to every tool hook/middleware call.

Not yet implemented anywhere in production — every test below is expected to
fail against the current code and turn green once the contract lands:

  1. ``agent.inline_tool_executors.tool_hook_ids`` surfaces ``agent._gateway_session_key``
     as ``"gateway_session_key"`` (``""`` when absent), preserving every existing id.
  2. ``model_tools._CallIds`` accepts an optional ``gateway_session_key`` field whose
     ``hook_kwargs()`` normalizes ``None`` to ``""`` like every other id field.
  3. ``model_tools.handle_function_call`` accepts an optional ``gateway_session_key``
     kwarg and threads the exact same value into both the ``pre_tool_call`` and
     ``post_tool_call`` hooks for a real (registry-dispatched) tool call.
  4. Omitting ``gateway_session_key`` anywhere in the chain is backward compatible
     and surfaces as ``""``, never ``None`` or a missing key.
"""

import json
from types import SimpleNamespace

import pytest

from agent.inline_tool_executors import tool_hook_ids
from hermes_cli import plugins as plugins_mod
from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest
from model_tools import _CallIds, handle_function_call
from tools.registry import registry

_PROBE_TOOL = "__probe_gateway_session_key_tool__"


@pytest.fixture
def hook_ctx(monkeypatch):
    """A fresh, discovery-skipping ``PluginManager`` wired as the active one so
    ``hermes_cli.plugins.invoke_hook``/``has_hook`` deliver to hooks we register here,
    exactly like ``tests/agent/test_auxiliary_hooks.py`` does for other hook pairs."""
    manager = PluginManager()
    manager._discovered = True  # skip the real on-disk plugin-tree scan
    monkeypatch.setattr(plugins_mod, "_plugin_manager", manager)
    monkeypatch.setattr(plugins_mod, "_plugin_managers_by_home", {})
    return PluginContext(PluginManifest(name="gsk-probe", source="test"), manager)


@pytest.fixture
def probe_tool():
    """A harmless tool registered for real ``registry.dispatch`` (not mocked)."""
    registry.register(
        name=_PROBE_TOOL, toolset="test",
        schema={"name": _PROBE_TOOL, "parameters": {"type": "object", "properties": {}}},
        handler=lambda args, **kwargs: json.dumps({"ok": True}),
    )
    try:
        yield _PROBE_TOOL
    finally:
        registry.deregister(_PROBE_TOOL)


def _capture_pre_and_post(ctx):
    """Register real ``pre_tool_call``/``post_tool_call`` hooks; return the dict they fill."""
    captured = {}
    ctx.register_hook("pre_tool_call", lambda **kw: captured.__setitem__("pre_tool_call", kw))
    ctx.register_hook("post_tool_call", lambda **kw: captured.__setitem__("post_tool_call", kw))
    return captured


class TestToolHookIdsGatewaySessionKey:
    """(1) agent.inline_tool_executors.tool_hook_ids"""

    def test_present_on_agent_is_surfaced_and_existing_ids_preserved(self):
        agent = SimpleNamespace(
            session_id="sess-1", _current_turn_id="turn-1", _current_api_request_id="req-1",
            _gateway_session_key="gsk-abc",
        )
        ids = tool_hook_ids(agent, "task-1", "call-1")
        assert ids == {
            "task_id": "task-1",
            "session_id": "sess-1",
            "tool_call_id": "call-1",
            "turn_id": "turn-1",
            "api_request_id": "req-1",
            "gateway_session_key": "gsk-abc",
        }

    def test_absent_on_agent_normalizes_to_empty_string(self):
        agent = SimpleNamespace(session_id="sess-1")
        ids = tool_hook_ids(agent, "task-1", None)
        assert ids["gateway_session_key"] == ""
        # Every other identity kwarg is still present and unaffected.
        assert ids["task_id"] == "task-1"
        assert ids["session_id"] == "sess-1"
        assert ids["tool_call_id"] == ""


class TestCallIdsGatewaySessionKey:
    """(2) model_tools._CallIds"""

    def test_accepts_gateway_session_key_and_hook_kwargs_carries_it(self):
        ids = _CallIds(
            task_id="t", session_id="s", tool_call_id="c", turn_id="tu", api_request_id="a",
            gateway_session_key="gsk-xyz",
        )
        assert ids.hook_kwargs()["gateway_session_key"] == "gsk-xyz"

    def test_omitted_gateway_session_key_normalizes_to_empty_string(self):
        ids = _CallIds(task_id="t")
        assert ids.hook_kwargs()["gateway_session_key"] == ""

    def test_none_gateway_session_key_normalizes_to_empty_string(self):
        ids = _CallIds(task_id="t", gateway_session_key=None)
        assert ids.hook_kwargs()["gateway_session_key"] == ""


class TestHandleFunctionCallGatewaySessionKey:
    """(3) & (4) model_tools.handle_function_call, real dispatch through the registry."""

    def test_pre_and_post_tool_call_receive_the_exact_stable_key(self, hook_ctx, probe_tool):
        captured = _capture_pre_and_post(hook_ctx)

        result = handle_function_call(
            probe_tool, {}, task_id="task-1", tool_call_id="call-1", session_id="sess-1",
            gateway_session_key="gsk-stable-123",
        )

        assert json.loads(result) == {"ok": True}
        assert "pre_tool_call" in captured, "pre_tool_call hook never fired for the probe tool"
        assert "post_tool_call" in captured, "post_tool_call hook never fired for the probe tool"
        assert captured["pre_tool_call"]["gateway_session_key"] == "gsk-stable-123"
        assert captured["post_tool_call"]["gateway_session_key"] == "gsk-stable-123"

    def test_absence_is_backward_compatible_and_emits_empty_string(self, hook_ctx, probe_tool):
        captured = _capture_pre_and_post(hook_ctx)

        result = handle_function_call(probe_tool, {}, task_id="task-1", tool_call_id="call-1")

        assert json.loads(result) == {"ok": True}
        assert captured["pre_tool_call"]["gateway_session_key"] == ""
        assert captured["post_tool_call"]["gateway_session_key"] == ""

    def test_gateway_session_key_is_keyword_only_and_optional(self, probe_tool):
        # No hooks registered at all: must still dispatch cleanly with the new
        # kwarg accepted but unused downstream.
        result = handle_function_call(
            probe_tool, {}, task_id="task-1", gateway_session_key="gsk-unused",
        )
        assert json.loads(result) == {"ok": True}
