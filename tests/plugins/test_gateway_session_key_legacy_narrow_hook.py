"""Guard: a legacy ``pre_tool_call`` plugin callback declared with only the original
narrow signature (``tool_name``, ``args`` — no ``**kwargs``) must keep firing without a
``TypeError`` now that ``model_tools.handle_function_call`` threads the additive
``gateway_session_key`` field into the ``pre_tool_call`` hook payload (see
``tests/plugins/test_gateway_session_key_tool_hooks.py``). Payload narrowing
(``PluginDispatchMixin._hook_callback_kwargs``) is supposed to strip any kwarg a
callback's declared signature doesn't name; this pins that promise for the newest
additive field specifically, using the same real-dispatch harness as the sibling
gateway_session_key hook tests.
"""

import json
from types import SimpleNamespace

import pytest

from hermes_cli import plugins as plugins_mod
from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest
from model_tools import handle_function_call
from tools.registry import registry

_PROBE_TOOL = "__probe_gateway_session_key_narrow_hook_tool__"


@pytest.fixture
def hook_ctx(monkeypatch):
    """A fresh, discovery-skipping ``PluginManager`` wired as the active one, matching
    ``tests/plugins/test_gateway_session_key_tool_hooks.py``'s ``hook_ctx`` fixture."""
    manager = PluginManager()
    manager._discovered = True
    monkeypatch.setattr(plugins_mod, "_plugin_manager", manager)
    monkeypatch.setattr(plugins_mod, "_plugin_managers_by_home", {})
    return PluginContext(PluginManifest(name="gsk-narrow-probe", source="test"), manager)


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


def test_legacy_narrow_pre_tool_call_fires_without_typeerror(hook_ctx, probe_tool):
    calls = []

    def narrow_pre_tool_call(tool_name, args):
        calls.append((tool_name, args))

    hook_ctx.register_hook("pre_tool_call", narrow_pre_tool_call)

    result = handle_function_call(
        probe_tool, {}, task_id="task-1", tool_call_id="call-1", session_id="sess-1",
        gateway_session_key="gsk-narrow-1",
    )

    assert json.loads(result) == {"ok": True}
    assert calls == [(probe_tool, {})]
