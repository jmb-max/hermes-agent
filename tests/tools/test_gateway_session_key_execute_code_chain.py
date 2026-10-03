"""RED: ``gateway_session_key`` must survive execute_code's whole nested-dispatch chain.

The top-level plumbing (``agent.inline_tool_executors.tool_hook_ids``,
``model_tools._CallIds``, ``model_tools.handle_function_call`` threading the key
into ``pre_tool_call``/``post_tool_call``) already lands — see
``tests/plugins/test_gateway_session_key_tool_hooks.py``. This file covers the
seams still missing it: the path a tool call takes when it originates INSIDE an
execute_code sandbox script rather than directly from the model. Not yet
implemented anywhere in production — every test below is expected to fail
against the current code and turn green once the contract lands:

  1. ``model_tools._execute_tool`` must include ``gateway_session_key`` in the
     dispatch_kwargs it hands to the registry — the seam that reaches the
     ``execute_code`` handler — exactly like ``task_id``/``session_id`` already do.
  2. ``tools.code_execution_tool._execute_code_handler`` must forward the
     ``gateway_session_key`` kwarg it receives into its call to ``execute_code(...)``.
  3. ``tools.code_execution_rpc._default_dispatch`` must accept an optional
     ``gateway_session_key`` and thread it into the nested ``handle_function_call``
     it dispatches; ``_rpc_poll_loop`` must accept the same optional kwarg and
     forward it into ``_default_dispatch``.
  4. ``tools.code_kernel.CellAuthority`` must accept an optional
     ``gateway_session_key`` at construction and thread it into the nested
     ``handle_function_call`` its cell dispatches RPC calls through.

Every seam is tested twice: the key present (and unchanged end to end), and the
key omitted, which must default to ``""`` — never ``None`` and never a missing
kwarg — matching the ``_CallIds.hook_kwargs()`` convention elsewhere in this
dispatch chain.
"""

import json
import threading

from model_tools import _CallIds, _execute_tool
from tools.code_execution_rpc import _default_dispatch, _rpc_poll_loop
from tools.code_execution_tool import _execute_code_handler
from tools.code_kernel import CellAuthority


class TestExecuteToolToExecuteCodeHandler:
    """(1) model_tools._execute_tool -> execute_code handler."""

    def _dispatch_kwargs(self, monkeypatch, **ids_kwargs):
        captured = {}

        def fake_dispatch(name, args, **kwargs):
            captured.update(kwargs)
            return json.dumps({"ok": True})

        monkeypatch.setattr("model_tools.registry.dispatch", fake_dispatch)
        ids = _CallIds(task_id="task-1", session_id="sess-1", **ids_kwargs)
        _execute_tool(
            "execute_code", {"code": "1"}, {"code": "1"}, ids,
            user_task=None, enabled_tools=[], skip_tool_execution_middleware=True,
        )
        return captured

    def test_present_key_reaches_dispatch_kwargs_unchanged(self, monkeypatch):
        captured = self._dispatch_kwargs(monkeypatch, gateway_session_key="gsk-exec-1")
        assert captured.get("gateway_session_key") == "gsk-exec-1"

    def test_absent_key_defaults_to_empty_string(self, monkeypatch):
        captured = self._dispatch_kwargs(monkeypatch)
        assert captured.get("gateway_session_key") == ""


class TestExecuteCodeHandlerToExecuteCode:
    """(2) tools.code_execution_tool._execute_code_handler -> execute_code."""

    def _forwarded_kwargs(self, monkeypatch, **handler_kwargs):
        captured = {}

        def fake_execute_code(**kwargs):
            captured.update(kwargs)
            return json.dumps({"status": "success"})

        monkeypatch.setattr("tools.code_execution_tool.execute_code", fake_execute_code)
        _execute_code_handler({"code": "print(1)"}, task_id="task-1", **handler_kwargs)
        return captured

    def test_present_key_forwarded_unchanged(self, monkeypatch):
        captured = self._forwarded_kwargs(monkeypatch, gateway_session_key="gsk-handler-1")
        assert captured.get("gateway_session_key") == "gsk-handler-1"

    def test_absent_key_defaults_to_empty_string(self, monkeypatch):
        captured = self._forwarded_kwargs(monkeypatch)
        assert captured.get("gateway_session_key") == ""


class TestDefaultDispatchToHandleFunctionCall:
    """(3a) tools.code_execution_rpc._default_dispatch -> nested handle_function_call."""

    def _forwarded_kwargs(self, monkeypatch, **dispatch_kwargs):
        captured = {}

        def fake_handle_function_call(tool_name, tool_args, **kwargs):
            captured.update(kwargs)
            return json.dumps({"ok": True})

        monkeypatch.setattr("model_tools.handle_function_call", fake_handle_function_call)
        dispatch = _default_dispatch("task-1", **dispatch_kwargs)
        dispatch("web_search", {"query": "q"})
        return captured

    def test_present_key_forwarded_unchanged(self, monkeypatch):
        captured = self._forwarded_kwargs(monkeypatch, gateway_session_key="gsk-rpc-1")
        assert captured.get("gateway_session_key") == "gsk-rpc-1"

    def test_absent_key_defaults_to_empty_string(self, monkeypatch):
        captured = self._forwarded_kwargs(monkeypatch)
        assert captured.get("gateway_session_key") == ""


class TestRpcPollLoopToDefaultDispatch:
    """(3b) tools.code_execution_rpc._rpc_poll_loop -> _default_dispatch.

    ``_default_dispatch`` is bound once, before the poll loop's ``while`` even
    starts, so setting ``stop_event`` up front isolates that one binding call
    without needing a fake remote filesystem/env.
    """

    def _forwarded_kwargs(self, monkeypatch, **poll_kwargs):
        captured = {}

        def fake_default_dispatch(task_id, **kwargs):
            captured.update(kwargs)
            return lambda tool_name, tool_args: json.dumps({"ok": True})

        monkeypatch.setattr("tools.code_execution_rpc._default_dispatch", fake_default_dispatch)
        stop_event = threading.Event()
        stop_event.set()
        _rpc_poll_loop(None, "/tmp/rpc", "task-1", [], [0], 5, frozenset(), stop_event, "tok", **poll_kwargs)
        return captured

    def test_present_key_forwarded_unchanged(self, monkeypatch):
        captured = self._forwarded_kwargs(monkeypatch, gateway_session_key="gsk-poll-1")
        assert captured.get("gateway_session_key") == "gsk-poll-1"

    def test_absent_key_defaults_to_empty_string(self, monkeypatch):
        captured = self._forwarded_kwargs(monkeypatch)
        assert captured.get("gateway_session_key") == ""


class TestCellAuthorityToHandleFunctionCall:
    """(4) tools.code_kernel.CellAuthority -> nested handle_function_call."""

    def _forwarded_kwargs(self, monkeypatch, **authority_kwargs):
        captured = {}

        def fake_handle_function_call(tool_name, tool_args, **kwargs):
            captured.update(kwargs)
            return json.dumps({"ok": True})

        monkeypatch.setattr("model_tools.handle_function_call", fake_handle_function_call)
        authority = CellAuthority("turn-1", **authority_kwargs)
        authority.dispatch("web_search", {"query": "q"})
        return captured

    def test_present_key_forwarded_unchanged(self, monkeypatch):
        captured = self._forwarded_kwargs(monkeypatch, gateway_session_key="gsk-cell-1")
        assert captured.get("gateway_session_key") == "gsk-cell-1"

    def test_absent_key_defaults_to_empty_string(self, monkeypatch):
        captured = self._forwarded_kwargs(monkeypatch)
        assert captured.get("gateway_session_key") == ""
