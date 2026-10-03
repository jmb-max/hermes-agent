"""RED: ``tui_gateway.methods_connectors._session_connector_rpc`` must thread the
session's stable gateway identity into ``model_tools.handle_function_call`` as
``gateway_session_key`` — the ``agent._gateway_session_key`` set at agent-init time
(see ``tests/plugins/test_gateway_session_key_tool_hooks.py`` for the rest of this
contract) when an agent is attached, falling back to ``session["session_key"]`` when
the session has no agent yet (e.g. connector RPCs fired before the agent finishes
building). Not yet implemented in production — both tests below are expected to fail
against the current code, which never passes ``gateway_session_key`` to
``handle_function_call`` at all.

Ownership/gating (``_session_connector_gate``, ``_connector_owner_matches``) is
stubbed out here so the test isolates the one seam under test: the kwargs
``_session_connector_rpc`` hands to ``handle_function_call``.
"""

import json
from types import SimpleNamespace

import pytest

from tui_gateway import server


@pytest.fixture(autouse=True)
def _bypass_gate_and_ownership(monkeypatch):
    monkeypatch.setattr(server, "_session_connector_gate", lambda rid, session, action: (None, None, None))
    monkeypatch.setattr(server, "_connector_owner_matches", lambda sid, session, profile_home: True)


def _capture_handle_function_call(monkeypatch):
    captured = {}

    def fake_handle_function_call(function_name, args, **kwargs):
        captured["function_name"] = function_name
        captured["args"] = args
        captured.update(kwargs)
        return json.dumps({"connectors": []})

    monkeypatch.setattr("model_tools.handle_function_call", fake_handle_function_call)
    return captured


def _status_request():
    return SimpleNamespace(owner=SimpleNamespace(session_id="sess-1"), reconnect=False, connectors=[])


def test_passes_agent_gateway_session_key_when_agent_present(monkeypatch):
    captured = _capture_handle_function_call(monkeypatch)
    agent = SimpleNamespace(_gateway_session_key="gsk-agent-1")
    session = {"session_key": "sk-1", "agent": agent, "profile_home": None}

    server._session_connector_rpc("rid-1", _status_request(), session, "status")

    assert captured.get("gateway_session_key") == "gsk-agent-1"


def test_falls_back_to_session_key_when_no_agent(monkeypatch):
    captured = _capture_handle_function_call(monkeypatch)
    session = {"session_key": "sk-2", "agent": None, "profile_home": None}

    server._session_connector_rpc("rid-1", _status_request(), session, "status")

    assert captured.get("gateway_session_key") == "sk-2"
