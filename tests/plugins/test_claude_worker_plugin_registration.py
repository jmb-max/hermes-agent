"""RED->GREEN tests for ``plugins/claude-worker/__init__.py`` — the final
wiring: ``register(ctx)`` registers the tool + hooks + auxiliary task, and
the end-to-end enforcement path (pre_gateway_dispatch -> canary cache ->
pre_tool_call gate) works through a single ``register(ctx)`` call, not
piecemeal hook registration. Also covers requirement invariants (17)/(18):
``model.default`` is never written and no Anthropic provider is registered
by any part of this plugin.
"""

from __future__ import annotations

import json

import pytest

import hermes_cli.plugins as hermes_plugins
from hermes_cli.plugins import resolve_pre_tool_block
from gateway.session_context import _UNSET, _VAR_MAP

from tests.plugins._claude_worker_helpers import load_plugin_package, load_submodule

canary = load_submodule("canary")
gate = load_submodule("gate")
breaker = load_submodule("breaker")
config = load_submodule("config")

CANARY_CHANNEL = "1527706694665113670"


class _FakeSourcePlatform:
    def __init__(self, value):
        self.value = value


class _FakeSource:
    def __init__(self, platform, chat_id, parent_chat_id=None):
        self.platform = _FakeSourcePlatform(platform)
        self.chat_id = chat_id
        self.parent_chat_id = parent_chat_id


class _FakeEvent:
    def __init__(self, source):
        self.source = source


class _FakeGateway:
    def __init__(self, session_key):
        self._session_key = session_key

    def _session_key_for_source(self, source):
        return self._session_key


@pytest.fixture()
def registered_plugin(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))

    saved_ctx = {name: var.get() for name, var in _VAR_MAP.items()}
    for var in _VAR_MAP.values():
        var.set(_UNSET)
    canary._identity_cache.clear()
    gate.reset_unlocks()

    from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest
    from tools.registry import registry as tool_registry

    plugin_module = load_plugin_package(force=True)
    manager = PluginManager()
    manifest = PluginManifest(name="claude-worker", key="claude-worker", source="bundled")
    ctx = PluginContext(manifest, manager)
    plugin_module.register(ctx)

    saved_manager = hermes_plugins._plugin_manager
    hermes_plugins._plugin_manager = manager

    yield plugin_module, manager, ctx

    hermes_plugins._plugin_manager = saved_manager
    tool_registry.deregister("claude_worker")
    for var, val in zip(_VAR_MAP.values(), saved_ctx.values()):
        var.set(val)
    canary._identity_cache.clear()
    gate.reset_unlocks()


class TestToolAndHookRegistration:
    def test_registers_claude_worker_tool(self, registered_plugin):
        from tools.registry import registry as tool_registry

        entry = tool_registry.get_entry("claude_worker")
        assert entry is not None
        assert entry.toolset == "claude_worker"

    def test_registers_expected_hooks(self, registered_plugin):
        _module, manager, _ctx = registered_plugin
        assert manager._hooks.get("pre_gateway_dispatch")
        assert manager._hooks.get("pre_tool_call")
        assert manager._hooks.get("post_tool_call")

    def test_registers_auxiliary_review_task(self, registered_plugin):
        _module, manager, _ctx = registered_plugin
        assert "claude_worker_review" in manager._aux_tasks
        entry = manager._aux_tasks["claude_worker_review"]
        assert entry["defaults"]["provider"] == "openai-codex"
        assert entry["defaults"]["model"] == "gpt-5.6-terra"


class TestEndToEndEnforcementThroughRegisterCtx:
    def test_direct_channel_patch_is_blocked_end_to_end(self, tmp_path, monkeypatch, registered_plugin):
        repo = tmp_path / "repo"
        (repo / ".git").mkdir(parents=True)
        target = repo / "app.py"
        target.write_text("x = 1\n")

        # Only gate.repo_roots is configured: the canary channels come from
        # the immutable policy layer, so the defaults are already live.
        raw_cfg = {
            "plugins": {"entries": {"claude-worker": {"gate": {"repo_roots": [str(repo)]}}}}
        }
        monkeypatch.setattr(config, "_load_raw_config", lambda: raw_cfg)

        _module, manager, _ctx = registered_plugin
        from hermes_cli.plugins import invoke_hook

        event = _FakeEvent(_FakeSource("discord", CANARY_CHANNEL))
        gateway = _FakeGateway("sess:e2e")
        invoke_hook("pre_gateway_dispatch", event=event, gateway=gateway, session_store=None)

        _VAR_MAP["HERMES_SESSION_KEY"].set("sess:e2e")

        message = resolve_pre_tool_block(
            "patch",
            {"path": str(target), "old_string": "x = 1", "new_string": "x = 2"},
            session_id="sess:e2e",
        )
        assert message is not None
        assert "claude_worker" in message

    def test_non_canary_chat_is_never_blocked_end_to_end(self, tmp_path, monkeypatch, registered_plugin):
        repo = tmp_path / "repo"
        (repo / ".git").mkdir(parents=True)
        target = repo / "app.py"
        target.write_text("x = 1\n")

        # Config tries to enrol a third channel, both through the old
        # top-level key and through the narrowing-only canary block. Neither
        # can widen the policy, so this chat is never gated.
        raw_cfg = {
            "plugins": {
                "entries": {
                    "claude-worker": {
                        "canary_channel_ids": ["999999999999999999"],
                        "canary": {"channel_ids": ["999999999999999999", CANARY_CHANNEL]},
                        "gate": {"repo_roots": [str(repo)]},
                    }
                }
            }
        }
        monkeypatch.setattr(config, "_load_raw_config", lambda: raw_cfg)

        from hermes_cli.plugins import invoke_hook

        event = _FakeEvent(_FakeSource("discord", "999999999999999999"))
        gateway = _FakeGateway("sess:e2e-non-canary")
        invoke_hook("pre_gateway_dispatch", event=event, gateway=gateway, session_store=None)
        _VAR_MAP["HERMES_SESSION_KEY"].set("sess:e2e-non-canary")

        message = resolve_pre_tool_block(
            "patch",
            {"path": str(target), "old_string": "x = 1", "new_string": "x = 2"},
            session_id="sess:e2e-non-canary",
        )
        assert message is None


class TestManifestOptIn:
    def test_plugin_yaml_is_standalone_opt_in(self):
        import yaml

        from tests.plugins._claude_worker_helpers import plugin_dir

        manifest = yaml.safe_load((plugin_dir() / "plugin.yaml").read_text())
        assert manifest.get("kind", "standalone") == "standalone"
        assert manifest["name"] == "claude-worker"


class TestInvariants:
    def test_model_default_config_key_never_written(self, tmp_path, monkeypatch, registered_plugin):
        repo = tmp_path / "repo"
        repo.mkdir()
        runner = load_submodule("runner")

        monkeypatch.setattr(
            config, "_load_raw_config",
            lambda: {"plugins": {"entries": {"claude-worker": {"gate": {"repo_roots": [str(repo)]}}}}},
        )
        monkeypatch.setattr(
            runner, "spawn_claude",
            lambda **kw: {
                "exit_code": 0, "stdout": json.dumps({"result": "ok"}), "stderr": "",
                "duration_ms": 10, "timed_out": False, "model": kw["model"],
            },
        )
        monkeypatch.setattr(runner, "_git_changed_files", lambda cwd, repo_roots=None: [])

        write_calls = []
        import hermes_cli.config as hc_config

        if hasattr(hc_config, "save_config_value"):
            monkeypatch.setattr(
                hc_config, "save_config_value",
                lambda *a, **k: write_calls.append((a, k)),
            )
        if hasattr(hc_config, "save_config"):
            monkeypatch.setattr(
                hc_config, "save_config",
                lambda *a, **k: write_calls.append((a, k)),
            )

        raw = runner.run_worker({"task": "fix it", "cwd": str(repo)}, session_id="sess:inv")
        result = json.loads(raw)
        assert result["success"] is True
        assert write_calls == []

    def test_no_anthropic_provider_registered_by_plugin_module(self, registered_plugin):
        import inspect

        _module, _manager, _ctx = registered_plugin
        source_files = [
            load_submodule(name).__file__
            for name in ("canary", "gate", "routing", "breaker", "runner", "telemetry", "review", "config")
        ]
        for path in source_files:
            text = open(path, encoding="utf-8").read()
            assert "PROVIDER_REGISTRY[" not in text
            assert "register_provider(" not in text
            assert "model.default" not in text
