"""Tests for ``plugins/claude-worker/__init__.py`` — the final wiring.

``register(ctx)`` registers the tool + hooks + auxiliary task, and the
end-to-end enforcement path (pre_gateway_dispatch -> identity cache ->
pre_tool_call gate) works through a single ``register(ctx)`` call, not
piecemeal hook registration.

The end-to-end cases here are the global-scope change seen from the outside:
an ARBITRARY Discord channel and an ARBITRARY Discord thread are both gated
with no configuration at all, a non-Discord chat never is, and a terminal
command that would run the Claude CLI on the host is refused. Also covers the
requirement invariants: ``model.default`` is never written and no Anthropic
provider is registered by any part of this plugin.
"""

from __future__ import annotations

import json

import pytest

import hermes_cli.plugins as hermes_plugins
from hermes_cli.plugins import resolve_pre_tool_block
from gateway.session_context import _UNSET, _VAR_MAP

from tests.plugins._claude_worker_helpers import (
    load_plugin_package,
    load_submodule,
    make_git_repo,
    require_safe_tmp,
    stub_fresh_oauth_preflight,
)

canary = load_submodule("canary")
gate = load_submodule("gate")
config = load_submodule("config")

#: Deliberately arbitrary: no channel id is special to the plugin any more.
ARBITRARY_CHANNEL = "204871330045198336"
UNENROLLED_CHANNEL = "999999999999999999"
THREAD_ID = "111222333444555666"


class _FakeSourcePlatform:
    def __init__(self, value):
        self.value = value


class _FakeSource:
    def __init__(self, platform, chat_id, parent_chat_id=None, thread_id=None):
        self.platform = _FakeSourcePlatform(platform)
        self.chat_id = chat_id
        self.parent_chat_id = parent_chat_id
        self.thread_id = thread_id


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
    require_safe_tmp(tmp_path)

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))

    # No plugin configuration at all: the defaults must already gate every
    # Discord-origin session.
    monkeypatch.setattr(config, "_load_raw_config", lambda: {})

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
    # ``register`` installs the OAuth refresh probe into module-global state
    # (``oauth.REFRESH_PROBE``), so it is saved and restored here — a probe
    # left installed by this fixture would otherwise follow the session into
    # every later test file.
    oauth_module = load_submodule("oauth")
    saved_probe = oauth_module.REFRESH_PROBE
    plugin_module.register(ctx)

    saved_manager = hermes_plugins._plugin_manager
    hermes_plugins._plugin_manager = manager

    yield plugin_module, manager, ctx

    oauth_module.set_refresh_probe(saved_probe)
    hermes_plugins._plugin_manager = saved_manager
    tool_registry.deregister("claude_worker")
    for var, val in zip(_VAR_MAP.values(), saved_ctx.values()):
        var.set(val)
    canary._identity_cache.clear()
    gate.reset_unlocks()


@pytest.fixture()
def repo(tmp_path):
    path = make_git_repo(tmp_path, "repo")
    (path / "app.py").write_text("x = 1\n", encoding="utf-8")
    return path


def _dispatch(source, session_key):
    from hermes_cli.plugins import invoke_hook

    invoke_hook(
        "pre_gateway_dispatch",
        event=_FakeEvent(source), gateway=_FakeGateway(session_key), session_store=None,
    )
    _VAR_MAP["HERMES_SESSION_KEY"].set(session_key)


def _patch(target, session_key):
    return resolve_pre_tool_block(
        "patch", {"path": str(target), "old_string": "x = 1", "new_string": "x = 2"},
        session_id=session_key,
    )


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

    def test_the_tool_schema_documents_dynamic_single_repo_scope(self, registered_plugin):
        module, _manager, _ctx = registered_plugin
        cwd_description = module.TOOL_SCHEMA["parameters"]["properties"]["cwd"]["description"]
        assert "Git worktree" in cwd_description
        # The retired static-allowlist language must be gone, not merely
        # supplemented — it told callers the wrong thing about scope.
        assert "allowlisted repo root" not in cwd_description
        assert "configured claude-worker repo root" not in cwd_description

    def test_allow_opus_is_a_dedicated_boolean_field(self, registered_plugin):
        """Requirement 2: Opus authorization must be an explicit, separate
        input the caller opts into — not folded into ``complexity``."""
        module, _manager, _ctx = registered_plugin
        props = module.TOOL_SCHEMA["parameters"]["properties"]
        assert props["allow_opus"]["type"] == "boolean"
        assert "true" in props["allow_opus"]["description"].lower()

    def test_complexity_alone_is_documented_as_insufficient_for_opus(self, registered_plugin):
        module, _manager, _ctx = registered_plugin
        props = module.TOOL_SCHEMA["parameters"]["properties"]
        complexity_description = props["complexity"]["description"]
        # The retired claim ("forces claude-opus-5 routing regardless of
        # keyword detection") must be gone — complexity alone must never
        # read as sufficient authorization on its own.
        assert "forces" not in complexity_description.lower()
        assert "regardless of keyword detection" not in complexity_description.lower()
        assert "allow_opus" in complexity_description
        assert "authorizes NOTHING" in complexity_description

    def test_description_documents_no_keyword_routing_and_no_escalation(self, registered_plugin):
        module, _manager, _ctx = registered_plugin
        description = module.TOOL_SCHEMA["description"]
        assert "never automatically retried" in description or "no automatic escalation" in description
        assert "Task text/keywords never choose" in description

    def test_description_documents_validation_status_semantics(self, registered_plugin):
        module, _manager, _ctx = registered_plugin
        description = module.TOOL_SCHEMA["description"]
        assert "validation_status" in description
        assert "parent_verification_required" in description
        assert "Bash is unavailable" in description or "Bash is denied" in description


class TestEndToEndEnforcementThroughRegisterCtx:
    def test_an_arbitrary_discord_channel_is_blocked_end_to_end(
        self, repo, registered_plugin,
    ):
        """No configuration whatsoever, and a channel that was never enrolled
        anywhere: the platform alone puts it in scope."""
        _dispatch(_FakeSource("discord", ARBITRARY_CHANNEL), "sess:e2e-channel")
        message = _patch(repo / "app.py", "sess:e2e-channel")
        assert message is not None
        assert "claude_worker" in message

    def test_a_discord_thread_is_blocked_end_to_end(self, repo, registered_plugin):
        """The thread case the old channel-list check silently missed: the
        session's ``chat_id`` is the THREAD id, and ``parent_chat_id`` is not
        a tracked session contextvar."""
        _dispatch(
            _FakeSource("discord", THREAD_ID, parent_chat_id=UNENROLLED_CHANNEL,
                        thread_id=THREAD_ID),
            "sess:e2e-thread",
        )
        message = _patch(repo / "app.py", "sess:e2e-thread")
        assert message is not None
        assert "claude_worker" in message

    def test_a_never_enrolled_channel_is_blocked_exactly_like_any_other(
        self, repo, registered_plugin,
    ):
        for channel in (ARBITRARY_CHANNEL, UNENROLLED_CHANNEL, "42"):
            _dispatch(_FakeSource("discord", channel), f"sess:e2e-uniform:{channel}")
            assert _patch(repo / "app.py", f"sess:e2e-uniform:{channel}") is not None

    def test_config_cannot_exempt_a_channel_end_to_end(
        self, repo, registered_plugin, monkeypatch,
    ):
        """Config tries every retired narrowing spelling at once. None of
        them can exempt this chat, because scope is the platform."""
        raw_cfg = {
            "plugins": {
                "entries": {
                    "claude-worker": {
                        "canary_channel_ids": [UNENROLLED_CHANNEL],
                        "canary": {"enabled": True, "channel_ids": [UNENROLLED_CHANNEL]},
                        "gate": {"repo_roots": ["/somewhere/else"]},
                    }
                }
            }
        }
        monkeypatch.setattr(config, "_load_raw_config", lambda: raw_cfg)

        _dispatch(_FakeSource("discord", ARBITRARY_CHANNEL), "sess:e2e-not-exempt")
        assert _patch(repo / "app.py", "sess:e2e-not-exempt") is not None

    def test_a_non_discord_chat_is_never_blocked_end_to_end(
        self, repo, registered_plugin,
    ):
        _dispatch(_FakeSource("telegram", ARBITRARY_CHANNEL), "sess:e2e-telegram")
        assert _patch(repo / "app.py", "sess:e2e-telegram") is None

    def test_a_path_outside_every_worktree_is_never_blocked_end_to_end(
        self, tmp_path, registered_plugin,
    ):
        scratch = tmp_path / "scratch"
        scratch.mkdir()
        _dispatch(_FakeSource("discord", ARBITRARY_CHANNEL), "sess:e2e-outside")
        assert resolve_pre_tool_block(
            "write_file", {"path": str(scratch / "notes.txt"), "content": "hi"},
            session_id="sess:e2e-outside",
        ) is None

    def test_the_terminal_bypass_is_closed_end_to_end(self, registered_plugin):
        _dispatch(_FakeSource("discord", ARBITRARY_CHANNEL), "sess:e2e-terminal")
        message = resolve_pre_tool_block(
            "terminal", {"command": "claude -p 'edit the parser'"},
            session_id="sess:e2e-terminal",
        )
        assert message is not None
        assert "claude_worker" in message

    def test_ordinary_terminal_work_survives_end_to_end(self, registered_plugin):
        _dispatch(_FakeSource("discord", ARBITRARY_CHANNEL), "sess:e2e-terminal-ok")
        assert resolve_pre_tool_block(
            "terminal", {"command": "make build && pytest -q"},
            session_id="sess:e2e-terminal-ok",
        ) is None

    def test_a_successful_worker_run_does_not_release_the_gate_end_to_end(
        self, repo, registered_plugin,
    ):
        """The worker is the coder for the whole session, not a one-time toll
        gate: a successful run proves the worker works and grants nothing."""
        from hermes_cli.plugins import invoke_hook

        _dispatch(_FakeSource("discord", ARBITRARY_CHANNEL), "sess:e2e-run")
        assert _patch(repo / "app.py", "sess:e2e-run") is not None

        invoke_hook(
            "post_tool_call", tool_name="claude_worker",
            args={"task": "fix it", "cwd": str(repo)},
            result=json.dumps({"success": True, "cwd": str(repo)}),
            session_id="sess:e2e-run", status="ok",
        )
        assert _patch(repo / "app.py", "sess:e2e-run") is not None

    def test_an_explicit_terra_fallback_releases_the_gate_end_to_end(
        self, repo, registered_plugin,
    ):
        """The one remaining release path — the worker could not run at all
        and Terra actually came back with guidance."""
        from hermes_cli.plugins import invoke_hook

        _dispatch(_FakeSource("discord", ARBITRARY_CHANNEL), "sess:e2e-fallback")
        assert _patch(repo / "app.py", "sess:e2e-fallback") is not None

        invoke_hook(
            "post_tool_call", tool_name="claude_worker",
            args={"task": "fix it", "cwd": str(repo)},
            result=json.dumps({
                "success": False, "status": "fallback_ready", "fallback_ready": True,
                "fallback_requested": True, "fallback_provenance": "terra_auxiliary",
                "cwd": str(repo),
                "fallback": {"ok": True, "ready": True, "notes": "Terra guidance"},
            }),
            session_id="sess:e2e-fallback", status="ok",
        )
        assert _patch(repo / "app.py", "sess:e2e-fallback") is None

    def test_a_discord_continuation_that_lost_its_contextvars_is_still_gated(
        self, repo, registered_plugin,
    ):
        """End-to-end form of the live regression: the ambient session
        ContextVars are cleared (context compression/continuation) and only
        the explicit dispatch ``session_id`` remains."""
        session_key = "agent:main:discord:group:1527706694665113670:970735341680082944"
        _dispatch(_FakeSource("discord", ARBITRARY_CHANNEL), session_key)
        canary._identity_cache.clear()
        for var in _VAR_MAP.values():
            var.set("")

        assert _patch(repo / "app.py", session_key) is not None
        assert resolve_pre_tool_block(
            "terminal", {"command": "claude -p 'edit the parser'"},
            session_id=session_key,
        ) is not None


class TestManifestOptIn:
    def test_plugin_yaml_is_standalone_opt_in(self):
        import yaml

        from tests.plugins._claude_worker_helpers import plugin_dir

        manifest = yaml.safe_load((plugin_dir() / "plugin.yaml").read_text())
        assert manifest.get("kind", "standalone") == "standalone"
        assert manifest["name"] == "claude-worker"

    def test_plugin_yaml_describes_the_global_discord_scope(self):
        import yaml

        from tests.plugins._claude_worker_helpers import plugin_dir

        manifest = yaml.safe_load((plugin_dir() / "plugin.yaml").read_text())
        description = manifest["description"]
        assert "Discord" in description
        assert "thread" in description.lower()
        # The retired two-channel canary language must be gone.
        assert "canary channel" not in description.lower()
        assert "allowlisted repo root" not in description

    def test_plugin_yaml_registers_the_hooks_the_gate_needs(self):
        import yaml

        from tests.plugins._claude_worker_helpers import plugin_dir

        manifest = yaml.safe_load((plugin_dir() / "plugin.yaml").read_text())
        assert set(manifest["hooks"]) >= {
            "pre_gateway_dispatch", "pre_tool_call", "post_tool_call",
        }


class TestOAuthRefreshProbeRegistration:
    """``register(ctx)`` is where the host-CLI refresh probe gets installed.

    The seam (``oauth.set_refresh_probe``) already existed; what registration
    adds is the ONE implementation of it, and it must live outside ``oauth.py``
    so that module keeps its no-write/no-subprocess tripwire. Registration is
    also the only place the probe is installed, so it has to be deterministic:
    the same object after any number of ``register(ctx)`` calls, and no probe
    at all when the knob is off.
    """

    @pytest.fixture(autouse=True)
    def _restore_probe(self):
        oauth = load_submodule("oauth")
        saved = oauth.REFRESH_PROBE
        yield
        oauth.set_refresh_probe(saved)

    def test_register_installs_the_probe_from_the_dedicated_module(self, registered_plugin):
        import inspect
        import os

        oauth = load_submodule("oauth")
        oauth_refresh = load_submodule("oauth_refresh")

        assert oauth.REFRESH_PROBE is oauth_refresh.refresh_probe
        source_file = inspect.getsourcefile(oauth.REFRESH_PROBE)
        assert os.path.realpath(source_file) != os.path.realpath(oauth.__file__)

    def test_each_register_installs_exactly_one_probe(self, registered_plugin, monkeypatch):
        """Not two, and not zero: the installed probe is one deterministic
        object, so repeated registration (plugin reload, a second manager)
        cannot stack probes or leave a stale one behind."""
        module, _manager, ctx = registered_plugin
        oauth = load_submodule("oauth")

        installed = []
        real_set = oauth.set_refresh_probe
        monkeypatch.setattr(
            oauth, "set_refresh_probe",
            lambda probe: (installed.append(probe), real_set(probe))[1],
        )

        module.register(ctx)
        module.register(ctx)

        assert len(installed) == 2
        assert installed[0] is installed[1] is oauth.REFRESH_PROBE

    def test_a_disabled_knob_leaves_no_probe_installed(self, registered_plugin, monkeypatch):
        module, _manager, ctx = registered_plugin
        oauth = load_submodule("oauth")
        assert oauth.REFRESH_PROBE is not None

        monkeypatch.setattr(
            config, "_load_raw_config",
            lambda: {"plugins": {"entries": {
                "claude-worker": {"oauth": {"auto_refresh": False}},
            }}},
        )
        module.register(ctx)

        assert oauth.REFRESH_PROBE is None


class TestInvariants:
    def test_model_default_config_key_never_written(self, tmp_path, monkeypatch, registered_plugin):
        repo = make_git_repo(tmp_path, "invariant-repo")
        runner = load_submodule("runner")

        # ``run_worker``'s OAuth freshness preflight reads the fixed host
        # credentials path, which does not exist under a test home; without
        # this the run would HOLD on auth and never reach the invariant.
        stub_fresh_oauth_preflight(monkeypatch)

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

    def test_the_model_literals_are_unchanged(self):
        policy = load_submodule("policy")
        assert policy.DEFAULT_MODEL == "claude-sonnet-5"
        assert policy.ESCALATION_MODEL == "claude-opus-5"
        assert policy.MODEL_ALLOWLIST == frozenset({"claude-sonnet-5", "claude-opus-5"})

    def test_no_anthropic_provider_registered_by_plugin_module(self, registered_plugin):
        _module, _manager, _ctx = registered_plugin
        source_files = [
            load_submodule(name).__file__
            for name in ("canary", "gate", "routing", "breaker", "runner", "telemetry",
                         "review", "config", "project", "terminal_guard", "trust")
        ]
        for path in source_files:
            text = open(path, encoding="utf-8").read()
            assert "PROVIDER_REGISTRY[" not in text
            assert "register_provider(" not in text
            assert "model.default" not in text
