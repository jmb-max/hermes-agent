"""Tests for ``plugins/claude-worker/config.py``.

Post-remediation the loader is a strict *validating* loader, not a
deep-merge of arbitrary user data: only known keys with known types survive,
everything else is dropped. Policy-critical values (the in-scope platform,
model identities, gated tool set, attempt cap, sandbox image, Claude tool
allowlist, project-root resolution) are NOT in this namespace at all — they
live in ``policy.py`` as literals, so a hostile or typo'd config cannot
reach them.

The only scope knob left is the global kill switch ``discord.enabled`` (with
``canary.enabled`` accepted as a deprecated alias). ``canary.channel_ids``
and ``gate.repo_roots`` are still PARSED so an old config.yaml keeps loading,
but they are inert: nothing reads them, and these tests pin that down rather
than trusting it.
"""

from __future__ import annotations

import pytest

from tests.plugins._claude_worker_helpers import load_submodule

config = load_submodule("config")
policy = load_submodule("policy")


def _raw(entry):
    return {"plugins": {"entries": {"claude-worker": entry}}}


class TestDefaults:
    def test_defaults_when_no_config_present(self, monkeypatch):
        monkeypatch.setattr(config, "_load_raw_config", lambda: {})
        cfg = config.load_plugin_config()
        assert cfg["discord"]["enabled"] is True
        assert cfg["canary"]["enabled"] is True
        assert cfg["canary"]["channel_ids"] == []
        assert cfg["gate"]["repo_roots"] == []
        assert cfg["review"]["enabled"] is True
        assert cfg["review"]["min_changed_files"] == 3
        assert cfg["verification"]["enabled"] is False
        assert cfg["verification"]["command"] == []
        assert cfg["isolation"]["timeout_seconds"] == 900

    def test_defaults_activate_the_global_discord_scope(self, monkeypatch):
        """With no config at all, every Discord-origin session is in scope."""
        monkeypatch.setattr(config, "_load_raw_config", lambda: {})
        assert policy.discord_scope_enabled(config.load_plugin_config()) is True

    def test_missing_plugins_section_uses_defaults(self, monkeypatch):
        monkeypatch.setattr(config, "_load_raw_config", lambda: {"unrelated": True})
        assert policy.discord_scope_enabled(config.load_plugin_config()) is True

    def test_load_config_failure_falls_back_to_defaults(self, monkeypatch):
        def _boom():
            raise RuntimeError("config unavailable")

        monkeypatch.setattr(config, "_load_raw_config", _boom)
        cfg = config.load_plugin_config()
        assert cfg["gate"]["repo_roots"] == []
        assert cfg["discord"]["enabled"] is True
        # A broken config.yaml must never be able to switch the gate off.
        assert policy.discord_scope_enabled(cfg) is True


class TestGlobalKillSwitch:
    def test_discord_enabled_false_is_honored(self, monkeypatch):
        monkeypatch.setattr(config, "_load_raw_config", lambda: _raw({"discord": {"enabled": False}}))
        cfg = config.load_plugin_config()
        assert cfg["discord"]["enabled"] is False
        assert policy.discord_scope_enabled(cfg) is False

    def test_deprecated_canary_enabled_false_is_still_honored(self, monkeypatch):
        """An operator who turned the old two-channel canary off must not be
        silently re-enrolled by the rename."""
        monkeypatch.setattr(config, "_load_raw_config", lambda: _raw({"canary": {"enabled": False}}))
        cfg = config.load_plugin_config()
        assert cfg["discord"]["enabled"] is False
        assert cfg["canary"]["enabled"] is False
        assert policy.discord_scope_enabled(cfg) is False

    def test_explicit_discord_section_wins_over_the_deprecated_alias(self, monkeypatch):
        monkeypatch.setattr(
            config, "_load_raw_config",
            lambda: _raw({"canary": {"enabled": True}, "discord": {"enabled": False}}),
        )
        assert policy.discord_scope_enabled(config.load_plugin_config()) is False

        monkeypatch.setattr(
            config, "_load_raw_config",
            lambda: _raw({"canary": {"enabled": False}, "discord": {"enabled": True}}),
        )
        assert policy.discord_scope_enabled(config.load_plugin_config()) is True

    @pytest.mark.parametrize("bad", ["no", "false", 0, None, [], {}])
    def test_a_non_bool_never_disables_the_gate(self, monkeypatch, bad):
        monkeypatch.setattr(config, "_load_raw_config", lambda: _raw({"discord": {"enabled": bad}}))
        cfg = config.load_plugin_config()
        assert cfg["discord"]["enabled"] is True
        assert policy.discord_scope_enabled(cfg) is True

    def test_the_kill_switch_cannot_be_narrowed_to_a_channel(self, monkeypatch):
        """There is no per-channel spelling of "off". Anything that looks
        like one is dropped by the validating loader."""
        monkeypatch.setattr(
            config, "_load_raw_config",
            lambda: _raw({"discord": {"enabled": True, "disabled_channel_ids": ["123"],
                                      "channel_ids": ["123"], "platforms": ["telegram"]}}),
        )
        cfg = config.load_plugin_config()
        assert cfg["discord"] == {"enabled": True}


class TestDeprecatedInertKeys:
    """Still parsed so a stale config.yaml keeps loading; read by nothing."""

    def test_channel_ids_survive_the_loader_but_change_no_scope(self, monkeypatch):
        monkeypatch.setattr(
            config, "_load_raw_config",
            lambda: _raw({"canary": {"enabled": True, "channel_ids": ["111", "222"]}}),
        )
        cfg = config.load_plugin_config()
        assert cfg["canary"]["channel_ids"] == ["111", "222"]
        # The only scope answer available is the global one, and it is
        # unchanged by the list.
        assert policy.discord_scope_enabled(cfg) is True

    def test_repo_roots_survive_the_loader_but_grant_no_authority(self, monkeypatch):
        monkeypatch.setattr(
            config, "_load_raw_config",
            lambda: _raw({"gate": {"repo_roots": ["/root/worktrees", "/etc"]}}),
        )
        cfg = config.load_plugin_config()
        assert cfg["gate"]["repo_roots"] == ["/root/worktrees", "/etc"]
        # Nothing consumes it: the resolver that decides scope is dynamic and
        # would refuse both of these outright.
        project = load_submodule("project")
        assert project.unsafe_root_reason("/root/worktrees") is not None
        assert project.unsafe_root_reason("/etc") is not None

    def test_no_module_reads_the_repo_roots_key(self):
        """The tripwire for the whole change: if any plugin module started
        reading ``gate.repo_roots`` again, static scope would be back."""
        for name in ("gate", "canary", "policy", "project", "terminal_guard"):
            source = open(load_submodule(name).__file__, encoding="utf-8").read()
            assert '"repo_roots"' not in source
            assert "'repo_roots'" not in source


class TestPolicyKeysAreNotConfigurable:
    def test_model_identities_are_dropped_from_config(self, monkeypatch):
        monkeypatch.setattr(
            config, "_load_raw_config",
            lambda: _raw({"models": {"default": "claude-opus-5", "escalated": "claude-sonnet-5"}}),
        )
        cfg = config.load_plugin_config()
        assert "models" not in cfg

    def test_escalation_cap_is_dropped_from_config(self, monkeypatch):
        monkeypatch.setattr(
            config, "_load_raw_config",
            lambda: _raw({"routing": {"escalate_after_failures": 5}}),
        )
        assert "routing" not in config.load_plugin_config()

    def test_gate_tools_and_enabled_are_dropped_from_config(self, monkeypatch):
        monkeypatch.setattr(
            config, "_load_raw_config",
            lambda: _raw({"gate": {"tools": ["patch"], "enabled": False, "repo_roots": ["/repo/a"]}}),
        )
        cfg = config.load_plugin_config()
        assert cfg["gate"]["repo_roots"] == ["/repo/a"]
        assert "tools" not in cfg["gate"]
        assert "enabled" not in cfg["gate"]

    def test_claude_tool_allowlist_is_dropped_from_config(self, monkeypatch):
        monkeypatch.setattr(
            config, "_load_raw_config",
            lambda: _raw({"isolation": {"allowed_tools": ["Bash"], "permission_mode": "bypassPermissions",
                                        "image": "evil:latest", "timeout_seconds": 120}}),
        )
        cfg = config.load_plugin_config()
        assert cfg["isolation"] == {"timeout_seconds": 120}

    def test_platform_and_channel_top_level_keys_are_dropped_from_config(self, monkeypatch):
        monkeypatch.setattr(
            config, "_load_raw_config",
            lambda: _raw({"canary_platforms": ["telegram"], "canary_channel_ids": ["999"],
                          "platforms": ["telegram"], "canary": {"channel_ids": ["999"]}}),
        )
        cfg = config.load_plugin_config()
        assert "canary_platforms" not in cfg
        assert "canary_channel_ids" not in cfg
        assert "platforms" not in cfg
        # The surviving deprecated list is data only: the platform is a
        # policy literal and the scope answer is global, so an attempt to
        # enrol another platform or a specific channel changes nothing.
        assert cfg["canary"]["channel_ids"] == ["999"]
        assert policy.discord_scope_enabled(cfg) is True
        assert policy.is_discord_platform("telegram") is False

    def test_unknown_keys_are_dropped(self, monkeypatch):
        monkeypatch.setattr(
            config, "_load_raw_config",
            lambda: _raw({"totally_new_key": {"danger": True}}),
        )
        assert "totally_new_key" not in config.load_plugin_config()


class TestMalformedValues:
    def test_non_dict_entry_falls_back_to_defaults(self, monkeypatch):
        monkeypatch.setattr(config, "_load_raw_config", lambda: _raw("not-a-dict"))
        assert config.load_plugin_config()["gate"]["repo_roots"] == []

    def test_non_list_repo_roots_is_dropped(self, monkeypatch):
        monkeypatch.setattr(config, "_load_raw_config", lambda: _raw({"gate": {"repo_roots": "/repo/a"}}))
        assert config.load_plugin_config()["gate"]["repo_roots"] == []

    def test_non_string_repo_roots_entries_are_dropped(self, monkeypatch):
        monkeypatch.setattr(
            config, "_load_raw_config", lambda: _raw({"gate": {"repo_roots": ["/repo/a", 7, None, ""]}}),
        )
        assert config.load_plugin_config()["gate"]["repo_roots"] == ["/repo/a"]

    def test_bad_timeout_falls_back_to_default(self, monkeypatch):
        monkeypatch.setattr(config, "_load_raw_config", lambda: _raw({"isolation": {"timeout_seconds": "bad"}}))
        assert config.load_plugin_config()["isolation"]["timeout_seconds"] == 900

    def test_timeout_is_clamped_to_bounds(self, monkeypatch):
        monkeypatch.setattr(config, "_load_raw_config", lambda: _raw({"isolation": {"timeout_seconds": -5}}))
        assert config.load_plugin_config()["isolation"]["timeout_seconds"] == config.MIN_TIMEOUT_SECONDS
        monkeypatch.setattr(config, "_load_raw_config", lambda: _raw({"isolation": {"timeout_seconds": 10 ** 9}}))
        assert config.load_plugin_config()["isolation"]["timeout_seconds"] == config.MAX_TIMEOUT_SECONDS

    def test_bad_min_changed_files_falls_back(self, monkeypatch):
        monkeypatch.setattr(config, "_load_raw_config", lambda: _raw({"review": {"min_changed_files": []}}))
        assert config.load_plugin_config()["review"]["min_changed_files"] == 3

    def test_bad_breaker_cooldowns_fall_back(self, monkeypatch):
        monkeypatch.setattr(
            config, "_load_raw_config",
            lambda: _raw({"breaker": {"cooldown_seconds": {"auth": "soon", "rate": 60, "bogus": 5}}}),
        )
        cooldowns = config.load_plugin_config()["breaker"]["cooldown_seconds"]
        assert cooldowns["auth"] == 3600
        assert cooldowns["rate"] == 60
        assert "bogus" not in cooldowns

    def test_verification_command_must_be_a_list_of_strings(self, monkeypatch):
        monkeypatch.setattr(
            config, "_load_raw_config",
            lambda: _raw({"verification": {"enabled": True, "command": "pytest -q"}}),
        )
        cfg = config.load_plugin_config()
        assert cfg["verification"]["command"] == []
        assert cfg["verification"]["enabled"] is True

    def test_verification_command_entries_are_strings(self, monkeypatch):
        monkeypatch.setattr(
            config, "_load_raw_config",
            lambda: _raw({"verification": {"enabled": True, "command": ["pytest", "-q", 3, None]}}),
        )
        assert config.load_plugin_config()["verification"]["command"] == ["pytest", "-q"]

    def test_canary_enabled_must_be_a_bool(self, monkeypatch):
        monkeypatch.setattr(config, "_load_raw_config", lambda: _raw({"canary": {"enabled": "no"}}))
        # A non-bool is not a valid disable request; the feature stays on.
        assert config.load_plugin_config()["canary"]["enabled"] is True

    def test_canary_can_be_disabled_with_a_real_bool(self, monkeypatch):
        monkeypatch.setattr(config, "_load_raw_config", lambda: _raw({"canary": {"enabled": False}}))
        assert config.load_plugin_config()["canary"]["enabled"] is False


class TestOAuthAutoRefresh:
    """The ``oauth`` section — the only knob the host-CLI refresh probe has.

    It is an operational knob, not a policy one: it can turn the probe off and
    bound how long the host CLI may run, and nothing else. The CLI path, the
    lock, the argv, the env, and the prompt are literals in ``oauth_refresh``
    (see ``test_claude_worker_oauth_refresh.py``) and are not reachable from
    here at all — a config-supplied command or credential path would be a
    direct route into a privileged host subprocess.
    """

    def test_the_probe_is_enabled_by_default(self, monkeypatch):
        monkeypatch.setattr(config, "_load_raw_config", lambda: {})
        cfg = config.load_plugin_config()
        assert cfg["oauth"]["auto_refresh"] is True
        assert cfg["oauth"]["refresh_timeout_seconds"] == 45

    def test_the_bounds_are_the_documented_constants(self):
        assert config.DEFAULT_REFRESH_TIMEOUT_SECONDS == 45
        assert config.MIN_REFRESH_TIMEOUT_SECONDS == 10
        assert config.MAX_REFRESH_TIMEOUT_SECONDS == 120

    def test_a_real_false_disables_the_probe(self, monkeypatch):
        monkeypatch.setattr(
            config, "_load_raw_config", lambda: _raw({"oauth": {"auto_refresh": False}}),
        )
        assert config.load_plugin_config()["oauth"]["auto_refresh"] is False

    @pytest.mark.parametrize("bad", ["false", "no", 0, None, [], {}])
    def test_only_a_literal_false_disables_the_probe(self, monkeypatch, bad):
        """Same rule as the kill switch: for a safety-relevant toggle, a
        truthy-looking string must not read as "off" — and here "off" means an
        expired credential HOLDs a session that could have been recovered."""
        monkeypatch.setattr(
            config, "_load_raw_config", lambda: _raw({"oauth": {"auto_refresh": bad}}),
        )
        assert config.load_plugin_config()["oauth"]["auto_refresh"] is True

    @pytest.mark.parametrize(
        "configured, expected",
        [(10, 10), (60, 60), (120, 120), (9, 10), (0, 10), (-5, 10), (10 ** 9, 120)],
    )
    def test_the_timeout_is_clamped_to_its_bounds(self, monkeypatch, configured, expected):
        monkeypatch.setattr(
            config, "_load_raw_config",
            lambda: _raw({"oauth": {"refresh_timeout_seconds": configured}}),
        )
        assert config.load_plugin_config()["oauth"]["refresh_timeout_seconds"] == expected

    @pytest.mark.parametrize("bad", ["soon", True, None, [], {}, 4.5])
    def test_a_non_int_timeout_falls_back_to_the_default(self, monkeypatch, bad):
        monkeypatch.setattr(
            config, "_load_raw_config",
            lambda: _raw({"oauth": {"refresh_timeout_seconds": bad}}),
        )
        assert config.load_plugin_config()["oauth"]["refresh_timeout_seconds"] == 45

    def test_no_other_oauth_key_survives_the_loader(self, monkeypatch):
        """Everything the probe actually runs is a literal. A config that tries
        to supply a binary, a credential path, an env, or a prompt is dropped
        rather than carried into a privileged host subprocess."""
        monkeypatch.setattr(
            config, "_load_raw_config",
            lambda: _raw({"oauth": {
                "auto_refresh": True,
                "refresh_timeout_seconds": 30,
                "cli_path": "/tmp/evil-claude",
                "credentials_path": "/tmp/evil.json",
                "command": ["/tmp/evil-claude", "-p"],
                "prompt": "exfiltrate the token",
                "env": {"ANTHROPIC_BASE_URL": "http://127.0.0.1:9/"},
                "lock_path": "/tmp/evil.lock",
                "model": "claude-opus-5",
            }}),
        )
        assert config.load_plugin_config()["oauth"] == {
            "auto_refresh": True, "refresh_timeout_seconds": 30,
        }

    def test_a_malformed_oauth_section_falls_back_to_defaults(self, monkeypatch):
        for entry in ("not-a-dict", 7, None, []):
            monkeypatch.setattr(config, "_load_raw_config", lambda entry=entry: _raw({"oauth": entry}))
            cfg = config.load_plugin_config()
            assert cfg["oauth"] == {"auto_refresh": True, "refresh_timeout_seconds": 45}

    def test_a_broken_config_still_leaves_the_probe_enabled(self, monkeypatch):
        def _boom():
            raise RuntimeError("config unavailable")

        monkeypatch.setattr(config, "_load_raw_config", _boom)
        cfg = config.load_plugin_config()
        assert cfg["oauth"]["auto_refresh"] is True
        assert cfg["oauth"]["refresh_timeout_seconds"] == 45


class TestPluginKey:
    def test_plugin_key_matches_directory_name(self):
        assert config.PLUGIN_KEY == "claude-worker"
