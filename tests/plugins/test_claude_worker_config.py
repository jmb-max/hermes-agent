"""Tests for ``plugins/claude-worker/config.py``.

Post-remediation the loader is a strict *validating* loader, not a
deep-merge of arbitrary user data: only known keys with known types survive,
everything else is dropped. Policy-critical values (canary channel ids /
platform, model identities, gated tool set, attempt cap, sandbox image,
Claude tool allowlist) are NOT in this namespace at all — they live in
``policy.py`` as literals, so a hostile or typo'd config cannot reach them.
"""

from __future__ import annotations

from tests.plugins._claude_worker_helpers import load_submodule

config = load_submodule("config")
policy = load_submodule("policy")


def _raw(entry):
    return {"plugins": {"entries": {"claude-worker": entry}}}


class TestDefaults:
    def test_defaults_when_no_config_present(self, monkeypatch):
        monkeypatch.setattr(config, "_load_raw_config", lambda: {})
        cfg = config.load_plugin_config()
        assert cfg["canary"]["enabled"] is True
        assert cfg["canary"]["channel_ids"] == []
        assert cfg["gate"]["repo_roots"] == []
        assert cfg["review"]["enabled"] is True
        assert cfg["review"]["min_changed_files"] == 3
        assert cfg["verification"]["enabled"] is False
        assert cfg["verification"]["command"] == []
        assert cfg["isolation"]["timeout_seconds"] == 900

    def test_defaults_activate_the_canary_policy(self, monkeypatch):
        """With no config at all, both required canary channels are live."""
        monkeypatch.setattr(config, "_load_raw_config", lambda: {})
        cfg = config.load_plugin_config()
        assert policy.enabled_canary_channel_ids(cfg) == policy.CANARY_CHANNEL_IDS

    def test_missing_plugins_section_uses_defaults(self, monkeypatch):
        monkeypatch.setattr(config, "_load_raw_config", lambda: {"unrelated": True})
        assert config.load_plugin_config()["gate"]["repo_roots"] == []

    def test_load_config_failure_falls_back_to_defaults(self, monkeypatch):
        def _boom():
            raise RuntimeError("config unavailable")

        monkeypatch.setattr(config, "_load_raw_config", _boom)
        cfg = config.load_plugin_config()
        assert cfg["gate"]["repo_roots"] == []
        assert cfg["canary"]["enabled"] is True


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

    def test_canary_platforms_are_dropped_from_config(self, monkeypatch):
        monkeypatch.setattr(
            config, "_load_raw_config",
            lambda: _raw({"canary_platforms": ["telegram"], "canary_channel_ids": ["999"],
                          "canary": {"channel_ids": ["999"]}}),
        )
        cfg = config.load_plugin_config()
        assert "canary_platforms" not in cfg
        assert "canary_channel_ids" not in cfg
        # A non-policy id survives the loader as data but is filtered by the
        # policy layer (defense in depth, asserted in the policy tests).
        assert policy.enabled_canary_channel_ids(cfg) == frozenset()

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


class TestPluginKey:
    def test_plugin_key_matches_directory_name(self):
        assert config.PLUGIN_KEY == "claude-worker"
