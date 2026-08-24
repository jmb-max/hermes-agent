"""Tests for ``plugins/claude-worker/policy.py`` — the immutable policy layer.

Every value Terra's review requires to be non-configurable lives here as a
module-level literal: the in-scope PLATFORM (Discord, and every session on
it — there is no channel allowlist any more), the mandatory gated tool set,
the two model identities, the structural attempt cap, the sandbox image tag,
the Claude tool allowlist, and the literals that bound the dynamic project-
root resolver. These tests are the tripwire: they assert the literals
themselves, that configuration can only turn the feature OFF, and that the
retired channel/repo-root allowlists are gone rather than merely unused.
"""

from __future__ import annotations

import os
import re

import pytest

from tests.plugins._claude_worker_helpers import load_submodule

policy = load_submodule("policy")


class TestGlobalDiscordScope:
    def test_platform_is_discord_only(self):
        assert policy.DISCORD_PLATFORMS == frozenset({"discord"})

    def test_platform_set_is_an_immutable_container(self):
        assert isinstance(policy.DISCORD_PLATFORMS, frozenset)
        with pytest.raises(AttributeError):
            policy.DISCORD_PLATFORMS.add("telegram")

    def test_platform_check_accepts_only_the_exact_literal(self):
        assert policy.is_discord_platform("discord") is True
        for bad in ("telegram", "slack", "cli", "", None, 7, [], "Discord", "DISCORD", "discord "):
            assert policy.is_discord_platform(bad) is False

    def test_channel_allowlist_helpers_are_gone_not_merely_unused(self):
        """A dormant narrowing helper is a narrowing helper: if it still
        existed, a future caller could reintroduce per-channel scope."""
        for removed in ("CANARY_CHANNEL_IDS", "CANARY_PLATFORMS",
                        "enabled_canary_channel_ids", "is_canary_platform"):
            assert not hasattr(policy, removed)

    def test_module_contains_no_channel_id_literals(self):
        source = open(policy.__file__, encoding="utf-8").read()
        assert re.search(r"\d{17,20}", source) is None


class TestKillSwitchIsTheOnlyKnob:
    @pytest.mark.parametrize(
        "cfg",
        [
            {},
            {"discord": {}},
            {"discord": {"enabled": True}},
            {"canary": {"enabled": True}},
            {"discord": None},
            {"canary": None},
            {"discord": "nonsense"},
            {"canary": {"channel_ids": ["1527706694665113670"]}},
        ],
    )
    def test_gate_stays_on_by_default_and_for_malformed_config(self, cfg):
        assert policy.discord_scope_enabled(cfg) is True

    def test_only_the_literal_false_disables(self):
        assert policy.discord_scope_enabled({"discord": {"enabled": False}}) is False
        assert policy.discord_scope_enabled({"canary": {"enabled": False}}) is False

    @pytest.mark.parametrize("truthy_looking", ["no", "false", "0", 0, [], None])
    def test_a_non_bool_never_reads_as_off(self, truthy_looking):
        """For a safety toggle, a mistyped value must not silently disable
        the gate."""
        assert policy.discord_scope_enabled({"discord": {"enabled": truthy_looking}}) is True

    def test_discord_section_wins_over_the_deprecated_alias(self):
        cfg = {"discord": {"enabled": False}, "canary": {"enabled": True}}
        assert policy.discord_scope_enabled(cfg) is False

    def test_non_dict_config_leaves_the_gate_on(self):
        for cfg in (None, "off", 0, []):
            assert policy.discord_scope_enabled(cfg) is True

    def test_config_cannot_narrow_scope_to_channels(self):
        """The one remaining knob is global. There is no shape of config that
        turns the gate on for one channel and off for another — every call
        below returns the SAME answer for the whole platform."""
        for cfg in (
            {"discord": {"enabled": True, "channel_ids": ["1"]}},
            {"canary": {"enabled": True, "channel_ids": ["1"]}},
            {"canary": {"enabled": True, "channel_ids": []}},
        ):
            assert policy.discord_scope_enabled(cfg) is True


class TestImmutableGateToolSet:
    def test_gated_tools_are_exactly_the_mandatory_three(self):
        assert policy.GATED_TOOLS == frozenset({"patch", "write_file", "skill_manage"})

    def test_all_skill_manage_actions_are_treated_as_mutating(self):
        # tools/skill_manager_tool.py enum — every one of these can mutate
        # repo files, so every one is gate-relevant.
        assert policy.SKILL_MANAGE_MUTATING_ACTIONS == frozenset(
            {"create", "edit", "patch", "delete", "write_file", "remove_file"}
        )


class TestImmutableModels:
    def test_model_literals(self):
        assert policy.DEFAULT_MODEL == "claude-sonnet-5"
        assert policy.ESCALATION_MODEL == "claude-opus-5"

    def test_allowlist_is_exactly_two_models(self):
        assert policy.MODEL_ALLOWLIST == frozenset({"claude-sonnet-5", "claude-opus-5"})

    def test_max_attempts_is_two(self):
        assert policy.MAX_ATTEMPTS == 2

    def test_no_config_lookup_in_policy_module(self):
        source = open(policy.__file__, encoding="utf-8").read()
        assert "load_plugin_config" not in source
        # Model identities must be literals in this module, never read from
        # a mutable mapping.
        assert '"claude-sonnet-5"' in source
        assert '"claude-opus-5"' in source


class TestImmutableSandboxPolicy:
    def test_image_tag_is_a_fixed_local_literal(self):
        assert policy.SANDBOX_IMAGE_TAG == "claude-worker-sandbox:2.1.237"
        assert policy.CLAUDE_CLI_VERSION == "2.1.237"

    def test_image_id_is_the_activated_immutable_local_image(self):
        assert policy.SANDBOX_IMAGE_ID == (
            "sha256:46dc23aaa53c845dacb081dbd71d865a1772fdb2af51fb28ee8f4bfa35f8dc80"
        )
        digest = policy.SANDBOX_IMAGE_ID.split(":", 1)[1]
        assert len(digest) == 64
        assert all(c in "0123456789abcdef" for c in digest)
        assert policy.SANDBOX_IMAGE_ID_CONFIGURED is True

    def test_docker_host_endpoint_is_the_fixed_unix_socket(self):
        assert policy.DOCKER_HOST_ENDPOINT == "unix:///var/run/docker.sock"

    def test_docker_bin_is_an_absolute_trusted_path_not_a_bare_command(self):
        assert policy.DOCKER_BIN == "/usr/bin/docker"
        assert os.path.isabs(policy.DOCKER_BIN)

    def test_claude_tool_allowlist_is_read_edit_write_glob_grep(self):
        assert list(policy.CLAUDE_ALLOWED_TOOLS) == ["Read", "Edit", "Write", "Glob", "Grep"]

    def test_dangerous_tools_are_explicitly_denied(self):
        for denied in ("Bash", "WebFetch", "WebSearch", "Agent", "Task"):
            assert denied in policy.CLAUDE_DENIED_TOOLS
            assert denied not in policy.CLAUDE_ALLOWED_TOOLS

    def test_permission_mode_is_accept_edits(self):
        assert policy.PERMISSION_MODE == "acceptEdits"

    def test_required_cli_flags(self):
        assert set(policy.REQUIRED_CLI_FLAGS) == {
            "--strict-mcp-config", "--settings", "--setting-sources",
        }

    def test_container_paths(self):
        assert policy.CONTAINER_WORKDIR == "/workspace"
        assert policy.CONTAINER_CREDENTIALS_PATH.endswith(".credentials.json")
        assert policy.CONTAINER_CREDENTIALS_PATH.startswith(policy.CONTAINER_HOME + "/")


class TestFixedSandboxIdentity:
    """Every container the runner ever starts — worker spawn or verifier —
    must run as the exact non-root identity baked into the sandbox image
    (``USER worker``, uid 10001, see ``sandbox/Dockerfile``), never as
    whatever uid the host runner process happens to be (root, since it must
    be able to read the root-owned OAuth credentials path)."""

    def test_sandbox_uid_and_gid_are_the_fixed_image_worker_identity(self):
        assert policy.SANDBOX_UID == 10001
        assert policy.SANDBOX_GID == 10001

    def test_sandbox_identity_is_never_root(self):
        assert policy.SANDBOX_UID != 0
        assert policy.SANDBOX_GID != 0


class TestCredentialStagingPolicy:
    """The one host OAuth credentials file is root-owned and cannot be read
    by the fixed non-root sandbox uid, so it is never mounted directly —
    only a staged, re-owned copy under this fixed trusted root ever is."""

    def test_staging_root_is_a_fixed_absolute_literal(self):
        assert policy.CREDENTIAL_STAGING_ROOT == "/run/hermes-claude-worker"
        assert os.path.isabs(policy.CREDENTIAL_STAGING_ROOT)

    def test_max_credential_bytes_is_exactly_one_mebibyte(self):
        assert policy.MAX_CREDENTIAL_BYTES == 1024 * 1024


class TestNoStaticRepoRootAuthorityRemains:
    """``gate.repo_roots`` used to be BOTH the authority for where the worker
    could run AND the thing that got mounted, which is how an operator entry
    of ``/root/worktrees`` handed every container every sibling checkout.
    Project scope is resolved dynamically now (``project.py``), so the
    configured-allowlist helpers must be gone, not dormant."""

    def test_configured_root_helpers_are_gone(self):
        for removed in ("canonical_repo_roots", "resolve_within_roots"):
            assert not hasattr(policy, removed)

    def test_policy_never_reads_a_repo_roots_config_key(self):
        source = open(policy.__file__, encoding="utf-8").read()
        for config_key in ('"repo_roots"', "'repo_roots'", '.get("gate")', ".get('gate')"):
            assert config_key not in source


class TestDynamicResolverLiterals:
    """The literals ``project.py``'s resolution is bounded by. They are
    policy, not configuration, for the same reason the model ids are."""

    def test_git_bin_is_an_absolute_trusted_path_not_a_bare_command(self):
        assert policy.GIT_BIN == "/usr/bin/git"
        assert os.path.isabs(policy.GIT_BIN)
        assert "/" in policy.GIT_BIN

    def test_git_resolution_timeout_is_bounded_and_short(self):
        assert isinstance(policy.GIT_RESOLVE_TIMEOUT_SECONDS, int)
        assert 0 < policy.GIT_RESOLVE_TIMEOUT_SECONDS <= 30

    def test_ancestor_walk_depth_is_bounded(self):
        assert isinstance(policy.MAX_PROJECT_WALK_DEPTH, int)
        assert 0 < policy.MAX_PROJECT_WALK_DEPTH <= 256

    @pytest.mark.parametrize(
        "sensitive",
        ["/", "/etc", "/usr", "/var", "/proc", "/sys", "/dev", "/root", "/home", "/tmp", "/run"],
    )
    def test_system_sensitive_directories_can_never_be_a_project_root(self, sensitive):
        assert sensitive in policy.UNSAFE_PROJECT_ROOTS

    def test_the_multi_project_worktree_container_is_never_a_project_root(self):
        assert "/root/worktrees" in policy.UNSAFE_PROJECT_ROOTS

    def test_unsafe_prefixes_cover_the_system_trees_at_any_depth(self):
        for prefix in ("/etc", "/usr", "/var", "/proc", "/sys", "/dev", "/run", "/bin", "/sbin"):
            assert prefix in policy.UNSAFE_PROJECT_PREFIXES

    def test_unsafe_containers_are_immutable(self):
        assert isinstance(policy.UNSAFE_PROJECT_ROOTS, frozenset)
        assert isinstance(policy.UNSAFE_PROJECT_PREFIXES, tuple)
        with pytest.raises(AttributeError):
            policy.UNSAFE_PROJECT_ROOTS.add("/anything")


class TestTerminalBypassPolicy:
    def test_terminal_tool_is_in_scope_for_the_guard(self):
        assert policy.TERMINAL_TOOLS == frozenset({"terminal"})

    def test_claude_cli_basename_is_the_exact_executable_name(self):
        assert policy.CLAUDE_CLI_BASENAME == "claude"
