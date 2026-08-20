"""Tests for ``plugins/claude-worker/policy.py`` — the immutable policy layer.

Every value Terra's review requires to be non-configurable lives here as a
module-level literal: the two canary Discord channel ids, the platform, the
mandatory gated tool set, the two model identities, the structural attempt
cap, the sandbox image tag, and the Claude tool allowlist. These tests are
the tripwire: they assert the literals themselves and that the helpers
derived from them can never be widened by configuration.
"""

from __future__ import annotations

import os
import sys

import pytest

from tests.plugins._claude_worker_helpers import load_submodule

policy = load_submodule("policy")

CHANNEL_A = "1527706694665113670"
CHANNEL_B = "1501268569697026140"


class TestImmutableCanaryPolicy:
    def test_exact_channel_ids(self):
        assert set(policy.CANARY_CHANNEL_IDS) == {CHANNEL_A, CHANNEL_B}

    def test_platform_is_discord_only(self):
        assert policy.CANARY_PLATFORMS == frozenset({"discord"})

    def test_channel_ids_are_an_immutable_container(self):
        assert isinstance(policy.CANARY_CHANNEL_IDS, frozenset)
        with pytest.raises(AttributeError):
            policy.CANARY_CHANNEL_IDS.add("999")

    def test_config_cannot_add_channels(self):
        cfg = {"canary": {"enabled": True, "channel_ids": ["999999999999999999", CHANNEL_A]}}
        assert policy.enabled_canary_channel_ids(cfg) == frozenset({CHANNEL_A})

    def test_config_may_select_a_subset(self):
        cfg = {"canary": {"enabled": True, "channel_ids": [CHANNEL_B]}}
        assert policy.enabled_canary_channel_ids(cfg) == frozenset({CHANNEL_B})

    def test_empty_selection_means_all_policy_channels(self):
        cfg = {"canary": {"enabled": True, "channel_ids": []}}
        assert policy.enabled_canary_channel_ids(cfg) == policy.CANARY_CHANNEL_IDS

    def test_feature_can_be_disabled_but_not_widened(self):
        cfg = {"canary": {"enabled": False, "channel_ids": ["999"]}}
        assert policy.enabled_canary_channel_ids(cfg) == frozenset()

    def test_malformed_canary_config_is_safe(self):
        for cfg in ({}, {"canary": None}, {"canary": {"channel_ids": "1527706694665113670"}},
                    {"canary": {"channel_ids": [None, 5, {}]}}):
            result = policy.enabled_canary_channel_ids(cfg)
            assert result <= policy.CANARY_CHANNEL_IDS

    def test_platform_check_rejects_non_discord(self):
        assert policy.is_canary_platform("discord") is True
        for bad in ("telegram", "slack", "cli", "", None, "Discord "):
            assert policy.is_canary_platform(bad) is False


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


class TestCanonicalRepoRoots:
    def test_roots_are_resolved_and_deduplicated(self, tmp_path):
        root = tmp_path / "repo"
        root.mkdir()
        cfg = {"gate": {"repo_roots": [str(root), str(root) + os.sep, str(root)]}}
        roots = policy.canonical_repo_roots(cfg)
        assert roots == [os.path.realpath(str(root))]

    def test_missing_roots_are_dropped(self, tmp_path):
        root = tmp_path / "repo"
        root.mkdir()
        cfg = {"gate": {"repo_roots": [str(root), str(tmp_path / "nope"), None, 7]}}
        assert policy.canonical_repo_roots(cfg) == [os.path.realpath(str(root))]

    def test_no_roots_configured_means_empty(self):
        assert policy.canonical_repo_roots({}) == []
        assert policy.canonical_repo_roots({"gate": {"repo_roots": []}}) == []

    def test_resolve_within_roots_accepts_nested_path(self, tmp_path):
        root = tmp_path / "repo"
        (root / "src").mkdir(parents=True)
        target = root / "src" / "app.py"
        target.write_text("x = 1\n", encoding="utf-8")
        roots = [os.path.realpath(str(root))]
        assert policy.resolve_within_roots(str(target), roots) == os.path.realpath(str(root))

    def test_resolve_within_roots_rejects_outside(self, tmp_path):
        root = tmp_path / "repo"
        root.mkdir()
        other = tmp_path / "other"
        other.mkdir()
        roots = [os.path.realpath(str(root))]
        assert policy.resolve_within_roots(str(other / "f.py"), roots) is None

    def test_resolve_within_roots_rejects_sibling_prefix_collision(self, tmp_path):
        root = tmp_path / "repo"
        root.mkdir()
        sibling = tmp_path / "repo-evil"
        sibling.mkdir()
        roots = [os.path.realpath(str(root))]
        assert policy.resolve_within_roots(str(sibling / "f.py"), roots) is None

    @pytest.mark.skipif(sys.platform == "win32", reason="symlinks need admin on Windows")
    def test_resolve_within_roots_is_symlink_safe(self, tmp_path):
        root = tmp_path / "repo"
        root.mkdir()
        outside = tmp_path / "secret"
        outside.mkdir()
        (outside / "creds.txt").write_text("token\n", encoding="utf-8")
        link = root / "escape"
        link.symlink_to(outside, target_is_directory=True)
        roots = [os.path.realpath(str(root))]
        assert policy.resolve_within_roots(str(link / "creds.txt"), roots) is None

    @pytest.mark.skipif(sys.platform == "win32", reason="symlinks need admin on Windows")
    def test_symlinked_root_itself_is_canonicalized(self, tmp_path):
        real = tmp_path / "real-repo"
        real.mkdir()
        link = tmp_path / "linked-repo"
        link.symlink_to(real, target_is_directory=True)
        cfg = {"gate": {"repo_roots": [str(link)]}}
        roots = policy.canonical_repo_roots(cfg)
        assert roots == [os.path.realpath(str(real))]
        assert policy.resolve_within_roots(str(real / "a.py"), roots) == os.path.realpath(str(real))

    def test_dotdot_traversal_is_rejected(self, tmp_path):
        root = tmp_path / "repo"
        root.mkdir()
        roots = [os.path.realpath(str(root))]
        assert policy.resolve_within_roots(str(root / ".." / "etc" / "passwd"), roots) is None
