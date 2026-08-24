"""Tests for the OS-isolation layer in ``plugins/claude-worker/runner.py``.

Terra's activation blocker was that isolation was not real: the worker ran
as a host subprocess with ``Bash``/``Write``/``Edit``, so nothing constrained
it to the repo. The remediated design never spawns a host ``claude``: every
model-driven phase runs in a locked-down ``docker run`` against a fixed local
image, with the repo as the only writable mount and a literal five-tool
allowlist.

These tests assert the container command shape (mounts, caps, limits,
network, tools, settings), the fail-closed preflight, and that the host-side
child environment carries no credentials.
"""

from __future__ import annotations

import json
import os
import shutil
import stat
from pathlib import Path

import pytest

from tests.plugins._claude_worker_helpers import (
    load_submodule,
    make_git_repo,
    require_safe_tmp,
)

runner = load_submodule("runner")
policy = load_submodule("policy")


# --------------------------------------------------------------------------
# argv helpers
# --------------------------------------------------------------------------


def _opt_values(cmd, flag):
    """All values following each occurrence of *flag* in an argv list."""
    return [cmd[i + 1] for i, tok in enumerate(cmd) if tok == flag and i + 1 < len(cmd)]


def _mount_field(mount, field):
    """One ``key=value`` field out of a structured ``--mount`` argument."""
    for kv in mount.split(","):
        key, _, value = kv.partition("=")
        if key == field:
            return value
    return None


def _image_index(cmd):
    return cmd.index(policy.SANDBOX_IMAGE_ID)


def _docker_opts(cmd):
    return cmd[:_image_index(cmd)]


def _container_args(cmd):
    return cmd[_image_index(cmd) + 1:]


@pytest.fixture()
def repo(tmp_path):
    """A real Git worktree.

    ``spawn_claude`` resolves its own scope through ``project.py`` whenever a
    caller does not pass an explicit ``repo_roots`` (the dynamic
    single-repository mount), so the fixture has to be something that
    resolver accepts — a bare directory is not.
    """
    require_safe_tmp(tmp_path)
    return make_git_repo(tmp_path, "repo")


@pytest.fixture()
def claude_cmd(repo):
    return runner.build_claude_docker_command(
        repo_root=str(repo), model="claude-sonnet-5", credentials_path=None,
    )


@pytest.fixture(autouse=True)
def _isolated_credential_staging_root(tmp_path, monkeypatch):
    """Every test in this module gets its own throwaway staging root, so no
    test — old or new — ever creates, chowns, or leaves files under the
    real fixed ``policy.CREDENTIAL_STAGING_ROOT`` (``/run/hermes-claude-worker``)."""
    monkeypatch.setattr(policy, "CREDENTIAL_STAGING_ROOT", str(tmp_path / "claude-worker-staging"))


class TestNoHostClaudeSpawn:
    def test_runner_never_invokes_a_host_claude_binary(self):
        source = open(runner.__file__, encoding="utf-8").read()
        assert '"claude", "--help"' not in source
        assert "claude_bin" not in source

    def test_docker_is_the_only_process_entrypoint(self, claude_cmd):
        assert claude_cmd[0] == policy.DOCKER_BIN == "/usr/bin/docker"
        assert os.path.isabs(claude_cmd[0])
        assert claude_cmd[1] == "--host"
        assert claude_cmd[2] == policy.DOCKER_HOST_ENDPOINT
        assert claude_cmd[3] == "run"


class TestContainerHardening:
    def test_ephemeral_and_read_only_rootfs(self, claude_cmd):
        assert "--rm" in claude_cmd
        assert "--read-only" in claude_cmd

    def test_all_capabilities_dropped(self, claude_cmd):
        assert _opt_values(claude_cmd, "--cap-drop") == ["ALL"]
        assert "--cap-add" not in claude_cmd
        assert "--privileged" not in claude_cmd

    def test_no_new_privileges(self, claude_cmd):
        assert "no-new-privileges" in " ".join(_opt_values(claude_cmd, "--security-opt"))

    def test_pid_memory_cpu_limits_are_set(self, claude_cmd):
        assert _opt_values(claude_cmd, "--pids-limit") == [str(policy.PIDS_LIMIT)]
        assert _opt_values(claude_cmd, "--memory") == [policy.MEMORY_LIMIT]
        assert _opt_values(claude_cmd, "--cpus") == [policy.CPU_LIMIT]

    def test_tmpfs_for_tmp_and_home_scratch(self, claude_cmd):
        destinations = [value.split(":", 1)[0] for value in _opt_values(claude_cmd, "--tmpfs")]
        assert "/tmp" in destinations
        assert policy.CONTAINER_HOME in destinations

    def test_tmpfs_is_nosuid_and_nodev(self, claude_cmd):
        for value in _opt_values(claude_cmd, "--tmpfs"):
            assert "nosuid" in value
            assert "nodev" in value

    def test_workdir_is_the_workspace_mount(self, claude_cmd):
        assert _opt_values(claude_cmd, "--workdir") == ["/workspace"]

    def test_runs_as_the_fixed_sandbox_uid_not_the_host_caller(self, claude_cmd):
        """The host runner process is root (it must read the root-owned
        OAuth credentials path), so mirroring its uid/gid — the old
        behavior — actually ran the container AS ROOT. The container must
        always run as the fixed non-root identity baked into the sandbox
        image instead, regardless of who invoked the runner."""
        assert _opt_values(claude_cmd, "--user") == [f"{policy.SANDBOX_UID}:{policy.SANDBOX_GID}"]

    def test_worker_container_never_runs_as_root(self, claude_cmd):
        assert "0:0" not in _opt_values(claude_cmd, "--user")

    def test_worker_keeps_container_stdin_open_for_print_prompt(self, claude_cmd):
        assert "--interactive" in claude_cmd

    def test_verifier_does_not_open_interactive_stdin(self, repo):
        cmd = runner.build_verification_docker_command(
            repo_root=str(repo), command=["/usr/bin/git", "status"],
        )
        assert "--interactive" not in cmd


class TestMounts:
    def test_repo_is_the_only_writable_mount(self, claude_cmd, repo):
        mounts = _opt_values(claude_cmd, "--mount")
        assert mounts == [
            f"type=bind,src={os.path.realpath(str(repo))},dst=/workspace"
        ]

    def test_credentials_mount_is_read_only_at_the_expected_path(self, repo, tmp_path):
        creds = tmp_path / ".credentials.json"
        creds.write_text("{}", encoding="utf-8")
        cmd = runner.build_claude_docker_command(
            repo_root=str(repo), model="claude-sonnet-5", credentials_path=str(creds),
        )
        mounts = _opt_values(cmd, "--mount")
        assert len(mounts) == 2
        assert mounts[1] == (
            f"type=bind,src={os.path.realpath(str(creds))},"
            f"dst={policy.CONTAINER_CREDENTIALS_PATH},readonly"
        )

    def test_mounts_are_structured_bind_mounts_not_volume_flags(self, claude_cmd):
        for forbidden in ("--volume", "-v", "--volumes-from", "--device"):
            assert forbidden not in claude_cmd
        for mount in _opt_values(claude_cmd, "--mount"):
            assert _mount_field(mount, "type") == "bind"
            assert _mount_field(mount, "src")
            assert _mount_field(mount, "dst")

    def test_repo_path_with_comma_is_refused_before_docker_argv(self, tmp_path):
        injected = tmp_path / "repo,src=etc"
        injected.mkdir()
        with pytest.raises(runner.SpawnRefused, match="comma-delimited"):
            runner.build_claude_docker_command(
                repo_root=str(injected), model="claude-sonnet-5", credentials_path=None,
            )

    def test_credentials_path_with_comma_is_refused_before_docker_argv(self, repo, tmp_path):
        injected = tmp_path / "oauth,src=etc"
        injected.write_text("{}", encoding="utf-8")
        with pytest.raises(runner.SpawnRefused, match="comma-delimited"):
            runner.build_claude_docker_command(
                repo_root=str(repo), model="claude-sonnet-5", credentials_path=str(injected),
            )

    def test_no_docker_socket_bind_mount(self, repo, tmp_path):
        creds = tmp_path / ".credentials.json"
        creds.write_text("{}", encoding="utf-8")
        cmd = runner.build_claude_docker_command(
            repo_root=str(repo), model="claude-sonnet-5", credentials_path=str(creds),
        )
        for mount in _opt_values(cmd, "--mount"):
            assert "docker.sock" not in mount
            assert "/var/run" not in mount

    def test_no_host_root_or_home_mount(self, repo, tmp_path):
        creds = tmp_path / ".credentials.json"
        creds.write_text("{}", encoding="utf-8")
        cmd = runner.build_claude_docker_command(
            repo_root=str(repo), model="claude-sonnet-5", credentials_path=str(creds),
        )
        sources = [_mount_field(m, "src") for m in _opt_values(cmd, "--mount")]
        assert "/" not in sources
        assert os.path.expanduser("~") not in sources

    def test_repo_path_is_realpathed_before_mounting(self, tmp_path):
        real = tmp_path / "real-repo"
        real.mkdir()
        link = tmp_path / "linked-repo"
        link.symlink_to(real, target_is_directory=True)
        cmd = runner.build_claude_docker_command(
            repo_root=str(link), model="claude-sonnet-5", credentials_path=None,
        )
        mount = _opt_values(cmd, "--mount")[0]
        assert _mount_field(mount, "src") == os.path.realpath(str(real))


class TestClaudeInvocation:
    def test_image_reference_is_the_immutable_id_not_the_tag(self, claude_cmd):
        assert claude_cmd.count(policy.SANDBOX_IMAGE_ID) == 1
        assert policy.SANDBOX_IMAGE_TAG not in claude_cmd

    def test_entrypoint_is_claude(self, claude_cmd):
        assert _opt_values(claude_cmd, "--entrypoint") == ["claude"]

    def test_tools_are_restricted_to_the_five_read_edit_tools(self, claude_cmd):
        allowed = _opt_values(_container_args(claude_cmd), "--allowed-tools")
        assert allowed == ["Read,Edit,Write,Glob,Grep"]

    @pytest.mark.parametrize("forbidden", ["Bash", "WebFetch", "WebSearch", "Agent", "Task"])
    def test_forbidden_tools_are_denied(self, claude_cmd, forbidden):
        allowed = _opt_values(_container_args(claude_cmd), "--allowed-tools")[0].split(",")
        assert forbidden not in allowed
        denied = _opt_values(_container_args(claude_cmd), "--disallowed-tools")[0].split(",")
        assert forbidden in denied

    def test_mcp_is_strict_and_empty(self, claude_cmd):
        args = _container_args(claude_cmd)
        assert "--strict-mcp-config" in args
        mcp = _opt_values(args, "--mcp-config")
        assert len(mcp) == 1
        assert json.loads(mcp[0]) == {"mcpServers": {}}

    def test_setting_sources_are_disabled(self, claude_cmd):
        args = _container_args(claude_cmd)
        assert _opt_values(args, "--setting-sources") == [""]

    def test_ephemeral_settings_deny_dangerous_tools(self, claude_cmd):
        settings = json.loads(_opt_values(_container_args(claude_cmd), "--settings")[0])
        assert settings["permissions"]["defaultMode"] == "acceptEdits"
        assert set(policy.CLAUDE_ALLOWED_TOOLS) <= set(settings["permissions"]["allow"])
        for denied in policy.CLAUDE_DENIED_TOOLS:
            assert denied in settings["permissions"]["deny"]

    def test_permission_mode_is_accept_edits(self, claude_cmd):
        assert _opt_values(_container_args(claude_cmd), "--permission-mode") == ["acceptEdits"]

    def test_never_skips_permissions(self, claude_cmd):
        assert "--dangerously-skip-permissions" not in claude_cmd
        assert "bypassPermissions" not in " ".join(claude_cmd)

    def test_model_is_passed_explicitly(self, repo):
        cmd = runner.build_claude_docker_command(
            repo_root=str(repo), model="claude-opus-5", credentials_path=None,
        )
        assert _opt_values(_container_args(cmd), "--model") == ["claude-opus-5"]

    def test_model_outside_the_allowlist_is_refused(self, repo):
        with pytest.raises(Exception):
            runner.build_claude_docker_command(
                repo_root=str(repo), model="claude-3-opus", credentials_path=None,
            )

    def test_prompt_is_not_placed_in_argv(self, repo):
        """The task text goes over stdin, so it never lands in the host
        process list, ``docker inspect``, or any log of the argv."""
        cmd = runner.build_claude_docker_command(
            repo_root=str(repo), model="claude-sonnet-5", credentials_path=None,
        )
        assert "--print" in cmd
        assert cmd[cmd.index("--print") + 1].startswith("--")

    def test_output_format_is_json(self, claude_cmd):
        assert _opt_values(_container_args(claude_cmd), "--output-format") == ["json"]


class TestContainerEnvironment:
    def test_only_home_and_config_dir_are_injected(self, claude_cmd):
        env_values = _opt_values(claude_cmd, "--env")
        keys = {value.split("=", 1)[0] for value in env_values}
        assert keys == {"HOME", "CLAUDE_CONFIG_DIR"}

    def test_no_anthropic_or_token_env_reaches_the_container(self, claude_cmd):
        joined = " ".join(claude_cmd)
        assert "ANTHROPIC_API_KEY" not in joined
        assert "ANTHROPIC_BASE_URL" not in joined
        assert "CLAUDE_CODE_OAUTH_TOKEN" not in joined

    def test_env_file_and_env_passthrough_are_not_used(self, claude_cmd):
        assert "--env-file" not in claude_cmd
        for value in _opt_values(claude_cmd, "--env"):
            assert "=" in value  # never a bare passthrough of a host var


class TestHostChildEnv:
    def test_no_credentials_are_handed_to_the_docker_cli(self):
        source = {
            "HOME": "/root", "PATH": "/usr/bin",
            "CLAUDE_CODE_OAUTH_TOKEN": "sk-oauth-abc",
            "ANTHROPIC_API_KEY": "sk-ant-abc",
            "OPENAI_API_KEY": "sk-openai", "GH_TOKEN": "ghp_x",
            "AWS_SECRET_ACCESS_KEY": "aws", "DISCORD_BOT_TOKEN": "disc",
        }
        env = runner.build_child_env(source)
        for leaked in ("CLAUDE_CODE_OAUTH_TOKEN", "ANTHROPIC_API_KEY", "OPENAI_API_KEY",
                       "GH_TOKEN", "AWS_SECRET_ACCESS_KEY", "DISCORD_BOT_TOKEN"):
            assert leaked not in env

    def test_required_runtime_vars_preserved(self):
        source = {"HOME": "/home/u", "PATH": "/usr/local/bin:/usr/bin", "LANG": "en_US.UTF-8"}
        env = runner.build_child_env(source)
        assert env["PATH"] == "/usr/bin:/bin"
        assert env["LANG"] == "en_US.UTF-8"

    def test_term_is_preserved_when_present(self):
        env = runner.build_child_env({"PATH": "/usr/bin", "TERM": "xterm-256color"})
        assert env["TERM"] == "xterm-256color"

    def test_home_is_always_the_isolated_literal_never_the_ambient_value(self):
        env = runner.build_child_env({"HOME": "/home/u", "PATH": "/usr/bin"})
        assert env["HOME"] == "/nonexistent"

    def test_hostile_home_is_ignored_not_sanitized(self):
        for hostile in ("/", "/root", "/home/attacker/.ssh", ""):
            env = runner.build_child_env({"HOME": hostile, "PATH": "/usr/bin"})
            assert env["HOME"] == "/nonexistent"

    def test_ambient_docker_host_is_ignored(self):
        env = runner.build_child_env(
            {"PATH": "/usr/bin", "DOCKER_HOST": "tcp://attacker.example:2375"}
        )
        assert "DOCKER_HOST" not in env

    def test_ambient_docker_context_and_config_are_ignored(self):
        env = runner.build_child_env({
            "PATH": "/usr/bin",
            "DOCKER_CONTEXT": "attacker-context",
            "DOCKER_CONFIG": "/tmp/attacker-docker-config",
        })
        assert "DOCKER_CONTEXT" not in env
        assert "DOCKER_CONFIG" not in env

    def test_hostile_ambient_env_is_fully_neutralized(self):
        source = {
            "HOME": "/root",
            "PATH": "/usr/local/bin:/usr/bin",
            "LANG": "en_US.UTF-8",
            "TERM": "xterm-256color",
            "DOCKER_HOST": "tcp://attacker.example:2375",
            "DOCKER_CONTEXT": "attacker-context",
            "DOCKER_CONFIG": "/tmp/attacker-docker-config",
            "ANTHROPIC_API_KEY": "sk-ant-abc",
            "CLAUDE_CODE_OAUTH_TOKEN": "sk-oauth-abc",
        }
        env = runner.build_child_env(source)
        assert env["HOME"] == "/nonexistent"
        assert env["PATH"] == "/usr/bin:/bin"
        assert env["LANG"] == "en_US.UTF-8"
        assert env["TERM"] == "xterm-256color"
        for blocked in (
            "DOCKER_HOST", "DOCKER_CONTEXT", "DOCKER_CONFIG",
            "ANTHROPIC_API_KEY", "CLAUDE_CODE_OAUTH_TOKEN",
        ):
            assert blocked not in env

    def test_default_source_is_os_environ(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "leak-if-copied-blindly")
        monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "leak-too")
        monkeypatch.setenv("DOCKER_HOST", "tcp://attacker.example:2375")
        monkeypatch.setenv("HOME", "/root")
        env = runner.build_child_env()
        assert "ANTHROPIC_API_KEY" not in env
        assert "CLAUDE_CODE_OAUTH_TOKEN" not in env
        assert "DOCKER_HOST" not in env
        assert env["HOME"] == "/nonexistent"

    def test_hostile_path_fake_docker_is_never_invoked(self, tmp_path, monkeypatch):
        """A hostile ``$PATH`` entry placed ahead of ``/usr/bin`` in the
        ambient environment must never influence what actually runs: the
        docker CLI's own child env gets the fixed literal ``$PATH``, and
        the real docker invocation ``_docker_base_argv`` builds always
        starts with the absolute ``policy.DOCKER_BIN``, never a bare
        ``"docker"`` resolved by searching ``$PATH``."""
        hostile_bin = tmp_path / "hostile"
        hostile_bin.mkdir()
        fake_docker = hostile_bin / "docker"
        fake_docker.write_text("#!/bin/sh\necho pwned\n", encoding="utf-8")
        fake_docker.chmod(0o755)

        source = {"PATH": f"{hostile_bin}:/usr/bin", "HOME": "/root"}
        env = runner.build_child_env(source)
        assert env["PATH"] == "/usr/bin:/bin"
        assert str(hostile_bin) not in env["PATH"]

        monkeypatch.setenv("PATH", f"{hostile_bin}:/usr/bin")
        argv = runner._docker_base_argv()
        assert argv[0] == policy.DOCKER_BIN == "/usr/bin/docker"
        assert os.path.isabs(argv[0])
        assert str(hostile_bin) not in argv[0]


class TestVerificationCommand:
    def test_verification_runs_with_no_network(self, repo):
        cmd = runner.build_verification_docker_command(
            repo_root=str(repo), command=["pytest", "-q"],
        )
        assert _opt_values(cmd, "--network") == ["none"]

    def test_verification_uses_the_fixed_docker_host_endpoint(self, repo):
        cmd = runner.build_verification_docker_command(repo_root=str(repo), command=["pytest"])
        assert cmd[0] == policy.DOCKER_BIN
        assert cmd[1] == "--host"
        assert cmd[2] == policy.DOCKER_HOST_ENDPOINT

    def test_verification_uses_the_immutable_image_id_not_the_tag(self, repo):
        cmd = runner.build_verification_docker_command(repo_root=str(repo), command=["pytest"])
        assert cmd.count(policy.SANDBOX_IMAGE_ID) == 1
        assert policy.SANDBOX_IMAGE_TAG not in cmd

    def test_verification_never_mounts_credentials(self, repo):
        cmd = runner.build_verification_docker_command(
            repo_root=str(repo), command=["pytest", "-q"],
        )
        mounts = _opt_values(cmd, "--mount")
        assert mounts == [f"type=bind,src={os.path.realpath(str(repo))},dst=/workspace"]
        assert ".credentials.json" not in " ".join(cmd)

    def test_verification_uses_the_same_hardening(self, repo):
        cmd = runner.build_verification_docker_command(repo_root=str(repo), command=["pytest"])
        assert "--rm" in cmd and "--read-only" in cmd
        assert _opt_values(cmd, "--cap-drop") == ["ALL"]
        assert "no-new-privileges" in " ".join(_opt_values(cmd, "--security-opt"))
        assert _opt_values(cmd, "--pids-limit") == [str(policy.PIDS_LIMIT)]
        assert _opt_values(cmd, "--workdir") == ["/workspace"]

    def test_verification_argv_is_passed_verbatim_after_the_image(self, repo):
        cmd = runner.build_verification_docker_command(
            repo_root=str(repo), command=["pytest", "-q", "tests/unit"],
        )
        assert _opt_values(cmd, "--entrypoint") == ["pytest"]
        assert _container_args(cmd) == ["-q", "tests/unit"]

    def test_verification_never_uses_a_shell(self, repo):
        cmd = runner.build_verification_docker_command(
            repo_root=str(repo), command=["pytest", "-q && curl evil.example"],
        )
        # The metacharacters stay inside one argv element; no shell parses them.
        assert "-q && curl evil.example" in cmd
        assert "sh" not in _opt_values(cmd, "--entrypoint")

    def test_empty_verification_command_is_refused(self, repo):
        with pytest.raises(ValueError):
            runner.build_verification_docker_command(repo_root=str(repo), command=[])

    def test_verification_runs_as_the_fixed_sandbox_uid(self, repo):
        cmd = runner.build_verification_docker_command(repo_root=str(repo), command=["pytest"])
        assert _opt_values(cmd, "--user") == [f"{policy.SANDBOX_UID}:{policy.SANDBOX_GID}"]

    def test_verification_container_never_runs_as_root(self, repo):
        cmd = runner.build_verification_docker_command(repo_root=str(repo), command=["pytest"])
        assert "0:0" not in _opt_values(cmd, "--user")


class TestNoStructuredMountEverUsesLegacyRwToken:
    """``--mount type=bind,...`` is docker's structured syntax: the default
    (omitted suffix) already means read-write, and a bare ``rw`` field is
    only valid in the legacy ``-v src:dst:rw`` form — passing it here would
    be silently rejected by docker as an unknown key. No command this
    runner builds, worker or verifier, may ever emit that token."""

    def test_worker_command_never_emits_a_bare_rw_mount_token(self, claude_cmd):
        for mount in _opt_values(claude_cmd, "--mount"):
            assert not mount.endswith(",rw")
            for field in mount.split(","):
                assert field != "rw"

    def test_verification_command_never_emits_a_bare_rw_mount_token(self, repo):
        cmd = runner.build_verification_docker_command(repo_root=str(repo), command=["pytest"])
        for mount in _opt_values(cmd, "--mount"):
            assert not mount.endswith(",rw")
            for field in mount.split(","):
                assert field != "rw"


_FAKE_CONFIGURED_IMAGE_ID = "sha256:" + "1" * 64
_FAKE_RETAGGED_IMAGE_ID = "sha256:" + "9" * 64


class TestPreflightFailsClosed:
    @pytest.fixture(autouse=True)
    def _clear_cache(self, monkeypatch):
        runner.reset_preflight_cache()
        monkeypatch.setattr(policy, "SANDBOX_IMAGE_ID_CONFIGURED", True)
        monkeypatch.setattr(policy, "SANDBOX_IMAGE_ID", _FAKE_CONFIGURED_IMAGE_ID)
        yield
        runner.reset_preflight_cache()

    def _install_fake_docker(
        self, monkeypatch, *, which=True, info_rc=0, inspect_rc=0,
        inspect_id=_FAKE_CONFIGURED_IMAGE_ID, help_text=None,
    ):
        if help_text is None:
            help_text = (
                "Usage: claude [options]\n  --strict-mcp-config\n  --settings <file-or-json>\n"
                "  --setting-sources <sources>\n"
            )
        monkeypatch.setattr(runner.shutil, "which", lambda name: "/usr/bin/docker" if which else None)
        calls = []
        envs = []

        class _Completed:
            def __init__(self, returncode, stdout=""):
                self.returncode = returncode
                self.stdout = stdout
                self.stderr = ""

        def _fake_run(cmd, **kwargs):
            calls.append(cmd)
            envs.append(kwargs.get("env"))
            if "info" in cmd:
                return _Completed(info_rc, "ServerVersion")
            if "image" in cmd and "inspect" in cmd:
                payload = json.dumps([{"Id": inspect_id}])
                return _Completed(inspect_rc, payload)
            return _Completed(0, help_text)

        monkeypatch.setattr(runner.subprocess, "run", _fake_run)
        return calls, envs

    def test_all_checks_pass(self, monkeypatch):
        self._install_fake_docker(monkeypatch)
        result = runner.docker_preflight()
        assert result["ok"] is True
        assert result["image_present"] is True

    def test_missing_docker_binary_fails_closed(self, monkeypatch):
        self._install_fake_docker(monkeypatch, which=False)
        result = runner.docker_preflight()
        assert result["ok"] is False
        assert "docker" in result["reason"].lower()

    def test_dead_daemon_fails_closed(self, monkeypatch):
        self._install_fake_docker(monkeypatch, info_rc=1)
        result = runner.docker_preflight()
        assert result["ok"] is False
        assert result["daemon_ok"] is False

    def test_missing_image_fails_closed(self, monkeypatch):
        self._install_fake_docker(monkeypatch, inspect_rc=1)
        result = runner.docker_preflight()
        assert result["ok"] is False
        assert result["image_present"] is False
        assert policy.SANDBOX_IMAGE_TAG in result["reason"]

    def test_placeholder_image_id_not_configured_fails_closed_without_touching_docker(self, monkeypatch):
        monkeypatch.setattr(policy, "SANDBOX_IMAGE_ID_CONFIGURED", False)
        calls, _ = self._install_fake_docker(monkeypatch)
        result = runner.docker_preflight()
        assert result["ok"] is False
        assert result["image_present"] is False
        assert calls == [], "must refuse before ever invoking docker while the id is a placeholder"

    def test_retagged_or_mismatched_image_fails_closed(self, monkeypatch):
        self._install_fake_docker(monkeypatch, inspect_id=_FAKE_RETAGGED_IMAGE_ID)
        result = runner.docker_preflight()
        assert result["ok"] is False
        assert result["image_present"] is False
        assert _FAKE_RETAGGED_IMAGE_ID in result["reason"] or "retag" in result["reason"].lower() \
            or "mismatch" in result["reason"].lower()

    @pytest.mark.parametrize("missing", ["--strict-mcp-config", "--settings", "--setting-sources"])
    def test_missing_required_cli_flag_fails_closed(self, monkeypatch, missing):
        help_text = " ".join(flag for flag in policy.REQUIRED_CLI_FLAGS if flag != missing)
        self._install_fake_docker(monkeypatch, help_text=help_text)
        result = runner.docker_preflight()
        assert result["ok"] is False
        assert missing in result["reason"]

    def test_probe_exception_fails_closed(self, monkeypatch):
        monkeypatch.setattr(runner.shutil, "which", lambda name: "/usr/bin/docker")

        def _boom(*a, **k):
            raise OSError("daemon exploded")

        monkeypatch.setattr(runner.subprocess, "run", _boom)
        assert runner.docker_preflight()["ok"] is False

    def test_result_is_cached(self, monkeypatch):
        calls, _ = self._install_fake_docker(monkeypatch)
        runner.docker_preflight()
        first = len(calls)
        runner.docker_preflight()
        assert len(calls) == first

    def test_help_probe_runs_in_a_container_with_no_network(self, monkeypatch):
        calls, _ = self._install_fake_docker(monkeypatch)
        runner.docker_preflight()
        help_cmds = [cmd for cmd in calls if "--help" in cmd]
        assert help_cmds, "preflight must confirm flags inside the container"
        assert "--network" in help_cmds[0]
        assert help_cmds[0][help_cmds[0].index("--network") + 1] == "none"

    def test_help_probe_uses_the_immutable_image_id(self, monkeypatch):
        calls, _ = self._install_fake_docker(monkeypatch)
        runner.docker_preflight()
        help_cmds = [cmd for cmd in calls if "--help" in cmd]
        assert help_cmds
        assert policy.SANDBOX_IMAGE_ID in help_cmds[0]
        assert policy.SANDBOX_IMAGE_TAG not in help_cmds[0]

    def test_all_docker_stages_use_the_fixed_host_and_the_isolated_child_env(self, monkeypatch):
        calls, envs = self._install_fake_docker(monkeypatch)
        runner.docker_preflight()
        assert calls, "expected the preflight to invoke docker at least once"
        for cmd, env in zip(calls, envs):
            assert cmd[0] == policy.DOCKER_BIN
            assert cmd[1] == "--host"
            assert cmd[2] == policy.DOCKER_HOST_ENDPOINT
            assert env is not None
            assert env.get("HOME") == "/nonexistent"
            assert "DOCKER_HOST" not in env
            assert "DOCKER_CONTEXT" not in env
            assert "DOCKER_CONFIG" not in env

    def test_spawn_refused_when_preflight_fails(self, monkeypatch, repo):
        self._install_fake_docker(monkeypatch, inspect_rc=1)
        ran = {"n": 0}
        monkeypatch.setattr(runner, "_run_subprocess", lambda *a, **k: ran.__setitem__("n", ran["n"] + 1))
        with pytest.raises(runner.SpawnRefused):
            runner.spawn_claude(
                task="fix a bug", cwd=str(repo), model="claude-sonnet-5", timeout_seconds=30,
            )
        assert ran["n"] == 0


class TestOAuthCredentialsValidation:
    """The one host OAuth credentials path is validated with ``lstat``, not
    a symlink-following ``stat``/``isfile`` check: a missing path, a
    symlink, a directory, or a FIFO at that exact location is refused
    before any docker run — only an exact regular file may ever be
    mounted into the sandbox."""

    def test_missing_credentials_file_refuses(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            policy, "HOST_CREDENTIALS_PATH", str(tmp_path / "nowhere" / ".credentials.json"),
        )
        with pytest.raises(runner.SpawnRefused):
            runner._resolve_credentials_path()

    def test_symlink_credentials_file_refuses(self, tmp_path, monkeypatch):
        target = tmp_path / "real.json"
        target.write_text("{}", encoding="utf-8")
        link = tmp_path / ".credentials.json"
        link.symlink_to(target)
        monkeypatch.setattr(policy, "HOST_CREDENTIALS_PATH", str(link))
        with pytest.raises(runner.SpawnRefused):
            runner._resolve_credentials_path()

    def test_directory_at_credentials_path_refuses(self, tmp_path, monkeypatch):
        path = tmp_path / ".credentials.json"
        path.mkdir()
        monkeypatch.setattr(policy, "HOST_CREDENTIALS_PATH", str(path))
        with pytest.raises(runner.SpawnRefused):
            runner._resolve_credentials_path()

    def test_fifo_at_credentials_path_refuses(self, tmp_path, monkeypatch):
        path = tmp_path / ".credentials.json"
        os.mkfifo(str(path))
        monkeypatch.setattr(policy, "HOST_CREDENTIALS_PATH", str(path))
        with pytest.raises(runner.SpawnRefused):
            runner._resolve_credentials_path()

    def test_regular_file_credentials_path_is_accepted(self, tmp_path, monkeypatch):
        path = tmp_path / ".credentials.json"
        path.write_text("{}", encoding="utf-8")
        monkeypatch.setattr(policy, "HOST_CREDENTIALS_PATH", str(path))
        assert runner._resolve_credentials_path() == str(path)


class TestSpawnPlumbing:
    @pytest.fixture(autouse=True)
    def _preflight_ok(self, monkeypatch):
        runner.reset_preflight_cache()
        monkeypatch.setattr(
            runner, "docker_preflight",
            lambda *a, **k: {"ok": True, "reason": "", "docker_present": True,
                             "daemon_ok": True, "image_present": True},
        )
        yield
        runner.reset_preflight_cache()

    @pytest.fixture(autouse=True)
    def _valid_credentials(self, monkeypatch, tmp_path):
        creds = tmp_path / "default-credentials" / ".credentials.json"
        creds.parent.mkdir()
        creds.write_text("{}", encoding="utf-8")
        monkeypatch.setattr(policy, "HOST_CREDENTIALS_PATH", str(creds))

    def test_task_is_written_to_stdin(self, monkeypatch, repo):
        captured = {}

        class _Completed:
            returncode = 0
            stdout = json.dumps({"result": "done"})
            stderr = ""

        def _fake_run(cmd, cwd=None, env=None, timeout_seconds=None, stdin_text=None):
            captured["cmd"] = cmd
            captured["stdin"] = stdin_text
            captured["env"] = env
            return _Completed()

        monkeypatch.setattr(runner, "_run_subprocess", _fake_run)
        result = runner.spawn_claude(
            task="fix the parser", cwd=str(repo), model="claude-sonnet-5", timeout_seconds=30,
        )
        assert captured["stdin"] == "fix the parser"
        assert "fix the parser" not in " ".join(captured["cmd"])
        assert result["exit_code"] == 0

    def test_timeout_is_reported_not_raised(self, monkeypatch, repo):
        removed = []
        monkeypatch.setattr(
            runner, "_force_remove_container",
            lambda cidfile, env=None: removed.append(cidfile.read_text().strip()),
            raising=False,
        )

        def _timeout(cmd, *args, **kwargs):
            cidfile = Path(cmd[cmd.index("--cidfile") + 1])
            cidfile.write_text("a" * 64, encoding="ascii")
            raise runner.subprocess.TimeoutExpired(cmd=["docker"], timeout=1)

        monkeypatch.setattr(runner, "_run_subprocess", _timeout)
        result = runner.spawn_claude(
            task="t", cwd=str(repo), model="claude-sonnet-5", timeout_seconds=1,
        )
        assert result["timed_out"] is True
        assert result["exit_code"] is None
        assert removed == ["a" * 64]

    def test_no_temp_files_are_left_behind(self, monkeypatch, repo, tmp_path):
        scratch = tmp_path / "tmpdir"
        scratch.mkdir()
        monkeypatch.setenv("TMPDIR", str(scratch))

        class _Completed:
            returncode = 0
            stdout = "{}"
            stderr = ""

        monkeypatch.setattr(runner, "_run_subprocess", lambda *a, **k: _Completed())
        runner.spawn_claude(task="t", cwd=str(repo), model="claude-sonnet-5", timeout_seconds=5)
        assert list(scratch.iterdir()) == []

    def test_missing_credentials_refuses_spawn_before_any_docker_run(self, monkeypatch, repo, tmp_path):
        monkeypatch.setattr(
            policy, "HOST_CREDENTIALS_PATH", str(tmp_path / "nowhere" / ".credentials.json"),
        )
        ran = {"n": 0}
        monkeypatch.setattr(runner, "_run_subprocess", lambda *a, **k: ran.__setitem__("n", ran["n"] + 1))
        with pytest.raises(runner.SpawnRefused):
            runner.spawn_claude(task="t", cwd=str(repo), model="claude-sonnet-5", timeout_seconds=5)
        assert ran["n"] == 0

    def test_symlinked_credentials_refuses_spawn_before_any_docker_run(self, monkeypatch, repo, tmp_path):
        target = tmp_path / "real.json"
        target.write_text("{}", encoding="utf-8")
        link = tmp_path / ".credentials.json"
        link.symlink_to(target)
        monkeypatch.setattr(policy, "HOST_CREDENTIALS_PATH", str(link))
        ran = {"n": 0}
        monkeypatch.setattr(runner, "_run_subprocess", lambda *a, **k: ran.__setitem__("n", ran["n"] + 1))
        with pytest.raises(runner.SpawnRefused):
            runner.spawn_claude(task="t", cwd=str(repo), model="claude-sonnet-5", timeout_seconds=5)
        assert ran["n"] == 0

    def test_credentials_present_adds_the_read_only_mount(self, monkeypatch, repo, tmp_path):
        creds = tmp_path / ".credentials.json"
        creds.write_text("{}", encoding="utf-8")
        monkeypatch.setattr(policy, "HOST_CREDENTIALS_PATH", str(creds))
        captured = {}

        class _Completed:
            returncode = 0
            stdout = json.dumps({"result": "done"})
            stderr = ""

        def _fake_run(cmd, **kwargs):
            captured["cmd"] = cmd
            return _Completed()

        monkeypatch.setattr(runner, "_run_subprocess", _fake_run)
        runner.spawn_claude(task="t", cwd=str(repo), model="claude-sonnet-5", timeout_seconds=5)

        mounts = _opt_values(captured["cmd"], "--mount")
        assert len(mounts) == 2
        assert mounts[0] == f"type=bind,src={os.path.realpath(str(repo))},dst=/workspace"
        # The ORIGINAL root-owned credentials path is never mounted — the
        # container runs as the fixed non-root sandbox uid, which could
        # never read it. Only the staged, re-owned copy is.
        creds_src = _mount_field(mounts[1], "src")
        assert creds_src != os.path.realpath(str(creds))
        assert _mount_field(mounts[1], "dst") == policy.CONTAINER_CREDENTIALS_PATH
        assert _mount_field(mounts[1], "readonly") == ""
        assert str(creds) not in " ".join(captured["cmd"])


class TestSpawnUnexpectedException:
    @pytest.fixture(autouse=True)
    def _preflight_ok(self, monkeypatch):
        runner.reset_preflight_cache()
        monkeypatch.setattr(
            runner, "docker_preflight",
            lambda *a, **k: {"ok": True, "reason": "", "docker_present": True,
                             "daemon_ok": True, "image_present": True},
        )
        yield
        runner.reset_preflight_cache()

    @pytest.fixture(autouse=True)
    def _valid_credentials(self, monkeypatch, tmp_path):
        creds = tmp_path / "default-credentials" / ".credentials.json"
        creds.parent.mkdir()
        creds.write_text("{}", encoding="utf-8")
        monkeypatch.setattr(policy, "HOST_CREDENTIALS_PATH", str(creds))

    def test_non_timeout_subprocess_error_propagates_to_the_caller(self, monkeypatch, repo):
        """spawn_claude only swallows ``TimeoutExpired`` — any other
        unexpected error (docker daemon dies mid-call, OS-level failure)
        must propagate rather than being reported as a normal result, so
        ``run_worker``'s outer handler is the one place that turns it into
        a single structured failure and telemetry record."""

        def _boom(*a, **k):
            raise OSError("docker vanished mid-run")

        monkeypatch.setattr(runner, "_run_subprocess", _boom)
        with pytest.raises(OSError):
            runner.spawn_claude(task="t", cwd=str(repo), model="claude-sonnet-5", timeout_seconds=5)


class TestVerificationExecution:
    @pytest.fixture(autouse=True)
    def _clear_cache(self):
        runner.reset_preflight_cache()
        yield
        runner.reset_preflight_cache()

    def test_preflight_failure_returns_a_structured_result_without_raising(self, monkeypatch, repo):
        monkeypatch.setattr(
            runner, "docker_preflight",
            lambda: {"ok": False, "reason": "sandbox image not present locally: x",
                      "docker_present": True, "daemon_ok": True, "image_present": False},
        )
        result = runner.run_verification_command(
            repo_root=str(repo), command=["pytest", "-q"], timeout_seconds=30,
        )
        assert result["ok"] is False
        assert result["failure_class"] == "isolation_refused"
        assert "not present" in result["reason"]

    def test_empty_command_returns_a_structured_result_without_raising(self, monkeypatch, repo):
        monkeypatch.setattr(
            runner, "docker_preflight",
            lambda: {"ok": True, "reason": "", "docker_present": True,
                      "daemon_ok": True, "image_present": True},
        )
        result = runner.run_verification_command(
            repo_root=str(repo), command=[], timeout_seconds=30,
        )
        assert result["ok"] is False
        assert result["failure_class"] == "invalid_command"

    def test_unexpected_subprocess_error_returns_a_structured_result_without_raising(self, monkeypatch, repo):
        monkeypatch.setattr(
            runner, "docker_preflight",
            lambda: {"ok": True, "reason": "", "docker_present": True,
                      "daemon_ok": True, "image_present": True},
        )

        def _boom(*a, **k):
            raise OSError("docker vanished mid-run")

        monkeypatch.setattr(runner, "_run_subprocess", _boom)
        result = runner.run_verification_command(
            repo_root=str(repo), command=["pytest", "-q"], timeout_seconds=30,
        )
        assert result["ok"] is False
        assert result["failure_class"] == "internal_error"

    def test_successful_run_captures_bounded_evidence(self, monkeypatch, repo):
        monkeypatch.setattr(
            runner, "docker_preflight",
            lambda: {"ok": True, "reason": "", "docker_present": True,
                      "daemon_ok": True, "image_present": True},
        )

        class _Completed:
            returncode = 0
            stdout = "collected 3 items"
            stderr = ""

        monkeypatch.setattr(runner, "_run_subprocess", lambda *a, **k: _Completed())
        result = runner.run_verification_command(
            repo_root=str(repo), command=["pytest", "-q"], timeout_seconds=30,
        )
        assert result["ok"] is True
        assert result["failure_class"] is None
        assert result["stdout"] == "collected 3 items"

    def test_nested_verification_mounts_configured_root_and_uses_nested_workdir(
        self, monkeypatch, repo,
    ):
        nested = repo / "sub" / "nested"
        nested.mkdir(parents=True)
        monkeypatch.setattr(
            runner, "docker_preflight",
            lambda: {"ok": True, "reason": "", "docker_present": True,
                     "daemon_ok": True, "image_present": True},
        )
        captured = {}

        class _Completed:
            returncode = 0
            stdout = "ok"
            stderr = ""

        def _fake_run(cmd, cwd=None, **kwargs):
            captured["cmd"] = cmd
            captured["cwd"] = cwd
            return _Completed()

        monkeypatch.setattr(runner, "_run_subprocess", _fake_run)
        result = runner.run_verification_command(
            repo_root=str(nested),
            command=["/usr/bin/git", "status"],
            timeout_seconds=30,
            repo_roots=[str(repo)],
        )

        assert result["ok"] is True
        assert captured["cwd"] == os.path.realpath(str(nested))
        assert _opt_values(captured["cmd"], "--mount") == [
            f"type=bind,src={os.path.realpath(str(repo))},dst=/workspace"
        ]
        assert _opt_values(captured["cmd"], "--workdir") == ["/workspace/sub/nested"]


class TestSpawnOpensCredentialsBeforeBuildingCommand:
    """``spawn_claude`` resolves the credentials path, opens it
    (``O_NOFOLLOW``) and snapshots its identity BEFORE calling
    ``docker_preflight`` or building the docker command, then re-validates
    both trusted path chains again immediately after preflight and once
    more directly before the ``_run_subprocess`` call that actually mounts
    them. Any trusted-path mutation in that window — a parent turned into a
    symlink, a parent or the file itself turned world-writable, or the file
    replaced outright — must refuse before ``_run_subprocess`` is ever
    called, and the held descriptor must always end up closed."""

    @pytest.fixture(autouse=True)
    def _preflight_ok(self, monkeypatch):
        runner.reset_preflight_cache()
        monkeypatch.setattr(
            runner, "docker_preflight",
            lambda *a, **k: {"ok": True, "reason": "", "docker_present": True,
                             "daemon_ok": True, "image_present": True},
        )
        yield
        runner.reset_preflight_cache()

    @pytest.fixture()
    def creds(self, tmp_path):
        parent = tmp_path / "cred-parent"
        parent.mkdir()
        os.chmod(str(parent), 0o755)  # mkdir() honors umask; pin it explicitly
        path = parent / ".credentials.json"
        path.write_text("{}", encoding="utf-8")
        path.chmod(0o600)
        return path

    @staticmethod
    def _spy_run_subprocess(monkeypatch):
        calls = {"n": 0}

        class _Completed:
            returncode = 0
            stdout = json.dumps({"result": "done"})
            stderr = ""

        def _fake(*a, **k):
            calls["n"] += 1
            return _Completed()

        monkeypatch.setattr(runner, "_run_subprocess", _fake)
        return calls

    def test_parent_directory_symlink_refuses_before_run(self, monkeypatch, repo, tmp_path, creds):
        linked_parent = tmp_path / "linked-cred-parent"
        linked_parent.symlink_to(creds.parent, target_is_directory=True)
        monkeypatch.setattr(policy, "HOST_CREDENTIALS_PATH", str(linked_parent / ".credentials.json"))

        ran = self._spy_run_subprocess(monkeypatch)
        with pytest.raises(runner.SpawnRefused):
            runner.spawn_claude(task="t", cwd=str(repo), model="claude-sonnet-5", timeout_seconds=5)
        assert ran["n"] == 0

    def test_parent_mode_world_writable_after_open_refuses(self, monkeypatch, repo, creds):
        monkeypatch.setattr(policy, "HOST_CREDENTIALS_PATH", str(creds))
        parent = creds.parent
        original_mode = stat.S_IMODE(os.lstat(str(parent)).st_mode)

        def _preflight_then_mutate(*a, **k):
            os.chmod(str(parent), 0o777)
            return {"ok": True, "reason": "", "docker_present": True,
                    "daemon_ok": True, "image_present": True}

        monkeypatch.setattr(runner, "docker_preflight", _preflight_then_mutate)
        ran = self._spy_run_subprocess(monkeypatch)
        try:
            with pytest.raises(runner.SpawnRefused):
                runner.spawn_claude(task="t", cwd=str(repo), model="claude-sonnet-5", timeout_seconds=5)
            assert ran["n"] == 0
        finally:
            os.chmod(str(parent), original_mode)

    def test_final_file_replaced_after_open_refuses(self, monkeypatch, repo, creds):
        monkeypatch.setattr(policy, "HOST_CREDENTIALS_PATH", str(creds))
        original_mode = stat.S_IMODE(os.lstat(str(creds)).st_mode)

        def _preflight_then_replace(*a, **k):
            os.remove(str(creds))
            creds.write_text("{}", encoding="utf-8")
            creds.chmod(original_mode)  # same mode, different inode
            return {"ok": True, "reason": "", "docker_present": True,
                    "daemon_ok": True, "image_present": True}

        monkeypatch.setattr(runner, "docker_preflight", _preflight_then_replace)
        ran = self._spy_run_subprocess(monkeypatch)
        with pytest.raises(runner.SpawnRefused):
            runner.spawn_claude(task="t", cwd=str(repo), model="claude-sonnet-5", timeout_seconds=5)
        assert ran["n"] == 0

    def test_final_mode_change_after_open_refuses(self, monkeypatch, repo, creds):
        monkeypatch.setattr(policy, "HOST_CREDENTIALS_PATH", str(creds))
        original_mode = stat.S_IMODE(os.lstat(str(creds)).st_mode)

        def _preflight_then_chmod(*a, **k):
            creds.chmod(0o666)
            return {"ok": True, "reason": "", "docker_present": True,
                    "daemon_ok": True, "image_present": True}

        monkeypatch.setattr(runner, "docker_preflight", _preflight_then_chmod)
        ran = self._spy_run_subprocess(monkeypatch)
        try:
            with pytest.raises(runner.SpawnRefused):
                runner.spawn_claude(task="t", cwd=str(repo), model="claude-sonnet-5", timeout_seconds=5)
            assert ran["n"] == 0
        finally:
            creds.chmod(original_mode)

    def test_held_descriptor_is_closed_after_success(self, monkeypatch, repo, creds):
        monkeypatch.setattr(policy, "HOST_CREDENTIALS_PATH", str(creds))
        closed_fds = []
        real_close = os.close

        def _spy_close(fd):
            closed_fds.append(fd)
            real_close(fd)

        monkeypatch.setattr(runner.os, "close", _spy_close)

        class _Completed:
            returncode = 0
            stdout = json.dumps({"result": "done"})
            stderr = ""

        monkeypatch.setattr(runner, "_run_subprocess", lambda *a, **k: _Completed())
        runner.spawn_claude(task="t", cwd=str(repo), model="claude-sonnet-5", timeout_seconds=5)

        # The held ORIGINAL credentials descriptor is opened first and
        # released only in spawn_claude's outer ``finally`` — after the
        # staged copy's own (also legitimately closed) descriptor — so it is
        # always the LAST fd this spy observes being closed.
        assert closed_fds
        with pytest.raises(OSError):
            os.fstat(closed_fds[-1])

    def test_held_descriptor_is_closed_after_run_subprocess_exception(self, monkeypatch, repo, creds):
        monkeypatch.setattr(policy, "HOST_CREDENTIALS_PATH", str(creds))
        closed_fds = []
        real_close = os.close

        def _spy_close(fd):
            closed_fds.append(fd)
            real_close(fd)

        monkeypatch.setattr(runner.os, "close", _spy_close)

        def _boom(*a, **k):
            raise OSError("docker vanished mid-run")

        monkeypatch.setattr(runner, "_run_subprocess", _boom)
        with pytest.raises(OSError):
            runner.spawn_claude(task="t", cwd=str(repo), model="claude-sonnet-5", timeout_seconds=5)

        assert closed_fds
        with pytest.raises(OSError):
            os.fstat(closed_fds[-1])


class TestDockerBinaryTrustHelper:
    """``_validate_docker_binary_trust`` walks the fixed ``policy.DOCKER_BIN``
    path chain fresh on every call. These tests substitute the ``lstat``
    result for individual chain components (the binary itself and its
    immediate parent directory) via monkeypatched snapshots rather than
    mutating the real, shared system ``/usr/bin/docker`` — a test must
    never chmod or replace a real system binary."""

    class _FakeStat:
        def __init__(self, mode, uid=0, gid=0, dev=1, ino=999):
            self.st_mode = mode
            self.st_uid = uid
            self.st_gid = gid
            self.st_dev = dev
            self.st_ino = ino

    @staticmethod
    def _patch_lstat(monkeypatch, overrides):
        real_lstat = os.lstat

        def _fake(path, *a, **k):
            if path in overrides:
                return overrides[path]
            return real_lstat(path, *a, **k)

        monkeypatch.setattr(runner.os, "lstat", _fake)

    def test_final_symlink_refuses(self, monkeypatch):
        self._patch_lstat(monkeypatch, {
            policy.DOCKER_BIN: self._FakeStat(stat.S_IFLNK | 0o777),
        })
        with pytest.raises(runner.TrustViolation):
            runner._validate_docker_binary_trust()

    def test_final_directory_refuses(self, monkeypatch):
        self._patch_lstat(monkeypatch, {
            policy.DOCKER_BIN: self._FakeStat(stat.S_IFDIR | 0o755),
        })
        with pytest.raises(runner.TrustViolation):
            runner._validate_docker_binary_trust()

    def test_final_non_executable_refuses(self, monkeypatch):
        self._patch_lstat(monkeypatch, {
            policy.DOCKER_BIN: self._FakeStat(stat.S_IFREG | 0o644),
        })
        with pytest.raises(runner.TrustViolation):
            runner._validate_docker_binary_trust()

    def test_final_world_writable_refuses(self, monkeypatch):
        self._patch_lstat(monkeypatch, {
            policy.DOCKER_BIN: self._FakeStat(stat.S_IFREG | 0o757),
        })
        with pytest.raises(runner.TrustViolation):
            runner._validate_docker_binary_trust()

    def test_parent_symlink_refuses(self, monkeypatch):
        parent = os.path.dirname(policy.DOCKER_BIN)
        self._patch_lstat(monkeypatch, {
            parent: self._FakeStat(stat.S_IFLNK | 0o777),
        })
        with pytest.raises(runner.TrustViolation):
            runner._validate_docker_binary_trust()

    def test_parent_world_writable_refuses(self, monkeypatch):
        parent = os.path.dirname(policy.DOCKER_BIN)
        self._patch_lstat(monkeypatch, {
            parent: self._FakeStat(stat.S_IFDIR | 0o777),
        })
        with pytest.raises(runner.TrustViolation):
            runner._validate_docker_binary_trust()

    def test_valid_path_passes_and_reaches_run(self, monkeypatch, repo, tmp_path):
        # No overrides: the real docker binary chain on this host is
        # already trusted (root-owned, non-symlink, owner-executable, never
        # group/world-writable) — the check must pass cleanly, and
        # spawn_claude must proceed all the way to ``_run_subprocess``.
        snapshot = runner._validate_docker_binary_trust()
        assert policy.DOCKER_BIN in snapshot

        runner.reset_preflight_cache()
        monkeypatch.setattr(
            runner, "docker_preflight",
            lambda *a, **k: {"ok": True, "reason": "", "docker_present": True,
                             "daemon_ok": True, "image_present": True},
        )
        creds = tmp_path / ".credentials.json"
        creds.write_text("{}", encoding="utf-8")
        creds.chmod(0o600)
        monkeypatch.setattr(policy, "HOST_CREDENTIALS_PATH", str(creds))

        ran = {"n": 0}

        class _Completed:
            returncode = 0
            stdout = json.dumps({"result": "done"})
            stderr = ""

        def _fake(*a, **k):
            ran["n"] += 1
            return _Completed()

        monkeypatch.setattr(runner, "_run_subprocess", _fake)
        try:
            result = runner.spawn_claude(
                task="t", cwd=str(repo), model="claude-sonnet-5", timeout_seconds=5,
            )
        finally:
            runner.reset_preflight_cache()
        assert ran["n"] == 1
        assert result["exit_code"] == 0


class TestCwdPathChainRace:
    """``spawn_claude`` snapshots the canonical repo cwd directory chain
    (``/`` through the resolved cwd — device/inode/mode/uid/gid, no
    symlinks anywhere) BEFORE preflight, then re-validates that whole chain
    AND containment inside the canonical ``repo_roots`` twice more: right
    after the docker command is built, and immediately before
    ``_run_subprocess``. A nested cwd (or any ancestor up to the repo root)
    replaced by a symlink escaping the allowlisted root between those
    checkpoints must refuse with ``SpawnRefused`` before any subprocess is
    started — never fall back to running with the swapped, now-out-of-scope
    directory.
    """

    @pytest.fixture(autouse=True)
    def _preflight_ok(self, monkeypatch):
        runner.reset_preflight_cache()
        monkeypatch.setattr(
            runner, "docker_preflight",
            lambda *a, **k: {"ok": True, "reason": "", "docker_present": True,
                             "daemon_ok": True, "image_present": True},
        )
        yield
        runner.reset_preflight_cache()

    @pytest.fixture(autouse=True)
    def _valid_credentials(self, monkeypatch, tmp_path):
        creds = tmp_path / "default-credentials" / ".credentials.json"
        creds.parent.mkdir()
        creds.write_text("{}", encoding="utf-8")
        monkeypatch.setattr(policy, "HOST_CREDENTIALS_PATH", str(creds))

    @pytest.fixture()
    def nested_cwd(self, repo):
        sub = repo / "sub" / "nested"
        sub.mkdir(parents=True)
        return sub

    @staticmethod
    def _fake_completed():
        class _Completed:
            returncode = 0
            stdout = json.dumps({"result": "done"})
            stderr = ""

        return _Completed()

    def test_valid_nested_cwd_runs_normally(self, monkeypatch, repo, nested_cwd):
        captured = {}

        def _fake_run(cmd, cwd=None, **kwargs):
            captured["cmd"] = cmd
            captured["cwd"] = cwd
            return self._fake_completed()

        monkeypatch.setattr(runner, "_run_subprocess", _fake_run)
        result = runner.spawn_claude(
            task="t", cwd=str(nested_cwd), model="claude-sonnet-5", timeout_seconds=5,
            repo_roots=[str(repo)],
        )
        assert result["exit_code"] == 0
        assert captured["cwd"] == os.path.realpath(str(nested_cwd))
        assert _opt_values(captured["cmd"], "--mount")[0] == (
            f"type=bind,src={os.path.realpath(str(repo))},dst=/workspace"
        )
        assert _opt_values(captured["cmd"], "--workdir") == ["/workspace/sub/nested"]

    def test_mount_root_is_revalidated_at_final_subprocess_checkpoint(
        self, monkeypatch, repo, nested_cwd,
    ):
        original_mode = stat.S_IMODE(os.stat(repo).st_mode)
        ran = {"n": 0}

        # Isolate the mount-root defense: if the older cwd-chain layer were
        # the only protection, neutralising it would let this mutation run.
        monkeypatch.setattr(runner, "_revalidate_repo_cwd_chain", lambda *a, **k: None)
        real_validate_docker = runner._validate_docker_binary_trust

        def _validate_then_mutate_mount_root():
            real_validate_docker()
            os.chmod(repo, original_mode ^ stat.S_IXUSR)

        monkeypatch.setattr(runner, "_validate_docker_binary_trust", _validate_then_mutate_mount_root)
        monkeypatch.setattr(
            runner, "_run_subprocess",
            lambda *a, **k: ran.__setitem__("n", ran["n"] + 1),
        )
        try:
            with pytest.raises(runner.SpawnRefused, match="mount"):
                runner.spawn_claude(
                    task="t", cwd=str(nested_cwd), model="claude-sonnet-5",
                    timeout_seconds=5, repo_roots=[str(repo)],
                )
            assert ran["n"] == 0
        finally:
            os.chmod(repo, original_mode)

    def test_mount_plan_uses_most_specific_configured_root(self, tmp_path):
        broad = tmp_path / "broad"
        specific = broad / "specific"
        nested = specific / "sub"
        nested.mkdir(parents=True)

        mount_root, container_cwd, _ = runner._mount_plan(
            os.path.realpath(str(nested)), [str(broad), str(specific)],
        )

        assert mount_root == os.path.realpath(str(specific))
        assert container_cwd == "/workspace/sub"

    def test_mount_plan_refuses_non_root_owned_parent(self, tmp_path):
        parent = tmp_path / "untrusted-parent"
        repo = parent / "repo"
        repo.mkdir(parents=True)
        os.chown(parent, policy.SANDBOX_UID, policy.SANDBOX_GID)

        with pytest.raises(runner.CwdRejected, match="not root-owned"):
            runner._mount_plan(os.path.realpath(str(repo)), [str(repo)])

    def test_mount_plan_refuses_non_sticky_world_writable_parent(self, tmp_path):
        parent = tmp_path / "writable-parent"
        repo = parent / "repo"
        repo.mkdir(parents=True)
        os.chmod(parent, 0o777)

        with pytest.raises(runner.CwdRejected, match="group/world-writable"):
            runner._mount_plan(os.path.realpath(str(repo)), [str(repo)])

    def test_repo_content_edits_do_not_invalidate_the_snapshot(self, monkeypatch, repo, nested_cwd):
        """Editing/adding files inside the repo between snapshot and use
        must never trip the chain check — only structural changes to the
        directory chain itself (replacement, symlinking, mode/identity
        changes) may."""
        ran = {"n": 0}

        def _fake_run(cmd, cwd=None, **kwargs):
            (nested_cwd / "new_file.py").write_text("x = 1\n", encoding="utf-8")
            (repo / "other.py").write_text("y = 2\n", encoding="utf-8")
            ran["n"] += 1
            return self._fake_completed()

        monkeypatch.setattr(runner, "_run_subprocess", _fake_run)
        result = runner.spawn_claude(
            task="t", cwd=str(nested_cwd), model="claude-sonnet-5", timeout_seconds=5,
            repo_roots=[str(repo)],
        )
        assert ran["n"] == 1
        assert result["exit_code"] == 0

    def test_cwd_replaced_by_escaping_symlink_during_preflight_refuses(
        self, monkeypatch, repo, nested_cwd, tmp_path,
    ):
        """The swap happens as a side effect of the (mocked) docker
        preflight call — i.e. strictly between the initial snapshot and the
        command-construction revalidation checkpoint. Must never reach
        ``_run_subprocess``."""
        outside = tmp_path / "outside-escape-1"
        outside.mkdir()

        def _swap_and_preflight(*a, **k):
            import shutil as _shutil

            _shutil.rmtree(str(nested_cwd))
            os.symlink(str(outside), str(nested_cwd), target_is_directory=True)
            return {"ok": True, "reason": "", "docker_present": True,
                    "daemon_ok": True, "image_present": True}

        monkeypatch.setattr(runner, "docker_preflight", _swap_and_preflight)

        ran = {"n": 0}
        monkeypatch.setattr(runner, "_run_subprocess", lambda *a, **k: ran.__setitem__("n", ran["n"] + 1))

        with pytest.raises(runner.SpawnRefused):
            runner.spawn_claude(
                task="t", cwd=str(nested_cwd), model="claude-sonnet-5", timeout_seconds=5,
                repo_roots=[str(repo)],
            )
        assert ran["n"] == 0

    def test_cwd_replaced_by_escaping_symlink_during_command_build_refuses(
        self, monkeypatch, repo, nested_cwd, tmp_path,
    ):
        """The swap happens as a side effect of building the docker
        command — strictly between the command-construction revalidation
        checkpoint and the immediately-before-``_run_subprocess`` checkpoint.
        Must never reach ``_run_subprocess``."""
        outside = tmp_path / "outside-escape-2"
        outside.mkdir()

        real_build = runner.build_claude_docker_command

        def _build_then_swap(*a, **k):
            cmd = real_build(*a, **k)
            import shutil as _shutil

            _shutil.rmtree(str(nested_cwd))
            os.symlink(str(outside), str(nested_cwd), target_is_directory=True)
            return cmd

        monkeypatch.setattr(runner, "build_claude_docker_command", _build_then_swap)

        ran = {"n": 0}
        monkeypatch.setattr(runner, "_run_subprocess", lambda *a, **k: ran.__setitem__("n", ran["n"] + 1))

        with pytest.raises(runner.SpawnRefused):
            runner.spawn_claude(
                task="t", cwd=str(nested_cwd), model="claude-sonnet-5", timeout_seconds=5,
                repo_roots=[str(repo)],
            )
        assert ran["n"] == 0

    def test_repo_root_itself_replaced_by_symlink_refuses(self, monkeypatch, repo, nested_cwd, tmp_path):
        """Not just the leaf cwd — any ancestor up to (and including) the
        allowlisted repo root swapped out for a symlink must refuse too."""
        outside = tmp_path / "outside-escape-root"
        outside.mkdir()
        (outside / "sub").mkdir()
        (outside / "sub" / "nested").mkdir()

        def _swap_root_and_preflight(*a, **k):
            import shutil as _shutil

            real_repo = str(repo)
            _shutil.rmtree(real_repo)
            os.symlink(str(outside), real_repo, target_is_directory=True)
            return {"ok": True, "reason": "", "docker_present": True,
                    "daemon_ok": True, "image_present": True}

        monkeypatch.setattr(runner, "docker_preflight", _swap_root_and_preflight)

        ran = {"n": 0}
        monkeypatch.setattr(runner, "_run_subprocess", lambda *a, **k: ran.__setitem__("n", ran["n"] + 1))

        with pytest.raises(runner.SpawnRefused):
            runner.spawn_claude(
                task="t", cwd=str(nested_cwd), model="claude-sonnet-5", timeout_seconds=5,
                repo_roots=[str(repo)],
            )
        assert ran["n"] == 0

    def test_cwd_vanishing_refuses(self, monkeypatch, repo, nested_cwd):
        def _vanish_and_preflight(*a, **k):
            import shutil as _shutil

            _shutil.rmtree(str(nested_cwd))
            return {"ok": True, "reason": "", "docker_present": True,
                    "daemon_ok": True, "image_present": True}

        monkeypatch.setattr(runner, "docker_preflight", _vanish_and_preflight)

        ran = {"n": 0}
        monkeypatch.setattr(runner, "_run_subprocess", lambda *a, **k: ran.__setitem__("n", ran["n"] + 1))

        with pytest.raises(runner.SpawnRefused):
            runner.spawn_claude(
                task="t", cwd=str(nested_cwd), model="claude-sonnet-5", timeout_seconds=5,
                repo_roots=[str(repo)],
            )
        assert ran["n"] == 0

    def test_cwd_mode_changed_refuses(self, monkeypatch, repo, nested_cwd):
        original_mode = os.stat(str(nested_cwd)).st_mode

        def _chmod_and_preflight(*a, **k):
            os.chmod(str(nested_cwd), 0o777)
            return {"ok": True, "reason": "", "docker_present": True,
                    "daemon_ok": True, "image_present": True}

        monkeypatch.setattr(runner, "docker_preflight", _chmod_and_preflight)

        ran = {"n": 0}
        monkeypatch.setattr(runner, "_run_subprocess", lambda *a, **k: ran.__setitem__("n", ran["n"] + 1))

        try:
            with pytest.raises(runner.SpawnRefused):
                runner.spawn_claude(
                    task="t", cwd=str(nested_cwd), model="claude-sonnet-5", timeout_seconds=5,
                    repo_roots=[str(repo)],
                )
            assert ran["n"] == 0
        finally:
            os.chmod(str(nested_cwd), original_mode)

    def test_repo_roots_default_falls_back_to_cwd_itself_for_direct_callers(
        self, monkeypatch, repo,
    ):
        """A caller that pre-validates its own cwd and does not pass
        ``repo_roots`` (every existing direct ``spawn_claude`` call in this
        test module) must keep working exactly as before: the chain/symlink
        protection still applies, but containment is trivially satisfied by
        treating the resolved cwd as its own sole root."""
        captured = {}

        def _fake_run(cmd, cwd=None, **kwargs):
            captured["cwd"] = cwd
            return self._fake_completed()

        monkeypatch.setattr(runner, "_run_subprocess", _fake_run)
        result = runner.spawn_claude(
            task="t", cwd=str(repo), model="claude-sonnet-5", timeout_seconds=5,
        )
        assert result["exit_code"] == 0
        assert captured["cwd"] == os.path.realpath(str(repo))


class TestCredentialStaging:
    """The one host OAuth credentials file (``policy.HOST_CREDENTIALS_PATH``)
    is root-owned, and every sandbox container now always runs as the fixed
    non-root ``policy.SANDBOX_UID`` — which could never read that file if it
    were mounted directly. ``spawn_claude`` must instead copy it, in bounded
    chunks from the already-open trusted descriptor, into a fresh, unique,
    ``policy.SANDBOX_UID``-owned, mode-0400 file under a root-owned 0700
    staging directory; mount ONLY that staged copy; and remove the staged
    file and its private directory on every exit path — success, a nonzero
    exit, a timeout, an unexpected exception, or a preflight/command-build
    failure."""

    @pytest.fixture(autouse=True)
    def _preflight_ok(self, monkeypatch):
        runner.reset_preflight_cache()
        monkeypatch.setattr(
            runner, "docker_preflight",
            lambda *a, **k: {"ok": True, "reason": "", "docker_present": True,
                             "daemon_ok": True, "image_present": True},
        )
        yield
        runner.reset_preflight_cache()

    @pytest.fixture()
    def creds(self, tmp_path, monkeypatch):
        parent = tmp_path / "cred-parent"
        parent.mkdir()
        os.chmod(str(parent), 0o755)
        path = parent / ".credentials.json"
        path.write_text('{"token": "sk-real-secret-value"}', encoding="utf-8")
        path.chmod(0o600)
        monkeypatch.setattr(policy, "HOST_CREDENTIALS_PATH", str(path))
        return path

    @staticmethod
    def _fake_completed():
        class _Completed:
            returncode = 0
            stdout = json.dumps({"result": "done"})
            stderr = ""

        return _Completed()

    @staticmethod
    def _creds_mount_src(cmd):
        mounts = _opt_values(cmd, "--mount")
        matches = [m for m in mounts if policy.CONTAINER_CREDENTIALS_PATH in m]
        assert len(matches) == 1
        return _mount_field(matches[0], "src")

    def test_staged_file_is_mounted_not_the_original(self, monkeypatch, repo, creds):
        captured = {}

        def _fake_run(cmd, **kwargs):
            captured["cmd"] = cmd
            return self._fake_completed()

        monkeypatch.setattr(runner, "_run_subprocess", _fake_run)
        runner.spawn_claude(task="t", cwd=str(repo), model="claude-sonnet-5", timeout_seconds=5)

        src = self._creds_mount_src(captured["cmd"])
        assert src != str(creds)
        assert src != os.path.realpath(str(creds))
        assert str(creds) not in " ".join(captured["cmd"])

    def test_staged_file_is_owned_by_the_sandbox_uid_and_mode_0400(self, monkeypatch, repo, creds):
        captured = {}

        def _fake_run(cmd, **kwargs):
            src = self._creds_mount_src(cmd)
            st = os.stat(src)
            captured["uid"] = st.st_uid
            captured["gid"] = st.st_gid
            captured["mode"] = stat.S_IMODE(st.st_mode)
            return self._fake_completed()

        monkeypatch.setattr(runner, "_run_subprocess", _fake_run)
        runner.spawn_claude(task="t", cwd=str(repo), model="claude-sonnet-5", timeout_seconds=5)

        assert captured["uid"] == policy.SANDBOX_UID
        assert captured["gid"] == policy.SANDBOX_GID
        assert captured["mode"] == 0o400

    def test_staged_file_content_matches_the_original_exactly(self, monkeypatch, repo, creds):
        captured = {}

        def _fake_run(cmd, **kwargs):
            src = self._creds_mount_src(cmd)
            with open(src, "rb") as fh:
                captured["content"] = fh.read()
            return self._fake_completed()

        monkeypatch.setattr(runner, "_run_subprocess", _fake_run)
        runner.spawn_claude(task="t", cwd=str(repo), model="claude-sonnet-5", timeout_seconds=5)

        assert captured["content"] == creds.read_bytes()

    def test_staging_private_dir_is_root_owned_and_mode_0700(self, monkeypatch, repo, creds):
        captured = {}

        def _fake_run(cmd, **kwargs):
            src = self._creds_mount_src(cmd)
            parent = os.path.dirname(src)
            st = os.lstat(parent)
            captured["uid"] = st.st_uid
            captured["mode"] = stat.S_IMODE(st.st_mode)
            return self._fake_completed()

        monkeypatch.setattr(runner, "_run_subprocess", _fake_run)
        runner.spawn_claude(task="t", cwd=str(repo), model="claude-sonnet-5", timeout_seconds=5)

        assert captured["uid"] == 0
        assert captured["mode"] == 0o700

    def test_oversize_credentials_file_is_refused_before_any_docker_run(self, monkeypatch, repo, creds):
        creds.write_bytes(b"x" * (policy.MAX_CREDENTIAL_BYTES + 1))
        ran = {"n": 0}
        monkeypatch.setattr(runner, "_run_subprocess", lambda *a, **k: ran.__setitem__("n", ran["n"] + 1))

        with pytest.raises(runner.SpawnRefused):
            runner.spawn_claude(task="t", cwd=str(repo), model="claude-sonnet-5", timeout_seconds=5)
        assert ran["n"] == 0

    def test_exact_max_size_credentials_file_is_accepted(self, monkeypatch, repo, creds):
        creds.write_bytes(b"x" * policy.MAX_CREDENTIAL_BYTES)
        monkeypatch.setattr(runner, "_run_subprocess", lambda *a, **k: self._fake_completed())

        result = runner.spawn_claude(task="t", cwd=str(repo), model="claude-sonnet-5", timeout_seconds=5)
        assert result["exit_code"] == 0

    def test_staged_file_and_private_dir_are_cleaned_up_after_success(self, monkeypatch, repo, creds):
        captured = {}

        def _fake_run(cmd, **kwargs):
            src = self._creds_mount_src(cmd)
            captured["src"] = src
            captured["dir"] = os.path.dirname(src)
            return self._fake_completed()

        monkeypatch.setattr(runner, "_run_subprocess", _fake_run)
        runner.spawn_claude(task="t", cwd=str(repo), model="claude-sonnet-5", timeout_seconds=5)

        assert not os.path.exists(captured["src"])
        assert not os.path.exists(captured["dir"])

    def test_staged_file_is_cleaned_up_after_nonzero_exit(self, monkeypatch, repo, creds):
        captured = {}

        class _Failed:
            returncode = 1
            stdout = ""
            stderr = "boom"

        def _fake_run(cmd, **kwargs):
            captured["src"] = self._creds_mount_src(cmd)
            return _Failed()

        monkeypatch.setattr(runner, "_run_subprocess", _fake_run)
        runner.spawn_claude(task="t", cwd=str(repo), model="claude-sonnet-5", timeout_seconds=5)

        assert not os.path.exists(captured["src"])

    def test_staged_file_is_cleaned_up_after_timeout(self, monkeypatch, repo, creds):
        captured = {}

        def _fake_run(cmd, **kwargs):
            captured["src"] = self._creds_mount_src(cmd)
            raise runner.subprocess.TimeoutExpired(cmd=["docker"], timeout=1)

        monkeypatch.setattr(runner, "_run_subprocess", _fake_run)
        runner.spawn_claude(task="t", cwd=str(repo), model="claude-sonnet-5", timeout_seconds=1)

        assert not os.path.exists(captured["src"])

    def test_staged_file_is_cleaned_up_after_unexpected_exception(self, monkeypatch, repo, creds):
        captured = {}

        def _fake_run(cmd, **kwargs):
            captured["src"] = self._creds_mount_src(cmd)
            raise OSError("docker vanished mid-run")

        monkeypatch.setattr(runner, "_run_subprocess", _fake_run)
        with pytest.raises(OSError):
            runner.spawn_claude(task="t", cwd=str(repo), model="claude-sonnet-5", timeout_seconds=5)

        assert not os.path.exists(captured["src"])

    def test_staged_file_is_cleaned_up_when_preflight_fails(self, monkeypatch, repo, creds):
        monkeypatch.setattr(
            runner, "docker_preflight",
            lambda *a, **k: {"ok": False, "reason": "nope", "docker_present": True,
                             "daemon_ok": True, "image_present": False},
        )
        with pytest.raises(runner.SpawnRefused):
            runner.spawn_claude(task="t", cwd=str(repo), model="claude-sonnet-5", timeout_seconds=5)

        # Preflight fails before staging is ever attempted.
        assert not os.path.isdir(policy.CREDENTIAL_STAGING_ROOT) or not os.listdir(
            policy.CREDENTIAL_STAGING_ROOT
        )

    def test_staged_file_is_cleaned_up_when_command_build_fails(self, monkeypatch, repo, creds):
        def _boom(*a, **k):
            raise ValueError("bad model")

        monkeypatch.setattr(runner, "build_claude_docker_command", _boom)
        with pytest.raises(ValueError):
            runner.spawn_claude(task="t", cwd=str(repo), model="claude-sonnet-5", timeout_seconds=5)

        assert not os.path.isdir(policy.CREDENTIAL_STAGING_ROOT) or not os.listdir(
            policy.CREDENTIAL_STAGING_ROOT
        )

    def test_staged_symlink_before_run_refuses(self, monkeypatch, repo, creds, tmp_path):
        """If the staged file is swapped for a symlink between staging and
        the subprocess call, the immediately-before-run revalidation must
        catch it and refuse — never mount whatever the symlink resolves to."""
        outside = tmp_path / "outside-secret.json"
        outside.write_text("{}", encoding="utf-8")

        real_build = runner.build_claude_docker_command

        def _build_then_swap(*a, **kwargs):
            cmd = real_build(*a, **kwargs)
            src = self._creds_mount_src(cmd)
            os.remove(src)
            os.symlink(str(outside), src)
            return cmd

        monkeypatch.setattr(runner, "build_claude_docker_command", _build_then_swap)
        ran = {"n": 0}
        monkeypatch.setattr(runner, "_run_subprocess", lambda *a, **k: ran.__setitem__("n", ran["n"] + 1))

        with pytest.raises(runner.SpawnRefused):
            runner.spawn_claude(task="t", cwd=str(repo), model="claude-sonnet-5", timeout_seconds=5)
        assert ran["n"] == 0

    def test_staged_file_replaced_before_run_refuses(self, monkeypatch, repo, creds):
        """A staged file removed and recreated (same path, different inode)
        between staging and the subprocess call must refuse too — an
        identity check, not merely an existence check."""
        real_build = runner.build_claude_docker_command

        def _build_then_replace(*a, **kwargs):
            cmd = real_build(*a, **kwargs)
            src = self._creds_mount_src(cmd)
            mode = stat.S_IMODE(os.lstat(src).st_mode)
            os.remove(src)
            fd = os.open(src, os.O_CREAT | os.O_EXCL | os.O_WRONLY, mode)
            os.write(fd, b"{}")
            os.close(fd)
            return cmd

        monkeypatch.setattr(runner, "build_claude_docker_command", _build_then_replace)
        ran = {"n": 0}
        monkeypatch.setattr(runner, "_run_subprocess", lambda *a, **k: ran.__setitem__("n", ran["n"] + 1))

        with pytest.raises(runner.SpawnRefused):
            runner.spawn_claude(task="t", cwd=str(repo), model="claude-sonnet-5", timeout_seconds=5)
        assert ran["n"] == 0

    def test_staged_file_mode_changed_before_run_refuses(self, monkeypatch, repo, creds):
        real_build = runner.build_claude_docker_command

        def _build_then_chmod(*a, **kwargs):
            cmd = real_build(*a, **kwargs)
            src = self._creds_mount_src(cmd)
            os.chmod(src, 0o644)
            return cmd

        monkeypatch.setattr(runner, "build_claude_docker_command", _build_then_chmod)
        ran = {"n": 0}
        monkeypatch.setattr(runner, "_run_subprocess", lambda *a, **k: ran.__setitem__("n", ran["n"] + 1))

        with pytest.raises(runner.SpawnRefused):
            runner.spawn_claude(task="t", cwd=str(repo), model="claude-sonnet-5", timeout_seconds=5)
        assert ran["n"] == 0

    def test_no_secret_content_is_logged(self, monkeypatch, repo, creds, caplog):
        monkeypatch.setattr(runner, "_run_subprocess", lambda *a, **k: self._fake_completed())
        with caplog.at_level("DEBUG"):
            runner.spawn_claude(task="t", cwd=str(repo), model="claude-sonnet-5", timeout_seconds=5)
        assert "sk-real-secret-value" not in caplog.text

    def test_no_credential_mutation_happens_without_a_spawn(self, repo, creds):
        """Merely importing/loading the runner module or resolving the
        credentials path must never touch the staging root — staging is a
        side effect of an actual ``spawn_claude`` call only."""
        assert not os.path.isdir(policy.CREDENTIAL_STAGING_ROOT)
