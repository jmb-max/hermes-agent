"""RED->GREEN tests for ``runner.run_worker`` orchestration and the
``_git_changed_files`` helper.

Covers requirements (1)/(4)/(6)/(9): the claude_worker tool handler itself —
routing + breaker + isolated spawn + evidence + exactly-one-telemetry-record
per invocation, with the hard "exactly one spawn, no failure-driven retry or
escalation" guarantee proven by call counting on a mocked ``spawn_claude``.
The container command shape itself
(mounts, hardening, preflight) is covered by
``test_claude_worker_isolation.py``; ``spawn_claude`` is mocked here at the
orchestration boundary — it never spawns a host ``claude`` process, only the
locked-down ``docker run`` sandbox.

``run_worker`` also runs an OAuth credential freshness preflight before it
commits to a spawn, and the fixed host credentials path it reads does not
exist under a test home. The autouse fixture below therefore defaults every
test in this module to a healthy credential; ``TestOAuthPreflightIntegration``
at the bottom restores the production implementation and drives it against a
real temporary credentials file.

Every ``repo`` here is a real Git worktree (``make_git_repo``) because scope
is resolved dynamically now: ``run_worker`` validates its cwd through
``project.resolve_project_root_strict``, the same resolver the gate uses, and
refuses anything that is not inside a real worktree. The retired
``gate.repo_roots`` key is still accepted by the loader and is deliberately
still passed in some configs below — to prove it grants nothing.
"""

from __future__ import annotations

import json
import subprocess
import time

import pytest

from tests.plugins._claude_worker_helpers import (
    load_submodule,
    make_git_repo,
    require_safe_tmp,
    stub_fresh_oauth_preflight,
    write_credentials,
)

runner = load_submodule("runner")
breaker = load_submodule("breaker")
telemetry = load_submodule("telemetry")
config = load_submodule("config")
policy = load_submodule("policy")
oauth = load_submodule("oauth")

#: Captured at import time, before any fixture stubs it, so the tests that are
#: ABOUT the preflight can restore the production implementation.
_REAL_OAUTH_PREFLIGHT = oauth.preflight


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    require_safe_tmp(tmp_path)
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    runner.reset_preflight_cache()
    # ``run_worker`` now refuses to spawn on a credential it can already see
    # is stale. The fixed host credentials path does not exist under a test
    # home, so default every test in this module to a healthy credential;
    # ``TestOAuthPreflightIntegration`` below overrides this deliberately.
    stub_fresh_oauth_preflight(monkeypatch)
    yield


def _cfg_with_roots(*roots, **overrides):
    """Build a plugin config carrying the DEPRECATED, inert ``gate.repo_roots``
    key. It is kept in these tests deliberately: every call below must behave
    identically whether or not the requested cwd appears in it."""
    entry = {"gate": {"repo_roots": [str(r) for r in roots]}}
    entry.update(overrides)
    return {"plugins": {"entries": {"claude-worker": entry}}}


def _spawn_result(exit_code=0, stdout="", stderr="", duration_ms=100, timed_out=False, model="claude-sonnet-5"):
    return {
        "exit_code": exit_code, "stdout": stdout, "stderr": stderr,
        "duration_ms": duration_ms, "timed_out": timed_out, "model": model,
    }


class TestSpawnIsInvokedWithoutLegacyHostArgs:
    """``_run_attempts`` must call the docker-backed ``spawn_claude`` with
    exactly its current isolated-sandbox signature — no ``claude_bin``,
    ``allowed_tools``, ``permission_mode``, or ``mcp_config_path``, all of
    which belonged to the retired host-subprocess spawn and are now fixed
    policy literals baked into ``build_claude_docker_command`` instead."""

    def test_run_worker_calls_spawn_claude_with_only_the_sandboxed_kwargs(self, tmp_path, monkeypatch):
        repo = make_git_repo(tmp_path, "repo")
        monkeypatch.setattr(config, "_load_raw_config", lambda: _cfg_with_roots(repo))

        calls = []

        def _fake_spawn(**kwargs):
            calls.append(kwargs)
            return _spawn_result(exit_code=0, stdout=json.dumps({"result": "ok"}), model=kwargs["model"])

        monkeypatch.setattr(runner, "spawn_claude", _fake_spawn)
        monkeypatch.setattr(runner, "_git_changed_files", lambda cwd, repo_roots=None: [])

        runner.run_worker({"task": "fix it", "cwd": str(repo)}, session_id="sess:spawn-shape")

        assert len(calls) == 1
        assert set(calls[0]) == {"task", "cwd", "model", "timeout_seconds"}


class TestGitChangedFiles:
    """``_git_changed_files`` must never shell out to a host ``git`` — it
    delegates entirely to the sandboxed ``run_verification_command`` with a
    fixed, operator-config-independent argv, and parses whatever bounded
    stdout that returns. These tests fake ``run_verification_command`` (the
    sandbox boundary itself — network-isolated docker — is covered by
    ``test_claude_worker_isolation.py``), running a real host ``git`` only
    from INSIDE the fake to produce realistic sandbox-shaped output, never
    from ``_git_changed_files`` itself."""

    def test_uses_fixed_verifier_argv(self, tmp_path, monkeypatch):
        calls = []

        def _fake_verify(repo_root, command, timeout_seconds):
            calls.append((repo_root, command, timeout_seconds))
            return {"ok": True, "stdout": "", "stderr": ""}

        monkeypatch.setattr(runner, "run_verification_command", _fake_verify)

        runner._git_changed_files(str(tmp_path))

        assert len(calls) == 1
        repo_root, command, timeout_seconds = calls[0]
        assert repo_root == str(tmp_path)
        assert command == [
            "/usr/bin/git",
            "-c", "safe.directory=/workspace",
            "-c", "core.hooksPath=/dev/null",
            "-c", "core.fsmonitor=false",
            "status", "--porcelain=v1", "--untracked-files=all",
        ]
        assert timeout_seconds == runner._GIT_STATUS_TIMEOUT_SECONDS

    def test_propagates_configured_roots_to_nested_verifier(self, tmp_path, monkeypatch):
        repo = tmp_path / "repo"
        nested = repo / "sub"
        nested.mkdir(parents=True)
        captured = {}

        def _fake_verify(repo_root, command, timeout_seconds, repo_roots=None):
            captured["repo_root"] = repo_root
            captured["repo_roots"] = repo_roots
            return {"ok": True, "stdout": "", "stderr": ""}

        monkeypatch.setattr(runner, "run_verification_command", _fake_verify)
        runner._git_changed_files(str(nested), repo_roots=[str(repo)])

        assert captured == {
            "repo_root": str(nested),
            "repo_roots": [str(repo)],
        }

    def test_real_git_repo_reports_changed_files(self, tmp_path, monkeypatch):
        repo = make_git_repo(tmp_path, "repo")
        subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
        subprocess.run(["git", "config", "user.email", "t@example.com"], cwd=repo, check=True)
        subprocess.run(["git", "config", "user.name", "t"], cwd=repo, check=True)
        (repo / "a.py").write_text("x = 1\n")
        subprocess.run(["git", "add", "a.py"], cwd=repo, check=True)
        subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=repo, check=True)
        (repo / "a.py").write_text("x = 2\n")
        (repo / "b.py").write_text("new\n")

        def _fake_verify(repo_root, command, timeout_seconds):
            # Simulate what the sandboxed verifier would return — this
            # subprocess call belongs to the TEST DOUBLE, not to
            # ``_git_changed_files`` itself.
            completed = subprocess.run(
                [command[0], "-C", repo_root, *command[5:]],
                capture_output=True, text=True,
            )
            return {"ok": True, "stdout": completed.stdout, "stderr": completed.stderr}

        monkeypatch.setattr(runner, "run_verification_command", _fake_verify)

        files = runner._git_changed_files(str(repo))
        assert "a.py" in files
        assert "b.py" in files

    def test_non_git_dir_returns_empty(self, tmp_path, monkeypatch):
        plain = tmp_path / "plain"
        plain.mkdir()
        monkeypatch.setattr(
            runner, "run_verification_command",
            lambda repo_root, command, timeout_seconds: {
                "ok": False, "stdout": "", "stderr": "not a git repository",
            },
        )
        assert runner._git_changed_files(str(plain)) == []

    def test_verifier_failure_is_best_effort_empty(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            runner, "run_verification_command",
            lambda repo_root, command, timeout_seconds: {"ok": False, "stdout": "junk", "stderr": ""},
        )
        assert runner._git_changed_files(str(tmp_path)) == []

    def test_verifier_exception_is_best_effort_empty(self, tmp_path, monkeypatch):
        def _boom(repo_root, command, timeout_seconds):
            raise RuntimeError("boom")

        monkeypatch.setattr(runner, "run_verification_command", _boom)
        assert runner._git_changed_files(str(tmp_path)) == []

    def test_hostile_ambient_path_fake_git_is_never_invoked(self, tmp_path, monkeypatch):
        """A hostile ``$PATH`` entry ahead of the real ``git`` must never be
        reached: ``_git_changed_files`` never resolves ``git`` off ``$PATH``
        and never calls a host ``subprocess.run`` at all — it only calls the
        sandboxed ``run_verification_command``."""
        hostile_bin = tmp_path / "hostile_bin"
        hostile_bin.mkdir()
        marker = tmp_path / "fake_git_ran"
        fake_git = hostile_bin / "git"
        fake_git.write_text(f"#!/bin/sh\ntouch {marker}\necho pwned\nexit 0\n")
        fake_git.chmod(0o755)
        monkeypatch.setenv("PATH", str(hostile_bin))

        monkeypatch.setattr(
            runner, "run_verification_command",
            lambda repo_root, command, timeout_seconds: {"ok": True, "stdout": "", "stderr": ""},
        )

        spy_calls = []

        def _spy_run(*args, **kwargs):
            spy_calls.append((args, kwargs))
            raise AssertionError("no host subprocess.run may be called by _git_changed_files")

        monkeypatch.setattr(runner.subprocess, "run", _spy_run)

        result = runner._git_changed_files(str(tmp_path))

        assert result == []
        assert not marker.exists()
        assert spy_calls == []


class TestRunWorkerSuccess:
    def test_sonnet_success_returns_full_evidence(self, tmp_path, monkeypatch):
        repo = make_git_repo(tmp_path, "repo")
        monkeypatch.setattr(config, "_load_raw_config", lambda: _cfg_with_roots(repo))

        calls = []

        def _fake_spawn(**kwargs):
            calls.append(kwargs)
            return _spawn_result(
                exit_code=0,
                stdout=json.dumps({"result": "Fixed the typo.", "is_error": False}),
                model=kwargs["model"],
            )

        monkeypatch.setattr(runner, "spawn_claude", _fake_spawn)
        statuses = iter([[], ["a.py"]])
        monkeypatch.setattr(
            runner, "_git_changed_files",
            lambda cwd, repo_roots=None: next(statuses),
        )

        raw = runner.run_worker(
            {"task": "fix a typo", "cwd": str(repo)}, session_id="sess:1",
        )
        result = json.loads(raw)

        assert result["success"] is True
        assert result["model"] == "claude-sonnet-5"
        assert result["route_reason"] == "default"
        assert result["attempts"] == 1
        assert result["escalated"] is False
        assert result["cwd"] == str(repo.resolve())
        assert result["exit_code"] == 0
        assert result["failure_class"] is None
        assert result["breaker"]["open"] is False
        assert result["files_touched"] == ["a.py"]
        assert len(calls) == 1

    def test_preexisting_dirty_file_unchanged_by_worker_is_not_attributed(
        self, tmp_path, monkeypatch,
    ):
        repo = make_git_repo(tmp_path, "repo")
        dirty = repo / "dirty.py"
        dirty.write_text("already dirty\n")
        monkeypatch.setattr(config, "_load_raw_config", lambda: _cfg_with_roots(repo))
        monkeypatch.setattr(
            runner,
            "spawn_claude",
            lambda **kwargs: _spawn_result(
                exit_code=0,
                stdout=json.dumps({"result": "read only", "is_error": False}),
                model=kwargs["model"],
            ),
        )
        status_calls = []

        def _dirty_before_and_after(cwd, repo_roots=None):
            status_calls.append(cwd)
            return ["dirty.py"]

        monkeypatch.setattr(runner, "_git_changed_files", _dirty_before_and_after)

        result = json.loads(runner.run_worker(
            {"task": "read only", "cwd": str(repo)},
            session_id="sess:preexisting-dirty",
        ))

        assert result["files_touched"] == []
        assert status_calls == [str(repo.resolve()), str(repo.resolve())]

    def test_preexisting_dirty_file_modified_by_worker_is_attributed(
        self, tmp_path, monkeypatch,
    ):
        repo = make_git_repo(tmp_path, "repo")
        dirty = repo / "dirty.py"
        dirty.write_text("already dirty\n")
        monkeypatch.setattr(config, "_load_raw_config", lambda: _cfg_with_roots(repo))

        def _fake_spawn(**kwargs):
            dirty.write_text("changed by worker with a different size\n")
            return _spawn_result(
                exit_code=0,
                stdout=json.dumps({"result": "edited", "is_error": False}),
                model=kwargs["model"],
            )

        monkeypatch.setattr(runner, "spawn_claude", _fake_spawn)
        monkeypatch.setattr(
            runner,
            "_git_changed_files",
            lambda cwd, repo_roots=None: ["dirty.py"],
        )

        result = json.loads(runner.run_worker(
            {"task": "edit dirty file", "cwd": str(repo)},
            session_id="sess:changed-dirty",
        ))

        assert result["files_touched"] == ["dirty.py"]


class TestRunWorkerBreakerOpen:
    def test_breaker_open_refuses_without_spawning(self, tmp_path, monkeypatch):
        repo = make_git_repo(tmp_path, "repo")
        monkeypatch.setattr(config, "_load_raw_config", lambda: _cfg_with_roots(repo))
        breaker.record_failure("auth", {"auth": 3600})

        def _must_not_spawn(**kwargs):
            raise AssertionError("spawn_claude must not be called while breaker is open")

        monkeypatch.setattr(runner, "spawn_claude", _must_not_spawn)

        raw = runner.run_worker({"task": "fix it", "cwd": str(repo)}, session_id="sess:2")
        result = json.loads(raw)

        assert result["success"] is False
        assert result["attempts"] == 0
        assert result["breaker"]["open"] is True
        assert "auth" in result["breaker"]["classes"]


class TestRunWorkerEscalation:
    def test_generic_failure_returns_as_is_without_escalation(self, tmp_path, monkeypatch):
        """A plain task failure (``other`` class) is exactly one spawn on
        Sonnet, reported as-is — no second, bigger-model attempt is ever
        made, whether it would have failed again or succeeded."""
        repo = make_git_repo(tmp_path, "repo")
        monkeypatch.setattr(config, "_load_raw_config", lambda: _cfg_with_roots(repo))

        calls = []

        def _fake_spawn(**kwargs):
            calls.append(kwargs)
            return _spawn_result(exit_code=1, stderr="Traceback: something broke", model=kwargs["model"])

        monkeypatch.setattr(runner, "spawn_claude", _fake_spawn)
        monkeypatch.setattr(runner, "_git_changed_files", lambda cwd, repo_roots=None: [])

        raw = runner.run_worker({"task": "hard bug", "cwd": str(repo)}, session_id="sess:3")
        result = json.loads(raw)

        assert len(calls) == 1
        assert calls[0]["model"] == "claude-sonnet-5"
        assert result["success"] is False
        assert result["escalated"] is False
        assert result["attempts"] == 1
        assert result["model"] == "claude-sonnet-5"
        assert result["failure_class"] == "other"

    def test_single_attempt_timeout_is_clamped_to_total_budget_and_never_retries(
        self, tmp_path, monkeypatch,
    ):
        """The single allowed attempt's timeout is still clamped to
        ``policy.MAX_TOTAL_ATTEMPT_SECONDS`` — that budget math survives even
        though there is no longer a second attempt to split it with — and a
        timeout on that lone attempt still never triggers a retry."""
        repo = make_git_repo(tmp_path, "repo")
        monkeypatch.setattr(
            config, "_load_raw_config",
            lambda: _cfg_with_roots(
                repo, isolation={"timeout_seconds": policy.MAX_TOTAL_ATTEMPT_SECONDS + 500},
            ),
        )
        calls = []

        def _fake_spawn(**kwargs):
            calls.append(kwargs)
            return _spawn_result(
                exit_code=1, timed_out=True,
                duration_ms=policy.MAX_TOTAL_ATTEMPT_SECONDS * 1000,
                model=kwargs["model"],
            )

        monkeypatch.setattr(runner, "spawn_claude", _fake_spawn)
        monkeypatch.setattr(runner, "_git_changed_files", lambda cwd, repo_roots=None: [])

        result = json.loads(runner.run_worker(
            {"task": "hard bug", "cwd": str(repo)}, session_id="sess:budget",
        ))

        assert len(calls) == 1
        assert calls[0]["timeout_seconds"] == policy.MAX_TOTAL_ATTEMPT_SECONDS
        assert result["success"] is False
        assert result["attempts"] == 1
        assert result["escalated"] is False

    def test_auth_failure_opens_breaker_and_does_not_escalate(self, tmp_path, monkeypatch):
        repo = make_git_repo(tmp_path, "repo")
        monkeypatch.setattr(config, "_load_raw_config", lambda: _cfg_with_roots(repo))

        calls = []

        def _fake_spawn(**kwargs):
            calls.append(kwargs)
            return _spawn_result(exit_code=1, stderr="Error: Invalid API key", model=kwargs["model"])

        monkeypatch.setattr(runner, "spawn_claude", _fake_spawn)

        raw = runner.run_worker({"task": "fix it", "cwd": str(repo)}, session_id="sess:5")
        result = json.loads(raw)

        assert len(calls) == 1
        assert result["success"] is False
        assert result["failure_class"] == "auth"
        assert result["breaker"]["open"] is True
        assert breaker.is_open() is True

    def test_structured_stdout_oauth_401_opens_breaker_and_does_not_escalate(self, tmp_path, monkeypatch):
        """Claude Code's ``--output-format json`` can exit 1 with empty stderr
        and the real failure in stdout, e.g. an expired OAuth token. That must
        classify as ``auth`` (one attempt, breaker opens, no Opus escalation),
        not ``other`` (which would retry on Opus)."""
        repo = make_git_repo(tmp_path, "repo")
        monkeypatch.setattr(config, "_load_raw_config", lambda: _cfg_with_roots(repo))

        calls = []
        raw_token_marker = "OAuth access token has expired"

        def _fake_spawn(**kwargs):
            calls.append(kwargs)
            stdout = json.dumps({
                "is_error": True,
                "terminal_reason": "api_error",
                "api_error_status": 401,
                "result": f"Failed to authenticate. API Error: 401 {raw_token_marker}...",
            })
            return _spawn_result(exit_code=1, stdout=stdout, stderr="", model=kwargs["model"])

        monkeypatch.setattr(runner, "spawn_claude", _fake_spawn)

        raw = runner.run_worker({"task": "fix it", "cwd": str(repo)}, session_id="sess:6")
        result = json.loads(raw)

        assert len(calls) == 1
        assert result["success"] is False
        assert result["attempts"] == 1
        assert result["escalated"] is False
        assert result["failure_class"] == "auth"
        assert result["breaker"]["open"] is True
        assert "auth" in result["breaker"]["classes"]
        assert breaker.is_open() is True
        assert raw_token_marker not in raw

    def test_malformed_json_stdout_with_generic_failure_returns_as_is_without_escalation(
        self, tmp_path, monkeypatch,
    ):
        """Non-JSON/garbled stdout must never raise and must not be
        misclassified as auth — it behaves exactly like any other
        generic-failure path: exactly one spawn, reported as-is, no
        escalation and no breaker mutation."""
        repo = make_git_repo(tmp_path, "repo")
        monkeypatch.setattr(config, "_load_raw_config", lambda: _cfg_with_roots(repo))

        calls = []

        def _fake_spawn(**kwargs):
            calls.append(kwargs)
            return _spawn_result(exit_code=1, stdout="not json{{{", stderr="", model=kwargs["model"])

        monkeypatch.setattr(runner, "spawn_claude", _fake_spawn)
        monkeypatch.setattr(runner, "_git_changed_files", lambda cwd, repo_roots=None: [])

        raw = runner.run_worker({"task": "fix it", "cwd": str(repo)}, session_id="sess:7")
        result = json.loads(raw)

        assert len(calls) == 1
        assert result["success"] is False
        assert result["escalated"] is False
        assert result["attempts"] == 1
        assert result["failure_class"] == "other"
        assert breaker.is_open() is False


class TestRunWorkerCwdRejection:
    """Scope comes from the request, resolved by ``project.py`` — the SAME
    resolver the gate consults, so the runner can never refuse a repository
    the gate is busy gating (Terra's deadlock finding)."""

    def _refuse(self, monkeypatch, cwd, session_id):
        def _must_not_spawn(**kwargs):
            raise AssertionError("spawn_claude must not be called for a rejected cwd")

        monkeypatch.setattr(runner, "spawn_claude", _must_not_spawn)
        return json.loads(runner.run_worker({"task": "fix it", "cwd": cwd}, session_id=session_id))

    def test_cwd_outside_any_git_worktree_refuses_without_spawning(self, tmp_path, monkeypatch):
        repo = make_git_repo(tmp_path, "repo")
        outside = tmp_path / "outside"
        outside.mkdir()
        monkeypatch.setattr(config, "_load_raw_config", lambda: _cfg_with_roots(repo))

        result = self._refuse(monkeypatch, str(outside), "sess:cwd-outside")
        assert result["success"] is False
        assert result["attempts"] == 0
        assert result["failure_class"] == "cwd_rejected"

    def test_a_configured_repo_root_cannot_authorise_a_non_git_cwd(self, tmp_path, monkeypatch):
        """The deprecated key is inert in BOTH directions: naming a directory
        in ``gate.repo_roots`` does not make it a valid worker cwd."""
        outside = tmp_path / "configured-but-not-a-repo"
        outside.mkdir()
        monkeypatch.setattr(config, "_load_raw_config", lambda: _cfg_with_roots(outside))

        result = self._refuse(monkeypatch, str(outside), "sess:cwd-configured-non-git")
        assert result["failure_class"] == "cwd_rejected"

    def test_an_unconfigured_git_repo_is_accepted(self, tmp_path, monkeypatch):
        """...and inert in the other direction too: a brand-new checkout that
        appears in no config at all is a perfectly valid worker cwd."""
        fresh = make_git_repo(tmp_path, "brand-new")
        monkeypatch.setattr(config, "_load_raw_config", lambda: _cfg_with_roots())
        monkeypatch.setattr(
            runner, "spawn_claude",
            lambda **kw: _spawn_result(exit_code=0, stdout=json.dumps({"result": "ok"}), model=kw["model"]),
        )
        monkeypatch.setattr(runner, "_git_changed_files", lambda cwd, repo_roots=None: [])

        result = json.loads(runner.run_worker(
            {"task": "fix it", "cwd": str(fresh)}, session_id="sess:cwd-unconfigured",
        ))
        assert result["success"] is True
        assert result["cwd"] == str(fresh.resolve())

    def test_a_relative_cwd_refuses_without_spawning(self, tmp_path, monkeypatch):
        make_git_repo(tmp_path, "repo")
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(config, "_load_raw_config", lambda: {})

        result = self._refuse(monkeypatch, "repo", "sess:cwd-relative")
        assert result["failure_class"] == "cwd_rejected"

    def test_a_nested_cwd_resolves_to_its_repo_root(self, tmp_path, monkeypatch):
        repo = make_git_repo(tmp_path, "repo")
        nested = repo / "src"
        nested.mkdir()
        monkeypatch.setattr(config, "_load_raw_config", lambda: {})
        monkeypatch.setattr(
            runner, "spawn_claude",
            lambda **kw: _spawn_result(exit_code=0, stdout=json.dumps({"result": "ok"}), model=kw["model"]),
        )
        monkeypatch.setattr(runner, "_git_changed_files", lambda cwd, repo_roots=None: [])

        result = json.loads(runner.run_worker(
            {"task": "fix it", "cwd": str(nested)}, session_id="sess:cwd-nested",
        ))
        assert result["success"] is True
        # The worker still RUNS in the nested directory; what gets mounted is
        # the repository root (asserted in test_claude_worker_isolation.py).
        assert result["cwd"] == str(nested.resolve())

    def test_a_symlink_escaping_the_repo_is_judged_where_it_lands(self, tmp_path, monkeypatch):
        repo = make_git_repo(tmp_path, "repo")
        outside = tmp_path / "outside"
        outside.mkdir()
        (repo / "escape").symlink_to(outside, target_is_directory=True)
        monkeypatch.setattr(config, "_load_raw_config", lambda: _cfg_with_roots(repo))

        result = self._refuse(monkeypatch, str(repo / "escape"), "sess:cwd-symlink-escape")
        assert result["failure_class"] == "cwd_rejected"

    def test_run_worker_never_asks_policy_for_configured_roots(self):
        """The tripwire: ``run_worker`` used to call a policy helper that
        turned ``gate.repo_roots`` into the worker's authority."""
        source = open(runner.__file__, encoding="utf-8").read()
        assert "canonical_repo_roots" not in source
        assert "resolve_within_roots" not in source


class TestTelemetryIntegration:
    def test_exactly_one_telemetry_record_per_invocation(self, tmp_path, monkeypatch):
        repo = make_git_repo(tmp_path, "repo")
        monkeypatch.setattr(config, "_load_raw_config", lambda: _cfg_with_roots(repo))

        calls = []

        def _fake_spawn(**kwargs):
            calls.append(kwargs)
            return _spawn_result(exit_code=1, stderr="Traceback: broke", model=kwargs["model"])

        monkeypatch.setattr(runner, "spawn_claude", _fake_spawn)

        telemetry_calls = []
        monkeypatch.setattr(
            telemetry, "append_record",
            lambda record, configured_path="": telemetry_calls.append(record),
        )

        runner.run_worker({"task": "hard bug", "cwd": str(repo)}, session_id="sess:7")

        assert len(telemetry_calls) == 1
        assert telemetry_calls[0]["session_id"] == "sess:7"

    def test_telemetry_emitted_even_when_breaker_open(self, tmp_path, monkeypatch):
        repo = make_git_repo(tmp_path, "repo")
        monkeypatch.setattr(config, "_load_raw_config", lambda: _cfg_with_roots(repo))
        breaker.record_failure("rate", {"rate": 900})

        telemetry_calls = []
        monkeypatch.setattr(
            telemetry, "append_record",
            lambda record, configured_path="": telemetry_calls.append(record),
        )

        runner.run_worker({"task": "fix it", "cwd": str(repo)}, session_id="sess:8")
        assert len(telemetry_calls) == 1


class TestReviewIntegration:
    def test_review_runs_for_substantial_success(self, tmp_path, monkeypatch):
        repo = make_git_repo(tmp_path, "repo")
        monkeypatch.setattr(
            config, "_load_raw_config",
            lambda: _cfg_with_roots(repo, review={"enabled": True, "min_changed_files": 2}),
        )
        monkeypatch.setattr(
            runner, "spawn_claude",
            lambda **kw: _spawn_result(exit_code=0, stdout=json.dumps({"result": "done"}), model=kw["model"]),
        )
        statuses = iter([[], ["a.py", "b.py", "c.py"]])
        monkeypatch.setattr(
            runner, "_git_changed_files",
            lambda cwd, repo_roots=None: next(statuses),
        )

        review_calls = []
        monkeypatch.setattr(
            runner, "run_review",
            lambda **kw: review_calls.append(kw) or {"reviewed": True, "notes": "looks fine"},
        )

        raw = runner.run_worker({"task": "add feature", "cwd": str(repo)}, session_id="sess:9")
        result = json.loads(raw)

        assert len(review_calls) == 1
        assert result["review"]["reviewed"] is True

    def test_review_skipped_below_threshold(self, tmp_path, monkeypatch):
        repo = make_git_repo(tmp_path, "repo")
        monkeypatch.setattr(
            config, "_load_raw_config",
            lambda: _cfg_with_roots(repo, review={"enabled": True, "min_changed_files": 5}),
        )
        monkeypatch.setattr(
            runner, "spawn_claude",
            lambda **kw: _spawn_result(exit_code=0, stdout=json.dumps({"result": "done"}), model=kw["model"]),
        )
        statuses = iter([[], ["a.py"]])
        monkeypatch.setattr(
            runner, "_git_changed_files",
            lambda cwd, repo_roots=None: next(statuses),
        )

        review_calls = []
        monkeypatch.setattr(runner, "run_review", lambda **kw: review_calls.append(kw) or {})

        raw = runner.run_worker({"task": "tiny fix", "cwd": str(repo)}, session_id="sess:10")
        result = json.loads(raw)

        assert len(review_calls) == 0
        assert result["review"] is None

    def test_review_failure_does_not_fail_worker_result(self, tmp_path, monkeypatch):
        repo = make_git_repo(tmp_path, "repo")
        monkeypatch.setattr(
            config, "_load_raw_config",
            lambda: _cfg_with_roots(repo, review={"enabled": True, "min_changed_files": 1}),
        )
        monkeypatch.setattr(
            runner, "spawn_claude",
            lambda **kw: _spawn_result(exit_code=0, stdout=json.dumps({"result": "done"}), model=kw["model"]),
        )
        statuses = iter([[], ["a.py"]])
        monkeypatch.setattr(
            runner, "_git_changed_files",
            lambda cwd, repo_roots=None: next(statuses),
        )
        monkeypatch.setattr(
            runner, "run_review", lambda **kw: {"reviewed": False, "error": "aux backend down"},
        )

        raw = runner.run_worker({"task": "fix", "cwd": str(repo)}, session_id="sess:11")
        result = json.loads(raw)

        assert result["success"] is True
        assert result["review"]["reviewed"] is False


class TestInputValidation:
    def test_missing_task_is_rejected_without_spawning(self, tmp_path, monkeypatch):
        repo = make_git_repo(tmp_path, "repo")
        monkeypatch.setattr(config, "_load_raw_config", lambda: _cfg_with_roots(repo))

        def _must_not_spawn(**kwargs):
            raise AssertionError("must not spawn without a task")

        monkeypatch.setattr(runner, "spawn_claude", _must_not_spawn)

        raw = runner.run_worker({"cwd": str(repo)}, session_id="sess:12")
        result = json.loads(raw)
        assert result["success"] is False

    def test_missing_cwd_is_rejected_without_spawning(self, monkeypatch):
        def _must_not_spawn(**kwargs):
            raise AssertionError("must not spawn without a cwd")

        monkeypatch.setattr(runner, "spawn_claude", _must_not_spawn)


class TestFallbackConsentStrictness:
    """``allow_terra_fallback`` must be the literal ``True`` — every other
    JSON-representable value (including other truthy ones) leaves the
    fallback un-invoked and the session on HOLD."""

    @pytest.mark.parametrize(
        "bad_value",
        [False, "false", "true", 0, 1, [], {}, None],
        ids=["False", "str-false", "str-true", "int-0", "int-1", "empty-list", "empty-dict", "None"],
    )
    def test_non_true_allow_terra_fallback_never_invokes_fallback(self, tmp_path, monkeypatch, bad_value):
        repo = make_git_repo(tmp_path, "repo")
        monkeypatch.setattr(config, "_load_raw_config", lambda: _cfg_with_roots(repo))
        breaker.record_failure("auth", {"auth": 3600})

        fallback_calls = []
        monkeypatch.setattr(
            runner, "run_fallback",
            lambda **kw: fallback_calls.append(kw) or {"ok": True, "notes": "guidance"},
        )

        raw = runner.run_worker(
            {"task": "fix it", "cwd": str(repo), "allow_terra_fallback": bad_value},
            session_id="sess:fallback-strict",
        )
        result = json.loads(raw)

        assert fallback_calls == []
        assert result["fallback_ready"] is False
        assert result["status"] == "HOLD"

    def test_missing_allow_terra_fallback_key_never_invokes_fallback(self, tmp_path, monkeypatch):
        repo = make_git_repo(tmp_path, "repo")
        monkeypatch.setattr(config, "_load_raw_config", lambda: _cfg_with_roots(repo))
        breaker.record_failure("auth", {"auth": 3600})

        fallback_calls = []
        monkeypatch.setattr(
            runner, "run_fallback",
            lambda **kw: fallback_calls.append(kw) or {"ok": True, "notes": "guidance"},
        )

        raw = runner.run_worker({"task": "fix it", "cwd": str(repo)}, session_id="sess:fallback-missing")
        result = json.loads(raw)

        assert fallback_calls == []
        assert result["fallback_ready"] is False
        assert result["status"] == "HOLD"

    def test_literal_true_invokes_fallback_and_marks_ready_with_provenance(self, tmp_path, monkeypatch):
        repo = make_git_repo(tmp_path, "repo")
        monkeypatch.setattr(config, "_load_raw_config", lambda: _cfg_with_roots(repo))
        breaker.record_failure("auth", {"auth": 3600})

        fallback_calls = []
        monkeypatch.setattr(
            runner, "run_fallback",
            lambda **kw: fallback_calls.append(kw) or {"ok": True, "notes": "Terra guidance"},
        )

        raw = runner.run_worker(
            {"task": "fix it", "cwd": str(repo), "allow_terra_fallback": True},
            session_id="sess:fallback-true",
        )
        result = json.loads(raw)

        assert len(fallback_calls) == 1
        assert result["status"] == "fallback_ready"
        assert result["fallback_ready"] is True
        assert result["fallback_requested"] is True
        assert result["fallback_provenance"] == "terra_auxiliary"
        assert result["fallback"]["ok"] is True
        assert result["fallback"]["ready"] is True
        assert result["fallback"]["notes"] == "Terra guidance"

    def test_failed_terra_fallback_still_holds_without_fallback_ready(self, tmp_path, monkeypatch):
        repo = make_git_repo(tmp_path, "repo")
        monkeypatch.setattr(config, "_load_raw_config", lambda: _cfg_with_roots(repo))
        breaker.record_failure("auth", {"auth": 3600})

        monkeypatch.setattr(
            runner, "run_fallback", lambda **kw: {"ok": False, "error": "terra unavailable"},
        )

        raw = runner.run_worker(
            {"task": "fix it", "cwd": str(repo), "allow_terra_fallback": True},
            session_id="sess:fallback-terra-down",
        )
        result = json.loads(raw)

        assert result["status"] == "HOLD"
        assert result["fallback_ready"] is False


class TestSessionEnvLookupSafety:
    """Every ``get_session_env`` read at the top of ``run_worker`` is
    potentially-raising (a gateway session backend, a monkeypatched hook, or
    a bug in the session-context layer). Each lookup's failure must be
    independently absorbed: one redacted ``internal_error`` result and
    exactly one telemetry append, never a leaked exception, never a crash
    out of tool dispatch."""

    def test_session_key_lookup_failure_yields_one_redacted_internal_error(self, tmp_path, monkeypatch):
        repo = make_git_repo(tmp_path, "repo")
        monkeypatch.setattr(config, "_load_raw_config", lambda: _cfg_with_roots(repo))

        def _boom(name, default=""):
            if name == "HERMES_SESSION_KEY":
                raise RuntimeError("session backend exploded: /secret/internal/path")
            return default

        monkeypatch.setattr(runner, "get_session_env", _boom)

        def _must_not_spawn(**kwargs):
            raise AssertionError("must not spawn when the session key lookup fails")

        monkeypatch.setattr(runner, "spawn_claude", _must_not_spawn)

        telemetry_calls = []
        monkeypatch.setattr(
            telemetry, "append_record",
            lambda record, configured_path="": telemetry_calls.append(record),
        )

        raw = runner.run_worker({"task": "fix it", "cwd": str(repo)}, session_id="sess:lookup-fail-key")
        result = json.loads(raw)

        assert result["success"] is False
        assert result["failure_class"] == "internal_error"
        assert "/secret/internal/path" not in raw
        assert len(telemetry_calls) == 1

    def test_chat_id_lookup_failure_yields_one_redacted_internal_error(self, tmp_path, monkeypatch):
        repo = make_git_repo(tmp_path, "repo")
        monkeypatch.setattr(config, "_load_raw_config", lambda: _cfg_with_roots(repo))

        def _boom(name, default=""):
            if name == "HERMES_SESSION_CHAT_ID":
                raise RuntimeError("chat id backend exploded: /secret/other/path")
            if name == "HERMES_SESSION_KEY":
                return "sess:real-key"
            return default

        monkeypatch.setattr(runner, "get_session_env", _boom)

        def _must_not_spawn(**kwargs):
            raise AssertionError("must not spawn when the chat id lookup fails")

        monkeypatch.setattr(runner, "spawn_claude", _must_not_spawn)

        telemetry_calls = []
        monkeypatch.setattr(
            telemetry, "append_record",
            lambda record, configured_path="": telemetry_calls.append(record),
        )

        raw = runner.run_worker({"task": "fix it", "cwd": str(repo)}, session_id="sess:lookup-fail-chat")
        result = json.loads(raw)

        assert result["success"] is False
        assert result["failure_class"] == "internal_error"
        assert "/secret/other/path" not in raw
        assert len(telemetry_calls) == 1
        # session_key was already resolved before chat_id's lookup raised —
        # _finish must still use it (local defaults exist only for fields
        # not yet resolved at the point of failure).
        assert telemetry_calls[0]["session_id"] == "sess:real-key"


class TestUnexpectedSpawnException:
    """A raw, unclassified exception out of the docker layer (docker daemon
    dying mid-call, an OS-level failure) is not a ``SpawnRefused`` and must
    not crash tool dispatch or silently drop telemetry: ``run_worker``'s
    outer handler is the single place that converts anything unexpected
    into one redacted failure result and exactly one telemetry record."""

    def test_run_worker_degrades_to_one_internal_error_result(self, tmp_path, monkeypatch):
        repo = make_git_repo(tmp_path, "repo")
        monkeypatch.setattr(config, "_load_raw_config", lambda: _cfg_with_roots(repo))

        def _boom(**kwargs):
            raise OSError("docker vanished mid-run")

        monkeypatch.setattr(runner, "spawn_claude", _boom)

        telemetry_calls = []
        monkeypatch.setattr(
            telemetry, "append_record",
            lambda record, configured_path="": telemetry_calls.append(record),
        )

        raw = runner.run_worker({"task": "fix it", "cwd": str(repo)}, session_id="sess:boom")
        result = json.loads(raw)

        assert result["success"] is False
        assert result["failure_class"] == "internal_error"
        assert "docker vanished mid-run" not in raw
        assert len(telemetry_calls) == 1

        raw = runner.run_worker({"task": "fix it"}, session_id="sess:13")
        result = json.loads(raw)
        assert result["success"] is False


class TestOAuthPreflightIntegration:
    """``run_worker`` must check OAuth credential freshness BEFORE it commits
    to a spawn — that is the whole point of ``oauth.py``: a credential we can
    already see is stale must never become an observed 401 that slams the
    ``auth`` breaker shut for an hour while every session sits on HOLD.

    These tests deliberately let the REAL ``oauth.preflight`` run, against a
    temporary credentials file that ``policy.HOST_CREDENTIALS_PATH`` is
    pointed at, so they prove the wiring rather than a stub's return value.
    ``spawn_claude`` is still mocked at the orchestration boundary, so the
    fixed host path is never actually mounted or read by the docker layer.
    """

    # ``_isolated`` is requested explicitly, not merely relied on: it is the
    # fixture that installs the module-wide fresh-preflight stub, and this
    # one has to undo it, so the ordering must be a dependency rather than an
    # assumption about how pytest sequences two autouse fixtures.
    @pytest.fixture(autouse=True)
    def _real_preflight(self, _isolated, tmp_path, monkeypatch):
        """Undo the module-wide fresh-preflight stub and point the fixed host
        credentials path at a temporary file, so every case below exercises
        the production implementation.

        The clock is deliberately NOT patched — ``time.time`` is a globally
        shared module attribute that ``breaker`` reads too. Every expiry
        below is instead written far from ``oauth.FRESHNESS_MARGIN_SECONDS``
        (an hour ahead, or ten seconds behind), so the outcome does not
        depend on how long the test itself takes. The margin boundary itself
        is pinned exactly, with an explicit ``now``, in
        ``test_claude_worker_oauth.py``.
        """
        monkeypatch.setattr(oauth, "preflight", _REAL_OAUTH_PREFLIGHT)
        monkeypatch.setattr(oauth, "REFRESH_PROBE", None)
        self.now = time.time()
        self.creds = tmp_path / "creds" / ".credentials.json"
        monkeypatch.setattr(policy, "HOST_CREDENTIALS_PATH", str(self.creds))

    def _repo(self, tmp_path, monkeypatch):
        repo = make_git_repo(tmp_path, "repo")
        monkeypatch.setattr(config, "_load_raw_config", lambda: _cfg_with_roots(repo))
        return repo

    def _no_spawn(self, monkeypatch):
        def _must_not_spawn(**kwargs):
            raise AssertionError("spawn_claude must not be called after a preflight HOLD")

        def _must_not_attempt(**kwargs):
            raise AssertionError("_run_attempts must not be called after a preflight HOLD")

        monkeypatch.setattr(runner, "spawn_claude", _must_not_spawn)
        monkeypatch.setattr(runner, "_run_attempts", _must_not_attempt)

    def _spawns(self, monkeypatch):
        calls = []

        def _fake_spawn(**kwargs):
            calls.append(kwargs)
            return _spawn_result(
                exit_code=0, stdout=json.dumps({"result": "done"}), model=kwargs["model"],
            )

        monkeypatch.setattr(runner, "spawn_claude", _fake_spawn)
        monkeypatch.setattr(runner, "_git_changed_files", lambda cwd, repo_roots=None: [])
        return calls

    # -- fresh credential proceeds -----------------------------------------

    def test_fresh_credential_proceeds_to_exactly_one_attempt(self, tmp_path, monkeypatch):
        repo = self._repo(tmp_path, monkeypatch)
        write_credentials(self.creds, expires_at=(self.now + 3600) * 1000)
        calls = self._spawns(monkeypatch)

        result = json.loads(runner.run_worker(
            {"task": "fix it", "cwd": str(repo)}, session_id="sess:oauth-fresh",
        ))

        assert len(calls) == 1
        assert result["success"] is True
        assert result["attempts"] == 1
        assert "oauth_state" not in result  # nothing extra on the happy path

    # -- failed preflight HOLDs, spawns nothing, touches no breaker --------

    def test_missing_credential_holds_without_spawning_or_touching_breaker(
        self, tmp_path, monkeypatch,
    ):
        repo = self._repo(tmp_path, monkeypatch)
        self._no_spawn(monkeypatch)

        recorded = []
        monkeypatch.setattr(
            breaker, "record_failure",
            lambda *a, **k: recorded.append((a, k)),
        )

        assert breaker.open_classes() == []
        result = json.loads(runner.run_worker(
            {"task": "fix it", "cwd": str(repo)}, session_id="sess:oauth-missing",
        ))

        assert result["status"] == "HOLD"
        assert result["success"] is False
        assert result["attempts"] == 0
        assert result["escalated"] is False
        assert result["failure_class"] == runner.OAUTH_PREFLIGHT_FAILURE_CLASS
        assert result["oauth_state"] == oauth.STATE_MISSING
        assert result["oauth_refresh_attempted"] is False
        # No breaker mutation of any kind: not recorded, not opened, not reset.
        assert recorded == []
        assert result["breaker"] == {"open": False, "classes": []}
        assert breaker.open_classes() == []

    def test_preflight_failure_class_is_never_a_breaker_class(self):
        """Structural, not incidental: the preflight's class sits outside
        ``BREAKER_CLASSES``, so no present or future "record the failure
        class" path can convert a HOLD into an hour of cooldown."""
        assert runner.OAUTH_PREFLIGHT_FAILURE_CLASS not in breaker.BREAKER_CLASSES

    def test_preflight_hold_leaves_an_already_open_breaker_untouched(
        self, tmp_path, monkeypatch,
    ):
        """A breaker opened by a real observed failure keeps its own cooldown.
        The open-breaker HOLD is reported first, and the preflight never
        clears or resets anything on the way past."""
        repo = self._repo(tmp_path, monkeypatch)
        self._no_spawn(monkeypatch)
        breaker.record_failure("auth", {"auth": 3600})

        result = json.loads(runner.run_worker(
            {"task": "fix it", "cwd": str(repo)}, session_id="sess:oauth-breaker-open",
        ))

        assert result["failure_class"] == "breaker_open"
        assert result["breaker"]["open"] is True
        assert breaker.open_classes() == ["auth"]

    def test_expired_without_refresh_token_holds(self, tmp_path, monkeypatch):
        repo = self._repo(tmp_path, monkeypatch)
        write_credentials(
            self.creds, refresh_token=None, expires_at=(self.now - 10) * 1000,
        )
        self._no_spawn(monkeypatch)

        result = json.loads(runner.run_worker(
            {"task": "fix it", "cwd": str(repo)}, session_id="sess:oauth-expired",
        ))

        assert result["status"] == "HOLD"
        assert result["oauth_state"] == oauth.STATE_EXPIRED
        assert result["oauth_refresh_attempted"] is False

    def test_refreshable_expired_without_a_probe_holds(self, tmp_path, monkeypatch):
        """Production installs no probe, so a refreshable-expired credential
        HOLDs rather than hand-rolling a privileged refresh."""
        repo = self._repo(tmp_path, monkeypatch)
        write_credentials(self.creds, expires_at=(self.now - 10) * 1000)
        self._no_spawn(monkeypatch)

        result = json.loads(runner.run_worker(
            {"task": "fix it", "cwd": str(repo)}, session_id="sess:oauth-refreshable",
        ))

        assert result["status"] == "HOLD"
        assert result["oauth_state"] == oauth.STATE_REFRESHABLE_EXPIRED
        assert result["oauth_refresh_attempted"] is False

    # -- a probe that actually restores the credential lets ONE attempt run -

    def test_successful_probe_permits_exactly_one_attempt(self, tmp_path, monkeypatch):
        repo = self._repo(tmp_path, monkeypatch)
        write_credentials(self.creds, expires_at=(self.now - 10) * 1000)
        calls = self._spawns(monkeypatch)

        probe_calls = []

        def _probe():
            probe_calls.append(True)
            write_credentials(self.creds, expires_at=(self.now + 3600) * 1000)
            return {"ok": True}

        monkeypatch.setattr(oauth, "REFRESH_PROBE", _probe)

        result = json.loads(runner.run_worker(
            {"task": "fix it", "cwd": str(repo)}, session_id="sess:oauth-probe-ok",
        ))

        assert len(probe_calls) == 1
        assert len(calls) == 1
        assert result["success"] is True
        assert result["attempts"] == 1

    def test_failing_probe_holds_after_exactly_one_attempt_at_refresh(
        self, tmp_path, monkeypatch,
    ):
        repo = self._repo(tmp_path, monkeypatch)
        write_credentials(self.creds, expires_at=(self.now - 10) * 1000)
        self._no_spawn(monkeypatch)

        probe_calls = []
        monkeypatch.setattr(
            oauth, "REFRESH_PROBE",
            lambda: probe_calls.append(True) or {"ok": False, "reason": "refresh rejected"},
        )

        result = json.loads(runner.run_worker(
            {"task": "fix it", "cwd": str(repo)}, session_id="sess:oauth-probe-fail",
        ))

        assert len(probe_calls) == 1
        assert result["status"] == "HOLD"
        assert result["oauth_refresh_attempted"] is True
        assert result["oauth_state"] == oauth.STATE_REFRESHABLE_EXPIRED

    # -- ordering ----------------------------------------------------------

    def test_cwd_is_validated_before_the_preflight_runs(self, tmp_path, monkeypatch):
        """An invalid cwd is still ``cwd_rejected``, not an OAuth HOLD — the
        preflight sits after cwd validation, never in front of it."""
        self._repo(tmp_path, monkeypatch)
        outside = tmp_path / "outside"
        outside.mkdir()
        self._no_spawn(monkeypatch)

        probe_calls = []
        monkeypatch.setattr(oauth, "preflight", lambda *a, **k: probe_calls.append(True))

        result = json.loads(runner.run_worker(
            {"task": "fix it", "cwd": str(outside)}, session_id="sess:oauth-after-cwd",
        ))

        assert result["failure_class"] == "cwd_rejected"
        assert probe_calls == []

    def test_preflight_runs_before_any_baseline_evidence_gathering(
        self, tmp_path, monkeypatch,
    ):
        """No git-status container is started for a run that can never spawn."""
        repo = self._repo(tmp_path, monkeypatch)
        self._no_spawn(monkeypatch)

        def _must_not_gather(cwd, repo_roots=None):
            raise AssertionError("baseline evidence must not be gathered before a spawn")

        monkeypatch.setattr(runner, "_git_changed_files", _must_not_gather)

        result = json.loads(runner.run_worker(
            {"task": "fix it", "cwd": str(repo)}, session_id="sess:oauth-before-evidence",
        ))
        assert result["status"] == "HOLD"

    # -- result hygiene ----------------------------------------------------

    def test_hold_payload_carries_no_token_or_raw_credential_json(
        self, tmp_path, monkeypatch,
    ):
        repo = self._repo(tmp_path, monkeypatch)
        access = "sk-ant-" + "oat01-" + "DEADBEEFACCESSTOKEN"
        refresh = "sk-ant-" + "ort01-" + "DEADBEEFREFRESHTOKEN"
        write_credentials(
            self.creds, access_token=access, refresh_token=refresh,
            expires_at=(self.now - 10) * 1000,
        )
        self._no_spawn(monkeypatch)
        monkeypatch.setattr(
            oauth, "REFRESH_PROBE",
            lambda: {"ok": False, "reason": f"server rejected {refresh}"},
        )

        raw = runner.run_worker(
            {"task": "fix it", "cwd": str(repo)}, session_id="sess:oauth-redaction",
        )

        assert access not in raw
        assert refresh not in raw
        assert "accessToken" not in raw
        assert "refreshToken" not in raw
        # ...and the free-text probe reason never rides out on the result at
        # all: the operator text is a canned literal keyed by state.
        assert "server rejected" not in raw

    def test_hold_payload_cannot_unlock_the_gate(self, tmp_path, monkeypatch):
        """Every field ``gate._fallback_delivered`` requires is pinned to the
        locked value, so a preflight HOLD can never release direct edits."""
        repo = self._repo(tmp_path, monkeypatch)
        self._no_spawn(monkeypatch)

        result = json.loads(runner.run_worker(
            {"task": "fix it", "cwd": str(repo), "allow_terra_fallback": True},
            session_id="sess:oauth-no-unlock",
        ))

        assert result["success"] is False
        assert result["fallback_ready"] is False
        assert result["fallback_requested"] is False
        assert result["fallback_provenance"] is None
        assert result["fallback"] is None

    # -- telemetry ---------------------------------------------------------

    def test_preflight_hold_emits_exactly_one_telemetry_record(
        self, tmp_path, monkeypatch,
    ):
        repo = self._repo(tmp_path, monkeypatch)
        self._no_spawn(monkeypatch)

        records = []
        monkeypatch.setattr(
            telemetry, "append_record",
            lambda record, configured_path="": records.append(record),
        )

        runner.run_worker({"task": "fix it", "cwd": str(repo)}, session_id="sess:oauth-telemetry")

        assert len(records) == 1
        record = records[0]
        assert record["attempt"] == 0
        assert record["success"] is False
        assert record["failure_class"] == runner.OAUTH_PREFLIGHT_FAILURE_CLASS
        assert record["breaker_state"] == "closed"

    def test_a_preflight_that_raises_fails_closed_with_one_record(
        self, tmp_path, monkeypatch,
    ):
        """``oauth.preflight`` is documented never to raise, but if it ever
        did the orchestrator must still degrade closed: no spawn, one
        redacted result, one telemetry record."""
        repo = self._repo(tmp_path, monkeypatch)
        self._no_spawn(monkeypatch)

        def _boom(*args, **kwargs):
            raise RuntimeError("credential check exploded: /root/.claude/.credentials.json")

        monkeypatch.setattr(oauth, "preflight", _boom)

        records = []
        monkeypatch.setattr(
            telemetry, "append_record",
            lambda record, configured_path="": records.append(record),
        )

        raw = runner.run_worker({"task": "fix it", "cwd": str(repo)}, session_id="sess:oauth-raises")
        result = json.loads(raw)

        assert result["success"] is False
        assert result["failure_class"] == "internal_error"
        assert "/root/.claude/.credentials.json" not in raw
        assert len(records) == 1


# ---------------------------------------------------------------------------
# Bounded, redacted failure diagnostics — the 2026-09-07/08 incident: a run
# of "other" failures at 03:50-04:41 UTC left telemetry with no stderr and
# the agent-visible tool result truncated before anything actionable, so
# root cause was unrecoverable. These cover the pure helpers, the
# ``_run_attempts`` wiring, and the end-to-end ``run_worker`` -> telemetry
# path, including that the diagnostic never leaks into the caller-facing
# JSON result and is never fabricated on a success/preflight/refusal path.
# ---------------------------------------------------------------------------


class TestBoundedFailureDiagnosticHelpers:
    """Direct unit coverage of ``runner._bounded_diagnostic`` and its two
    callers. Pure functions — no spawn, no orchestration."""

    def test_prefers_stderr_over_stdout_when_both_are_present(self):
        diag = runner._diagnostics_from_spawn("stdout body", "the real error")
        assert diag["error_source"] == "stderr"
        assert diag["error_excerpt"] == "the real error"

    def test_falls_back_to_stdout_json_result_when_stderr_is_empty(self):
        stdout = json.dumps({"is_error": True, "result": "task failed: bad input"})
        diag = runner._diagnostics_from_spawn(stdout, "")
        assert diag["error_source"] == "stdout_json_result"
        assert diag["error_excerpt"] == "task failed: bad input"

    def test_falls_back_to_raw_stdout_when_it_is_not_json(self):
        diag = runner._diagnostics_from_spawn("not json at all", "")
        assert diag["error_source"] == "stdout"
        assert diag["error_excerpt"] == "not json at all"

    def test_falls_back_to_raw_stdout_when_json_has_no_result_field(self):
        stdout = json.dumps({"is_error": True, "exit_code": 2})
        diag = runner._diagnostics_from_spawn(stdout, "")
        assert diag["error_source"] == "stdout"
        assert diag["error_excerpt"] == stdout

    def test_empty_stdout_and_stderr_is_an_empty_but_well_shaped_diagnostic(self):
        diag = runner._diagnostics_from_spawn("", "")
        assert diag == {
            "error_source": "", "error_excerpt": "", "error_fingerprint": "",
            "api_error_status": None,
        }

    @pytest.mark.parametrize(
        "status_field", ["api_error_status", "status", "status_code", "code"],
    )
    def test_numeric_status_is_captured_from_every_known_field_name(self, status_field):
        stdout = json.dumps({"is_error": True, status_field: 500, "result": "server error"})
        diag = runner._diagnostics_from_spawn(stdout, "")
        assert diag["api_error_status"] == 500

    @pytest.mark.parametrize("status_field", ["status", "status_code", "code"])
    def test_numeric_status_is_captured_when_nested_under_error(self, status_field):
        stdout = json.dumps({"is_error": True, "error": {status_field: 503}, "result": "oops"})
        diag = runner._diagnostics_from_spawn(stdout, "")
        assert diag["api_error_status"] == 503

    def test_a_non_integer_status_field_is_never_captured(self):
        stdout = json.dumps({"is_error": True, "status": "unauthorized", "result": "oops"})
        diag = runner._diagnostics_from_spawn(stdout, "")
        assert diag["api_error_status"] is None

    def test_an_absolute_path_is_scrubbed_out_of_the_excerpt(self):
        diag = runner._diagnostics_from_text(
            "exception", "failed to read /home/operator/secret-project/config.yaml",
        )
        assert "/home/operator" not in diag["error_excerpt"]
        assert "secret-project" not in diag["error_excerpt"]
        assert "[PATH]" in diag["error_excerpt"]

    @pytest.mark.parametrize(
        "secret",
        [
            "sk-ant-" + "api03-" + "AAAABBBBCCCCDDDDEEEEFFFF",
            "ghp_" + "1234567890abcdef1234567890abcdef",
            "AKIA" + "ABCDEFGHIJKLMNOP",
            "xox" + "b-" + "1234567890-abcdefghijklmnop",
        ],
        ids=["anthropic-key", "github-pat", "aws-key", "slack-token"],
    )
    def test_bearer_and_api_key_shaped_secrets_are_redacted(self, secret):
        diag = runner._diagnostics_from_text("exception", f"auth failed with token {secret}")
        assert secret not in diag["error_excerpt"]
        assert "[REDACTED]" in diag["error_excerpt"]

    def test_the_excerpt_is_bounded_regardless_of_input_size(self):
        huge = "x" * 50_000
        diag = runner._diagnostics_from_text("exception", huge)
        assert len(diag["error_excerpt"]) <= runner._MAX_ERROR_EXCERPT_CHARS

    def test_the_fingerprint_is_deterministic_for_identical_input(self):
        a = runner._diagnostics_from_text("exception", "boom: connection reset")
        b = runner._diagnostics_from_text("exception", "boom: connection reset")
        assert a["error_fingerprint"] == b["error_fingerprint"]
        assert a["error_fingerprint"].startswith("sha256:")

    def test_the_fingerprint_differs_for_different_input(self):
        a = runner._diagnostics_from_text("exception", "boom: connection reset")
        b = runner._diagnostics_from_text("exception", "boom: disk full")
        assert a["error_fingerprint"] != b["error_fingerprint"]

    def test_the_fingerprint_differs_by_source_for_identical_text(self):
        a = runner._diagnostics_from_text("stderr", "boom")
        b = runner._diagnostics_from_text("exception", "boom")
        assert a["error_fingerprint"] != b["error_fingerprint"]

    def test_the_fingerprint_is_computed_after_redaction_not_before(self):
        """Two inputs that differ only in WHICH secret they carry redact to
        the identical excerpt, so they must fingerprint identically — the
        fingerprint is evidence over the redacted shape, never a side
        channel back to the original secret."""
        a = runner._diagnostics_from_text(
            "exception",
            "token " + "sk-ant-" + "api03-" + "AAAABBBBCCCCDDDDEEEEFFFF" + " rejected",
        )
        b = runner._diagnostics_from_text(
            "exception",
            "token " + "sk-ant-" + "api03-" + "ZZZZYYYYXXXXWWWWVVVVUUUU" + " rejected",
        )
        assert a["error_excerpt"] == b["error_excerpt"]
        assert a["error_fingerprint"] == b["error_fingerprint"]

    def test_never_includes_the_full_raw_stdout_body_beyond_the_bound(self):
        stdout = json.dumps({"is_error": True, "result": "y" * 10_000})
        diag = runner._diagnostics_from_spawn(stdout, "")
        assert stdout not in diag["error_excerpt"]
        assert len(diag["error_excerpt"]) <= runner._MAX_ERROR_EXCERPT_CHARS


class TestRunAttemptsDiagnostics:
    """``_run_attempts`` must capture a bounded, redacted diagnostic for a
    failing attempt and never fabricate one on success."""

    def test_success_never_carries_a_diagnostic(self, tmp_path, monkeypatch):
        repo = make_git_repo(tmp_path, "repo")
        monkeypatch.setattr(
            runner, "spawn_claude",
            lambda **kw: _spawn_result(exit_code=0, stdout=json.dumps({"result": "ok"}), model=kw["model"]),
        )
        outcome = runner._run_attempts(task="fix it", cwd=str(repo), complexity=None, cfg={})
        assert outcome["success"] is True
        assert outcome["diagnostic"] is None

    def test_other_failure_carries_a_bounded_diagnostic(self, tmp_path, monkeypatch):
        repo = make_git_repo(tmp_path, "repo")
        monkeypatch.setattr(
            runner, "spawn_claude",
            lambda **kw: _spawn_result(exit_code=1, stderr="Traceback: something broke", model=kw["model"]),
        )
        outcome = runner._run_attempts(task="fix it", cwd=str(repo), complexity=None, cfg={})
        assert outcome["success"] is False
        assert outcome["failure_class"] == "other"
        assert outcome["diagnostic"]["error_source"] == "stderr"
        assert outcome["diagnostic"]["error_excerpt"] == "Traceback: something broke"
        assert outcome["diagnostic"]["error_fingerprint"].startswith("sha256:")

    def test_auth_breaker_failure_still_carries_a_diagnostic(self, tmp_path, monkeypatch):
        repo = make_git_repo(tmp_path, "repo")
        monkeypatch.setattr(
            runner, "spawn_claude",
            lambda **kw: _spawn_result(exit_code=1, stderr="Error: Invalid API key", model=kw["model"]),
        )
        outcome = runner._run_attempts(task="fix it", cwd=str(repo), complexity=None, cfg={})
        assert outcome["failure_class"] == "auth"
        assert outcome["diagnostic"] is not None

    def test_timeout_carries_a_diagnostic_from_whatever_partial_output_exists(
        self, tmp_path, monkeypatch,
    ):
        repo = make_git_repo(tmp_path, "repo")
        monkeypatch.setattr(
            runner, "spawn_claude",
            lambda **kw: _spawn_result(exit_code=None, stderr="", timed_out=True, model=kw["model"]),
        )
        outcome = runner._run_attempts(task="fix it", cwd=str(repo), complexity=None, cfg={})
        assert outcome["failure_class"] == "timeout"
        assert outcome["diagnostic"] is not None

    def test_isolation_refused_does_not_fabricate_a_diagnostic(self, tmp_path, monkeypatch):
        repo = make_git_repo(tmp_path, "repo")

        def _refuse(**kw):
            raise runner.SpawnRefused("docker preflight failed")

        monkeypatch.setattr(runner, "spawn_claude", _refuse)
        outcome = runner._run_attempts(task="fix it", cwd=str(repo), complexity=None, cfg={})
        assert outcome["failure_class"] == "isolation_refused"
        assert outcome["diagnostic"] is None


class TestFailureDiagnosticTelemetryIntegration:
    """End-to-end: a failing spawn leaves actionable, bounded, redacted
    evidence in telemetry, while success, OAuth-preflight-HOLD, and
    breaker-open refusal never fabricate one, and it never leaks into the
    caller-facing JSON tool result."""

    def test_other_failure_telemetry_carries_the_diagnostic(self, tmp_path, monkeypatch):
        repo = make_git_repo(tmp_path, "repo")
        monkeypatch.setattr(config, "_load_raw_config", lambda: _cfg_with_roots(repo))
        monkeypatch.setattr(
            runner, "spawn_claude",
            lambda **kw: _spawn_result(exit_code=1, stderr="Traceback: it broke", model=kw["model"]),
        )
        records = []
        monkeypatch.setattr(
            telemetry, "append_record",
            lambda record, configured_path="": records.append(record),
        )

        raw = runner.run_worker({"task": "fix it", "cwd": str(repo)}, session_id="sess:diag-other")
        result = json.loads(raw)

        assert result["failure_class"] == "other"
        assert "diagnostic" not in result  # never leaks into the caller-facing result
        assert len(records) == 1
        diag = records[0]["diagnostic"]
        assert diag["error_source"] == "stderr"
        assert diag["error_excerpt"] == "Traceback: it broke"
        assert diag["error_fingerprint"].startswith("sha256:")

    def test_success_telemetry_never_fabricates_a_diagnostic(self, tmp_path, monkeypatch):
        repo = make_git_repo(tmp_path, "repo")
        monkeypatch.setattr(config, "_load_raw_config", lambda: _cfg_with_roots(repo))
        monkeypatch.setattr(
            runner, "spawn_claude",
            lambda **kw: _spawn_result(exit_code=0, stdout=json.dumps({"result": "done"}), model=kw["model"]),
        )
        monkeypatch.setattr(runner, "_git_changed_files", lambda cwd, repo_roots=None: [])
        records = []
        monkeypatch.setattr(
            telemetry, "append_record",
            lambda record, configured_path="": records.append(record),
        )

        runner.run_worker({"task": "fix it", "cwd": str(repo)}, session_id="sess:diag-success")

        assert len(records) == 1
        assert records[0]["diagnostic"] is None


class TestRunAttemptsThreadsAllowOpus:
    """``_run_attempts`` must pass ``allow_opus`` straight through to
    ``routing.choose_model`` unchanged — that function is the single source
    of truth for what counts as authorization (see
    ``test_claude_worker_routing.py``); this only proves the wiring."""

    def test_allow_opus_and_complexity_reach_choose_model(self, tmp_path, monkeypatch):
        repo = make_git_repo(tmp_path, "repo")
        calls = []

        def _fake_choose(task, complexity=None, allow_opus=False):
            calls.append({"task": task, "complexity": complexity, "allow_opus": allow_opus})
            return "claude-sonnet-5", "default"

        monkeypatch.setattr(runner._routing, "choose_model", _fake_choose)
        monkeypatch.setattr(
            runner, "spawn_claude",
            lambda **kw: _spawn_result(exit_code=0, model=kw["model"]),
        )

        runner._run_attempts(
            task="harden auth", cwd=str(repo), complexity="security", cfg={}, allow_opus=True,
        )

        assert calls == [{"task": "harden auth", "complexity": "security", "allow_opus": True}]

    def test_omitted_allow_opus_defaults_to_false(self, tmp_path, monkeypatch):
        repo = make_git_repo(tmp_path, "repo")
        calls = []

        def _fake_choose(task, complexity=None, allow_opus=False):
            calls.append(allow_opus)
            return "claude-sonnet-5", "default"

        monkeypatch.setattr(runner._routing, "choose_model", _fake_choose)
        monkeypatch.setattr(
            runner, "spawn_claude",
            lambda **kw: _spawn_result(exit_code=0, model=kw["model"]),
        )

        runner._run_attempts(task="fix it", cwd=str(repo), complexity=None, cfg={})

        assert calls == [False]


class TestRunWorkerModelAuthorization:
    """End-to-end through ``run_worker``: Opus is selected only when a call
    supplies BOTH an allowed ``complexity`` AND the literal boolean
    ``allow_opus=True``. Existing callers that omit ``allow_opus`` entirely,
    and callers that send a truthy-but-not-``True`` value, all stay on
    Sonnet — never Opus."""

    def _run(self, repo, monkeypatch, args, session_id):
        monkeypatch.setattr(config, "_load_raw_config", lambda: _cfg_with_roots(repo))
        monkeypatch.setattr(
            runner, "spawn_claude",
            lambda **kw: _spawn_result(
                exit_code=0, stdout=json.dumps({"result": "ok", "is_error": False}),
                model=kw["model"],
            ),
        )
        monkeypatch.setattr(runner, "_git_changed_files", lambda cwd, repo_roots=None: [])
        return json.loads(runner.run_worker(args, session_id=session_id))

    def test_complexity_and_allow_opus_true_selects_opus(self, tmp_path, monkeypatch):
        repo = make_git_repo(tmp_path, "repo")
        result = self._run(
            repo, monkeypatch,
            {"task": "harden auth", "cwd": str(repo), "complexity": "security", "allow_opus": True},
            "sess:opus-authorized",
        )
        assert result["model"] == "claude-opus-5"
        assert "authoriz" in result["route_reason"].lower()

    def test_complexity_alone_stays_sonnet(self, tmp_path, monkeypatch):
        """The exact hole this hardening closes: ``complexity`` alone used
        to be sufficient to reach Opus."""
        repo = make_git_repo(tmp_path, "repo")
        result = self._run(
            repo, monkeypatch,
            {"task": "harden auth", "cwd": str(repo), "complexity": "architecture"},
            "sess:opus-complexity-only",
        )
        assert result["model"] == "claude-sonnet-5"
        assert result["route_reason"] == "default"

    def test_allow_opus_alone_stays_sonnet(self, tmp_path, monkeypatch):
        repo = make_git_repo(tmp_path, "repo")
        result = self._run(
            repo, monkeypatch,
            {"task": "harden auth", "cwd": str(repo), "allow_opus": True},
            "sess:opus-allow-only",
        )
        assert result["model"] == "claude-sonnet-5"

    def test_existing_callers_omitting_allow_opus_remain_sonnet(self, tmp_path, monkeypatch):
        repo = make_git_repo(tmp_path, "repo")
        result = self._run(
            repo, monkeypatch,
            {"task": "harden auth", "cwd": str(repo)},
            "sess:opus-omitted",
        )
        assert result["model"] == "claude-sonnet-5"

    @pytest.mark.parametrize("truthy_but_not_true", ["true", 1, "yes"])
    def test_truthy_non_bool_allow_opus_never_authorizes(
        self, tmp_path, monkeypatch, truthy_but_not_true,
    ):
        repo = make_git_repo(tmp_path, "repo")
        result = self._run(
            repo, monkeypatch,
            {
                "task": "harden auth", "cwd": str(repo), "complexity": "security",
                "allow_opus": truthy_but_not_true,
            },
            f"sess:opus-truthy-{truthy_but_not_true}",
        )
        assert result["model"] == "claude-sonnet-5"


class TestResolveValidationStatusUnit:
    """Direct unit coverage of ``_resolve_validation_status`` — the closed
    default is ``unverified``/``parent_verification_required=True``; only a
    real, operator-configured verifier that actually ran and exited 0 can
    move it to ``verified_by_configured_verifier``."""

    def test_task_failure_short_circuits_without_running_any_verifier(self, monkeypatch):
        def _must_not_run(*a, **k):
            raise AssertionError("a failed task must never trigger the verifier")

        monkeypatch.setattr(runner, "run_verification_command", _must_not_run)
        status, required, result = runner._resolve_validation_status(
            {"verification": {"enabled": True, "command": ["pytest"]}}, "/tmp/x", False,
        )
        assert status == runner.VALIDATION_STATUS_UNVERIFIED
        assert required is True
        assert result is None

    def test_missing_verification_section_stays_unverified(self):
        status, required, result = runner._resolve_validation_status({}, "/tmp/x", True)
        assert status == runner.VALIDATION_STATUS_UNVERIFIED
        assert required is True
        assert result is None

    def test_enabled_without_a_command_stays_unverified(self, monkeypatch):
        def _must_not_run(*a, **k):
            raise AssertionError("no command configured — the verifier must never run")

        monkeypatch.setattr(runner, "run_verification_command", _must_not_run)
        status, required, result = runner._resolve_validation_status(
            {"verification": {"enabled": True, "command": []}}, "/tmp/x", True,
        )
        assert status == runner.VALIDATION_STATUS_UNVERIFIED
        assert required is True
        assert result is None

    def test_truthy_non_bool_enabled_never_authorizes_running_the_verifier(self, monkeypatch):
        def _must_not_run(*a, **k):
            raise AssertionError("a non-bool 'enabled' must never authorize a run")

        monkeypatch.setattr(runner, "run_verification_command", _must_not_run)
        status, required, result = runner._resolve_validation_status(
            {"verification": {"enabled": "true", "command": ["pytest"]}}, "/tmp/x", True,
        )
        assert status == runner.VALIDATION_STATUS_UNVERIFIED
        assert required is True
        assert result is None

    def test_configured_verifier_that_passes_is_verified_distinctly(self, monkeypatch):
        monkeypatch.setattr(
            runner, "run_verification_command",
            lambda repo_root, command, timeout_seconds: {
                "ok": True, "exit_code": 0, "stdout": "", "stderr": "",
                "duration_ms": 5, "timed_out": False, "failure_class": None, "reason": "",
            },
        )
        status, required, result = runner._resolve_validation_status(
            {"verification": {"enabled": True, "command": ["pytest", "-q"]}}, "/tmp/x", True,
        )
        assert status == runner.VALIDATION_STATUS_VERIFIED
        assert required is False
        assert result["ok"] is True

    def test_configured_verifier_that_fails_stays_verification_failed(self, monkeypatch):
        monkeypatch.setattr(
            runner, "run_verification_command",
            lambda repo_root, command, timeout_seconds: {
                "ok": False, "exit_code": 1, "stdout": "", "stderr": "2 failed",
                "duration_ms": 5, "timed_out": False,
                "failure_class": "verification_failed", "reason": "",
            },
        )
        status, required, result = runner._resolve_validation_status(
            {"verification": {"enabled": True, "command": ["pytest", "-q"]}}, "/tmp/x", True,
        )
        assert status == runner.VALIDATION_STATUS_VERIFICATION_FAILED
        assert required is True
        assert result["ok"] is False

    def test_verifier_timeout_is_the_fixed_policy_constant_not_the_configured_worker_timeout(
        self, monkeypatch,
    ):
        """A configured ``isolation.timeout_seconds`` of up to 900s must
        never be handed to the verifier as-is — the verifier always gets the
        fixed, non-configurable ``policy.VERIFICATION_TIMEOUT_SECONDS``."""
        captured = {}

        def _fake_run_verification_command(repo_root, command, timeout_seconds):
            captured["timeout_seconds"] = timeout_seconds
            return {
                "ok": True, "exit_code": 0, "stdout": "", "stderr": "",
                "duration_ms": 5, "timed_out": False, "failure_class": None, "reason": "",
            }

        monkeypatch.setattr(runner, "run_verification_command", _fake_run_verification_command)
        runner._resolve_validation_status(
            {
                "verification": {"enabled": True, "command": ["pytest", "-q"]},
                "isolation": {"timeout_seconds": 900},
            },
            "/tmp/x", True,
        )
        assert captured["timeout_seconds"] == policy.VERIFICATION_TIMEOUT_SECONDS
        assert captured["timeout_seconds"] < 900

    def test_verifier_timeout_is_bounded_further_by_a_shorter_isolation_timeout(
        self, monkeypatch,
    ):
        """When the operator's isolation timeout is shorter than the fixed
        verification timeout, the verifier must never outlive the worker's
        own isolated run."""
        captured = {}

        def _fake_run_verification_command(repo_root, command, timeout_seconds):
            captured["timeout_seconds"] = timeout_seconds
            return {
                "ok": True, "exit_code": 0, "stdout": "", "stderr": "",
                "duration_ms": 5, "timed_out": False, "failure_class": None, "reason": "",
            }

        monkeypatch.setattr(runner, "run_verification_command", _fake_run_verification_command)
        short_timeout = policy.VERIFICATION_TIMEOUT_SECONDS - 1
        runner._resolve_validation_status(
            {
                "verification": {"enabled": True, "command": ["pytest", "-q"]},
                "isolation": {"timeout_seconds": short_timeout},
            },
            "/tmp/x", True,
        )
        assert captured["timeout_seconds"] == short_timeout


class TestRunWorkerValidationStatus:
    """End-to-end: the worker's own self-reported stdout — even a claim like
    "all tests pass" — must never be able to flip ``validation_status``.
    Only a real, operator-configured verifier process actually run in its
    own sandbox and inspected for its own exit code can do that."""

    def _run(self, repo, monkeypatch, cfg_overrides, spawn_stdout_result, session_id):
        monkeypatch.setattr(
            config, "_load_raw_config",
            lambda: _cfg_with_roots(repo, **cfg_overrides),
        )
        monkeypatch.setattr(
            runner, "spawn_claude",
            lambda **kw: _spawn_result(
                exit_code=0,
                stdout=json.dumps({"result": spawn_stdout_result, "is_error": False}),
                model=kw["model"],
            ),
        )
        monkeypatch.setattr(runner, "_git_changed_files", lambda cwd, repo_roots=None: ["a.py"])
        return json.loads(runner.run_worker(
            {"task": "fix it", "cwd": str(repo)}, session_id=session_id,
        ))

    def test_no_verifier_configured_is_unverified_despite_a_confident_summary(
        self, tmp_path, monkeypatch,
    ):
        repo = make_git_repo(tmp_path, "repo")

        def _must_not_run(*a, **k):
            raise AssertionError("no verifier is configured — it must never run")

        monkeypatch.setattr(runner, "run_verification_command", _must_not_run)

        result = self._run(
            repo, monkeypatch, {}, "All tests pass, fully verified and ready to merge.",
            "sess:validation-unverified",
        )

        assert result["success"] is True
        assert result["validation_status"] == "unverified"
        assert result["parent_verification_required"] is True
        assert result["verification"] is None

    def test_configured_verifier_passing_is_verified_distinctly(self, tmp_path, monkeypatch):
        repo = make_git_repo(tmp_path, "repo")
        monkeypatch.setattr(
            runner, "run_verification_command",
            lambda repo_root, command, timeout_seconds: {
                "ok": True, "exit_code": 0, "stdout": "1 passed", "stderr": "",
                "duration_ms": 5, "timed_out": False, "failure_class": None, "reason": "",
            },
        )

        result = self._run(
            repo, monkeypatch,
            {"verification": {"enabled": True, "command": ["pytest", "-q"]}},
            "done", "sess:validation-verified",
        )

        assert result["validation_status"] == "verified_by_configured_verifier"
        assert result["parent_verification_required"] is False
        assert result["verification"]["ok"] is True

    def test_worker_claiming_tests_pass_cannot_override_a_failing_verifier(
        self, tmp_path, monkeypatch,
    ):
        """The exact regression this hardening prevents: Claude's own
        stdout says the tests pass, but the REAL configured verifier
        disagrees — the verifier's actual exit code must win."""
        repo = make_git_repo(tmp_path, "repo")
        monkeypatch.setattr(
            runner, "run_verification_command",
            lambda repo_root, command, timeout_seconds: {
                "ok": False, "exit_code": 1, "stdout": "", "stderr": "1 failed",
                "duration_ms": 5, "timed_out": False,
                "failure_class": "verification_failed", "reason": "",
            },
        )

        result = self._run(
            repo, monkeypatch,
            {"verification": {"enabled": True, "command": ["pytest", "-q"]}},
            "All tests pass!", "sess:validation-worker-lies",
        )

        assert result["validation_status"] == "verification_failed"
        assert result["parent_verification_required"] is True
        assert result["verification"]["ok"] is False

    def test_task_failure_is_unverified_and_never_runs_the_verifier(
        self, tmp_path, monkeypatch,
    ):
        repo = make_git_repo(tmp_path, "repo")
        monkeypatch.setattr(
            config, "_load_raw_config",
            lambda: _cfg_with_roots(repo, verification={"enabled": True, "command": ["pytest"]}),
        )
        monkeypatch.setattr(
            runner, "spawn_claude",
            lambda **kw: _spawn_result(exit_code=1, stderr="boom", model=kw["model"]),
        )
        monkeypatch.setattr(runner, "_git_changed_files", lambda cwd, repo_roots=None: [])

        def _must_not_run(*a, **k):
            raise AssertionError("the worker spawn itself failed — the verifier must never run")

        monkeypatch.setattr(runner, "run_verification_command", _must_not_run)

        result = json.loads(runner.run_worker(
            {"task": "fix it", "cwd": str(repo)}, session_id="sess:validation-task-failed",
        ))

        assert result["success"] is False
        assert result["validation_status"] == "unverified"
        assert result["parent_verification_required"] is True
        assert result["verification"] is None

    def test_breaker_open_result_reports_unverified_by_default(self, tmp_path, monkeypatch):
        repo = make_git_repo(tmp_path, "repo")
        monkeypatch.setattr(config, "_load_raw_config", lambda: _cfg_with_roots(repo))
        breaker.record_failure("auth", {"auth": 3600})

        result = json.loads(runner.run_worker(
            {"task": "fix it", "cwd": str(repo)}, session_id="sess:validation-breaker-open",
        ))

        assert result["validation_status"] == "unverified"
        assert result["parent_verification_required"] is True
        assert result["verification"] is None


class TestFailureDiagnosticTelemetryIntegrationContinued:
    """Continuation of ``TestFailureDiagnosticTelemetryIntegration`` above —
    split into its own class only because the validation-status coverage
    was inserted between them; behavior and fixtures are unchanged."""

    def test_breaker_open_refusal_never_fabricates_a_diagnostic(self, tmp_path, monkeypatch):
        repo = make_git_repo(tmp_path, "repo")
        monkeypatch.setattr(config, "_load_raw_config", lambda: _cfg_with_roots(repo))
        breaker.record_failure("auth", {"auth": 3600})
        records = []
        monkeypatch.setattr(
            telemetry, "append_record",
            lambda record, configured_path="": records.append(record),
        )

        runner.run_worker({"task": "fix it", "cwd": str(repo)}, session_id="sess:diag-breaker-open")

        assert len(records) == 1
        assert records[0]["diagnostic"] is None

    def test_oauth_preflight_hold_never_fabricates_a_diagnostic(self, tmp_path, monkeypatch):
        repo = make_git_repo(tmp_path, "repo")
        monkeypatch.setattr(config, "_load_raw_config", lambda: _cfg_with_roots(repo))
        monkeypatch.setattr(
            oauth, "preflight",
            lambda *a, **k: {
                "ok": False, "state": oauth.STATE_MISSING, "classification": "auth",
                "refresh_attempted": False, "reason": "", "freshness": {},
            },
        )
        records = []
        monkeypatch.setattr(
            telemetry, "append_record",
            lambda record, configured_path="": records.append(record),
        )

        runner.run_worker({"task": "fix it", "cwd": str(repo)}, session_id="sess:diag-preflight-hold")

        assert len(records) == 1
        assert records[0]["diagnostic"] is None

    def test_secret_and_path_shaped_content_in_a_failure_never_reaches_telemetry(
        self, tmp_path, monkeypatch,
    ):
        repo = make_git_repo(tmp_path, "repo")
        monkeypatch.setattr(config, "_load_raw_config", lambda: _cfg_with_roots(repo))
        leaked_secret = "sk-ant-" + "api03-" + "AAAABBBBCCCCDDDDEEEEFFFF"
        monkeypatch.setattr(
            runner, "spawn_claude",
            lambda **kw: _spawn_result(
                exit_code=1,
                stderr=f"failed reading /home/operator/project/config with key {leaked_secret}",
                model=kw["model"],
            ),
        )
        records = []
        monkeypatch.setattr(
            telemetry, "append_record",
            lambda record, configured_path="": records.append(record),
        )

        runner.run_worker({"task": "fix it", "cwd": str(repo)}, session_id="sess:diag-redact")

        assert len(records) == 1
        excerpt = records[0]["diagnostic"]["error_excerpt"]
        assert leaked_secret not in excerpt
        assert "/home/operator" not in excerpt
        assert "[REDACTED]" in excerpt
        assert "[PATH]" in excerpt


class TestPostSpawnAuthSessionIntegration:
    """The 2026-09-07/08 incident itself: the CLI exits 1 in ~1.5-1.7s with a
    structured session-invalid signal in stdout and no numeric 401. That must
    classify as ``auth_preflight`` end-to-end — no retry, no breaker
    mutation, reauthentication guidance in the result, and a diagnostic in
    telemetry."""

    def test_spawn_reported_session_invalid_signal_classifies_end_to_end(
        self, tmp_path, monkeypatch,
    ):
        repo = make_git_repo(tmp_path, "repo")
        monkeypatch.setattr(config, "_load_raw_config", lambda: _cfg_with_roots(repo))

        calls = []

        def _fake_spawn(**kwargs):
            calls.append(kwargs)
            stdout = json.dumps({
                "is_error": True,
                "subtype": "error_during_execution",
                "result": "Invalid API key · Please run /login",
            })
            return _spawn_result(exit_code=1, stdout=stdout, stderr="", model=kwargs["model"])

        monkeypatch.setattr(runner, "spawn_claude", _fake_spawn)

        records = []
        monkeypatch.setattr(
            telemetry, "append_record",
            lambda record, configured_path="": records.append(record),
        )

        raw = runner.run_worker(
            {"task": "fix it", "cwd": str(repo)}, session_id="sess:auth-preflight-spawn",
        )
        result = json.loads(raw)

        assert len(calls) == 1  # never retried
        assert result["success"] is False
        assert result["failure_class"] == breaker.AUTH_PREFLIGHT_FAILURE_CLASS
        assert result["breaker"]["open"] is False
        assert breaker.is_open() is False
        assert "re-authenticate" in result["error"].lower()
        assert len(records) == 1
        assert records[0]["failure_class"] == breaker.AUTH_PREFLIGHT_FAILURE_CLASS
        assert records[0]["diagnostic"] is not None

    def test_bare_401_in_task_output_never_opens_the_auth_breaker(self, tmp_path, monkeypatch):
        repo = make_git_repo(tmp_path, "repo")
        monkeypatch.setattr(config, "_load_raw_config", lambda: _cfg_with_roots(repo))
        monkeypatch.setattr(
            runner, "spawn_claude",
            lambda **kw: _spawn_result(
                exit_code=1, stderr="test_401.py::test_x FAILED", model=kw["model"],
            ),
        )

        result = json.loads(runner.run_worker(
            {"task": "fix it", "cwd": str(repo)}, session_id="sess:bare-401",
        ))

        assert result["failure_class"] == "other"
        assert result["breaker"]["open"] is False
        assert breaker.is_open() is False
