"""RED->GREEN tests for ``runner.run_worker`` orchestration and the
``_git_changed_files`` helper.

Covers requirements (1)/(4)/(6)/(9): the claude_worker tool handler itself —
routing + breaker + isolated spawn + evidence + exactly-one-telemetry-record
per invocation, with the hard "no third spawn" guarantee proven by call
counting on a mocked ``spawn_claude``. The container command shape itself
(mounts, hardening, preflight) is covered by
``test_claude_worker_isolation.py``; ``spawn_claude`` is mocked here at the
orchestration boundary — it never spawns a host ``claude`` process, only the
locked-down ``docker run`` sandbox.
"""

from __future__ import annotations

import json
import subprocess

import pytest

from tests.plugins._claude_worker_helpers import load_submodule

runner = load_submodule("runner")
breaker = load_submodule("breaker")
telemetry = load_submodule("telemetry")
config = load_submodule("config")
policy = load_submodule("policy")


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    runner.reset_preflight_cache()
    yield


def _cfg_with_roots(*roots, **overrides):
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
        repo = tmp_path / "repo"
        repo.mkdir()
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
        repo = tmp_path / "repo"
        repo.mkdir()
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
        repo = tmp_path / "repo"
        repo.mkdir()
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
        repo = tmp_path / "repo"
        repo.mkdir()
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
        repo = tmp_path / "repo"
        repo.mkdir()
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
        repo = tmp_path / "repo"
        repo.mkdir()
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
    def test_sonnet_failure_escalates_once_to_opus_success(self, tmp_path, monkeypatch):
        repo = tmp_path / "repo"
        repo.mkdir()
        monkeypatch.setattr(config, "_load_raw_config", lambda: _cfg_with_roots(repo))

        calls = []

        def _fake_spawn(**kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                return _spawn_result(exit_code=1, stderr="Traceback: something broke", model=kwargs["model"])
            return _spawn_result(exit_code=0, stdout=json.dumps({"result": "done", "is_error": False}), model=kwargs["model"])

        monkeypatch.setattr(runner, "spawn_claude", _fake_spawn)
        monkeypatch.setattr(runner, "_git_changed_files", lambda cwd, repo_roots=None: [])

        raw = runner.run_worker({"task": "hard bug", "cwd": str(repo)}, session_id="sess:3")
        result = json.loads(raw)

        assert len(calls) == 2
        assert calls[0]["model"] == "claude-sonnet-5"
        assert calls[1]["model"] == "claude-opus-5"
        assert result["success"] is True
        assert result["escalated"] is True
        assert result["attempts"] == 2
        assert result["model"] == "claude-opus-5"

    def test_escalation_uses_only_remaining_total_budget(self, tmp_path, monkeypatch):
        repo = tmp_path / "repo"
        repo.mkdir()
        monkeypatch.setattr(
            config, "_load_raw_config",
            lambda: _cfg_with_roots(repo, isolation={"timeout_seconds": 900}),
        )
        calls = []

        def _fake_spawn(**kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                return _spawn_result(
                    exit_code=1, timed_out=True, duration_ms=900_000,
                    model=kwargs["model"],
                )
            return _spawn_result(
                exit_code=0, stdout=json.dumps({"result": "done"}),
                duration_ms=100, model=kwargs["model"],
            )

        monkeypatch.setattr(runner, "spawn_claude", _fake_spawn)
        monkeypatch.setattr(runner, "_git_changed_files", lambda cwd, repo_roots=None: [])

        result = json.loads(runner.run_worker(
            {"task": "hard bug", "cwd": str(repo)}, session_id="sess:budget",
        ))

        assert result["success"] is True
        assert [call["timeout_seconds"] for call in calls] == [
            900, policy.MAX_TOTAL_ATTEMPT_SECONDS - 900,
        ]

    def test_no_third_spawn_after_two_failures(self, tmp_path, monkeypatch):
        repo = tmp_path / "repo"
        repo.mkdir()
        monkeypatch.setattr(config, "_load_raw_config", lambda: _cfg_with_roots(repo))

        calls = []

        def _fake_spawn(**kwargs):
            calls.append(kwargs)
            return _spawn_result(exit_code=1, stderr="Traceback: still broken", model=kwargs["model"])

        monkeypatch.setattr(runner, "spawn_claude", _fake_spawn)

        raw = runner.run_worker({"task": "hard bug", "cwd": str(repo)}, session_id="sess:4")
        result = json.loads(raw)

        assert len(calls) == 2
        assert result["success"] is False
        assert result["attempts"] == 2
        assert result["escalated"] is True

    def test_auth_failure_opens_breaker_and_does_not_escalate(self, tmp_path, monkeypatch):
        repo = tmp_path / "repo"
        repo.mkdir()
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
        repo = tmp_path / "repo"
        repo.mkdir()
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

    def test_malformed_json_stdout_with_generic_failure_still_escalates_safely(self, tmp_path, monkeypatch):
        """Non-JSON/garbled stdout must never raise and must not be
        misclassified as auth — it should behave exactly like the existing
        generic-failure path (escalate once to Opus)."""
        repo = tmp_path / "repo"
        repo.mkdir()
        monkeypatch.setattr(config, "_load_raw_config", lambda: _cfg_with_roots(repo))

        calls = []

        def _fake_spawn(**kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                return _spawn_result(exit_code=1, stdout="not json{{{", stderr="", model=kwargs["model"])
            return _spawn_result(exit_code=0, stdout=json.dumps({"result": "done", "is_error": False}), model=kwargs["model"])

        monkeypatch.setattr(runner, "spawn_claude", _fake_spawn)
        monkeypatch.setattr(runner, "_git_changed_files", lambda cwd, repo_roots=None: [])

        raw = runner.run_worker({"task": "fix it", "cwd": str(repo)}, session_id="sess:7")
        result = json.loads(raw)

        assert len(calls) == 2
        assert result["success"] is True
        assert result["escalated"] is True
        assert breaker.is_open() is False


class TestRunWorkerCwdRejection:
    def test_cwd_outside_repo_roots_refuses_without_spawning(self, tmp_path, monkeypatch):
        repo = tmp_path / "repo"
        repo.mkdir()
        outside = tmp_path / "outside"
        outside.mkdir()
        monkeypatch.setattr(config, "_load_raw_config", lambda: _cfg_with_roots(repo))

        def _must_not_spawn(**kwargs):
            raise AssertionError("spawn_claude must not be called for a rejected cwd")

        monkeypatch.setattr(runner, "spawn_claude", _must_not_spawn)

        raw = runner.run_worker({"task": "fix it", "cwd": str(outside)}, session_id="sess:6")
        result = json.loads(raw)

        assert result["success"] is False
        assert result["attempts"] == 0
        assert result["failure_class"] == "cwd_rejected"


class TestTelemetryIntegration:
    def test_exactly_one_telemetry_record_per_invocation(self, tmp_path, monkeypatch):
        repo = tmp_path / "repo"
        repo.mkdir()
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
        repo = tmp_path / "repo"
        repo.mkdir()
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
        repo = tmp_path / "repo"
        repo.mkdir()
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
        repo = tmp_path / "repo"
        repo.mkdir()
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
        repo = tmp_path / "repo"
        repo.mkdir()
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
        repo = tmp_path / "repo"
        repo.mkdir()
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
        repo = tmp_path / "repo"
        repo.mkdir()
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
        repo = tmp_path / "repo"
        repo.mkdir()
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
        repo = tmp_path / "repo"
        repo.mkdir()
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
        repo = tmp_path / "repo"
        repo.mkdir()
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
        repo = tmp_path / "repo"
        repo.mkdir()
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
        repo = tmp_path / "repo"
        repo.mkdir()
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
        repo = tmp_path / "repo"
        repo.mkdir()
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
