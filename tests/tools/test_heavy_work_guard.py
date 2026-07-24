from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tools.heavy_work_guard import (
    HeavyWorkConflict,
    acquire_heavy_work_lease,
    classify_heavy_work,
    heavy_work_requests_detach,
)


class _FakeLease:
    def __init__(self):
        self.release_count = 0

    def release(self):
        self.release_count += 1

    def child_pass_fds(self):
        return ()


@pytest.mark.parametrize(
    ("command", "category"),
    [
        ("pytest -q", "test-suite"),
        ("python -m pytest tests", "test-suite"),
        ("python3.13 -m pytest tests", "test-suite"),
        ("uv run pytest -q", "test-suite"),
        ("poetry run python -m pytest", "test-suite"),
        ("bash -lc 'cd /tmp && pytest -q'", "test-suite"),
        ("if true; then pytest -q; fi", "test-suite"),
        ("{ pytest -q; }", "test-suite"),
        ("( pytest -q )", "test-suite"),
        ("nohup pytest -q >/tmp/t.log 2>&1 &", "test-suite"),
        ("setsid pytest -q >/tmp/t.log 2>&1 &", "test-suite"),
        ("command pytest -q", "test-suite"),
        ("python3 -c \"import subprocess; subprocess.run(['pytest','-q'])\"", "test-suite"),
        ("sudo -u root pytest -q", "dynamic-execution"),
        ("timeout --kill-after 3 10 pytest -q", "dynamic-execution"),
        ("eval pytest -q", "dynamic-execution"),
        ("sh -c 'eval pytest -q'", "dynamic-execution"),
        ("cmd=pytest; $cmd -q", "dynamic-execution"),
        (
            "python -c \"import subprocess; subprocess.run(['py' + 'test','-q'])\"",
            "dynamic-execution",
        ),
        (
            "python -c \"import os; os.system('py' + 'test -q')\" --help",
            "dynamic-execution",
        ),
        ("bash ./opaque-script.sh --help", "dynamic-execution"),
        ("printf 'pytest -q' | bash", "dynamic-execution"),
        ("xargs pytest -q", "dynamic-execution"),
        ("node -e \"require('child_process').spawn('pytest')\"", "dynamic-execution"),
        ("env -i pytest -q", "dynamic-execution"),
        ("env --ignore-environment pytest -q", "dynamic-execution"),
        ("nice -n 10 pytest -q", "dynamic-execution"),
        ("ionice -c 3 pytest -q", "dynamic-execution"),
        ("time -p pytest -q", "dynamic-execution"),
        ("flock /tmp/l pytest -q", "dynamic-execution"),
        ("chronic pytest -q", "dynamic-execution"),
        ("npx --yes vitest run", "dynamic-execution"),
        ("npm exec -- vitest", "dynamic-execution"),
        ("pnpm test", "test-suite"),
        ("npm run test:unit", "test-suite"),
        ("cargo test --workspace", "test-suite"),
        ("go test ./...", "test-suite"),
        ("npx vitest run", "test-suite"),
        ("supabase start", "database-lab"),
        ("npx supabase db reset", "database-lab"),
        ("pglite ./scripts/check.ts", "database-lab"),
        ("claude -p 'review this diff'", "ai-reviewer"),
        ("codex exec --full-auto 'review'", "ai-reviewer"),
        ("opencode run 'review'", "ai-reviewer"),
    ],
)
def test_classifies_heavy_commands(command, category):
    assert classify_heavy_work(command) == category


@pytest.mark.parametrize(
    "command",
    [
        "echo pytest",
        "echo 'pytest && supabase start'",
        "pytest --help",
        "supabase status",
        "supabase stop",
        "codex --version",
        "claude --help",
        "python --help",
        "bash --version",
        "node --help",
        "command -v pytest",
        "command -V pytest",
        "ps aux | grep pglite",
    ],
)
def test_does_not_block_inspection_cleanup_or_mentions(command):
    assert classify_heavy_work(command) is None


@pytest.mark.parametrize(
    "command",
    [
        "nohup pytest -q >/tmp/t.log 2>&1 & exit",
        "pytest -q & disown",
        "python worker.py --daemon",
        "systemd-run pytest -q",
    ],
)
def test_detects_self_detaching_heavy_commands(command):
    assert heavy_work_requests_detach(command) is True


@pytest.mark.parametrize(
    "command",
    [
        "pytest -q",
        "nohup pytest -q",
        "setsid pytest -q",
        "echo harmless &",
    ],
)
def test_does_not_reject_managed_or_nonheavy_commands(command):
    assert heavy_work_requests_detach(command) is False


def test_kernel_lease_blocks_second_process_and_recovers_after_release(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    lease, conflict = acquire_heavy_work_lease(
        "pytest -q", limit=1, session_key="first"
    )
    assert lease is not None
    assert conflict is None

    script = """
import json
from tools.heavy_work_guard import acquire_heavy_work_lease
lease, conflict = acquire_heavy_work_lease('codex exec review', limit=1, session_key='second')
print(json.dumps({'acquired': lease is not None, 'owner': conflict.owner if conflict else None}))
if lease:
    lease.release()
"""
    env = dict(os.environ)
    env["HERMES_HOME"] = str(tmp_path)
    blocked = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).parents[2],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    payload = json.loads(blocked.stdout)
    assert payload["acquired"] is False
    assert payload["owner"]["category"] == "test-suite"
    assert payload["owner"]["session_key"] == "first"

    lease.release()

    acquired = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).parents[2],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(acquired.stdout)["acquired"] is True


@pytest.mark.skipif(os.name == "nt", reason="POSIX FD inheritance semantics")
def test_child_inherited_fd_keeps_slot_after_holder_crash(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    holder_script = r'''
import os
import subprocess
from tools.heavy_work_guard import acquire_heavy_work_lease
lease, conflict = acquire_heavy_work_lease("pytest -q", limit=1, session_key="crash-holder")
assert lease is not None and conflict is None
child = subprocess.Popen(
    ["sleep", "2"],
    stdin=subprocess.DEVNULL,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
    start_new_session=True,
    pass_fds=lease.child_pass_fds(),
)
print(child.pid, flush=True)
os._exit(0)
'''
    env = dict(os.environ)
    env["HERMES_HOME"] = str(tmp_path)
    holder = subprocess.run(
        [sys.executable, "-c", holder_script],
        cwd=Path(__file__).parents[2],
        env=env,
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    )
    child_pid = int(holder.stdout.strip())
    # Signal 0 only probes liveness and is permitted by the live-system guard.
    os.kill(child_pid, 0)
    lease, conflict = acquire_heavy_work_lease("codex exec review", limit=1)
    assert lease is None
    assert conflict is not None

    # Once the surviving job exits naturally, the kernel releases the inherited lock.
    import time

    deadline = time.time() + 5
    recovered = None
    while time.time() < deadline:
        recovered, _ = acquire_heavy_work_lease("codex exec review", limit=1)
        if recovered is not None:
            break
        time.sleep(0.05)
    assert recovered is not None
    recovered.release()


@pytest.mark.skipif(os.name == "nt", reason="POSIX FD inheritance semantics")
def test_parent_release_does_not_unlock_inherited_child_fd(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    lease, conflict = acquire_heavy_work_lease(
        "pytest -q", limit=1, session_key="release-parent"
    )
    assert lease is not None and conflict is None
    child = subprocess.Popen(
        ["sleep", "2"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
        pass_fds=lease.child_pass_fds(),
    )

    lease.release()
    rival, rival_conflict = acquire_heavy_work_lease(
        "codex exec review", limit=1, session_key="rival"
    )
    blocked_while_child_alive = rival is None and rival_conflict is not None
    if rival is not None:
        rival.release()
    assert blocked_while_child_alive

    child.wait(timeout=5)
    recovered, recovered_conflict = acquire_heavy_work_lease(
        "codex exec review", limit=1, session_key="recovered"
    )
    assert recovered is not None and recovered_conflict is None
    recovered.release()


def test_disabled_limit_does_not_acquire(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    lease, conflict = acquire_heavy_work_lease("pytest -q", limit=0)
    assert lease is None
    assert conflict is None


def test_lock_metadata_does_not_persist_command_or_inline_secret(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    lease, conflict = acquire_heavy_work_lease(
        "API_TOKEN=do-not-store pytest -q", limit=1, session_key="safe-session"
    )
    assert lease is not None
    assert conflict is None
    payload = json.loads(lease.path.read_text(encoding="utf-8"))
    assert "command" not in payload
    assert "do-not-store" not in lease.path.read_text(encoding="utf-8")
    lease.release()


def test_limit_two_allows_two_then_blocks_third(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    first, _ = acquire_heavy_work_lease("pytest -q", limit=2)
    second, _ = acquire_heavy_work_lease("supabase start", limit=2)
    third, conflict = acquire_heavy_work_lease("codex exec review", limit=2)
    assert first is not None
    assert second is not None
    assert third is None
    assert conflict is not None
    first.release()
    second.release()


def test_terminal_config_bridge_exports_heavy_limit():
    from hermes_cli.config import apply_terminal_config_to_env

    target = {}
    apply_terminal_config_to_env(
        env=target,
        config={"terminal": {"max_concurrent_heavy_jobs": 1}},
        override=True,
    )
    assert target["TERMINAL_MAX_CONCURRENT_HEAVY_JOBS"] == "1"


def test_registry_releases_background_lease_once(monkeypatch):
    from tools.process_registry import ProcessRegistry, ProcessSession

    registry = ProcessRegistry()
    monkeypatch.setattr(registry, "_write_checkpoint", lambda: None)
    lease = _FakeLease()
    session = ProcessSession(
        id="proc_test",
        command="pytest -q",
        task_id="task",
        session_key="session",
        cwd=".",
        started_at=0.0,
        _heavy_work_lease=lease,
    )
    registry._running[session.id] = session

    registry._move_to_finished(session)
    registry._move_to_finished(session)

    assert lease.release_count == 1
    assert session._heavy_work_lease is None


@pytest.mark.skipif(os.name == "nt", reason="POSIX PTY pass_fds semantics")
def test_registry_pty_spawn_inherits_kernel_lock_fd(tmp_path, monkeypatch):
    from ptyprocess import PtyProcess
    from tools.process_registry import ProcessRegistry

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    lease, conflict = acquire_heavy_work_lease("pytest -q", limit=1)
    assert lease is not None and conflict is None
    expected = lease.child_pass_fds()
    seen = {}
    original_spawn = PtyProcess.spawn

    def _spy_spawn(*args, **kwargs):
        seen["pass_fds"] = kwargs.get("pass_fds")
        return original_spawn(*args, **kwargs)

    monkeypatch.setattr(PtyProcess, "spawn", staticmethod(_spy_spawn))
    registry = ProcessRegistry()
    monkeypatch.setattr(registry, "_write_checkpoint", lambda: None)

    session = registry.spawn_local(
        command="printf pty-ok",
        cwd=str(tmp_path),
        task_id="pty-heavy",
        use_pty=True,
        heavy_work_lease=lease,
    )
    finished = registry.wait(session.id, timeout=5)

    assert finished["status"] == "exited"
    assert seen["pass_fds"] == expected


def _terminal_test_config():
    return {
        "env_type": "local",
        "timeout": 180,
        "cwd": "/tmp",
        "host_cwd": None,
        "modal_mode": "auto",
        "docker_image": "",
        "singularity_image": "",
        "modal_image": "",
        "daytona_image": "",
    }


def test_terminal_foreground_releases_lease(monkeypatch):
    from tools import heavy_work_guard
    from tools import terminal_tool as terminal_module

    lease = _FakeLease()
    monkeypatch.setattr(terminal_module, "_get_env_config", _terminal_test_config)
    monkeypatch.setattr(terminal_module, "_start_cleanup_thread", lambda: None)
    monkeypatch.setattr(
        heavy_work_guard,
        "acquire_heavy_work_lease",
        lambda *args, **kwargs: (lease, None),
    )

    result = json.loads(terminal_module.terminal_tool(command="printf ok"))

    assert result["exit_code"] == 0
    assert lease.release_count == 1


@pytest.mark.skipif(os.name == "nt", reason="POSIX FD inheritance semantics")
def test_terminal_foreground_child_inherits_kernel_lock_fd(tmp_path, monkeypatch):
    from tools import heavy_work_guard
    from tools import terminal_tool as terminal_module

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    lease, conflict = acquire_heavy_work_lease("pytest -q", limit=1)
    assert lease is not None and conflict is None
    fd = lease.child_pass_fds()[0]
    monkeypatch.setattr(terminal_module, "_get_env_config", _terminal_test_config)
    monkeypatch.setattr(terminal_module, "_start_cleanup_thread", lambda: None)
    monkeypatch.setattr(
        heavy_work_guard,
        "acquire_heavy_work_lease",
        lambda *args, **kwargs: (lease, None),
    )

    command = (
        f"{sys.executable} -c 'import os; "
        f"print(os.readlink(\"/proc/self/fd/{fd}\"))'"
    )
    result = json.loads(
        terminal_module.terminal_tool(
            command=command,
            task_id="heavy-foreground-inherited-fd",
        )
    )

    assert result["exit_code"] == 0
    assert "heavy-work-0.lock" in result["output"]


def test_terminal_background_holds_lease_until_process_finishes(monkeypatch):
    from tools import heavy_work_guard
    from tools import terminal_tool as terminal_module
    from tools.process_registry import process_registry

    lease = _FakeLease()
    monkeypatch.setattr(terminal_module, "_get_env_config", _terminal_test_config)
    monkeypatch.setattr(terminal_module, "_start_cleanup_thread", lambda: None)
    monkeypatch.setattr(
        heavy_work_guard,
        "acquire_heavy_work_lease",
        lambda *args, **kwargs: (lease, None),
    )

    result = json.loads(
        terminal_module.terminal_tool(command="sleep 0.4", background=True)
    )
    assert result["exit_code"] == 0
    assert lease.release_count == 0

    finished = process_registry.wait(result["session_id"], timeout=5)
    assert finished["status"] == "exited"
    assert lease.release_count == 1


def test_terminal_background_kill_releases_lease(monkeypatch):
    from tools import heavy_work_guard
    from tools import terminal_tool as terminal_module
    from tools.process_registry import process_registry

    lease = _FakeLease()
    monkeypatch.setattr(terminal_module, "_get_env_config", _terminal_test_config)
    monkeypatch.setattr(terminal_module, "_start_cleanup_thread", lambda: None)
    monkeypatch.setattr(
        heavy_work_guard,
        "acquire_heavy_work_lease",
        lambda *args, **kwargs: (lease, None),
    )

    result = json.loads(
        terminal_module.terminal_tool(command="sleep 30", background=True)
    )
    assert lease.release_count == 0
    killed = process_registry.kill_process(result["session_id"], source="test")
    assert killed["status"] == "killed"
    assert lease.release_count == 1


def test_terminal_rejects_second_heavy_job_without_execution(monkeypatch):
    from tools import heavy_work_guard
    from tools import terminal_tool as terminal_module

    monkeypatch.setattr(terminal_module, "_get_env_config", _terminal_test_config)
    monkeypatch.setattr(terminal_module, "_start_cleanup_thread", lambda: None)
    monkeypatch.setattr(
        terminal_module,
        "_create_environment",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("environment must not be created while heavy work is busy")
        ),
    )
    monkeypatch.setattr(
        heavy_work_guard,
        "acquire_heavy_work_lease",
        lambda *args, **kwargs: (
            None,
            HeavyWorkConflict(owner={"category": "test-suite", "pid": 123}),
        ),
    )

    result = json.loads(
        terminal_module.terminal_tool(
            command="printf never-runs",
            task_id="heavy-conflict-before-env",
        )
    )

    assert result["status"] == "busy"
    assert result["exit_code"] == 75
    assert "test-suite" in result["error"]


def test_terminal_rejects_self_detaching_heavy_work_before_execution(monkeypatch):
    from tools import heavy_work_guard
    from tools import terminal_tool as terminal_module

    monkeypatch.setenv("TERMINAL_MAX_CONCURRENT_HEAVY_JOBS", "1")
    monkeypatch.setattr(terminal_module, "_get_env_config", _terminal_test_config)
    monkeypatch.setattr(terminal_module, "_start_cleanup_thread", lambda: None)
    monkeypatch.setattr(
        terminal_module,
        "_create_environment",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("self-detaching heavy work must not create an environment")
        ),
    )
    monkeypatch.setattr(
        heavy_work_guard,
        "acquire_heavy_work_lease",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("self-detaching heavy work must be rejected before acquisition")
        ),
    )

    result = json.loads(
        terminal_module.terminal_tool(
            command="nohup pytest -q >/tmp/t.log 2>&1 & exit",
            task_id="heavy-self-detach",
            background=True,
        )
    )

    assert result["status"] == "unsupported"
    assert result["exit_code"] == 78
    assert result["heavy_work"] is True
    assert "self-detaches" in result["error"]


def test_terminal_fails_closed_for_non_local_heavy_work(monkeypatch):
    from tools import heavy_work_guard
    from tools import terminal_tool as terminal_module

    lease = _FakeLease()
    config = _terminal_test_config()
    config["env_type"] = "docker"
    config["docker_image"] = "unused"
    monkeypatch.setattr(terminal_module, "_get_env_config", lambda: config)
    monkeypatch.setattr(
        terminal_module,
        "_create_environment",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("unsupported guarded backend must not be created")
        ),
    )
    monkeypatch.setattr(
        heavy_work_guard,
        "acquire_heavy_work_lease",
        lambda *args, **kwargs: (lease, None),
    )

    result = json.loads(terminal_module.terminal_tool(command="pytest -q"))

    assert result["status"] == "unsupported"
    assert result["exit_code"] == 78
    assert lease.release_count == 1


def test_terminal_reports_remote_background_failed_start(monkeypatch):
    from tools import heavy_work_guard
    from tools import terminal_tool as terminal_module

    class _FailedEnv:
        cwd = "/tmp"
        cwd_owner = ""

        def execute(self, *args, **kwargs):
            return {"output": "launcher failed", "returncode": 2}

    config = _terminal_test_config()
    config["env_type"] = "docker"
    config["docker_image"] = "unused"
    monkeypatch.setattr(terminal_module, "_get_env_config", lambda: config)
    monkeypatch.setattr(terminal_module, "_start_cleanup_thread", lambda: None)
    monkeypatch.setattr(
        heavy_work_guard,
        "acquire_heavy_work_lease",
        lambda *args, **kwargs: (None, None),
    )
    monkeypatch.setitem(terminal_module._active_environments, "default", _FailedEnv())

    result = json.loads(
        terminal_module.terminal_tool(
            command="printf not-started",
            background=True,
            task_id="remote-failed-start",
        )
    )

    assert result["status"] == "failed_start"
    assert result["exit_code"] == 2
    assert "launcher failed" in result["output"]
