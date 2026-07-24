"""Regression tests for bounded LSP client lifecycle management."""
from __future__ import annotations

import asyncio
import io
import json
import os
import signal
import sys
import time
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

import pytest

from agent.lsp.manager import LSPService
from agent.lsp.servers import SERVERS, ServerContext, ServerDef, SpawnSpec

MOCK_SERVER = str(Path(__file__).parent / "_mock_lsp_server.py")


@pytest.fixture
def mock_pyright(monkeypatch):
    target_index = next(i for i, server in enumerate(SERVERS) if server.server_id == "pyright")
    original = SERVERS[target_index]

    def build_spawn(root: str, ctx: ServerContext) -> SpawnSpec:
        return SpawnSpec(
            command=[sys.executable, MOCK_SERVER],
            workspace_root=root,
            cwd=root,
            env={"MOCK_LSP_SCRIPT": "clean"},
            initialization_options={},
        )

    SERVERS[target_index] = ServerDef(
        server_id="pyright",
        extensions=original.extensions,
        resolve_root=lambda file_path, workspace: workspace,
        build_spawn=build_spawn,
        seed_first_push=False,
        description="mock pyright",
    )
    yield
    SERVERS[target_index] = original


def make_repo(tmp_path: Path, name: str) -> Path:
    repo = tmp_path / name
    repo.mkdir()
    (repo / ".git").mkdir()
    (repo / "pyproject.toml").write_text("")
    (repo / "x.py").write_text("print('ok')\n")
    return repo


def wait_until(predicate, timeout: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


def make_service(**overrides: Any) -> LSPService:
    kwargs: dict[str, Any] = {
        "enabled": True,
        "wait_mode": "document",
        "wait_timeout": 1.0,
        "install_strategy": "manual",
        "idle_timeout": 600.0,
        "reaper_interval": 0.0,
        "max_clients": 2,
    }
    kwargs.update(overrides)
    return LSPService(**kwargs)


def test_periodic_reaper_stops_idle_client(mock_pyright, tmp_path, monkeypatch):
    repo = make_repo(tmp_path, "one")
    monkeypatch.chdir(repo)
    service = make_service(idle_timeout=0.05, reaper_interval=0.01)
    try:
        assert service.get_diagnostics_sync(str(repo / "x.py")) == []
        assert len(service.get_status()["clients"]) == 1
        assert wait_until(lambda: service.get_status()["clients"] == [])
        assert service.get_status()["reaped_total"] == 1
    finally:
        service.shutdown()


def test_max_clients_evicts_inactive_lru(mock_pyright, tmp_path, monkeypatch):
    repos = [make_repo(tmp_path, name) for name in ("one", "two", "three")]
    service = make_service(max_clients=2)
    try:
        for repo in repos:
            monkeypatch.chdir(repo)
            assert service.get_diagnostics_sync(str(repo / "x.py")) == []
        status = service.get_status()
        roots = {client["workspace_root"] for client in status["clients"]}
        assert roots == {str(repos[1]), str(repos[2])}
        assert status["evicted_total"] == 1
    finally:
        service.shutdown()


def test_reaper_never_removes_client_with_active_lease():
    class FakeClient:
        server_id = "typescript"
        workspace_root = "/tmp/repo"
        state = "running"
        is_running = True
        process_id = 123

        def __init__(self):
            self.shutdown_calls = 0

        async def shutdown(self):
            self.shutdown_calls += 1

    service = make_service(enabled=False)
    key = ("typescript", "/tmp/repo")
    client = FakeClient()
    service._clients[key] = client  # type: ignore[assignment]
    service._last_used[key] = time.monotonic() - 999
    service._active_operations[key] = 1

    asyncio.run(service._reap_idle_async(now=time.monotonic()))

    assert service._clients[key] is client
    assert client.shutdown_calls == 0


def test_failed_capacity_reservation_does_not_partially_evict_clients():
    class FakeClient:
        state = "running"
        is_running = True
        process_id = 123

        def __init__(self, server_id: str, workspace_root: str):
            self.server_id = server_id
            self.workspace_root = workspace_root
            self.shutdown_calls = 0

        async def shutdown(self):
            self.shutdown_calls += 1

    async def scenario():
        service = make_service(enabled=False, max_clients=2)
        inactive_key = ("pyright", "/tmp/inactive")
        active_key = ("pyright", "/tmp/active")
        inactive = FakeClient(*inactive_key)
        active = FakeClient(*active_key)
        service._clients[inactive_key] = inactive  # type: ignore[assignment]
        service._clients[active_key] = active  # type: ignore[assignment]
        service._last_used[inactive_key] = 1.0
        service._last_used[active_key] = 2.0
        service._active_operations[active_key] = 1
        loop = asyncio.get_running_loop()
        service._spawning[("pyright", "/tmp/new-a")] = loop.create_future()
        service._spawning[("pyright", "/tmp/new-b")] = loop.create_future()

        assert await service._ensure_capacity_async() is False
        assert set(service._clients) == {inactive_key, active_key}
        assert inactive.shutdown_calls == 0
        assert service.get_status()["evicted_total"] == 0

    asyncio.run(scenario())


def test_config_wires_lifecycle_limits(monkeypatch, tmp_path):
    import hermes_cli.config as config

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    lsp_defaults = config.DEFAULT_CONFIG["lsp"]
    assert isinstance(lsp_defaults, dict)
    assert lsp_defaults["idle_timeout_seconds"] == 600
    assert lsp_defaults["reaper_interval_seconds"] == 30
    assert lsp_defaults["max_clients"] == 2
    monkeypatch.setattr(
        config,
        "load_config",
        lambda: {
            "lsp": {
                "enabled": True,
                "idle_timeout_seconds": 300,
                "reaper_interval_seconds": 30,
                "max_clients": 2,
            }
        },
    )
    service = LSPService.create_from_config()
    assert service is not None
    try:
        status = service.get_status()
        assert status["idle_timeout_seconds"] == 300
        assert status["reaper_interval_seconds"] == 30
        assert status["max_clients"] == 2
    finally:
        service.shutdown()


def test_status_exposes_live_client_runtime_fields(mock_pyright, tmp_path, monkeypatch):
    repo = make_repo(tmp_path, "status")
    monkeypatch.chdir(repo)
    service = make_service()
    try:
        service.get_diagnostics_sync(str(repo / "x.py"))
        client = service.get_status()["clients"][0]
        assert client["pid"] > 0
        assert client["idle_seconds"] >= 0
        assert client["active_operations"] == 0
    finally:
        service.shutdown()


def test_default_client_limit_is_two(monkeypatch, tmp_path):
    import hermes_cli.config as config

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(config, "load_config", lambda: {"lsp": {"enabled": True}})
    service = LSPService.create_from_config()
    assert service is not None
    try:
        assert service.get_status()["max_clients"] == 2
    finally:
        service.shutdown()


def test_shutdown_cancels_and_cleans_spawn_in_progress(tmp_path, monkeypatch):
    target_index = next(i for i, server in enumerate(SERVERS) if server.server_id == "pyright")
    original = SERVERS[target_index]
    repo = make_repo(tmp_path, "slow-spawn")
    pid_file = tmp_path / "server.pid"

    def build_spawn(root: str, ctx: ServerContext) -> SpawnSpec:
        return SpawnSpec(
            command=[sys.executable, MOCK_SERVER],
            workspace_root=root,
            cwd=root,
            env={
                "MOCK_LSP_SCRIPT": "slow",
                "MOCK_LSP_PID_FILE": str(pid_file),
            },
            initialization_options={},
        )

    SERVERS[target_index] = ServerDef(
        server_id="pyright",
        extensions=original.extensions,
        resolve_root=lambda file_path, workspace: workspace,
        build_spawn=build_spawn,
        seed_first_push=False,
        description="slow mock pyright",
    )
    monkeypatch.chdir(repo)
    service = make_service()
    server_pid = None
    spawn = service._loop.submit(service._get_or_spawn(str(repo / "x.py")))
    try:
        assert wait_until(pid_file.exists)
        server_pid = int(pid_file.read_text())
        assert Path(f"/proc/{server_pid}").exists()
        service.shutdown()
        assert wait_until(lambda: not Path(f"/proc/{server_pid}").exists())
        assert spawn.done()
    finally:
        SERVERS[target_index] = original
        service.shutdown()
        if server_pid and Path(f"/proc/{server_pid}").exists():
            try:
                os.killpg(os.getpgid(server_pid), signal.SIGKILL)
            except ProcessLookupError:
                pass


def test_shutdown_kills_descendants_in_lsp_process_group(tmp_path, monkeypatch):
    target_index = next(i for i, server in enumerate(SERVERS) if server.server_id == "pyright")
    original = SERVERS[target_index]
    repo = make_repo(tmp_path, "children")
    pid_file = tmp_path / "child.pid"

    def build_spawn(root: str, ctx: ServerContext) -> SpawnSpec:
        return SpawnSpec(
            command=[sys.executable, MOCK_SERVER],
            workspace_root=root,
            cwd=root,
            env={
                "MOCK_LSP_SCRIPT": "clean",
                "MOCK_LSP_CHILD_PID_FILE": str(pid_file),
                "MOCK_LSP_CHILD_IGNORE_TERM": "1",
            },
            initialization_options={},
        )

    SERVERS[target_index] = ServerDef(
        server_id="pyright",
        extensions=original.extensions,
        resolve_root=lambda file_path, workspace: workspace,
        build_spawn=build_spawn,
        seed_first_push=False,
        description="mock pyright with child",
    )
    monkeypatch.chdir(repo)
    service = make_service()
    child_pid = None
    try:
        service.get_diagnostics_sync(str(repo / "x.py"))
        assert wait_until(pid_file.exists)
        child_pid = int(pid_file.read_text())
        assert Path(f"/proc/{child_pid}").exists()
        service.shutdown()
        assert wait_until(lambda: not Path(f"/proc/{child_pid}").exists())
    finally:
        SERVERS[target_index] = original
        service.shutdown()
        if child_pid and Path(f"/proc/{child_pid}").exists():
            try:
                os.killpg(os.getpgid(child_pid), signal.SIGKILL)
            except ProcessLookupError:
                pass


def test_gateway_marker_is_required_for_runtime_publication(monkeypatch):
    import agent.lsp as lsp_package

    class FakeService:
        @staticmethod
        def is_active():
            return True

    sentinel = FakeService()
    calls = []

    def fake_create_from_config(cls, **kwargs):
        calls.append(kwargs)
        return sentinel

    monkeypatch.setattr(lsp_package, "_service", None)
    monkeypatch.setattr(lsp_package, "_atexit_registered", True)
    monkeypatch.setattr(lsp_package, "_gateway_runtime_publisher", False, raising=False)
    monkeypatch.setattr(
        lsp_package.LSPService,
        "create_from_config",
        classmethod(fake_create_from_config),
    )

    assert lsp_package.get_service() is sentinel
    assert calls == [{}]

    monkeypatch.setattr(lsp_package, "_service", None)
    lsp_package.mark_gateway_process()
    assert lsp_package.get_service() is sentinel
    assert calls[-1] == {"publish_runtime_status": True}


def test_gateway_runtime_status_is_published_and_removed(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    service = make_service(reaper_interval=0.0, publish_runtime_status=True)
    status_path = tmp_path / "runtime" / "lsp-status.json"
    try:
        assert wait_until(status_path.exists)
        payload = json.loads(status_path.read_text(encoding="utf-8"))
        assert payload["publisher_pid"] == os.getpid()
        assert payload["source"] == "gateway-runtime"
        assert payload["max_clients"] == 2
    finally:
        service.shutdown()
    assert not status_path.exists()


def test_runtime_status_removal_preserves_newer_publisher(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    first = make_service(reaper_interval=0.0, publish_runtime_status=True)
    second = make_service(reaper_interval=0.0, publish_runtime_status=True)
    status_path = tmp_path / "runtime" / "lsp-status.json"
    try:
        second_payload = json.loads(status_path.read_text(encoding="utf-8"))
        assert second_payload["publisher_instance_id"] == second._runtime_publisher_instance_id
        assert first._runtime_publisher_instance_id != second._runtime_publisher_instance_id

        first.shutdown()

        surviving = json.loads(status_path.read_text(encoding="utf-8"))
        assert surviving["publisher_instance_id"] == second._runtime_publisher_instance_id
    finally:
        first.shutdown()
        second.shutdown()
    assert not status_path.exists()


def test_cli_status_invalid_lifecycle_config_uses_safe_defaults(monkeypatch, tmp_path):
    import hermes_cli.config as config
    from agent.lsp import cli as lsp_cli

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(
        config,
        "load_config",
        lambda: {
            "lsp": {
                "wait_timeout": "invalid",
                "idle_timeout_seconds": -1,
                "reaper_interval_seconds": "invalid",
                "max_clients": "invalid",
            }
        },
    )
    service_instance = LSPService.create_from_config()
    assert service_instance is not None
    try:
        assert service_instance.get_status()["wait_timeout"] == 5.0
        assert service_instance.get_status()["idle_timeout_seconds"] == 600.0
        assert service_instance.get_status()["reaper_interval_seconds"] == 30.0
        assert service_instance.get_status()["max_clients"] == 2
    finally:
        service_instance.shutdown()

    buffer = io.StringIO()
    with redirect_stdout(buffer):
        assert lsp_cli._cmd_status(emit_json=True) == 0
    service = json.loads(buffer.getvalue())["service"]
    assert service["wait_timeout"] == 5.0
    assert service["idle_timeout_seconds"] == 600.0
    assert service["reaper_interval_seconds"] == 30.0
    assert service["max_clients"] == 2


def test_cli_status_non_mapping_config_uses_safe_defaults(monkeypatch, tmp_path):
    import hermes_cli.config as config
    from agent.lsp import cli as lsp_cli

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    for invalid in ({"lsp": "invalid"}, "invalid"):
        monkeypatch.setattr(config, "load_config", lambda value=invalid: value)
        status = lsp_cli._configured_status()
        assert status["idle_timeout_seconds"] == 600.0
        assert status["reaper_interval_seconds"] == 30.0
        assert status["max_clients"] == 2


def test_cli_status_reads_gateway_snapshot_without_local_singleton(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    payload = {
        "source": "gateway-runtime",
        "publisher_pid": os.getpid(),
        "updated_at_epoch": time.time(),
        "enabled": True,
        "wait_mode": "document",
        "wait_timeout": 5.0,
        "install_strategy": "manual",
        "idle_timeout_seconds": 300,
        "reaper_interval_seconds": 30,
        "max_clients": 2,
        "reaped_total": 4,
        "evicted_total": 1,
        "clients": [],
        "broken": [],
        "disabled_servers": [],
    }
    (runtime / "lsp-status.json").write_text(json.dumps(payload), encoding="utf-8")

    from agent.lsp import cli as lsp_cli
    import agent.lsp as lsp_package

    monkeypatch.setattr(
        lsp_package,
        "get_service",
        lambda: (_ for _ in ()).throw(AssertionError("must not create local singleton")),
    )
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        lsp_cli._cmd_status(emit_json=True)
    rendered = json.loads(buffer.getvalue())
    assert rendered["service"]["source"] == "gateway-runtime"
    assert rendered["service"]["publisher_pid"] == os.getpid()
