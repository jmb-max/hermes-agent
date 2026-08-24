"""Tests for ``plugins/claude-worker/oauth_refresh.py`` — the isolated refresh
probe that delegates credential refresh to the trusted host Claude CLI.

RED phase: no production ``oauth_refresh`` module exists yet, so every case
here fails until it does. The tests ARE the specification, so the contract the
implementation has to satisfy is written out here in full.

Why the probe lives outside ``oauth.py``
---------------------------------------
``oauth.py`` is tripwired against ever gaining a write path, a subprocess, or
network egress (``test_claude_worker_oauth.py::
TestDefaultsAndInvariants::test_the_module_never_writes_the_credentials_file``),
and that tripwire is load-bearing: it is what makes "this plugin never owns the
privileged credential write" a structural fact rather than a promise. Refresh
therefore cannot be implemented there, and it is not reimplemented anywhere —
it is DELEGATED to the one program that legitimately owns the write and knows
the grant parameters: the host's own Claude CLI. ``oauth.set_refresh_probe`` is
the seam that already exists for exactly this, and ``oauth.preflight`` remains
the authority: after the probe runs it re-reads the credential from disk and
believes only what it finds there.

The contract under test
-----------------------
Module ``oauth_refresh`` exposes:

* ``HOST_CLAUDE_CLI``            — fixed ABSOLUTE path of the host Claude CLI:
                                   exactly ``HOST_CLAUDE_EXECUTABLE`` below,
                                   the packaged native binary (basename
                                   ``claude.exe``, NOT the ``claude`` wrapper
                                   name on ``$PATH``); never a bare name
                                   resolved off ``$PATH``.
* ``REFRESH_LOCK_PATH``          — fixed absolute lock path under the
                                   root-owned mode-0700
                                   ``policy.CREDENTIAL_STAGING_ROOT``.
* ``REFRESH_LOCK_TIMEOUT_SECONDS``, ``REFRESH_PROMPT``, ``REFRESH_MODEL``.
* ``REASON_*`` literals and ``REASONS`` — the CLOSED set of reason strings the
  probe may ever return. Fixed, non-secret, no interpolation.
* ``RefreshLockUnavailable`` — raised by the lock when it cannot be taken.
* ``build_refresh_argv() -> list[str]`` — takes no arguments, by construction.
* ``build_refresh_env() -> dict[str, str]`` — built from literals, never from
  the ambient environment.
* ``refresh_lock()`` — context manager taking the isolated lock; looked up as a
  module attribute at call time so tests can substitute it.
* ``_validate_staging_dir(directory)`` — raises ``trust.TrustViolation`` unless
  *directory* is an existing, non-symlink, real directory owned by
  ``trust.TRUSTED_UID`` with no group or world bits at all, every parent
  likewise trusted. Guards the lock directory, the staging root, and the
  throwaway cwd — none of which may be taken on faith from
  ``makedirs(exist_ok=True)``.
* ``REFRESH_KILL_GRACE_SECONDS`` — the fixed, short grace between the group
  SIGTERM and the group SIGKILL of a timed-out refresh turn.
* ``_run_refresh(argv, *, env, cwd, timeout_seconds)`` — the ONE subprocess
  seam, likewise looked up at call time. Creates the child with
  ``start_new_session=True`` so a timeout can reap the whole process group
  rather than leaving the CLI's descendants orphaned and running.
* ``refresh_probe(now: float | None = None) -> {"ok": bool, "reason": str}`` —
  the zero-argument-callable probe ``oauth.preflight`` invokes.
* ``install_refresh_probe(cfg: dict | None = None) -> bool`` — installs
  ``refresh_probe`` (or ``None`` when the knob is off) via
  ``oauth.set_refresh_probe``; what ``register(ctx)`` calls.

``refresh_probe`` does, in this order: read the validated config knob;
trust-validate the fixed CLI path and RETAIN the snapshot; take the isolated
lock; RE-READ freshness inside the lock and return success without spawning if
another process already refreshed; re-validate the retained snapshot
immediately before process creation; otherwise run the host CLI exactly once
with a fixed argv, a minimal env, a throwaway non-project cwd inside the
root-owned staging root, and the bounded timeout; and finally — still holding
the lock — RE-READ the credential and report success only if it is now
``STATE_FRESH``. A clean exit over an untouched credential is not a refresh.
It never parses, writes, or returns credential material, and no
stdout/stderr/exception text ever reaches its result or the log.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import shutil
import signal
import stat
import subprocess
import tempfile
import threading
import time

import pytest

from tests.plugins._claude_worker_helpers import load_submodule, write_credentials

oauth = load_submodule("oauth")
policy = load_submodule("policy")
config = load_submodule("config")
trust = load_submodule("trust")

#: A fixed instant every freshness assertion is computed against.
NOW = 1_700_000_000.0

ACCESS = "sk-ant-oat01-AAAABBBBCCCCDDDDEEEEFFFF-accesstoken"
REFRESH = "sk-ant-ort01-1111222233334444555566667777-refreshtoken"

#: Anything that could redirect, impersonate, or hijack the refresh. None of
#: these may reach the child no matter what the ambient environment holds.
HOSTILE_ENV = {
    "ANTHROPIC_API_KEY": "sk-ant-api-hostile",
    "ANTHROPIC_AUTH_TOKEN": "hostile-auth-token",
    "ANTHROPIC_BASE_URL": "http://127.0.0.1:9/",
    "ANTHROPIC_MODEL": "evil-model",
    "CLAUDE_CODE_OAUTH_TOKEN": "hostile-oauth-token",
    "CLAUDE_CODE_USE_BEDROCK": "1",
    "CLAUDE_CONFIG_DIR": "/tmp/hostile-config",
    "HTTP_PROXY": "http://127.0.0.1:9/",
    "http_proxy": "http://127.0.0.1:9/",
    "HTTPS_PROXY": "http://127.0.0.1:9/",
    "https_proxy": "http://127.0.0.1:9/",
    "ALL_PROXY": "socks5://127.0.0.1:9",
    "all_proxy": "socks5://127.0.0.1:9",
    "NO_PROXY": "example.invalid",
    "NODE_OPTIONS": "--require /tmp/pwn.js",
    "NODE_EXTRA_CA_CERTS": "/tmp/pwn.pem",
    "NPM_CONFIG_PREFIX": "/tmp/pwn",
    "LD_PRELOAD": "/tmp/pwn.so",
    "LD_LIBRARY_PATH": "/tmp/pwn",
    "PYTHONPATH": "/tmp/pwn",
    "BASH_ENV": "/tmp/pwn.sh",
    "SSL_CERT_FILE": "/tmp/pwn.pem",
    "REQUESTS_CA_BUNDLE": "/tmp/pwn.pem",
    "GIT_SSH_COMMAND": "/tmp/pwn.sh",
}

#: THIS deployment's real Claude Code CLI: the root-owned, non-symlink,
#: mode-0755 executable the global npm install leaves inside the package
#: itself, and the only claude binary on this host that
#: ``trust.validate_trusted_path_chain`` can accept. Note the basename: the
#: packaged program is ``claude.exe``, not ``claude`` — ``claude`` is only the
#: wrapper name npm exposes on ``$PATH``.
HOST_CLAUDE_EXECUTABLE = (
    "/usr/lib/node_modules/@anthropic-ai/claude-code/bin/claude.exe"
)

#: The basename that goes with it. Deliberately NOT
#: ``policy.CLAUDE_CLI_BASENAME``: that constant names the ``claude``
#: invocation token ``terminal_guard`` matches on, which is a different thing
#: from the on-disk executable this probe must exec.
HOST_CLAUDE_EXECUTABLE_BASENAME = "claude.exe"

#: The npm bin SYMLINK on this host — the convenience path on ``$PATH``, which
#: points at ``HOST_CLAUDE_EXECUTABLE``. Symlinks are refused by design, so a
#: probe pointed here fails closed on every call and can never refresh
#: anything.
NPM_BIN_SYMLINK = "/usr/bin/claude"

#: The unusable literal this regression exists to keep out: production's
#: current ``HOST_CLAUDE_CLI``, which does not exist on this host at all —
#: neither as a binary nor as a symlink — so every refresh fails closed.
ABSENT_PRODUCTION_LITERAL = "/usr/local/bin/claude"

#: Every ``REASON_*`` constant the probe must expose. Names are pinned (the
#: code paths below map onto them); the wording is the implementation's.
REASON_NAMES = (
    "REASON_REFRESHED",
    "REASON_ALREADY_FRESH",
    "REASON_DISABLED",
    "REASON_UNTRUSTED_CLI",
    "REASON_LOCK_BUSY",
    "REASON_TIMEOUT",
    "REASON_CLI_FAILED",
    "REASON_MALFORMED",
    "REASON_ERROR",
)

#: The two reasons that mean "the credential is usable now". Everything else in
#: ``REASONS`` is a HOLD.
SUCCESS_REASON_NAMES = ("REASON_REFRESHED", "REASON_ALREADY_FRESH")

#: The pid of the stand-in child in ``fake_child``. Never a real process — with
#: ``start_new_session=True`` the child is its own group leader, so this doubles
#: as the process-group id the timeout path must signal.
FAKE_CHILD_PID = 987654


# ---------------------------------------------------------------------------
# Fixtures — every seam the probe touches is injected, nothing real is run
# ---------------------------------------------------------------------------


@pytest.fixture
def refresh():
    """The module under test. Absent in the RED phase, which is the point."""
    return load_submodule("oauth_refresh")


@pytest.fixture(autouse=True)
def _no_ambient_state(monkeypatch):
    """No probe leaks out of this file, and no test reads the real config.yaml
    (the knob is read per probe call, so an operator's file would otherwise
    decide these outcomes)."""
    monkeypatch.setattr(oauth, "REFRESH_PROBE", None)
    monkeypatch.setattr(config, "_load_raw_config", lambda: {})


def _cfg(**oauth_entry):
    return {"plugins": {"entries": {"claude-worker": {"oauth": dict(oauth_entry)}}}}


@pytest.fixture
def creds(tmp_path, monkeypatch):
    """A refreshable-expired host credential at the fixed host path."""
    path = tmp_path / "root" / ".claude" / ".credentials.json"
    write_credentials(
        path, access_token=ACCESS, refresh_token=REFRESH,
        expires_at=(NOW - 10) * 1000,
    )
    monkeypatch.setattr(policy, "HOST_CREDENTIALS_PATH", str(path))
    return path


@pytest.fixture
def trusted(monkeypatch):
    """Neutralize (and record) trusted-path validation.

    The fixed host CLI path is root-owned in production and is not present at
    all in a test environment, so without this every case would short-circuit
    on ``REASON_UNTRUSTED_CLI``. Trust behavior itself is asserted separately.
    """
    calls = []

    def _validate(path, *, executable=False):
        calls.append({"path": path, "executable": executable})
        return {}

    monkeypatch.setattr(trust, "validate_trusted_path_chain", _validate)
    return calls


@pytest.fixture
def staging_at_tmp(refresh, tmp_path, monkeypatch):
    """Point the root-owned staging tree — which now holds the throwaway cwd
    as well as the lock — at a writable temporary directory, and neutralize
    the trust check that guards it.

    Neutralizing that check is not a hole: a test process is not root, so the
    genuine check could only ever refuse here, and every case would then fail
    for a reason that has nothing to do with what it is about. The check's own
    behavior is asserted directly in
    ``TestTheStagingDirectoryChainIsValidated``. Returns the root.
    """
    root = tmp_path / "staging"
    root.mkdir(parents=True, exist_ok=True)
    os.chmod(str(root), 0o700)
    monkeypatch.setattr(policy, "CREDENTIAL_STAGING_ROOT", str(root))
    monkeypatch.setattr(refresh, "_validate_staging_dir", lambda *a, **k: None, raising=False)
    return root


@pytest.fixture
def lock_at_tmp(refresh, staging_at_tmp, monkeypatch):
    """Point the isolated lock inside that temporary staging root so the REAL
    lock implementation runs (production's ``/run/hermes-claude-worker``
    neither exists nor would be ours in a test environment). Returns the lock
    path."""
    path = staging_at_tmp / "oauth-refresh.lock"
    monkeypatch.setattr(refresh, "REFRESH_LOCK_PATH", str(path))
    return path


def _completed(returncode=0, stdout="", stderr=""):
    return subprocess.CompletedProcess(
        args=["claude"], returncode=returncode, stdout=stdout, stderr=stderr,
    )


@contextlib.contextmanager
def _null_lock():
    """A lock that is always free — for cases whose subject is what happens
    INSIDE it rather than how it is taken."""
    yield


@pytest.fixture
def spawns(refresh, monkeypatch):
    """Record every host-CLI invocation and exit 0 WITHOUT touching the
    credential.

    Deliberately not a "success" fixture any more: exit 0 over an untouched
    credential is exactly the case ``TestExitZeroIsNotRefreshSuccess`` says is
    not a refresh. Tests about the happy path use ``spawns_that_refresh``.
    """
    calls = []

    def _fake(argv, *, env=None, cwd=None, timeout_seconds=None):
        live = bool(cwd) and os.path.isdir(cwd)
        calls.append({
            "argv": list(argv),
            "env": dict(env or {}),
            "cwd": cwd,
            "timeout_seconds": timeout_seconds,
            "cwd_is_dir": live,
            "cwd_entries": sorted(os.listdir(cwd)) if live else None,
            "cwd_mode": stat.S_IMODE(os.lstat(cwd).st_mode) if live else None,
            "cwd_is_symlink": (
                stat.S_ISLNK(os.lstat(cwd).st_mode)
                if cwd and os.path.lexists(cwd) else None
            ),
        })
        return _completed(0)

    monkeypatch.setattr(refresh, "_run_refresh", _fake)
    return calls


@pytest.fixture
def spawns_that_refresh(refresh, creds, monkeypatch):
    """The only shape a real refresh has: the host CLI — which OWNS the
    privileged write — rewrites the credential itself, and only then exits 0.

    Every test that is about the happy path has to simulate that write, because
    the probe now confirms it by re-reading the file under its own lock instead
    of believing the exit status.
    """
    calls = []

    def _fake(argv, *, env=None, cwd=None, timeout_seconds=None):
        calls.append({
            "argv": list(argv),
            "env": dict(env or {}),
            "cwd": cwd,
            "timeout_seconds": timeout_seconds,
        })
        write_credentials(
            creds, access_token=ACCESS, refresh_token=REFRESH,
            expires_at=(NOW + 3600) * 1000,
        )
        return _completed(0)

    monkeypatch.setattr(refresh, "_run_refresh", _fake)
    return calls


@pytest.fixture
def staging_chain(monkeypatch):
    """Let the staging checks run for real against a ``tmp_path`` directory.

    Two things are in the way otherwise: ``tmp_path``'s ANCESTORS are not
    root-owned (and ``/tmp`` itself is not ours), and neither is the directory
    under test. So this treats the test process's own uid as the trusted one,
    and records-but-permits every ancestor while leaving the ONE directory a
    case is about fully enforced — including when the implementation validates
    it through the same shared ``trust`` primitive.

    ``chain["focus"](path)`` names that directory; ``chain["checked"]`` is
    every component the implementation actually looked at.
    """
    real_check = trust.check_dir_component
    state = {"subject": None, "checked": []}

    def _check(path):
        state["checked"].append(path)
        if state["subject"] is not None and os.path.normpath(path) == state["subject"]:
            return real_check(path)
        return os.lstat(path)

    def _focus(path):
        state["subject"] = os.path.normpath(str(path))
        return state["subject"]

    monkeypatch.setattr(trust, "TRUSTED_UID", os.getuid())
    monkeypatch.setattr(trust, "check_dir_component", _check)
    state["focus"] = _focus
    return state


def _staging_shape(tmp_path, shape):
    """Build one staging-root shape and return its path. ``"private"`` is the
    only one a refresh may ever run under."""
    path = tmp_path / f"staging-{shape}"
    if shape == "missing":
        return str(path)
    if shape == "symlink":
        target = tmp_path / "staging-symlink-target"
        target.mkdir()
        os.chmod(str(target), 0o700)
        path.symlink_to(target, target_is_directory=True)
        return str(path)
    if shape == "regular_file":
        path.write_text("not a directory", encoding="utf-8")
        return str(path)
    path.mkdir()
    os.chmod(str(path), {
        "private": 0o700,
        "wrong_owner": 0o700,
        "world_writable": 0o777,
        "group_writable": 0o770,
        "world_readable": 0o755,
    }[shape])
    return str(path)


@pytest.fixture
def trust_events(refresh, monkeypatch, spawns):
    """Record BOTH halves of the trusted-path protocol — the validation and
    the re-validation done immediately before process creation — together with
    the spawn count at each, so what is asserted is the ORDER, not merely that
    both happened."""
    snapshot = {
        "/": (1, 2, 0o40755, 0, 0),
        refresh.HOST_CLAUDE_CLI: (1, 4242, 0o100755, 0, 0),
    }
    events = []

    def _validate(path, *, executable=False):
        events.append({
            "event": "validate", "path": path, "executable": executable,
            "spawns": len(spawns),
        })
        return dict(snapshot)

    def _revalidate(seen):
        events.append({
            "event": "revalidate", "snapshot": dict(seen or {}),
            "spawns": len(spawns),
        })

    monkeypatch.setattr(trust, "validate_trusted_path_chain", _validate)
    monkeypatch.setattr(trust, "revalidate_trusted_path_chain", _revalidate)
    return {"snapshot": snapshot, "events": events}


@pytest.fixture
def fake_child(monkeypatch):
    """A scriptable stand-in for the host CLI process.

    No real process is ever created, but the seam production creates one
    THROUGH is exercised for real, so ``start_new_session`` and the signals a
    timeout sends are directly observable. ``state["hangs"]`` makes every wait
    raise ``TimeoutExpired`` until the child's GROUP is signalled;
    ``state["dies_on_term"]`` decides whether SIGTERM alone is enough to reap
    it, which is what separates "SIGTERM" from "SIGTERM then SIGKILL".
    """
    state = {
        "children": [], "signals": [], "events": [],
        "exit_code": 0, "hangs": False, "dies_on_term": True,
    }
    real_rmtree = shutil.rmtree

    class _Child:
        def __init__(self, args=None, **kwargs):
            self.argv = list(args if args is not None else kwargs.get("args") or [])
            self.args = self.argv  # what ``subprocess`` itself reads back
            self.kwargs = dict(kwargs)
            self.pid = FAKE_CHILD_PID
            self.returncode = None
            self.alive = True
            state["children"].append(self)

        def _settle(self, timeout=None):
            if state["hangs"] and self.alive:
                raise subprocess.TimeoutExpired(cmd=self.argv, timeout=timeout or 0)
            if self.returncode is None:
                self.returncode = state["exit_code"] if self.alive else -signal.SIGKILL
            self.alive = False
            return self.returncode

        def communicate(self, input=None, timeout=None):
            self._settle(timeout)
            return (None, None)

        def wait(self, timeout=None):
            return self._settle(timeout)

        def poll(self):
            return self.returncode

        def terminate(self):
            state["signals"].append(("direct", signal.SIGTERM))

        def kill(self):
            state["signals"].append(("direct", signal.SIGKILL))
            self.alive = False

        def __enter__(self):
            return self

        def __exit__(self, *exc_info):
            return False

    def _killpg(pgid, sig):
        state["signals"].append((pgid, sig))
        state["events"].append(("signal", sig))
        if sig == signal.SIGKILL or (sig == signal.SIGTERM and state["dies_on_term"]):
            for child in state["children"]:
                child.alive = False

    def _rmtree(path, *args, **kwargs):
        state["events"].append(("rmtree", path))
        return real_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(subprocess, "Popen", _Child)
    monkeypatch.setattr(os, "killpg", _killpg)
    monkeypatch.setattr(os, "getpgid", lambda pid: pid)
    monkeypatch.setattr(shutil, "rmtree", _rmtree)
    return state


def _spawn_returning(refresh, monkeypatch, calls, outcome):
    """Replace the subprocess seam with one that returns/raises *outcome*."""

    def _fake(argv, *, env=None, cwd=None, timeout_seconds=None):
        calls.append({"argv": list(argv), "cwd": cwd, "timeout_seconds": timeout_seconds})
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    monkeypatch.setattr(refresh, "_run_refresh", _fake)


def _value_after(argv, flag):
    """The single value following *flag* in *argv*."""
    assert argv.count(flag) == 1, f"{flag!r} must appear exactly once in {argv!r}"
    return argv[argv.index(flag) + 1]


# ---------------------------------------------------------------------------
# The fixed literals
# ---------------------------------------------------------------------------


class TestFixedLiterals:
    def test_the_host_cli_is_a_fixed_absolute_path(self, refresh):
        """A bare ``claude`` would let any earlier ``$PATH`` entry substitute
        an attacker's binary for the program we are about to hand the host's
        refresh token to.

        The path is pinned to the ONE canonical literal — no prefix or
        substring latitude, which would let ``…/bin/claude.exe`` be swapped for
        some other ``…/claude.exe`` — and its basename is pinned to the
        packaged executable's real name, ``claude.exe``.
        """
        assert refresh.HOST_CLAUDE_CLI == HOST_CLAUDE_EXECUTABLE
        assert os.path.isabs(refresh.HOST_CLAUDE_CLI)
        assert (
            os.path.basename(refresh.HOST_CLAUDE_CLI)
            == HOST_CLAUDE_EXECUTABLE_BASENAME
            == "claude.exe"
        )
        assert refresh.HOST_CLAUDE_CLI != policy.CLAUDE_CLI_BASENAME
        assert refresh.HOST_CLAUDE_CLI != HOST_CLAUDE_EXECUTABLE_BASENAME
        assert "$" not in refresh.HOST_CLAUDE_CLI
        assert ".." not in refresh.HOST_CLAUDE_CLI.split(os.sep)

    def test_the_lock_is_isolated_in_the_root_owned_staging_root(self, refresh):
        """A lock in a world-writable directory is not a lock: any local user
        could hold it forever (stampede back on) or replace it."""
        assert os.path.isabs(refresh.REFRESH_LOCK_PATH)
        assert refresh.REFRESH_LOCK_PATH.startswith(
            policy.CREDENTIAL_STAGING_ROOT.rstrip("/") + "/"
        )
        assert not refresh.REFRESH_LOCK_PATH.startswith("/tmp/")
        assert refresh.REFRESH_LOCK_PATH != policy.HOST_CREDENTIALS_PATH
        assert 0 < float(refresh.REFRESH_LOCK_TIMEOUT_SECONDS) <= 120

    def test_the_refresh_prompt_is_a_fixed_low_cost_literal(self, refresh):
        """One cheap turn whose only job is to make the CLI use — and so
        refresh — its own credential. Nothing caller-shaped may be in it."""
        assert isinstance(refresh.REFRESH_PROMPT, str)
        assert refresh.REFRESH_PROMPT.strip()
        assert len(refresh.REFRESH_PROMPT) <= 200
        for placeholder in ("{", "}", "%s", "%("):
            assert placeholder not in refresh.REFRESH_PROMPT

    def test_the_refresh_model_is_the_cheap_default_never_the_escalation(self, refresh):
        assert refresh.REFRESH_MODEL == policy.DEFAULT_MODEL == "claude-sonnet-5"
        assert refresh.REFRESH_MODEL != policy.ESCALATION_MODEL

    def test_every_reason_is_a_closed_fixed_non_secret_literal(self, refresh):
        values = []
        for name in REASON_NAMES:
            value = getattr(refresh, name)
            assert isinstance(value, str) and value.strip(), name
            values.append(value)

        # The pinned names must all be there and all be reachable. The set may
        # be LARGER — confirming a refresh against the credential itself adds a
        # failure mode ("the run finished but nothing was renewed") that
        # deserves its own literal rather than being folded into an existing
        # one — but it stays closed: every member is checked below.
        assert refresh.REASONS >= frozenset(values)
        assert isinstance(refresh.REASONS, frozenset)
        for value in refresh.REASONS:
            assert isinstance(value, str) and value.strip()
            for placeholder in ("{", "}", "%s", "%("):
                assert placeholder not in value, value
            assert "sk-ant" not in value
        # Distinct reasons, so an operator can tell the failure modes apart.
        assert len(set(values)) == len(values)


class TestTheHostCliIsThisHostsRealExecutable:
    """The literal above is absolute and well-shaped — and still names a file
    this host cannot run.

    ``TestFixedLiterals`` only checks the SHAPE of ``HOST_CLAUDE_CLI``, so a
    path that is absolute, correctly-based, and simply wrong passes it. On this
    deployment the wrong path costs the whole feature: the probe validates the
    binary immediately before every run, so a CLI path that cannot survive
    validation turns every refresh into ``REASON_UNTRUSTED_CLI`` — silently,
    since the probe never reports which path failed. These tests run the REAL
    validator against the REAL filesystem (no ``trusted`` fixture) because the
    identity of the host binary is exactly what is under test.
    """

    def test_the_configured_cli_is_this_hosts_real_claude_executable(self, refresh):
        assert refresh.HOST_CLAUDE_CLI == HOST_CLAUDE_EXECUTABLE
        assert refresh.HOST_CLAUDE_CLI != NPM_BIN_SYMLINK
        assert refresh.HOST_CLAUDE_CLI != ABSENT_PRODUCTION_LITERAL

    def test_the_configured_cli_is_a_root_owned_non_symlink_executable(self, refresh):
        """Every property ``trust`` demands of a final component, asserted
        directly, so a failure names the property rather than just the
        exception."""
        st = os.lstat(refresh.HOST_CLAUDE_CLI)

        assert not stat.S_ISLNK(st.st_mode)
        assert stat.S_ISREG(st.st_mode)
        assert st.st_uid == trust.TRUSTED_UID == 0
        assert st.st_mode & stat.S_IXUSR
        assert not st.st_mode & (stat.S_IWGRP | stat.S_IWOTH)

    def test_the_configured_cli_passes_real_trusted_path_validation(self, refresh):
        """The whole chain, not just the final component: one user-owned or
        world-writable parent directory would fail the probe just as totally."""
        snapshot = trust.validate_trusted_path_chain(
            refresh.HOST_CLAUDE_CLI, executable=True,
        )

        assert refresh.HOST_CLAUDE_CLI in snapshot
        assert set(snapshot) >= set(trust.parent_dirs(refresh.HOST_CLAUDE_CLI))

    def test_the_npm_bin_symlink_is_refused_by_the_unweakened_validator(self):
        """Why the literal has to change, pinned as a fact about this host: the
        convenience path npm leaves on ``$PATH`` is a symlink onto the packaged
        binary, and validation refuses symlinks — the fix is to name the real
        target, never to relax this check."""
        assert stat.S_ISLNK(os.lstat(NPM_BIN_SYMLINK).st_mode)
        assert os.path.realpath(NPM_BIN_SYMLINK) == HOST_CLAUDE_EXECUTABLE

        with pytest.raises(trust.TrustViolation):
            trust.validate_trusted_path_chain(NPM_BIN_SYMLINK, executable=True)

    def test_the_literal_production_carries_is_not_on_this_host_at_all(self):
        """The other half of the same fact: production's current
        ``/usr/local/bin/claude`` is not merely a symlink here — nothing exists
        at that path, so validation fails on the missing final component and
        every refresh returns ``REASON_UNTRUSTED_CLI``."""
        assert not os.path.lexists(ABSENT_PRODUCTION_LITERAL)

        with pytest.raises(trust.TrustViolation):
            trust.validate_trusted_path_chain(
                ABSENT_PRODUCTION_LITERAL, executable=True,
            )


# ---------------------------------------------------------------------------
# argv — fixed, and nothing caller-controlled can enter it
# ---------------------------------------------------------------------------


class TestRefreshArgv:
    def test_the_argv_builder_accepts_no_input_at_all(self, refresh):
        """Structural, not incidental: with no parameters there is no task, no
        path, no model, and no env for a caller to smuggle into argv."""
        import inspect

        signature = inspect.signature(refresh.build_refresh_argv)
        assert list(signature.parameters) == []

    def test_the_argv_is_deterministic(self, refresh):
        first = refresh.build_refresh_argv()
        second = refresh.build_refresh_argv()
        assert first == second
        assert all(isinstance(item, str) for item in first)
        assert all("\x00" not in item for item in first)

    def test_the_argv_runs_exactly_the_fixed_trusted_binary(self, refresh):
        argv = refresh.build_refresh_argv()
        assert argv[0] == refresh.HOST_CLAUDE_CLI
        assert argv.count(refresh.HOST_CLAUDE_CLI) == 1
        # No shell, ever — and nothing that would reintroduce one.
        for shellish in ("sh", "-c", "bash", "/bin/sh", "/bin/bash", "env"):
            assert shellish not in argv[1:]

    def test_the_argv_pins_the_cheap_model(self, refresh):
        argv = refresh.build_refresh_argv()
        assert _value_after(argv, "--model") == refresh.REFRESH_MODEL
        assert policy.ESCALATION_MODEL not in argv

    def test_the_argv_uses_an_empty_strict_mcp_config(self, refresh):
        argv = refresh.build_refresh_argv()
        assert "--strict-mcp-config" in argv
        assert json.loads(_value_after(argv, "--mcp-config")) == {"mcpServers": {}}

    def test_the_argv_caps_the_turn_and_prints(self, refresh):
        argv = refresh.build_refresh_argv()
        assert _value_after(argv, "--max-turns") == "1"
        assert "--print" in argv

    def test_the_argv_persists_no_session(self, refresh):
        """A refresh must leave no session, history, or settings behind for a
        later run to inherit."""
        argv = refresh.build_refresh_argv()
        assert _value_after(argv, "--setting-sources") == ""
        for flag in ("--continue", "-c", "--resume", "--session-id", "--fork-session"):
            assert flag not in argv

    def test_the_fixed_prompt_is_the_only_positional(self, refresh):
        argv = refresh.build_refresh_argv()
        assert argv.count(refresh.REFRESH_PROMPT) == 1
        assert argv[-1] == refresh.REFRESH_PROMPT

    def test_no_host_or_caller_path_reaches_the_argv(self, refresh, tmp_path, monkeypatch):
        """Even the credential path — the one path this module knows — stays
        out of argv: the CLI finds its own credential through its own config
        directory, not through an argument we hand it."""
        monkeypatch.setattr(policy, "HOST_CREDENTIALS_PATH", str(tmp_path / "creds.json"))
        argv = refresh.build_refresh_argv()
        assert str(tmp_path) not in " ".join(argv)
        assert policy.HOST_CREDENTIALS_PATH not in argv


class TestNoToolIsEverGranted:
    """A refresh turn needs no capability whatsoever — and "no capability" has
    to be spelled the way Claude Code's permission model actually reads.

    ``--allowed-tools ""`` is not a denial. In Claude Code's semantics the
    allow list only ADDS permissions; it never subtracts one, so an empty
    allow list grants nothing while ALSO forbidding nothing — every default
    tool remains reachable, and the flag reads as a guarantee it does not
    make. The enforcing half is the deny rule, and the mode that decides what
    happens to anything not covered by a rule. So the refresh turn is denied
    by wildcard on the command line AND in the settings payload, and runs in a
    permission mode that neither prompts (there is no one to answer: stdin is
    ``/dev/null``) nor bypasses the rules it just installed.
    """

    def test_the_argv_denies_every_tool_by_wildcard(self, refresh):
        argv = refresh.build_refresh_argv()

        assert _value_after(argv, "--disallowed-tools") == "*"
        # ``--tools ""`` is the other half: it disables every built-in tool, so
        # the turn starts with nothing to deny in the first place.
        assert _value_after(argv, "--tools") == ""
        # No allow-list claim at all: an empty one denies nothing, and saying
        # it here would read as an enforcement that is not happening.
        assert "--allowed-tools" not in argv
        for tool in policy.CLAUDE_ALLOWED_TOOLS + policy.CLAUDE_DENIED_TOOLS:
            assert tool not in argv
        assert "--dangerously-skip-permissions" not in argv

    def test_the_settings_payload_denies_every_tool(self, refresh):
        """Belt and braces, and not redundant: ``--settings`` is the file-shaped
        control plane the CLI merges its rules from, so a deny that lives only
        in argv is one flag-name change away from silently granting the world."""
        argv = refresh.build_refresh_argv()
        settings = json.loads(_value_after(argv, "--settings"))

        assert settings["permissions"]["deny"] == ["*"]
        assert not settings["permissions"].get("allow")
        assert not settings["permissions"].get("ask")
        # The rest of the isolation is unchanged by the deny rule.
        assert "--strict-mcp-config" in argv
        assert json.loads(_value_after(argv, "--mcp-config")) == {"mcpServers": {}}
        assert _value_after(argv, "--setting-sources") == ""

    def test_the_permission_mode_neither_asks_nor_bypasses(self, refresh):
        """``acceptEdits`` would auto-approve the very tools this turn must
        never get, and ``bypassPermissions`` would ignore the deny rule
        outright. The turn runs non-interactively against ``/dev/null`` stdin,
        so the only correct mode is the one that never prompts and still
        enforces."""
        argv = refresh.build_refresh_argv()
        mode = _value_after(argv, "--permission-mode")

        assert mode == "dontAsk"
        assert mode != policy.PERMISSION_MODE
        assert "bypassPermissions" not in argv
        assert "acceptEdits" not in argv
        settings = json.loads(_value_after(argv, "--settings"))
        assert settings["permissions"].get("defaultMode", mode) == mode


# ---------------------------------------------------------------------------
# env — minimal, sanitized, and never inherited
# ---------------------------------------------------------------------------


class TestRefreshEnv:
    ALLOWED_KEYS = {"HOME", "PATH", "CLAUDE_CONFIG_DIR", "LANG"}

    def test_the_env_is_minimal_and_deterministic(self, refresh):
        env = refresh.build_refresh_env()
        assert set(env) <= self.ALLOWED_KEYS, f"unexpected keys: {set(env) - self.ALLOWED_KEYS}"
        assert set(env) >= {"HOME", "PATH"}
        assert all(isinstance(k, str) and isinstance(v, str) for k, v in env.items())
        assert refresh.build_refresh_env() == env

    def test_the_path_is_a_fixed_list_of_absolute_system_directories(self, refresh):
        entries = refresh.build_refresh_env()["PATH"].split(os.pathsep)
        assert entries
        for entry in entries:
            assert entry and os.path.isabs(entry)
            assert entry != "."
            assert not entry.startswith(("/tmp", "/home", "/var/tmp"))

    def test_the_credential_home_is_derived_from_the_fixed_host_path(
        self, refresh, tmp_path, monkeypatch,
    ):
        """The CLI must land on the SAME credential the preflight reads —
        otherwise it would happily refresh some other profile and the
        preflight's re-read would keep failing."""
        creds = tmp_path / "root" / ".claude" / ".credentials.json"
        monkeypatch.setattr(policy, "HOST_CREDENTIALS_PATH", str(creds))

        env = refresh.build_refresh_env()
        assert env["HOME"] == os.path.dirname(os.path.dirname(str(creds)))
        if "CLAUDE_CONFIG_DIR" in env:
            assert env["CLAUDE_CONFIG_DIR"] == os.path.dirname(str(creds))

    def test_a_hostile_ambient_environment_reaches_the_child_nowhere(
        self, refresh, monkeypatch,
    ):
        for key, value in HOSTILE_ENV.items():
            monkeypatch.setenv(key, value)

        env = refresh.build_refresh_env()

        for key, value in HOSTILE_ENV.items():
            assert key not in env or env[key] != value, key
        assert set(env) <= self.ALLOWED_KEYS

    @pytest.mark.parametrize(
        "key",
        [
            "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL",
            "CLAUDE_CODE_OAUTH_TOKEN", "HTTP_PROXY", "http_proxy", "HTTPS_PROXY",
            "https_proxy", "ALL_PROXY", "NO_PROXY", "NODE_OPTIONS",
            "NODE_EXTRA_CA_CERTS", "LD_PRELOAD", "LD_LIBRARY_PATH", "PYTHONPATH",
            "BASH_ENV", "SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "GIT_SSH_COMMAND",
        ],
    )
    def test_the_named_hazards_are_absent_from_the_child_env(self, refresh, monkeypatch, key):
        monkeypatch.setenv(key, "hostile")
        assert key not in refresh.build_refresh_env()

    def test_the_probe_hands_the_cli_exactly_this_env(
        self, refresh, creds, trusted, lock_at_tmp, spawns, monkeypatch,
    ):
        for key, value in HOSTILE_ENV.items():
            monkeypatch.setenv(key, value)

        refresh.refresh_probe(now=NOW)

        assert len(spawns) == 1
        assert spawns[0]["env"] == refresh.build_refresh_env()
        assert spawns[0]["argv"] == refresh.build_refresh_argv()


# ---------------------------------------------------------------------------
# The cwd is a throwaway directory, never a project
# ---------------------------------------------------------------------------


class TestThrowawayCwd:
    def test_the_cli_runs_in_an_empty_temporary_non_project_directory(
        self, refresh, creds, trusted, lock_at_tmp, spawns, tmp_path,
    ):
        """Running the refresh inside a repository would give a fully-trusted
        host CLI a project to read; it gets an empty scratch directory
        instead."""
        refresh.refresh_probe(now=NOW)

        assert len(spawns) == 1
        cwd = spawns[0]["cwd"]
        assert isinstance(cwd, str) and os.path.isabs(cwd)
        assert spawns[0]["cwd_is_dir"] is True
        assert spawns[0]["cwd_entries"] == []
        assert os.path.realpath(cwd) != os.path.realpath(os.getcwd())
        assert not os.path.exists(os.path.join(cwd, ".git"))

    def test_the_temporary_cwd_is_removed_afterwards(
        self, refresh, creds, trusted, lock_at_tmp, spawns,
    ):
        refresh.refresh_probe(now=NOW)
        assert not os.path.exists(spawns[0]["cwd"])

    def test_each_probe_gets_its_own_cwd(
        self, refresh, creds, trusted, lock_at_tmp, spawns,
    ):
        refresh.refresh_probe(now=NOW)
        refresh.refresh_probe(now=NOW)
        assert len(spawns) == 2
        assert spawns[0]["cwd"] != spawns[1]["cwd"]


class TestTheStagingDirectoryChainIsValidated:
    """A throwaway directory is not automatically a directory it is safe to
    throw the host's credential into.

    A scratch directory under the ambient temp root is world-traversable and
    attacker-influenceable: ``TMPDIR`` decides where it lands, any local user
    can pre-create or watch the parent, and the CLI running there is fully
    trusted with the credential we are asking it to renew. So the cwd is
    created INSIDE the fixed root-owned mode-0700
    ``policy.CREDENTIAL_STAGING_ROOT`` — the same isolated tree the lock lives
    in — and that root is validated on every run rather than taken on faith
    from a ``makedirs(exist_ok=True)`` that would happily accept a directory
    someone else already made.
    """

    def test_the_throwaway_cwd_is_created_inside_the_staging_root(
        self, refresh, creds, trusted, lock_at_tmp, spawns, tmp_path, monkeypatch,
    ):
        """And a hostile ``TMPDIR`` cannot move it: the location is a policy
        literal, not something the ambient environment gets a vote in."""
        decoy = tmp_path / "ambient-tmp"
        decoy.mkdir()
        for key in ("TMPDIR", "TMP", "TEMP"):
            monkeypatch.setenv(key, str(decoy))
        # tempfile caches its answer on first use; clear it so the decoy would
        # genuinely be picked up by anything still asking the environment.
        monkeypatch.setattr(tempfile, "tempdir", None)

        refresh.refresh_probe(now=NOW)

        assert len(spawns) == 1
        cwd = spawns[0]["cwd"]
        assert os.path.dirname(cwd) == policy.CREDENTIAL_STAGING_ROOT.rstrip("/")
        assert os.listdir(str(decoy)) == []
        # Private and real, exactly like the root it sits in.
        assert spawns[0]["cwd_mode"] == 0o700
        assert spawns[0]["cwd_is_symlink"] is False

    def test_an_unsafe_staging_root_fails_closed_without_spawning(
        self, refresh, creds, trusted, spawns, staging_chain, tmp_path, monkeypatch,
    ):
        """A symlinked staging root is someone else's directory wearing the
        right name — the classic way to redirect a privileged write into a
        place its owner can read. Refuse the run; never follow it."""
        target = tmp_path / "elsewhere"
        target.mkdir()
        os.chmod(str(target), 0o700)
        hostile = tmp_path / "staging"
        hostile.symlink_to(target, target_is_directory=True)
        staging_chain["focus"](hostile)
        monkeypatch.setattr(policy, "CREDENTIAL_STAGING_ROOT", str(hostile))
        monkeypatch.setattr(refresh, "refresh_lock", _null_lock)

        result = refresh.refresh_probe(now=NOW)

        assert spawns == []
        assert result["ok"] is False
        assert result["reason"] in refresh.REASONS
        assert os.listdir(str(target)) == []

    @pytest.mark.parametrize(
        "shape",
        [
            "missing", "symlink", "regular_file", "wrong_owner",
            "world_writable", "group_writable", "world_readable",
        ],
    )
    def test_a_hostile_staging_directory_is_refused(
        self, refresh, staging_chain, tmp_path, monkeypatch, shape,
    ):
        """Every shape that would let another account reach the throwaway cwd
        or the lock: absent, a symlink, not a directory, owned by someone else,
        or carrying ANY group/world bit — including a merely readable one,
        since ``x`` on the directory is enough to walk into it."""
        if shape == "wrong_owner":
            path = _staging_shape(tmp_path, "private")
            monkeypatch.setattr(trust, "TRUSTED_UID", os.getuid() + 1)
        else:
            path = _staging_shape(tmp_path, shape)
        staging_chain["focus"](path)

        with pytest.raises(trust.TrustViolation):
            refresh._validate_staging_dir(path)

    def test_a_private_trusted_staging_directory_is_accepted_chain_and_all(
        self, refresh, staging_chain, tmp_path,
    ):
        """The accepting case — and the whole chain, not just the final
        component: one group-writable parent is enough to replace the
        directory underneath a check that only looked at its leaf."""
        path = _staging_shape(tmp_path, "private")
        staging_chain["focus"](path)

        refresh._validate_staging_dir(path)  # must not raise

        assert set(staging_chain["checked"]) >= set(trust.parent_dirs(path))

    def test_the_lock_refuses_a_staging_directory_it_cannot_validate(
        self, refresh, tmp_path, monkeypatch,
    ):
        """The lock is subject to the same check, and fails closed BEFORE it
        creates anything: a lock file made inside a directory that failed
        validation is a lock another account already controls."""
        path = tmp_path / "locks" / "oauth-refresh.lock"
        path.parent.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(refresh, "REFRESH_LOCK_PATH", str(path))

        def _refuse(directory, *args, **kwargs):
            raise trust.TrustViolation(f"staging directory is not private: {directory!r}")

        monkeypatch.setattr(refresh, "_validate_staging_dir", _refuse, raising=False)

        with pytest.raises(refresh.RefreshLockUnavailable):
            with refresh.refresh_lock():  # pragma: no cover - must not be entered
                pass

        assert not path.exists()


# ---------------------------------------------------------------------------
# A timed-out refresh leaves no process behind
# ---------------------------------------------------------------------------


class TestTimedOutChildrenAreReaped:
    """``subprocess`` kills the process it started, and nothing else.

    The host CLI is a Node program that spawns its own children; if it hangs,
    killing only the direct child leaves those descendants running — still
    holding the credential, still able to write it, and now orphaned past the
    end of the probe that was supposed to bound them. So the child is started
    in its OWN session (making it a process-group leader), and a timeout
    signals the GROUP: SIGTERM, one fixed short grace, then SIGKILL for
    whatever ignored it — all before the throwaway cwd is removed, so nothing
    is still running in a directory that is being deleted underneath it.
    """

    def test_the_child_runs_in_its_own_process_session(
        self, refresh, creds, trusted, lock_at_tmp, fake_child,
    ):
        refresh.refresh_probe(now=NOW)

        assert len(fake_child["children"]) == 1
        child = fake_child["children"][0]
        assert child.kwargs.get("start_new_session") is True
        assert child.argv == refresh.build_refresh_argv()
        assert child.kwargs.get("env") == refresh.build_refresh_env()
        assert os.path.dirname(child.kwargs.get("cwd") or "") == (
            policy.CREDENTIAL_STAGING_ROOT.rstrip("/")
        )
        for stream in ("stdin", "stdout", "stderr"):
            assert child.kwargs.get(stream) == subprocess.DEVNULL
        # A run that finished on its own is never signalled.
        assert fake_child["signals"] == []

    @pytest.mark.parametrize(
        "dies_on_term, expected",
        [(True, [signal.SIGTERM]), (False, [signal.SIGTERM, signal.SIGKILL])],
        ids=["sigterm-is-enough", "sigterm-ignored"],
    )
    def test_a_timed_out_group_gets_sigterm_then_sigkill_as_needed(
        self, refresh, creds, trusted, lock_at_tmp, fake_child, dies_on_term, expected,
    ):
        fake_child["hangs"] = True
        fake_child["dies_on_term"] = dies_on_term

        result = refresh.refresh_probe(now=NOW)

        assert result == {"ok": False, "reason": refresh.REASON_TIMEOUT}
        # Addressed to the GROUP (pid == pgid for a session leader), not to the
        # single direct child, which is what leaves descendants behind.
        to_group = [
            sig for target, sig in fake_child["signals"] if target == FAKE_CHILD_PID
        ]
        assert to_group == expected
        assert 0 < float(refresh.REFRESH_KILL_GRACE_SECONDS) <= 5

    def test_the_group_is_reaped_before_the_throwaway_cwd_is_removed(
        self, refresh, creds, trusted, lock_at_tmp, fake_child,
    ):
        fake_child["hangs"] = True
        fake_child["dies_on_term"] = False

        result = refresh.refresh_probe(now=NOW)

        kinds = [kind for kind, _ in fake_child["events"]]
        assert kinds.count("signal") == 2
        assert "rmtree" in kinds
        assert kinds.index("rmtree") > max(
            index for index, kind in enumerate(kinds) if kind == "signal"
        )
        assert not os.path.exists(fake_child["children"][0].kwargs["cwd"])
        assert result["reason"] == refresh.REASON_TIMEOUT


# ---------------------------------------------------------------------------
# Trust, the lock, and exactly-one invocation
# ---------------------------------------------------------------------------


class TestTrustValidation:
    def test_the_cli_is_trust_validated_as_an_executable_before_it_is_run(
        self, refresh, creds, lock_at_tmp, spawns, monkeypatch,
    ):
        seen = []

        def _validate(path, *, executable=False):
            # Recording the spawn count AT VALIDATION TIME is what proves the
            # ordering, rather than merely that both happened.
            seen.append({"path": path, "executable": executable, "spawns": len(spawns)})
            return {}

        monkeypatch.setattr(trust, "validate_trusted_path_chain", _validate)

        refresh.refresh_probe(now=NOW)

        assert {"path": refresh.HOST_CLAUDE_CLI, "executable": True, "spawns": 0} in seen
        assert len(spawns) == 1

    def test_an_untrusted_cli_fails_closed_without_spawning(
        self, refresh, creds, lock_at_tmp, spawns, monkeypatch,
    ):
        def _boom(path, *, executable=False):
            raise trust.TrustViolation(f"trusted path is a symlink: {path!r}")

        monkeypatch.setattr(trust, "validate_trusted_path_chain", _boom)

        result = refresh.refresh_probe(now=NOW)

        assert result == {"ok": False, "reason": refresh.REASON_UNTRUSTED_CLI}
        assert spawns == []

    def test_the_validated_snapshot_is_rechecked_immediately_before_the_run(
        self, refresh, creds, lock_at_tmp, spawns, trust_events,
    ):
        """Validation returns an identity snapshot precisely so it can be
        re-checked at the last moment. Between the check and the exec sits a
        lock acquisition that can block for up to a minute — ample room to
        swap the binary — so a snapshot that is validated and then dropped
        turns the whole trusted-path chain into a time-of-check/time-of-use
        window over a process we hand the host's credential to."""
        refresh.refresh_probe(now=NOW)

        assert [event["event"] for event in trust_events["events"]] == [
            "validate", "revalidate",
        ]
        revalidate = trust_events["events"][1]
        assert revalidate["snapshot"] == trust_events["snapshot"]
        assert revalidate["spawns"] == 0  # before the process, not after it
        assert len(spawns) == 1

    def test_a_cli_swapped_after_validation_fails_closed_without_spawning(
        self, refresh, creds, lock_at_tmp, spawns, trust_events, monkeypatch,
    ):
        def _changed(snapshot):
            raise trust.TrustViolation(
                "trusted path component identity changed before use"
            )

        monkeypatch.setattr(trust, "revalidate_trusted_path_chain", _changed)

        result = refresh.refresh_probe(now=NOW)

        assert result == {"ok": False, "reason": refresh.REASON_UNTRUSTED_CLI}
        assert spawns == []


class TestTheIsolatedLock:
    def test_a_refreshable_credential_spawns_the_cli_exactly_once(
        self, refresh, creds, trusted, lock_at_tmp, spawns_that_refresh,
    ):
        result = refresh.refresh_probe(now=NOW)

        assert result == {"ok": True, "reason": refresh.REASON_REFRESHED}
        assert len(spawns_that_refresh) == 1

    def test_freshness_is_re_checked_only_after_the_lock_is_taken(
        self, refresh, creds, trusted, staging_at_tmp, spawns, monkeypatch,
    ):
        """The whole point of the lock: checking first and locking second is
        exactly the stampede — every waiter would have decided to spawn before
        anyone won the race."""
        events = []
        real_read = oauth.read_freshness

        @contextlib.contextmanager
        def _lock():
            events.append("lock")
            try:
                yield
            finally:
                events.append("unlock")

        def _read(*args, **kwargs):
            events.append("read")
            return real_read(*args, **kwargs)

        monkeypatch.setattr(refresh, "refresh_lock", _lock)
        monkeypatch.setattr(oauth, "read_freshness", _read)

        refresh.refresh_probe(now=NOW)

        assert events[0] == "lock"
        assert "read" in events
        assert events.index("read") < len(events) - 1  # the read is inside the lock
        assert events[-1] == "unlock"
        assert len(spawns) == 1

    def test_a_credential_another_process_already_refreshed_never_spawns(
        self, refresh, creds, trusted, spawns, monkeypatch,
    ):
        """Two workers HOLD on the same expired credential; the first refreshes
        it while the second waits on the lock. The second must NOT burn a
        second CLI run — it re-reads and finds the work already done."""

        @contextlib.contextmanager
        def _lock_that_someone_else_won():
            write_credentials(
                creds, access_token=ACCESS, refresh_token=REFRESH,
                expires_at=(NOW + 3600) * 1000,
            )
            yield

        monkeypatch.setattr(refresh, "refresh_lock", _lock_that_someone_else_won)

        result = refresh.refresh_probe(now=NOW)

        assert result == {"ok": True, "reason": refresh.REASON_ALREADY_FRESH}
        assert spawns == []

    def test_a_lock_it_cannot_take_fails_closed_without_spawning(
        self, refresh, creds, trusted, spawns, monkeypatch,
    ):
        @contextlib.contextmanager
        def _busy():
            raise refresh.RefreshLockUnavailable("another refresh holds the lock")
            yield  # pragma: no cover - unreachable, keeps this a generator

        monkeypatch.setattr(refresh, "refresh_lock", _busy)

        result = refresh.refresh_probe(now=NOW)

        assert result == {"ok": False, "reason": refresh.REASON_LOCK_BUSY}
        assert spawns == []

    def test_the_lock_file_is_never_readable_or_writable_by_others(
        self, refresh, trusted, lock_at_tmp,
    ):
        with refresh.refresh_lock():
            assert os.path.exists(str(lock_at_tmp))
            mode = os.lstat(str(lock_at_tmp)).st_mode
            assert not stat.S_ISLNK(mode)
            assert mode & (stat.S_IRWXG | stat.S_IRWXO) == 0

    def test_two_concurrent_probes_never_run_the_cli_at_the_same_time(
        self, refresh, creds, trusted, lock_at_tmp, monkeypatch, tmp_path,
    ):
        """The stampede this lock exists to stop: two HOLDing sessions hitting
        an expired credential at the same instant."""
        write_credentials(
            creds, access_token=ACCESS, refresh_token=REFRESH,
            expires_at=(time.time() - 3600) * 1000,
        )

        guard = threading.Lock()
        state = {"inside": 0, "max_inside": 0, "runs": 0}

        def _fake(argv, *, env=None, cwd=None, timeout_seconds=None):
            with guard:
                state["inside"] += 1
                state["runs"] += 1
                state["max_inside"] = max(state["max_inside"], state["inside"])
            time.sleep(0.05)
            with guard:
                state["inside"] -= 1
            return _completed(0)

        monkeypatch.setattr(refresh, "_run_refresh", _fake)

        results = []
        threads = [
            threading.Thread(target=lambda: results.append(refresh.refresh_probe()))
            for _ in range(2)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=30)

        assert all(not thread.is_alive() for thread in threads)
        assert len(results) == 2
        assert state["max_inside"] == 1, "two refreshes ran concurrently"
        assert state["runs"] == 2  # neither credential state changed in between

    def test_only_one_of_two_concurrent_probes_actually_refreshes(
        self, refresh, creds, trusted, lock_at_tmp, monkeypatch,
    ):
        """With the lock serializing them, the loser re-reads a credential the
        winner already refreshed and returns success having spawned nothing."""
        write_credentials(
            creds, access_token=ACCESS, refresh_token=REFRESH,
            expires_at=(time.time() - 3600) * 1000,
        )
        runs = []

        def _fake(argv, *, env=None, cwd=None, timeout_seconds=None):
            runs.append(argv)
            write_credentials(
                creds, access_token=ACCESS, refresh_token=REFRESH,
                expires_at=(time.time() + 3600) * 1000,
            )
            return _completed(0)

        monkeypatch.setattr(refresh, "_run_refresh", _fake)

        results = []
        threads = [
            threading.Thread(target=lambda: results.append(refresh.refresh_probe()))
            for _ in range(2)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=30)

        assert len(runs) == 1
        assert all(result["ok"] is True for result in results)
        reasons = sorted(result["reason"] for result in results)
        assert reasons == sorted([refresh.REASON_REFRESHED, refresh.REASON_ALREADY_FRESH])


# ---------------------------------------------------------------------------
# Exit 0 is a claim, not a refresh
# ---------------------------------------------------------------------------


class TestExitZeroIsNotRefreshSuccess:
    """The exit status answers "did the CLI finish?", never "was the
    credential renewed?".

    ``claude --print`` exits 0 for plenty of turns that never touched the
    credential — a cached session, a refusal, a reply produced from an access
    token that was still valid for one more second. Believing 0 makes the
    probe report a refresh that did not happen, which turns an honest HOLD
    into a spawn against an expired credential and buries the real fault. So
    the answer is re-read from the file the CLI itself owns, while the lock is
    STILL held — releasing first would let the next probe's winner rewrite the
    file underneath this one's confirmation, and it is the same read the
    stampede check already does, so it costs nothing.
    """

    def test_a_clean_exit_over_a_still_expired_credential_is_not_success(
        self, refresh, creds, trusted, lock_at_tmp, spawns,
    ):
        result = refresh.refresh_probe(now=NOW)

        assert len(spawns) == 1
        assert result["ok"] is False
        assert result["reason"] in refresh.REASONS
        assert result["reason"] not in {
            getattr(refresh, name) for name in SUCCESS_REASON_NAMES
        }
        # Still closed and non-secret on the way out.
        serialized = json.dumps(result)
        assert ACCESS not in serialized and REFRESH not in serialized

    def test_only_a_credential_confirmed_fresh_under_the_lock_is_a_refresh(
        self, refresh, creds, trusted, lock_at_tmp, spawns_that_refresh,
    ):
        result = refresh.refresh_probe(now=NOW)

        assert result == {"ok": True, "reason": refresh.REASON_REFRESHED}
        assert len(spawns_that_refresh) == 1

    def test_the_confirming_re_read_happens_while_the_lock_is_still_held(
        self, refresh, creds, trusted, staging_at_tmp, monkeypatch,
    ):
        events = []
        real_read = oauth.read_freshness

        @contextlib.contextmanager
        def _lock():
            events.append("lock")
            try:
                yield
            finally:
                events.append("unlock")

        def _read(*args, **kwargs):
            events.append("read")
            return real_read(*args, **kwargs)

        def _run(argv, *, env=None, cwd=None, timeout_seconds=None):
            events.append("spawn")
            write_credentials(
                creds, access_token=ACCESS, refresh_token=REFRESH,
                expires_at=(NOW + 3600) * 1000,
            )
            return _completed(0)

        monkeypatch.setattr(refresh, "refresh_lock", _lock)
        monkeypatch.setattr(oauth, "read_freshness", _read)
        monkeypatch.setattr(refresh, "_run_refresh", _run)

        result = refresh.refresh_probe(now=NOW)

        assert result == {"ok": True, "reason": refresh.REASON_REFRESHED}
        assert events[0] == "lock"
        assert events[-1] == "unlock"
        assert events.count("read") >= 2, "the run's outcome was never re-read"
        spawned_at = events.index("spawn")
        confirmed_at = len(events) - 1 - events[::-1].index("read")
        assert spawned_at < confirmed_at < events.index("unlock")


# ---------------------------------------------------------------------------
# Every failure mode fails closed
# ---------------------------------------------------------------------------


class TestFailClosed:
    def test_a_timeout_fails_closed(self, refresh, creds, trusted, lock_at_tmp, monkeypatch):
        calls = []
        _spawn_returning(
            refresh, monkeypatch, calls,
            subprocess.TimeoutExpired(cmd=["claude"], timeout=45),
        )

        result = refresh.refresh_probe(now=NOW)

        assert result == {"ok": False, "reason": refresh.REASON_TIMEOUT}
        assert len(calls) == 1

    def test_a_non_zero_exit_fails_closed(self, refresh, creds, trusted, lock_at_tmp, monkeypatch):
        calls = []
        _spawn_returning(refresh, monkeypatch, calls, _completed(1, stdout="nope", stderr="nope"))

        result = refresh.refresh_probe(now=NOW)

        assert result == {"ok": False, "reason": refresh.REASON_CLI_FAILED}

    @pytest.mark.parametrize(
        "outcome",
        [None, "ok", 0, [], object(), _completed(None)],
        ids=["none", "string", "int", "list", "object", "no-returncode"],
    )
    def test_a_malformed_result_fails_closed(
        self, refresh, creds, trusted, lock_at_tmp, monkeypatch, outcome,
    ):
        """The probe believes an exit status and nothing else; anything it
        cannot read as one is a failure, never a success."""
        calls = []
        _spawn_returning(refresh, monkeypatch, calls, outcome)

        result = refresh.refresh_probe(now=NOW)

        assert result == {"ok": False, "reason": refresh.REASON_MALFORMED}

    @pytest.mark.parametrize(
        "exc",
        [OSError("exec format error"), RuntimeError("boom"), ValueError("bad")],
        ids=["oserror", "runtimeerror", "valueerror"],
    )
    def test_an_exploding_subprocess_never_propagates(
        self, refresh, creds, trusted, lock_at_tmp, monkeypatch, exc,
    ):
        """``oauth.preflight`` already guards itself, but a probe that raises
        into it would still lose the distinction between "refresh failed" and
        "the preflight itself broke"."""
        calls = []
        _spawn_returning(refresh, monkeypatch, calls, exc)

        result = refresh.refresh_probe(now=NOW)

        assert result == {"ok": False, "reason": refresh.REASON_ERROR}

    def test_a_missing_credential_is_never_a_reason_to_spawn(
        self, refresh, tmp_path, trusted, lock_at_tmp, spawns, monkeypatch,
    ):
        """Only a refreshable-expired credential is worth a refresh; there is
        nothing to refresh into an absent file."""
        monkeypatch.setattr(policy, "HOST_CREDENTIALS_PATH", str(tmp_path / "absent.json"))

        result = refresh.refresh_probe(now=NOW)

        assert result["ok"] is False
        assert result["reason"] in refresh.REASONS
        assert spawns == []

    def test_every_result_is_the_closed_two_key_shape(
        self, refresh, creds, trusted, lock_at_tmp, monkeypatch,
    ):
        outcomes = [
            _completed(0), _completed(2), None, "junk",
            subprocess.TimeoutExpired(cmd=["claude"], timeout=45), RuntimeError("boom"),
        ]
        for outcome in outcomes:
            calls = []
            _spawn_returning(refresh, monkeypatch, calls, outcome)
            write_credentials(
                creds, access_token=ACCESS, refresh_token=REFRESH,
                expires_at=(NOW - 10) * 1000,
            )

            result = refresh.refresh_probe(now=NOW)

            assert set(result) == {"ok", "reason"}
            assert isinstance(result["ok"], bool)
            assert result["reason"] in refresh.REASONS


# ---------------------------------------------------------------------------
# Nothing from the child ever escapes
# ---------------------------------------------------------------------------


class TestNoOutputEverEscapes:
    def test_stdout_and_stderr_never_reach_the_result_or_the_log(
        self, refresh, creds, trusted, lock_at_tmp, monkeypatch, caplog,
    ):
        """A failing ``claude`` prints its own diagnostics, and those can quote
        the credential it just tried to use."""
        calls = []
        _spawn_returning(
            refresh, monkeypatch, calls,
            _completed(
                1,
                stdout=json.dumps({"claudeAiOauth": {"accessToken": ACCESS}}),
                stderr=f"refresh failed for {REFRESH}",
            ),
        )

        with caplog.at_level(logging.DEBUG):
            result = refresh.refresh_probe(now=NOW)

        serialized = json.dumps(result)
        for secret in (ACCESS, REFRESH, "accessToken", "claudeAiOauth"):
            assert secret not in serialized
            assert secret not in caplog.text
        assert result["reason"] in refresh.REASONS

    def test_an_exception_string_never_reaches_the_result_or_the_log(
        self, refresh, creds, trusted, lock_at_tmp, monkeypatch, caplog,
    ):
        calls = []
        _spawn_returning(refresh, monkeypatch, calls, RuntimeError(f"token {REFRESH} rejected"))

        with caplog.at_level(logging.DEBUG):
            result = refresh.refresh_probe(now=NOW)

        assert REFRESH not in json.dumps(result)
        assert REFRESH not in caplog.text
        assert result["reason"] == refresh.REASON_ERROR

    def test_timeout_output_never_reaches_the_result_or_the_log(
        self, refresh, creds, trusted, lock_at_tmp, monkeypatch, caplog,
    ):
        calls = []
        _spawn_returning(
            refresh, monkeypatch, calls,
            subprocess.TimeoutExpired(
                cmd=["claude"], timeout=45,
                output=ACCESS.encode(), stderr=REFRESH.encode(),
            ),
        )

        with caplog.at_level(logging.DEBUG):
            result = refresh.refresh_probe(now=NOW)

        for secret in (ACCESS, REFRESH):
            assert secret not in json.dumps(result)
            assert secret not in caplog.text

    def test_the_happy_path_logs_no_credential_material(
        self, refresh, creds, trusted, lock_at_tmp, spawns_that_refresh, caplog,
    ):
        with caplog.at_level(logging.DEBUG):
            refresh.refresh_probe(now=NOW)

        assert ACCESS not in caplog.text
        assert REFRESH not in caplog.text


# ---------------------------------------------------------------------------
# The plugin still owns no credential write
# ---------------------------------------------------------------------------


class TestTheCredentialIsNeverTouchedByThePlugin:
    def test_the_credential_file_is_byte_for_byte_unchanged(
        self, refresh, creds, trusted, lock_at_tmp, spawns,
    ):
        """Rotated refresh tokens are persisted by the Claude CLI itself. This
        module opens the credential to READ freshness and for nothing else —
        it does not rewrite it, re-own it, or re-mode it."""
        before_bytes = creds.read_bytes()
        before_stat = os.lstat(str(creds))

        refresh.refresh_probe(now=NOW)

        after_stat = os.lstat(str(creds))
        assert creds.read_bytes() == before_bytes
        assert stat.S_IMODE(after_stat.st_mode) == stat.S_IMODE(before_stat.st_mode)
        assert (after_stat.st_uid, after_stat.st_gid) == (before_stat.st_uid, before_stat.st_gid)
        assert after_stat.st_mtime == before_stat.st_mtime

    def test_the_module_cannot_write_parse_or_network(self, refresh):
        """Source-level tripwire, the same shape as ``oauth.py``'s: it fails
        when someone ADDS the capability, which is the moment worth catching.
        This module runs a subprocess (that is its entire job), so the needles
        are narrower than ``oauth.py``'s — but writing a credential, parsing
        one, changing its ownership, or reaching the network directly are all
        still out of bounds."""
        source = open(refresh.__file__, encoding="utf-8").read()
        for forbidden in (
            # never persist or re-own a credential. ``os.chmod`` is NOT on this
            # list: the module has to force mode 0700 on the private staging
            # directories it creates (a bare ``makedirs`` obeys the umask, and
            # the resulting 0755 directory is exactly what
            # ``_validate_staging_dir`` must refuse). That the credential file
            # itself is never re-moded or re-owned is asserted behaviorally, by
            # ``test_the_credential_file_is_byte_for_byte_unchanged``.
            "write_text", "write_bytes", "os.write(", "os.chown",
            "shutil.chown", "json.dump(",
            # never parse credential material
            "accessToken", "refreshToken", "access_token", "refresh_token",
            "Authorization", "Bearer",
            # never talk to an auth endpoint directly
            "urllib", "socket", "requests", "http.client", "httpx",
            # never reintroduce a shell
            "shell=True", "os.system", "popen",
        ):
            assert forbidden not in source, f"oauth_refresh.py must not use {forbidden!r}"

    def test_the_module_adds_no_environment_knobs(self, refresh):
        """Configuration for this feature is the validated config.yaml knob and
        nothing else — an env var would be an unvalidated second control plane
        for a privileged path."""
        source = open(refresh.__file__, encoding="utf-8").read()
        for forbidden in ("os.environ", "getenv", "HERMES_", "environb"):
            assert forbidden not in source, f"oauth_refresh.py must not read {forbidden!r}"

    def test_the_module_never_touches_the_breaker(self, refresh):
        source = open(refresh.__file__, encoding="utf-8").read()
        assert "breaker" not in source

    def test_oauth_does_not_import_the_refresh_module(self):
        """The dependency points one way. ``oauth.py`` keeps its no-subprocess,
        no-write tripwire precisely because it knows nothing about this."""
        source = open(oauth.__file__, encoding="utf-8").read()
        assert "oauth_refresh" not in source


# ---------------------------------------------------------------------------
# oauth.preflight remains the authority
# ---------------------------------------------------------------------------


class TestPreflightRemainsTheAuthority:
    def test_a_probe_that_really_refreshed_lets_the_preflight_pass(
        self, refresh, creds, trusted, lock_at_tmp, monkeypatch,
    ):
        def _fake(argv, *, env=None, cwd=None, timeout_seconds=None):
            # The real CLI rewrites the credential; that disk state — not the
            # probe's return value — is what the preflight believes.
            write_credentials(
                creds, access_token=ACCESS, refresh_token=REFRESH,
                expires_at=(NOW + 3600) * 1000,
            )
            return _completed(0)

        monkeypatch.setattr(refresh, "_run_refresh", _fake)
        oauth.set_refresh_probe(refresh.refresh_probe)
        try:
            result = oauth.preflight(str(creds), now=NOW)
        finally:
            oauth.set_refresh_probe(None)

        assert result["ok"] is True
        assert result["state"] == oauth.STATE_FRESH
        assert result["refresh_attempted"] is True

    def test_a_probe_claiming_success_over_a_stale_file_still_holds(
        self, refresh, creds, trusted, lock_at_tmp, spawns,
    ):
        """The CLI exited 0 but the credential on disk is still expired. The
        probe already refuses that (``TestExitZeroIsNotRefreshSuccess``) — and
        the preflight would HOLD anyway, because the disk is the authority and
        this layering is deliberate: neither check is load-bearing alone."""
        oauth.set_refresh_probe(refresh.refresh_probe)
        try:
            result = oauth.preflight(str(creds), now=NOW)
        finally:
            oauth.set_refresh_probe(None)

        assert len(spawns) == 1
        assert result["ok"] is False
        assert result["state"] == oauth.STATE_REFRESHABLE_EXPIRED
        assert result["classification"] == "auth"
        assert result["refresh_attempted"] is True

    def test_the_probe_result_never_carries_a_token_into_the_preflight(
        self, refresh, creds, trusted, lock_at_tmp, monkeypatch,
    ):
        calls = []
        _spawn_returning(
            refresh, monkeypatch, calls,
            _completed(1, stdout=ACCESS, stderr=REFRESH),
        )
        oauth.set_refresh_probe(refresh.refresh_probe)
        try:
            result = oauth.preflight(str(creds), now=NOW)
        finally:
            oauth.set_refresh_probe(None)

        serialized = json.dumps(result)
        assert ACCESS not in serialized
        assert REFRESH not in serialized

    def test_the_refresh_path_neither_opens_nor_resets_a_breaker(
        self, refresh, creds, trusted, lock_at_tmp, spawns, tmp_path, monkeypatch,
    ):
        """A breaker opened by a real observed failure keeps its own cooldown,
        and a failing refresh never starts one: an unrefreshable credential is
        not an observed rejection by the API."""
        home = tmp_path / ".hermes"
        home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(home))
        breaker = load_submodule("breaker")

        assert breaker.open_classes() == []
        breaker.record_failure("auth", {"auth": 3600})
        state_before = breaker._state_path().read_bytes()

        oauth.set_refresh_probe(refresh.refresh_probe)
        try:
            oauth.preflight(str(creds), now=NOW)
        finally:
            oauth.set_refresh_probe(None)

        assert breaker.open_classes() == ["auth"]
        assert breaker._state_path().read_bytes() == state_before


# ---------------------------------------------------------------------------
# The config knob
# ---------------------------------------------------------------------------


class TestTheConfigKnob:
    def test_the_probe_is_enabled_by_default(
        self, refresh, creds, trusted, lock_at_tmp, spawns_that_refresh,
    ):
        assert refresh.refresh_probe(now=NOW)["ok"] is True
        assert len(spawns_that_refresh) == 1

    def test_a_literal_false_disables_the_probe_entirely(
        self, refresh, creds, trusted, lock_at_tmp, spawns, monkeypatch,
    ):
        monkeypatch.setattr(config, "_load_raw_config", lambda: _cfg(auto_refresh=False))

        result = refresh.refresh_probe(now=NOW)

        assert result == {"ok": False, "reason": refresh.REASON_DISABLED}
        assert spawns == []

    @pytest.mark.parametrize("value", ["false", "no", 0, None, [], {}])
    def test_only_a_literal_false_disables_the_probe(
        self, refresh, creds, trusted, lock_at_tmp, spawns_that_refresh, monkeypatch, value,
    ):
        monkeypatch.setattr(config, "_load_raw_config", lambda: _cfg(auto_refresh=value))

        assert refresh.refresh_probe(now=NOW)["ok"] is True
        assert len(spawns_that_refresh) == 1

    def test_the_default_timeout_is_the_bounded_default(
        self, refresh, creds, trusted, lock_at_tmp, spawns,
    ):
        refresh.refresh_probe(now=NOW)
        assert spawns[0]["timeout_seconds"] == config.DEFAULT_REFRESH_TIMEOUT_SECONDS == 45

    @pytest.mark.parametrize(
        "configured, expected",
        [(10, 10), (60, 60), (120, 120), (1, 10), (0, 10), (-5, 10), (10_000, 120),
         ("soon", 45), (True, 45), (None, 45)],
    )
    def test_the_timeout_reaching_the_subprocess_is_always_in_range(
        self, refresh, creds, trusted, lock_at_tmp, spawns, monkeypatch,
        configured, expected,
    ):
        """A caller-visible timeout that is unbounded (or zero) turns this into
        either a hung privileged subprocess or a refresh that never completes."""
        monkeypatch.setattr(
            config, "_load_raw_config", lambda: _cfg(refresh_timeout_seconds=configured),
        )

        refresh.refresh_probe(now=NOW)

        assert len(spawns) == 1
        assert spawns[0]["timeout_seconds"] == expected
        assert (
            config.MIN_REFRESH_TIMEOUT_SECONDS
            <= spawns[0]["timeout_seconds"]
            <= config.MAX_REFRESH_TIMEOUT_SECONDS
        )


# ---------------------------------------------------------------------------
# Installation — the seam the plugin registration uses
# ---------------------------------------------------------------------------


class TestInstallRefreshProbe:
    def test_installing_puts_this_modules_probe_behind_the_oauth_seam(self, refresh):
        assert refresh.install_refresh_probe() is True
        assert oauth.REFRESH_PROBE is refresh.refresh_probe

    def test_the_installed_probe_is_implemented_outside_oauth(self, refresh):
        import inspect

        refresh.install_refresh_probe()
        source_file = inspect.getsourcefile(oauth.REFRESH_PROBE)
        assert os.path.realpath(source_file) == os.path.realpath(refresh.__file__)
        assert os.path.realpath(source_file) != os.path.realpath(oauth.__file__)

    def test_the_probe_is_a_zero_argument_callable(
        self, refresh, creds, trusted, lock_at_tmp, spawns,
    ):
        """``oauth.preflight`` calls ``probe()`` with no arguments; a probe
        needing one would raise there and be reported as a refresh failure."""
        refresh.install_refresh_probe()
        result = oauth.REFRESH_PROBE()
        assert set(result) == {"ok", "reason"}

    def test_installing_repeatedly_is_idempotent(self, refresh):
        refresh.install_refresh_probe()
        first = oauth.REFRESH_PROBE
        refresh.install_refresh_probe()
        assert oauth.REFRESH_PROBE is first

    def test_a_disabled_knob_installs_no_probe(self, refresh, monkeypatch):
        monkeypatch.setattr(config, "_load_raw_config", lambda: _cfg(auto_refresh=False))
        assert refresh.install_refresh_probe() is False
        assert oauth.REFRESH_PROBE is None

    def test_a_disabled_knob_removes_a_previously_installed_probe(
        self, refresh, monkeypatch,
    ):
        """Deterministic, not merely additive: whatever an earlier
        registration installed, the answer after this call is the knob's."""
        refresh.install_refresh_probe()
        assert oauth.REFRESH_PROBE is not None

        monkeypatch.setattr(config, "_load_raw_config", lambda: _cfg(auto_refresh=False))
        refresh.install_refresh_probe()

        assert oauth.REFRESH_PROBE is None

    def test_an_explicit_cfg_is_used_instead_of_loading_config(self, refresh, monkeypatch):
        def _must_not_load():
            raise AssertionError("install_refresh_probe(cfg) must not re-load config")

        monkeypatch.setattr(config, "_load_raw_config", _must_not_load)

        assert refresh.install_refresh_probe({"oauth": {"auto_refresh": False}}) is False
        assert oauth.REFRESH_PROBE is None

    def test_a_broken_config_leaves_the_probe_enabled(self, refresh, monkeypatch):
        """``load_plugin_config`` already fails closed to defaults; the probe
        must not become a second, quieter way for a broken config.yaml to turn
        the feature off."""
        def _boom():
            raise RuntimeError("config unavailable")

        monkeypatch.setattr(config, "_load_raw_config", _boom)

        assert refresh.install_refresh_probe() is True
        assert oauth.REFRESH_PROBE is refresh.refresh_probe
