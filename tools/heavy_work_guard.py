"""Cross-process concurrency guard for resource-heavy terminal commands.

The guard is opt-in via ``terminal.max_concurrent_heavy_jobs``. Ownership is
held by a kernel file lock. On supported POSIX-local execution, spawned jobs
inherit the lock FD so gateway death cannot release a slot while the job tree is
still alive. The JSON stored in each slot is informational only and never
trusted for lock ownership.
"""
from __future__ import annotations

import json
import os
import re
import shlex
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Optional

from hermes_cli.config import get_hermes_home

_INFO_FLAGS = {"--help", "-h", "--version", "-V", "version", "help"}
_SHELL_OPERATORS = {"&", "&&", "|", "||", ";", ";;", "(", ")", "{", "}"}
_SHELL_CONTROL_WORDS = {
    "if", "then", "elif", "else", "fi", "for", "while", "until", "do", "done", "!",
}
_DYNAMIC_EXECUTABLES = {
    ".",
    "chronic",
    "chrt",
    "daemon",
    "daemonize",
    "doas",
    "docker",
    "env",
    "eval",
    "flock",
    "ionice",
    "nice",
    "parallel",
    "podman",
    "runuser",
    "source",
    "start-stop-daemon",
    "stdbuf",
    "sudo",
    "systemd-run",
    "taskset",
    "time",
    "timeout",
    "unbuffer",
    "xargs",
}
_DETACHING_EXECUTABLES = {
    "daemon",
    "daemonize",
    "start-stop-daemon",
    "systemd-run",
}
_DYNAMIC_INTERPRETERS = {
    "node",
    "nodejs",
    "perl",
    "php",
    "ruby",
}
_HEAVY_CHILD_FDS: ContextVar[tuple[int, ...]] = ContextVar(
    "heavy_work_child_fds", default=()
)


def _token_segments(command: str) -> list[list[str]]:
    """Split a shell command without treating quoted operators as separators."""
    lexer = shlex.shlex(
        command,
        posix=os.name != "nt",
        punctuation_chars="();&|{}",
    )
    lexer.whitespace_split = True
    lexer.commenters = ""
    segments: list[list[str]] = []
    current: list[str] = []
    for token in lexer:
        if token in _SHELL_OPERATORS:
            if current:
                segments.append(current)
                current = []
        else:
            current.append(token)
    if current:
        segments.append(current)
    return segments


def _strip_wrappers(tokens: list[str]) -> list[str]:
    current = list(tokens)
    while current:
        executable = Path(current[0]).name.lower()
        if executable in _SHELL_CONTROL_WORDS:
            current = current[1:]
            continue
        if executable == "sudo":
            current = current[1:]
            while current and current[0].startswith("-"):
                current = current[1:]
            continue
        if executable == "env":
            current = current[1:]
            while current and "=" in current[0] and not current[0].startswith("="):
                current = current[1:]
            continue
        if executable == "time":
            current = current[1:]
            continue
        if executable in {"nohup", "command", "exec", "setsid"}:
            current = current[1:]
            while current and current[0].startswith("-"):
                current = current[1:]
            continue
        if executable == "timeout":
            current = current[1:]
            while current and current[0].startswith("-"):
                current = current[1:]
            if current:
                current = current[1:]
            continue
        while current and "=" in current[0] and not current[0].startswith("="):
            current = current[1:]
        break
    return current


def _classify_tokens(tokens: list[str]) -> Optional[str]:
    # Variable expansion, command substitution, and generic execution wrappers
    # cannot be resolved reliably without executing the shell. Serialize them
    # fail-closed rather than allowing an obfuscated heavy child to bypass the
    # lease.
    if any("$" in token or "`" in token for token in tokens):
        return "dynamic-execution"
    if (
        tokens
        and Path(tokens[0]).name.lower() == "command"
        and len(tokens) > 1
        and tokens[1] in {"-v", "-V"}
    ):
        return None
    if tokens and Path(tokens[0]).name.lower() in _DYNAMIC_EXECUTABLES:
        return "dynamic-execution"

    tokens = _strip_wrappers(tokens)
    if not tokens:
        return None

    executable = Path(tokens[0]).name.lower()
    args = tokens[1:]
    informational = any(arg in _INFO_FLAGS for arg in args)
    direct_informational = bool(args) and args[0] in _INFO_FLAGS

    if re.fullmatch(r"python(?:\d+(?:\.\d+)*)?", executable):
        if len(args) >= 2 and args[0] == "-m" and args[1] in {"pytest", "tox", "nox"}:
            return None if informational else "test-suite"
        if len(args) >= 2 and args[0] == "-c":
            indirect = args[1].lower()
            if re.search(r"\b(pytest|py\.test|tox|nox|jest|vitest)\b", indirect):
                return "test-suite"
            if re.search(r"\b(supabase|pglite)\b", indirect):
                return "database-lab"
            if re.search(r"\b(claude|codex|opencode)\b", indirect):
                return "ai-reviewer"
        # Arbitrary modules, snippets, and scripts can spawn a heavy child using
        # computed strings that lexical matching cannot recover.
        return None if direct_informational else "dynamic-execution"

    if executable in {"uv", "poetry", "pipenv"} and args[:1] == ["run"]:
        return _classify_tokens(args[1:])
    if executable in {"bash", "dash", "ksh", "sh", "zsh", "fish"}:
        for index, arg in enumerate(args):
            if arg.startswith("-") and "c" in arg[1:] and index + 1 < len(args):
                nested = classify_heavy_work(args[index + 1])
                return nested
        # A script path or stdin-fed shell is opaque to this command line.
        return None if direct_informational else "dynamic-execution"
    if executable in _DYNAMIC_INTERPRETERS:
        return None if direct_informational else "dynamic-execution"

    if executable in {"pytest", "py.test", "tox", "nox", "jest", "vitest"}:
        return None if informational else "test-suite"
    if executable == "cargo" and args[:1] == ["test"]:
        return None if informational else "test-suite"
    if executable == "go" and args[:1] == ["test"]:
        return None if informational else "test-suite"
    if executable in {"npm", "pnpm", "yarn", "bun"}:
        if executable == "npm" and args[:1] in (["exec"], ["x"]):
            return "dynamic-execution"
        runner_args = args[1:] if args[:1] == ["run"] else args
        if runner_args and runner_args[0].lower().startswith("test"):
            return None if informational else "test-suite"
    if executable == "make" and args and args[0].lower() in {"test", "tests", "check"}:
        return None if informational else "test-suite"

    if executable == "npx" and args:
        if args[0].startswith("-"):
            return "dynamic-execution"
        return _classify_tokens(args)

    if executable == "supabase" and args:
        action = args[0].lower()
        if action == "start" or action in {"db", "test"}:
            return None if informational else "database-lab"
    if executable.startswith("pglite"):
        return None if informational else "database-lab"

    if executable in {"claude", "codex", "opencode"}:
        return None if informational else "ai-reviewer"

    return None


def classify_heavy_work(command: str) -> Optional[str]:
    """Return the heavy-work category for a shell command, if any."""
    if not isinstance(command, str) or not command.strip():
        return None
    try:
        segments = _token_segments(command)
    except ValueError:
        return "dynamic-execution"
    for tokens in segments:
        category = _classify_tokens(tokens)
        if category:
            return category
    return None


def heavy_work_requests_detach(command: str) -> bool:
    """Return whether guarded work asks the shell/program to self-detach.

    Hermes can keep a lease across its own background/PTY lifecycle, but it
    cannot guarantee ownership after a command deliberately closes inherited
    descriptors. Reject recognized detach forms instead of running unguarded.
    """
    if classify_heavy_work(command) is None:
        return False
    try:
        lexer = shlex.shlex(
            command,
            posix=os.name != "nt",
            punctuation_chars="();&|{}",
        )
        lexer.whitespace_split = True
        lexer.commenters = ""
        tokens = list(lexer)
    except ValueError:
        return True

    if "&" in tokens:
        return True
    lowered = [Path(token).name.lower() for token in tokens]
    if "disown" in lowered or any(
        executable in _DETACHING_EXECUTABLES for executable in lowered
    ):
        return True
    return any(
        token.lower() in {"--background", "--daemon", "--detach", "--fork"}
        for token in tokens
    )


def configured_heavy_work_limit() -> int:
    """Read the internal env bridge for terminal.max_concurrent_heavy_jobs."""
    raw = os.getenv("TERMINAL_MAX_CONCURRENT_HEAVY_JOBS", "0")
    try:
        return max(0, min(int(raw), 16))
    except (TypeError, ValueError):
        return 0


@dataclass(frozen=True)
class HeavyWorkConflict:
    owner: dict[str, Any]


class HeavyWorkLease:
    """One held kernel-lock slot. Release is idempotent."""

    def __init__(self, handle, path: Path, owner: dict[str, Any]) -> None:
        self._handle = handle
        self.path = path
        self.owner = owner
        self._released = False

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        handle = self._handle
        self._handle = None
        if handle is None:
            return
        try:
            # Keep owner metadata intact. POSIX flock locks belong to the open
            # file description shared by inherited FDs. Closing only this copy
            # preserves ownership until the last descendant closes its copy;
            # an explicit LOCK_UN here would incorrectly unlock every holder.
            if os.name == "nt":
                import msvcrt

                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        finally:
            handle.close()

    def child_pass_fds(self) -> tuple[int, ...]:
        """Return POSIX FDs a child must inherit to keep the lease alive."""
        if self._released or self._handle is None or os.name == "nt":
            return ()
        return (self._handle.fileno(),)

    def __enter__(self) -> "HeavyWorkLease":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()


@contextmanager
def inherit_heavy_work_lease(lease: Optional[HeavyWorkLease]) -> Iterator[None]:
    """Expose a lease FD to LocalEnvironment's next foreground spawn."""
    fds = lease.child_pass_fds() if lease is not None else ()
    token = _HEAVY_CHILD_FDS.set(fds)
    try:
        yield
    finally:
        _HEAVY_CHILD_FDS.reset(token)


def current_heavy_work_child_fds() -> tuple[int, ...]:
    """Return task-local lease FDs for a local subprocess spawn."""
    return _HEAVY_CHILD_FDS.get()


def _try_lock(handle) -> bool:
    try:
        if os.name == "nt":
            import msvcrt

            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write(b"\0")
                handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except (BlockingIOError, OSError):
        return False


def _read_owner(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except (OSError, ValueError, TypeError):
        return {}


def acquire_heavy_work_lease(
    command: str,
    *,
    limit: Optional[int] = None,
    session_key: str = "",
) -> tuple[Optional[HeavyWorkLease], Optional[HeavyWorkConflict]]:
    """Try to claim a heavy-work slot without waiting.

    ``(None, None)`` means the command is not classified or the guard is
    disabled. ``(None, conflict)`` means every configured slot is held.
    """
    category = classify_heavy_work(command)
    effective_limit = configured_heavy_work_limit() if limit is None else max(0, limit)
    if category is None or effective_limit <= 0:
        return None, None

    runtime_dir = get_hermes_home() / "runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    owner = {
        "pid": os.getpid(),
        "category": category,
        "session_key": session_key,
        "started_at_epoch": time.time(),
    }
    occupied: list[dict[str, Any]] = []

    for slot in range(effective_limit):
        path = runtime_dir / f"heavy-work-{slot}.lock"
        handle = open(path, "a+b")
        try:
            os.chmod(path, 0o600)
            if not _try_lock(handle):
                handle.close()
                occupied.append(_read_owner(path))
                continue
            handle.seek(0)
            handle.truncate(0)
            handle.write(json.dumps(owner, sort_keys=True).encode("utf-8"))
            handle.flush()
            os.fsync(handle.fileno())
            return HeavyWorkLease(handle, path, owner), None
        except Exception:
            handle.close()
            raise

    conflict_owner = next((item for item in occupied if item), {})
    return None, HeavyWorkConflict(owner=conflict_owner)
