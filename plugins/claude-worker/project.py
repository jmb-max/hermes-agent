"""Dynamic, safe project-root resolution — the ONE authority the gate and
the runner both consult.

This replaces the static ``gate.repo_roots`` allowlist entirely. That
allowlist had two problems a configured list can't fix: it went stale the
moment a new worktree appeared (the worker simply refused to run there), and
whatever it named was what got MOUNTED — so a well-meaning operator entry of
``/root/worktrees`` handed every sandbox container every sibling project at
once. Scope is now derived from the request itself: given any requested cwd,
resolve the canonical Git worktree root that actually contains it, and mount
exactly that.

What is accepted
----------------
An existing, absolute path whose canonical form lives inside a real Git
worktree. The resolved root is the nearest ancestor (or the path itself)
holding a ``.git`` entry.

What is rejected — every one of these is a hard ``ProjectRejected``
-------------------------------------------------------------------
* a relative path, an empty path, or one containing a NUL byte;
* a path that does not exist, or is not a directory;
* a directory that is not inside any Git worktree at all (a plain scratch
  directory), and a BARE repository (no worktree — ``git rev-parse
  --is-inside-work-tree`` is ``false`` there, and a bare repo has no ``.git``
  entry for the ancestor walk to find either);
* a symlink escape: ``realpath`` runs BEFORE anything else, so a link planted
  inside a repo that points outside it resolves to where it really goes and
  is judged there, never smuggled back in;
* a ``.git`` entry that is a symlink, or is neither a directory nor a regular
  file, and a ``.git`` FILE whose contents are not the ``gitdir:`` pointer a
  linked worktree/submodule actually writes;
* a resolved root that is system-sensitive (``policy.UNSAFE_PROJECT_ROOTS`` /
  ``UNSAFE_PROJECT_PREFIXES``), a whole home directory, or the multi-project
  worktree CONTAINER ``/root/worktrees``.

Sibling prefixes are not containment: ``/repo-evil`` is never "inside"
``/repo`` — every containment test here is segment-wise.

Talking to git
--------------
The ancestor walk above is the primary resolver: it is pure filesystem
metadata, so it is deterministic, needs no subprocess, and works for linked
worktrees whose ``.git`` is a file. When a trusted git binary is available it
is consulted as a SECOND opinion, and a disagreement is fatal rather than
ignored — git seeing a different top-level than the walk means something
about the checkout is not what it looks like.

That subprocess is hardened on every axis the threat model needs:

* absolute ``policy.GIT_BIN`` only, whose entire path chain must pass
  ``trust.validate_trusted_path_chain`` (root-owned, non-symlink, never
  group/world-writable) *before* it is executed — if it doesn't, git is
  simply not consulted rather than being trusted anyway;
* a fixed argv list, never a command string and never a shell, so nothing in
  a path can be interpreted as syntax;
* a bounded timeout (``policy.GIT_RESOLVE_TIMEOUT_SECONDS``);
* an environment built from scratch — no ambient ``$PATH``/``$HOME``, and
  ``GIT_CONFIG_NOSYSTEM`` / ``GIT_CONFIG_GLOBAL=/dev/null`` /
  ``GIT_CONFIG_SYSTEM=/dev/null`` so no system or user gitconfig (and
  therefore no ``core.pager``, ``alias.*``, ``core.fsmonitor`` or
  ``include.path`` injected from a repo-adjacent config) can influence it;
* explicit ``-c`` overrides for the settings a hostile checkout could
  otherwise weaponize (``core.hooksPath=/dev/null``,
  ``core.fsmonitor=false``, ``protocol.ext.allow=never``);
* ``safe.directory=*`` passed as the literal glob rather than by
  interpolating the requested path into a config value — that both defuses
  git's "dubious ownership" refusal for root-owned worktrees and leaves no
  place for a path to inject a second config key;
* output validation: exit status 0, exactly three lines, ``true``/``false``
  for the worktree/bare probes, and an absolute NUL-free top-level that must
  itself resolve to a real directory.
"""

from __future__ import annotations

import logging
import os
import stat
import subprocess
from typing import Any, Dict, List, Optional

from . import policy as _policy
from . import trust as _trust

logger = logging.getLogger(__name__)

#: The marker entry an ancestor must carry to be a worktree root. A directory
#: in an ordinary checkout; a regular FILE holding ``gitdir: <path>`` in a
#: linked worktree or a submodule — both are supported, a symlink is not.
GIT_MARKER = ".git"

#: A ``.git`` file is tiny by construction; refuse to read more than this.
MAX_GIT_FILE_BYTES = 4096

_GITDIR_PREFIX = "gitdir:"


class ProjectRejected(ValueError):
    """The requested path is not a safe project root, or does not live inside
    one. Always carries a specific reason; callers surface it (runner) or
    treat it as "out of scope" (gate)."""


# ---------------------------------------------------------------------------
# Safety predicates
# ---------------------------------------------------------------------------


def _is_home_directory(real: str) -> bool:
    """``/root`` or a top-level ``/home/<user>``.

    A whole home directory is never one project: mounting it would expose
    ``.ssh``, ``.claude``, shell history and every unrelated checkout under it
    to a worker asked to touch one repository.
    """
    if real == "/root":
        return True
    return real.startswith("/home/") and real.count("/") == 2


def unsafe_root_reason(real: str) -> Optional[str]:
    """Why *real* may never be a project root, or ``None`` if it may."""
    if real in _policy.UNSAFE_PROJECT_ROOTS:
        return f"{real!r} is a system-sensitive or multi-project directory"
    for prefix in _policy.UNSAFE_PROJECT_PREFIXES:
        if real == prefix or real.startswith(prefix.rstrip("/") + "/"):
            return f"{real!r} is inside the system-sensitive tree {prefix!r}"
    if _is_home_directory(real):
        return f"{real!r} is a home directory, not a single project"
    return None


def contains(root: str, path: str) -> bool:
    """Segment-wise containment: ``/repo-evil`` is not inside ``/repo``."""
    if not root or not path:
        return False
    return path == root or path.startswith(root.rstrip(os.sep) + os.sep)


# ---------------------------------------------------------------------------
# The filesystem ancestor walk — the primary resolver
# ---------------------------------------------------------------------------


def _validate_git_file(marker: str) -> None:
    """A ``.git`` regular file must be the ``gitdir:`` pointer git itself
    writes for a linked worktree or a submodule. Anything else at that path
    is not a worktree marker and must not be treated as one."""
    try:
        with open(marker, "rb") as handle:
            head = handle.read(MAX_GIT_FILE_BYTES)
    except OSError as exc:
        raise ProjectRejected(f".git file could not be read: {marker!r}: {exc}") from exc
    try:
        text = head.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise ProjectRejected(f".git file is not valid UTF-8: {marker!r}") from exc
    if not text.strip().startswith(_GITDIR_PREFIX):
        raise ProjectRejected(
            f".git file is not a {_GITDIR_PREFIX!r} worktree pointer: {marker!r}"
        )


def _worktree_root_from_walk(real: str) -> Optional[str]:
    """Nearest ancestor of (or) *real* carrying a valid ``.git`` marker.

    *real* must already be canonical (``realpath``), which makes every
    ancestor canonical too — so this walk can never be redirected by a
    symlink it did not already resolve.
    """
    current = real
    for _ in range(_policy.MAX_PROJECT_WALK_DEPTH):
        marker = os.path.join(current, GIT_MARKER)
        try:
            st = os.lstat(marker)
        except OSError:
            st = None
        if st is not None:
            if stat.S_ISLNK(st.st_mode):
                raise ProjectRejected(f".git marker is a symlink: {marker!r}")
            if stat.S_ISDIR(st.st_mode):
                return current
            if stat.S_ISREG(st.st_mode):
                _validate_git_file(marker)
                return current
            raise ProjectRejected(
                f".git marker is neither a directory nor a regular file: {marker!r}"
            )
        parent = os.path.dirname(current)
        if parent == current:
            return None
        current = parent
    return None


# ---------------------------------------------------------------------------
# The git second opinion
# ---------------------------------------------------------------------------

#: Built from scratch — nothing ambient crosses into it. See the module
#: docstring for why each entry is here.
_GIT_ENV: Dict[str, str] = {
    "PATH": "/usr/bin:/bin",
    "HOME": "/nonexistent",
    "LC_ALL": "C",
    "LANG": "C",
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_CONFIG_GLOBAL": "/dev/null",
    "GIT_CONFIG_SYSTEM": "/dev/null",
    "GIT_TERMINAL_PROMPT": "0",
    "GIT_OPTIONAL_LOCKS": "0",
    "GIT_PAGER": "cat",
    "GIT_ASKPASS": "",
}

#: ``safe.directory=*`` is the literal glob, never the requested path
#: interpolated into a config value — see the module docstring.
_GIT_HARDENING_ARGS = (
    "-c", "safe.directory=*",
    "-c", "core.hooksPath=/dev/null",
    "-c", "core.fsmonitor=false",
    "-c", "core.pager=cat",
    "-c", "protocol.ext.allow=never",
)


def git_available() -> bool:
    """Whether the fixed git binary's whole path chain is currently trusted.
    Never cached: a binary swapped in after an earlier success is caught."""
    try:
        _trust.validate_trusted_path_chain(_policy.GIT_BIN, executable=True)
    except _trust.TrustViolation:
        return False
    return True


def _git_toplevel(real: str) -> Optional[str]:
    """Git's own opinion of the worktree top-level containing *real*.

    Returns ``None`` when git cannot be consulted at all (untrusted/absent
    binary, timeout, unparseable output) — the walk result then stands alone.
    Raises ``ProjectRejected`` when git answers CLEARLY and the answer is
    disqualifying (not inside a worktree, a bare repository, a non-absolute
    top-level): a definite "no" is never downgraded to "no opinion".
    """
    if not git_available():
        logger.debug("claude-worker project: git binary is not trusted; skipping cross-check")
        return None

    argv: List[str] = [
        _policy.GIT_BIN, *_GIT_HARDENING_ARGS, "-C", real, "rev-parse",
        "--is-inside-work-tree", "--is-bare-repository", "--show-toplevel",
    ]
    try:
        completed = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=_policy.GIT_RESOLVE_TIMEOUT_SECONDS,
            env=dict(_GIT_ENV),
            cwd=real,
        )
    except (subprocess.TimeoutExpired, OSError, ValueError):
        logger.debug("claude-worker project: git rev-parse unavailable", exc_info=True)
        return None

    if completed.returncode != 0:
        # "not a git repository" and friends — no usable opinion. The walk
        # already decided whether a marker exists; it stays authoritative.
        return None

    lines = [line.strip() for line in (completed.stdout or "").splitlines() if line.strip()]
    if len(lines) != 3:
        logger.debug("claude-worker project: git rev-parse returned an unexpected shape")
        return None

    inside, bare, toplevel = lines
    if inside != "true":
        raise ProjectRejected(f"git reports {real!r} is not inside a worktree")
    if bare != "false":
        raise ProjectRejected(f"git reports {real!r} belongs to a bare repository")
    if not toplevel or "\x00" in toplevel or not os.path.isabs(toplevel):
        raise ProjectRejected("git returned a non-absolute or malformed worktree top-level")
    resolved = os.path.realpath(toplevel)
    if not os.path.isdir(resolved):
        raise ProjectRejected("git worktree top-level is not an existing directory")
    return resolved


# ---------------------------------------------------------------------------
# Public API — the resolver the gate and the runner share
# ---------------------------------------------------------------------------


def resolve_project_root_strict(path: Any, *, verify_with_git: bool = True) -> str:
    """Canonical Git worktree root containing *path*, or ``ProjectRejected``.

    *path* must be an existing, absolute directory. This is the resolution
    the runner uses to decide what to MOUNT and the gate uses to decide what
    is in scope — one function, so the two can never disagree and manufacture
    a deadlock.
    """
    if not isinstance(path, str) or not path.strip():
        raise ProjectRejected("a project path is required")
    if "\x00" in path:
        raise ProjectRejected("project path contains a NUL byte")
    if not os.path.isabs(path):
        raise ProjectRejected(f"project path must be absolute: {path!r}")

    try:
        real = os.path.realpath(path)
    except (OSError, ValueError) as exc:
        raise ProjectRejected(f"project path could not be resolved: {path!r}: {exc}") from exc
    if not os.path.isdir(real):
        raise ProjectRejected(f"project path is not an existing directory: {path!r}")

    root = _worktree_root_from_walk(real)
    if root is None:
        raise ProjectRejected(
            f"{path!r} is not inside a Git worktree (no .git marker in any ancestor)"
        )

    reason = unsafe_root_reason(root)
    if reason is not None:
        raise ProjectRejected(f"refusing unsafe project root: {reason}")

    if verify_with_git:
        git_root = _git_toplevel(real)
        if git_root is not None and git_root != root:
            raise ProjectRejected(
                "git and the filesystem disagree about the worktree root "
                f"({git_root!r} vs {root!r}) — refusing rather than guessing"
            )

    return root


def resolve_project_root(path: Any, *, verify_with_git: bool = True) -> Optional[str]:
    """:func:`resolve_project_root_strict`, returning ``None`` instead of
    raising. For callers (the gate) whose "not resolvable" answer is simply
    "out of scope"."""
    try:
        return resolve_project_root_strict(path, verify_with_git=verify_with_git)
    except ProjectRejected:
        return None
    except Exception:  # pragma: no cover - defensive
        logger.debug("claude-worker project: unexpected resolution failure", exc_info=True)
        return None


def resolve_project_root_for_target(path: Any) -> Optional[str]:
    """The project root a FILE path would be written into, or ``None``.

    Unlike a worker cwd, a gated write target need not exist yet
    (``write_file`` creating a new file, possibly in a new subdirectory), so
    this walks up to the nearest existing directory and resolves from there,
    then re-confirms the canonical target really lands inside the root it
    found. A relative target is resolved against the process cwd — the same
    thing the underlying tool would do — rather than being waved through.
    """
    if not isinstance(path, str) or not path.strip():
        return None
    if "\x00" in path:
        return None
    try:
        real = os.path.realpath(path if os.path.isabs(path) else os.path.abspath(path))
    except (OSError, ValueError):
        return None

    directory = real if os.path.isdir(real) else os.path.dirname(real)
    while directory and not os.path.isdir(directory):
        parent = os.path.dirname(directory)
        if parent == directory:
            return None
        directory = parent
    if not directory:
        return None

    root = resolve_project_root(directory)
    if root is None:
        return None
    return root if contains(root, real) else None
