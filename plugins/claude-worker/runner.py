"""Isolated execution for claude_worker (requirements 1, 4, 5) — a locked-down
Docker sandbox, never a host subprocess.

Every model-driven phase runs inside ``docker run`` against the fixed
sandbox image, referenced by its immutable ``policy.SANDBOX_IMAGE_ID``
(never the mutable ``policy.SANDBOX_IMAGE_TAG``): ``--rm --read-only``,
every Linux capability dropped, ``no-new-privileges``, bounded
pids/memory/cpu, tmpfs-only scratch space, the fixed non-root
``policy.SANDBOX_UID``/``SANDBOX_GID`` identity (never root, never the host
runner's own uid — see ``_hardening_opts``), and exactly one writable
structured bind mount — the resolved repo at ``/workspace`` — plus a staged,
re-owned, mode-0400 copy of the operator's Claude OAuth credentials, mounted
read-only (see ``_stage_credentials``); the original root-owned credentials
path is never mounted, since the container could never read it as
``SANDBOX_UID``. There is no docker-socket mount, no other host directory,
and no shell inside the container command: the task text goes over stdin,
never argv, and the verification command is fixed operator-config argv, run
in its own network-isolated, credential-less container. Every docker
invocation — preflight probe or real run — talks to the one fixed
``policy.DOCKER_HOST_ENDPOINT``, never an ambient ``DOCKER_HOST``.

Six independent safety layers, each fail-closed on its own:

  * ``build_child_env`` — an EXPLICIT allowlist (not a blocklist) for the
    *host* ``docker`` CLI's own environment. Only the runtime/locale
    variables docker itself needs cross into it; every provider API key and
    every Hermes integration secret is left behind by construction, nothing
    related to Anthropic credentials is ever passed as an environment
    variable at all, and ``HOME``/``PATH``/``DOCKER_HOST``/``DOCKER_CONTEXT``/
    ``DOCKER_CONFIG`` are never taken from the caller — ``HOME`` and ``PATH``
    are always the isolated literals, and the daemon endpoint is always the
    fixed policy literal, so a hostile ambient ``$PATH`` entry can never
    substitute a fake ``docker`` (which is unreachable anyway, since
    ``policy.DOCKER_BIN`` is an absolute path, never resolved off ``$PATH``).
  * ``validate_cwd`` — delegates to ``project.resolve_project_root_strict``,
    the ONE dynamic resolver the write gate also consults, so the runner and
    the gate can never disagree about what a request's repository is. It
    resolves symlinks (``os.path.realpath``) BEFORE anything else, so a link
    planted inside a repo that points outside it is judged where it really
    goes rather than smuggling the worker out of scope; it accepts only an
    absolute, existing directory inside a real (non-bare) Git worktree; and
    the ONE thing mounted at ``/workspace`` is that exact Git root — never a
    parent holding several checkouts, never a system-sensitive directory,
    and never a configured allowlist entry (there is no ``gate.repo_roots``
    authority anymore). The same realpath-before-mount discipline applies to
    the repo path handed to the docker command builders.
  * ``_validate_trusted_path_chain`` / ``_validate_docker_binary_trust`` /
    ``_open_trusted_credentials`` — every component of the docker binary's
    and the OAuth credentials file's absolute path chains (every parent
    directory AND the final component) must be a real, non-symlink object,
    owned by root, and never writable by anyone but its owner (a
    sticky-bit directory like ``/tmp`` excepted). The docker binary check
    is never cached and re-runs before every preflight and immediately
    before every real ``docker run``; the credentials file is opened with
    ``O_NOFOLLOW`` and its identity — an ``fstat`` on the held-open fd, not
    a re-``stat`` of the path — is re-compared immediately before the
    subprocess call that mounts it, so a replacement between validation and
    use is caught rather than trusted.
  * ``_resolve_credentials_path`` — validates the one fixed host OAuth
    credentials path with ``lstat``, not a symlink-following check: a
    missing path, a symlink, a directory, or a FIFO all raise
    ``SpawnRefused`` before any docker command is even built.
  * ``_stage_credentials`` — copies the held credentials descriptor, in
    bounded chunks (refusing anything over ``policy.MAX_CREDENTIAL_BYTES``),
    into a fresh ``O_CREAT|O_EXCL|O_NOFOLLOW`` file under a root-owned
    mode-0700 staging directory, then ``fchown``s it to ``SANDBOX_UID``/
    ``SANDBOX_GID`` and ``fchmod``s it to 0400 — never logging the content.
    That staged copy's own identity is re-validated immediately before the
    subprocess call exactly like the trusted path chains above, and its
    private directory is removed in ``spawn_claude``'s outer ``finally`` on
    every exit path.
  * ``docker_preflight`` / ``spawn_claude`` — a cached preflight that
    confirms the ``docker`` executable exists, the sandbox image id is a
    real configured value (not the activation placeholder), the daemon
    answers, the fixed sandbox tag resolves to exactly that policy image
    id, and the claude CLI baked into that image actually supports every
    isolation flag this runner depends on. If any of that is not true,
    ``spawn_claude`` refuses to run at all (``SpawnRefused``) rather than
    silently running with weaker isolation or a retagged image.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import stat
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from . import acl as _acl
from . import breaker as _breaker
from . import config as _config
from . import oauth as _oauth
from . import policy as _policy
from . import project as _project
from . import routing as _routing
from . import telemetry as _telemetry
from . import trust as _trust
from .review import run_fallback, run_review, should_review
from gateway.session_context import get_session_env

logger = logging.getLogger(__name__)

_MAX_CAPTURED_OUTPUT_CHARS = 200_000

# ---------------------------------------------------------------------------
# Host-side child environment — explicit allowlist for the docker CLI itself
# ---------------------------------------------------------------------------

#: Only the runtime/locale variables docker itself needs. Deliberately
#: excludes ``PATH`` (always forced to ``_FIXED_DOCKER_PATH`` below — a
#: hostile or merely unexpected ambient ``$PATH`` entry earlier than the
#: real ``/usr/bin`` can never substitute what any *other* PATH-searching
#: tool the docker CLI might shell out to would resolve), ``HOME`` (always
#: forced to ``_ISOLATED_HOME`` below, never taken from the caller), and
#: ``DOCKER_HOST``/``DOCKER_CONTEXT``/``DOCKER_CONFIG`` (the daemon endpoint
#: is the fixed ``policy.DOCKER_HOST_ENDPOINT`` literal, never redirectable
#: by the ambient environment).
_ALLOWED_PASSTHROUGH_ENV = ("LANG", "LC_ALL", "LC_CTYPE", "TERM")

#: The docker CLI's ``$PATH`` is always this literal, regardless of what the
#: ambient environment sets. ``policy.DOCKER_BIN`` itself is already an
#: absolute path and is never resolved off ``$PATH``, but a fixed, minimal
#: ``$PATH`` means nothing docker itself might exec keeps looking at an
#: attacker-controlled directory either.
_FIXED_DOCKER_PATH = "/usr/bin:/bin"

#: The docker CLI's ``$HOME`` is always this literal, regardless of what the
#: ambient environment sets — a hostile or merely unexpected ``HOME`` (``/``,
#: another user's home, a symlinked directory) can never influence where the
#: docker CLI looks for config, plugins, or credential helpers.
_ISOLATED_HOME = "/nonexistent"

# Defense in depth: even though the allowlist above already excludes these
# by construction, scrub them explicitly so a future accidental addition to
# _ALLOWED_PASSTHROUGH_ENV can't silently reintroduce a leak.
_BLOCKED_DOCKER_ENV = ("DOCKER_HOST", "DOCKER_CONTEXT", "DOCKER_CONFIG")
_BLOCKED_ANTHROPIC_ENV = ("ANTHROPIC_API_KEY", "ANTHROPIC_BASE_URL", "ANTHROPIC_TOKEN")


def build_child_env(source_env: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """Build the host ``docker`` CLI's environment from scratch.

    ``source_env`` defaults to ``os.environ``. Only allowlisted runtime/
    locale vars are copied — no provider API key, no Hermes integration
    secret, and no Claude OAuth token ever crosses into this environment;
    the container gets its credentials from the mounted file, never from a
    passed-through variable. ``HOME`` is never copied from *source_env* at
    all — it is always the isolated literal — and the ambient
    ``DOCKER_HOST``/``DOCKER_CONTEXT``/``DOCKER_CONFIG`` are ignored so the
    daemon the docker CLI talks to is always ``policy.DOCKER_HOST_ENDPOINT``,
    never something the caller's environment can redirect.
    """
    source = source_env if source_env is not None else os.environ
    child: Dict[str, str] = {}
    for key in _ALLOWED_PASSTHROUGH_ENV:
        value = source.get(key)
        if value is not None:
            child[key] = value
    child["HOME"] = _ISOLATED_HOME
    child["PATH"] = _FIXED_DOCKER_PATH
    for blocked in _BLOCKED_DOCKER_ENV + _BLOCKED_ANTHROPIC_ENV:
        child.pop(blocked, None)
    return child


# ---------------------------------------------------------------------------
# cwd validation
# ---------------------------------------------------------------------------


class CwdRejected(ValueError):
    """Raised when the requested cwd is missing, is not inside a real Git
    worktree, or resolves to an unsafe project root."""


def validate_cwd(cwd: str, repo_roots: Optional[List[str]] = None) -> str:
    """Resolve *cwd* and confirm it is a safe place for the worker to run.

    Authority is ``project.resolve_project_root_strict`` — the SAME resolver
    the gate uses — so the two can never disagree: an absolute, existing
    directory inside a real (non-bare) Git worktree whose canonical root is
    not system-sensitive. Symlinks resolve BEFORE anything else, so a link
    that escapes its repository is judged where it really points.

    *repo_roots*, when supplied, is an ADDITIONAL containment constraint for
    a caller that has its own explicit allowlist (the isolation tests, and
    any direct caller that pre-validated scope itself). It can only narrow:
    a cwd outside every supplied root is rejected even if its Git root is
    otherwise fine. It is never widened into an alternative to the Git check.
    """
    if not cwd:
        raise CwdRejected("cwd is required")
    try:
        _project.resolve_project_root_strict(cwd)
    except _project.ProjectRejected as exc:
        raise CwdRejected(str(exc)) from exc

    real = os.path.realpath(cwd)
    if repo_roots:
        if _contained_root(real, repo_roots) is None:
            raise CwdRejected(f"cwd {cwd!r} is outside every allowlisted repo root")
    return real


# ---------------------------------------------------------------------------
# Repo cwd directory-chain snapshot/revalidation — closes the TOCTOU window
# between "this cwd was validated inside an allowlisted root" and "this cwd
# is the one docker actually mounts and runs in". Distinct from the trusted
# path-chain helpers above (docker binary, OAuth credentials): those enforce
# root ownership because they protect fixed HOST paths outside the caller's
# control, but a repo checkout is ordinarily owned by the calling user, so
# only non-symlink-ness and identity stability are enforced here — never
# ownership/writability.
# ---------------------------------------------------------------------------


def _dir_chain_components(path: str) -> List[str]:
    """Every directory component of an absolute directory *path*, from
    ``/`` down to and including *path* itself."""
    parts = [p for p in path.split(os.sep) if p]
    components = [os.sep]
    current = os.sep
    for part in parts:
        current = os.path.join(current, part)
        components.append(current)
    return components


def _check_cwd_dir_component(path: str) -> os.stat_result:
    """Validate one directory component of the repo cwd chain: must exist,
    must not be a symlink (anywhere in the chain a symlink could redirect
    the resolved cwd outside every allowlisted repo root), and must be a
    real directory."""
    try:
        st = os.lstat(path)
    except OSError as exc:
        raise CwdRejected(f"cwd path component vanished: {path!r}: {exc}") from exc
    if stat.S_ISLNK(st.st_mode):
        raise CwdRejected(f"cwd path component is a symlink: {path!r}")
    if not stat.S_ISDIR(st.st_mode):
        raise CwdRejected(f"cwd path component is not a directory: {path!r}")
    return st


def _contained_root(real_cwd: str, repo_roots: List[str]) -> Optional[str]:
    """Return the most-specific canonical root from *repo_roots* containing cwd."""
    matches: List[str] = []
    for root in repo_roots or []:
        try:
            real_root = os.path.realpath(root)
        except (OSError, ValueError):
            continue
        if real_cwd == real_root or real_cwd.startswith(real_root.rstrip(os.sep) + os.sep):
            matches.append(real_root)
    return max(matches, key=len) if matches else None


def _effective_roots(real_cwd: str, repo_roots: Optional[List[str]]) -> List[str]:
    """The roots *real_cwd* must be contained in.

    With no explicit *repo_roots*, that is exactly one root: the canonical
    Git worktree root ``project.py`` resolves for this cwd — never a parent
    holding several projects, and never ``/root/worktrees``. An explicit
    list (a caller with its own allowlist, e.g. the isolation tests) is used
    verbatim and can only narrow.
    """
    if repo_roots:
        return list(repo_roots)
    try:
        return [_project.resolve_project_root_strict(real_cwd)]
    except _project.ProjectRejected as exc:
        raise CwdRejected(str(exc)) from exc


def _snapshot_mount_root(
    mount_root: str,
) -> Dict[str, Tuple[int, int, int, int, int]]:
    """Prove dockerd's late pathname lookup cannot be redirected by a
    non-root actor.

    Every component is a real root-owned directory. Parents must not be
    writable by group/other, except a sticky directory (e.g. /tmp), whose
    sticky semantics prevent a non-owner from replacing the root-owned next
    component — parents stay PROTECTED; only the mount root itself may ever
    carry sandbox-uid provisioning.

    The mount root itself may expose a group-write mode bit, but never on
    trust alone: once a directory carries a POSIX ACL, the kernel reports
    the ACL_MASK entry — not the real owning-group permission — in the
    "group" bits ``stat`` shows, so a genuinely-provisioned root (a named
    ACL entry for ``policy.SANDBOX_UID``, real group/other left at r-x) and
    a careless ``chmod g+w`` can look identical to a bare mode check.
    ``acl.evaluate_sandbox_uid_provisioning`` reads the actual ACL to tell
    them apart; a group-write bit with no such verified ACL is refused
    exactly like world-writable, never silently trusted. Replacing the
    final entry still requires write access to its protected parent, not
    write access inside the repo.
    """
    snapshot: Dict[str, Tuple[int, int, int, int, int]] = {}
    components = _dir_chain_components(mount_root)
    for component in components:
        st = _check_cwd_dir_component(component)
        if st.st_uid != _TRUSTED_UID:
            raise CwdRejected(
                f"mount-root path component is not root-owned: {component!r}"
            )
        mode = stat.S_IMODE(st.st_mode)
        if component == mount_root:
            if mode & stat.S_IWOTH:
                raise CwdRejected(f"mount root is world-writable: {component!r}")
            if mode & stat.S_IWGRP:
                safe, reason = _acl.evaluate_sandbox_uid_provisioning(
                    component, _policy.SANDBOX_UID,
                )
                if not safe:
                    raise CwdRejected(
                        "mount root is group-writable without a verified named "
                        f"POSIX ACL for the sandbox uid: {reason}"
                    )
        elif mode & (stat.S_IWGRP | stat.S_IWOTH):
            if not (mode & stat.S_ISVTX):
                raise CwdRejected(
                    f"mount-root parent is group/world-writable: {component!r}"
                )
        snapshot[component] = _stat_identity(st)
    return snapshot


def _mount_plan(
    real_cwd: str, repo_roots: Optional[List[str]],
) -> Tuple[str, str, Dict[str, Tuple[int, int, int, int, int]]]:
    """Return protected host mount root, in-container cwd, and root snapshot.

    With no explicit *repo_roots* the mount root is the canonical Git
    worktree root of *real_cwd* — the ONE directory that gets bind-mounted
    at ``/workspace``. That is deliberately the repository itself and never
    an enclosing directory: mounting ``/root/worktrees`` (or any parent
    holding several checkouts) would hand a worker asked to touch one
    project write access to all of its siblings.
    """
    roots = _effective_roots(real_cwd, repo_roots)
    mount_root = _contained_root(real_cwd, roots)
    if mount_root is None:
        raise CwdRejected(f"cwd {real_cwd!r} is outside every allowlisted repo root")
    root_snapshot = _snapshot_mount_root(mount_root)
    relative = os.path.relpath(real_cwd, mount_root)
    if relative == os.curdir:
        container_cwd = _policy.CONTAINER_WORKDIR
    else:
        normalized = os.path.normpath(relative)
        if normalized == os.pardir or normalized.startswith(os.pardir + os.sep):
            raise CwdRejected("container workdir would escape the mounted repo root")
        container_cwd = os.path.join(_policy.CONTAINER_WORKDIR, normalized)
    return mount_root, container_cwd, root_snapshot


def _revalidate_mount_root(
    mount_root: str,
    snapshot: Dict[str, Tuple[int, int, int, int, int]],
) -> None:
    for component, expected in snapshot.items():
        try:
            st = os.lstat(component)
        except OSError as exc:
            raise CwdRejected(
                f"mount-root component vanished before use: {component!r}: {exc}"
            ) from exc
        if stat.S_ISLNK(st.st_mode) or _stat_identity(st) != expected:
            raise CwdRejected(
                f"mount-root component identity changed before use: {component!r}"
            )
    # Re-run policy checks as well as identity comparison for clarity and
    # defense if the snapshot representation changes later.
    _snapshot_mount_root(mount_root)


def _normalize_container_workdir(value: Optional[str]) -> str:
    workdir = value or _policy.CONTAINER_WORKDIR
    normalized = os.path.normpath(workdir)
    prefix = _policy.CONTAINER_WORKDIR.rstrip("/") + "/"
    if normalized != _policy.CONTAINER_WORKDIR and not normalized.startswith(prefix):
        raise SpawnRefused("container workdir escapes /workspace")
    return normalized


def snapshot_repo_cwd_chain(
    cwd: str, repo_roots: Optional[List[str]] = None,
) -> Tuple[str, Dict[str, Tuple[int, int, int, int, int]]]:
    """Resolve *cwd*, confirm it is contained in one of *repo_roots*, then
    ``lstat`` every directory component from ``/`` through the resolved cwd
    — none may be a symlink.

    When *repo_roots* is not supplied, containment is checked against the
    canonical Git worktree root ``project.py`` resolves for this cwd — the
    same answer the gate gets — so a cwd that is not inside a real Git
    worktree, or whose root is system-sensitive, is rejected here rather
    than silently treated as its own root. An explicit *repo_roots* list is
    honored as-is for callers that already validated containment themselves.

    Returns the resolved cwd and an identity snapshot — ``{component: (dev,
    ino, mode, uid, gid)}`` — for later re-validation via
    :func:`_revalidate_repo_cwd_chain`. Raises ``CwdRejected`` on any
    violation. Editing FILE CONTENTS inside the repo never changes a
    directory's own identity tuple (mtime is deliberately excluded), so this
    snapshot survives ordinary repo writes untouched.
    """
    if not cwd:
        raise CwdRejected("cwd is required")
    real = os.path.realpath(cwd)
    if not os.path.isdir(real):
        raise CwdRejected(f"cwd does not exist or is not a directory: {cwd!r}")

    roots = _effective_roots(real, repo_roots)
    if _contained_root(real, roots) is None:
        raise CwdRejected(f"cwd {cwd!r} is outside every allowlisted repo root")

    snapshot: Dict[str, Tuple[int, int, int, int, int]] = {}
    for component in _dir_chain_components(real):
        st = _check_cwd_dir_component(component)
        snapshot[component] = _stat_identity(st)
    return real, snapshot


def _revalidate_repo_cwd_chain(
    real_cwd: str,
    snapshot: Dict[str, Tuple[int, int, int, int, int]],
    repo_roots: Optional[List[str]],
) -> None:
    """Re-``lstat`` every component recorded in *snapshot* and compare its
    full identity — device, inode, mode, uid, gid — against the value
    captured at snapshot time, then re-confirm canonical containment.

    A component that vanished, was replaced (even by another object at the
    same path — the inode differs), was symlinked, had its mode/ownership
    changed, or no longer resolves inside any allowlisted root raises
    ``CwdRejected`` rather than trusting a cwd that may have moved
    underneath the earlier check.
    """
    for component, expected in snapshot.items():
        try:
            st = os.lstat(component)
        except OSError as exc:
            raise CwdRejected(
                f"cwd path component vanished before use: {component!r}: {exc}"
            ) from exc
        if stat.S_ISLNK(st.st_mode):
            raise CwdRejected(f"cwd path component became a symlink before use: {component!r}")
        if _stat_identity(st) != expected:
            raise CwdRejected(f"cwd path component identity changed before use: {component!r}")

    roots = repo_roots if repo_roots else [real_cwd]
    if _contained_root(real_cwd, roots) is None:
        raise CwdRejected(f"cwd {real_cwd!r} is no longer inside any allowlisted repo root")


# ---------------------------------------------------------------------------
# Trusted absolute path-chain validation — the docker binary and the OAuth
# credentials file are both fixed host paths; both are validated the same
# way before every use, not merely once at startup.
# ---------------------------------------------------------------------------


# The primitives themselves live in ``trust.py`` so ``project.py`` can
# validate the git binary with the identical checks instead of growing a
# second, subtly different copy. These module-level aliases keep every
# existing internal call site — and every test that reaches for
# ``runner.TrustViolation`` / ``runner._validate_trusted_path_chain`` —
# working unchanged.
TrustViolation = _trust.TrustViolation
_TRUSTED_UID = _trust.TRUSTED_UID
_parent_dirs = _trust.parent_dirs
_stat_identity = _trust.stat_identity
_writable_by_others = _trust.writable_by_others
_check_dir_component = _trust.check_dir_component
_check_final_component = _trust.check_final_component
_validate_trusted_path_chain = _trust.validate_trusted_path_chain
_revalidate_trusted_path_chain = _trust.revalidate_trusted_path_chain


def _validate_docker_binary_trust() -> Dict[str, Tuple[int, int, int, int, int]]:
    """Validate the fixed ``policy.DOCKER_BIN`` path chain is fully
    trusted: every parent directory and the binary itself are real,
    non-symlink, root-owned, never writable by anyone but their owner, and
    the binary is owner-executable. Computed fresh on every call — never
    cached — so a binary replaced after an earlier successful check is
    still caught the next time this runs."""
    return _validate_trusted_path_chain(_policy.DOCKER_BIN, executable=True)


def _open_trusted_credentials(
    path: str,
) -> Tuple[int, Dict[str, Tuple[int, int, int, int, int]]]:
    """Validate the OAuth credentials parent-directory chain, then open
    *path* itself with ``O_RDONLY | O_NOFOLLOW | O_CLOEXEC`` and validate
    the *opened descriptor* via ``fstat`` — race-free relative to the open
    itself, unlike a separate ``lstat`` — for exactly regular, root-owned,
    never group/world-writable.

    Returns the open fd (the caller owns it and must close it, in a
    ``finally``, once the subprocess that mounts it has finished) and an
    identity snapshot of the whole chain (parents from ``lstat``, the file
    itself from the ``fstat`` on this fd) for ``_revalidate_trusted_path_chain``
    immediately before that subprocess runs.
    """
    snapshot: Dict[str, Tuple[int, int, int, int, int]] = {}
    for parent in _parent_dirs(path):
        try:
            st = _check_dir_component(parent)
        except TrustViolation as exc:
            raise SpawnRefused(
                f"OAuth credentials path parent failed trust check: {exc}"
            ) from exc
        snapshot[parent] = _stat_identity(st)

    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise SpawnRefused(
            f"OAuth credentials path could not be opened trusted ({exc}): {path!r}"
        ) from exc

    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            raise SpawnRefused(f"OAuth credentials path is not a regular file: {path!r}")
        if st.st_uid != _TRUSTED_UID:
            raise SpawnRefused(
                f"OAuth credentials path is not owned by uid {_TRUSTED_UID}: {path!r}"
            )
        if st.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
            raise SpawnRefused(f"OAuth credentials path is group/world-writable: {path!r}")
    except Exception:
        os.close(fd)
        raise

    snapshot[path] = _stat_identity(st)
    return fd, snapshot


# ---------------------------------------------------------------------------
# OAuth credentials staging — the sandbox container always runs as the
# fixed non-root policy.SANDBOX_UID, which can never read the root-owned
# HOST_CREDENTIALS_PATH directly. Every spawn instead copies the already
# trust-validated, already-open descriptor into a fresh, unique,
# SANDBOX_UID-owned, mode-0400 file under a root-owned 0700 staging
# directory, and mounts ONLY that staged copy — the original path is never
# passed to docker at all.
# ---------------------------------------------------------------------------

_STAGING_DIR_MODE = 0o700
_STAGED_FILE_MODE = 0o400
_STAGE_COPY_CHUNK_BYTES = 65536


def _validate_staging_root() -> None:
    """Ensure the fixed ``policy.CREDENTIAL_STAGING_ROOT`` exists as a
    root-owned, non-symlink, exactly-mode-0700 directory, creating it (and
    forcing that exact mode, regardless of umask) if entirely absent.
    Refuses (``SpawnRefused``) rather than staging into anything that
    doesn't match — a symlink, a directory owned by someone else, or one
    that is group/world-accessible."""
    root = _policy.CREDENTIAL_STAGING_ROOT
    try:
        st = os.lstat(root)
    except FileNotFoundError:
        try:
            os.mkdir(root, _STAGING_DIR_MODE)
            os.chmod(root, _STAGING_DIR_MODE)
        except OSError as exc:
            raise SpawnRefused(
                f"credential staging root could not be created: {exc}: {root!r}"
            ) from exc
        st = os.lstat(root)
    except OSError as exc:
        raise SpawnRefused(
            f"credential staging root could not be checked: {exc}: {root!r}"
        ) from exc

    if stat.S_ISLNK(st.st_mode):
        raise SpawnRefused(f"credential staging root is a symlink: {root!r}")
    if not stat.S_ISDIR(st.st_mode):
        raise SpawnRefused(f"credential staging root is not a directory: {root!r}")
    if st.st_uid != _TRUSTED_UID:
        raise SpawnRefused(
            f"credential staging root is not owned by uid {_TRUSTED_UID}: {root!r}"
        )
    if stat.S_IMODE(st.st_mode) != _STAGING_DIR_MODE:
        raise SpawnRefused(f"credential staging root has an unexpected mode: {root!r}")


def _remove_staging_dir(private_dir: str) -> None:
    """Best-effort removal of one private per-spawn staging directory and
    everything under it. Called from ``spawn_claude``'s outer ``finally``
    on every exit path — success, a nonzero exit, a timeout, an unexpected
    exception, or a preflight/command-build failure — so a staged
    credentials copy never outlives the container run it was made for.
    Never raises: cleanup must not be able to mask or replace the real
    outcome of the spawn."""
    try:
        shutil.rmtree(private_dir, ignore_errors=True)
    except Exception:
        logger.warning(
            "claude_worker: failed to remove staged credentials directory", exc_info=True,
        )


#: Stale-staging reap — best-effort cleanup for staging directories left
#: behind by a worker that never reached its own ``spawn_claude`` ``finally``
#: (killed, host reboot, an exception in a code path added later that
#: forgets to re-raise through the existing ``finally``). This is opportunistic
#: hygiene, not a security boundary, so it is deliberately conservative: an
#: entry is removed ONLY when its ownership, exact directory shape, age, and
#: (if a cidfile is present) the CONFIRMED absence of any container are all
#: provably true. Anything that cannot be proven safe is left in place and
#: logged so an operator can inspect and remove it by hand — this must never
#: grow into a second broad-deletion mechanism.
_STALE_STAGING_MAX_AGE_SECONDS = 2 * _config.MAX_TIMEOUT_SECONDS
_STALE_STAGING_REAP_MAX_ENTRIES = 20
_STAGED_CREDENTIALS_NAME = "credentials.json"
_STAGED_CIDFILE_NAME = "container.cid"
_STAGING_ALLOWED_ENTRY_NAMES = frozenset({_STAGED_CREDENTIALS_NAME, _STAGED_CIDFILE_NAME})


def _staging_entry_is_stale_and_safe(entry_path: str, now: float) -> Tuple[bool, str]:
    """Is *entry_path* (one direct child of ``CREDENTIAL_STAGING_ROOT``)
    provably a leftover from a finished spawn that is safe to remove?

    Every one of these must hold, or the entry is left alone:

    * a real directory, never a symlink;
    * owned by the trusted root uid and exactly mode 0700 — the identity
      ``_stage_credentials`` always creates it with;
    * every entry inside it is one of the two known staged filenames, and
      each present one is itself a regular, non-symlink file — anything
      else means this is not a shape this module ever produced;
    * older than a large fixed multiple of the maximum configured spawn
      timeout, never merely "not the newest one";
    * if a cidfile is present: its content is a well-formed container id
      AND docker confirms — not merely fails to deny — that the container
      no longer exists. An unreadable/malformed cidfile, or one docker
      cannot answer for, blocks the reap rather than being treated as
      harmless.
    """
    try:
        st = os.lstat(entry_path)
    except OSError as exc:
        return False, f"could not stat: {exc}"
    if stat.S_ISLNK(st.st_mode):
        return False, "is a symlink"
    if not stat.S_ISDIR(st.st_mode):
        return False, "is not a directory"
    if st.st_uid != _TRUSTED_UID:
        return False, "is not root-owned"
    if stat.S_IMODE(st.st_mode) != _STAGING_DIR_MODE:
        return False, "does not have the expected staging directory mode"

    if (now - st.st_mtime) < _STALE_STAGING_MAX_AGE_SECONDS:
        return False, "not old enough to be considered stale"

    try:
        names = os.listdir(entry_path)
    except OSError as exc:
        return False, f"could not list contents: {exc}"
    if any(name not in _STAGING_ALLOWED_ENTRY_NAMES for name in names):
        return False, "contains unexpected entries"

    if _STAGED_CIDFILE_NAME in names:
        cidfile_path = os.path.join(entry_path, _STAGED_CIDFILE_NAME)
        try:
            cid_st = os.lstat(cidfile_path)
        except OSError as exc:
            return False, f"could not stat cidfile: {exc}"
        if stat.S_ISLNK(cid_st.st_mode) or not stat.S_ISREG(cid_st.st_mode):
            return False, "cidfile is not a regular file"
        try:
            container_id = Path(cidfile_path).read_text(encoding="ascii").strip()
        except (OSError, UnicodeError):
            return False, "cidfile could not be read as a container id"
        if len(container_id) != 64 or any(
            char not in "0123456789abcdef" for char in container_id.lower()
        ):
            return False, "cidfile does not contain a well-formed container id"
        if _container_still_exists(container_id, build_child_env()) is not False:
            return False, "container existence could not be confirmed absent"

    if _STAGED_CREDENTIALS_NAME in names:
        creds_path = os.path.join(entry_path, _STAGED_CREDENTIALS_NAME)
        try:
            creds_st = os.lstat(creds_path)
        except OSError as exc:
            return False, f"could not stat staged credentials: {exc}"
        if stat.S_ISLNK(creds_st.st_mode) or not stat.S_ISREG(creds_st.st_mode):
            return False, "staged credentials entry is not a regular file"

    return True, "stale, root-owned, exact expected shape, no active container"


def reap_stale_staging_dirs() -> None:
    """Best-effort removal of staging directories left behind by a worker
    that never reached its own ``finally`` (crash, kill -9, host reboot).

    Bounded to a small fixed number of entries per call and conservative by
    construction (see ``_staging_entry_is_stale_and_safe``). Never raises —
    called opportunistically before staging a new credentials copy, so a bug
    here must never be able to block a spawn; anything left un-reaped is
    logged for an operator to clean up by hand.
    """
    root = _policy.CREDENTIAL_STAGING_ROOT
    try:
        root_st = os.lstat(root)
    except OSError:
        return
    if stat.S_ISLNK(root_st.st_mode) or root_st.st_uid != _TRUSTED_UID:
        return

    try:
        entries = os.listdir(root)
    except OSError:
        return

    now = time.time()
    for name in entries[:_STALE_STAGING_REAP_MAX_ENTRIES]:
        entry_path = os.path.join(root, name)
        try:
            safe, reason = _staging_entry_is_stale_and_safe(entry_path, now)
        except Exception:
            logger.warning(
                "claude_worker: stale-staging check failed unexpectedly for %r",
                name, exc_info=True,
            )
            continue
        if not safe:
            continue
        try:
            shutil.rmtree(entry_path)
        except Exception:
            logger.warning(
                "claude_worker: failed to reap stale staging directory %r", name, exc_info=True,
            )
        else:
            logger.warning(
                "claude_worker: reaped stale staging directory %r (%s)", name, reason,
            )


def _stage_credentials(
    creds_fd: int,
) -> Tuple[str, str, Dict[str, Tuple[int, int, int, int, int]]]:
    """Copy the already-open, already-trust-validated OAuth credentials
    descriptor *creds_fd* into a fresh, unique file under a root-owned
    0700 staging directory, owned by ``policy.SANDBOX_UID``/``SANDBOX_GID``
    and mode 0400 — the ONLY thing ever mounted into the container.

    Copies in bounded chunks and refuses (``SpawnRefused``) rather than
    truncating silently if the source exceeds ``policy.MAX_CREDENTIAL_BYTES``.
    Never logs the copied content. Returns ``(private_dir, staged_path,
    snapshot)``; the caller must remove ``private_dir`` (via
    ``_remove_staging_dir``) in a ``finally`` on every exit path. Also
    opportunistically reaps other, unrelated staging directories proven
    stale (see ``reap_stale_staging_dirs``) so a crashed prior worker's
    leftovers do not accumulate forever.
    """
    _validate_staging_root()
    reap_stale_staging_dirs()
    private_dir = tempfile.mkdtemp(dir=_policy.CREDENTIAL_STAGING_ROOT)
    os.chmod(private_dir, _STAGING_DIR_MODE)
    staged_path = os.path.join(private_dir, "credentials.json")

    flags = os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_WRONLY
    try:
        dest_fd = os.open(staged_path, flags, _STAGED_FILE_MODE)
    except OSError as exc:
        _remove_staging_dir(private_dir)
        raise SpawnRefused(f"could not create staged credentials file: {exc}") from exc

    try:
        os.lseek(creds_fd, 0, os.SEEK_SET)
        total = 0
        while True:
            chunk = os.read(creds_fd, _STAGE_COPY_CHUNK_BYTES)
            if not chunk:
                break
            total += len(chunk)
            if total > _policy.MAX_CREDENTIAL_BYTES:
                raise SpawnRefused(
                    "OAuth credentials file exceeds the maximum staged size "
                    f"({_policy.MAX_CREDENTIAL_BYTES} bytes) — refusing to stage it"
                )
            os.write(dest_fd, chunk)

        os.fchown(dest_fd, _policy.SANDBOX_UID, _policy.SANDBOX_GID)
        os.fchmod(dest_fd, _STAGED_FILE_MODE)

        st = os.fstat(dest_fd)
        if not stat.S_ISREG(st.st_mode):
            raise SpawnRefused(f"staged credentials path is not a regular file: {staged_path!r}")
        if st.st_uid != _policy.SANDBOX_UID or st.st_gid != _policy.SANDBOX_GID:
            raise SpawnRefused(f"staged credentials file has an unexpected owner: {staged_path!r}")
        if stat.S_IMODE(st.st_mode) != _STAGED_FILE_MODE:
            raise SpawnRefused(f"staged credentials file has an unexpected mode: {staged_path!r}")

        os.fsync(dest_fd)
    except Exception:
        os.close(dest_fd)
        _remove_staging_dir(private_dir)
        raise
    else:
        os.close(dest_fd)

    return private_dir, staged_path, {staged_path: _stat_identity(st)}


# ---------------------------------------------------------------------------
# Docker preflight — confirm the sandbox is actually usable before any spawn
# ---------------------------------------------------------------------------

_PREFLIGHT_TIMEOUT_SECONDS = 15
_preflight_result: Optional[Dict[str, Any]] = None


def reset_preflight_cache() -> None:
    global _preflight_result
    _preflight_result = None


def _docker_base_argv() -> List[str]:
    """The fixed ``docker --host <endpoint>`` prefix every docker
    invocation — preflight probe or real run — starts with, so the daemon
    the CLI talks to is always the one policy literal, never whatever the
    ambient ``DOCKER_HOST`` happens to say."""
    return [_policy.DOCKER_BIN, "--host", _policy.DOCKER_HOST_ENDPOINT]


def _help_probe_command() -> List[str]:
    return [
        *_docker_base_argv(), "run", "--rm", "--network", "none",
        "--entrypoint", "claude", _policy.SANDBOX_IMAGE_ID, "--help",
    ]


def docker_preflight() -> Dict[str, Any]:
    """Confirm the sandbox is usable: the docker binary's whole path chain
    is trusted, the docker executable exists, the sandbox image id is a
    real configured value (not the activation placeholder), its daemon
    answers, the fixed sandbox tag resolves to exactly that policy image
    id, and the CLI baked into that image supports every isolation flag
    this runner depends on. The docker-subprocess checks (info/inspect/
    help) are cached after the first call; the path-chain trust check is
    NOT cached and re-runs on every call — it is a cheap local ``lstat``
    walk, so re-validating it every time closes the window where a binary
    swapped in after an earlier successful preflight would otherwise keep
    being trusted from the cache. Degrades CLOSED (``ok`` False) on every
    failure mode, including an exception raised by docker itself.
    """
    global _preflight_result

    try:
        _validate_docker_binary_trust()
    except TrustViolation as exc:
        return {
            "ok": False,
            "reason": f"docker binary trust check failed: {exc}",
            "docker_present": False, "daemon_ok": False, "image_present": False,
        }

    if _preflight_result is not None:
        return _preflight_result

    result: Dict[str, Any] = {
        "ok": False, "reason": "", "docker_present": False,
        "daemon_ok": False, "image_present": False,
    }

    if not shutil.which(_policy.DOCKER_BIN):
        result["reason"] = "docker executable not found on PATH"
        _preflight_result = result
        return result
    result["docker_present"] = True

    if not _policy.SANDBOX_IMAGE_ID_CONFIGURED:
        result["reason"] = (
            "sandbox image id is still the activation placeholder "
            f"({_policy.SANDBOX_IMAGE_ID}) — refusing until the real image "
            "id is configured"
        )
        _preflight_result = result
        return result

    env = build_child_env()

    try:
        info = subprocess.run(
            [*_docker_base_argv(), "info"],
            capture_output=True, text=True, timeout=_PREFLIGHT_TIMEOUT_SECONDS, env=env,
        )
    except Exception as exc:
        result["reason"] = f"docker daemon check failed: {exc}"
        _preflight_result = result
        return result
    if info.returncode != 0:
        result["reason"] = "docker daemon is not reachable"
        _preflight_result = result
        return result
    result["daemon_ok"] = True

    try:
        inspect = subprocess.run(
            [*_docker_base_argv(), "image", "inspect", _policy.SANDBOX_IMAGE_TAG],
            capture_output=True, text=True, timeout=_PREFLIGHT_TIMEOUT_SECONDS, env=env,
        )
    except Exception as exc:
        result["reason"] = f"docker image inspect failed: {exc}"
        _preflight_result = result
        return result
    if inspect.returncode != 0:
        result["reason"] = f"sandbox image not present locally: {_policy.SANDBOX_IMAGE_TAG}"
        _preflight_result = result
        return result

    try:
        inspected = json.loads(inspect.stdout or "[]")
    except (ValueError, TypeError):
        result["reason"] = "docker image inspect returned unparseable output"
        _preflight_result = result
        return result
    actual_id = (
        inspected[0].get("Id")
        if inspected and isinstance(inspected[0], dict)
        else None
    )
    if actual_id != _policy.SANDBOX_IMAGE_ID:
        result["reason"] = (
            f"sandbox tag {_policy.SANDBOX_IMAGE_TAG!r} resolves to image id "
            f"{actual_id!r}, not the policy image id "
            f"{_policy.SANDBOX_IMAGE_ID!r} — refusing a retagged or "
            "mismatched image"
        )
        _preflight_result = result
        return result
    result["image_present"] = True

    try:
        help_run = subprocess.run(
            _help_probe_command(),
            capture_output=True, text=True, timeout=_PREFLIGHT_TIMEOUT_SECONDS, env=env,
        )
    except Exception as exc:
        result["reason"] = f"in-container CLI help probe failed: {exc}"
        _preflight_result = result
        return result

    help_text = (help_run.stdout or "") + (help_run.stderr or "")
    missing = [flag for flag in _policy.REQUIRED_CLI_FLAGS if flag not in help_text]
    if help_run.returncode != 0 or missing:
        result["reason"] = (
            "sandboxed CLI is missing required flag(s): "
            f"{', '.join(missing) or 'help probe exited non-zero'}"
        )
        _preflight_result = result
        return result

    result["ok"] = True
    _preflight_result = result
    return result


# ---------------------------------------------------------------------------
# Docker command construction — the only writable mount is the repo itself
# ---------------------------------------------------------------------------


def _bind_mount(src: str, dst: str, *, readonly: bool) -> str:
    """One structured ``--mount type=bind`` argument — never ``--volume``,
    so the mount is described in unambiguous key=value fields rather than a
    colon-delimited string that a path containing ``:`` could confuse."""
    if "," in src or "," in dst:
        raise SpawnRefused(
            "bind mount path contains ',' and cannot be represented safely "
            "in Docker's comma-delimited --mount grammar"
        )
    # Docker's structured --mount syntax uses a bare ``readonly`` flag for
    # RO binds and NO field at all for the default read-write mode. ``rw``
    # is valid in legacy ``-v src:dst:rw`` syntax but is rejected here as an
    # unknown key/value field.
    suffix = ",readonly" if readonly else ""
    return f"type=bind,src={src},dst={dst}{suffix}"


def _hardening_opts(mounts: List[str]) -> List[str]:
    """Flags shared by every sandbox container: ephemeral, read-only
    rootfs, no capabilities, no privilege escalation, bounded resources,
    and tmpfs-only scratch space. ``--user`` is deliberately NOT set here —
    the fixed ``policy.SANDBOX_UID``/``SANDBOX_GID`` (never root, never the
    host caller's own uid) is added explicitly by each caller instead, so
    it can never be silently omitted for one container type but not
    another."""
    opts = [
        "--rm",
        "--read-only",
        "--cap-drop", "ALL",
        "--security-opt", "no-new-privileges:true",
        "--pids-limit", str(_policy.PIDS_LIMIT),
        "--memory", _policy.MEMORY_LIMIT,
        "--cpus", _policy.CPU_LIMIT,
        "--tmpfs", "/tmp:rw,nosuid,nodev,size=64m",
        "--tmpfs", f"{_policy.CONTAINER_HOME}:rw,nosuid,nodev,size=64m",
    ]
    for mount in mounts:
        opts += ["--mount", mount]
    return opts


def build_claude_docker_command(
    repo_root: str,
    model: str,
    credentials_path: Optional[str],
    container_workdir: Optional[str] = None,
    cidfile_path: Optional[str] = None,
) -> List[str]:
    """Build the full ``docker run`` argv for one isolated claude spawn.

    Every hardening flag, mount, and CLI flag lives here as one literal
    argv — there is no shell, so nothing in *repo_root* or *model* is ever
    interpreted; the task text itself is never placed in argv at all (see
    ``spawn_claude``, which pipes it over stdin instead). The image is
    referenced by its immutable id, never the mutable tag: by the time this
    is called, ``docker_preflight`` has already confirmed the tag resolves
    to that exact id, so pinning the id here means a retag after preflight
    can never change what actually runs.
    """
    if model not in _policy.MODEL_ALLOWLIST:
        raise ValueError(f"model {model!r} is not in the sandbox model allowlist")

    real_repo = os.path.realpath(repo_root)
    safe_workdir = _normalize_container_workdir(container_workdir)
    mounts = [_bind_mount(real_repo, _policy.CONTAINER_WORKDIR, readonly=False)]
    if credentials_path:
        # credentials_path has already been trust-validated (no symlink
        # anywhere in its chain) by the time this is called, so its realpath
        # must be identical to itself. Any discrepancy means the path moved
        # underneath an earlier check — refuse rather than silently mount
        # whatever the realpath resolves to instead.
        real_creds = os.path.realpath(credentials_path)
        if real_creds != credentials_path:
            raise SpawnRefused(
                "OAuth credentials path is not canonical — refusing to mount "
                f"a path that resolves elsewhere: {credentials_path!r} != {real_creds!r}"
            )
        mounts.append(_bind_mount(real_creds, _policy.CONTAINER_CREDENTIALS_PATH, readonly=True))

    docker_opts = _hardening_opts(mounts)
    if cidfile_path is not None:
        if not os.path.isabs(cidfile_path) or "\x00" in cidfile_path:
            raise SpawnRefused("container cidfile path must be an absolute safe path")
        docker_opts += ["--cidfile", cidfile_path]
    docker_opts += [
        # Claude --print reads the task from stdin. Docker closes container
        # stdin unless --interactive/-i is explicit, which otherwise makes
        # every real spawn fail with "Input must be provided" even though
        # subprocess.run(input=task) supplied the prompt to the docker CLI.
        "--interactive",
        "--user", f"{_policy.SANDBOX_UID}:{_policy.SANDBOX_GID}",
        "--workdir", safe_workdir,
        "--env", f"HOME={_policy.CONTAINER_HOME}",
        "--env", f"CLAUDE_CONFIG_DIR={_policy.CONTAINER_CLAUDE_CONFIG_DIR}",
        "--entrypoint", "claude",
        _policy.SANDBOX_IMAGE_ID,
    ]

    settings = {
        "permissions": {
            "defaultMode": _policy.PERMISSION_MODE,
            "allow": list(_policy.CLAUDE_ALLOWED_TOOLS),
            "deny": list(_policy.CLAUDE_DENIED_TOOLS),
        }
    }
    claude_args = [
        "--print",
        "--model", model,
        "--strict-mcp-config",
        "--mcp-config", json.dumps({"mcpServers": {}}),
        "--allowed-tools", ",".join(_policy.CLAUDE_ALLOWED_TOOLS),
        "--disallowed-tools", ",".join(_policy.CLAUDE_DENIED_TOOLS),
        "--settings", json.dumps(settings),
        "--setting-sources", "",
        "--permission-mode", _policy.PERMISSION_MODE,
        "--output-format", "json",
    ]

    return [*_docker_base_argv(), "run", *docker_opts, *claude_args]


def build_verification_docker_command(
    repo_root: str,
    command: List[str],
    container_workdir: Optional[str] = None,
) -> List[str]:
    """Build the ``docker run`` argv for one operator-configured
    verification command.

    Same repo mount and hardening as the claude spawn, but network-isolated
    and never given credentials — a verification command has no business
    reaching the network or reading OAuth secrets. *command* is trusted argv
    from operator config only (never task/model input) and is passed
    through verbatim after the image; there is no shell to interpret it.
    Like the claude spawn, the image is referenced by its immutable id, not
    the tag.
    """
    if not command:
        raise ValueError("verification command must not be empty")

    real_repo = os.path.realpath(repo_root)
    safe_workdir = _normalize_container_workdir(container_workdir)
    mounts = [_bind_mount(real_repo, _policy.CONTAINER_WORKDIR, readonly=False)]

    docker_opts = _hardening_opts(mounts)
    docker_opts += [
        "--user", f"{_policy.SANDBOX_UID}:{_policy.SANDBOX_GID}",
        "--workdir", safe_workdir,
        "--network", "none",
        "--entrypoint", command[0],
        _policy.SANDBOX_IMAGE_ID,
    ]

    return [*_docker_base_argv(), "run", *docker_opts, *command[1:]]


# ---------------------------------------------------------------------------
# Spawn
# ---------------------------------------------------------------------------


class SpawnRefused(RuntimeError):
    """Raised when spawning claude would be unsafe (docker sandbox preflight
    failed, or the OAuth credentials path is not exactly a regular file) —
    degrade closed, never spawn outside the hardened container and never
    mount anything but a confirmed regular file as credentials."""


def _run_subprocess(
    cmd: List[str],
    cwd: Optional[str] = None,
    env: Optional[Dict[str, str]] = None,
    timeout_seconds: Optional[float] = None,
    stdin_text: Optional[str] = None,
):
    return subprocess.run(
        cmd, cwd=cwd, env=env, input=stdin_text,
        capture_output=True, text=True, timeout=timeout_seconds,
    )


#: Bounded verification that a force-removed container is actually gone —
#: closes the "no orphan workers" requirement: docker's client-side timeout
#: kills the ``docker run`` process but does not guarantee the daemon-side
#: container has actually exited, and ``docker rm --force`` itself can
#: return success while the container is still tearing down. A small,
#: fixed number of short polls, never an unbounded wait.
_CONTAINER_REMOVAL_VERIFY_ATTEMPTS = 3
_CONTAINER_REMOVAL_VERIFY_SLEEP_SECONDS = 1.0


def _container_still_exists(container_id: str, env: Dict[str, str]) -> Optional[bool]:
    """``True``/``False`` if docker could answer whether *container_id*
    still exists, or ``None`` if docker itself could not be asked (trust
    check failed, daemon unreachable, timeout, nonzero/malformed result) —
    treated as "cannot confirm" by the caller, never as "gone".

    Uses ``docker container ls -a --filter id=<id>`` rather than ``docker
    inspect``: ``inspect``'s exit code alone cannot distinguish "confirmed
    absent" from "daemon error" — both are nonzero. ``container ls`` exits 0
    in both cases, so its exit code carries no ambiguity, and the actual
    answer is read from stdout: an exact (``--no-trunc``) match on
    *container_id* means present, empty output means confirmed absent. Any
    unexpected/non-empty-but-non-matching output, a nonzero exit, or an
    exception all resolve to "cannot confirm" — never "gone".
    """
    try:
        _validate_docker_binary_trust()
    except TrustViolation:
        return None
    try:
        result = subprocess.run(
            [
                *_docker_base_argv(), "container", "ls", "-a", "--no-trunc",
                "--filter", f"id={container_id}", "--format", "{{.ID}}",
            ],
            capture_output=True, text=True, timeout=10, env=env,
        )
    except Exception:
        return None
    if result.returncode != 0:
        return None
    lines = [line.strip() for line in (result.stdout or "").splitlines() if line.strip()]
    if lines == [container_id]:
        return True
    if not lines:
        return False
    return None


def _force_remove_container(
    cidfile: Path, env: Optional[Dict[str, str]] = None,
) -> None:
    """Force-remove the exact container created by one spawn, then verify —
    within a bounded number of short polls — that it is actually gone.

    Docker's client-side timeout kills ``docker run`` but does not guarantee
    that the daemon-side container exits. The cidfile lives inside the
    root-owned per-spawn staging directory and must contain a full 64-character
    hexadecimal container id; task/model input can never select the removal
    target. If removal cannot be confirmed within the bound, an explicit,
    non-secret diagnostic (the container id, which is an opaque hex handle,
    never a path or credential) is logged so an operator can act on a
    possible orphan rather than one going unnoticed. Cleanup never masks
    the worker's real result — every branch here only logs, never raises.
    """
    try:
        container_id = cidfile.read_text(encoding="ascii").strip()
    except (OSError, UnicodeError):
        return

    try:
        if len(container_id) != 64 or any(
            char not in "0123456789abcdef" for char in container_id.lower()
        ):
            logger.error("claude_worker: refusing cleanup for malformed container cidfile")
            return

        run_env = env or build_child_env()
        try:
            _validate_docker_binary_trust()
            subprocess.run(
                [*_docker_base_argv(), "rm", "--force", container_id],
                capture_output=True,
                text=True,
                timeout=15,
                env=run_env,
            )
        except Exception:
            logger.exception("claude_worker: failed to remove worker container %s", container_id)

        for attempt in range(_CONTAINER_REMOVAL_VERIFY_ATTEMPTS):
            still_exists = _container_still_exists(container_id, run_env)
            if still_exists is False:
                return
            if attempt < _CONTAINER_REMOVAL_VERIFY_ATTEMPTS - 1:
                time.sleep(_CONTAINER_REMOVAL_VERIFY_SLEEP_SECONDS)
        logger.error(
            "claude_worker: could not confirm removal of sandbox container %s after "
            "%d attempt(s) — it may still be running; a manual "
            "`docker rm --force %s` may be required to avoid an orphaned worker",
            container_id, _CONTAINER_REMOVAL_VERIFY_ATTEMPTS, container_id,
        )
    finally:
        try:
            cidfile.unlink(missing_ok=True)
        except OSError:
            logger.warning("claude_worker: failed to remove container cidfile")


def _resolve_credentials_path() -> str:
    """Validate the one fixed host OAuth credentials path with ``lstat`` —
    never a symlink-following ``stat``/``os.path.isfile`` — and return it
    only if it is exactly a regular, non-symlink file.

    Raises ``SpawnRefused`` for anything else: missing, a symlink (which
    could point anywhere, including outside any allowlisted root), a
    directory, or a FIFO. This runs before any docker command is built, so
    a hostile or absent credentials path refuses the spawn outright rather
    than silently mounting the wrong thing or running without isolation
    guarantees.
    """
    path = _policy.HOST_CREDENTIALS_PATH
    try:
        st = os.lstat(path)
    except OSError as exc:
        raise SpawnRefused(
            f"OAuth credentials path is not usable ({exc}): {path!r}"
        ) from exc
    if not stat.S_ISREG(st.st_mode):
        raise SpawnRefused(f"OAuth credentials path is not a regular file: {path!r}")
    return path


def spawn_claude(
    task: str,
    cwd: str,
    model: str,
    timeout_seconds: float,
    source_env: Optional[Dict[str, str]] = None,
    repo_roots: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Spawn one isolated, non-interactive claude run inside the fixed
    sandbox container.

    Returns a dict: ``exit_code``, ``stdout``, ``stderr``, ``duration_ms``,
    ``timed_out``, ``model``. Raises ``SpawnRefused`` (no container started)
    when the docker sandbox preflight fails, when the OAuth credentials path
    is missing or is not exactly a regular file, or when the repo cwd
    directory chain — snapshotted up front and re-validated twice more
    below — fails containment, identity, or symlink checks at any point
    before the subprocess actually runs (see ``snapshot_repo_cwd_chain`` /
    ``_revalidate_repo_cwd_chain``).

    *repo_roots* is an OPTIONAL extra containment constraint for a caller
    that has already validated scope itself; it can only narrow. The normal
    path (``run_worker``/``_run_attempts``) omits it entirely, so containment
    and the ``/workspace`` mount source are both the one canonical Git
    worktree root ``project.resolve_project_root_strict`` derives from *cwd*
    — the same answer the gate gets, never a configured allowlist.
    """
    try:
        real_cwd, cwd_snapshot = snapshot_repo_cwd_chain(cwd, repo_roots)
        mount_root, container_cwd, mount_snapshot = _mount_plan(real_cwd, repo_roots)
    except CwdRejected as exc:
        raise SpawnRefused(f"cwd/mount path chain failed trust check: {exc}") from exc

    credentials_path = _resolve_credentials_path()

    # Open the credentials file first, with O_NOFOLLOW so a symlink race at
    # open time is an OSError rather than silently followed, and hold the fd
    # open for the rest of this call — not because docker mounts by fd (it
    # mounts by path), but so the exact filesystem object validated here has
    # an ``fstat`` snapshot that every later re-check below compares against,
    # race-free relative to the open itself.
    creds_fd, creds_snapshot = _open_trusted_credentials(credentials_path)
    staged_dir: Optional[str] = None
    try:
        preflight = docker_preflight()
        if not preflight.get("ok"):
            raise SpawnRefused(
                f"docker sandbox preflight failed ({preflight.get('reason') or 'unknown reason'}) "
                "— refusing to spawn rather than run without container isolation"
            )

        try:
            _revalidate_trusted_path_chain(creds_snapshot)
        except TrustViolation as exc:
            raise SpawnRefused(
                f"OAuth credentials trust check failed after docker preflight: {exc}"
            ) from exc

        # The sandbox container always runs as the fixed non-root
        # policy.SANDBOX_UID, which can never read the root-owned
        # credentials_path directly — stage a re-owned, mode-0400 copy and
        # mount ONLY that; the original path is never passed to docker.
        staged_dir, staged_credentials_path, staged_snapshot = _stage_credentials(creds_fd)
        cidfile = Path(staged_dir) / "container.cid"

        cmd = build_claude_docker_command(
            repo_root=mount_root,
            model=model,
            credentials_path=staged_credentials_path,
            container_workdir=container_cwd,
            cidfile_path=str(cidfile),
        )
        env = build_child_env(source_env)

        # Re-validate the repo cwd chain, the docker binary, and both the
        # OAuth credentials file and its staged copy — immediately after
        # building the command and directly before the subprocess call that
        # actually uses them, narrowing the TOCTOU window to as small as
        # possible.
        try:
            _revalidate_repo_cwd_chain(real_cwd, cwd_snapshot, repo_roots)
            _revalidate_mount_root(mount_root, mount_snapshot)
        except CwdRejected as exc:
            raise SpawnRefused(
                f"cwd/mount path chain failed trust check after command construction: {exc}"
            ) from exc
        try:
            _validate_docker_binary_trust()
        except TrustViolation as exc:
            raise SpawnRefused(
                f"docker binary trust check failed immediately before run: {exc}"
            ) from exc
        try:
            _revalidate_trusted_path_chain(creds_snapshot)
        except TrustViolation as exc:
            raise SpawnRefused(
                f"OAuth credentials trust check failed immediately before run: {exc}"
            ) from exc
        try:
            _revalidate_trusted_path_chain(staged_snapshot)
        except TrustViolation as exc:
            raise SpawnRefused(
                f"staged credentials trust check failed immediately before run: {exc}"
            ) from exc
        try:
            _revalidate_repo_cwd_chain(real_cwd, cwd_snapshot, repo_roots)
            _revalidate_mount_root(mount_root, mount_snapshot)
        except CwdRejected as exc:
            raise SpawnRefused(
                f"cwd/mount path chain failed trust check immediately before run: {exc}"
            ) from exc

        start = time.monotonic()
        timed_out = False
        try:
            try:
                completed = _run_subprocess(
                    cmd, cwd=real_cwd, env=env, timeout_seconds=timeout_seconds,
                    stdin_text=task,
                )
                exit_code = completed.returncode
                stdout = completed.stdout or ""
                stderr = completed.stderr or ""
            except subprocess.TimeoutExpired as exc:
                timed_out = True
                exit_code = None
                stdout = (exc.stdout or "") if isinstance(exc.stdout, str) else ""
                stderr = (exc.stderr or "") if isinstance(exc.stderr, str) else ""
        finally:
            _force_remove_container(cidfile, env=env)
        duration_ms = int((time.monotonic() - start) * 1000)
    finally:
        os.close(creds_fd)
        if staged_dir is not None:
            _remove_staging_dir(staged_dir)

    return {
        "exit_code": exit_code,
        "stdout": stdout[:_MAX_CAPTURED_OUTPUT_CHARS],
        "stderr": stderr[:_MAX_CAPTURED_OUTPUT_CHARS],
        "duration_ms": duration_ms,
        "timed_out": timed_out,
        "model": model,
    }


def run_verification_command(
    repo_root: str,
    command: List[str],
    timeout_seconds: float,
    repo_roots: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Run one fixed, operator-configured verification command in its own
    network-isolated, credential-less container against the same repo
    mount.

    Never raises: a docker/preflight/subprocess failure (including an
    empty or malformed *command*) resolves to one structured, bounded
    result rather than propagating — this is evidence gathering, not a
    load-bearing safety check, so it must not be able to take the caller
    down with it.
    """
    try:
        real_cwd, cwd_snapshot = snapshot_repo_cwd_chain(repo_root, repo_roots)
        mount_root, container_cwd, mount_snapshot = _mount_plan(real_cwd, repo_roots)

        preflight = docker_preflight()
        if not preflight.get("ok"):
            return {
                "ok": False, "exit_code": None, "stdout": "", "stderr": "",
                "duration_ms": 0, "timed_out": False,
                "failure_class": "isolation_refused",
                "reason": preflight.get("reason") or "docker preflight failed",
            }

        cmd = build_verification_docker_command(
            repo_root=mount_root,
            command=command,
            container_workdir=container_cwd,
        )

        try:
            _validate_docker_binary_trust()
            _revalidate_repo_cwd_chain(real_cwd, cwd_snapshot, repo_roots)
            _revalidate_mount_root(mount_root, mount_snapshot)
        except (TrustViolation, CwdRejected) as exc:
            return {
                "ok": False, "exit_code": None, "stdout": "", "stderr": "",
                "duration_ms": 0, "timed_out": False,
                "failure_class": "isolation_refused",
                "reason": f"verification trust check failed: {exc}",
            }

        start = time.monotonic()
        timed_out = False
        try:
            completed = _run_subprocess(
                cmd, cwd=real_cwd, env=build_child_env(), timeout_seconds=timeout_seconds,
            )
            exit_code = completed.returncode
            stdout = completed.stdout or ""
            stderr = completed.stderr or ""
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            exit_code = None
            stdout = (exc.stdout or "") if isinstance(exc.stdout, str) else ""
            stderr = (exc.stderr or "") if isinstance(exc.stderr, str) else ""
        duration_ms = int((time.monotonic() - start) * 1000)

        ok = (not timed_out) and exit_code == 0
        return {
            "ok": ok,
            "exit_code": exit_code,
            "stdout": stdout[:_MAX_CAPTURED_OUTPUT_CHARS],
            "stderr": stderr[:_MAX_CAPTURED_OUTPUT_CHARS],
            "duration_ms": duration_ms,
            "timed_out": timed_out,
            "failure_class": None if ok else ("timeout" if timed_out else "verification_failed"),
            "reason": "",
        }
    except (CwdRejected, SpawnRefused) as exc:
        return {
            "ok": False, "exit_code": None, "stdout": "", "stderr": "",
            "duration_ms": 0, "timed_out": False,
            "failure_class": "isolation_refused", "reason": str(exc),
        }
    except ValueError as exc:
        return {
            "ok": False, "exit_code": None, "stdout": "", "stderr": "",
            "duration_ms": 0, "timed_out": False,
            "failure_class": "invalid_command", "reason": str(exc),
        }
    except Exception as exc:
        logger.exception("claude_worker: verification command failed unexpectedly")
        return {
            "ok": False, "exit_code": None, "stdout": "", "stderr": "",
            "duration_ms": 0, "timed_out": False,
            "failure_class": "internal_error", "reason": str(exc),
        }


# ---------------------------------------------------------------------------
# Evidence gathering
# ---------------------------------------------------------------------------

_GIT_STATUS_TIMEOUT_SECONDS = 15

#: Fixed, operator-config-independent argv for the git-status evidence
#: verifier — never task/model input, never derived from the caller's
#: ``$PATH`` (an absolute path to the in-image git). Runs inside the same
#: hardened, network-isolated, credential-less sandbox as any other
#: verification command (see ``run_verification_command``): a hostile
#: ambient host ``$PATH`` entry can never substitute a fake ``git`` here,
#: because this never becomes a host subprocess at all.
_GIT_STATUS_COMMAND: List[str] = [
    "/usr/bin/git",
    "-c", "safe.directory=/workspace",
    "-c", "core.hooksPath=/dev/null",
    "-c", "core.fsmonitor=false",
    "status", "--porcelain=v1", "--untracked-files=all",
]


def _git_changed_files(
    cwd: str, repo_roots: Optional[List[str]] = None,
) -> List[str]:
    """Best-effort list of files changed in *cwd*, gathered via the same
    isolated, network-isolated, credential-less sandbox as any other
    verification command — never a host ``git`` subprocess.

    Returns ``[]`` (never raises) for a non-git directory, a sandbox/preflight
    failure, or any other failure — this is evidence, not a load-bearing
    safety check. Parses the sandbox's bounded stdout exactly as the retired
    host implementation did.
    """
    try:
        verification_kwargs: Dict[str, Any] = {}
        if repo_roots is not None:
            verification_kwargs["repo_roots"] = repo_roots
        result = run_verification_command(
            cwd,
            list(_GIT_STATUS_COMMAND),
            _GIT_STATUS_TIMEOUT_SECONDS,
            **verification_kwargs,
        )
    except Exception:
        return []
    if not isinstance(result, dict) or not result.get("ok"):
        return []
    stdout = result.get("stdout") or ""
    files: List[str] = []
    for line in stdout.splitlines():
        if len(line) < 4:
            continue
        path = line[3:].strip()
        if " -> " in path:  # rename: "old -> new"
            path = path.split(" -> ", 1)[1].strip()
        if path:
            files.append(path)
    return files


def _snapshot_file_states(
    cwd: str, files: List[str],
) -> Dict[str, Optional[Tuple[int, int, int, int, int]]]:
    """Capture lightweight metadata for already-dirty paths.

    Evidence only, not a security boundary: this distinguishes unchanged
    pre-existing dirt from paths modified during one worker run without
    hashing large repositories.  Lexical containment plus ``lstat`` avoids
    following a repository symlink outside the configured root.
    """
    root = os.path.realpath(cwd)
    states: Dict[str, Optional[Tuple[int, int, int, int, int]]] = {}
    for path in files:
        if not isinstance(path, str) or not path or os.path.isabs(path):
            continue
        candidate = os.path.abspath(os.path.join(root, path))
        try:
            if os.path.commonpath([root, candidate]) != root:
                continue
        except ValueError:
            continue
        try:
            st = os.lstat(candidate)
        except OSError:
            states[path] = None
            continue
        states[path] = (
            stat.S_IFMT(st.st_mode), st.st_size, st.st_mtime_ns,
            st.st_ctime_ns, st.st_ino,
        )
    return states


def _files_changed_since_baseline(
    before_files: List[str],
    before_states: Dict[str, Optional[Tuple[int, int, int, int, int]]],
    after_files: List[str],
    after_states: Dict[str, Optional[Tuple[int, int, int, int, int]]],
) -> List[str]:
    """Return paths whose Git presence or filesystem state changed."""
    before_set = set(before_files)
    after_set = set(after_files)
    ordered = list(dict.fromkeys(list(after_files) + list(before_files)))
    return [
        path for path in ordered
        if (path in before_set) != (path in after_set)
        or before_states.get(path) != after_states.get(path)
    ]


def _parse_worker_summary(stdout: str) -> str:
    """Best-effort extraction of Claude's own result text from
    ``--output-format json`` stdout. Never raises; returns "" if stdout
    isn't the expected shape."""
    try:
        parsed = json.loads(stdout)
    except (ValueError, TypeError):
        return ""
    if isinstance(parsed, dict):
        result = parsed.get("result")
        if isinstance(result, str):
            return result[:2000]
    return ""


# ---------------------------------------------------------------------------
# Bounded, redacted failure diagnostics for telemetry
#
# The 2026-09-07/08 incident: telemetry had no stderr and the agent-visible
# tool result truncates before anything actionable, so a run of "other"
# failures at 03:50-04:41 UTC left no trail to diagnose from. These helpers
# build a SMALL, best-effort diagnostic — never the full stdout/stderr, never
# the task prompt, never an unbounded string — that ``run_worker`` threads
# into telemetry only (never into the caller-facing tool result, which
# already carries a coarse ``failure_class``). Redaction happens HERE, before
# the excerpt is ever handed to telemetry: absolute-path-looking substrings
# are scrubbed and known secret/bearer/API-key SHAPES (not just named
# fields — the same pattern ``telemetry.py`` already trusts for its own
# scan) are replaced, both before the excerpt is truncated and before the
# fingerprint is computed. ``telemetry.append_record``'s own secret scan
# independently re-applies the same redaction as defense in depth, not as
# the only line of defense.
# ---------------------------------------------------------------------------

_MAX_ERROR_EXCERPT_CHARS = 300
_ABS_PATH_PATTERN = re.compile(r"/(?:[\w.\-]+/)+[\w.\-]+")

#: The same alternate spellings a real ``claude --output-format json`` error
#: body's numeric HTTP status can arrive under — see
#: ``breaker._STATUS_FIELD_NAMES``. Kept as an independent literal (not a
#: shared import) so this best-effort diagnostic extractor can never change
#: breaker's own classification behavior, and vice versa.
_DIAGNOSTIC_STATUS_FIELD_NAMES = ("api_error_status", "status", "status_code", "code")


def _scrub_paths(text: str) -> str:
    return _ABS_PATH_PATTERN.sub("[PATH]", text)


def _status_from_parsed(parsed: Dict[str, Any]) -> Optional[int]:
    for source in (parsed, parsed.get("error")):
        if not isinstance(source, dict):
            continue
        for key in _DIAGNOSTIC_STATUS_FIELD_NAMES:
            value = source.get(key)
            if isinstance(value, int) and not isinstance(value, bool):
                return value
    return None


def _bounded_diagnostic(source: str, text: str, api_error_status: Optional[int] = None) -> Dict[str, Any]:
    """Build the one small, redacted, bounded diagnostic object shape.

    Order matters: scrub filesystem paths, then redact secret/bearer/API-key
    SHAPED substrings, THEN truncate to the excerpt bound, and finally
    fingerprint the already-redacted, already-truncated, already-normalized
    excerpt — never the raw input — so the fingerprint can never itself
    encode something that was redacted, and two failures that differ only in
    the part that got truncated away fingerprint identically.
    """
    scrubbed = _scrub_paths(text) if text else ""
    redacted = _telemetry.redact_secret_shapes(scrubbed) if scrubbed else ""
    excerpt = redacted[:_MAX_ERROR_EXCERPT_CHARS]
    normalized = f"{source}|{api_error_status}|{excerpt}"
    fingerprint = _telemetry.hash_value(normalized) if (source or excerpt) else ""
    return {
        "error_source": source,
        "error_excerpt": excerpt,
        "error_fingerprint": fingerprint,
        "api_error_status": api_error_status,
    }


def _diagnostics_from_text(source: str, text: str) -> Dict[str, Any]:
    """Bounded diagnostic from a plain exception/refusal message — never the
    task text, never raw stdout/stderr."""
    return _bounded_diagnostic(source, text or "")


def _diagnostics_from_spawn(stdout: str, stderr: str) -> Dict[str, Any]:
    """Bounded diagnostic from a failed spawn's captured stdout/stderr.

    Prefers stderr when present (the traditional Unix error channel);
    otherwise best-effort parses ``--output-format json`` stdout for a
    structured numeric HTTP status (``api_error_status`` or one of the
    alternate spellings real CLI error bodies have also been observed to
    use — see :data:`_DIAGNOSTIC_STATUS_FIELD_NAMES`) and the ``result`` text
    specifically — never the whole stdout blob — so the excerpt stays small
    even though the full (bounded-to-200k-char) stdout is much larger.
    """
    if stderr:
        return _bounded_diagnostic("stderr", stderr)
    if not stdout:
        return _bounded_diagnostic("", "")

    api_error_status: Optional[int] = None
    text = stdout
    source = "stdout"
    try:
        parsed = json.loads(stdout)
    except (ValueError, TypeError):
        parsed = None
    if isinstance(parsed, dict):
        api_error_status = _status_from_parsed(parsed)
        result_text = parsed.get("result")
        if isinstance(result_text, str):
            text = result_text
            source = "stdout_json_result"
    return _bounded_diagnostic(source, text, api_error_status)


# ---------------------------------------------------------------------------
# Orchestration — the claude_worker tool handler
# ---------------------------------------------------------------------------


_GENERIC_ERROR = (
    "claude_worker failed with an internal error; details were withheld from "
    "the result and the run was recorded in telemetry."
)

# ---------------------------------------------------------------------------
# Validation status — Bash is deliberately denied/unavailable inside the
# sandboxed worker (see ``policy.CLAUDE_DENIED_TOOLS``), so the worker can
# never run a test suite itself. A successful spawn (exit 0) proves only
# that the isolated Claude CLI process exited cleanly — never that any code
# it touched is correct. Claude's own prose (``summary``) and the observed
# ``files_touched`` are self-report/evidence, not test verification, and
# NEITHER is ever consulted below: the only thing that can move a result off
# "unverified" is a trusted, operator-configured verifier command that this
# module itself runs, in its own network-isolated, credential-less
# container, and inspects for a real process exit code.
# ---------------------------------------------------------------------------

#: No trusted verifier ran (not configured, not enabled, or the spawn itself
#: did not succeed) — the parent must independently verify before trusting
#: the result. This is the closed default: see ``_finish``'s ``setdefault``.
VALIDATION_STATUS_UNVERIFIED = "unverified"

#: The operator-configured ``verification.command`` ran in the sandbox and
#: exited 0. Represented distinctly from "unverified" so a caller can tell
#: "a real command actually ran and passed" apart from "nothing checked
#: this" — both would otherwise collapse into the same ``success: true``.
VALIDATION_STATUS_VERIFIED = "verified_by_configured_verifier"

#: The operator-configured verifier ran but did not pass (nonzero exit,
#: timeout, or the sandbox itself could not run it). Still requires parent
#: verification — this is evidence of a problem, not proof of one.
VALIDATION_STATUS_VERIFICATION_FAILED = "verification_failed"


def _resolve_validation_status(
    cfg: Dict[str, Any], resolved_cwd: str, task_succeeded: bool,
) -> Tuple[str, bool, Optional[Dict[str, Any]]]:
    """Return ``(validation_status, parent_verification_required,
    verification_result)`` for one ``claude_worker`` call.

    Fails closed to ``VALIDATION_STATUS_UNVERIFIED`` /
    ``parent_verification_required=True`` unless BOTH
    ``verification.enabled`` is ``True`` and ``verification.command`` is a
    non-empty configured argv — an operator opt-in, never something a task
    description or the worker's own output can trigger. When configured,
    the command is run for real (``run_verification_command``, the same
    hardened, network-isolated, credential-less container used elsewhere in
    this module) and only its actual exit code decides the outcome; no
    stdout text — from the verifier OR from the worker's own run — is
    parsed for phrases like "tests pass".

    The verifier's timeout is the fixed ``policy.VERIFICATION_TIMEOUT_SECONDS``
    — never the operator-configured ``isolation.timeout_seconds`` (which can
    be as high as 900s) — bounded further by the isolation timeout when that
    is the shorter of the two, so verification is never given more time than
    the worker's own isolated run was allowed.
    """
    if not task_succeeded:
        return VALIDATION_STATUS_UNVERIFIED, True, None

    verification_cfg = cfg.get("verification") or {}
    command = verification_cfg.get("command") or []
    if verification_cfg.get("enabled") is not True or not command:
        return VALIDATION_STATUS_UNVERIFIED, True, None

    isolation_timeout_seconds = float((cfg.get("isolation") or {}).get("timeout_seconds", 900))
    timeout_seconds = min(_policy.VERIFICATION_TIMEOUT_SECONDS, isolation_timeout_seconds)
    result = run_verification_command(resolved_cwd, list(command), timeout_seconds)
    if isinstance(result, dict) and result.get("ok") is True:
        return VALIDATION_STATUS_VERIFIED, False, result
    return VALIDATION_STATUS_VERIFICATION_FAILED, True, result


def run_worker(
    args: Dict[str, Any],
    task_id: Optional[str] = None,
    session_id: Optional[str] = None,
    user_task: Optional[str] = None,
    **_: Any,
) -> str:
    """The ``claude_worker`` tool handler.

    Orchestrates requirements (1)/(2)/(4)/(6)/(9)/(10) in one call: cwd
    validation, breaker check (no spawn while open) with the explicit Terra
    fallback, an OAuth credential freshness preflight (no spawn on a
    credential we can already see is stale), auto model routing, an isolated
    spawn capped structurally at ``policy.MAX_ATTEMPTS``, evidence assembly,
    exactly one telemetry record, and an advisory Terra review for
    substantial successful changes. Returns a JSON string (the
    ``tools.registry.tool_result`` contract) — deterministic, no raw
    prompts/stdout embedded.

    Every exit path — including an unexpected exception anywhere in the
    orchestration — goes through ``_finish``, which emits exactly one
    telemetry record. Terra's review found several paths (a bad config
    value, a subprocess error after preflight) that returned nothing and
    logged nothing; the outer handler below closes that class of gap
    generically rather than one exception type at a time.
    """
    from tools.registry import tool_result

    # Local defaults for everything ``_finish``/``_failure`` close over, set
    # BEFORE the telemetry-protected block below runs. ``get_session_env``
    # and config loading are potentially-raising (a gateway session backend,
    # a monkeypatched hook, a broken contextvar): if either raises, the
    # outer ``except`` below must still be able to build a redacted result
    # and emit exactly one telemetry record using whatever was resolved so
    # far, rather than raising ``UnboundLocalError`` on a never-assigned name.
    cfg: Dict[str, Any] = _config.DEFAULT_CONFIG
    session_key = ""
    chat_id = ""
    cwd_arg: Optional[str] = None

    emitted = {"telemetry": False}

    def _finish(payload: Dict[str, Any], diagnostic: Optional[Dict[str, Any]] = None) -> str:
        payload.setdefault("session_id", session_key)
        payload.setdefault("cwd", cwd_arg if isinstance(cwd_arg, str) else "")
        payload.setdefault("review", None)
        payload.setdefault("status", "ok" if payload.get("success") else "failed")
        payload.setdefault("fallback_ready", False)
        payload.setdefault("fallback", None)
        # Fail-closed default: Bash is unavailable inside the sandboxed
        # worker, so unless a trusted operator-configured verifier actually
        # ran and passed (see ``_resolve_validation_status``), a result must
        # never read as more than self-reported prose/``files_touched`` —
        # never as proven-correct.
        payload.setdefault("validation_status", VALIDATION_STATUS_UNVERIFIED)
        payload.setdefault("parent_verification_required", True)
        payload.setdefault("verification", None)
        if not emitted["telemetry"]:
            emitted["telemetry"] = True
            record = _telemetry.build_record(
                ts=datetime.now(timezone.utc).isoformat(),
                session_id=session_key,
                chat_id=chat_id,
                model=payload.get("model") or "",
                route_reason=payload.get("route_reason") or "",
                attempt=payload.get("attempts", 0),
                escalated=bool(payload.get("escalated")),
                duration_ms=payload.get("duration_ms", 0),
                exit_code=payload.get("exit_code"),
                failure_class=payload.get("failure_class"),
                breaker_state="open" if (payload.get("breaker") or {}).get("open") else "closed",
                cwd=payload.get("cwd") or "",
                files_touched=payload.get("files_touched") or [],
                success=bool(payload.get("success")),
                diagnostic=diagnostic,
            )
            _telemetry.append_record(
                record, configured_path=(cfg.get("telemetry") or {}).get("path", "")
            )
        return tool_result(payload)

    def _failure(failure_class: str, error: str, **extra: Any) -> str:
        payload: Dict[str, Any] = {
            "success": False, "attempts": 0, "escalated": False,
            "failure_class": failure_class, "error": error,
            "breaker": {"open": False, "classes": []},
            "model": "", "route_reason": "", "exit_code": None,
            "duration_ms": 0, "files_touched": [],
        }
        payload.update(extra)
        return _finish(payload)

    try:
        try:
            cfg = _config.load_plugin_config()
        except Exception:  # pragma: no cover - load_plugin_config already fails closed
            cfg = _config.DEFAULT_CONFIG
        session_key = get_session_env("HERMES_SESSION_KEY") or session_id or ""
        chat_id = get_session_env("HERMES_SESSION_CHAT_ID") or ""

        task = args.get("task") if isinstance(args, dict) else None
        cwd_arg = args.get("cwd") if isinstance(args, dict) else None
        complexity = args.get("complexity") if isinstance(args, dict) else None
        # ``routing.choose_model`` itself is the single source of truth for
        # what counts as authorization (``allow_opus is True``, nothing
        # else) — the raw value is passed through unchanged rather than
        # coerced here, so a caller that omits it, or sends a truthy
        # non-``True`` value, is guaranteed to land on Sonnet exactly the
        # way ``routing.py``'s own tests pin down, with no second place this
        # logic could drift from that contract.
        allow_opus = args.get("allow_opus") if isinstance(args, dict) else None
        allow_fallback = isinstance(args, dict) and args.get("allow_terra_fallback") is True

        if not isinstance(task, str) or not task.strip():
            return _failure("invalid_args", "task is required")
        if not isinstance(cwd_arg, str) or not cwd_arg.strip():
            return _failure("invalid_args", "cwd is required")

        # Scope is resolved dynamically from the request itself — the SAME
        # ``project.py`` resolution the gate uses — not from a configured
        # ``gate.repo_roots`` allowlist (which no longer has any authority;
        # a stale key is parsed by the loader and never read). ``repo_roots``
        # is therefore left unset here so every nested call re-derives the
        # one canonical Git worktree root of this cwd.
        try:
            resolved_cwd = validate_cwd(cwd_arg)
        except CwdRejected as exc:
            return _failure("cwd_rejected", str(exc))

        open_classes_now = _breaker.open_classes()
        if open_classes_now:
            return _finish(_breaker_open_payload(
                task=task, cwd=resolved_cwd, open_classes=open_classes_now,
                allow_fallback=allow_fallback,
            ))

        # OAuth freshness, BEFORE any evidence gathering and before any
        # docker/Claude spawn: a credential we can already see is stale must
        # never be allowed to become an observed 401 that slams the ``auth``
        # breaker shut for a full hour. This check is deliberately
        # breaker-NEUTRAL — it never opens, clears, or resets any class (see
        # ``oauth.py``), and a failure here is a HOLD, not a recorded failure.
        auth_preflight = _oauth.preflight()
        if not auth_preflight.get("ok"):
            return _finish(_oauth_preflight_hold_payload(
                auth_preflight, cwd=resolved_cwd, open_classes=open_classes_now,
            ))

        baseline_files = _git_changed_files(resolved_cwd)
        baseline_states = _snapshot_file_states(resolved_cwd, baseline_files)

        outcome = _run_attempts(
            task=task, cwd=resolved_cwd, complexity=complexity, cfg=cfg,
            allow_opus=allow_opus,
        )

        after_files = _git_changed_files(resolved_cwd)
        after_states = _snapshot_file_states(resolved_cwd, after_files)
        files_touched = _files_changed_since_baseline(
            baseline_files, baseline_states, after_files, after_states,
        )
        review_result: Optional[Dict[str, Any]] = None
        if outcome["success"]:
            review_cfg = cfg.get("review") or {}
            if review_cfg.get("enabled", True) and should_review(
                files_touched, review_cfg.get("min_changed_files", 3)
            ):
                review_result = run_review(
                    task=task, files_touched=files_touched, summary=outcome["summary"],
                )

        validation_status, parent_verification_required, verification_result = (
            _resolve_validation_status(cfg, resolved_cwd, outcome["success"])
        )

        final_open_classes = _breaker.open_classes()
        payload: Dict[str, Any] = {
            "success": outcome["success"],
            "status": "ok" if outcome["success"] else "failed",
            "model": outcome["model"],
            "route_reason": outcome["route_reason"],
            "session_id": session_key,
            "cwd": resolved_cwd,
            "attempts": outcome["attempts"],
            "escalated": outcome["attempts"] > 1,
            "duration_ms": outcome["duration_ms"],
            "exit_code": outcome["exit_code"],
            "failure_class": None if outcome["success"] else outcome["failure_class"],
            "breaker": {"open": bool(final_open_classes), "classes": final_open_classes},
            "files_touched": files_touched,
            "summary": outcome["summary"],
            "review": review_result,
            "validation_status": validation_status,
            "parent_verification_required": parent_verification_required,
            "verification": verification_result,
        }
        if not outcome["success"]:
            if outcome["failure_class"] == _breaker.AUTH_PREFLIGHT_FAILURE_CLASS:
                # The post-spawn counterpart of ``_oauth_preflight_hold_payload``:
                # the CLI itself reported a stale/revoked OAuth SESSION (a
                # structured signal a local freshness check cannot see — the
                # credential file can look unexpired while the server has
                # already rejected it). Same operator vocabulary, same
                # "no breaker was opened, waiting will not fix this" framing.
                payload["error"] = (
                    "claude_worker failed: the Claude OAuth session appears expired "
                    "or revoked. No circuit breaker was opened — waiting will not "
                    "fix this. Re-authenticate the host Claude Code session and call "
                    "claude_worker again."
                )
            else:
                payload["error"] = (
                    f"claude_worker failed after {outcome['attempts']} attempt(s): "
                    f"{outcome['failure_class']}"
                )
        return _finish(
            payload,
            diagnostic=None if outcome["success"] else outcome.get("diagnostic"),
        )
    except Exception:
        # Anything unexpected: one redacted result, one telemetry record.
        # The exception text itself is logged, not returned — it can quote a
        # path, a command, or an environment value.
        logger.exception("claude_worker: unexpected orchestration error")
        try:
            return _failure("internal_error", _GENERIC_ERROR)
        except Exception:
            logger.exception("claude_worker: failed to emit failure result")
            return tool_result({
                "success": False, "status": "failed", "failure_class": "internal_error",
                "error": _GENERIC_ERROR,
                "validation_status": VALIDATION_STATUS_UNVERIFIED,
                "parent_verification_required": True,
                "verification": None,
            })


#: The failure class a failed OAuth freshness preflight reports. Deliberately
#: NOT the ``auth`` breaker class: ``auth`` means the API actually rejected a
#: request and a one-hour cooldown is now in force, while this means the local
#: credential was inspected and found unusable before anything was spawned and
#: NOTHING was opened. Keeping the string outside ``breaker.BREAKER_CLASSES``
#: is also structural: no current or future "record the failure class" path
#: can accidentally turn a preflight HOLD into a breaker cooldown.
OAUTH_PREFLIGHT_FAILURE_CLASS = "auth_preflight"

#: Canned, per-state operator text — a closed mapping over ``oauth``'s own
#: state constants, never the preflight's free-text ``reason``. The state is
#: provably non-secret (it is one of a handful of module literals); a reason
#: string can quote whatever an operator-installed refresh probe put in its
#: own error message, which ``oauth.redact`` can only scrub for the secrets
#: currently in the credentials file. Nothing token-shaped can reach a result
#: through a fixed literal.
_OAUTH_HOLD_DETAIL: Dict[str, str] = {
    _oauth.STATE_MISSING: "no Claude OAuth credentials file is readable on this host",
    _oauth.STATE_MALFORMED: "the Claude OAuth credentials file is malformed or oversized",
    _oauth.STATE_INVALID: "the Claude OAuth credentials file carries no usable OAuth material",
    _oauth.STATE_EXPIRED: (
        "the Claude OAuth access token is expired and carries no refresh token"
    ),
    _oauth.STATE_REFRESHABLE_EXPIRED: (
        "the Claude OAuth access token is expired and could not be refreshed in isolation"
    ),
    _oauth.STATE_ERROR: "the Claude OAuth credential could not be checked",
}


def _oauth_preflight_hold_payload(
    preflight_result: Dict[str, Any], cwd: str, open_classes: List[str],
) -> Dict[str, Any]:
    """Build the no-spawn HOLD result for a failed OAuth freshness preflight.

    Structurally identical in its safety properties to
    :func:`_breaker_open_payload` — ``attempts`` 0, no model, no exit code,
    no touched files, and every fallback-provenance field pinned to the
    locked values so ``gate._fallback_delivered`` can never read an unlock
    out of it — but with two deliberate differences:

    * ``breaker`` reports whatever the breaker ALREADY said (the empty
      classes list the caller just read), because this path must not open,
      clear, or reset any class. A stale credential is not an observed
      rejection by the API, and burning an hour of cooldown on one would be
      exactly the overreaction ``oauth.py`` exists to prevent.
    * the only OAuth material that crosses into the result is the state name
      and the ``refresh_attempted`` boolean. No token, no expiry instant, no
      scope list, no raw credential JSON, and no free-text probe reason —
      the operator-facing text is looked up from the closed
      :data:`_OAUTH_HOLD_DETAIL` mapping by state.
    """
    state = preflight_result.get("state")
    if not isinstance(state, str) or state not in _OAUTH_HOLD_DETAIL:
        state = _oauth.STATE_ERROR
    return {
        "success": False,
        "status": "HOLD",
        "attempts": 0,
        "escalated": False,
        "failure_class": OAUTH_PREFLIGHT_FAILURE_CLASS,
        "breaker": {"open": bool(open_classes), "classes": list(open_classes)},
        "model": "", "route_reason": "", "exit_code": None,
        "duration_ms": 0, "files_touched": [], "cwd": cwd,
        "oauth_state": state,
        "oauth_refresh_attempted": preflight_result.get("refresh_attempted") is True,
        "fallback_ready": False,
        "fallback_requested": False,
        "fallback_provenance": None,
        "fallback": None,
        "error": (
            f"claude_worker unavailable: {_OAUTH_HOLD_DETAIL[state]}. This session "
            "remains on HOLD: direct edits stay gated. No worker was spawned and no "
            "circuit breaker was opened — re-authenticate the host Claude Code "
            "session and call claude_worker again."
        ),
    }


def _breaker_open_payload(
    task: str, cwd: str, open_classes: List[str], allow_fallback: bool,
) -> Dict[str, Any]:
    """Build the no-spawn result for an open breaker.

    Default is an explicit HOLD: the worker is unavailable and the session
    stays gated. The contingency is opt-in per call — ``allow_fallback`` here
    must already be the strict ``args.get("allow_terra_fallback") is True``
    check done by the caller, never a truthiness coercion — and only counts
    when Terra actually answers with a non-empty result. A successful
    fallback carries explicit, unforgeable-by-coincidence provenance
    (``fallback_requested=True``, ``fallback_provenance="terra_auxiliary"``,
    and a nested ``fallback.ready=True``) so ``gate._fallback_delivered``
    can require every one of those fields independently rather than trusting
    ``fallback_ready`` alone; a Terra failure leaves the hold in place.
    """
    reason = f"breaker open: {', '.join(open_classes)}"
    payload: Dict[str, Any] = {
        "success": False,
        "attempts": 0,
        "escalated": False,
        "failure_class": "breaker_open",
        "breaker": {"open": True, "classes": list(open_classes)},
        "model": "", "route_reason": "", "exit_code": None,
        "duration_ms": 0, "files_touched": [], "cwd": cwd,
        "fallback_ready": False,
        "fallback_requested": False,
        "fallback_provenance": None,
        "fallback": None,
    }

    if allow_fallback is not True:
        payload["status"] = "HOLD"
        payload["error"] = (
            f"claude_worker unavailable ({reason}). This session remains on HOLD: "
            "direct edits stay gated. Re-call with allow_terra_fallback=true to "
            "request explicit Terra fallback guidance."
        )
        return payload

    payload["fallback_requested"] = True
    fallback = run_fallback(task=task, reason=reason)
    notes = fallback.get("notes") if isinstance(fallback, dict) else None
    delivered = (
        isinstance(fallback, dict)
        and fallback.get("ok") is True
        and isinstance(notes, str)
        and bool(notes.strip())
    )
    if delivered:
        payload["status"] = "fallback_ready"
        payload["fallback_ready"] = True
        payload["fallback_provenance"] = "terra_auxiliary"
        payload["fallback"] = {"ok": True, "ready": True, "notes": notes}
        payload["summary"] = notes
    else:
        payload["status"] = "HOLD"
        payload["fallback"] = fallback if isinstance(fallback, dict) else None
        payload["error"] = (
            f"claude_worker unavailable ({reason}) and the Terra fallback did not "
            "return a result. This session remains on HOLD."
        )
    return payload


def _run_attempts(
    task: str, cwd: str, complexity: Optional[str], cfg: Dict[str, Any],
    allow_opus: Any = False,
) -> Dict[str, Any]:
    """Run at most ``policy.MAX_ATTEMPTS`` spawns and summarize the outcome.

    The cap is the ``range`` bound above, which is ``1`` — there is no
    failure-driven retry or escalation (see ``policy.MAX_ATTEMPTS`` and
    ``routing.py``) — and is not reachable from configuration.
    """
    timeout_seconds = (cfg.get("isolation") or {}).get("timeout_seconds", 900)

    attempts = 0
    previous_failure_class: Optional[str] = None
    model = ""
    route_reason = ""
    exit_code: Optional[int] = None
    duration_ms = 0
    success = False
    summary = ""
    diagnostic: Optional[Dict[str, Any]] = None

    for attempt in range(_policy.MAX_ATTEMPTS):
        if attempt > 0 and _breaker.is_open():
            break

        remaining_budget_seconds = max(
            0.0,
            _policy.MAX_TOTAL_ATTEMPT_SECONDS - (duration_ms / 1000.0),
        )
        if remaining_budget_seconds <= 0:
            break
        attempt_timeout_seconds = min(
            float(timeout_seconds), remaining_budget_seconds,
        )

        try:
            # ``routing.choose_model`` takes no ``attempt``/
            # ``previous_failure_class`` — there is no failure-driven
            # escalation any more (see ``policy.MAX_ATTEMPTS`` and
            # ``routing.py``'s own docstring): every call here routes
            # identically, and the ``range`` bound above is the only cap.
            # ``allow_opus`` is passed through verbatim: only the literal
            # ``True`` (never a truthy non-bool) combined with an allowed
            # ``complexity`` can select Opus — see ``routing.choose_model``.
            model, route_reason = _routing.choose_model(
                task=task, complexity=complexity, allow_opus=allow_opus,
            )
        except RuntimeError:
            # Defensive: routing.choose_model does not currently raise this,
            # but a refusal here must still stop the loop rather than spawn
            # with an unresolved model.
            break

        try:
            spawn_result = spawn_claude(
                task=task, cwd=cwd, model=model,
                timeout_seconds=attempt_timeout_seconds,
            )
        except SpawnRefused as exc:
            attempts += 1
            previous_failure_class = "isolation_refused"
            exit_code = None
            logger.warning("claude_worker: spawn refused: %s", exc)
            break

        attempts += 1
        exit_code = spawn_result.get("exit_code")
        duration_ms += int(spawn_result.get("duration_ms") or 0)
        stdout = spawn_result.get("stdout", "") or ""
        stderr = spawn_result.get("stderr", "") or ""
        timed_out = bool(spawn_result.get("timed_out"))

        if not timed_out and exit_code == 0:
            summary = _parse_worker_summary(stdout)
            success = True
            previous_failure_class = None
            break

        previous_failure_class = (
            "timeout" if timed_out else _breaker.classify_failure(exit_code, stderr, stdout)
        )
        # Bounded, redacted evidence for THIS attempt's failure — see the
        # module-level "Bounded, redacted failure diagnostics" section below.
        # Overwritten (never accumulated) on every failing attempt, so only
        # the most recent attempt's evidence survives to the caller.
        diagnostic = _diagnostics_from_spawn(stdout, stderr)
        if previous_failure_class in _breaker.BREAKER_CLASSES:
            _breaker.record_failure(
                previous_failure_class,
                (cfg.get("breaker") or {}).get("cooldown_seconds") or {},
            )
            break
        # "other"/"timeout": fall through to the single escalation attempt.

    return {
        "success": success,
        "attempts": attempts,
        "model": model,
        "route_reason": route_reason,
        "exit_code": exit_code,
        "duration_ms": duration_ms,
        "failure_class": previous_failure_class,
        "summary": summary,
        "diagnostic": None if success else diagnostic,
    }
