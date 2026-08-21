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
  * ``validate_cwd`` — resolves symlinks (``os.path.realpath``) BEFORE the
    allowlist check, so a symlink planted inside an allowlisted root that
    points outside it cannot smuggle the worker out of scope. The same
    realpath-before-mount discipline applies to the repo path handed to the
    docker command builders.
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
import shutil
import stat
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from . import breaker as _breaker
from . import config as _config
from . import policy as _policy
from . import routing as _routing
from . import telemetry as _telemetry
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
    """Raised when the requested cwd is missing or outside every allowlisted root."""


def validate_cwd(cwd: str, repo_roots: List[str]) -> str:
    """Resolve *cwd* and ensure it is inside one of *repo_roots*.

    Resolves symlinks BEFORE the allowlist check so a symlink that escapes
    an allowlisted root is rejected rather than followed.
    """
    if not cwd:
        raise CwdRejected("cwd is required")
    real = os.path.realpath(cwd)
    if not os.path.isdir(real):
        raise CwdRejected(f"cwd does not exist or is not a directory: {cwd!r}")
    for root in repo_roots or []:
        try:
            real_root = os.path.realpath(root)
        except (OSError, ValueError):
            continue
        if real == real_root or real.startswith(real_root.rstrip(os.sep) + os.sep):
            return real
    raise CwdRejected(f"cwd {cwd!r} is outside every allowlisted repo root")


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
    """Return the most-specific canonical configured root containing cwd."""
    matches: List[str] = []
    for root in repo_roots or []:
        try:
            real_root = os.path.realpath(root)
        except (OSError, ValueError):
            continue
        if real_cwd == real_root or real_cwd.startswith(real_root.rstrip(os.sep) + os.sep):
            matches.append(real_root)
    return max(matches, key=len) if matches else None


def _snapshot_mount_root(
    mount_root: str,
) -> Dict[str, Tuple[int, int, int, int, int]]:
    """Prove dockerd's late pathname lookup cannot be redirected by a
    non-root actor.

    Every component is a real root-owned directory. Parents must not be
    writable by group/other, except a sticky directory (e.g. /tmp), whose
    sticky semantics prevent a non-owner from replacing the root-owned next
    component. The mount root itself may expose a group-write mode bit from
    the explicitly provisioned UID-10001 ACL mask, but it must remain
    root-owned and never world-writable. Replacing that final entry requires
    write access to its protected parent, not write access inside the repo.
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
    """Return protected host mount root, in-container cwd, and root snapshot."""
    roots = repo_roots if repo_roots else [real_cwd]
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

    When *repo_roots* is not supplied, the resolved cwd is treated as its
    own sole allowed root — this preserves a direct ``spawn_claude`` caller
    that has already validated containment itself (every isolation test in
    this suite that predates ``repo_roots``), while the chain/symlink
    protection below still applies unconditionally.

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

    roots = repo_roots if repo_roots else [cwd]
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


class TrustViolation(RuntimeError):
    """A component of a trusted absolute path chain failed validation:
    missing, a symlink, the wrong type, not owned by ``_TRUSTED_UID``, or
    writable by anyone but its owner (sticky-bit directories like ``/tmp``
    excepted). Raised by ``_validate_trusted_path_chain`` and by the
    identity re-check done immediately before a subprocess call."""


#: The only uid ever allowed to own a component of a trusted path chain.
_TRUSTED_UID = 0


def _parent_dirs(path: str) -> List[str]:
    """Every directory component of an absolute *path*, from ``/`` up to
    (but not including) the final component itself."""
    parts = [p for p in path.split(os.sep) if p][:-1]
    parents = [os.sep]
    current = os.sep
    for part in parts:
        current = os.path.join(current, part)
        parents.append(current)
    return parents


def _stat_identity(st: os.stat_result) -> Tuple[int, int, int, int, int]:
    """The subset of ``stat`` fields that together identify one exact
    filesystem object: device, inode, mode, owner, group. A replacement —
    even one with the same path, same size, same content — changes at
    least the inode."""
    return (st.st_dev, st.st_ino, st.st_mode, st.st_uid, st.st_gid)


def _writable_by_others(mode: int) -> bool:
    """True if *mode* is group- or world-writable, EXCEPT a world-writable
    directory that also has the sticky bit set (``/tmp`` and friends): the
    sticky bit means only a file's own owner may rename or delete it, which
    is the standard, safe multi-writer directory convention every major OS
    and security tool (sshd, sudo, PAM) already treats as non-hostile."""
    if mode & stat.S_ISVTX:
        return False
    return bool(mode & (stat.S_IWGRP | stat.S_IWOTH))


def _check_dir_component(path: str) -> os.stat_result:
    """Validate one directory component of a trusted path chain: must
    exist, must not be a symlink, must be a real directory, must be owned
    by ``_TRUSTED_UID``, and must not be writable by anyone but its owner
    (subject to the sticky-bit exception above)."""
    try:
        st = os.lstat(path)
    except OSError as exc:
        raise TrustViolation(
            f"cannot stat trusted path component {path!r}: {exc}"
        ) from exc
    if stat.S_ISLNK(st.st_mode):
        raise TrustViolation(f"trusted path component is a symlink: {path!r}")
    if not stat.S_ISDIR(st.st_mode):
        raise TrustViolation(f"trusted path component is not a directory: {path!r}")
    if st.st_uid != _TRUSTED_UID:
        raise TrustViolation(
            f"trusted path component is not owned by uid {_TRUSTED_UID}: {path!r}"
        )
    if _writable_by_others(st.st_mode):
        raise TrustViolation(
            f"trusted path component is group/world-writable: {path!r}"
        )
    return st


def _check_final_component(path: str, *, executable: bool) -> os.stat_result:
    """Validate the final (non-directory) component of a trusted path
    chain: must exist, must not be a symlink, must be exactly a regular
    file, owned by ``_TRUSTED_UID``, never group/world-writable, and (when
    *executable*) owner-executable."""
    try:
        st = os.lstat(path)
    except OSError as exc:
        raise TrustViolation(f"cannot stat trusted path {path!r}: {exc}") from exc
    if stat.S_ISLNK(st.st_mode):
        raise TrustViolation(f"trusted path is a symlink: {path!r}")
    if not stat.S_ISREG(st.st_mode):
        raise TrustViolation(f"trusted path is not a regular file: {path!r}")
    if st.st_uid != _TRUSTED_UID:
        raise TrustViolation(
            f"trusted path is not owned by uid {_TRUSTED_UID}: {path!r}"
        )
    if st.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
        raise TrustViolation(f"trusted path is group/world-writable: {path!r}")
    if executable and not (st.st_mode & stat.S_IXUSR):
        raise TrustViolation(f"trusted path is not owner-executable: {path!r}")
    return st


def _validate_trusted_path_chain(
    path: str, *, executable: bool = False
) -> Dict[str, Tuple[int, int, int, int, int]]:
    """Validate every component of an absolute trusted *path* and return
    an identity snapshot — ``{component: (dev, ino, mode, uid, gid)}`` —
    for every parent directory plus the final component, so a caller can
    re-check the exact same objects immediately before using the path.

    Fail-closed on the first violation: a missing component, a symlink
    anywhere in the chain, a parent that is not a real directory, a final
    component that is not exactly a regular file, anything not owned by
    ``_TRUSTED_UID`` (root), or anything writable by anyone but its owner
    (sticky-bit directories excepted) all raise ``TrustViolation``.
    """
    if not os.path.isabs(path):
        raise TrustViolation(f"trusted path is not absolute: {path!r}")

    snapshot: Dict[str, Tuple[int, int, int, int, int]] = {}
    for parent in _parent_dirs(path):
        st = _check_dir_component(parent)
        snapshot[parent] = _stat_identity(st)

    final_st = _check_final_component(path, executable=executable)
    snapshot[path] = _stat_identity(final_st)
    return snapshot


def _revalidate_trusted_path_chain(
    snapshot: Dict[str, Tuple[int, int, int, int, int]],
) -> None:
    """Re-``lstat`` every component recorded in *snapshot* and compare its
    full identity — device, inode, mode, uid, gid — against the value
    captured at validation time. A component that vanished, was replaced
    (even by another equally "valid" object — the inode differs), was
    symlinked, or had its mode changed since validation raises
    ``TrustViolation`` rather than trusting a path that may have moved
    underneath the earlier check."""
    for component, expected in snapshot.items():
        try:
            st = os.lstat(component)
        except OSError as exc:
            raise TrustViolation(
                f"trusted path component vanished before use: {component!r}: {exc}"
            ) from exc
        if _stat_identity(st) != expected:
            raise TrustViolation(
                f"trusted path component identity changed before use: {component!r}"
            )


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
    ``_remove_staging_dir``) in a ``finally`` on every exit path.
    """
    _validate_staging_root()
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


def _force_remove_container(
    cidfile: Path, env: Optional[Dict[str, str]] = None,
) -> None:
    """Best-effort removal of the exact container created by one spawn.

    Docker's client-side timeout kills ``docker run`` but does not guarantee
    that the daemon-side container exits. The cidfile lives inside the
    root-owned per-spawn staging directory and must contain a full 64-character
    hexadecimal container id; task/model input can never select the removal
    target. Cleanup never masks the worker's real result.
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
        try:
            _validate_docker_binary_trust()
            subprocess.run(
                [*_docker_base_argv(), "rm", "--force", container_id],
                capture_output=True,
                text=True,
                timeout=15,
                env=env or build_child_env(),
            )
        except Exception:
            logger.exception("claude_worker: failed to remove worker container %s", container_id)
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
    ``_revalidate_repo_cwd_chain``). *repo_roots* should be the SAME
    canonical roots the caller resolved *cwd* against (``_run_attempts``
    passes ``policy.canonical_repo_roots(cfg)``); omitting it treats the
    resolved cwd as its own sole allowed root, for callers that already
    validated containment themselves.
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
# Orchestration — the claude_worker tool handler
# ---------------------------------------------------------------------------


_GENERIC_ERROR = (
    "claude_worker failed with an internal error; details were withheld from "
    "the result and the run was recorded in telemetry."
)


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
    fallback, auto model routing, an isolated spawn capped structurally at
    ``policy.MAX_ATTEMPTS``, evidence assembly, exactly one telemetry
    record, and an advisory Terra review for substantial successful
    changes. Returns a JSON string (the ``tools.registry.tool_result``
    contract) — deterministic, no raw prompts/stdout embedded.

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

    def _finish(payload: Dict[str, Any]) -> str:
        payload.setdefault("session_id", session_key)
        payload.setdefault("cwd", cwd_arg if isinstance(cwd_arg, str) else "")
        payload.setdefault("review", None)
        payload.setdefault("status", "ok" if payload.get("success") else "failed")
        payload.setdefault("fallback_ready", False)
        payload.setdefault("fallback", None)
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
        allow_fallback = isinstance(args, dict) and args.get("allow_terra_fallback") is True

        if not isinstance(task, str) or not task.strip():
            return _failure("invalid_args", "task is required")
        if not isinstance(cwd_arg, str) or not cwd_arg.strip():
            return _failure("invalid_args", "cwd is required")

        repo_roots = _policy.canonical_repo_roots(cfg)
        try:
            resolved_cwd = validate_cwd(cwd_arg, repo_roots)
        except CwdRejected as exc:
            return _failure("cwd_rejected", str(exc))

        open_classes_now = _breaker.open_classes()
        if open_classes_now:
            return _finish(_breaker_open_payload(
                task=task, cwd=resolved_cwd, open_classes=open_classes_now,
                allow_fallback=allow_fallback,
            ))

        baseline_files = _git_changed_files(resolved_cwd, repo_roots=repo_roots)
        baseline_states = _snapshot_file_states(resolved_cwd, baseline_files)

        outcome = _run_attempts(
            task=task, cwd=resolved_cwd, complexity=complexity, cfg=cfg,
        )

        after_files = _git_changed_files(resolved_cwd, repo_roots=repo_roots)
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
        }
        if not outcome["success"]:
            payload["error"] = (
                f"claude_worker failed after {outcome['attempts']} attempt(s): "
                f"{outcome['failure_class']}"
            )
        return _finish(payload)
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
            })


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
) -> Dict[str, Any]:
    """Run at most ``policy.MAX_ATTEMPTS`` spawns and summarize the outcome.

    The cap is the ``range`` bound AND ``routing.choose_model``'s own
    structural refusal — two independent enforcements of "no third spawn",
    neither of which is reachable from configuration.
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
            model, route_reason = _routing.choose_model(
                task=task, attempt=attempt, complexity=complexity,
                previous_failure_class=previous_failure_class,
            )
        except RuntimeError:
            # Routing refused this attempt (cap reached, breaker-class
            # failure, or a task that already started on Opus).
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
    }
