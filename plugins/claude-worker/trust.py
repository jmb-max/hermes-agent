"""Trusted absolute path-chain validation, shared by every claude_worker
component that is about to execute or mount a fixed host path.

The docker binary (``runner``), the git binary (``project``), and the OAuth
credentials file (``runner``) are all fixed HOST paths outside the caller's
control, and all three are validated the same way before every use, never
merely once at startup: every component of the absolute path — each parent
directory AND the final component — must be a real, non-symlink object owned
by root and never writable by anyone but its owner (a sticky-bit directory
like ``/tmp`` excepted).

This lives in its own module rather than inside ``runner`` so that
``project.py`` can validate the git binary with the exact same primitives
instead of growing a second, subtly different copy of them. ``runner`` keeps
private aliases for backward compatibility with callers (and tests) that
reference ``runner._validate_trusted_path_chain`` and friends.

Note the deliberate asymmetry with the repo cwd chain in ``runner``: a repo
checkout is ordinarily owned by the calling user, so ownership is NOT
enforced there — only non-symlink-ness and identity stability. Ownership is
enforced *here* precisely because these paths are not the caller's to own.
"""

from __future__ import annotations

import os
import stat
from typing import Dict, List, Tuple


class TrustViolation(RuntimeError):
    """A component of a trusted absolute path chain failed validation:
    missing, a symlink, the wrong type, not owned by ``TRUSTED_UID``, or
    writable by anyone but its owner (sticky-bit directories like ``/tmp``
    excepted). Raised by :func:`validate_trusted_path_chain` and by the
    identity re-check done immediately before a subprocess call."""


#: The only uid ever allowed to own a component of a trusted path chain.
TRUSTED_UID = 0

#: ``{component: (dev, ino, mode, uid, gid)}``
Snapshot = Dict[str, Tuple[int, int, int, int, int]]


def parent_dirs(path: str) -> List[str]:
    """Every directory component of an absolute *path*, from ``/`` up to
    (but not including) the final component itself."""
    parts = [p for p in path.split(os.sep) if p][:-1]
    parents = [os.sep]
    current = os.sep
    for part in parts:
        current = os.path.join(current, part)
        parents.append(current)
    return parents


def stat_identity(st: os.stat_result) -> Tuple[int, int, int, int, int]:
    """The subset of ``stat`` fields that together identify one exact
    filesystem object: device, inode, mode, owner, group. A replacement —
    even one with the same path, same size, same content — changes at
    least the inode."""
    return (st.st_dev, st.st_ino, st.st_mode, st.st_uid, st.st_gid)


def writable_by_others(mode: int) -> bool:
    """True if *mode* is group- or world-writable, EXCEPT a world-writable
    directory that also has the sticky bit set (``/tmp`` and friends): the
    sticky bit means only a file's own owner may rename or delete it, which
    is the standard, safe multi-writer directory convention every major OS
    and security tool (sshd, sudo, PAM) already treats as non-hostile."""
    if mode & stat.S_ISVTX:
        return False
    return bool(mode & (stat.S_IWGRP | stat.S_IWOTH))


def check_dir_component(path: str) -> os.stat_result:
    """Validate one directory component of a trusted path chain: must
    exist, must not be a symlink, must be a real directory, must be owned
    by ``TRUSTED_UID``, and must not be writable by anyone but its owner
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
    if st.st_uid != TRUSTED_UID:
        raise TrustViolation(
            f"trusted path component is not owned by uid {TRUSTED_UID}: {path!r}"
        )
    if writable_by_others(st.st_mode):
        raise TrustViolation(
            f"trusted path component is group/world-writable: {path!r}"
        )
    return st


def check_final_component(path: str, *, executable: bool) -> os.stat_result:
    """Validate the final (non-directory) component of a trusted path
    chain: must exist, must not be a symlink, must be exactly a regular
    file, owned by ``TRUSTED_UID``, never group/world-writable, and (when
    *executable*) owner-executable."""
    try:
        st = os.lstat(path)
    except OSError as exc:
        raise TrustViolation(f"cannot stat trusted path {path!r}: {exc}") from exc
    if stat.S_ISLNK(st.st_mode):
        raise TrustViolation(f"trusted path is a symlink: {path!r}")
    if not stat.S_ISREG(st.st_mode):
        raise TrustViolation(f"trusted path is not a regular file: {path!r}")
    if st.st_uid != TRUSTED_UID:
        raise TrustViolation(
            f"trusted path is not owned by uid {TRUSTED_UID}: {path!r}"
        )
    if st.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
        raise TrustViolation(f"trusted path is group/world-writable: {path!r}")
    if executable and not (st.st_mode & stat.S_IXUSR):
        raise TrustViolation(f"trusted path is not owner-executable: {path!r}")
    return st


def validate_trusted_path_chain(path: str, *, executable: bool = False) -> Snapshot:
    """Validate every component of an absolute trusted *path* and return
    an identity snapshot — ``{component: (dev, ino, mode, uid, gid)}`` —
    for every parent directory plus the final component, so a caller can
    re-check the exact same objects immediately before using the path.

    Fail-closed on the first violation: a missing component, a symlink
    anywhere in the chain, a parent that is not a real directory, a final
    component that is not exactly a regular file, anything not owned by
    ``TRUSTED_UID`` (root), or anything writable by anyone but its owner
    (sticky-bit directories excepted) all raise ``TrustViolation``.
    """
    if not os.path.isabs(path):
        raise TrustViolation(f"trusted path is not absolute: {path!r}")

    snapshot: Snapshot = {}
    for parent in parent_dirs(path):
        st = check_dir_component(parent)
        snapshot[parent] = stat_identity(st)

    final_st = check_final_component(path, executable=executable)
    snapshot[path] = stat_identity(final_st)
    return snapshot


def revalidate_trusted_path_chain(snapshot: Snapshot) -> None:
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
        if stat_identity(st) != expected:
            raise TrustViolation(
                f"trusted path component identity changed before use: {component!r}"
            )
