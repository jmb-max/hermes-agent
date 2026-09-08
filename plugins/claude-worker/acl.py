"""Named POSIX ACL verification for the worktree mount root.

The operational failure this closes: an operator needs the fixed non-root
sandbox identity (``policy.SANDBOX_UID``) to be able to write inside a
root-owned worktree, and reaches for ``chmod 775``/``chmod 777`` on the repo
root to get there. Both are refused (``isolation_refused``) — 777 because it
is world-writable, and 775 because a bare group-write bit cannot be trusted:
it grants write to EVERY member of the owning group, not just the sandbox
uid, and ``runner.py`` has no way to know who else is in that group.

The supported provisioning method is a NAMED POSIX ACL for
``policy.SANDBOX_UID`` on the root-owned worktree root: an explicit
``ACL_USER`` entry for that one uid, with the real owning-group entry
(``ACL_GROUP_OBJ``) and the ``ACL_OTHER`` entry left at read+execute (no
write). The subtlety this module exists to handle: once a directory carries
any ACL entries beyond the traditional owner/group/other three, POSIX
requires the kernel to report the ACL_MASK entry — not the real
ACL_GROUP_OBJ permission — in the "group" bits ``stat``/``ls -l`` show. A
worktree provisioned exactly as recommended can therefore legitimately show
mode ``755`` or ``775`` depending on the mask, and a bare mode check cannot
tell that apart from an operator who actually ran ``chmod g+w``. Only
reading the ACL itself can.

This module never shells out to ``getfacl``/``setfacl`` — it reads the two
ACL extended attributes directly (``system.posix_acl_access`` for the
directory itself, ``system.posix_acl_default`` for what new files inside it
inherit) and parses the kernel's fixed binary encoding
(``linux/posix_acl_xattr.h``): a 4-byte little-endian version, followed by
8-byte entries of ``(tag: u16, perm: u16, id: u32)``, all little-endian.
Parsing is defensive: anything that doesn't match this exact shape is
treated as "no usable ACL" rather than raising, which keeps the runner's
mount-root check fail-closed.

Recommended provisioning (see ``recommended_setfacl_commands``):

    setfacl -m u:10001:rwx /path/to/worktree
    setfacl -d -m u:10001:rwx /path/to/worktree   # default ACL for new files

Only ``runner.py`` treats the outcome here as security-relevant (refusing a
spawn when a group-writable mount root is not explained by a verified named
ACL). Everything else here is read-only introspection.
"""

from __future__ import annotations

import os
import struct
from typing import Dict, List, NamedTuple, Optional, Tuple

#: ACL entry tags, from the kernel's ``include/uapi/linux/posix_acl.h``.
ACL_USER_OBJ = 0x01
ACL_USER = 0x02
ACL_GROUP_OBJ = 0x04
ACL_GROUP = 0x08
ACL_MASK = 0x10
ACL_OTHER = 0x20

#: Permission bits within one entry — same numbering as a normal Unix mode's
#: low three bits.
PERM_READ = 0x4
PERM_WRITE = 0x2
PERM_EXECUTE = 0x1

_ACCESS_XATTR = "system.posix_acl_access"
_DEFAULT_XATTR = "system.posix_acl_default"

#: The only version this parser understands. A different version is treated
#: as unparseable (fail closed), not guessed at.
_ACL_XATTR_VERSION = 0x0002

_HEADER = struct.Struct("<I")
_ENTRY = struct.Struct("<HHI")


class AclEntry(NamedTuple):
    tag: int
    perm: int
    entry_id: Optional[int]  # None for tags that carry no id (OBJ/MASK/OTHER)


def _getxattr(path: str, name: str) -> Optional[bytes]:
    """Read one extended attribute, or ``None`` if it is absent/unsupported.

    Never raises: a missing xattr, a filesystem with no xattr support, and a
    permission error are all indistinguishable from "no ACL is provisioned"
    for this module's purposes, and the caller (``runner.py``) fails closed
    on that outcome anyway.
    """
    getter = getattr(os, "getxattr", None)
    if getter is None:  # pragma: no cover - non-Linux platform
        return None
    try:
        return getter(path, name, follow_symlinks=False)
    except OSError:
        return None
    except ValueError:
        return None


def parse_acl_blob(blob: bytes) -> List[AclEntry]:
    """Parse one ``system.posix_acl_{access,default}`` xattr value.

    Raises ``ValueError`` for anything that is not exactly the expected
    fixed binary shape: wrong version, a header/entry that doesn't fit, or
    trailing bytes that are not a whole number of entries. Callers treat a
    ``ValueError`` as "no usable ACL" rather than propagating it.
    """
    if len(blob) < _HEADER.size:
        raise ValueError("ACL blob shorter than the version header")
    (version,) = _HEADER.unpack_from(blob, 0)
    if version != _ACL_XATTR_VERSION:
        raise ValueError(f"unsupported ACL xattr version: {version!r}")
    body = blob[_HEADER.size:]
    if len(body) % _ENTRY.size != 0:
        raise ValueError("ACL blob body is not a whole number of entries")

    entries: List[AclEntry] = []
    for offset in range(0, len(body), _ENTRY.size):
        tag, perm, entry_id = _ENTRY.unpack_from(body, offset)
        has_id = tag in (ACL_USER, ACL_GROUP)
        entries.append(AclEntry(tag=tag, perm=perm, entry_id=entry_id if has_id else None))
    return entries


def read_acl(path: str, *, default: bool = False) -> Optional[List[AclEntry]]:
    """Parsed ACL entries for *path*, or ``None`` if no usable ACL is present.

    *default* selects ``system.posix_acl_default`` (what new files created
    inside a directory inherit) instead of ``system.posix_acl_access``.
    Malformed xattr content is treated exactly like an absent one — this
    reads state, it never raises. ``_getxattr`` itself never raises, but
    this call is wrapped defensively too — a caller that swaps it out
    (tests, or a future refactor) must not be able to turn a read of ACL
    state into an uncaught exception that breaks the fail-closed contract
    ``runner.py`` depends on.
    """
    try:
        blob = _getxattr(path, _DEFAULT_XATTR if default else _ACCESS_XATTR)
    except OSError:
        return None
    if blob is None:
        return None
    try:
        return parse_acl_blob(blob)
    except ValueError:
        return None


class AclAssessment(NamedTuple):
    acl_present: bool
    user_perms: Dict[int, int]
    group_obj_perm: Optional[int]
    other_perm: Optional[int]
    mask_perm: Optional[int]


def _assess(entries: Optional[List[AclEntry]]) -> AclAssessment:
    if not entries:
        return AclAssessment(False, {}, None, None, None)
    user_perms: Dict[int, int] = {}
    group_obj_perm: Optional[int] = None
    other_perm: Optional[int] = None
    mask_perm: Optional[int] = None
    for entry in entries:
        if entry.tag == ACL_USER and entry.entry_id is not None:
            user_perms[entry.entry_id] = entry.perm
        elif entry.tag == ACL_GROUP_OBJ:
            group_obj_perm = entry.perm
        elif entry.tag == ACL_OTHER:
            other_perm = entry.perm
        elif entry.tag == ACL_MASK:
            mask_perm = entry.perm
    return AclAssessment(True, user_perms, group_obj_perm, other_perm, mask_perm)


def describe(path: str) -> AclAssessment:
    """Best-effort description of *path*'s access ACL (never raises)."""
    return _assess(read_acl(path, default=False))


def evaluate_sandbox_uid_provisioning(path: str, uid: int) -> Tuple[bool, str]:
    """Is *path*'s group-writable mode bit explained by a verified named
    POSIX ACL that grants write to *uid* only?

    Returns ``(safe, reason)``. ``safe`` is True only when ALL of:

    * an access ACL is actually present (not merely a bare ``chmod g+w``,
      which carries no ACL xattr at all);
    * it grants ``uid`` a named ``ACL_USER`` entry that includes write;
    * the ACL mask (if present) does not clip that grant back below write;
    * the REAL owning-group entry (``ACL_GROUP_OBJ`` — never the mask bits
      ``stat`` may substitute for it) does not itself include write; and
    * the ``ACL_OTHER`` entry does not include write.

    Any other shape — no ACL at all, no entry for this uid, a group or
    other entry that is itself writable, or a mask that would restrict the
    uid's grant — is unsafe: the group-write bit is either unexplained or
    grants more than the one intended uid.
    """
    info = describe(path)
    if not info.acl_present:
        return False, (
            f"{path!r} has a group-writable mode bit but no POSIX ACL is present — "
            "a bare chmod grants every member of the owning group, not just the "
            "sandbox uid; provision access with a named ACL instead (see README)"
        )

    uid_perm = info.user_perms.get(uid)
    if uid_perm is None:
        return False, f"{path!r} has no named ACL entry for uid {uid}"
    if not (uid_perm & PERM_WRITE):
        return False, f"{path!r}'s named ACL entry for uid {uid} does not grant write"

    if info.mask_perm is not None and (info.mask_perm & uid_perm) != uid_perm:
        return False, (
            f"{path!r}'s ACL mask ({info.mask_perm:#o}) would restrict uid {uid}'s "
            f"granted permission ({uid_perm:#o})"
        )

    if info.group_obj_perm is not None and (info.group_obj_perm & PERM_WRITE):
        return False, (
            f"{path!r}'s real owning-group ACL entry grants write — provisioning "
            f"must scope write to uid {uid} only, not the whole group"
        )

    if info.other_perm is not None and (info.other_perm & PERM_WRITE):
        return False, f"{path!r}'s ACL 'other' entry grants write"

    return True, f"{path!r} carries a verified named ACL scoping write to uid {uid} only"


def default_acl_grants_uid(path: str, uid: int) -> bool:
    """Whether *path*'s DEFAULT ACL (inherited by new files/directories
    created inside it) already grants *uid* access — informational only,
    never security-blocking: files the sandbox itself creates are already
    owned by ``SANDBOX_UID``, so an absent default ACL only matters for
    files something else creates later."""
    info = _assess(read_acl(path, default=True))
    perm = info.user_perms.get(uid)
    return perm is not None and bool(perm & PERM_WRITE)


def recommended_setfacl_commands(path: str, uid: int) -> List[str]:
    """Operator-facing ``setfacl`` commands that provision *path* the
    supported way: a named access ACL for *uid*, plus a default ACL so new
    files created later inherit the same grant. Documentation only — never
    executed by this plugin."""
    return [
        f"setfacl -m u:{uid}:rwx {path}",
        f"setfacl -d -m u:{uid}:rwx {path}",
    ]
