"""Tests for ``plugins/claude-worker/acl.py``.

Covers the "isolation_refused after chmod 775/777" operational gap: a bare
group-write mode bit must never be trusted as sandbox-uid provisioning, and
a genuine named POSIX ACL for the sandbox uid — with the real group/other
entries left non-writable — must be recognized as safe even though ``stat``
may show the same group-writable-looking mode bits (the ACL mask).

These tests build the kernel's fixed binary ACL xattr encoding by hand
(version header + 8-byte ``(tag, perm, id)`` entries) and monkeypatch
``acl._getxattr`` directly, so they need no real filesystem ACL support —
tmpfs/overlay test environments frequently have none.
"""

from __future__ import annotations

import struct

import pytest

from tests.plugins._claude_worker_helpers import load_submodule

acl = load_submodule("acl")

_HEADER = struct.Struct("<I")
_ENTRY = struct.Struct("<HHI")

_SANDBOX_UID = 10001
_NOID = 0xFFFFFFFF


def _blob(entries):
    """Build a raw ``system.posix_acl_access``-shaped xattr value.

    *entries* is a list of ``(tag, perm, id_or_none)``.
    """
    out = _HEADER.pack(acl._ACL_XATTR_VERSION)
    for tag, perm, entry_id in entries:
        out += _ENTRY.pack(tag, perm, _NOID if entry_id is None else entry_id)
    return out


def _install_xattrs(monkeypatch, access=None, default=None):
    values = {}
    if access is not None:
        values[acl._ACCESS_XATTR] = access
    if default is not None:
        values[acl._DEFAULT_XATTR] = default

    def _fake(path, name):
        return values.get(name)

    monkeypatch.setattr(acl, "_getxattr", _fake)


class TestParseAclBlob:
    def test_parses_well_formed_blob(self):
        blob = _blob([
            (acl.ACL_USER_OBJ, 0o7, None),
            (acl.ACL_USER, 0o7, _SANDBOX_UID),
            (acl.ACL_GROUP_OBJ, 0o5, None),
            (acl.ACL_MASK, 0o7, None),
            (acl.ACL_OTHER, 0o5, None),
        ])
        entries = acl.parse_acl_blob(blob)
        assert len(entries) == 5
        user_entry = next(e for e in entries if e.tag == acl.ACL_USER)
        assert user_entry.entry_id == _SANDBOX_UID
        assert user_entry.perm == 0o7

    def test_wrong_version_raises(self):
        blob = _HEADER.pack(0x0099) + _ENTRY.pack(acl.ACL_USER_OBJ, 0o7, _NOID)
        with pytest.raises(ValueError):
            acl.parse_acl_blob(blob)

    def test_truncated_header_raises(self):
        with pytest.raises(ValueError):
            acl.parse_acl_blob(b"\x00\x00")

    def test_partial_trailing_entry_raises(self):
        blob = _HEADER.pack(acl._ACL_XATTR_VERSION) + _ENTRY.pack(acl.ACL_USER_OBJ, 7, _NOID) + b"\x01\x02"
        with pytest.raises(ValueError):
            acl.parse_acl_blob(blob)


class TestReadAcl:
    def test_missing_xattr_returns_none(self, monkeypatch):
        _install_xattrs(monkeypatch)
        assert acl.read_acl("/some/path") is None

    def test_malformed_xattr_returns_none_not_raise(self, monkeypatch):
        _install_xattrs(monkeypatch, access=b"garbage")
        assert acl.read_acl("/some/path") is None

    def test_getxattr_oserror_is_treated_as_absent(self, monkeypatch):
        def _boom(path, name):
            raise OSError("no such attribute")

        monkeypatch.setattr(acl, "_getxattr", _boom)
        assert acl.read_acl("/some/path") is None


class TestEvaluateSandboxUidProvisioning:
    def test_no_acl_at_all_is_unsafe(self, monkeypatch):
        _install_xattrs(monkeypatch)
        safe, reason = acl.evaluate_sandbox_uid_provisioning("/repo", _SANDBOX_UID)
        assert safe is False
        assert "no POSIX ACL" in reason or "no POSIX ACL" in reason.replace("no POSIX ACL", "no POSIX ACL")

    def test_acl_present_but_no_entry_for_uid_is_unsafe(self, monkeypatch):
        blob = _blob([
            (acl.ACL_USER_OBJ, 0o7, None),
            (acl.ACL_USER, 0o7, 99999),
            (acl.ACL_GROUP_OBJ, 0o5, None),
            (acl.ACL_MASK, 0o7, None),
            (acl.ACL_OTHER, 0o5, None),
        ])
        _install_xattrs(monkeypatch, access=blob)
        safe, reason = acl.evaluate_sandbox_uid_provisioning("/repo", _SANDBOX_UID)
        assert safe is False
        assert str(_SANDBOX_UID) in reason

    def test_named_entry_without_write_is_unsafe(self, monkeypatch):
        blob = _blob([
            (acl.ACL_USER_OBJ, 0o7, None),
            (acl.ACL_USER, 0o5, _SANDBOX_UID),  # r-x, no write
            (acl.ACL_GROUP_OBJ, 0o5, None),
            (acl.ACL_MASK, 0o7, None),
            (acl.ACL_OTHER, 0o5, None),
        ])
        _install_xattrs(monkeypatch, access=blob)
        safe, _reason = acl.evaluate_sandbox_uid_provisioning("/repo", _SANDBOX_UID)
        assert safe is False

    def test_broad_group_obj_write_is_unsafe_even_with_named_entry(self, monkeypatch):
        """A named grant for the sandbox uid does not excuse the real
        (unmasked) owning-group entry ALSO being writable — that would still
        hand write to every group member, not just the sandbox uid."""
        blob = _blob([
            (acl.ACL_USER_OBJ, 0o7, None),
            (acl.ACL_USER, 0o7, _SANDBOX_UID),
            (acl.ACL_GROUP_OBJ, 0o7, None),  # real group perm includes write
            (acl.ACL_MASK, 0o7, None),
            (acl.ACL_OTHER, 0o5, None),
        ])
        _install_xattrs(monkeypatch, access=blob)
        safe, reason = acl.evaluate_sandbox_uid_provisioning("/repo", _SANDBOX_UID)
        assert safe is False
        assert "group" in reason.lower()

    def test_other_write_is_unsafe(self, monkeypatch):
        blob = _blob([
            (acl.ACL_USER_OBJ, 0o7, None),
            (acl.ACL_USER, 0o7, _SANDBOX_UID),
            (acl.ACL_GROUP_OBJ, 0o5, None),
            (acl.ACL_MASK, 0o7, None),
            (acl.ACL_OTHER, 0o7, None),  # world write
        ])
        _install_xattrs(monkeypatch, access=blob)
        safe, reason = acl.evaluate_sandbox_uid_provisioning("/repo", _SANDBOX_UID)
        assert safe is False
        assert "other" in reason.lower()

    def test_mask_restricting_the_named_grant_is_unsafe(self, monkeypatch):
        blob = _blob([
            (acl.ACL_USER_OBJ, 0o7, None),
            (acl.ACL_USER, 0o7, _SANDBOX_UID),
            (acl.ACL_GROUP_OBJ, 0o5, None),
            (acl.ACL_MASK, 0o5, None),  # mask clips the uid's write bit
            (acl.ACL_OTHER, 0o5, None),
        ])
        _install_xattrs(monkeypatch, access=blob)
        safe, reason = acl.evaluate_sandbox_uid_provisioning("/repo", _SANDBOX_UID)
        assert safe is False
        assert "mask" in reason.lower()

    def test_correctly_provisioned_named_acl_is_safe(self, monkeypatch):
        """The exact recommended shape: stat's group column would show the
        mask (7 -> looks like 'rwx group'), but the REAL group entry is r-x
        and only the named uid entry carries write."""
        blob = _blob([
            (acl.ACL_USER_OBJ, 0o7, None),
            (acl.ACL_USER, 0o7, _SANDBOX_UID),
            (acl.ACL_GROUP_OBJ, 0o5, None),
            (acl.ACL_MASK, 0o7, None),
            (acl.ACL_OTHER, 0o5, None),
        ])
        _install_xattrs(monkeypatch, access=blob)
        safe, reason = acl.evaluate_sandbox_uid_provisioning("/repo", _SANDBOX_UID)
        assert safe is True
        assert str(_SANDBOX_UID) in reason

    def test_a_different_uids_grant_does_not_authorize_this_uid(self, monkeypatch):
        blob = _blob([
            (acl.ACL_USER_OBJ, 0o7, None),
            (acl.ACL_USER, 0o7, _SANDBOX_UID + 1),
            (acl.ACL_GROUP_OBJ, 0o5, None),
            (acl.ACL_MASK, 0o7, None),
            (acl.ACL_OTHER, 0o5, None),
        ])
        _install_xattrs(monkeypatch, access=blob)
        safe, _reason = acl.evaluate_sandbox_uid_provisioning("/repo", _SANDBOX_UID)
        assert safe is False


class TestDefaultAclGrantsUid:
    def test_absent_default_acl_is_false(self, monkeypatch):
        _install_xattrs(monkeypatch)
        assert acl.default_acl_grants_uid("/repo", _SANDBOX_UID) is False

    def test_default_acl_with_write_grant_is_true(self, monkeypatch):
        blob = _blob([
            (acl.ACL_USER_OBJ, 0o7, None),
            (acl.ACL_USER, 0o7, _SANDBOX_UID),
            (acl.ACL_GROUP_OBJ, 0o5, None),
            (acl.ACL_MASK, 0o7, None),
            (acl.ACL_OTHER, 0o5, None),
        ])
        _install_xattrs(monkeypatch, default=blob)
        assert acl.default_acl_grants_uid("/repo", _SANDBOX_UID) is True

    def test_default_acl_without_write_grant_is_false(self, monkeypatch):
        blob = _blob([
            (acl.ACL_USER_OBJ, 0o7, None),
            (acl.ACL_USER, 0o5, _SANDBOX_UID),
            (acl.ACL_GROUP_OBJ, 0o5, None),
            (acl.ACL_MASK, 0o7, None),
            (acl.ACL_OTHER, 0o5, None),
        ])
        _install_xattrs(monkeypatch, default=blob)
        assert acl.default_acl_grants_uid("/repo", _SANDBOX_UID) is False


class TestRecommendedSetfaclCommands:
    def test_returns_access_and_default_acl_commands(self):
        commands = acl.recommended_setfacl_commands("/repo", _SANDBOX_UID)
        assert commands == [
            f"setfacl -m u:{_SANDBOX_UID}:rwx /repo",
            f"setfacl -d -m u:{_SANDBOX_UID}:rwx /repo",
        ]

    def test_never_executes_anything(self, monkeypatch):
        def _boom(*a, **k):
            raise AssertionError("recommended_setfacl_commands must not execute anything")

        monkeypatch.setattr(acl.os, "system", _boom, raising=False)
        acl.recommended_setfacl_commands("/repo", _SANDBOX_UID)
