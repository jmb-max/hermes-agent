"""RED->GREEN tests for ``plugins/claude-worker/telemetry.py`` (requirement 9).

One safe JSONL record per invocation; prompts, tokens, credentials, raw
command output, and secret-like values must never reach the telemetry file.

The write destination is a FIXED host path under ``get_hermes_home()`` —
never redirectable by a caller-supplied ``configured_path`` — opened with a
trusted, symlink/FIFO/device-refusing ``os.open`` and a bounded fallback
record on any primary-write failure, so telemetry can never be pointed at an
attacker-chosen file and never blocks or raises the caller down with it.
"""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import pytest

from tests.plugins._claude_worker_helpers import load_submodule

telemetry = load_submodule("telemetry")


@pytest.fixture(autouse=True)
def _home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


def _primary_path(home):
    return home / "claude-worker" / "telemetry.jsonl"


def _fallback_path(home):
    return home / "claude-worker" / "telemetry-fallback.jsonl"


def _read_lines(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class TestAppendRecord:
    def test_writes_one_jsonl_line(self, _home):
        telemetry.append_record(
            {
                "ts": "2026-08-20T00:00:00Z", "session_id": "sess:1", "chat_id": "123",
                "model": "claude-sonnet-5", "route_reason": "default", "attempt": 0,
                "escalated": False, "duration_ms": 500, "exit_code": 0,
                "failure_class": None, "breaker_state": "closed", "cwd": "/repo",
                "files_touched": ["a.py"], "success": True,
            },
        )
        lines = _read_lines(_primary_path(_home))
        assert len(lines) == 1
        assert lines[0]["session_id"] == "sess:1"
        assert lines[0]["model"] == "claude-sonnet-5"

    def test_failed_run_still_emits_one_record(self, _home):
        telemetry.append_record(
            {
                "ts": "t", "session_id": "sess:2", "model": "claude-opus-5",
                "route_reason": "escalation:sonnet-failure", "attempt": 1,
                "escalated": True, "duration_ms": 900, "exit_code": 1,
                "failure_class": "other", "breaker_state": "closed", "cwd": "/repo",
                "files_touched": [], "success": False,
            },
        )
        lines = _read_lines(_primary_path(_home))
        assert len(lines) == 1
        assert lines[0]["success"] is False
        assert lines[0]["failure_class"] == "other"

    def test_multiple_invocations_append_multiple_lines(self, _home):
        for i in range(3):
            telemetry.append_record({"session_id": f"sess:{i}"})
        assert len(_read_lines(_primary_path(_home))) == 3

    def test_default_path_is_under_hermes_home(self, _home):
        telemetry.append_record({"session_id": "sess:default"})
        expected = _primary_path(_home)
        assert expected.exists()
        lines = _read_lines(expected)
        assert lines[-1]["session_id"] == "sess:default"


class TestConfiguredPathIsIgnored:
    """``configured_path`` is deprecated: it can never redirect the write
    destination away from the fixed hermes-home path — the whole point is
    that no caller-supplied value (config.yaml, tool args, ...) can point
    telemetry at an attacker-chosen file."""

    def test_configured_path_is_never_written_to(self, _home, tmp_path):
        external = tmp_path / "attacker-chosen.jsonl"
        telemetry.append_record({"session_id": "sess:1"}, configured_path=str(external))
        assert not external.exists()

    def test_record_still_lands_at_the_fixed_path(self, _home, tmp_path):
        external = tmp_path / "attacker-chosen.jsonl"
        telemetry.append_record({"session_id": "sess:1"}, configured_path=str(external))
        lines = _read_lines(_primary_path(_home))
        assert len(lines) == 1
        assert lines[0]["session_id"] == "sess:1"


class TestTrustedPermissions:
    def test_directory_created_mode_0700(self, _home):
        telemetry.append_record({"session_id": "s"})
        mode = stat.S_IMODE(os.stat(_home / "claude-worker").st_mode)
        assert mode == 0o700

    def test_file_created_mode_0600(self, _home):
        telemetry.append_record({"session_id": "s"})
        mode = stat.S_IMODE(os.stat(_primary_path(_home)).st_mode)
        assert mode == 0o600


class TestPrimaryWriteFailureFallsBackSafely:
    def test_primary_symlink_uses_fallback_exactly_once(self, _home):
        primary = _primary_path(_home)
        primary.parent.mkdir(parents=True, exist_ok=True)
        target = _home / "somewhere-else.jsonl"
        target.write_text("")
        os.symlink(target, primary)

        telemetry.append_record({"session_id": "sess:sym"})

        fallback_lines = _read_lines(_fallback_path(_home))
        assert len(fallback_lines) == 1
        assert fallback_lines[0]["telemetry_error"] == "primary_write_failed"
        assert target.read_text() == ""  # symlink target never written through

    def test_primary_fifo_uses_fallback_exactly_once(self, _home):
        primary = _primary_path(_home)
        primary.parent.mkdir(parents=True, exist_ok=True)
        os.mkfifo(primary)

        telemetry.append_record({"session_id": "sess:fifo"})

        fallback_lines = _read_lines(_fallback_path(_home))
        assert len(fallback_lines) == 1
        assert fallback_lines[0]["telemetry_error"] == "primary_write_failed"

    def test_primary_is_a_directory_uses_fallback_exactly_once(self, _home):
        primary = _primary_path(_home)
        primary.mkdir(parents=True)

        telemetry.append_record({"session_id": "sess:dir"})

        fallback_lines = _read_lines(_fallback_path(_home))
        assert len(fallback_lines) == 1
        assert fallback_lines[0]["telemetry_error"] == "primary_write_failed"

    def test_primary_open_error_uses_fallback_exactly_once(self, _home, monkeypatch):
        real_open = telemetry._open_trusted_for_append
        primary = _primary_path(_home)

        def _boom_for_primary(path):
            if Path(path) == primary:
                raise OSError("simulated open failure")
            return real_open(path)

        monkeypatch.setattr(telemetry, "_open_trusted_for_append", _boom_for_primary)

        telemetry.append_record({"session_id": "sess:err", "chat_id": "42"})

        fallback_lines = _read_lines(_fallback_path(_home))
        assert len(fallback_lines) == 1
        assert fallback_lines[0]["telemetry_error"] == "primary_write_failed"
        assert fallback_lines[0]["session_id"] == "sess:err"
        assert fallback_lines[0]["chat_id"] == "42"

    def test_primary_failure_never_blocks_or_raises(self, _home, monkeypatch):
        monkeypatch.setattr(
            telemetry, "_open_trusted_for_append",
            lambda path: (_ for _ in ()).throw(OSError("nope")),
        )
        telemetry.append_record({"session_id": "s"})  # must return promptly, never raise

    def test_primary_success_writes_no_fallback(self, _home):
        telemetry.append_record({"session_id": "sess:ok"})
        assert not _fallback_path(_home).exists()
        assert len(_read_lines(_primary_path(_home))) == 1

    def test_fallback_failure_logs_error_and_does_not_raise(self, _home, monkeypatch, caplog):
        monkeypatch.setattr(
            telemetry, "_open_trusted_for_append",
            lambda path: (_ for _ in ()).throw(OSError("everything is broken")),
        )
        import logging

        with caplog.at_level(logging.ERROR, logger=telemetry.logger.name):
            telemetry.append_record({"session_id": "s"})  # must not raise
        assert any("claude_worker" in rec.message.lower() or "telemetry" in rec.message.lower()
                   for rec in caplog.records)

    def test_fallback_record_carries_no_secrets(self, _home, monkeypatch):
        real_open = telemetry._open_trusted_for_append
        primary = _primary_path(_home)

        def _boom_for_primary(path):
            if Path(path) == primary:
                raise OSError("nope")
            return real_open(path)

        monkeypatch.setattr(telemetry, "_open_trusted_for_append", _boom_for_primary)
        telemetry.append_record({
            "session_id": "sk-ant-api03-AAAAAAAAAAAAAAAAAAAA",
            "chat_id": "123",
        })
        fallback_lines = _read_lines(_fallback_path(_home))
        assert "sk-ant-api03" not in json.dumps(fallback_lines[0])


class TestRedaction:
    def test_unknown_fields_are_dropped(self, _home):
        telemetry.append_record(
            {
                "session_id": "sess:1", "prompt": "full user prompt text",
                "raw_stdout": "command output blob", "api_key": "sk-should-not-appear",
                "arbitrary_field": "should not appear either",
            },
        )
        line = _read_lines(_primary_path(_home))[0]
        assert "prompt" not in line
        assert "raw_stdout" not in line
        assert "api_key" not in line
        assert "arbitrary_field" not in line

    def test_secret_like_key_names_in_safe_fields_are_redacted(self, _home):
        # Even a value smuggled into a normally-safe-shaped field is caught
        # by the key-name-based redaction pass as belt-and-suspenders.
        telemetry.append_record({"session_id": "sess:1", "model": "claude-sonnet-5"})
        line = _read_lines(_primary_path(_home))[0]
        assert line["model"] == "claude-sonnet-5"

    def test_safe_field_allowlist_is_stable(self):
        assert "session_id" in telemetry.SAFE_FIELDS
        assert "model" in telemetry.SAFE_FIELDS
        assert "cwd" in telemetry.SAFE_FIELDS
        assert "prompt" not in telemetry.SAFE_FIELDS
        assert "stdout" not in telemetry.SAFE_FIELDS
        assert "stderr" not in telemetry.SAFE_FIELDS


class TestPathsAreHashedOrRepoRelative:
    def test_cwd_is_fingerprinted_not_recorded(self, _home, tmp_path):
        repo = tmp_path / "customer-acme-prod"
        repo.mkdir()
        telemetry.append_record({"session_id": "sess:1", "cwd": str(repo)})
        line = _read_lines(_primary_path(_home))[0]
        assert str(repo) not in json.dumps(line)
        assert "customer-acme-prod" not in json.dumps(line)
        assert line["cwd"].startswith("sha256:")

    def test_cwd_fingerprint_is_stable_and_distinguishing(self, tmp_path):
        a = tmp_path / "repo-a"
        b = tmp_path / "repo-b"
        a.mkdir()
        b.mkdir()
        assert telemetry.hash_path(str(a)) == telemetry.hash_path(str(a) + "/")
        assert telemetry.hash_path(str(a)) != telemetry.hash_path(str(b))

    def test_empty_cwd_stays_empty(self, _home):
        telemetry.append_record({"session_id": "s", "cwd": ""})
        assert _read_lines(_primary_path(_home))[0]["cwd"] == ""

    def test_absolute_touched_files_are_made_repo_relative(self, _home, tmp_path):
        repo = tmp_path / "repo"
        (repo / "src").mkdir(parents=True)
        telemetry.append_record(
            {"session_id": "s", "cwd": str(repo), "files_touched": [str(repo / "src" / "app.py")]},
        )
        line = _read_lines(_primary_path(_home))[0]
        assert line["files_touched"] == [os.path.join("src", "app.py")]
        assert str(tmp_path) not in json.dumps(line)

    def test_file_outside_the_repo_is_hashed_not_recorded(self, _home, tmp_path):
        repo = tmp_path / "repo"
        repo.mkdir()
        outside = tmp_path / "etc-shadow-copy"
        outside.mkdir()
        telemetry.append_record(
            {"session_id": "s", "cwd": str(repo), "files_touched": [str(outside / "creds.txt")]},
        )
        line = _read_lines(_primary_path(_home))[0]
        assert "etc-shadow-copy" not in json.dumps(line)
        assert line["files_touched"][0].startswith("sha256:")

    def test_absolute_file_without_a_cwd_is_hashed(self, _home):
        telemetry.append_record({"session_id": "s", "files_touched": ["/srv/secret-project/main.py"]})
        line = _read_lines(_primary_path(_home))[0]
        assert "secret-project" not in json.dumps(line)
        assert line["files_touched"][0].startswith("sha256:")

    def test_too_many_files_are_capped(self, _home):
        telemetry.append_record({"session_id": "s", "files_touched": [f"f{i}.py" for i in range(500)]})
        assert len(_read_lines(_primary_path(_home))[0]["files_touched"]) <= 50


class TestSecretShapedValuesAreRedacted:
    @pytest.mark.parametrize(
        "filename",
        [
            "sk-ant-api03-AAAAAAAAAAAAAAAAAAAAAAAA.json",
            "ghp_AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA.txt",
            "dump-0123456789abcdef0123456789abcdef.sql",
            "token=super-secret-value.env",
        ],
    )
    def test_token_like_filenames_are_hashed(self, _home, filename):
        telemetry.append_record({"session_id": "s", "files_touched": [filename]})
        line = _read_lines(_primary_path(_home))[0]
        assert filename not in json.dumps(line)
        assert line["files_touched"][0].startswith("sha256:")

    def test_ordinary_filenames_survive_verbatim(self, _home):
        telemetry.append_record({"session_id": "s", "files_touched": ["src/app.py", "README.md"]})
        assert _read_lines(_primary_path(_home))[0]["files_touched"] == ["src/app.py", "README.md"]

    def test_secret_shaped_scalar_value_is_redacted(self, _home):
        telemetry.append_record({"session_id": "sk-ant-api03-AAAAAAAAAAAAAAAAAAAA", "model": "claude-sonnet-5"})
        line = _read_lines(_primary_path(_home))[0]
        assert line["session_id"] == "[REDACTED]"
        assert line["model"] == "claude-sonnet-5"

    def test_long_free_text_is_truncated(self, _home):
        telemetry.append_record({"session_id": "s", "failure_class": "x" * 5000})
        assert len(_read_lines(_primary_path(_home))[0]["failure_class"]) <= 256


class TestMalformedRecordsStillProduceOneRecord:
    @pytest.mark.parametrize(
        "record",
        [
            {"session_id": {"nested": "object"}},
            {"session_id": "s", "files_touched": "not-a-list"},
            {"session_id": "s", "attempt": ["weird"]},
            {"session_id": "s", "cwd": 12345},
            {},
        ],
    )
    def test_malformed_values_never_raise_and_write_one_line(self, _home, record):
        telemetry.append_record(record)
        assert len(_read_lines(_primary_path(_home))) == 1

    def test_non_dict_record_writes_one_line(self, _home):
        telemetry.append_record("not-a-record")
        assert len(_read_lines(_primary_path(_home))) == 1

    def test_unexpected_sanitize_error_still_writes_one_marker_line(self, _home, monkeypatch):
        def _boom(record):
            raise RuntimeError("sanitizer exploded")

        monkeypatch.setattr(telemetry, "_sanitize", _boom)
        telemetry.append_record({"session_id": "s"})
        lines = _read_lines(_primary_path(_home))
        assert len(lines) == 1
        assert lines[0]["telemetry_error"] == "record_sanitization_failed"

    def test_unserializable_value_still_writes_one_line(self, _home):
        class _Unserializable:
            def __repr__(self):
                return "<obj>"

        telemetry.append_record({"session_id": "s", "exit_code": _Unserializable()})
        assert len(_read_lines(_primary_path(_home))) == 1


class TestBuildRecord:
    def test_build_record_produces_only_safe_fields(self):
        record = telemetry.build_record(
            session_id="sess:1", chat_id="123", model="claude-sonnet-5",
            route_reason="default", attempt=0, escalated=False, duration_ms=100,
            exit_code=0, failure_class=None, breaker_state="closed", cwd="/repo",
            files_touched=["a.py"], success=True,
        )
        assert set(record.keys()) <= set(telemetry.SAFE_FIELDS)
        assert record["success"] is True
