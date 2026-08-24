"""Tests for ``plugins/claude-worker/oauth.py`` — the pre-spawn OAuth
credential freshness preflight.

The failure this module closes: the worker spawned with an already-expired
access token, the container came back with a 401, ``breaker.classify_failure``
called it ``auth``, and the breaker slammed shut for a full hour over
something a refresh would have fixed in a second. So the credential's
freshness is read BEFORE anything is committed to, and the reading itself has
to be trustworthy on two independent axes:

* **Determinism.** Every case below passes an explicit ``now``, so a token
  that is "fresh" or "expired" is decided by arithmetic, not by how long the
  test took to run. The margin boundary is pinned to the exact second on both
  sides.
* **Secrecy.** Nothing this module returns, formats, or logs may contain a
  token — not in the freshness summary, not in a preflight ``reason``, not in
  a probe's own error text, and not in a traceback. Several tests below
  assert on ``json.dumps(result)`` and on ``caplog.text`` rather than on one
  hand-picked field, so a token leaking through a field nobody thought to
  check still fails them.

The orchestration side — that ``run_worker`` actually calls this, HOLDs on
failure, spawns nothing, and mutates no breaker state — lives in
``TestOAuthPreflightIntegration`` in ``test_claude_worker_runner.py``.
"""

from __future__ import annotations

import json
import logging
import os

import pytest

from tests.plugins._claude_worker_helpers import load_submodule, write_credentials

oauth = load_submodule("oauth")
policy = load_submodule("policy")

#: A fixed instant every case computes expiries against. Nothing here reads
#: the wall clock.
NOW = 1_700_000_000.0

ACCESS = "sk-ant-oat01-AAAABBBBCCCCDDDDEEEEFFFF-accesstoken"
REFRESH = "sk-ant-ort01-1111222233334444555566667777-refreshtoken"


@pytest.fixture(autouse=True)
def _no_ambient_probe(monkeypatch):
    """No refresh probe is installed in production, and none may leak in
    from another test's module-level assignment."""
    monkeypatch.setattr(oauth, "REFRESH_PROBE", None)


@pytest.fixture
def creds(tmp_path):
    return tmp_path / "creds" / ".credentials.json"


def _write(path, **kwargs):
    kwargs.setdefault("access_token", ACCESS)
    kwargs.setdefault("refresh_token", REFRESH)
    return write_credentials(path, **kwargs)


# ---------------------------------------------------------------------------
# read_freshness — expiry normalization and the margin
# ---------------------------------------------------------------------------


class TestFreshness:
    def test_millisecond_expiry_is_fresh(self, creds):
        """Claude Code itself writes ``expiresAt`` in MILLISECONDS."""
        _write(creds, expires_at=int((NOW + 3600) * 1000))

        result = oauth.read_freshness(str(creds), now=NOW)

        assert result["state"] == oauth.STATE_FRESH
        assert result["expired"] is False
        assert result["present"] is True
        assert result["has_access_token"] is True
        assert result["has_refresh_token"] is True
        assert result["expires_at_epoch"] == NOW + 3600
        assert result["seconds_remaining"] == 3600
        assert result["reason"] == ""

    def test_second_expiry_is_fresh_and_agrees_with_the_millisecond_form(self, creds):
        """Other tooling writes SECONDS. Both must normalize to the same
        instant — a seconds value silently read as milliseconds would make a
        live token look ~50 years expired, which is precisely the surprise
        401 this module exists to prevent."""
        _write(creds, expires_at=int(NOW + 3600))
        seconds = oauth.read_freshness(str(creds), now=NOW)

        _write(creds, expires_at=int((NOW + 3600) * 1000))
        millis = oauth.read_freshness(str(creds), now=NOW)

        assert seconds["state"] == oauth.STATE_FRESH
        assert seconds["expires_at_epoch"] == millis["expires_at_epoch"] == NOW + 3600
        assert seconds["seconds_remaining"] == millis["seconds_remaining"] == 3600

    def test_one_second_past_the_margin_is_fresh(self, creds):
        _write(creds, expires_at=(NOW + oauth.FRESHNESS_MARGIN_SECONDS + 1) * 1000)

        result = oauth.read_freshness(str(creds), now=NOW)

        assert result["state"] == oauth.STATE_FRESH
        assert result["expired"] is False
        assert result["seconds_remaining"] == oauth.FRESHNESS_MARGIN_SECONDS + 1

    def test_exactly_at_the_margin_is_already_expired(self, creds):
        """A token with exactly the margin left expires mid-run. Treating the
        boundary as fresh is the failure mode; the comparison is strict."""
        _write(creds, expires_at=(NOW + oauth.FRESHNESS_MARGIN_SECONDS) * 1000)

        result = oauth.read_freshness(str(creds), now=NOW)

        assert result["state"] == oauth.STATE_REFRESHABLE_EXPIRED
        assert result["expired"] is True
        assert result["seconds_remaining"] == oauth.FRESHNESS_MARGIN_SECONDS

    def test_expired_with_a_refresh_token_is_refreshable(self, creds):
        _write(creds, expires_at=(NOW - 10) * 1000)

        result = oauth.read_freshness(str(creds), now=NOW)

        assert result["state"] == oauth.STATE_REFRESHABLE_EXPIRED
        assert result["expired"] is True
        assert result["has_refresh_token"] is True
        assert result["seconds_remaining"] == -10

    def test_expired_without_a_refresh_token_is_terminal(self, creds):
        _write(creds, refresh_token=None, expires_at=(NOW - 10) * 1000)

        result = oauth.read_freshness(str(creds), now=NOW)

        assert result["state"] == oauth.STATE_EXPIRED
        assert result["expired"] is True
        assert result["has_refresh_token"] is False

    def test_non_secret_labels_are_reported(self, creds):
        _write(creds, expires_at=(NOW + 3600) * 1000)

        result = oauth.read_freshness(str(creds), now=NOW)

        assert result["scopes"] == ["user:inference", "user:profile"]
        assert result["subscription_type"] == "max"

    @pytest.mark.parametrize(
        "expiry",
        [None, True, False, 0, -1, "1700000000", [], {}],
        ids=["missing", "true", "false", "zero", "negative", "string", "list", "dict"],
    )
    def test_an_unusable_expiry_is_never_treated_as_eternal(self, creds, expiry):
        """A credential with no usable expiry is exactly the case that
        produced surprise 401s. It is never assumed to be good forever."""
        _write(creds, expires_at=expiry)

        result = oauth.read_freshness(str(creds), now=NOW)

        assert result["expired"] is True
        assert result["state"] == oauth.STATE_REFRESHABLE_EXPIRED

    def test_an_unusable_expiry_without_a_refresh_token_is_invalid(self, creds):
        _write(creds, refresh_token=None, expires_at=None)

        result = oauth.read_freshness(str(creds), now=NOW)

        assert result["state"] == oauth.STATE_INVALID
        assert result["expired"] is True

    def test_a_flat_credential_without_an_oauth_section_still_reads(self, tmp_path):
        """Not every writer nests under ``claudeAiOauth``; a flat object is
        read from the top level rather than reported as unusable."""
        path = tmp_path / "flat.json"
        path.write_text(
            json.dumps({
                "accessToken": ACCESS,
                "refreshToken": REFRESH,
                "expiresAt": int((NOW + 3600) * 1000),
            }),
            encoding="utf-8",
        )

        assert oauth.read_freshness(str(path), now=NOW)["state"] == oauth.STATE_FRESH

    def test_snake_case_key_spellings_are_accepted(self, tmp_path):
        path = tmp_path / "snake.json"
        path.write_text(
            json.dumps({"oauth": {
                "access_token": ACCESS,
                "refresh_token": REFRESH,
                "expires_at": int(NOW + 3600),
            }}),
            encoding="utf-8",
        )

        result = oauth.read_freshness(str(path), now=NOW)
        assert result["state"] == oauth.STATE_FRESH
        assert result["has_refresh_token"] is True


# ---------------------------------------------------------------------------
# read_freshness — every unusable file shape fails closed
# ---------------------------------------------------------------------------


class TestUnusableCredentialFiles:
    def test_missing_file_is_missing_not_fresh(self, tmp_path):
        result = oauth.read_freshness(str(tmp_path / "nope.json"), now=NOW)

        assert result["state"] == oauth.STATE_MISSING
        assert result["present"] is False
        assert result["expired"] is True
        assert result["has_access_token"] is False

    def test_malformed_json_is_malformed(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("{not json{{{", encoding="utf-8")

        result = oauth.read_freshness(str(path), now=NOW)

        assert result["state"] == oauth.STATE_MALFORMED
        assert result["present"] is True
        assert result["expired"] is True

    def test_undecodable_bytes_are_malformed(self, tmp_path):
        path = tmp_path / "bytes.json"
        path.write_bytes(b"\xff\xfe\x00garbage")

        assert oauth.read_freshness(str(path), now=NOW)["state"] == oauth.STATE_MALFORMED

    @pytest.mark.parametrize(
        "body", ["[]", '"a string"', "42", "null", "true"],
        ids=["list", "string", "number", "null", "bool"],
    )
    def test_a_non_object_document_is_malformed(self, tmp_path, body):
        path = tmp_path / "nonobject.json"
        path.write_text(body, encoding="utf-8")

        result = oauth.read_freshness(str(path), now=NOW)

        assert result["state"] == oauth.STATE_MALFORMED
        assert result["expired"] is True

    def test_an_oversized_file_is_refused_without_parsing(self, tmp_path):
        """A bound this generous is a DoS backstop, not a functional limit —
        but it must refuse rather than parse an unbounded blob."""
        path = tmp_path / "huge.json"
        filler = "a" * (policy.MAX_CREDENTIAL_BYTES + 16)
        path.write_text(json.dumps({"claudeAiOauth": {"accessToken": filler}}), encoding="utf-8")
        assert path.stat().st_size > policy.MAX_CREDENTIAL_BYTES

        result = oauth.read_freshness(str(path), now=NOW)

        assert result["state"] == oauth.STATE_MALFORMED
        assert result["present"] is True
        assert result["has_access_token"] is False

    def test_a_credential_with_no_access_token_is_invalid(self, tmp_path):
        path = tmp_path / "noaccess.json"
        path.write_text(
            json.dumps({"claudeAiOauth": {
                "refreshToken": REFRESH, "expiresAt": int((NOW + 3600) * 1000),
            }}),
            encoding="utf-8",
        )

        result = oauth.read_freshness(str(path), now=NOW)

        assert result["state"] == oauth.STATE_INVALID
        assert result["has_access_token"] is False
        assert result["expired"] is True

    @pytest.mark.parametrize("blank", ["", "   ", "\n"], ids=["empty", "spaces", "newline"])
    def test_a_blank_access_token_is_not_an_access_token(self, tmp_path, blank):
        path = tmp_path / "blank.json"
        path.write_text(
            json.dumps({"claudeAiOauth": {
                "accessToken": blank, "expiresAt": int((NOW + 3600) * 1000),
            }}),
            encoding="utf-8",
        )

        assert oauth.read_freshness(str(path), now=NOW)["state"] == oauth.STATE_INVALID

    def test_a_directory_at_the_credentials_path_is_missing_not_a_crash(self, tmp_path):
        directory = tmp_path / "credentials.json"
        directory.mkdir()

        assert oauth.read_freshness(str(directory), now=NOW)["state"] == oauth.STATE_MISSING


@pytest.mark.skipif(
    not hasattr(os, "O_NOFOLLOW"), reason="platform has no O_NOFOLLOW",
)
class TestSymlinkIsNeverFollowed:
    """The credentials file is opened with ``O_NOFOLLOW``: a symlink planted
    at the fixed host path could point anywhere, so it is refused outright
    rather than followed to whatever it names."""

    def test_a_symlink_to_a_fresh_credential_reads_as_missing(self, tmp_path):
        real = tmp_path / "real.json"
        _write(real, expires_at=(NOW + 3600) * 1000)
        link = tmp_path / "link.json"
        link.symlink_to(real)

        # The target itself is unambiguously fresh...
        assert oauth.read_freshness(str(real), now=NOW)["state"] == oauth.STATE_FRESH
        # ...and the link to it is still refused, so it is the LINK that is
        # rejected, not the content.
        via_link = oauth.read_freshness(str(link), now=NOW)
        assert via_link["state"] == oauth.STATE_MISSING
        assert via_link["present"] is False

    def test_a_symlinked_credential_holds_the_preflight(self, tmp_path):
        real = tmp_path / "real.json"
        _write(real, expires_at=(NOW + 3600) * 1000)
        link = tmp_path / "link.json"
        link.symlink_to(real)

        result = oauth.preflight(str(link), now=NOW)

        assert result["ok"] is False
        assert result["state"] == oauth.STATE_MISSING
        assert result["classification"] == "auth"

    def test_a_dangling_symlink_reads_as_missing(self, tmp_path):
        link = tmp_path / "dangling.json"
        link.symlink_to(tmp_path / "does-not-exist.json")

        assert oauth.read_freshness(str(link), now=NOW)["state"] == oauth.STATE_MISSING


# ---------------------------------------------------------------------------
# preflight — states, and the at-most-one probe
# ---------------------------------------------------------------------------


class TestPreflightWithoutAProbe:
    def test_a_fresh_credential_passes(self, creds):
        _write(creds, expires_at=(NOW + 3600) * 1000)

        result = oauth.preflight(str(creds), now=NOW)

        assert result["ok"] is True
        assert result["state"] == oauth.STATE_FRESH
        assert result["classification"] is None
        assert result["refresh_attempted"] is False
        assert result["reason"] == ""

    @pytest.mark.parametrize(
        "kwargs, expected_state",
        [
            ({"refresh_token": None, "expires_at": (NOW - 10) * 1000}, "expired"),
            ({"expires_at": (NOW - 10) * 1000}, "refreshable_expired"),
        ],
        ids=["expired", "refreshable-expired"],
    )
    def test_an_expired_credential_holds_with_an_auth_classification(
        self, creds, kwargs, expected_state,
    ):
        _write(creds, **kwargs)

        result = oauth.preflight(str(creds), now=NOW)

        assert result["ok"] is False
        assert result["state"] == expected_state
        assert result["classification"] == "auth"

    def test_refreshable_expired_holds_when_no_probe_is_installed(self, creds):
        """Production installs no probe: writing a refreshed token back to the
        root-owned credentials path would mean owning exactly the privileged
        credential-writing logic this plugin is careful never to own. HOLD is
        the safe direction."""
        _write(creds, expires_at=(NOW - 10) * 1000)
        assert oauth.REFRESH_PROBE is None

        result = oauth.preflight(str(creds), now=NOW)

        assert result["ok"] is False
        assert result["refresh_attempted"] is False
        assert "no isolated refresh probe is configured" in result["reason"]

    @pytest.mark.parametrize(
        "setup",
        [
            lambda path: None,  # missing
            lambda path: path.write_text("{{{", encoding="utf-8"),
            lambda path: path.write_text("[]", encoding="utf-8"),
        ],
        ids=["missing", "malformed", "non-object"],
    )
    def test_an_unusable_file_holds_and_never_probes(self, tmp_path, monkeypatch, setup):
        path = tmp_path / "creds.json"
        setup(path)
        probe_calls = []
        monkeypatch.setattr(
            oauth, "REFRESH_PROBE", lambda: probe_calls.append(True) or {"ok": True},
        )

        result = oauth.preflight(str(path), now=NOW)

        assert result["ok"] is False
        assert result["classification"] == "auth"
        # There is nothing to refresh: only a refreshable-expired credential
        # is ever worth a probe.
        assert probe_calls == []

    def test_a_fresh_credential_never_probes(self, creds, monkeypatch):
        _write(creds, expires_at=(NOW + 3600) * 1000)
        probe_calls = []
        monkeypatch.setattr(
            oauth, "REFRESH_PROBE", lambda: probe_calls.append(True) or {"ok": True},
        )

        assert oauth.preflight(str(creds), now=NOW)["ok"] is True
        assert probe_calls == []


class TestExactlyOneRefreshProbe:
    """At most one probe, structurally, on one code path — no retry loop
    lives in this module."""

    def _expired(self, creds):
        return _write(creds, expires_at=(NOW - 10) * 1000)

    def test_a_probe_that_restores_the_credential_passes(self, creds):
        self._expired(creds)
        probe_calls = []

        def _probe():
            probe_calls.append(True)
            _write(creds, expires_at=(NOW + 3600) * 1000)
            return {"ok": True}

        result = oauth.preflight(str(creds), now=NOW, refresh_probe=_probe)

        assert len(probe_calls) == 1
        assert result["ok"] is True
        assert result["state"] == oauth.STATE_FRESH
        assert result["refresh_attempted"] is True
        assert result["reason"] == ""

    def test_the_credential_is_re_read_from_disk_not_assumed(self, creds):
        """A probe claiming success proves nothing on its own — the file is
        re-read, and a credential that is still expired still HOLDs."""
        self._expired(creds)
        probe_calls = []

        result = oauth.preflight(
            str(creds), now=NOW,
            refresh_probe=lambda: probe_calls.append(True) or {"ok": True},
        )

        assert len(probe_calls) == 1
        assert result["ok"] is False
        assert result["state"] == oauth.STATE_REFRESHABLE_EXPIRED
        assert result["refresh_attempted"] is True
        assert result["classification"] == "auth"
        assert "did not restore a fresh credential" in result["reason"]

    def test_a_failing_probe_is_called_exactly_once(self, creds):
        self._expired(creds)
        probe_calls = []

        result = oauth.preflight(
            str(creds), now=NOW,
            refresh_probe=lambda: probe_calls.append(True) or {
                "ok": False, "reason": "refresh endpoint said no",
            },
        )

        assert len(probe_calls) == 1
        assert result["ok"] is False
        assert result["refresh_attempted"] is True
        assert "refresh endpoint said no" in result["reason"]

    def test_a_raising_probe_is_called_exactly_once_and_never_propagates(self, creds):
        self._expired(creds)
        probe_calls = []

        def _probe():
            probe_calls.append(True)
            raise RuntimeError("probe exploded")

        result = oauth.preflight(str(creds), now=NOW, refresh_probe=_probe)

        assert len(probe_calls) == 1
        assert result["ok"] is False
        assert result["refresh_attempted"] is True
        assert result["classification"] == "auth"

    @pytest.mark.parametrize(
        "outcome", [None, "ok", 1, [], object()],
        ids=["none", "string", "int", "list", "object"],
    )
    def test_a_probe_returning_a_non_mapping_is_treated_as_no_success(
        self, creds, outcome,
    ):
        self._expired(creds)
        probe_calls = []

        result = oauth.preflight(
            str(creds), now=NOW,
            refresh_probe=lambda: probe_calls.append(True) or outcome,
        )

        assert len(probe_calls) == 1
        assert result["ok"] is False
        assert result["refresh_attempted"] is True

    def test_the_argument_probe_takes_precedence_over_the_installed_one(self, creds, monkeypatch):
        self._expired(creds)
        installed_calls, argument_calls = [], []
        monkeypatch.setattr(
            oauth, "REFRESH_PROBE", lambda: installed_calls.append(True) or {"ok": True},
        )

        oauth.preflight(
            str(creds), now=NOW,
            refresh_probe=lambda: argument_calls.append(True) or {"ok": True},
        )

        assert len(argument_calls) == 1
        assert installed_calls == []

    def test_set_refresh_probe_installs_and_clears(self, creds):
        self._expired(creds)
        probe_calls = []
        try:
            oauth.set_refresh_probe(lambda: probe_calls.append(True) or {"ok": True})
            assert oauth.REFRESH_PROBE is not None
            oauth.preflight(str(creds), now=NOW)
            assert len(probe_calls) == 1
        finally:
            oauth.set_refresh_probe(None)
        assert oauth.REFRESH_PROBE is None

        oauth.preflight(str(creds), now=NOW)
        assert len(probe_calls) == 1  # still one: no probe is installed now


# ---------------------------------------------------------------------------
# Secrets never leave this module
# ---------------------------------------------------------------------------


class TestRedact:
    def test_a_secret_is_replaced_everywhere_it_appears(self):
        assert oauth.redact(f"{ACCESS} and again {ACCESS}", (ACCESS,)) == (
            "[redacted] and again [redacted]"
        )

    def test_multiple_secrets_are_all_replaced(self):
        out = oauth.redact(f"a={ACCESS} r={REFRESH}", (ACCESS, REFRESH))
        assert ACCESS not in out
        assert REFRESH not in out
        assert out.count("[redacted]") == 2

    @pytest.mark.parametrize("short", ["", "a", "ab", "abc"])
    def test_a_too_short_secret_is_not_used_as_a_needle(self, short):
        """Redacting on a 1-3 character needle would shred ordinary text into
        noise without protecting anything real."""
        assert oauth.redact("abc abcdef", (short,)) == "abc abcdef"

    @pytest.mark.parametrize(
        "value, expected",
        [(None, ""), (123, "123"), (["a"], "['a']"), (b"x", "b'x'")],
        ids=["none", "int", "list", "bytes"],
    )
    def test_non_string_input_is_coerced_not_crashed(self, value, expected):
        assert oauth.redact(value, ()) == expected

    @pytest.mark.parametrize("secrets", [None, (), [None, 1, object()]])
    def test_a_missing_or_junk_secret_list_is_harmless(self, secrets):
        assert oauth.redact("unchanged text", secrets) == "unchanged text"


class TestNoSecretEverLeaves:
    def test_a_freshness_summary_contains_no_token(self, creds):
        _write(creds, expires_at=(NOW + 3600) * 1000)

        serialized = json.dumps(oauth.read_freshness(str(creds), now=NOW))

        assert ACCESS not in serialized
        assert REFRESH not in serialized
        assert "accessToken" not in serialized
        assert "refreshToken" not in serialized

    def test_a_preflight_result_contains_no_token(self, creds):
        _write(creds, expires_at=(NOW + 3600) * 1000)

        serialized = json.dumps(oauth.preflight(str(creds), now=NOW))

        assert ACCESS not in serialized
        assert REFRESH not in serialized

    def test_a_probe_quoting_the_token_back_is_redacted_out_of_the_reason(self, creds):
        """The exact leak this closes: an operator-installed probe talking to
        an auth endpoint quotes the credential it just sent in its own error
        string, and that string is otherwise copied verbatim into the
        result's ``reason``."""
        _write(creds, expires_at=(NOW - 10) * 1000)

        result = oauth.preflight(
            str(creds), now=NOW,
            refresh_probe=lambda: {
                "ok": False,
                "reason": f"POST /oauth/token with refresh_token={REFRESH} rejected",
            },
        )

        assert REFRESH not in result["reason"]
        assert "[redacted]" in result["reason"]
        assert REFRESH not in json.dumps(result)
        assert ACCESS not in json.dumps(result)

    def test_a_probe_raising_with_the_token_leaks_it_nowhere(self, creds, caplog):
        """Not into the result, and — because a traceback embeds the raw
        exception text — not into the log either."""
        _write(creds, expires_at=(NOW - 10) * 1000)

        def _probe():
            raise RuntimeError(f"refresh failed for token {REFRESH}")

        with caplog.at_level(logging.DEBUG):
            result = oauth.preflight(str(creds), now=NOW, refresh_probe=_probe)

        assert result["ok"] is False
        assert REFRESH not in json.dumps(result)
        assert REFRESH not in caplog.text
        assert ACCESS not in caplog.text
        # Still actionable: the operator learns what failed and how.
        assert "RuntimeError" in caplog.text
        assert "[redacted]" in caplog.text

    def test_nothing_logs_the_credential_on_the_ordinary_paths(self, creds, caplog):
        _write(creds, expires_at=(NOW + 3600) * 1000)
        with caplog.at_level(logging.DEBUG):
            oauth.preflight(str(creds), now=NOW)
            _write(creds, expires_at=(NOW - 10) * 1000)
            oauth.preflight(str(creds), now=NOW)

        assert ACCESS not in caplog.text
        assert REFRESH not in caplog.text

    def test_the_result_shape_is_exactly_the_documented_surface(self, creds):
        """A closed key set, asserted directly: a future field carrying
        anything from the parsed credential has to break this test first."""
        _write(creds, expires_at=(NOW + 3600) * 1000)

        result = oauth.preflight(str(creds), now=NOW)
        assert set(result) == {
            "ok", "state", "classification", "refresh_attempted", "reason", "freshness",
        }
        assert set(result["freshness"]) == {
            "state", "present", "has_access_token", "has_refresh_token",
            "expires_at_epoch", "seconds_remaining", "expired", "scopes",
            "subscription_type", "reason",
        }


# ---------------------------------------------------------------------------
# Defaults and invariants
# ---------------------------------------------------------------------------


class TestDefaultsAndInvariants:
    def test_the_default_path_is_the_fixed_host_credentials_path(self, monkeypatch, tmp_path):
        """Neither function derives its path from the environment: with no
        argument, both read exactly ``policy.HOST_CREDENTIALS_PATH``."""
        creds = tmp_path / "fixed.json"
        _write(creds, expires_at=(NOW + 3600) * 1000)
        monkeypatch.setattr(policy, "HOST_CREDENTIALS_PATH", str(creds))

        assert oauth.read_freshness(now=NOW)["state"] == oauth.STATE_FRESH
        assert oauth.preflight(now=NOW)["ok"] is True

    def test_read_freshness_never_raises_on_any_input(self, tmp_path, monkeypatch):
        """"Never raises" covers every rejection ``os.open`` can produce, not
        just ``OSError`` — an empty path, a directory, an absent file, and a
        path carrying an embedded NUL (a ``ValueError``) all fail closed."""
        monkeypatch.setattr(policy, "HOST_CREDENTIALS_PATH", str(tmp_path / "absent.json"))
        for candidate in ("", str(tmp_path), str(tmp_path / "gone"), "with\x00nul"):
            result = oauth.read_freshness(candidate, now=NOW)
            assert result["expired"] is True
            assert result["state"] in {oauth.STATE_MISSING, oauth.STATE_MALFORMED}

    def test_the_module_never_writes_the_credentials_file(self):
        """The one structural guarantee behind "refresh is deliberately NOT
        reimplemented here": nothing in this module can create, write, or
        exec anything, so it can never own the privileged credential write
        or hand-roll a refresh.

        A source-level tripwire is the right shape for this: it fails when
        someone ADDS the capability, which is the moment worth catching, and
        no behavioral test can prove the absence of a write path."""
        source = open(oauth.__file__, encoding="utf-8").read()
        for forbidden in (
            "O_WRONLY", "O_RDWR", "O_CREAT", "O_APPEND", "O_TRUNC",
            "os.write(", "write_text", "write_bytes", "subprocess", "shutil",
            "urllib", "socket", "requests",
        ):
            assert forbidden not in source, f"oauth.py must not use {forbidden!r}"

    def test_the_state_constants_are_the_documented_wire_values(self):
        """These strings cross into ``run_worker``'s result as
        ``oauth_state``; renaming one silently changes the tool contract."""
        assert oauth.STATE_FRESH == "fresh"
        assert oauth.STATE_REFRESHABLE_EXPIRED == "refreshable_expired"
        assert oauth.STATE_EXPIRED == "expired"
        assert oauth.STATE_INVALID == "invalid"
        assert oauth.STATE_MALFORMED == "malformed"
        assert oauth.STATE_MISSING == "missing"
        assert oauth.STATE_ERROR == "error"

    def test_the_freshness_margin_is_a_meaningful_bound(self):
        assert oauth.FRESHNESS_MARGIN_SECONDS >= 30
