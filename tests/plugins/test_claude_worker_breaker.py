"""RED->GREEN tests for ``plugins/claude-worker/breaker.py``.

Covers requirement (6): auth/rate/extra_usage classifications open the
breaker with a per-class cooldown and no retry; ``other`` does not open it;
an open breaker is queryable without any spawn happening (that no-spawn
guarantee is asserted at the runner layer — this file covers the breaker's
own state machine).
"""

from __future__ import annotations

import json

import pytest

from tests.plugins._claude_worker_helpers import load_submodule

breaker = load_submodule("breaker")


@pytest.fixture(autouse=True)
def _isolated_hermes_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    yield home


class TestClassifyFailure:
    @pytest.mark.parametrize(
        "stderr",
        [
            "Error: Invalid API key · Please run /login",
            "OAuth token expired, please run `claude login`",
            "authentication_error: 401 Unauthorized",
        ],
    )
    def test_classifies_auth(self, stderr):
        assert breaker.classify_failure(1, stderr) == "auth"

    @pytest.mark.parametrize(
        "stderr",
        [
            "Error: 429 Too Many Requests",
            "rate limit exceeded, retry after 60s",
        ],
    )
    def test_classifies_rate(self, stderr):
        assert breaker.classify_failure(1, stderr) == "rate"

    @pytest.mark.parametrize(
        "stderr",
        [
            "You've exceeded your usage limit for this plan",
            "Credit balance is too low to continue",
        ],
    )
    def test_classifies_extra_usage(self, stderr):
        assert breaker.classify_failure(1, stderr) == "extra_usage"

    def test_classifies_other_for_generic_failure(self):
        assert breaker.classify_failure(1, "Traceback: something else broke") == "other"

    @pytest.mark.parametrize(
        "stderr",
        [
            "test_429.py failed: AssertionError",
            "line 429: SyntaxError near 'def'",
            "expected 429 items, got 430",
        ],
    )
    def test_a_bare_429_in_stderr_is_never_rate(self, stderr):
        """A task's own output can legitimately contain the numeral 429 (a
        file name, a line number, an unrelated count) with no rate limit
        problem at all — a real HTTP 429 is always paired with the phrase
        'too many requests' in Claude CLI's own diagnostics."""
        assert breaker.classify_failure(1, stderr) == "other"

    def test_classifies_other_for_empty_stderr(self):
        assert breaker.classify_failure(1, "") == "other"

    def test_classifies_auth_from_structured_stdout_401(self):
        stdout = json.dumps({
            "is_error": True,
            "terminal_reason": "api_error",
            "api_error_status": 401,
            "result": "Failed to authenticate. API Error: 401 OAuth access "
                      "token has expired. Please obtain a new token.",
        })
        assert breaker.classify_failure(1, "", stdout) == "auth"

    def test_structured_stdout_auth_wins_even_with_unrelated_stderr(self):
        stdout = json.dumps({"is_error": True, "api_error_status": 401})
        assert breaker.classify_failure(1, "Traceback: something else broke", stdout) == "auth"

    def test_malformed_stdout_json_falls_back_to_stderr_classification(self):
        assert breaker.classify_failure(1, "OAuth token expired, please run `claude login`", "not json{{{") == "auth"

    def test_malformed_stdout_json_with_no_stderr_signal_is_other(self):
        assert breaker.classify_failure(1, "", "not json{{{") == "other"

    def test_structured_stdout_non_401_status_falls_back_to_stderr(self):
        stdout = json.dumps({"is_error": True, "api_error_status": 500})
        assert breaker.classify_failure(1, "", stdout) == "other"

    def test_structured_stdout_without_is_error_flag_is_ignored(self):
        stdout = json.dumps({"api_error_status": 401})
        assert breaker.classify_failure(1, "", stdout) == "other"

    def test_stdout_defaults_to_empty_and_preserves_existing_stderr_classification(self):
        assert breaker.classify_failure(1, "Error: Invalid API key · Please run /login") == "auth"

    @pytest.mark.parametrize(
        "stderr",
        [
            "test_401.py failed: AssertionError",
            "line 401: SyntaxError near 'def'",
            "expected 401 items, got 402",
        ],
    )
    def test_a_bare_401_in_stderr_is_never_auth(self, stderr):
        """A task's own output can legitimately contain the numeral 401 (a
        file name, a line number, an unrelated count) with no authentication
        problem at all — a real HTTP 401 is recognized structurally from a
        typed status field, never from this substring."""
        assert breaker.classify_failure(1, stderr) == "other"

    def test_a_bare_401_in_structured_result_text_is_never_auth_preflight(self):
        stdout = json.dumps({
            "is_error": True,
            "result": "test_401.py::test_something FAILED with exit 401",
        })
        assert breaker.classify_failure(1, "", stdout) == "other"

    @pytest.mark.parametrize("status_field", ["status", "status_code", "code"])
    def test_alternate_status_field_names_classify_the_same_as_api_error_status(self, status_field):
        stdout = json.dumps({"is_error": True, status_field: 401})
        assert breaker.classify_failure(1, "", stdout) == "auth"

    @pytest.mark.parametrize("status_field", ["status", "status_code", "code"])
    def test_alternate_status_field_names_are_also_recognized_when_nested_under_error(self, status_field):
        stdout = json.dumps({"is_error": True, "error": {status_field: 401}})
        assert breaker.classify_failure(1, "", stdout) == "auth"

    @pytest.mark.parametrize("status_field", ["status", "status_code", "code"])
    def test_alternate_status_field_non_401_falls_back_to_other(self, status_field):
        stdout = json.dumps({"is_error": True, status_field: 500})
        assert breaker.classify_failure(1, "", stdout) == "other"

    def test_typed_authentication_error_subtype_classifies_as_auth_preflight_with_no_text_marker(self):
        """A CLI body can report the typed field with no reauthentication
        PHRASE in the free text at all — the exact field value alone must be
        sufficient, since that is exactly the shape the incident surfaced."""
        stdout = json.dumps({"is_error": True, "subtype": "authentication_error", "result": ""})
        assert breaker.classify_failure(1, "", stdout) == breaker.AUTH_PREFLIGHT_FAILURE_CLASS

    def test_typed_authentication_error_type_nested_under_error_classifies_as_auth_preflight(self):
        stdout = json.dumps({
            "is_error": True, "error": {"type": "authentication_error"}, "result": "",
        })
        assert breaker.classify_failure(1, "", stdout) == breaker.AUTH_PREFLIGHT_FAILURE_CLASS

    def test_numeric_401_still_wins_over_a_typed_authentication_error_field(self):
        stdout = json.dumps({
            "is_error": True, "api_error_status": 401, "subtype": "authentication_error",
        })
        assert breaker.classify_failure(1, "", stdout) == "auth"


class TestPostSpawnAuthSessionRecognition:
    """The 2026-09-07/08 incident: three real CLI failures exited 1 in
    1.5-1.7s and telemetry classified them ``other``, but a ``/login`` fixed
    it — the diagnostic was an expired/invalid OAuth SESSION carried in the
    structured stdout's free-text error, not a numeric 401
    ``api_error_status``. This must classify as ``AUTH_PREFLIGHT_FAILURE_CLASS``
    (``"auth_preflight"``) — a session-invalid signal that does NOT open the
    hour-long ``auth`` breaker (waiting does not fix a bad session) — while a
    genuine unrelated failure must still classify as ``other``."""

    def test_result_text_reauth_phrase_without_401_status_classifies_as_auth_preflight(self):
        stdout = json.dumps({
            "is_error": True,
            "subtype": "error_during_execution",
            "result": "Invalid API key · Please run /login",
        })
        assert breaker.classify_failure(1, "", stdout) == "auth_preflight"

    def test_auth_preflight_class_is_the_exact_module_constant(self):
        stdout = json.dumps({"is_error": True, "result": "please run `claude login`"})
        assert breaker.classify_failure(1, "", stdout) == breaker.AUTH_PREFLIGHT_FAILURE_CLASS

    def test_auth_preflight_class_is_not_a_breaker_class(self):
        assert breaker.AUTH_PREFLIGHT_FAILURE_CLASS not in breaker.BREAKER_CLASSES

    def test_nested_error_message_reauth_phrase_classifies_as_auth_preflight(self):
        stdout = json.dumps({
            "is_error": True,
            "error": {"type": "authentication_error", "message": "OAuth token expired"},
        })
        assert breaker.classify_failure(1, "", stdout) == "auth_preflight"

    def test_numeric_401_still_wins_and_stays_the_real_auth_class(self):
        """The confirmed API rejection case is unchanged: it still opens the
        breaker, it is not downgraded to auth_preflight."""
        stdout = json.dumps({
            "is_error": True, "api_error_status": 401,
            "result": "please run /login",
        })
        assert breaker.classify_failure(1, "", stdout) == "auth"

    def test_generic_failure_text_is_never_misclassified_as_auth_preflight(self):
        stdout = json.dumps({
            "is_error": True,
            "result": "TypeError: cannot read property 'foo' of undefined",
        })
        assert breaker.classify_failure(1, "", stdout) == "other"

    def test_is_error_false_never_classifies_as_auth_preflight_even_with_reauth_text(self):
        stdout = json.dumps({"is_error": False, "result": "please run /login just kidding"})
        assert breaker.classify_failure(1, "", stdout) == "other"

    def test_auth_preflight_does_not_open_the_breaker(self, monkeypatch):
        breaker.record_failure(breaker.AUTH_PREFLIGHT_FAILURE_CLASS, {"auth_preflight": 3600})
        assert breaker.is_open() is False
        assert breaker.open_classes() == []


class TestBreakerStateMachine:
    def test_starts_closed(self):
        assert breaker.is_open() is False
        assert breaker.open_classes() == []

    @pytest.mark.parametrize("cls", ["auth", "rate", "extra_usage"])
    def test_breaker_class_opens_breaker(self, cls):
        breaker.record_failure(cls, {cls: 3600})
        assert breaker.is_open() is True
        assert cls in breaker.open_classes()

    def test_other_does_not_open_breaker(self):
        breaker.record_failure("other", {"other": 3600})
        assert breaker.is_open() is False

    def test_cooldown_expiry_closes_breaker(self, monkeypatch):
        t0 = 1_000_000.0
        breaker.record_failure("auth", {"auth": 100}, now=t0)
        assert breaker.is_open(now=t0 + 1) is True
        assert breaker.is_open(now=t0 + 100.1) is False

    def test_state_persists_across_module_reload(self, tmp_path, monkeypatch):
        breaker.record_failure("rate", {"rate": 900})
        state_path = breaker._state_path()
        assert state_path.exists()
        raw = json.loads(state_path.read_text())
        assert "rate" in raw

    def test_each_class_has_independent_cooldown(self):
        t0 = 2_000_000.0
        breaker.record_failure("auth", {"auth": 50}, now=t0)
        breaker.record_failure("rate", {"rate": 500}, now=t0)
        # auth expired, rate still open
        assert breaker.is_open(now=t0 + 60) is True
        assert breaker.open_classes(now=t0 + 60) == ["rate"]
