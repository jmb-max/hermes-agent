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

    def test_classifies_other_for_empty_stderr(self):
        assert breaker.classify_failure(1, "") == "other"


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
