"""Tests for ``plugins/claude-worker/routing.py``.

Routing is a closed, non-configurable contract: literal ``claude-sonnet-5``
by default, literal ``claude-opus-5`` for architecture/security/hard-debugging
or for the single escalation after one non-breaker Sonnet failure. The
attempt cap is structural (``policy.MAX_ATTEMPTS == 2``) and there is no
config-supplied model or cap parameter to swap.
"""

from __future__ import annotations

import inspect

import pytest

from tests.plugins._claude_worker_helpers import load_submodule

routing = load_submodule("routing")
policy = load_submodule("policy")


class TestModelAllowlist:
    def test_allowlist_contains_exactly_sonnet_and_opus(self):
        assert routing.MODEL_ALLOWLIST == frozenset({"claude-sonnet-5", "claude-opus-5"})

    def test_validate_model_accepts_allowlisted(self):
        assert routing.validate_model("claude-sonnet-5") == "claude-sonnet-5"
        assert routing.validate_model("claude-opus-5") == "claude-opus-5"

    @pytest.mark.parametrize("bad", ["claude-3-opus", "gpt-5.6-terra", "", "claude-sonnet-5 "])
    def test_validate_model_rejects_non_allowlisted(self, bad):
        with pytest.raises(routing.ModelNotAllowedError):
            routing.validate_model(bad)


class TestNoConfigurableModelsOrCap:
    def test_choose_model_has_no_model_or_cap_parameters(self):
        params = set(inspect.signature(routing.choose_model).parameters)
        assert "default_model" not in params
        assert "escalated_model" not in params
        assert "escalate_after_failures" not in params

    def test_module_never_loads_plugin_config(self):
        source = open(routing.__file__, encoding="utf-8").read()
        assert "load_plugin_config" not in source


class TestDefaultRouting:
    def test_default_is_sonnet(self):
        model, reason = routing.choose_model(task="fix a typo in the README", attempt=0)
        assert model == policy.DEFAULT_MODEL == "claude-sonnet-5"
        assert "default" in reason

    def test_plain_bugfix_task_stays_sonnet(self):
        model, _reason = routing.choose_model(task="the button click handler is off by one", attempt=0)
        assert model == "claude-sonnet-5"


class TestComplexityClassification:
    @pytest.mark.parametrize(
        "task",
        [
            "review the overall system architecture for this service",
            "propose a system design for the new ingestion pipeline",
        ],
    )
    def test_architecture_keyword_routes_to_opus(self, task):
        model, reason = routing.choose_model(task=task, attempt=0)
        assert model == "claude-opus-5"
        assert "architecture" in reason

    @pytest.mark.parametrize(
        "task",
        [
            "investigate this potential SQL injection vulnerability",
            "assess whether this endpoint has an auth bypass",
        ],
    )
    def test_security_keyword_routes_to_opus(self, task):
        model, reason = routing.choose_model(task=task, attempt=0)
        assert model == "claude-opus-5"
        assert "security" in reason

    @pytest.mark.parametrize(
        "task",
        [
            "track down this flaky race condition in the test suite",
            "diagnose an intermittent deadlock under load",
        ],
    )
    def test_hard_debugging_keyword_routes_to_opus(self, task):
        model, reason = routing.choose_model(task=task, attempt=0)
        assert model == "claude-opus-5"
        assert "hard_debugging" in reason

    def test_explicit_complexity_arg_overrides_keyword_scan(self):
        model, reason = routing.choose_model(task="fix a typo", attempt=0, complexity="security")
        assert model == "claude-opus-5"
        assert "security" in reason

    def test_unknown_complexity_arg_falls_back_to_keyword_scan(self):
        model, _reason = routing.choose_model(
            task="fix a typo", attempt=0, complexity="not-a-real-category",
        )
        assert model == "claude-sonnet-5"


class TestEscalation:
    def test_escalates_to_opus_after_one_sonnet_non_breaker_failure(self):
        model, reason = routing.choose_model(
            task="fix a typo", attempt=1, previous_failure_class="other",
        )
        assert model == "claude-opus-5"
        assert "escalat" in reason.lower()

    def test_no_third_spawn_raises(self):
        with pytest.raises(RuntimeError):
            routing.choose_model(task="fix a typo", attempt=2, previous_failure_class="other")

    @pytest.mark.parametrize("attempt", [-1, 3, 99])
    def test_out_of_range_attempts_raise(self, attempt):
        with pytest.raises(RuntimeError):
            routing.choose_model(task="fix a typo", attempt=attempt, previous_failure_class="other")

    @pytest.mark.parametrize("failure_class", ["auth", "rate", "extra_usage"])
    def test_breaker_class_failure_never_escalates(self, failure_class):
        with pytest.raises(RuntimeError):
            routing.choose_model(task="fix a typo", attempt=1, previous_failure_class=failure_class)

    def test_escalation_requires_a_previous_failure(self):
        with pytest.raises(RuntimeError):
            routing.choose_model(task="fix a typo", attempt=1, previous_failure_class=None)

    def test_initial_opus_task_gets_no_second_attempt(self):
        """A category task starts on Opus; there is nothing to escalate to,
        so a second attempt is structurally refused."""
        first, _reason = routing.choose_model(task="security audit of the auth flow", attempt=0)
        assert first == "claude-opus-5"
        with pytest.raises(RuntimeError):
            routing.choose_model(task="security audit of the auth flow", attempt=1,
                                 previous_failure_class="other")

    def test_explicit_complexity_also_blocks_a_second_attempt(self):
        with pytest.raises(RuntimeError):
            routing.choose_model(task="fix a typo", attempt=1, complexity="architecture",
                                 previous_failure_class="other")
