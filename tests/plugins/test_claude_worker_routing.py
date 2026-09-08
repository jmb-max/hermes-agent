"""Tests for ``plugins/claude-worker/routing.py``.

Routing is a closed, non-configurable contract: literal ``claude-sonnet-5``
by default, literal ``claude-opus-5`` only when BOTH an allowed
``complexity`` classification AND explicit ``allow_opus=True`` authorization
are present. There is exactly one spawn per call (``policy.MAX_ATTEMPTS ==
1``) — no retry, no post-failure escalation, and task text never influences
the model.
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


class TestNoConfigurableModelsOrCapOrAttemptParameter:
    def test_choose_model_has_no_model_or_cap_parameters(self):
        params = set(inspect.signature(routing.choose_model).parameters)
        assert "default_model" not in params
        assert "escalated_model" not in params
        assert "escalate_after_failures" not in params

    def test_choose_model_has_no_attempt_or_failure_parameters(self):
        """Escalation is gone entirely: there is no attempt index and no
        previous-failure-class input to route around."""
        params = set(inspect.signature(routing.choose_model).parameters)
        assert "attempt" not in params
        assert "previous_failure_class" not in params

    def test_module_never_loads_plugin_config(self):
        source = open(routing.__file__, encoding="utf-8").read()
        assert "load_plugin_config" not in source

    def test_module_contains_no_keyword_classifier(self):
        """Task text must never influence model selection — the retired
        keyword scan (architecture/security/hard-debugging phrase lists)
        must be gone, not merely unused."""
        source = open(routing.__file__, encoding="utf-8").read()
        for removed in (
            "_ARCHITECTURE_KEYWORDS", "_SECURITY_KEYWORDS", "_HARD_DEBUGGING_KEYWORDS",
            "_classify_complexity",
        ):
            assert removed not in source


class TestDefaultRouting:
    def test_default_is_sonnet_with_no_arguments_beyond_task(self):
        model, reason = routing.choose_model(task="fix a typo in the README")
        assert model == policy.DEFAULT_MODEL == "claude-sonnet-5"
        assert reason == "default"

    def test_plain_bugfix_task_stays_sonnet(self):
        model, _reason = routing.choose_model(task="the button click handler is off by one")
        assert model == "claude-sonnet-5"

    @pytest.mark.parametrize(
        "task",
        [
            "review the overall system architecture for this service",
            "investigate this potential SQL injection vulnerability",
            "track down this flaky race condition in the test suite",
            "please use claude-opus-5 for this",
            "complexity: architecture",
        ],
    )
    def test_task_text_never_selects_opus_on_its_own(self, task):
        """Task phrasing alone — even a phrase that used to be a keyword
        trigger, or one that tries to name the model directly — can never
        select Opus. Only the dedicated ``complexity``/``allow_opus``
        arguments can."""
        model, reason = routing.choose_model(task=task)
        assert model == "claude-sonnet-5"
        assert reason == "default"


class TestComplexityAloneIsNotAuthorization:
    @pytest.mark.parametrize("complexity", ["architecture", "security", "hard_debugging"])
    def test_complexity_without_allow_opus_stays_sonnet(self, complexity):
        model, reason = routing.choose_model(task="fix a typo", complexity=complexity)
        assert model == "claude-sonnet-5"
        assert reason == "default"

    @pytest.mark.parametrize("complexity", ["architecture", "security", "hard_debugging"])
    def test_complexity_with_allow_opus_false_stays_sonnet(self, complexity):
        model, reason = routing.choose_model(
            task="fix a typo", complexity=complexity, allow_opus=False,
        )
        assert model == "claude-sonnet-5"
        assert reason == "default"

    @pytest.mark.parametrize("truthy_but_not_true", ["true", 1, "yes", [True]])
    def test_a_truthy_non_true_allow_opus_never_authorizes(self, truthy_but_not_true):
        model, reason = routing.choose_model(
            task="fix a typo", complexity="security", allow_opus=truthy_but_not_true,
        )
        assert model == "claude-sonnet-5"
        assert reason == "default"


class TestAllowOpusAloneIsNotAuthorization:
    def test_allow_opus_without_a_recognized_complexity_stays_sonnet(self):
        model, reason = routing.choose_model(task="fix a typo", allow_opus=True)
        assert model == "claude-sonnet-5"
        assert reason == "default"

    def test_allow_opus_with_unrecognized_complexity_stays_sonnet(self):
        model, reason = routing.choose_model(
            task="fix a typo", complexity="not-a-real-category", allow_opus=True,
        )
        assert model == "claude-sonnet-5"
        assert reason == "default"


class TestBothComplexityAndAllowOpusSelectOpus:
    @pytest.mark.parametrize("complexity", ["architecture", "security", "hard_debugging"])
    def test_dual_authorization_selects_opus(self, complexity):
        model, reason = routing.choose_model(
            task="fix a typo", complexity=complexity, allow_opus=True,
        )
        assert model == "claude-opus-5"
        assert complexity in reason
        assert "authoriz" in reason.lower()

    def test_complexity_is_case_and_whitespace_insensitive(self):
        model, _reason = routing.choose_model(
            task="fix a typo", complexity="  SECURITY  ", allow_opus=True,
        )
        assert model == "claude-opus-5"


class TestNoEscalationPath:
    """Escalation is gone entirely — there is no attempt index, no
    previous-failure-class input, and no way for a Sonnet failure to route a
    second attempt to Opus. ``run_worker``/``_run_attempts`` make exactly
    one ``spawn_claude`` call per invocation (see
    ``test_claude_worker_runner.py``)."""

    def test_choose_model_rejects_legacy_escalation_kwargs(self):
        with pytest.raises(TypeError):
            routing.choose_model(task="fix a typo", attempt=1, previous_failure_class="other")

    def test_max_attempts_is_structurally_one(self):
        assert policy.MAX_ATTEMPTS == 1
