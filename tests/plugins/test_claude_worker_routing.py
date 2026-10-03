"""Tests for ``plugins/claude-worker/routing.py``.

Routing is a closed, non-configurable contract: literal ``claude-sonnet-5``
is ALWAYS the selection when the caller omits ``complexity``, and literal
``claude-opus-5`` is selected ONLY when the caller passes an explicit,
non-null ``complexity`` that exactly matches one of the values the tool
schema already declares (``architecture`` / ``security`` /
``hard_debugging``). The text of ``task`` never influences routing — there
is no keyword/heuristic scan of any kind — and there is no failure-driven
escalation: ``choose_model`` takes no ``attempt`` or ``previous_failure_class``
input at all, because a Sonnet failure of any class is never retried on a
different model.
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

    def test_choose_model_has_no_escalation_surface(self):
        """There is no escalation of any kind left to parameterize: no
        ``attempt`` counter and no ``previous_failure_class`` input — a
        failure is never a reason to pick a different model."""
        params = set(inspect.signature(routing.choose_model).parameters)
        assert "attempt" not in params
        assert "previous_failure_class" not in params

    def test_attempt_kwarg_is_rejected(self):
        with pytest.raises(TypeError):
            routing.choose_model(task="fix a typo", attempt=1)

    def test_previous_failure_class_kwarg_is_rejected(self):
        with pytest.raises(TypeError):
            routing.choose_model(task="fix a typo", previous_failure_class="other")

    def test_module_never_loads_plugin_config(self):
        source = open(routing.__file__, encoding="utf-8").read()
        assert "load_plugin_config" not in source

    def test_module_has_no_keyword_heuristic_scan(self):
        """The whole keyword/heuristic classifier is gone, not merely
        unreachable — asserted at the source level so a reintroduced keyword
        list fails this test even if no behavioral test happens to trip it."""
        source = open(routing.__file__, encoding="utf-8").read()
        for token in (
            "KEYWORDS", "lower_task", "_classify_complexity",
            "segfault", "vulnerability", "system design",
        ):
            assert token not in source


class TestDefaultRouting:
    def test_default_is_sonnet(self):
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
            "propose a system design for the new ingestion pipeline",
            "investigate this potential SQL injection vulnerability",
            "assess whether this endpoint has an auth bypass",
            "track down this flaky race condition in the test suite",
            "diagnose an intermittent deadlock under load",
        ],
    )
    def test_sensitive_vocabulary_without_complexity_stays_sonnet(self, task):
        """Requirement 3: the CONTENT of ``task`` must never change the
        model. These read exactly like the retired keyword-triggered
        categories (architecture/security/hard_debugging) but carry no
        explicit ``complexity`` — they must route to Sonnet like any other
        task, with the plain ``"default"`` reason."""
        model, reason = routing.choose_model(task=task)
        assert model == "claude-sonnet-5"
        assert reason == "default"


class TestExplicitComplexityIsTheOnlyOpusPath:
    @pytest.mark.parametrize("complexity", ["architecture", "security", "hard_debugging"])
    def test_each_schema_allowed_value_routes_to_opus(self, complexity):
        model, reason = routing.choose_model(task="fix a typo", complexity=complexity)
        assert model == "claude-opus-5"
        assert complexity in reason

    def test_explicit_complexity_wins_even_over_a_plain_task(self):
        model, _reason = routing.choose_model(
            task="trivial one-line fix", complexity="security",
        )
        assert model == "claude-opus-5"

    @pytest.mark.parametrize(
        "bad", ["not-a-real-category", "", "ARCHITECTURE", "Security", None, "  security  "],
    )
    def test_non_matching_complexity_falls_back_to_default(self, bad):
        """Only an EXACT match against the schema's declared enum values
        selects Opus — no case-folding, no whitespace trimming, and no
        fallback to any other classification. Anything else (an unknown
        string, an empty string, wrong case, untrimmed whitespace, or
        ``None``) is the plain default."""
        model, reason = routing.choose_model(task="fix a typo", complexity=bad)
        assert model == "claude-sonnet-5"
        assert reason == "default"
