"""RED->GREEN tests for ``plugins/claude-worker/review.py`` (requirement 10).

Terra review reuses the existing auxiliary-task system
(``agent.auxiliary_client.call_llm(task="claude_worker_review")``) — no new
provider, no ``model.default`` change. Runs only for substantial successful
changes, and a review failure must never fail the worker's own result
(advisory / non-fatal).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tests.plugins._claude_worker_helpers import load_submodule

review = load_submodule("review")


class TestShouldReview:
    def test_below_threshold_skips(self):
        assert review.should_review(["a.py", "b.py"], min_changed_files=3) is False

    def test_at_or_above_threshold_reviews(self):
        assert review.should_review(["a.py", "b.py", "c.py"], min_changed_files=3) is True
        assert review.should_review(["a.py", "b.py", "c.py", "d.py"], min_changed_files=3) is True

    def test_empty_files_never_reviews(self):
        assert review.should_review([], min_changed_files=1) is False

    def test_threshold_floor_is_at_least_one(self):
        assert review.should_review(["a.py"], min_changed_files=0) is True


class TestRunReview:
    def test_calls_auxiliary_task_claude_worker_review(self, monkeypatch):
        captured = {}

        def _fake_call_llm(task=None, *, messages, **kwargs):
            captured["task"] = task
            captured["messages"] = messages
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="Looks fine, minor nit on naming."))]
            )

        monkeypatch.setattr(review, "_call_llm", _fake_call_llm)
        result = review.run_review(task="add a feature", files_touched=["a.py", "b.py", "c.py"], summary="did stuff")

        assert captured["task"] == "claude_worker_review"
        assert result["reviewed"] is True
        assert "minor nit" in result["notes"]

    def test_never_registers_a_provider_or_touches_model_default(self, monkeypatch):
        # run_review must go ONLY through the auxiliary call_llm bridge —
        # never through provider registration or config writes.
        import hermes_cli.config as hc_config

        calls = {"save": 0}
        original_save = getattr(hc_config, "save_config_value", None)
        if original_save is not None:
            monkeypatch.setattr(
                hc_config, "save_config_value",
                lambda *a, **k: calls.__setitem__("save", calls["save"] + 1),
            )

        def _fake_call_llm(task=None, *, messages, **kwargs):
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))]
            )

        monkeypatch.setattr(review, "_call_llm", _fake_call_llm)
        review.run_review(task="t", files_touched=["a.py"] * 5, summary="s")
        assert calls["save"] == 0

    def test_review_failure_is_advisory_not_fatal(self, monkeypatch):
        def _boom(task=None, *, messages, **kwargs):
            raise RuntimeError("auxiliary model unavailable")

        monkeypatch.setattr(review, "_call_llm", _boom)
        result = review.run_review(task="t", files_touched=["a.py"] * 5, summary="s")
        assert result["reviewed"] is False
        assert "error" in result

    def test_run_review_never_raises(self, monkeypatch):
        def _boom(task=None, *, messages, **kwargs):
            raise RuntimeError("boom")

        monkeypatch.setattr(review, "_call_llm", _boom)
        # Must not raise even on a totally broken auxiliary backend.
        review.run_review(task="t", files_touched=["a.py"], summary="s")


class TestTerraFallback:
    """The explicit contingency when the worker itself is unavailable: only a
    real, non-empty Terra result counts. Anything else stays a HOLD, which is
    what ``gate.on_post_tool_call`` keys the unlock off."""

    def _stub(self, monkeypatch, content):
        def _fake_call_llm(task=None, *, messages, **kwargs):
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
            )

        monkeypatch.setattr(review, "_call_llm", _fake_call_llm)

    def test_valid_result_is_ok_with_notes(self, monkeypatch):
        self._stub(monkeypatch, "Edit config.py line 40 and re-run the suite.")
        result = review.run_fallback(task="fix the loader", reason="breaker open: auth")
        assert result["ok"] is True
        assert "config.py" in result["notes"]

    def test_uses_the_same_auxiliary_task(self, monkeypatch):
        captured = {}

        def _fake_call_llm(task=None, *, messages, **kwargs):
            captured["task"] = task
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))])

        monkeypatch.setattr(review, "_call_llm", _fake_call_llm)
        review.run_fallback(task="t")
        assert captured["task"] == "claude_worker_review"

    @pytest.mark.parametrize("content", ["", "   \n\t ", None, 42])
    def test_empty_or_malformed_result_is_not_ok(self, monkeypatch, content):
        self._stub(monkeypatch, content)
        result = review.run_fallback(task="t")
        assert result["ok"] is False
        assert "error" in result

    def test_backend_failure_is_not_ok_and_never_raises(self, monkeypatch):
        def _boom(task=None, *, messages, **kwargs):
            raise RuntimeError("terra unavailable")

        monkeypatch.setattr(review, "_call_llm", _boom)
        result = review.run_fallback(task="t", reason="breaker open: rate")
        assert result["ok"] is False
        assert "terra unavailable" in result["error"]

    def test_malformed_response_shape_is_not_ok(self, monkeypatch):
        monkeypatch.setattr(
            review, "_call_llm",
            lambda task=None, *, messages, **kwargs: SimpleNamespace(choices=[]),
        )
        assert review.run_fallback(task="t")["ok"] is False


class TestAuxiliaryTaskKey:
    def test_task_key_constant(self):
        assert review.AUX_TASK_KEY == "claude_worker_review"

    def test_default_routing_targets_terra(self):
        assert review.AUX_TASK_DEFAULTS["provider"] == "openai-codex"
        assert review.AUX_TASK_DEFAULTS["model"] == "gpt-5.6-terra"
