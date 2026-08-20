"""Tests for ``plugins/claude-worker/gate.py`` — the write gate.

Asserted through the REAL enforcement path
(``hermes_cli.plugins.resolve_pre_tool_block``) so these prove the actual
tool-dispatch sites block, not merely that a callback returns a dict.

Post-remediation invariants:
  * the gated tool set (patch/write_file/skill_manage) and the gate itself
    are immutable inside an eligible canary session — config cannot relax them;
  * scope is exactly the canonical configured repo roots shared with the
    runner — no ``.git``-ancestor expansion, so the gate can never block a
    repo the worker is not allowed to run in;
  * an OPEN BREAKER NEVER UNLOCKS direct edits. Only a successful worker run,
    or an explicit Terra fallback that actually returned a result, unlocks;
  * a mutating ``skill_manage`` call whose target path cannot be resolved
    fails closed.
"""

from __future__ import annotations

import json

import pytest

import hermes_cli.plugins as hermes_plugins
from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest, invoke_hook, resolve_pre_tool_block
from gateway.session_context import _UNSET, _VAR_MAP

from tests.plugins._claude_worker_helpers import load_submodule

canary = load_submodule("canary")
gate = load_submodule("gate")
breaker = load_submodule("breaker")
config = load_submodule("config")
policy = load_submodule("policy")

CANARY_KEY = "sess:gate-canary"
NON_CANARY_KEY = "sess:gate-non-canary"


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))

    saved_ctx = {name: var.get() for name, var in _VAR_MAP.items()}
    for var in _VAR_MAP.values():
        var.set(_UNSET)
    canary._identity_cache.clear()
    gate.reset_unlocks()

    saved_manager = hermes_plugins._plugin_manager
    manager = PluginManager()
    manifest = PluginManifest(name="claude-worker", key="claude-worker", source="bundled")
    ctx = PluginContext(manifest, manager)
    ctx.register_hook("pre_tool_call", gate.on_pre_tool_call)
    ctx.register_hook("post_tool_call", gate.on_post_tool_call)
    hermes_plugins._plugin_manager = manager

    yield

    hermes_plugins._plugin_manager = saved_manager
    for var, val in zip(_VAR_MAP.values(), saved_ctx.values()):
        var.set(val)
    canary._identity_cache.clear()
    gate.reset_unlocks()


#: A cached source identity is never a bare bool anymore — it is re-applied
#: against the CURRENT config on every lookup (see canary.py). These tests
#: run under ``_gate_config(...)``, which never overrides ``canary`` at all,
#: so the full default policy channel set is in force; picking one of those
#: ids (or a chat id that is provably outside it) reproduces the old
#: True/False shorthand these tests relied on.
_CANARY_CHAT_ID = next(iter(policy.CANARY_CHANNEL_IDS))
_NON_CANARY_CHAT_ID = "000000000000000000"


def _canary_identity(eligible: bool = True) -> dict:
    chat_id = _CANARY_CHAT_ID if eligible else _NON_CANARY_CHAT_ID
    return {"platform": "discord", "chat_id": chat_id, "parent_chat_id": "", "thread_id": ""}


def _mark_canary(session_key: str, eligible: bool = True) -> None:
    canary._record(session_key, _canary_identity(eligible))
    _VAR_MAP["HERMES_SESSION_KEY"].set(session_key)


def _gate_config(repo_roots, **extra):
    entry = {"gate": {"repo_roots": [str(r) for r in repo_roots], **extra}}
    return {"plugins": {"entries": {"claude-worker": entry}}}


@pytest.fixture()
def repo(tmp_path):
    path = tmp_path / "repo"
    (path / ".git").mkdir(parents=True)
    target = path / "app.py"
    target.write_text("x = 1\n", encoding="utf-8")
    return path


def _patch(target, session_id=CANARY_KEY):
    return resolve_pre_tool_block(
        "patch", {"path": str(target), "old_string": "x = 1", "new_string": "x = 2"},
        session_id=session_id,
    )


class TestBlocksInsideCanaryAndConfiguredRoot:
    def test_patch_blocked_before_worker_run(self, repo, monkeypatch):
        monkeypatch.setattr(config, "_load_raw_config", lambda: _gate_config([repo]))
        _mark_canary(CANARY_KEY)
        message = _patch(repo / "app.py")
        assert message is not None
        assert "claude_worker" in message

    def test_write_file_blocked_too(self, repo, monkeypatch):
        monkeypatch.setattr(config, "_load_raw_config", lambda: _gate_config([repo]))
        _mark_canary(CANARY_KEY)
        message = resolve_pre_tool_block(
            "write_file", {"path": str(repo / "new.py"), "content": "print(1)"}, session_id=CANARY_KEY,
        )
        assert message is not None

    def test_write_file_with_missing_path_fails_closed(self, repo, monkeypatch):
        monkeypatch.setattr(config, "_load_raw_config", lambda: _gate_config([repo]))
        _mark_canary(CANARY_KEY)
        message = resolve_pre_tool_block("write_file", {"content": "print(1)"}, session_id=CANARY_KEY)
        assert message is not None


class TestGateIsImmutable:
    @pytest.mark.parametrize("relaxation", [
        {"enabled": False},
        {"tools": []},
        {"tools": ["read_file"]},
    ])
    def test_config_cannot_disable_or_narrow_the_gate(self, repo, monkeypatch, relaxation):
        monkeypatch.setattr(config, "_load_raw_config", lambda: _gate_config([repo], **relaxation))
        _mark_canary(CANARY_KEY)
        assert _patch(repo / "app.py") is not None

    def test_all_three_mandatory_tools_are_gated(self, repo, monkeypatch):
        monkeypatch.setattr(config, "_load_raw_config", lambda: _gate_config([repo]))
        monkeypatch.setattr(gate, "_resolve_skill_manage_path", lambda args: str(repo / "skills" / "s" / "SKILL.md"))
        _mark_canary(CANARY_KEY)
        assert _patch(repo / "app.py") is not None
        assert resolve_pre_tool_block(
            "write_file", {"path": str(repo / "b.py"), "content": "x"}, session_id=CANARY_KEY) is not None
        assert resolve_pre_tool_block(
            "skill_manage", {"action": "write_file", "name": "s", "file_content": "x"},
            session_id=CANARY_KEY) is not None


class TestAntiOverreach:
    def test_patch_allowed_outside_canary_session(self, repo, monkeypatch):
        monkeypatch.setattr(config, "_load_raw_config", lambda: _gate_config([repo]))
        _mark_canary(NON_CANARY_KEY, eligible=False)
        assert _patch(repo / "app.py", session_id=NON_CANARY_KEY) is None

    def test_unrelated_tool_name_is_never_gated(self, repo, monkeypatch):
        monkeypatch.setattr(config, "_load_raw_config", lambda: _gate_config([repo]))
        _mark_canary(CANARY_KEY)
        assert resolve_pre_tool_block(
            "read_file", {"path": str(repo / "app.py")}, session_id=CANARY_KEY) is None

    def test_path_outside_every_configured_root_is_allowed(self, tmp_path, repo, monkeypatch):
        scratch = tmp_path / "scratch"
        scratch.mkdir()
        (scratch / "notes.txt").write_text("hello\n", encoding="utf-8")
        monkeypatch.setattr(config, "_load_raw_config", lambda: _gate_config([repo]))
        _mark_canary(CANARY_KEY)
        assert resolve_pre_tool_block(
            "write_file", {"path": str(scratch / "notes.txt"), "content": "hi"}, session_id=CANARY_KEY,
        ) is None


class TestScopeConsistencyWithRunner:
    def test_unconfigured_git_repo_is_not_gated(self, tmp_path, monkeypatch):
        """Terra's deadlock: the gate used to block any ``.git`` ancestor
        while the runner only accepts configured roots, leaving a session
        that could neither edit nor delegate. Gate scope is now exactly the
        runner's scope."""
        other = tmp_path / "other-checkout"
        (other / ".git").mkdir(parents=True)
        (other / "app.py").write_text("x = 1\n", encoding="utf-8")
        configured = tmp_path / "configured"
        configured.mkdir()

        monkeypatch.setattr(config, "_load_raw_config", lambda: _gate_config([configured]))
        _mark_canary(CANARY_KEY)
        assert _patch(other / "app.py") is None

    def test_gate_and_runner_share_one_root_resolution(self, tmp_path, monkeypatch):
        runner = load_submodule("runner")
        root = tmp_path / "repo"
        (root / "src").mkdir(parents=True)
        cfg = {"gate": {"repo_roots": [str(root)]}}
        roots = policy.canonical_repo_roots(cfg)
        assert runner.validate_cwd(str(root / "src"), roots) == str((root / "src").resolve())
        assert policy.resolve_within_roots(str(root / "src" / "a.py"), roots) == str(root.resolve())

    def test_no_git_fallback_in_gate_source(self):
        source = open(gate.__file__, encoding="utf-8").read()
        assert '".git"' not in source


class TestBreakerNeverUnlocks:
    @pytest.mark.parametrize("failure_class", ["auth", "rate", "extra_usage"])
    def test_open_breaker_does_not_unlock_patch(self, repo, monkeypatch, failure_class):
        monkeypatch.setattr(config, "_load_raw_config", lambda: _gate_config([repo]))
        _mark_canary(CANARY_KEY)
        breaker.record_failure(failure_class, {failure_class: 3600})
        message = _patch(repo / "app.py")
        assert message is not None
        assert "claude_worker" in message

    def test_open_breaker_does_not_unlock_write_file(self, repo, monkeypatch):
        monkeypatch.setattr(config, "_load_raw_config", lambda: _gate_config([repo]))
        _mark_canary(CANARY_KEY)
        breaker.record_failure("auth", {"auth": 3600})
        assert resolve_pre_tool_block(
            "write_file", {"path": str(repo / "b.py"), "content": "x"}, session_id=CANARY_KEY) is not None

    def test_open_breaker_does_not_unlock_skill_manage(self, repo, monkeypatch):
        monkeypatch.setattr(config, "_load_raw_config", lambda: _gate_config([repo]))
        monkeypatch.setattr(gate, "_resolve_skill_manage_path", lambda args: str(repo / "skills" / "s" / "SKILL.md"))
        _mark_canary(CANARY_KEY)
        breaker.record_failure("rate", {"rate": 900})
        assert resolve_pre_tool_block(
            "skill_manage", {"action": "patch", "name": "s"}, session_id=CANARY_KEY) is not None

    def test_breaker_block_message_explains_the_hold(self, repo, monkeypatch):
        monkeypatch.setattr(config, "_load_raw_config", lambda: _gate_config([repo]))
        _mark_canary(CANARY_KEY)
        breaker.record_failure("auth", {"auth": 3600})
        message = _patch(repo / "app.py")
        assert "unavailable" in message.lower() or "hold" in message.lower()


class TestUnlockAfterSuccessfulRun:
    def _record_worker_result(self, payload, session_id=CANARY_KEY, status="ok"):
        invoke_hook(
            "post_tool_call", tool_name="claude_worker",
            args={"task": "fix it", "cwd": payload.get("cwd", "")},
            result=json.dumps(payload), session_id=session_id, status=status,
        )

    def test_gate_releases_after_successful_worker_run(self, repo, monkeypatch):
        monkeypatch.setattr(config, "_load_raw_config", lambda: _gate_config([repo]))
        _mark_canary(CANARY_KEY)
        assert _patch(repo / "app.py") is not None
        self._record_worker_result({"success": True, "cwd": str(repo)})
        assert _patch(repo / "app.py") is None

    def test_failed_worker_run_does_not_release_gate(self, repo, monkeypatch):
        monkeypatch.setattr(config, "_load_raw_config", lambda: _gate_config([repo]))
        _mark_canary(CANARY_KEY)
        self._record_worker_result({"success": False, "cwd": str(repo)}, status="error")
        assert _patch(repo / "app.py") is not None

    def test_success_in_a_different_repo_does_not_release_this_one(self, tmp_path, repo, monkeypatch):
        repo_b = tmp_path / "repo_b"
        repo_b.mkdir()
        monkeypatch.setattr(config, "_load_raw_config", lambda: _gate_config([repo, repo_b]))
        _mark_canary(CANARY_KEY)
        self._record_worker_result({"success": True, "cwd": str(repo_b)})
        assert _patch(repo / "app.py") is not None

    def test_success_in_a_different_session_does_not_release_this_one(self, repo, monkeypatch):
        monkeypatch.setattr(config, "_load_raw_config", lambda: _gate_config([repo]))
        canary._record("sess:other-canary", _canary_identity(True))
        _mark_canary(CANARY_KEY)
        self._record_worker_result({"success": True, "cwd": str(repo)}, session_id="sess:other-canary")
        _VAR_MAP["HERMES_SESSION_KEY"].set(CANARY_KEY)
        assert _patch(repo / "app.py") is not None

    def test_success_outside_configured_roots_is_ignored(self, tmp_path, repo, monkeypatch):
        outside = tmp_path / "outside"
        outside.mkdir()
        monkeypatch.setattr(config, "_load_raw_config", lambda: _gate_config([repo]))
        _mark_canary(CANARY_KEY)
        self._record_worker_result({"success": True, "cwd": str(outside)})
        assert _patch(repo / "app.py") is not None


class TestExplicitTerraFallbackUnlock:
    def _record(self, payload, session_id=CANARY_KEY, status="ok"):
        invoke_hook(
            "post_tool_call", tool_name="claude_worker",
            args={"task": "fix it", "cwd": payload.get("cwd", "")},
            result=json.dumps(payload), session_id=session_id, status=status,
        )

    def _valid_payload(self, repo):
        return {
            "success": False, "status": "fallback_ready", "fallback_ready": True,
            "fallback_requested": True, "fallback_provenance": "terra_auxiliary",
            "cwd": str(repo),
            "fallback": {"ok": True, "ready": True, "notes": "Terra guidance"},
        }

    def test_fallback_ready_marker_unlocks_the_gate(self, repo, monkeypatch):
        monkeypatch.setattr(config, "_load_raw_config", lambda: _gate_config([repo]))
        _mark_canary(CANARY_KEY)
        breaker.record_failure("auth", {"auth": 3600})
        assert _patch(repo / "app.py") is not None

        self._record(self._valid_payload(repo))
        assert _patch(repo / "app.py") is None

    def test_hold_status_keeps_the_gate_blocked(self, repo, monkeypatch):
        monkeypatch.setattr(config, "_load_raw_config", lambda: _gate_config([repo]))
        _mark_canary(CANARY_KEY)
        breaker.record_failure("auth", {"auth": 3600})
        self._record({
            "success": False, "status": "HOLD", "fallback_ready": False,
            "cwd": str(repo), "fallback": {"ok": False, "error": "terra unavailable"},
        })
        assert _patch(repo / "app.py") is not None

    def test_fallback_marker_is_scoped_to_its_repo(self, tmp_path, repo, monkeypatch):
        repo_b = tmp_path / "repo_b"
        repo_b.mkdir()
        monkeypatch.setattr(config, "_load_raw_config", lambda: _gate_config([repo, repo_b]))
        _mark_canary(CANARY_KEY)
        self._record(self._valid_payload(repo_b))
        assert _patch(repo / "app.py") is not None

    def test_fallback_marker_is_scoped_to_its_session(self, repo, monkeypatch):
        monkeypatch.setattr(config, "_load_raw_config", lambda: _gate_config([repo]))
        canary._record("sess:other-canary", _canary_identity(True))
        _mark_canary(CANARY_KEY)
        self._record(self._valid_payload(repo), session_id="sess:other-canary")
        _VAR_MAP["HERMES_SESSION_KEY"].set(CANARY_KEY)
        assert _patch(repo / "app.py") is not None

    def test_forged_marker_from_another_tool_is_ignored(self, repo, monkeypatch):
        monkeypatch.setattr(config, "_load_raw_config", lambda: _gate_config([repo]))
        _mark_canary(CANARY_KEY)
        invoke_hook(
            "post_tool_call", tool_name="write_file",
            args={"path": str(repo / "x.py")},
            result=json.dumps({"fallback_ready": True, "cwd": str(repo)}),
            session_id=CANARY_KEY, status="ok",
        )
        assert _patch(repo / "app.py") is not None

    def test_fallback_ready_without_a_fallback_result_is_ignored(self, repo, monkeypatch):
        monkeypatch.setattr(config, "_load_raw_config", lambda: _gate_config([repo]))
        _mark_canary(CANARY_KEY)
        self._record({"success": False, "fallback_ready": True, "cwd": str(repo), "fallback": {"ok": False}})
        assert _patch(repo / "app.py") is not None


class TestForgedFallbackProvenanceCannotUnlock:
    """A payload that gets every OTHER check right but forges/omits one
    provenance field must still stay HOLD — ``_fallback_delivered`` checks
    each of these independently, not just ``fallback_ready``."""

    def _record(self, payload, session_id=CANARY_KEY, status="ok"):
        invoke_hook(
            "post_tool_call", tool_name="claude_worker",
            args={"task": "fix it", "cwd": payload.get("cwd", "")},
            result=json.dumps(payload), session_id=session_id, status=status,
        )

    def _valid_payload(self, repo):
        return {
            "success": False, "status": "fallback_ready", "fallback_ready": True,
            "fallback_requested": True, "fallback_provenance": "terra_auxiliary",
            "cwd": str(repo),
            "fallback": {"ok": True, "ready": True, "notes": "Terra guidance"},
        }

    def _assert_still_blocked(self, repo, monkeypatch, payload):
        monkeypatch.setattr(config, "_load_raw_config", lambda: _gate_config([repo]))
        _mark_canary(CANARY_KEY)
        breaker.record_failure("auth", {"auth": 3600})
        self._record(payload)
        assert _patch(repo / "app.py") is not None

    def test_missing_fallback_requested_does_not_unlock(self, repo, monkeypatch):
        payload = self._valid_payload(repo)
        del payload["fallback_requested"]
        self._assert_still_blocked(repo, monkeypatch, payload)

    def test_string_true_fallback_requested_does_not_unlock(self, repo, monkeypatch):
        payload = self._valid_payload(repo)
        payload["fallback_requested"] = "true"
        self._assert_still_blocked(repo, monkeypatch, payload)

    def test_missing_fallback_provenance_does_not_unlock(self, repo, monkeypatch):
        payload = self._valid_payload(repo)
        del payload["fallback_provenance"]
        self._assert_still_blocked(repo, monkeypatch, payload)

    def test_wrong_fallback_provenance_does_not_unlock(self, repo, monkeypatch):
        payload = self._valid_payload(repo)
        payload["fallback_provenance"] = "attacker_supplied"
        self._assert_still_blocked(repo, monkeypatch, payload)

    def test_fallback_ready_truthy_but_not_literal_true_does_not_unlock(self, repo, monkeypatch):
        payload = self._valid_payload(repo)
        payload["fallback_ready"] = 1
        self._assert_still_blocked(repo, monkeypatch, payload)

    def test_missing_nested_ready_does_not_unlock(self, repo, monkeypatch):
        payload = self._valid_payload(repo)
        del payload["fallback"]["ready"]
        self._assert_still_blocked(repo, monkeypatch, payload)

    def test_nested_ready_false_does_not_unlock(self, repo, monkeypatch):
        payload = self._valid_payload(repo)
        payload["fallback"]["ready"] = False
        self._assert_still_blocked(repo, monkeypatch, payload)

    def test_nested_ready_truthy_but_not_literal_true_does_not_unlock(self, repo, monkeypatch):
        payload = self._valid_payload(repo)
        payload["fallback"]["ready"] = 1
        self._assert_still_blocked(repo, monkeypatch, payload)

    def test_empty_notes_does_not_unlock(self, repo, monkeypatch):
        payload = self._valid_payload(repo)
        payload["fallback"]["notes"] = "   "
        self._assert_still_blocked(repo, monkeypatch, payload)

    def test_fully_valid_payload_does_unlock_as_a_control(self, repo, monkeypatch):
        """Sanity control proving the helper payload itself is otherwise
        correct — every "forged" test above changes exactly one field away
        from this unlocking baseline."""
        monkeypatch.setattr(config, "_load_raw_config", lambda: _gate_config([repo]))
        _mark_canary(CANARY_KEY)
        breaker.record_failure("auth", {"auth": 3600})
        self._record(self._valid_payload(repo))
        assert _patch(repo / "app.py") is None


class TestSkillManageScope:
    @pytest.mark.parametrize("action", sorted(policy.SKILL_MANAGE_MUTATING_ACTIONS))
    def test_every_mutating_action_is_gated_inside_a_repo_root(self, repo, monkeypatch, action):
        skill_dir = repo / "skills" / "my-skill"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text("# skill\n", encoding="utf-8")
        monkeypatch.setattr(config, "_load_raw_config", lambda: _gate_config([repo]))
        monkeypatch.setattr(gate, "_resolve_skill_manage_path", lambda args: str(skill_dir / "SKILL.md"))
        _mark_canary(CANARY_KEY)
        assert resolve_pre_tool_block(
            "skill_manage", {"action": action, "name": "my-skill"}, session_id=CANARY_KEY) is not None

    def test_skill_outside_repo_roots_is_not_gated(self, tmp_path, repo, monkeypatch):
        outside_skill = tmp_path / "hermes-skills" / "my-skill" / "SKILL.md"
        outside_skill.parent.mkdir(parents=True)
        outside_skill.write_text("# skill\n", encoding="utf-8")
        monkeypatch.setattr(config, "_load_raw_config", lambda: _gate_config([repo]))
        monkeypatch.setattr(gate, "_resolve_skill_manage_path", lambda args: str(outside_skill))
        _mark_canary(CANARY_KEY)
        assert resolve_pre_tool_block(
            "skill_manage", {"action": "edit", "name": "my-skill"}, session_id=CANARY_KEY) is None

    def test_unknown_non_mutating_action_is_not_gated(self, repo, monkeypatch):
        monkeypatch.setattr(config, "_load_raw_config", lambda: _gate_config([repo]))
        _mark_canary(CANARY_KEY)
        assert resolve_pre_tool_block(
            "skill_manage", {"action": "view", "name": "my-skill"}, session_id=CANARY_KEY) is None

    def test_resolution_error_fails_closed(self, repo, monkeypatch):
        monkeypatch.setattr(config, "_load_raw_config", lambda: _gate_config([repo]))

        def _boom(args):
            raise RuntimeError("skill dir resolution exploded")

        monkeypatch.setattr(gate, "_resolve_skill_manage_path", _boom)
        _mark_canary(CANARY_KEY)
        assert resolve_pre_tool_block(
            "skill_manage", {"action": "write_file", "name": "my-skill"}, session_id=CANARY_KEY) is not None

    def test_missing_skill_name_fails_closed(self, repo, monkeypatch):
        monkeypatch.setattr(config, "_load_raw_config", lambda: _gate_config([repo]))
        _mark_canary(CANARY_KEY)
        assert resolve_pre_tool_block(
            "skill_manage", {"action": "patch"}, session_id=CANARY_KEY) is not None

    def test_unresolvable_path_fails_closed(self, repo, monkeypatch):
        monkeypatch.setattr(config, "_load_raw_config", lambda: _gate_config([repo]))
        monkeypatch.setattr(gate, "_resolve_skill_manage_path", lambda args: None)
        _mark_canary(CANARY_KEY)
        assert resolve_pre_tool_block(
            "skill_manage", {"action": "delete", "name": "my-skill"}, session_id=CANARY_KEY) is not None


class TestFailClosedWithinConfirmedCanary:
    def test_config_load_error_fails_closed(self, repo, monkeypatch):
        def _boom():
            raise RuntimeError("config backend unavailable")

        monkeypatch.setattr(config, "_load_raw_config", _boom)
        _mark_canary(CANARY_KEY)
        # Defaults have no repo roots, so a config error must not silently
        # become "nothing is in scope" — it blocks.
        monkeypatch.setattr(policy, "canonical_repo_roots", lambda cfg: (_ for _ in ()).throw(RuntimeError("x")))
        assert _patch(repo / "app.py") is not None

    def test_canary_lookup_error_fails_closed_not_open(self, repo, monkeypatch):
        # Cache-loss remediation: an internal error resolving eligibility is
        # UNKNOWN, not "not canary" — a gated write inside a configured root
        # must block, not sail through.
        monkeypatch.setattr(config, "_load_raw_config", lambda: _gate_config([repo]))
        monkeypatch.setattr(
            canary, "current_session_eligibility_state",
            lambda: (_ for _ in ()).throw(RuntimeError("x")),
        )
        assert _patch(repo / "app.py") is not None


def _mark_thread_unknown(session_key: str, thread_id: str = "sess-thread-id") -> None:
    """A Discord thread session with no recorded pre_gateway_dispatch
    decision for this key — UNKNOWN, the cache-loss scenario."""
    _VAR_MAP["HERMES_SESSION_PLATFORM"].set("discord")
    _VAR_MAP["HERMES_SESSION_CHAT_ID"].set(thread_id)
    _VAR_MAP["HERMES_SESSION_THREAD_ID"].set(thread_id)
    _VAR_MAP["HERMES_SESSION_KEY"].set(session_key)


class TestUnknownEligibilityCacheLoss:
    """The cache-loss finding, exercised through the real enforcement path:
    an UNKNOWN eligibility state (Discord thread, cache miss/reset/restart,
    or a gateway session-key failure at dispatch time) must BLOCK a gated
    write inside a configured root — never silently allow it through — while
    still leaving every irrelevant tool call and every out-of-scope path
    completely unaffected."""

    UNKNOWN_KEY = "sess:gate-thread-unknown"

    def test_thread_cache_miss_blocks_gated_write_in_configured_root(self, repo, monkeypatch):
        monkeypatch.setattr(config, "_load_raw_config", lambda: _gate_config([repo]))
        _mark_thread_unknown(self.UNKNOWN_KEY)
        message = _patch(repo / "app.py", session_id=self.UNKNOWN_KEY)
        assert message is not None

    def test_cache_reset_restart_blocks_previously_recorded_thread(self, repo, monkeypatch):
        monkeypatch.setattr(config, "_load_raw_config", lambda: _gate_config([repo]))
        canary._record(self.UNKNOWN_KEY, _canary_identity(True))
        _VAR_MAP["HERMES_SESSION_KEY"].set(self.UNKNOWN_KEY)
        assert _patch(repo / "app.py", session_id=self.UNKNOWN_KEY) is not None

        canary.reset_cache()
        _mark_thread_unknown(self.UNKNOWN_KEY)
        assert _patch(repo / "app.py", session_id=self.UNKNOWN_KEY) is not None

    def test_gateway_session_key_failure_blocks_current_discord_thread(self, repo, monkeypatch):
        monkeypatch.setattr(config, "_load_raw_config", lambda: _gate_config([repo]))

        class _BrokenGateway:
            def _session_key_for_source(self, source):
                raise RuntimeError("gateway session-key resolution exploded")

        class _P:
            def __init__(self, value):
                self.value = value

        class _Source:
            platform = _P("discord")
            chat_id = "thread-under-canary-parent"
            parent_chat_id = "1527706694665113670"

        class _Event:
            source = _Source()

        canary.compute_and_cache_eligibility(_Event(), _BrokenGateway(), cfg=None)

        _mark_thread_unknown(self.UNKNOWN_KEY, thread_id="thread-under-canary-parent")
        assert _patch(repo / "app.py", session_id=self.UNKNOWN_KEY) is not None

    def test_no_eviction_after_over_4096_recorded_sessions(self, repo, monkeypatch):
        monkeypatch.setattr(config, "_load_raw_config", lambda: _gate_config([repo]))
        for i in range(4200):
            canary._record(f"sess:gate-bulk:{i}", _canary_identity(False))
        canary._record(self.UNKNOWN_KEY, _canary_identity(True))
        assert canary._identity_cache.get("sess:gate-bulk:0") == _canary_identity(False)
        _VAR_MAP["HERMES_SESSION_KEY"].set(self.UNKNOWN_KEY)
        assert _patch(repo / "app.py", session_id=self.UNKNOWN_KEY) is not None

    def test_non_discord_platform_remains_allowed_without_cache(self, repo, monkeypatch):
        monkeypatch.setattr(config, "_load_raw_config", lambda: _gate_config([repo]))
        session_key = "sess:gate-non-discord"
        _VAR_MAP["HERMES_SESSION_PLATFORM"].set("telegram")
        _VAR_MAP["HERMES_SESSION_CHAT_ID"].set("1527706694665113670")
        _VAR_MAP["HERMES_SESSION_KEY"].set(session_key)
        assert _patch(repo / "app.py", session_id=session_key) is None

    def test_known_direct_non_canary_discord_remains_allowed(self, repo, monkeypatch):
        monkeypatch.setattr(config, "_load_raw_config", lambda: _gate_config([repo]))
        session_key = "sess:gate-direct-non-canary"
        _VAR_MAP["HERMES_SESSION_PLATFORM"].set("discord")
        _VAR_MAP["HERMES_SESSION_CHAT_ID"].set("000000000000000000")
        _VAR_MAP["HERMES_SESSION_THREAD_ID"].set("")
        _VAR_MAP["HERMES_SESSION_KEY"].set(session_key)
        assert _patch(repo / "app.py", session_id=session_key) is None

    def test_irrelevant_tool_allowed_even_when_eligibility_unknown(self, repo, monkeypatch):
        monkeypatch.setattr(config, "_load_raw_config", lambda: _gate_config([repo]))
        _mark_thread_unknown(self.UNKNOWN_KEY)
        assert resolve_pre_tool_block(
            "read_file", {"path": str(repo / "app.py")}, session_id=self.UNKNOWN_KEY) is None

    def test_path_outside_every_root_allowed_even_when_eligibility_unknown(self, tmp_path, repo, monkeypatch):
        scratch = tmp_path / "scratch"
        scratch.mkdir()
        (scratch / "notes.txt").write_text("hello\n", encoding="utf-8")
        monkeypatch.setattr(config, "_load_raw_config", lambda: _gate_config([repo]))
        _mark_thread_unknown(self.UNKNOWN_KEY)
        assert resolve_pre_tool_block(
            "write_file", {"path": str(scratch / "notes.txt"), "content": "hi"},
            session_id=self.UNKNOWN_KEY,
        ) is None
