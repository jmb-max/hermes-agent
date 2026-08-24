"""Tests for ``plugins/claude-worker/gate.py`` — the write gate and the
terminal bypass.

Asserted through the REAL enforcement path
(``hermes_cli.plugins.resolve_pre_tool_block``) so these prove the actual
tool-dispatch sites block, not merely that a callback returns a dict.

Post-remediation invariants:

* scope is EVERY Discord-origin session — an arbitrary channel, a DM, and a
  thread are all gated identically, and no configuration can narrow that to
  a channel list;
* project scope is the DYNAMIC canonical Git worktree root of the request
  (``project.py``), shared with the runner, so the gate can never block a
  repository the worker is not allowed to run in;
* a path outside every Git worktree is never gated, even in a confirmed
  Discord session — gating it would deadlock the session with nothing able
  to satisfy the gate;
* the gated tool set (patch/write_file/skill_manage) and the gate itself are
  immutable — config cannot relax them;
* a SUCCESSFUL WORKER RUN RELEASES NOTHING. The worker is the coder for the
  whole session, not a one-time toll gate; the only release is an explicit
  Terra fallback that actually returned a result, and an OPEN BREAKER never
  releases anything either;
* a direct host Claude CLI invocation through ``terminal`` is refused
  outright, release state or not, while ordinary build/test/git commands are
  untouched;
* UNKNOWN eligibility fails CLOSED for anything that is actually in scope,
  and never manufactures a block for anything that is not;
* the EXPLICIT ``session_id`` passed to ``pre_tool_call`` is authoritative
  Discord provenance, so a continuation whose ambient session ContextVars
  were cleared cannot bypass either the write gate or the terminal guard;
* and — because the real dispatch sites pass the TRANSCRIPT session id under
  ``session_id`` — the stable per-chat ``gateway_session_key`` passed
  alongside it is what Discord provenance and the Terra-fallback release are
  actually resolved from (see the last two classes in this file).
"""

from __future__ import annotations

import json
import os

import pytest

import hermes_cli.plugins as hermes_plugins
from hermes_cli.plugins import (
    PluginContext,
    PluginManager,
    PluginManifest,
    invoke_hook,
    resolve_pre_tool_block,
)
from gateway.session_context import _UNSET, _VAR_MAP

from tests.plugins._claude_worker_helpers import (
    load_submodule,
    make_git_repo,
    require_safe_tmp,
)

canary = load_submodule("canary")
gate = load_submodule("gate")
breaker = load_submodule("breaker")
config = load_submodule("config")
policy = load_submodule("policy")
project = load_submodule("project")

DISCORD_KEY = "sess:gate-discord"
NON_DISCORD_KEY = "sess:gate-non-discord"

#: Deliberately arbitrary — no id is special to the plugin any more.
ARBITRARY_CHANNEL = "204871330045198336"
ANOTHER_CHANNEL = "999999999999999999"
THREAD_ID = "111222333444555666"


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    require_safe_tmp(tmp_path)

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))

    # Pin config loading to the plugin's own defaults so "the feature is on"
    # is an assertion rather than an accident of the host's config.yaml.
    monkeypatch.setattr(config, "_load_raw_config", lambda: {})

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


def _identity(platform="discord", chat_id=ARBITRARY_CHANNEL, parent="", thread=""):
    return {
        "platform": platform, "chat_id": chat_id,
        "parent_chat_id": parent, "thread_id": thread,
    }


def _mark_discord(session_key=DISCORD_KEY, chat_id=ARBITRARY_CHANNEL, parent="", thread=""):
    canary._record(session_key, _identity("discord", chat_id, parent, thread))
    _VAR_MAP["HERMES_SESSION_KEY"].set(session_key)


def _mark_platform(session_key, platform):
    canary._record(session_key, _identity(platform))
    _VAR_MAP["HERMES_SESSION_KEY"].set(session_key)


def _mark_unknown(session_key, thread_id=THREAD_ID):
    """A real chat session whose platform nothing can confirm — no recorded
    dispatch identity, no platform contextvar, no platform in the key."""
    canary._identity_cache.pop(session_key, None)
    _VAR_MAP["HERMES_SESSION_CHAT_ID"].set(thread_id)
    _VAR_MAP["HERMES_SESSION_THREAD_ID"].set(thread_id)
    _VAR_MAP["HERMES_SESSION_KEY"].set(session_key)


def _raw_config(**entry):
    return {"plugins": {"entries": {"claude-worker": entry}}}


@pytest.fixture()
def repo(tmp_path):
    path = make_git_repo(tmp_path, "repo")
    (path / "app.py").write_text("x = 1\n", encoding="utf-8")
    return path


def _patch(target, session_id=DISCORD_KEY):
    return resolve_pre_tool_block(
        "patch", {"path": str(target), "old_string": "x = 1", "new_string": "x = 2"},
        session_id=session_id,
    )


def _terminal(command, session_id=DISCORD_KEY):
    return resolve_pre_tool_block("terminal", {"command": command}, session_id=session_id)


# ---------------------------------------------------------------------------
# A. Global Discord scope
# ---------------------------------------------------------------------------


class TestEveryDiscordSessionIsGated:
    @pytest.mark.parametrize(
        "chat_id",
        [ARBITRARY_CHANNEL, ANOTHER_CHANNEL, "42", "1527706694665113670", "dm:8241"],
    )
    def test_an_arbitrary_discord_channel_is_gated(self, repo, chat_id):
        """No allowlist to be on: any channel id gates identically."""
        session = f"sess:{chat_id}"
        _mark_discord(session, chat_id=chat_id)
        message = _patch(repo / "app.py", session_id=session)
        assert message is not None
        assert "claude_worker" in message

    def test_a_discord_thread_is_gated(self, repo):
        """A thread's ``chat_id`` is the THREAD id and ``parent_chat_id`` is
        not a tracked contextvar — which is exactly how a channel-list check
        missed every thread."""
        _mark_discord(DISCORD_KEY, chat_id=THREAD_ID, parent=ANOTHER_CHANNEL, thread=THREAD_ID)
        assert _patch(repo / "app.py") is not None

    def test_a_thread_with_no_recorded_parent_is_still_gated(self, repo):
        _mark_discord(DISCORD_KEY, chat_id=THREAD_ID, parent="", thread=THREAD_ID)
        assert _patch(repo / "app.py") is not None

    def test_a_discord_thread_gates_from_contextvars_alone(self, repo):
        """No dispatch identity at all — the platform contextvar is enough
        now that scope is platform-wide."""
        canary._identity_cache.clear()
        _VAR_MAP["HERMES_SESSION_PLATFORM"].set("discord")
        _VAR_MAP["HERMES_SESSION_CHAT_ID"].set(THREAD_ID)
        _VAR_MAP["HERMES_SESSION_THREAD_ID"].set(THREAD_ID)
        _VAR_MAP["HERMES_SESSION_KEY"].set(DISCORD_KEY)
        assert _patch(repo / "app.py") is not None

    def test_a_discord_session_gates_from_the_session_key_platform_alone(self, repo):
        canary._identity_cache.clear()
        key = f"agent:main:discord:{ARBITRARY_CHANNEL}"
        _VAR_MAP["HERMES_SESSION_CHAT_ID"].set(ARBITRARY_CHANNEL)
        _VAR_MAP["HERMES_SESSION_KEY"].set(key)
        assert _patch(repo / "app.py", session_id=key) is not None

    def test_write_file_is_gated_too(self, repo):
        _mark_discord()
        assert resolve_pre_tool_block(
            "write_file", {"path": str(repo / "new.py"), "content": "print(1)"},
            session_id=DISCORD_KEY,
        ) is not None

    def test_all_three_mandatory_tools_are_gated(self, repo, monkeypatch):
        skill_dir = repo / "skills" / "s"
        skill_dir.mkdir(parents=True)
        monkeypatch.setattr(
            gate, "_resolve_skill_manage_path", lambda args: str(skill_dir / "SKILL.md"),
        )
        _mark_discord()
        assert _patch(repo / "app.py") is not None
        assert resolve_pre_tool_block(
            "write_file", {"path": str(repo / "b.py"), "content": "x"}, session_id=DISCORD_KEY,
        ) is not None
        assert resolve_pre_tool_block(
            "skill_manage", {"action": "write_file", "name": "s", "file_content": "x"},
            session_id=DISCORD_KEY,
        ) is not None


class TestNonDiscordIsNeverGated:
    @pytest.mark.parametrize("platform", ["telegram", "slack", "whatsapp", "matrix", "cli"])
    def test_other_platforms_pass_through(self, repo, platform):
        session = f"sess:{platform}"
        _mark_platform(session, platform)
        assert _patch(repo / "app.py", session_id=session) is None

    def test_non_discord_platform_from_contextvars_passes_through(self, repo):
        canary._identity_cache.clear()
        _VAR_MAP["HERMES_SESSION_PLATFORM"].set("telegram")
        _VAR_MAP["HERMES_SESSION_CHAT_ID"].set(ARBITRARY_CHANNEL)
        _VAR_MAP["HERMES_SESSION_KEY"].set(NON_DISCORD_KEY)
        assert _patch(repo / "app.py", session_id=NON_DISCORD_KEY) is None

    def test_a_cli_session_with_no_chat_identity_passes_through(self, repo):
        canary._identity_cache.clear()
        _VAR_MAP["HERMES_SESSION_KEY"].set("agent:main:local:cli")
        assert _patch(repo / "app.py", session_id="agent:main:local:cli") is None

    def test_a_non_discord_session_may_run_the_claude_cli_directly(self, repo):
        """The terminal guard is scoped to Discord-origin sessions too — it
        must not become a global restriction on the host."""
        _mark_platform(NON_DISCORD_KEY, "telegram")
        assert _terminal("claude -p 'fix it'", session_id=NON_DISCORD_KEY) is None


class TestGlobalKillSwitch:
    def test_disabling_the_feature_ungates_every_discord_session(self, repo, monkeypatch):
        monkeypatch.setattr(
            config, "_load_raw_config", lambda: _raw_config(discord={"enabled": False}),
        )
        _mark_discord()
        assert _patch(repo / "app.py") is None

    def test_the_deprecated_canary_toggle_still_disables(self, repo, monkeypatch):
        monkeypatch.setattr(
            config, "_load_raw_config", lambda: _raw_config(canary={"enabled": False}),
        )
        _mark_discord()
        assert _patch(repo / "app.py") is None

    @pytest.mark.parametrize(
        "entry",
        [
            {"canary": {"enabled": True, "channel_ids": [ANOTHER_CHANNEL]}},
            {"canary": {"enabled": True, "channel_ids": []}},
            {"discord": {"enabled": True, "channel_ids": [ANOTHER_CHANNEL]}},
            {"gate": {"repo_roots": []}},
        ],
    )
    def test_config_cannot_exempt_one_channel(self, repo, monkeypatch, entry):
        """The stale narrowing keys are inert: a session whose channel is
        absent from every list is gated exactly like one that is present."""
        monkeypatch.setattr(config, "_load_raw_config", lambda: _raw_config(**entry))
        _mark_discord(DISCORD_KEY, chat_id=ARBITRARY_CHANNEL)
        assert _patch(repo / "app.py") is not None


class TestGateIsImmutable:
    @pytest.mark.parametrize("relaxation", [
        {"gate": {"enabled": False}},
        {"gate": {"tools": []}},
        {"gate": {"tools": ["read_file"]}},
        {"gate": {"repo_roots": ["/nowhere"]}},
    ])
    def test_config_cannot_disable_or_narrow_the_gate(self, repo, monkeypatch, relaxation):
        monkeypatch.setattr(config, "_load_raw_config", lambda: _raw_config(**relaxation))
        _mark_discord()
        assert _patch(repo / "app.py") is not None


# ---------------------------------------------------------------------------
# B. Dynamic project scope, shared with the runner
# ---------------------------------------------------------------------------


class TestDynamicProjectScope:
    def test_a_repo_that_was_never_configured_anywhere_is_gated(self, tmp_path):
        """The old failure mode inverted: with a static allowlist a brand-new
        checkout was silently ungated. Scope comes from the request now."""
        fresh = make_git_repo(tmp_path, "brand-new-checkout")
        (fresh / "app.py").write_text("x = 1\n", encoding="utf-8")
        _mark_discord()
        assert _patch(fresh / "app.py") is not None

    def test_a_nested_path_is_gated_against_the_repo_root(self, repo):
        nested = repo / "src" / "pkg"
        nested.mkdir(parents=True)
        (nested / "mod.py").write_text("x = 1\n", encoding="utf-8")
        _mark_discord()
        message = _patch(nested / "mod.py")
        assert message is not None
        assert os.path.realpath(str(repo)) in message

    def test_a_dot_git_FILE_linked_worktree_is_gated(self, tmp_path):
        """A linked worktree/submodule has a ``.git`` FILE, not a directory —
        the shape a naive ``isdir('.git')`` check misses entirely."""
        linked = make_git_repo(tmp_path, "linked", marker="file")
        (linked / "app.py").write_text("x = 1\n", encoding="utf-8")
        _mark_discord()
        message = _patch(linked / "app.py")
        assert message is not None
        assert os.path.realpath(str(linked)) in message

    def test_a_new_file_in_a_new_subdirectory_is_gated(self, repo):
        _mark_discord()
        assert resolve_pre_tool_block(
            "write_file", {"path": str(repo / "brand" / "new" / "f.py"), "content": "x"},
            session_id=DISCORD_KEY,
        ) is not None

    def test_the_block_message_names_the_resolved_root(self, repo):
        _mark_discord()
        message = _patch(repo / "app.py")
        assert os.path.realpath(str(repo)) in message

    def test_a_sibling_repo_is_a_separate_scope(self, tmp_path, repo):
        """``/repo-evil`` is not inside ``/repo``: releasing one must not
        release the other. Asserted through the ONLY thing that still
        releases a repository — an explicit Terra fallback result."""
        evil = make_git_repo(tmp_path, "repo-evil")
        (evil / "app.py").write_text("x = 1\n", encoding="utf-8")
        _mark_discord()
        _record_worker_result(_fallback_payload(repo))
        assert _patch(repo / "app.py") is None
        assert _patch(evil / "app.py") is not None


class TestOutsideGitNeverDeadlocks:
    """A path outside every Git worktree is never gated, even for a CONFIRMED
    Discord session: the worker could not run there either, so gating it
    would leave the session with nothing able to satisfy the gate."""

    def test_a_scratch_directory_is_not_gated(self, tmp_path):
        scratch = tmp_path / "scratch"
        scratch.mkdir()
        (scratch / "notes.txt").write_text("hello\n", encoding="utf-8")
        _mark_discord()
        assert resolve_pre_tool_block(
            "write_file", {"path": str(scratch / "notes.txt"), "content": "hi"},
            session_id=DISCORD_KEY,
        ) is None

    def test_a_bare_repository_is_not_gated(self, tmp_path):
        bare = tmp_path / "project.git"
        (bare / "objects").mkdir(parents=True)
        (bare / "refs").mkdir()
        (bare / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
        _mark_discord()
        assert resolve_pre_tool_block(
            "write_file", {"path": str(bare / "hooks" / "x"), "content": "hi"},
            session_id=DISCORD_KEY,
        ) is None

    def test_a_symlink_escaping_a_repo_is_judged_where_it_lands(self, tmp_path, repo):
        outside = tmp_path / "outside"
        outside.mkdir()
        (repo / "escape").symlink_to(outside, target_is_directory=True)
        _mark_discord()
        assert resolve_pre_tool_block(
            "write_file", {"path": str(repo / "escape" / "f.py"), "content": "x"},
            session_id=DISCORD_KEY,
        ) is None

    def test_an_unsafe_resolved_root_is_not_gated(self, monkeypatch, repo):
        """A root the worker may never mount must not be gated either — the
        two rules are the same rule."""
        monkeypatch.setattr(
            policy, "UNSAFE_PROJECT_ROOTS",
            frozenset(policy.UNSAFE_PROJECT_ROOTS | {os.path.realpath(str(repo))}),
        )
        _mark_discord()
        assert _patch(repo / "app.py") is None


class TestGateAndRunnerShareOneRootResolution:
    def test_both_resolve_the_same_root_for_the_same_request(self, repo):
        runner = load_submodule("runner")
        nested = repo / "src"
        nested.mkdir()
        (nested / "a.py").write_text("x = 1\n", encoding="utf-8")

        expected = os.path.realpath(str(repo))
        # What the runner would validate and MOUNT...
        assert runner.validate_cwd(str(nested)) == os.path.realpath(str(nested))
        assert project.resolve_project_root_strict(str(nested)) == expected
        # ...is exactly what the gate scopes the unlock to.
        assert project.resolve_project_root_for_target(str(nested / "a.py")) == expected

    def test_a_cwd_the_runner_refuses_is_a_cwd_the_gate_does_not_gate(self, tmp_path):
        runner = load_submodule("runner")
        scratch = tmp_path / "scratch"
        scratch.mkdir()

        with pytest.raises(runner.CwdRejected):
            runner.validate_cwd(str(scratch))

        _mark_discord()
        assert resolve_pre_tool_block(
            "write_file", {"path": str(scratch / "f.py"), "content": "x"},
            session_id=DISCORD_KEY,
        ) is None

    def test_the_gate_never_reimplements_git_detection(self):
        """A second ``.git`` check in the gate is how the two scopes drifted
        apart and deadlocked a session in the first place."""
        source = open(gate.__file__, encoding="utf-8").read()
        assert '".git"' not in source
        assert "_project.resolve_project_root_for_target(" in source
        assert "_project.resolve_project_root(" in source


# ---------------------------------------------------------------------------
# C. Unlocking
# ---------------------------------------------------------------------------


def _record_worker_result(payload, session_id=DISCORD_KEY, status="ok"):
    invoke_hook(
        "post_tool_call", tool_name="claude_worker",
        args={"task": "fix it", "cwd": payload.get("cwd", "")},
        result=json.dumps(payload), session_id=session_id, status=status,
    )


def _fallback_payload(repo):
    """The one payload shape that still releases a repository: an explicit
    Terra fallback that actually came back with a result."""
    return {
        "success": False, "status": "fallback_ready", "fallback_ready": True,
        "fallback_requested": True, "fallback_provenance": "terra_auxiliary",
        "cwd": str(repo),
        "fallback": {"ok": True, "ready": True, "notes": "Terra guidance"},
    }


class TestASuccessfulRunNeverReleasesDirectEdits:
    """Policy change, from the live incident.

    A successful ``claude_worker`` run used to unlock ``(session, repo)`` for
    the rest of the session, so the worker was a ONE-TIME toll gate: run it
    once, then edit the repository directly forever. The operator expectation
    is the opposite — in a Discord session the worker is the coder, every
    time. A successful run now proves only that the worker works; it grants
    nothing.
    """

    def test_a_successful_run_does_not_release_patch(self, repo):
        _mark_discord()
        assert _patch(repo / "app.py") is not None
        _record_worker_result({"success": True, "cwd": str(repo)})
        assert _patch(repo / "app.py") is not None

    def test_a_successful_run_does_not_release_write_file(self, repo):
        _mark_discord()
        _record_worker_result({"success": True, "cwd": str(repo)})
        assert resolve_pre_tool_block(
            "write_file", {"path": str(repo / "b.py"), "content": "x"}, session_id=DISCORD_KEY,
        ) is not None

    def test_a_successful_run_does_not_release_skill_manage(self, repo, monkeypatch):
        skill_dir = repo / "skills" / "s"
        skill_dir.mkdir(parents=True)
        monkeypatch.setattr(
            gate, "_resolve_skill_manage_path", lambda args: str(skill_dir / "SKILL.md"),
        )
        _mark_discord()
        _record_worker_result({"success": True, "cwd": str(repo)})
        assert resolve_pre_tool_block(
            "skill_manage", {"action": "patch", "name": "s"}, session_id=DISCORD_KEY,
        ) is not None

    def test_a_run_from_a_nested_cwd_releases_nothing_either(self, repo):
        nested = repo / "src"
        nested.mkdir()
        _mark_discord()
        _record_worker_result({"success": True, "cwd": str(nested)})
        assert _patch(repo / "app.py") is not None

    def test_many_successful_runs_never_accumulate_into_a_release(self, repo):
        _mark_discord()
        for _ in range(5):
            _record_worker_result({"success": True, "cwd": str(repo)})
        assert _patch(repo / "app.py") is not None

    def test_the_block_message_does_not_promise_a_release_after_a_run(self, repo):
        """The old message told the session to call the worker once and then
        edit freely — an instruction the gate must no longer be able to give."""
        _mark_discord()
        _record_worker_result({"success": True, "cwd": str(repo)})
        message = _patch(repo / "app.py")
        assert "claude_worker" in message
        lowered = message.lower()
        assert "unblocked for the rest of the session" not in lowered
        assert "until a claude_worker run has completed" not in lowered

    def test_a_successful_run_does_not_release_the_terminal_guard(self, repo):
        _mark_discord()
        _record_worker_result({"success": True, "cwd": str(repo)})
        assert _terminal("claude -p x") is not None

    def test_the_gate_records_no_success_unlock_state_at_all(self, repo):
        """Structural: there is no "successful run" unlock set left to drift
        back into being consulted."""
        _mark_discord()
        _record_worker_result({"success": True, "cwd": str(repo)})
        assert not hasattr(gate, "_successful_runs")

    def test_an_explicit_terra_fallback_still_releases(self, repo):
        """The contingency is unchanged: when the worker cannot run at all,
        an explicit Terra fallback that returned a result still releases the
        repository, or the session would have no way forward."""
        _mark_discord()
        _record_worker_result(_fallback_payload(repo))
        assert _patch(repo / "app.py") is None


class TestReleaseRecordingGuards:
    """``on_post_tool_call``'s guards, asserted on the ONE payload shape that
    can still release a repository.

    The first three are the original success-payload controls. They now hold
    for two independent reasons — the payload does not release anything at all
    any more, AND it is scoped/keyed wrongly — and are kept as regression
    guards on the first of those. The rest exercise the guards themselves,
    using a fallback payload that WOULD release if the guard were missing.
    """

    def test_a_failed_run_does_not_release_the_gate(self, repo):
        _mark_discord()
        _record_worker_result({"success": False, "cwd": str(repo)}, status="error")
        assert _patch(repo / "app.py") is not None

    def test_success_in_a_different_repo_does_not_release_this_one(self, tmp_path, repo):
        other = make_git_repo(tmp_path, "repo_b")
        _mark_discord()
        _record_worker_result({"success": True, "cwd": str(other)})
        assert _patch(repo / "app.py") is not None

    def test_success_in_a_different_session_does_not_release_this_one(self, repo):
        canary._record("sess:other-discord", _identity())
        _mark_discord()
        _record_worker_result({"success": True, "cwd": str(repo)}, session_id="sess:other-discord")
        _VAR_MAP["HERMES_SESSION_KEY"].set(DISCORD_KEY)
        assert _patch(repo / "app.py") is not None

    def test_a_fallback_outside_any_git_worktree_releases_nothing(self, tmp_path, repo):
        outside = tmp_path / "outside"
        outside.mkdir()
        _mark_discord()
        _record_worker_result(_fallback_payload(outside))
        assert _patch(repo / "app.py") is not None

    def test_a_missing_cwd_in_the_payload_releases_nothing(self, repo):
        payload = _fallback_payload(repo)
        del payload["cwd"]
        _mark_discord()
        _record_worker_result(payload)
        assert _patch(repo / "app.py") is not None

    def test_an_empty_session_id_releases_nothing(self, repo):
        """The release is keyed on a session. With no session key on either
        the explicit argument or the ambient contextvar there is nothing to
        key it to, and a global release would be exactly wrong."""
        _mark_discord()
        # No explicit key AND no ambient key to fall back to.
        _VAR_MAP["HERMES_SESSION_KEY"].set("")
        _record_worker_result(_fallback_payload(repo), session_id="")
        _VAR_MAP["HERMES_SESSION_KEY"].set(DISCORD_KEY)
        assert _patch(repo / "app.py") is not None

    def test_another_tools_result_can_never_release(self, repo):
        _mark_discord()
        invoke_hook(
            "post_tool_call", tool_name="write_file",
            args={"path": str(repo / "x.py")},
            result=json.dumps(_fallback_payload(repo)),
            session_id=DISCORD_KEY, status="ok",
        )
        assert _patch(repo / "app.py") is not None

    def test_an_unparseable_result_releases_nothing(self, repo):
        _mark_discord()
        invoke_hook(
            "post_tool_call", tool_name="claude_worker",
            args={"task": "fix it", "cwd": str(repo)},
            result="not json at all", session_id=DISCORD_KEY, status="ok",
        )
        assert _patch(repo / "app.py") is not None


class TestBreakerNeverUnlocks:
    @pytest.mark.parametrize("failure_class", ["auth", "rate", "extra_usage"])
    def test_open_breaker_does_not_unlock_patch(self, repo, failure_class):
        _mark_discord()
        breaker.record_failure(failure_class, {failure_class: 3600})
        message = _patch(repo / "app.py")
        assert message is not None
        assert "claude_worker" in message

    def test_open_breaker_does_not_unlock_write_file(self, repo):
        _mark_discord()
        breaker.record_failure("auth", {"auth": 3600})
        assert resolve_pre_tool_block(
            "write_file", {"path": str(repo / "b.py"), "content": "x"}, session_id=DISCORD_KEY,
        ) is not None

    def test_open_breaker_does_not_unlock_skill_manage(self, repo, monkeypatch):
        skill_dir = repo / "skills" / "s"
        skill_dir.mkdir(parents=True)
        monkeypatch.setattr(
            gate, "_resolve_skill_manage_path", lambda args: str(skill_dir / "SKILL.md"),
        )
        _mark_discord()
        breaker.record_failure("rate", {"rate": 900})
        assert resolve_pre_tool_block(
            "skill_manage", {"action": "patch", "name": "s"}, session_id=DISCORD_KEY,
        ) is not None

    def test_the_breaker_message_explains_the_hold(self, repo):
        _mark_discord()
        breaker.record_failure("auth", {"auth": 3600})
        message = _patch(repo / "app.py")
        assert "unavailable" in message.lower() or "hold" in message.lower()

    def test_an_open_breaker_does_not_relax_the_terminal_bypass(self, repo):
        _mark_discord()
        breaker.record_failure("auth", {"auth": 3600})
        assert _terminal("claude -p 'just this once'") is not None


class TestExplicitTerraFallbackUnlock:
    def _valid_payload(self, repo):
        return {
            "success": False, "status": "fallback_ready", "fallback_ready": True,
            "fallback_requested": True, "fallback_provenance": "terra_auxiliary",
            "cwd": str(repo),
            "fallback": {"ok": True, "ready": True, "notes": "Terra guidance"},
        }

    def test_fallback_ready_marker_unlocks_the_gate(self, repo):
        _mark_discord()
        breaker.record_failure("auth", {"auth": 3600})
        assert _patch(repo / "app.py") is not None
        _record_worker_result(self._valid_payload(repo))
        assert _patch(repo / "app.py") is None

    def test_hold_status_keeps_the_gate_blocked(self, repo):
        _mark_discord()
        breaker.record_failure("auth", {"auth": 3600})
        _record_worker_result({
            "success": False, "status": "HOLD", "fallback_ready": False,
            "cwd": str(repo), "fallback": {"ok": False, "error": "terra unavailable"},
        })
        assert _patch(repo / "app.py") is not None

    def test_fallback_marker_is_scoped_to_its_repo(self, tmp_path, repo):
        other = make_git_repo(tmp_path, "repo_b")
        _mark_discord()
        _record_worker_result(self._valid_payload(other))
        assert _patch(repo / "app.py") is not None

    def test_fallback_marker_is_scoped_to_its_session(self, repo):
        canary._record("sess:other-discord", _identity())
        _mark_discord()
        _record_worker_result(self._valid_payload(repo), session_id="sess:other-discord")
        _VAR_MAP["HERMES_SESSION_KEY"].set(DISCORD_KEY)
        assert _patch(repo / "app.py") is not None

    def test_a_fallback_unlock_does_not_relax_the_terminal_bypass(self, repo):
        _mark_discord()
        _record_worker_result(self._valid_payload(repo))
        assert _patch(repo / "app.py") is None
        assert _terminal("claude -p x") is not None


class TestForgedFallbackProvenanceCannotUnlock:
    """A payload that gets every OTHER check right but forges/omits one
    provenance field must still stay HOLD — ``_fallback_delivered`` requires
    each of these independently, not just ``fallback_ready``."""

    def _valid_payload(self, repo):
        return {
            "success": False, "status": "fallback_ready", "fallback_ready": True,
            "fallback_requested": True, "fallback_provenance": "terra_auxiliary",
            "cwd": str(repo),
            "fallback": {"ok": True, "ready": True, "notes": "Terra guidance"},
        }

    def _assert_still_blocked(self, repo, payload):
        _mark_discord()
        breaker.record_failure("auth", {"auth": 3600})
        _record_worker_result(payload)
        assert _patch(repo / "app.py") is not None

    def test_missing_fallback_requested_does_not_unlock(self, repo):
        payload = self._valid_payload(repo)
        del payload["fallback_requested"]
        self._assert_still_blocked(repo, payload)

    def test_string_true_fallback_requested_does_not_unlock(self, repo):
        payload = self._valid_payload(repo)
        payload["fallback_requested"] = "true"
        self._assert_still_blocked(repo, payload)

    def test_missing_fallback_provenance_does_not_unlock(self, repo):
        payload = self._valid_payload(repo)
        del payload["fallback_provenance"]
        self._assert_still_blocked(repo, payload)

    def test_wrong_fallback_provenance_does_not_unlock(self, repo):
        payload = self._valid_payload(repo)
        payload["fallback_provenance"] = "attacker_supplied"
        self._assert_still_blocked(repo, payload)

    def test_fallback_ready_truthy_but_not_literal_true_does_not_unlock(self, repo):
        payload = self._valid_payload(repo)
        payload["fallback_ready"] = 1
        self._assert_still_blocked(repo, payload)

    def test_missing_nested_ready_does_not_unlock(self, repo):
        payload = self._valid_payload(repo)
        del payload["fallback"]["ready"]
        self._assert_still_blocked(repo, payload)

    def test_nested_ready_false_does_not_unlock(self, repo):
        payload = self._valid_payload(repo)
        payload["fallback"]["ready"] = False
        self._assert_still_blocked(repo, payload)

    def test_nested_ready_truthy_but_not_literal_true_does_not_unlock(self, repo):
        payload = self._valid_payload(repo)
        payload["fallback"]["ready"] = 1
        self._assert_still_blocked(repo, payload)

    def test_empty_notes_does_not_unlock(self, repo):
        payload = self._valid_payload(repo)
        payload["fallback"]["notes"] = "   "
        self._assert_still_blocked(repo, payload)

    def test_fully_valid_payload_does_unlock_as_a_control(self, repo):
        """Sanity control proving the helper payload is otherwise correct —
        every "forged" test above changes exactly one field away from this
        unlocking baseline."""
        _mark_discord()
        breaker.record_failure("auth", {"auth": 3600})
        _record_worker_result(self._valid_payload(repo))
        assert _patch(repo / "app.py") is None


# ---------------------------------------------------------------------------
# D. The terminal bypass, through the real enforcement path
# ---------------------------------------------------------------------------


class TestTerminalBypassIsBlockedForDiscordSessions:
    @pytest.mark.parametrize(
        "command",
        [
            "claude",
            "claude -p 'edit the parser'",
            'claude --print --model claude-opus-5 "do it"',
            "/usr/local/bin/claude -p x",
            "./claude",
            "~/.local/bin/claude -p x",
            "env claude",
            "env FOO=1 claude -p x",
            "sudo claude",
            "sudo -n claude",
            "sudo -u root claude -p x",
            "command claude",
            "nohup claude -p x",
            "exec claude",
            "setsid claude",
            "nice -n 5 claude",
            "timeout 300 claude",
            "FOO=bar claude",
            "make build && claude -p x",
            "make build; claude",
            "make build || claude",
            "cat prompt.txt | claude -p -",
            "make build\nclaude -p x",
            "make build &&\nclaude -p x",
            'bash -c "claude -p x"',
            "sh -c 'claude'",
            "sudo bash -c 'claude -p x'",
            'bash -c "sh -c \'claude -p x\'"',
            'claude -p "unterminated',
        ],
    )
    def test_every_direct_host_invocation_is_refused(self, command):
        _mark_discord()
        message = _terminal(command)
        assert message is not None, command
        assert "claude_worker" in message

    def test_the_message_explains_the_sandbox_bypass(self):
        _mark_discord()
        message = _terminal("claude -p x")
        assert "sandbox" in message.lower()

    def test_it_is_refused_in_a_thread_too(self):
        _mark_discord(DISCORD_KEY, chat_id=THREAD_ID, parent=ANOTHER_CHANNEL, thread=THREAD_ID)
        assert _terminal("claude -p x") is not None

    def test_it_is_refused_regardless_of_any_repository(self, repo):
        """Releasing a repository for direct ``patch`` edits is a statement
        about that repository, never a licence to run an unsandboxed Claude
        on the host."""
        _mark_discord()
        _record_worker_result(_fallback_payload(repo))
        assert _patch(repo / "app.py") is None
        assert _terminal("claude -p x") is not None

    def test_it_is_refused_from_outside_every_repository(self, tmp_path):
        """The terminal guard is not repo-scoped: there is no cwd from which
        a host Claude CLI invocation is acceptable in a Discord session."""
        _mark_discord()
        assert _terminal(f"cd {tmp_path} && claude -p x") is not None

    def test_unknown_eligibility_also_refuses(self):
        _mark_unknown("sess:terminal-unknown")
        assert _terminal("claude -p x", session_id="sess:terminal-unknown") is not None

    def test_a_guard_error_fails_closed(self, monkeypatch):
        terminal_guard = load_submodule("terminal_guard")

        def _boom(command):
            raise RuntimeError("guard exploded")

        monkeypatch.setattr(terminal_guard, "evaluate", _boom)
        _mark_discord()
        assert _terminal("make build") is not None


class TestOrdinaryTerminalCommandsStillWork:
    @pytest.mark.parametrize(
        "command",
        [
            "make build",
            "make build && make test",
            "pytest -q",
            "pytest -q tests/plugins/test_claude_worker_gate.py",
            "npm ci && npm test",
            "cargo build --release",
            "git status --porcelain",
            'git commit -m "wire up claude_worker"',
            "git log --grep claude",
            "grep -r claude .",
            "ls /usr/bin/claude",
            "cat plugins/claude-worker/README.md",
            "echo claude",
            "which claude",
            "timeout 300 pytest -q",
            "env FOO=1 pytest -q",
            "bash -c 'make build && pytest -q'",
            "make build\npytest -q",
            "./scripts/claude-helper.sh",
            "docker build -t claude-worker-sandbox:2.1.237 .",
        ],
    )
    def test_build_test_and_git_work_is_untouched(self, command):
        _mark_discord()
        assert _terminal(command) is None, command

    def test_an_empty_command_is_not_blocked(self):
        _mark_discord()
        assert _terminal("") is None
        assert resolve_pre_tool_block("terminal", {}, session_id=DISCORD_KEY) is None

    def test_ordinary_commands_are_allowed_when_eligibility_is_unknown(self):
        """Fail-closed must not become fail-dark: an unconfirmable session
        still has to be able to build and test."""
        _mark_unknown("sess:terminal-unknown-ok")
        assert _terminal("make build && pytest -q", session_id="sess:terminal-unknown-ok") is None


# ---------------------------------------------------------------------------
# Anti-overreach and fail-closed
# ---------------------------------------------------------------------------


class TestAntiOverreach:
    def test_an_unrelated_tool_name_is_never_gated(self, repo):
        _mark_discord()
        assert resolve_pre_tool_block(
            "read_file", {"path": str(repo / "app.py")}, session_id=DISCORD_KEY,
        ) is None

    @pytest.mark.parametrize("action", ["view", "list", "search", "read", "describe"])
    def test_a_non_mutating_skill_manage_action_is_not_gated(self, repo, action):
        _mark_discord()
        assert resolve_pre_tool_block(
            "skill_manage", {"action": action, "name": "my-skill"}, session_id=DISCORD_KEY,
        ) is None

    def test_an_irrelevant_tool_is_allowed_even_when_eligibility_is_unknown(self, repo):
        _mark_unknown("sess:unknown-irrelevant")
        assert resolve_pre_tool_block(
            "read_file", {"path": str(repo / "app.py")}, session_id="sess:unknown-irrelevant",
        ) is None

    def test_a_path_outside_git_is_allowed_even_when_eligibility_is_unknown(self, tmp_path):
        scratch = tmp_path / "scratch"
        scratch.mkdir()
        _mark_unknown("sess:unknown-outside")
        assert resolve_pre_tool_block(
            "write_file", {"path": str(scratch / "notes.txt"), "content": "hi"},
            session_id="sess:unknown-outside",
        ) is None


class TestSkillManageScope:
    @pytest.mark.parametrize("action", sorted(policy.SKILL_MANAGE_MUTATING_ACTIONS))
    def test_every_mutating_action_is_gated_inside_a_repo(self, repo, monkeypatch, action):
        skill_dir = repo / "skills" / "my-skill"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text("# skill\n", encoding="utf-8")
        monkeypatch.setattr(
            gate, "_resolve_skill_manage_path", lambda args: str(skill_dir / "SKILL.md"),
        )
        _mark_discord()
        assert resolve_pre_tool_block(
            "skill_manage", {"action": action, "name": "my-skill"}, session_id=DISCORD_KEY,
        ) is not None

    def test_a_skill_outside_every_worktree_is_not_gated(self, tmp_path, monkeypatch):
        outside = tmp_path / "hermes-skills" / "my-skill" / "SKILL.md"
        outside.parent.mkdir(parents=True)
        outside.write_text("# skill\n", encoding="utf-8")
        monkeypatch.setattr(gate, "_resolve_skill_manage_path", lambda args: str(outside))
        _mark_discord()
        assert resolve_pre_tool_block(
            "skill_manage", {"action": "edit", "name": "my-skill"}, session_id=DISCORD_KEY,
        ) is None

    def test_a_resolution_error_fails_closed(self, repo, monkeypatch):
        def _boom(args):
            raise RuntimeError("skill dir resolution exploded")

        monkeypatch.setattr(gate, "_resolve_skill_manage_path", _boom)
        _mark_discord()
        assert resolve_pre_tool_block(
            "skill_manage", {"action": "write_file", "name": "my-skill"}, session_id=DISCORD_KEY,
        ) is not None

    def test_a_missing_skill_name_fails_closed(self, repo):
        _mark_discord()
        assert resolve_pre_tool_block(
            "skill_manage", {"action": "patch"}, session_id=DISCORD_KEY,
        ) is not None

    def test_an_unresolvable_path_fails_closed(self, repo, monkeypatch):
        monkeypatch.setattr(gate, "_resolve_skill_manage_path", lambda args: None)
        _mark_discord()
        assert resolve_pre_tool_block(
            "skill_manage", {"action": "delete", "name": "my-skill"}, session_id=DISCORD_KEY,
        ) is not None


class TestFailClosed:
    def test_a_missing_path_fails_closed(self, repo):
        _mark_discord()
        assert resolve_pre_tool_block(
            "write_file", {"content": "print(1)"}, session_id=DISCORD_KEY,
        ) is not None

    def test_an_eligibility_lookup_error_fails_closed_not_open(self, repo, monkeypatch):
        monkeypatch.setattr(
            canary, "current_session_eligibility_state",
            lambda session_id="": (_ for _ in ()).throw(RuntimeError("x")),
        )
        assert _patch(repo / "app.py") is not None

    def test_the_gate_hands_the_explicit_session_id_to_the_resolver(self, repo, monkeypatch):
        """Structural: the resolver must be ASKED about the explicit key. A
        gate that resolves eligibility from the ambient contextvars alone is
        the exact shape of the live bypass."""
        seen = []

        def _record_and_gate(session_id=""):
            seen.append(session_id)
            return True

        monkeypatch.setattr(canary, "current_session_eligibility_state", _record_and_gate)
        _patch(repo / "app.py", session_id=DISCORD_KEY)
        assert seen == [DISCORD_KEY]

    def test_a_config_load_error_fails_closed(self, repo, monkeypatch):
        """A config backend failure makes eligibility UNKNOWN, which must
        block a gated edit inside a real repository rather than silently
        becoming "nothing is in scope"."""
        def _boom():
            raise RuntimeError("config backend unavailable")

        monkeypatch.setattr(config, "load_plugin_config", _boom)
        _mark_discord()
        assert _patch(repo / "app.py") is not None

    def test_a_project_resolution_error_fails_closed(self, repo, monkeypatch):
        def _boom(path):
            raise RuntimeError("resolver exploded")

        monkeypatch.setattr(gate._project, "resolve_project_root_for_target", _boom)
        _mark_discord()
        assert _patch(repo / "app.py") is not None


class TestContinuationContextvarLossCannotBypassTheGate:
    """The live incident, reproduced at the enforcement boundary.

    A compressed/continued turn ran with every ambient session ContextVar
    cleared to ``""`` (what ``clear_session_vars`` leaves behind, which
    deliberately suppresses the ``os.environ`` fallback). Eligibility then
    matched the "no chat identity at all" branch — the CLI answer — and
    returned a confident ``False``, so BOTH the write gate and the terminal
    guard were silently dropped for a session that was plainly Discord.

    ``pre_tool_call`` is handed the Discord session key explicitly on every
    single call, so nothing about this was unknowable.
    """

    #: The session key from the incident, verbatim in shape.
    DISCORD_SESSION_ID = "agent:main:discord:group:1527706694665113670:970735341680082944"
    TELEGRAM_SESSION_ID = "agent:main:telegram:group:55555"
    CLI_SESSION_ID = "agent:main:local:cli"

    def _lose_the_ambient_session(self):
        """Every contextvar cleared, and no recorded dispatch identity — the
        state a continuation after context compression was observed in."""
        canary._identity_cache.clear()
        for var in _VAR_MAP.values():
            var.set("")

    def test_patch_is_still_gated_from_the_explicit_session_id_alone(self, repo):
        self._lose_the_ambient_session()
        message = _patch(repo / "app.py", session_id=self.DISCORD_SESSION_ID)
        assert message is not None
        assert "claude_worker" in message

    def test_write_file_is_still_gated_from_the_explicit_session_id_alone(self, repo):
        self._lose_the_ambient_session()
        assert resolve_pre_tool_block(
            "write_file", {"path": str(repo / "new.py"), "content": "print(1)"},
            session_id=self.DISCORD_SESSION_ID,
        ) is not None

    def test_skill_manage_is_still_gated_from_the_explicit_session_id_alone(
        self, repo, monkeypatch,
    ):
        skill_dir = repo / "skills" / "s"
        skill_dir.mkdir(parents=True)
        monkeypatch.setattr(
            gate, "_resolve_skill_manage_path", lambda args: str(skill_dir / "SKILL.md"),
        )
        self._lose_the_ambient_session()
        assert resolve_pre_tool_block(
            "skill_manage", {"action": "edit", "name": "s"},
            session_id=self.DISCORD_SESSION_ID,
        ) is not None

    def test_the_terminal_guard_still_fires_from_the_explicit_session_id_alone(self):
        """The other half of the incident: the same session went on to launch
        a host ``claude -p`` through ``terminal``."""
        self._lose_the_ambient_session()
        message = _terminal("claude -p 'fix it'", session_id=self.DISCORD_SESSION_ID)
        assert message is not None
        assert "claude_worker" in message

    def test_a_recorded_dispatch_identity_is_found_by_the_explicit_session_id(self, repo):
        """Cache lookups keyed only on the ambient key missed a recorded
        identity the moment the ambient key was gone."""
        canary._record(self.DISCORD_SESSION_ID, _identity())
        for var in _VAR_MAP.values():
            var.set("")
        assert _patch(repo / "app.py", session_id=self.DISCORD_SESSION_ID) is not None

    def test_a_telegram_session_id_still_passes_through(self, repo):
        """Fail-closed must not become fail-dark for other platforms."""
        self._lose_the_ambient_session()
        assert _patch(repo / "app.py", session_id=self.TELEGRAM_SESSION_ID) is None
        assert _terminal("claude -p x", session_id=self.TELEGRAM_SESSION_ID) is None

    def test_a_cli_session_id_still_passes_through(self, repo):
        self._lose_the_ambient_session()
        assert _patch(repo / "app.py", session_id=self.CLI_SESSION_ID) is None

    def test_ordinary_terminal_work_survives_the_contextvar_loss(self):
        self._lose_the_ambient_session()
        assert _terminal(
            "make build && pytest -q", session_id=self.DISCORD_SESSION_ID,
        ) is None

    def test_a_path_outside_every_worktree_is_still_not_gated(self, tmp_path):
        scratch = tmp_path / "scratch"
        scratch.mkdir()
        self._lose_the_ambient_session()
        assert resolve_pre_tool_block(
            "write_file", {"path": str(scratch / "notes.txt"), "content": "hi"},
            session_id=self.DISCORD_SESSION_ID,
        ) is None

    def test_a_successful_run_does_not_release_this_path_either(self, repo):
        """The two fixes compose: recovering Discord provenance from the
        explicit key must not hand back the unlock that was just removed."""
        self._lose_the_ambient_session()
        _record_worker_result(
            {"success": True, "cwd": str(repo)}, session_id=self.DISCORD_SESSION_ID,
        )
        assert _patch(repo / "app.py", session_id=self.DISCORD_SESSION_ID) is not None

    def test_patch_and_terminal_are_both_blocked_from_the_explicit_session_id_alone(
        self, repo,
    ):
        """The regression, as a single scenario: with every ambient session
        ContextVar AND the identity cache empty, the explicit Discord-shaped
        ``session_id`` alone must still gate BOTH halves of this module — the
        write gate and the terminal bypass — not just whichever one a prior
        test happened to exercise."""
        self._lose_the_ambient_session()
        patch_message = _patch(repo / "app.py", session_id=self.DISCORD_SESSION_ID)
        assert patch_message is not None
        assert "claude_worker" in patch_message

        terminal_message = _terminal(
            "claude -p 'fix it'", session_id=self.DISCORD_SESSION_ID,
        )
        assert terminal_message is not None
        assert "claude_worker" in terminal_message

    def test_a_successful_run_does_not_release_write_file_or_skill_manage_either(
        self, repo, monkeypatch,
    ):
        """Extends ``test_a_successful_run_does_not_release_this_path_either``
        (patch only) to the other two gated tools: recording a successful
        ``claude_worker`` run for this Discord session/repo — session
        identity recovered from the explicit key alone, ambient context lost
        — must not unlock ``write_file`` or a mutating ``skill_manage``
        either."""
        skill_dir = repo / "skills" / "s"
        skill_dir.mkdir(parents=True)
        monkeypatch.setattr(
            gate, "_resolve_skill_manage_path", lambda args: str(skill_dir / "SKILL.md"),
        )
        self._lose_the_ambient_session()
        _record_worker_result(
            {"success": True, "cwd": str(repo)}, session_id=self.DISCORD_SESSION_ID,
        )
        assert _patch(repo / "app.py", session_id=self.DISCORD_SESSION_ID) is not None
        assert resolve_pre_tool_block(
            "write_file", {"path": str(repo / "new2.py"), "content": "print(1)"},
            session_id=self.DISCORD_SESSION_ID,
        ) is not None
        assert resolve_pre_tool_block(
            "skill_manage", {"action": "patch", "name": "s"},
            session_id=self.DISCORD_SESSION_ID,
        ) is not None


class TestUnknownEligibilityBlocksInScopeEdits:
    """UNKNOWN — a real chat session whose platform could not be confirmed
    (evicted/reset/not-yet-populated cache, a gateway session-key failure) —
    must BLOCK a gated write inside a Git worktree, never silently allow it."""

    UNKNOWN_KEY = "sess:gate-unknown"

    def test_an_unconfirmable_session_blocks_a_gated_write(self, repo):
        _mark_unknown(self.UNKNOWN_KEY)
        message = _patch(repo / "app.py", session_id=self.UNKNOWN_KEY)
        assert message is not None
        assert "claude_worker" in message

    def test_the_message_names_the_repository_and_the_tool(self, repo):
        _mark_unknown(self.UNKNOWN_KEY)
        message = _patch(repo / "app.py", session_id=self.UNKNOWN_KEY)
        assert os.path.realpath(str(repo)) in message
        assert "patch" in message

    def test_a_cache_reset_restart_blocks_a_previously_recorded_session(self, repo):
        canary._record(self.UNKNOWN_KEY, _identity())
        _VAR_MAP["HERMES_SESSION_KEY"].set(self.UNKNOWN_KEY)
        assert _patch(repo / "app.py", session_id=self.UNKNOWN_KEY) is not None

        canary.reset_cache()
        _mark_unknown(self.UNKNOWN_KEY)
        assert _patch(repo / "app.py", session_id=self.UNKNOWN_KEY) is not None

    def test_a_gateway_session_key_failure_leaves_the_session_blocked(self, repo):
        class _BrokenGateway:
            def _session_key_for_source(self, source):
                raise RuntimeError("gateway session-key resolution exploded")

        class _P:
            def __init__(self, value):
                self.value = value

        class _Source:
            platform = _P("discord")
            chat_id = THREAD_ID
            parent_chat_id = ANOTHER_CHANNEL
            thread_id = THREAD_ID

        class _Event:
            source = _Source()

        canary.compute_and_cache_eligibility(_Event(), _BrokenGateway(), cfg=None)
        _mark_unknown(self.UNKNOWN_KEY)
        assert _patch(repo / "app.py", session_id=self.UNKNOWN_KEY) is not None

    def test_no_eviction_after_over_4096_recorded_sessions(self, repo):
        for i in range(4200):
            canary._record(f"sess:gate-bulk:{i}", _identity("telegram"))
        canary._record(self.UNKNOWN_KEY, _identity())
        assert canary._identity_cache["sess:gate-bulk:0"] == _identity("telegram")
        _VAR_MAP["HERMES_SESSION_KEY"].set(self.UNKNOWN_KEY)
        assert _patch(repo / "app.py", session_id=self.UNKNOWN_KEY) is not None


# ---------------------------------------------------------------------------
# F. The gateway session key — what the REAL dispatch sites actually pass
# ---------------------------------------------------------------------------
#
# Every test above hands ``resolve_pre_tool_block`` a gateway-shaped
# ``session_id``. Production does not: the agent tool-execution paths pass
# ``agent.session_id``, which is the TRANSCRIPT id — ``<timestamp>_<uuid6>``,
# minted by ``agent_init`` and re-minted by context compression. It names no
# platform and it is not stable across a continuation, so a gate that resolves
# Discord provenance from it alone is not enforcing anything in production.
#
# The stable per-chat key (``agent._gateway_session_key``) travels alongside it
# as ``gateway_session_key`` — see
# ``tests/agent/test_pre_tool_call_gateway_session_key.py`` for the dispatch
# half of this contract. These two classes are the enforcement half.

#: A transcript session id, in the exact shape ``agent_init`` mints.
TRANSCRIPT_SESSION_ID = "20260823_101500_ab12cd"

#: The same conversation AFTER a compression — new transcript id, same chat.
ROTATED_TRANSCRIPT_SESSION_ID = "20260823_113000_ef34gh"

DISCORD_GATEWAY_KEY = "agent:main:discord:group:1527706694665113670:970735341680082944"
OTHER_DISCORD_GATEWAY_KEY = "agent:main:discord:dm:8241"
TELEGRAM_GATEWAY_KEY = "agent:main:telegram:group:55555"


def _lose_the_ambient_session():
    """Every contextvar cleared and no recorded dispatch identity — the state a
    continuation after context compression was observed in, and the state that
    leaves the explicit arguments as the ONLY identity available."""
    canary._identity_cache.clear()
    for var in _VAR_MAP.values():
        var.set("")


def _patch_from_chat(
    target,
    *,
    session_id=TRANSCRIPT_SESSION_ID,
    gateway_session_key=DISCORD_GATEWAY_KEY,
):
    return resolve_pre_tool_block(
        "patch", {"path": str(target), "old_string": "x = 1", "new_string": "x = 2"},
        session_id=session_id,
        gateway_session_key=gateway_session_key,
    )


def _terminal_from_chat(
    command,
    *,
    session_id=TRANSCRIPT_SESSION_ID,
    gateway_session_key=DISCORD_GATEWAY_KEY,
):
    return resolve_pre_tool_block(
        "terminal", {"command": command},
        session_id=session_id,
        gateway_session_key=gateway_session_key,
    )


def _record_worker_result_from_chat(
    payload, *, session_id, gateway_session_key, status="ok",
):
    invoke_hook(
        "post_tool_call", tool_name="claude_worker",
        args={"task": "fix it", "cwd": payload.get("cwd", "")},
        result=json.dumps(payload),
        session_id=session_id,
        gateway_session_key=gateway_session_key,
        status=status,
    )


class TestATranscriptSessionIdWithADiscordGatewayKeyIsGated:
    """The BLOCKING half of the continuation bypass, at the enforcement
    boundary and in the shape production actually calls it.

    ``session_id`` is transcript-shaped (it names no platform and
    ``_is_gateway_session_key`` rejects it), every ambient session ContextVar
    is cleared, and the identity cache is empty. The Discord key reaches the
    gate ONLY as ``gateway_session_key`` — and that must be enough to gate
    both halves of this module.
    """

    def test_patch_and_terminal_are_both_blocked(self, repo):
        _lose_the_ambient_session()

        patch_message = _patch_from_chat(repo / "app.py")
        assert patch_message is not None
        assert "claude_worker" in patch_message

        terminal_message = _terminal_from_chat("claude -p 'fix it'")
        assert terminal_message is not None
        assert "claude_worker" in terminal_message

    def test_the_block_message_still_names_the_resolved_repository(self, repo):
        _lose_the_ambient_session()
        message = _patch_from_chat(repo / "app.py")
        assert os.path.realpath(str(repo)) in message

    def test_a_recorded_dispatch_identity_is_found_by_the_gateway_key(self, repo):
        """``pre_gateway_dispatch`` records identity under the key the GATEWAY
        resolved, so the cache can only ever be hit by looking that key up.

        ``DISCORD_KEY`` is deliberately not platform-readable on its own, so
        the block here can only come from the cache lookup — proving the
        gateway key is what the identity cache is consulted with, rather than
        the block riding on the ``:discord:`` segment of a well-shaped key.
        """
        canary._record(DISCORD_KEY, _identity())
        for var in _VAR_MAP.values():
            var.set("")
        assert _patch_from_chat(
            repo / "app.py", gateway_session_key=DISCORD_KEY,
        ) is not None

    def test_a_telegram_gateway_key_still_passes_through(self, repo):
        """Fail-closed must not become fail-dark: a confirmed non-Discord chat
        keeps direct edits and its own CLI usage."""
        _lose_the_ambient_session()
        assert _patch_from_chat(
            repo / "app.py", gateway_session_key=TELEGRAM_GATEWAY_KEY,
        ) is None
        assert _terminal_from_chat(
            "claude -p x", gateway_session_key=TELEGRAM_GATEWAY_KEY,
        ) is None

    def test_ordinary_terminal_work_is_untouched(self):
        _lose_the_ambient_session()
        assert _terminal_from_chat("make build && pytest -q") is None

    def test_a_path_outside_every_worktree_is_still_not_gated(self, tmp_path):
        scratch = tmp_path / "scratch"
        scratch.mkdir()
        _lose_the_ambient_session()
        assert resolve_pre_tool_block(
            "write_file", {"path": str(scratch / "notes.txt"), "content": "hi"},
            session_id=TRANSCRIPT_SESSION_ID,
            gateway_session_key=DISCORD_GATEWAY_KEY,
        ) is None

    def test_the_legacy_call_shape_without_a_gateway_key_is_unchanged(self, repo):
        """Compatibility: a caller that passes no ``gateway_session_key`` at
        all (the CLI, and every pre-existing call site) resolves exactly as it
        does today — a transcript id with no chat identity anywhere is the CLI,
        and the CLI is not gated."""
        _lose_the_ambient_session()
        assert resolve_pre_tool_block(
            "patch", {"path": str(repo / "app.py"), "old_string": "x = 1", "new_string": "x = 2"},
            session_id=TRANSCRIPT_SESSION_ID,
        ) is None


class TestTheFallbackReleaseIsKeyedByTheGatewaySessionKey:
    """The Terra fallback release — the ONE thing that releases a repository —
    must be recorded against the CHAT, not the transcript slice.

    Keyed by transcript id it is both too narrow and wrong: the very next
    context compression mints a new id and silently revokes a release the
    session was told it had, and no ``pre_tool_call`` can match it against the
    gateway-key identity eligibility is now resolved from.
    """

    def test_a_delivered_fallback_releases_the_chat_it_was_delivered_to(self, repo):
        _lose_the_ambient_session()
        assert _patch_from_chat(repo / "app.py") is not None

        _record_worker_result_from_chat(
            _fallback_payload(repo),
            session_id=TRANSCRIPT_SESSION_ID,
            gateway_session_key=DISCORD_GATEWAY_KEY,
        )
        assert _patch_from_chat(repo / "app.py") is None

    def test_the_release_survives_a_transcript_id_rotation(self, repo):
        """Compression rotates ``agent.session_id`` mid-conversation. Same
        chat, same repository, new transcript id — the release holds."""
        _lose_the_ambient_session()
        _record_worker_result_from_chat(
            _fallback_payload(repo),
            session_id=TRANSCRIPT_SESSION_ID,
            gateway_session_key=DISCORD_GATEWAY_KEY,
        )
        assert _patch_from_chat(
            repo / "app.py", session_id=ROTATED_TRANSCRIPT_SESSION_ID,
        ) is None

    def test_a_release_in_another_chat_does_not_transfer(self, repo):
        """Scope is unchanged in the direction that matters: another Discord
        chat's fallback releases nothing here, even when both chats are served
        by an agent carrying the same transcript id."""
        _lose_the_ambient_session()
        _record_worker_result_from_chat(
            _fallback_payload(repo),
            session_id=TRANSCRIPT_SESSION_ID,
            gateway_session_key=OTHER_DISCORD_GATEWAY_KEY,
        )
        assert _patch_from_chat(repo / "app.py") is not None

    def test_a_successful_run_still_releases_nothing(self, repo):
        """The policy the incident produced is untouched by the re-keying: the
        worker is the coder for the whole session, so a success grants nothing
        under the gateway key either."""
        _lose_the_ambient_session()
        _record_worker_result_from_chat(
            {"success": True, "cwd": str(repo)},
            session_id=TRANSCRIPT_SESSION_ID,
            gateway_session_key=DISCORD_GATEWAY_KEY,
        )
        assert _patch_from_chat(repo / "app.py") is not None

    def test_a_release_never_relaxes_the_terminal_guard(self, repo):
        """Releasing a repository for direct edits is a statement about that
        repository, never a licence to run an unsandboxed Claude on the host."""
        _lose_the_ambient_session()
        _record_worker_result_from_chat(
            _fallback_payload(repo),
            session_id=TRANSCRIPT_SESSION_ID,
            gateway_session_key=DISCORD_GATEWAY_KEY,
        )
        assert _patch_from_chat(repo / "app.py") is None
        assert _terminal_from_chat("claude -p 'fix it'") is not None
