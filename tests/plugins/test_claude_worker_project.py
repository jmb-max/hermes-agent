"""Tests for ``plugins/claude-worker/project.py`` — the ONE dynamic, safe
Git-worktree-root resolver the gate and the runner both consult.

This replaced the static ``gate.repo_roots`` allowlist entirely. That
allowlist went stale the moment a new worktree appeared (the worker simply
refused to run there), and whatever it named was what got MOUNTED — so an
operator entry of ``/root/worktrees`` handed every sandbox container every
sibling project at once. Scope is derived from the request now: given a
requested path, resolve the canonical Git worktree root that actually
contains it, and mount exactly that.

These tests pin down both halves: what the resolver ACCEPTS (an absolute,
existing directory inside a real worktree, including a linked worktree whose
``.git`` is a FILE), and every shape it must REJECT (relative, missing,
non-Git, bare, symlink escape, malformed ``.git`` marker, system-sensitive or
multi-project root). They also pin the hardening of the one git subprocess it
is allowed to run: absolute trusted binary, fixed argv with no shell, bounded
timeout, sanitized environment.
"""

from __future__ import annotations

import os
import subprocess
import types

import pytest

from tests.plugins._claude_worker_helpers import (
    load_submodule,
    make_git_repo,
    require_safe_tmp,
)

project = load_submodule("project")
policy = load_submodule("policy")
trust = load_submodule("trust")


@pytest.fixture(autouse=True)
def _safe_tmp(tmp_path):
    require_safe_tmp(tmp_path)


@pytest.fixture()
def repo(tmp_path):
    return make_git_repo(tmp_path, "repo")


def _fake_subprocess(run):
    """A stand-in for the ``subprocess`` module inside ``project.py``.

    Patching the module attribute (rather than ``subprocess.run`` globally)
    keeps the fake scoped to the resolver and leaves the exception types the
    real ``except`` clause references intact.
    """
    return types.SimpleNamespace(
        run=run,
        TimeoutExpired=subprocess.TimeoutExpired,
        CalledProcessError=subprocess.CalledProcessError,
    )


def _completed(stdout="", returncode=0):
    return types.SimpleNamespace(returncode=returncode, stdout=stdout, stderr="")


# ---------------------------------------------------------------------------
# Accepted shapes
# ---------------------------------------------------------------------------


class TestAcceptsRealWorktrees:
    def test_repo_root_itself_resolves_to_itself(self, repo):
        assert project.resolve_project_root_strict(str(repo)) == os.path.realpath(str(repo))

    def test_nested_directory_resolves_to_the_repo_root(self, repo):
        nested = repo / "src" / "deep" / "deeper"
        nested.mkdir(parents=True)
        assert project.resolve_project_root_strict(str(nested)) == os.path.realpath(str(repo))

    def test_nearest_ancestor_wins_for_a_nested_checkout(self, tmp_path):
        """A checkout inside a checkout resolves to the INNER one — mounting
        the outer would hand the worker a repository it was not asked for."""
        outer = make_git_repo(tmp_path, "outer")
        inner = make_git_repo(outer, "vendor-inner")
        assert project.resolve_project_root_strict(str(inner)) == os.path.realpath(str(inner))
        assert project.resolve_project_root_strict(str(outer)) == os.path.realpath(str(outer))

    def test_dot_git_FILE_linked_worktree_is_supported(self, tmp_path):
        """A linked worktree (and a submodule) has a ``.git`` regular FILE
        holding a ``gitdir:`` pointer, not a directory. A naive
        ``isdir(".git")`` check misses every one of them."""
        linked = make_git_repo(tmp_path, "linked", marker="file")
        assert (linked / ".git").is_file()
        assert project.resolve_project_root_strict(str(linked)) == os.path.realpath(str(linked))

    def test_dot_git_file_worktree_resolves_from_a_nested_path(self, tmp_path):
        linked = make_git_repo(tmp_path, "linked", marker="file")
        nested = linked / "pkg" / "mod"
        nested.mkdir(parents=True)
        assert project.resolve_project_root_strict(str(nested)) == os.path.realpath(str(linked))

    def test_trailing_slash_and_dot_segments_normalize(self, repo):
        expected = os.path.realpath(str(repo))
        assert project.resolve_project_root_strict(str(repo) + "/") == expected
        assert project.resolve_project_root_strict(str(repo) + "/./") == expected

    def test_a_symlinked_repo_root_is_canonicalized(self, tmp_path, repo):
        link = tmp_path / "linked-repo"
        link.symlink_to(repo, target_is_directory=True)
        assert project.resolve_project_root_strict(str(link)) == os.path.realpath(str(repo))


# ---------------------------------------------------------------------------
# Rejected shapes
# ---------------------------------------------------------------------------


class TestRejectsMalformedPaths:
    @pytest.mark.parametrize("bad", ["", "   ", None, 7, [], {}])
    def test_empty_or_non_string_paths_are_rejected(self, bad):
        with pytest.raises(project.ProjectRejected):
            project.resolve_project_root_strict(bad)

    def test_nul_byte_is_rejected(self):
        with pytest.raises(project.ProjectRejected, match="NUL"):
            project.resolve_project_root_strict("/repo\x00/etc")

    @pytest.mark.parametrize("relative", ["repo", "./repo", "../repo", "repo/src", "~/repo"])
    def test_relative_paths_are_rejected(self, relative):
        """A relative path would be resolved against whatever the process cwd
        happens to be at the moment — a completely different repository from
        one call to the next."""
        with pytest.raises(project.ProjectRejected, match="absolute"):
            project.resolve_project_root_strict(relative)

    def test_relative_path_is_rejected_even_when_it_names_a_real_repo(self, repo, monkeypatch):
        monkeypatch.chdir(repo.parent)
        with pytest.raises(project.ProjectRejected, match="absolute"):
            project.resolve_project_root_strict("repo")

    def test_missing_path_is_rejected(self, tmp_path):
        with pytest.raises(project.ProjectRejected, match="existing directory"):
            project.resolve_project_root_strict(str(tmp_path / "does-not-exist"))

    def test_a_file_is_not_a_project_root(self, repo):
        target = repo / "app.py"
        target.write_text("x = 1\n", encoding="utf-8")
        with pytest.raises(project.ProjectRejected, match="existing directory"):
            project.resolve_project_root_strict(str(target))


class TestRejectsNonWorktrees:
    def test_plain_scratch_directory_is_not_a_worktree(self, tmp_path):
        scratch = tmp_path / "scratch"
        scratch.mkdir()
        with pytest.raises(project.ProjectRejected, match="not inside a Git worktree"):
            project.resolve_project_root_strict(str(scratch))

    def test_bare_repository_has_no_worktree_and_is_rejected(self, tmp_path):
        """A bare repo carries HEAD/objects/refs at its top level and has no
        ``.git`` entry at all, so the ancestor walk finds no marker — and git
        itself reports ``--is-bare-repository true`` if it is consulted."""
        bare = tmp_path / "project.git"
        (bare / "objects").mkdir(parents=True)
        (bare / "refs").mkdir()
        (bare / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
        (bare / "config").write_text("[core]\n\tbare = true\n", encoding="utf-8")
        with pytest.raises(project.ProjectRejected):
            project.resolve_project_root_strict(str(bare))

    def test_git_reporting_a_bare_repository_is_fatal(self, monkeypatch, repo):
        monkeypatch.setattr(project, "git_available", lambda: True)
        monkeypatch.setattr(
            project, "subprocess",
            _fake_subprocess(lambda *a, **k: _completed("true\ntrue\n/somewhere\n")),
        )
        with pytest.raises(project.ProjectRejected, match="bare repository"):
            project.resolve_project_root_strict(str(repo))

    def test_git_reporting_not_inside_a_worktree_is_fatal(self, monkeypatch, repo):
        monkeypatch.setattr(project, "git_available", lambda: True)
        monkeypatch.setattr(
            project, "subprocess",
            _fake_subprocess(lambda *a, **k: _completed("false\nfalse\n/somewhere\n")),
        )
        with pytest.raises(project.ProjectRejected, match="not inside a worktree"):
            project.resolve_project_root_strict(str(repo))


class TestRejectsMalformedGitMarkers:
    def test_dot_git_symlink_is_rejected(self, tmp_path):
        elsewhere = tmp_path / "elsewhere.git"
        elsewhere.mkdir()
        sneaky = tmp_path / "sneaky"
        sneaky.mkdir()
        (sneaky / ".git").symlink_to(elsewhere, target_is_directory=True)
        with pytest.raises(project.ProjectRejected, match="symlink"):
            project.resolve_project_root_strict(str(sneaky))

    def test_dot_git_file_without_a_gitdir_pointer_is_rejected(self, tmp_path):
        fake = tmp_path / "fake"
        fake.mkdir()
        (fake / ".git").write_text("not a worktree pointer at all\n", encoding="utf-8")
        with pytest.raises(project.ProjectRejected, match="gitdir"):
            project.resolve_project_root_strict(str(fake))

    def test_dot_git_file_that_is_not_utf8_is_rejected(self, tmp_path):
        fake = tmp_path / "binary-marker"
        fake.mkdir()
        (fake / ".git").write_bytes(b"\xff\xfe\x00\x01gitdir: /tmp\n")
        with pytest.raises(project.ProjectRejected, match="UTF-8"):
            project.resolve_project_root_strict(str(fake))

    def test_dot_git_fifo_is_rejected(self, tmp_path):
        fake = tmp_path / "fifo-marker"
        fake.mkdir()
        try:
            os.mkfifo(str(fake / ".git"))
        except (AttributeError, OSError):  # pragma: no cover - platform dependent
            pytest.skip("mkfifo unavailable on this platform")
        with pytest.raises(project.ProjectRejected, match="neither a directory nor a regular file"):
            project.resolve_project_root_strict(str(fake))


class TestRejectsUnsafeRoots:
    @pytest.mark.parametrize(
        "sensitive",
        ["/", "/etc", "/usr", "/var", "/proc", "/sys", "/dev", "/bin", "/sbin", "/run", "/tmp"],
    )
    def test_system_sensitive_roots_are_named_unsafe(self, sensitive):
        assert project.unsafe_root_reason(sensitive) is not None

    @pytest.mark.parametrize(
        "nested", ["/etc/anything", "/usr/local/src/repo", "/var/lib/checkout", "/run/user/0/repo"],
    )
    def test_anything_under_a_system_prefix_is_unsafe_at_any_depth(self, nested):
        assert project.unsafe_root_reason(nested) is not None

    @pytest.mark.parametrize("home", ["/root", "/home/alice", "/home/bob"])
    def test_a_whole_home_directory_is_never_one_project(self, home):
        """Mounting a home directory would expose ``.ssh``, ``.claude``,
        shell history and every unrelated checkout under it."""
        assert project.unsafe_root_reason(home) is not None

    def test_the_multi_project_worktree_container_is_unsafe(self):
        assert project.unsafe_root_reason("/root/worktrees") is not None

    def test_a_project_inside_a_home_directory_is_still_fine(self):
        assert project.unsafe_root_reason("/home/alice/projects/thing") is None
        assert project.unsafe_root_reason("/root/worktrees/one-repo") is None

    def test_an_unsafe_resolved_root_is_refused_by_the_resolver(self, monkeypatch, repo):
        """Exercised through ``resolve_project_root_strict`` itself, not just
        the predicate: a real ``.git`` marker at an unsafe root must not be
        enough to make it mountable."""
        real = os.path.realpath(str(repo))
        monkeypatch.setattr(
            policy, "UNSAFE_PROJECT_ROOTS", frozenset(policy.UNSAFE_PROJECT_ROOTS | {real}),
        )
        with pytest.raises(project.ProjectRejected, match="unsafe project root"):
            project.resolve_project_root_strict(str(repo))


class TestSymlinkAndSiblingEscapes:
    def test_a_symlink_inside_a_repo_pointing_outside_is_judged_where_it_lands(
        self, tmp_path, repo,
    ):
        """``realpath`` runs BEFORE anything else, so a link planted inside a
        repo that points out of it resolves to where it really goes and is
        judged there — never smuggled back in as "inside the repo"."""
        outside = tmp_path / "secrets"
        outside.mkdir()
        (repo / "escape").symlink_to(outside, target_is_directory=True)

        with pytest.raises(project.ProjectRejected, match="not inside a Git worktree"):
            project.resolve_project_root_strict(str(repo / "escape"))

    def test_a_symlink_escaping_into_ANOTHER_repo_resolves_to_that_repo(self, tmp_path, repo):
        other = make_git_repo(tmp_path, "other-repo")
        (repo / "escape").symlink_to(other, target_is_directory=True)
        assert project.resolve_project_root_strict(str(repo / "escape")) == os.path.realpath(str(other))

    def test_a_symlinked_target_file_is_never_attributed_to_the_link_repo(self, tmp_path, repo):
        outside = tmp_path / "secrets"
        outside.mkdir()
        (outside / "creds.txt").write_text("token\n", encoding="utf-8")
        (repo / "escape").symlink_to(outside, target_is_directory=True)
        assert project.resolve_project_root_for_target(str(repo / "escape" / "creds.txt")) is None

    def test_sibling_prefix_is_not_containment(self):
        assert project.contains("/repo", "/repo") is True
        assert project.contains("/repo", "/repo/src/a.py") is True
        assert project.contains("/repo", "/repo-evil") is False
        assert project.contains("/repo", "/repo-evil/a.py") is False
        assert project.contains("/repo/", "/repo-evil/a.py") is False

    def test_a_sibling_repo_resolves_to_itself_not_its_neighbour(self, tmp_path):
        make_git_repo(tmp_path, "repo")
        evil = make_git_repo(tmp_path, "repo-evil")
        (evil / "a.py").write_text("x = 1\n", encoding="utf-8")
        assert project.resolve_project_root_for_target(str(evil / "a.py")) == os.path.realpath(str(evil))

    def test_dotdot_traversal_out_of_a_repo_is_not_attributed_to_it(self, tmp_path, repo):
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "f.py").write_text("x = 1\n", encoding="utf-8")
        traversal = str(repo / ".." / "outside" / "f.py")
        assert project.resolve_project_root_for_target(traversal) is None


# ---------------------------------------------------------------------------
# The non-raising wrappers used by the gate
# ---------------------------------------------------------------------------


class TestNonRaisingWrappers:
    def test_resolve_project_root_returns_none_instead_of_raising(self, tmp_path):
        scratch = tmp_path / "scratch"
        scratch.mkdir()
        assert project.resolve_project_root(str(scratch)) is None
        assert project.resolve_project_root("relative/path") is None
        assert project.resolve_project_root(None) is None

    def test_resolve_project_root_returns_the_root_on_success(self, repo):
        assert project.resolve_project_root(str(repo)) == os.path.realpath(str(repo))

    def test_target_resolution_handles_a_file_that_does_not_exist_yet(self, repo):
        """``write_file`` legitimately creates new files, sometimes in new
        subdirectories — the gate must still know which repo they land in."""
        assert project.resolve_project_root_for_target(
            str(repo / "brand" / "new" / "file.py")
        ) == os.path.realpath(str(repo))

    def test_target_resolution_for_an_existing_file(self, repo):
        target = repo / "app.py"
        target.write_text("x = 1\n", encoding="utf-8")
        assert project.resolve_project_root_for_target(str(target)) == os.path.realpath(str(repo))

    def test_target_outside_every_worktree_is_none(self, tmp_path):
        scratch = tmp_path / "scratch"
        scratch.mkdir()
        assert project.resolve_project_root_for_target(str(scratch / "notes.txt")) is None

    @pytest.mark.parametrize("bad", ["", "   ", None, 7, "/repo\x00/x"])
    def test_malformed_targets_are_none(self, bad):
        assert project.resolve_project_root_for_target(bad) is None

    def test_a_relative_target_is_resolved_against_the_process_cwd(self, repo, monkeypatch):
        """The same thing the underlying tool would do — a relative target is
        resolved, not waved through as "unknown, therefore out of scope"."""
        monkeypatch.chdir(repo)
        assert project.resolve_project_root_for_target("app.py") == os.path.realpath(str(repo))

    def test_a_relative_target_outside_any_repo_is_none(self, tmp_path, monkeypatch):
        scratch = tmp_path / "scratch"
        scratch.mkdir()
        monkeypatch.chdir(scratch)
        assert project.resolve_project_root_for_target("notes.txt") is None


# ---------------------------------------------------------------------------
# The git second opinion, and how it is invoked
# ---------------------------------------------------------------------------


class TestGitSubprocessHardening:
    def _capture(self, monkeypatch, stdout, returncode=0):
        captured = {}

        def _run(argv, **kwargs):
            captured["argv"] = argv
            captured["kwargs"] = kwargs
            return _completed(stdout, returncode)

        monkeypatch.setattr(project, "git_available", lambda: True)
        monkeypatch.setattr(project, "subprocess", _fake_subprocess(_run))
        return captured

    def test_uses_the_absolute_trusted_binary_never_a_bare_command(self, monkeypatch, repo):
        real = os.path.realpath(str(repo))
        captured = self._capture(monkeypatch, f"true\nfalse\n{real}\n")
        project.resolve_project_root_strict(str(repo))
        assert captured["argv"][0] == policy.GIT_BIN
        assert os.path.isabs(captured["argv"][0])

    def test_argv_is_a_list_and_there_is_no_shell(self, monkeypatch, repo):
        real = os.path.realpath(str(repo))
        captured = self._capture(monkeypatch, f"true\nfalse\n{real}\n")
        project.resolve_project_root_strict(str(repo))
        assert isinstance(captured["argv"], list)
        assert all(isinstance(token, str) for token in captured["argv"])
        assert captured["kwargs"].get("shell") in (None, False)

    def test_timeout_is_bounded(self, monkeypatch, repo):
        real = os.path.realpath(str(repo))
        captured = self._capture(monkeypatch, f"true\nfalse\n{real}\n")
        project.resolve_project_root_strict(str(repo))
        assert captured["kwargs"]["timeout"] == policy.GIT_RESOLVE_TIMEOUT_SECONDS

    def test_environment_is_built_from_scratch_and_sanitized(self, monkeypatch, repo):
        monkeypatch.setenv("GIT_CONFIG_GLOBAL", "/attacker/gitconfig")
        monkeypatch.setenv("GIT_DIR", "/attacker/git")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-should-never-cross")
        real = os.path.realpath(str(repo))
        captured = self._capture(monkeypatch, f"true\nfalse\n{real}\n")
        project.resolve_project_root_strict(str(repo))

        env = captured["kwargs"]["env"]
        assert env["GIT_CONFIG_NOSYSTEM"] == "1"
        assert env["GIT_CONFIG_GLOBAL"] == "/dev/null"
        assert env["GIT_CONFIG_SYSTEM"] == "/dev/null"
        assert "GIT_DIR" not in env
        assert "ANTHROPIC_API_KEY" not in env
        assert env["HOME"] == "/nonexistent"
        assert env["PATH"] == "/usr/bin:/bin"

    def test_hostile_repo_config_knobs_are_overridden(self, monkeypatch, repo):
        real = os.path.realpath(str(repo))
        captured = self._capture(monkeypatch, f"true\nfalse\n{real}\n")
        project.resolve_project_root_strict(str(repo))
        argv = captured["argv"]
        for override in ("core.hooksPath=/dev/null", "core.fsmonitor=false",
                         "protocol.ext.allow=never", "safe.directory=*"):
            assert override in argv

    def test_safe_directory_is_the_literal_glob_not_the_requested_path(self, monkeypatch, repo):
        """Interpolating the requested path into a config value would give a
        path somewhere to inject a second config key."""
        real = os.path.realpath(str(repo))
        captured = self._capture(monkeypatch, f"true\nfalse\n{real}\n")
        project.resolve_project_root_strict(str(repo))
        assert "safe.directory=*" in captured["argv"]
        assert f"safe.directory={real}" not in captured["argv"]

    def test_untrusted_git_binary_is_never_executed(self, monkeypatch, repo):
        ran = {"n": 0}

        def _run(*a, **k):  # pragma: no cover - must never be reached
            ran["n"] += 1
            raise AssertionError("git must not run when its path chain is untrusted")

        monkeypatch.setattr(project, "git_available", lambda: False)
        monkeypatch.setattr(project, "subprocess", _fake_subprocess(_run))

        # The walk still resolves the repo — an untrusted git means "no
        # opinion", not "refuse everything".
        assert project.resolve_project_root_strict(str(repo)) == os.path.realpath(str(repo))
        assert ran["n"] == 0

    def test_git_availability_is_never_cached(self, monkeypatch):
        """A binary swapped in after an earlier success must be caught."""
        calls = []

        def _validate(path, executable=False):
            calls.append(path)
            if len(calls) > 1:
                raise trust.TrustViolation("binary was replaced")
            return {}

        monkeypatch.setattr(project._trust, "validate_trusted_path_chain", _validate)
        assert project.git_available() is True
        assert project.git_available() is False
        assert calls == [policy.GIT_BIN, policy.GIT_BIN]


class TestGitDisagreementIsFatal:
    def test_a_different_toplevel_refuses_rather_than_guessing(self, monkeypatch, tmp_path, repo):
        other = make_git_repo(tmp_path, "other")
        monkeypatch.setattr(project, "git_available", lambda: True)
        monkeypatch.setattr(
            project, "subprocess",
            _fake_subprocess(
                lambda *a, **k: _completed(f"true\nfalse\n{os.path.realpath(str(other))}\n")
            ),
        )
        with pytest.raises(project.ProjectRejected, match="disagree"):
            project.resolve_project_root_strict(str(repo))

    def test_agreement_resolves_normally(self, monkeypatch, repo):
        real = os.path.realpath(str(repo))
        monkeypatch.setattr(project, "git_available", lambda: True)
        monkeypatch.setattr(
            project, "subprocess",
            _fake_subprocess(lambda *a, **k: _completed(f"true\nfalse\n{real}\n")),
        )
        assert project.resolve_project_root_strict(str(repo)) == real

    def test_verify_with_git_false_skips_the_cross_check(self, monkeypatch, repo):
        ran = {"n": 0}

        def _run(*a, **k):  # pragma: no cover - must never be reached
            ran["n"] += 1
            raise AssertionError("git must not be consulted when verification is off")

        monkeypatch.setattr(project, "git_available", lambda: True)
        monkeypatch.setattr(project, "subprocess", _fake_subprocess(_run))
        assert project.resolve_project_root_strict(
            str(repo), verify_with_git=False
        ) == os.path.realpath(str(repo))
        assert ran["n"] == 0


class TestGitNoOpinionCases:
    """A git failure means "no usable opinion" and the deterministic
    filesystem walk stands alone — it must never silently widen or narrow
    scope, and it must never propagate an exception to the caller."""

    def test_nonzero_exit_leaves_the_walk_authoritative(self, monkeypatch, repo):
        monkeypatch.setattr(project, "git_available", lambda: True)
        monkeypatch.setattr(
            project, "subprocess",
            _fake_subprocess(lambda *a, **k: _completed("fatal: not a git repository\n", 128)),
        )
        assert project.resolve_project_root_strict(str(repo)) == os.path.realpath(str(repo))

    def test_timeout_leaves_the_walk_authoritative(self, monkeypatch, repo):
        def _timeout(*a, **k):
            raise subprocess.TimeoutExpired(cmd="git", timeout=1)

        monkeypatch.setattr(project, "git_available", lambda: True)
        monkeypatch.setattr(project, "subprocess", _fake_subprocess(_timeout))
        assert project.resolve_project_root_strict(str(repo)) == os.path.realpath(str(repo))

    def test_os_error_leaves_the_walk_authoritative(self, monkeypatch, repo):
        def _oserror(*a, **k):
            raise OSError("git vanished")

        monkeypatch.setattr(project, "git_available", lambda: True)
        monkeypatch.setattr(project, "subprocess", _fake_subprocess(_oserror))
        assert project.resolve_project_root_strict(str(repo)) == os.path.realpath(str(repo))

    @pytest.mark.parametrize(
        "stdout", ["", "true\n", "true\nfalse\n", "true\nfalse\n/a\n/b\n", "junk\n"],
    )
    def test_unexpected_output_shape_is_no_opinion(self, monkeypatch, repo, stdout):
        monkeypatch.setattr(project, "git_available", lambda: True)
        monkeypatch.setattr(
            project, "subprocess", _fake_subprocess(lambda *a, **k: _completed(stdout)),
        )
        assert project.resolve_project_root_strict(str(repo)) == os.path.realpath(str(repo))

    @pytest.mark.parametrize("toplevel", ["relative/path", "../escape", "/has\x00nul"])
    def test_a_malformed_toplevel_is_fatal_not_ignored(self, monkeypatch, repo, toplevel):
        monkeypatch.setattr(project, "git_available", lambda: True)
        monkeypatch.setattr(
            project, "subprocess",
            _fake_subprocess(lambda *a, **k: _completed(f"true\nfalse\n{toplevel}\n")),
        )
        with pytest.raises(project.ProjectRejected):
            project.resolve_project_root_strict(str(repo))

    def test_a_toplevel_that_is_not_a_directory_is_fatal(self, monkeypatch, tmp_path, repo):
        monkeypatch.setattr(project, "git_available", lambda: True)
        monkeypatch.setattr(
            project, "subprocess",
            _fake_subprocess(lambda *a, **k: _completed(f"true\nfalse\n{tmp_path / 'nope'}\n")),
        )
        with pytest.raises(project.ProjectRejected, match="not an existing directory"):
            project.resolve_project_root_strict(str(repo))


class TestRealGitCheckoutEndToEnd:
    """One end-to-end pass against a genuine ``git init`` checkout, so the
    hardened argv is proven to actually work against real git rather than
    only against a fake."""

    def test_real_checkout_resolves_and_git_agrees(self, tmp_path):
        if not project.git_available():
            pytest.skip("the fixed git binary is not trusted in this environment")
        repo = tmp_path / "real-repo"
        repo.mkdir()
        try:
            subprocess.run(["git", "init", "-q"], cwd=repo, check=True, timeout=30)
        except (OSError, subprocess.SubprocessError):  # pragma: no cover - env dependent
            pytest.skip("git is not usable in this environment")

        nested = repo / "src"
        nested.mkdir()
        expected = os.path.realpath(str(repo))
        assert project.resolve_project_root_strict(str(repo)) == expected
        assert project.resolve_project_root_strict(str(nested)) == expected
