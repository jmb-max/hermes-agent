"""Tests for ``plugins/claude-worker/terminal_guard.py`` — the terminal
bypass.

The write gate makes ``patch``/``write_file``/``skill_manage`` go through
``claude_worker``, but ``terminal`` takes a free-form command string, so
``claude -p "edit the file"`` routed around the entire sandbox: no image pin,
no non-root uid, no empty MCP config, no tool allowlist, no resource caps,
and full host filesystem access.

This module answers exactly one question: *does this command string directly
invoke the Claude CLI?* Two failure modes matter equally and both are tested
here:

* **Under-blocking** — a variant that still reaches a host ``claude``
  (wrappers, executors such as ``xargs``/``watch``/``parallel``/``find
  -exec``, GNU ``env -S``, busybox applet dispatch, shell chains, nested
  ``sh -c`` payloads including option clusters (``bash -cx``) and options
  that take values (``bash -O extglob -c``), scripts fed to a shell as
  visible text (``bash <<< "claude -p x"``), segments headed by shell
  grammar (``then``/``do``/``!``/``{``), executable substitutions, path
  forms).
* **Over-blocking** — ordinary build/test/git work is what these sessions
  exist to do. A token only counts in COMMAND HEAD position, so merely
  mentioning claude never blocks, a reserved word in an ARGUMENT
  (``echo "then claude"``) is just a word, and the executor rules must not
  swallow ``find . -name claude`` or ``xargs grep claude``.

The guard is lexical, and ``TestDocumentedScopeLimits`` pins the boundary it
does NOT claim: a command that constructs its target at runtime has no
``claude`` token to find, and the sandbox — not this module — contains that.

Enforcement through the real ``pre_tool_call`` dispatch path lives in
``test_claude_worker_gate.py``; this module is the detection matrix.
"""

from __future__ import annotations

import pytest

from tests.plugins._claude_worker_helpers import load_submodule

terminal_guard = load_submodule("terminal_guard")
policy = load_submodule("policy")


def _blocked(command):
    verdict = terminal_guard.evaluate(command)
    assert set(verdict) == {"blocked", "reason", "malformed"}
    return verdict["blocked"]


class TestReportedShellNativeBypasses:
    """The exact command strings adversarial testing ran against a host and
    found ALLOWED while every one of them executes the Claude CLI.

    Kept together and verbatim so a regression in any single parsing rule
    shows up as "this reported bypass is open again" rather than as a failure
    somewhere in the detection matrix below. Each one is also covered by the
    class that owns its rule."""

    @pytest.mark.parametrize(
        "command",
        [
            'env -S "claude -p x"',
            'env --split-string="claude -p x"',
            'busybox sh -c "claude -p x"',
            "busybox env claude -p x",
            "xargs claude -p x",
            "printf x | xargs claude -p",
            "find . -exec claude -p x ;",
            "find . -execdir /usr/local/bin/claude -p x ;",
            "echo `claude -p x`",
            "cat <(claude -p x)",
            "cat >(claude -p x)",
        ],
    )
    def test_each_reported_bypass_is_closed(self, command):
        assert _blocked(command) is True


class TestReportedShellSemanticsBypasses:
    """The second reported set, verbatim: shell ``-c`` option semantics,
    scripts fed to a shell as text, and shell grammar that made the scan stop
    one token before claude. Every one of these executes the host CLI and was
    allowed. Each is also covered by the class that owns its rule."""

    @pytest.mark.parametrize(
        "command",
        [
            # -c semantics and shell options
            'bash -cx "claude -p x"',
            'sh -cv "claude -p x"',
            'bash -ce "claude -p x"',
            'bash -O extglob -c "claude -p x"',
            'bash --rcfile /dev/null -c "claude -p x"',
            'bash --noprofile --norc -c "claude -p x"',
            # a script fed to the shell as visible text
            'bash <<< "claude -p x"',
            'sh <<< "/usr/local/bin/claude -p x"',
            # shell reserved words, grouping and function grammar
            "if true; then claude -p x; fi",
            "if claude -p x; then true; fi",
            "while true; do claude -p x; done",
            "until false; do /usr/local/bin/claude -p x; done",
            "! claude -p x",
            "{ claude -p x; }",
            "f(){ claude -p x; }; f",
            "case x in x) claude -p x;; esac",
        ],
    )
    def test_each_reported_bypass_is_closed(self, command):
        assert _blocked(command) is True


class TestBareAndPathInvocations:
    @pytest.mark.parametrize(
        "command",
        [
            "claude",
            "claude -p 'fix the bug'",
            'claude -p "fix the bug"',
            "claude --print --model claude-opus-5 'do it'",
            "claude   -p   spaced",
            "/usr/local/bin/claude",
            "/usr/local/bin/claude -p x",
            "/usr/bin/claude --print hi",
            "./claude",
            "./claude -p x",
            "../bin/claude",
            "~/.local/bin/claude -p x",
            "bin/claude",
        ],
    )
    def test_direct_invocations_are_blocked(self, command):
        assert _blocked(command) is True

    def test_the_reason_names_the_offending_token(self):
        verdict = terminal_guard.evaluate("/usr/local/bin/claude -p x")
        assert verdict["blocked"] is True
        assert "claude" in verdict["reason"]


class TestWrapperInvocations:
    @pytest.mark.parametrize(
        "command",
        [
            "env claude",
            "env claude -p x",
            "env FOO=1 claude",
            "env FOO=1 BAR=2 claude -p x",
            "sudo claude",
            "sudo -n claude",
            "sudo -u root claude -p x",
            "doas claude",
            "command claude",
            "command -v claude -p x",
            "builtin claude",
            "exec claude",
            "nohup claude -p x",
            "setsid claude",
            "stdbuf -o0 claude",
            "nice claude",
            "nice -n 5 claude",
            "ionice -c 3 claude -p x",
            "time claude",
            "timeout 300 claude",
            "timeout 30s claude -p x",
            "runuser -u worker claude",
            "/usr/bin/env claude",
            "/usr/bin/sudo /usr/local/bin/claude -p x",
        ],
    )
    def test_wrapped_invocations_are_blocked(self, command):
        assert _blocked(command) is True

    def test_leading_env_assignments_are_not_the_command(self):
        assert _blocked("FOO=bar claude") is True
        assert _blocked("FOO=bar BAZ=qux claude -p x") is True

    def test_a_value_taking_flag_does_not_hide_the_real_head(self):
        """Without the value-consuming flag table, ``sudo -u root claude``
        would stop scanning at ``root`` and miss the invocation entirely."""
        assert _blocked("sudo -u root claude -p x") is True
        assert _blocked("sudo --user root claude") is True
        assert _blocked("timeout --signal KILL 30 claude") is True


class TestShellChainedInvocations:
    @pytest.mark.parametrize(
        "command",
        [
            "make build && claude -p x",
            "make build; claude",
            "make build || claude -p x",
            "cat prompt.txt | claude -p -",
            "make build & claude",
            "true && env FOO=1 claude -p x",
            "cd /repo && sudo -n claude",
            "make&&claude",
            "make;claude",
            "make|claude",
            "echo hi\nclaude -p x",
        ],
    )
    def test_any_segment_that_invokes_claude_blocks(self, command):
        assert _blocked(command) is True

    def test_the_invocation_may_be_the_last_of_many_segments(self):
        assert _blocked("a; b; c; d && claude") is True


class TestNewlineSeparatedCommands:
    """A newline ends a command exactly as ``;`` does. shlex discards it by
    default, which merged a second command into the first segment and left it
    unexamined."""

    @pytest.mark.parametrize(
        "command",
        [
            "make build\nclaude -p x",
            "cd /repo\nclaude",
            "make build\n\nclaude -p x",
            "make build\r\nclaude -p x",
            "make build &&\nclaude -p x",
            "make build;\nclaude",
            "make build\nsudo -n claude",
            "make build\nenv FOO=1 claude -p x",
            "make build\n/usr/local/bin/claude",
            "make # a note\nclaude -p x",
            "claude -p x\nmake build",
        ],
    )
    def test_a_claude_invocation_on_any_line_blocks(self, command):
        assert _blocked(command) is True

    @pytest.mark.parametrize(
        "command",
        [
            "make build\npytest -q",
            "cd /repo\ngit status\ngit diff",
            'git commit -m "first line\nmentions claude"',
            "echo 'claude\nclaude'",
            "make # claude\npytest -q",
        ],
    )
    def test_multi_line_work_that_does_not_run_claude_is_allowed(self, command):
        assert _blocked(command) is False

    def test_a_newline_inside_quotes_is_still_one_word(self):
        tokens = terminal_guard._tokenize('git commit -m "a\nb"')
        assert tokens == ["git", "commit", "-m", "a\nb"]

    def test_a_bare_newline_is_emitted_as_a_separator(self):
        assert terminal_guard._is_segment_separator("\n") is True
        assert terminal_guard._is_segment_separator("&&\n") is True
        assert terminal_guard._is_segment_separator(";\n") is True
        assert terminal_guard._is_segment_separator("claude") is False
        assert terminal_guard._is_segment_separator(">") is False
        assert terminal_guard._is_segment_separator("") is False

    def test_a_redirection_does_not_split_a_segment(self):
        """``echo hi > claude`` writes a file named claude; it does not run
        one. Splitting on ``>`` would turn that into a spurious block."""
        assert _blocked("echo hi > claude") is False
        assert _blocked("pytest -q > claude") is False


class TestShellWrappedPayloads:
    @pytest.mark.parametrize(
        "command",
        [
            'bash -c "claude -p x"',
            "sh -c 'claude'",
            'zsh -c "claude -p x"',
            "dash -c 'claude'",
            'bash -lc "claude -p x"',
            'bash -c"claude -p x"',
            "sh -c 'make && claude -p x'",
            'bash -c "env FOO=1 claude"',
            "sudo bash -c 'claude -p x'",
        ],
    )
    def test_payloads_smuggled_through_an_explicit_shell_are_blocked(self, command):
        assert _blocked(command) is True

    def test_nested_shells_are_followed_to_the_bounded_depth(self):
        assert _blocked('bash -c "sh -c \'claude -p x\'"') is True
        assert _blocked('bash -c "sh -c \'sh -c \\"claude -p x\\"\'"') is True

    def test_the_reason_records_that_it_was_shell_wrapped(self):
        verdict = terminal_guard.evaluate('bash -c "claude -p x"')
        assert verdict["blocked"] is True
        assert "shell-wrapped" in verdict["reason"]


class TestNestingDepthBoundFailsClosed:
    """One level past ``MAX_SHELL_DEPTH`` used to be a free pass.

    Recursion is bounded so a hostile nest cannot spin the guard forever, but
    the bound returned ``None`` — ALLOW — so an attacker only had to stack
    enough layers to push the invocation past it. Every command in the first
    group has a lexically visible head-position ``claude`` at the bottom and
    every one of them was allowed. The bound itself is unchanged; reaching it
    now runs the fail-closed raw scan instead of giving up.
    """

    @pytest.mark.parametrize(
        "command",
        [
            # four shell layers: the innermost payload lands ON the bound
            'bash -c "bash -c \'bash -c \\"bash -c claude\\"\'"',
            'sh -c "sh -c \'sh -c \\"sh -c claude -p x\\"\'"',
            'bash -c "sh -c \'bash -c \\"sh -c /usr/local/bin/claude\\"\'"',
            # a wrapper spends a level of its own, so three shells reach the
            # bound: env -S and busybox applet dispatch both do this.
            'env -S "sh -c \'sh -c \\"sh -c claude\\"\'"',
            'env --split-string="sh -c \'sh -c \\"sh -c claude -p x\\"\'"',
            'busybox sh -c "sh -c \'sh -c \\"claude -p x\\"\'"',
            'busybox sh -c "sh -c \'sh -c \\"/usr/local/bin/claude\\"\'"',
        ],
    )
    def test_an_invocation_past_the_bound_is_blocked(self, command):
        assert _blocked(command) is True

    def test_the_reason_says_the_bound_was_what_stopped_the_parse(self):
        verdict = terminal_guard.evaluate(
            'bash -c "bash -c \'bash -c \\"bash -c claude\\"\'"'
        )
        assert verdict["blocked"] is True
        assert "nesting depth exceeded" in verdict["reason"]
        assert verdict["malformed"] is False

    @pytest.mark.parametrize(
        "command",
        [
            # the same depth, same shapes, no claude anywhere
            'bash -c "bash -c \'bash -c \\"bash -c echo true\\"\'"',
            'sh -c "sh -c \'sh -c \\"sh -c make build\\"\'"',
            'env -S "sh -c \'sh -c \\"sh -c echo true\\"\'"',
            'busybox sh -c "sh -c \'sh -c \\"echo true\\"\'"',
            # a lookalike at the bound is not the CLI
            'bash -c "bash -c \'bash -c \\"bash -c myclaude\\"\'"',
            'bash -c "bash -c \'bash -c \\"bash -c claudex -p x\\"\'"',
        ],
    )
    def test_depth_alone_does_not_block(self, command):
        """The bound blocks only when the raw scan still SEES an invocation.
        Deep nesting on its own is not evidence of one, so a nest of the same
        shape carrying ordinary work stays allowed."""
        assert _blocked(command) is False

    def test_the_scan_at_the_bound_is_cruder_than_the_parser_it_replaces(self):
        """Named rather than discovered later: past the bound there is no
        head-position analysis left, so a mere MENTION matches the raw scan
        and blocks. Inside the bound the same mention is correctly allowed.
        This is the documented fail-closed trade, not head-position parsing
        that reaches arbitrarily deep."""
        assert _blocked("busybox sh -c \"sh -c 'grep -r claude .'\"") is False
        assert _blocked(
            'busybox sh -c "sh -c \'sh -c \\"grep -r claude .\\"\'"'
        ) is True

    def test_the_bound_itself_is_unchanged(self):
        assert terminal_guard.MAX_SHELL_DEPTH == 3


class TestNestingDepthBoundInDispatchers:
    """The same bound, reached through the two OTHER recursive dispatchers.

    ``_nested_reason`` was made to fail closed at ``MAX_SHELL_DEPTH``, but the
    busybox applet scan and the ``find`` exec-primary scan kept their original
    ending — ``return None``, ALLOW — so the hole simply moved: spend the
    depth budget on shells and dispatch the invocation with the last token
    (``sh -c "sh -c 'sh -c \\"busybox env claude\\"'"``,
    ``… \\"find . -exec claude -p x ;\\"``) and nothing looked at it. Every
    command in the first two groups has a lexically visible invocation at the
    bottom and every one of them was allowed.

    The bound is untouched; reaching it runs the fail-closed raw scan over the
    text that would actually be EXECUTED there — the applet tail, and each
    exec primary's own argument slice.
    """

    BOUND_REASON = "direct Claude CLI invocation (nesting depth exceeded)"

    @pytest.mark.parametrize(
        "command",
        [
            # three shell layers put the busybox dispatch ON the bound
            'sh -c "sh -c \'sh -c \\"busybox env claude -p x\\"\'"',
            'sh -c "sh -c \'sh -c \\"busybox env claude\\"\'"',
            'sh -c "sh -c \'sh -c \\"busybox sh -c claude\\"\'"',
            'sh -c "sh -c \'sh -c \\"busybox env /usr/local/bin/claude\\"\'"',
            'bash -c "sh -c \'bash -c \\"busybox timeout 30 claude\\"\'"',
            # busybox spending a level of its own gets there just as well
            'busybox sh -c "sh -c \'busybox env claude\'"',
        ],
    )
    def test_busybox_applet_dispatch_at_the_bound_is_blocked(self, command):
        assert _blocked(command) is True

    @pytest.mark.parametrize(
        "command",
        [
            'sh -c "sh -c \'sh -c \\"find . -exec claude -p x ;\\"\'"',
            'sh -c "sh -c \'sh -c \\"find . -exec claude ;\\"\'"',
            'sh -c "sh -c \'sh -c \\"find . -execdir /usr/local/bin/claude -p x ;\\"\'"',
            'sh -c "sh -c \'sh -c \\"find . -type f -exec claude -p {} +\\"\'"',
            'sh -c "sh -c \'sh -c \\"find . -ok claude ;\\"\'"',
            'sh -c "sh -c \'sh -c \\"find . -okdir claude ;\\"\'"',
            # the invocation is in the SECOND exec primary of the expression
            'sh -c "sh -c \'sh -c \\"find . -exec echo {} + -exec claude -p x ;\\"\'"',
        ],
    )
    def test_find_exec_primaries_at_the_bound_are_blocked(self, command):
        assert _blocked(command) is True

    def test_the_reason_is_the_bounds_own_stable_string(self):
        for command in (
            'sh -c "sh -c \'sh -c \\"busybox env claude\\"\'"',
            'sh -c "sh -c \'sh -c \\"find . -exec claude -p x ;\\"\'"',
        ):
            verdict = terminal_guard.evaluate(command)
            assert verdict["blocked"] is True
            assert verdict["reason"].endswith(self.BOUND_REASON)
            assert verdict["malformed"] is False

    def test_all_three_dispatchers_report_the_same_bound_reason(self):
        """One string for one condition: a reason that drifted per dispatcher
        would read as three different findings in the audit log."""
        depth = terminal_guard.MAX_SHELL_DEPTH
        assert terminal_guard._nested_reason("claude -p x", depth) == self.BOUND_REASON
        assert terminal_guard._busybox_reason(
            ["busybox", "env", "claude"], 0, "busybox", depth,
        ) == self.BOUND_REASON
        assert terminal_guard._find_exec_reason(
            [".", "-exec", "claude", "-p", "x"], depth,
        ) == self.BOUND_REASON

    @pytest.mark.parametrize(
        "command",
        [
            # the same shapes at the same depth, carrying ordinary work
            'sh -c "sh -c \'sh -c \\"busybox env pytest -q\\"\'"',
            'sh -c "sh -c \'sh -c \\"busybox echo hi\\"\'"',
            'sh -c "sh -c \'sh -c \\"find . -type f -exec chmod 644 {} +\\"\'"',
            'sh -c "sh -c \'sh -c \\"find . -newer a -delete\\"\'"',
            # a filename TEST is not a command, at the bound as anywhere else
            'sh -c "sh -c \'sh -c \\"find . -name claude\\"\'"',
            'sh -c "sh -c \'sh -c \\"find / -type f -name claude -print\\"\'"',
            'sh -c "sh -c \'sh -c \\"find . -name claude -exec chmod 644 {} +\\"\'"',
            # text BEFORE the applet is not the command busybox would run
            'sh -c "sh -c \'sh -c \\"env FOO=claude busybox echo hi\\"\'"',
        ],
    )
    def test_depth_alone_does_not_block_in_either_dispatcher(self, command):
        assert _blocked(command) is False

    def test_the_bound_scan_is_scoped_to_what_would_be_executed(self):
        """Directly, because the end-to-end nests above only show the verdict:
        busybox scans the applet TAIL and not the tokens in front of it, and
        ``find`` scans each exec primary's argument slice up to its ``;``/``+``
        and not the predicates around it."""
        depth = terminal_guard.MAX_SHELL_DEPTH

        assert terminal_guard._busybox_reason(
            ["busybox", "echo", "hi"], 0, "busybox", depth,
        ) is None
        assert terminal_guard._busybox_reason(
            ["busybox"], 0, "busybox", depth,
        ) is None
        # `env FOO=claude busybox echo hi`: the mention is in the prefix the
        # applet never sees, so joining the whole segment would over-block.
        assert terminal_guard._busybox_reason(
            ["env", "FOO=claude", "busybox", "echo", "hi"], 2, "busybox", depth,
        ) is None

        assert terminal_guard._find_exec_reason([".", "-name", "claude"], depth) is None
        assert terminal_guard._find_exec_reason([".", "-exec"], depth) is None
        # `+` ends the argument list: the `-name claude` after it is a test.
        assert terminal_guard._find_exec_reason(
            [".", "-exec", "chmod", "644", "{}", "+", "-name", "claude"], depth,
        ) is None
        assert terminal_guard._find_exec_reason(
            [".", "-exec", "echo", "{}", "+", "-exec", "claude", "+"], depth,
        ) == self.BOUND_REASON

    @pytest.mark.parametrize(
        "command",
        [
            "busybox echo claude",
            "sh -c 'busybox echo claude'",
            'sh -c "sh -c \'busybox echo claude\'"',
            "find . -name claude",
            "sh -c 'find . -name claude'",
            'sh -c "sh -c \'find . -name claude\'"',
            "find . -exec grep claude {} ;",
            "sh -c 'find . -exec grep claude {} ;'",
            'sh -c "sh -c \'find . -exec grep claude {} ;\'"',
        ],
    )
    def test_the_preserved_controls_stay_allowed_at_every_depth_that_parses(
        self, command,
    ):
        """Inside the bound the parser is intact, so an applet or an ``-exec``
        that merely MENTIONS claude is head-position analysed and allowed —
        ``echo`` and ``grep`` are the commands being run, not the CLI."""
        assert _blocked(command) is False

    def test_the_scan_at_the_bound_is_cruder_than_the_parser_it_replaces(self):
        """The same documented trade the shell bound already makes, named here
        for the two dispatchers rather than discovered later: ON the bound
        there is no head-position analysis left, so a mention inside the
        scanned slice matches the raw scan and blocks. One level in, the same
        command is parsed and allowed."""
        assert _blocked('sh -c "sh -c \'busybox echo claude\'"') is False
        assert _blocked('sh -c "sh -c \'sh -c \\"busybox echo claude\\"\'"') is True

        assert _blocked('sh -c "sh -c \'find . -exec grep claude {} ;\'"') is False
        assert _blocked(
            'sh -c "sh -c \'sh -c \\"find . -exec grep claude {} ;\\"\'"'
        ) is True

    def test_the_bound_is_still_what_ends_the_recursion(self, monkeypatch):
        """Fail-closed is not "recurse one more level": the bound is unchanged
        and neither dispatcher descends once it is reached, which is what keeps
        a hostile nest from spinning the guard forever."""
        assert terminal_guard.MAX_SHELL_DEPTH == 3

        def _must_not_recurse(segment, depth):  # pragma: no cover - must not run
            raise AssertionError(f"recursed past the bound into {segment!r}")

        monkeypatch.setattr(terminal_guard, "_segment_reason", _must_not_recurse)
        depth = terminal_guard.MAX_SHELL_DEPTH

        assert terminal_guard._busybox_reason(
            ["busybox", "sh", "-c", "echo hi"], 0, "busybox", depth,
        ) is None
        assert terminal_guard._busybox_reason(
            ["busybox", "sh", "-c", "claude"], 0, "busybox", depth,
        ) == self.BOUND_REASON
        assert terminal_guard._find_exec_reason(
            [".", "-exec", "grep", "hi", "{}", "+"], depth,
        ) is None
        assert terminal_guard._find_exec_reason(
            [".", "-exec", "claude", ";"], depth,
        ) == self.BOUND_REASON


class TestShellOptionSemantics:
    """``-c`` is not a getopt option with an attached argument. A shell reads
    ``c`` as "take the command from the first operand", so the rest of the
    cluster is MORE OPTIONS and the command string is the next argv word.
    Reading the cluster tail as the payload made ``bash -cx "claude -p x"``
    report a payload of ``"x"`` and never look at the real one; not skipping
    option values made ``bash -O extglob -c …`` stop at ``extglob``."""

    @pytest.mark.parametrize(
        "command",
        [
            'bash -cx "claude -p x"',
            'sh -cv "claude -p x"',
            'bash -ce "claude -p x"',
            'bash -cex "claude -p x"',
            'bash -xc "claude -p x"',
            "zsh -cv 'claude'",
        ],
    )
    def test_a_short_option_cluster_does_not_hide_the_command_string(self, command):
        assert _blocked(command) is True

    @pytest.mark.parametrize(
        "command",
        [
            'bash -O extglob -c "claude -p x"',
            'bash +O extglob -c "claude -p x"',
            'bash -o pipefail -c "claude -p x"',
            'bash --rcfile /dev/null -c "claude -p x"',
            'bash --init-file /dev/null -c "claude -p x"',
            'bash --noprofile --norc -c "claude -p x"',
            'bash --posix -c "claude -p x"',
            'bash --rcfile=/dev/null -c "claude -p x"',
            'bash -O extglob -O globstar -c "claude -p x"',
            'sh -x -c "claude -p x"',
            'bash -c -- "claude -p x"',
        ],
    )
    def test_shell_options_and_their_values_are_skipped_before_c(self, command):
        assert _blocked(command) is True

    def test_an_option_value_is_not_consumed_when_it_is_itself_an_option(self):
        """``-O`` takes a value, but consuming one unconditionally would eat
        the ``-c`` that follows it and hide the payload behind it."""
        assert _blocked('bash -O -c "claude -p x"') is True

    def test_the_command_string_is_the_next_word_not_the_cluster_tail(self):
        assert terminal_guard._shell_command_strings(["-cx", "claude -p x"]) == [
            "x", "claude -p x",
        ]
        assert terminal_guard._shell_command_strings(["-c", "claude -p x"]) == [
            "claude -p x",
        ]
        assert terminal_guard._shell_command_strings(
            ["-O", "extglob", "-c", "claude -p x"]
        ) == ["claude -p x"]
        assert terminal_guard._shell_command_strings(
            ["--rcfile", "/dev/null", "-c", "claude -p x"]
        ) == ["claude -p x"]
        assert terminal_guard._shell_command_strings(["-O", "-c", "claude"]) == ["claude"]

    def test_a_shell_with_no_c_has_no_command_string(self):
        assert terminal_guard._shell_command_strings(["script.sh"]) == []
        assert terminal_guard._shell_command_strings(["-x", "script.sh", "claude"]) == []
        assert terminal_guard._shell_command_strings(["-c"]) == []

    @pytest.mark.parametrize(
        "command",
        [
            "bash -c 'echo claude'",
            "bash -O extglob -c 'echo claude'",
            "bash -cx 'echo claude'",
            "bash --noprofile --norc -c 'pytest -q'",
            "bash --rcfile /dev/null -c 'grep -r claude .'",
            "bash script.sh",
            "bash -x deploy.sh claude",
            "sh -s < input.txt",
        ],
    )
    def test_ordinary_shell_invocations_are_untouched(self, command):
        assert _blocked(command) is False


class TestShellScriptFedAsText:
    """A here-string IS the shell's script, so its text is a command string
    exactly as a ``-c`` operand is — and it carries no ``-c``, so the payload
    scan never fired. A here-doc body needs no rule of its own: a newline
    already ends a segment, so its lines are scanned in head position."""

    @pytest.mark.parametrize(
        "command",
        [
            'bash <<< "claude -p x"',
            'sh <<< "/usr/local/bin/claude -p x"',
            "bash <<< 'claude'",
            "bash <<<claude",
            'bash -s <<< "claude -p x"',
            'zsh <<< "env FOO=1 claude"',
            'busybox sh <<< "claude -p x"',
            'make build && bash <<< "claude -p x"',
        ],
    )
    def test_a_here_string_is_a_command_string(self, command):
        assert _blocked(command) is True

    @pytest.mark.parametrize(
        "command",
        [
            "bash <<EOF\nclaude -p x\nEOF",
            "bash <<'EOF'\n/usr/local/bin/claude -p x\nEOF",
            "sh <<EOF\nmake build\nclaude -p x\nEOF",
        ],
    )
    def test_here_doc_command_text_is_visible_and_blocks(self, command):
        assert _blocked(command) is True

    def test_only_the_here_string_operator_yields_a_payload(self):
        assert terminal_guard._here_string_payloads(["<<<", "claude -p x"]) == [
            "claude -p x",
        ]
        assert terminal_guard._here_string_payloads(["<<", "EOF"]) == []
        assert terminal_guard._here_string_payloads(["<", "in.txt"]) == []
        assert terminal_guard._here_string_payloads(["<<<"]) == []

    @pytest.mark.parametrize(
        "command",
        [
            'echo \'bash <<< "claude -p x"\'',
            "cat <<< claude",
            "grep claude <<< 'text'",
            "bash <<< 'make build && pytest -q'",
            "bash < script.sh",
        ],
    )
    def test_a_quoted_or_non_shell_here_string_is_allowed(self, command):
        """``cat <<< claude`` prints a word, and a here-string quoted inside
        another command's argument is one token that is never a command."""
        assert _blocked(command) is False


class TestShellGrammarSegments:
    """Reserved words, grouping and function grammar head a segment without
    being the command being run, so the scan used to stop one token before
    claude: ``then claude -p x`` looked like a command named ``then``."""

    @pytest.mark.parametrize(
        "command",
        [
            "if true; then claude -p x; fi",
            "if claude -p x; then true; fi",
            "if false; then true; else claude -p x; fi",
            "if false; then true; elif true; then claude; fi",
            "while true; do claude -p x; done",
            "until false; do /usr/local/bin/claude -p x; done",
            "for f in a b; do claude -p $f; done",
            "! claude -p x",
            "{ claude -p x; }",
            "f(){ claude -p x; }; f",
            "function f { claude -p x; }; f",
            "case x in x) claude -p x;; esac",
            "if true; then sudo -n claude; fi",
            "while true; do bash -c 'claude -p x'; done",
            "make build && { claude -p x; }",
        ],
    )
    def test_grammar_before_the_command_does_not_end_the_scan(self, command):
        assert _blocked(command) is True

    def test_a_standalone_brace_is_grouping_not_a_command_or_argument(self):
        assert terminal_guard._is_segment_separator("{") is True
        assert terminal_guard._is_segment_separator("}") is True
        assert terminal_guard._is_segment_separator("{}") is False

    def test_the_reason_still_names_the_token(self):
        verdict = terminal_guard.evaluate("if true; then claude -p x; fi")
        assert verdict["blocked"] is True
        assert "claude" in verdict["reason"]


class TestShellGrammarLookalikesRemainAllowed:
    """A reserved word only counts in COMMAND HEAD position. As an argument
    it is an ordinary word, and treating every mention as grammar would block
    the commit messages and echoes these sessions write all day."""

    @pytest.mark.parametrize(
        "command",
        [
            'echo "then claude"',
            "printf 'do claude'",
            "git commit -m 'if claude'",
            "find . -name claude",
            "python3 -c 'print(\"claude\")'",
            "bash -c 'echo claude'",
            "bash -O extglob -c 'echo claude'",
            'echo \'bash <<< "claude -p x"\'',
        ],
    )
    def test_the_preserved_controls_stay_allowed(self, command):
        assert _blocked(command) is False

    @pytest.mark.parametrize(
        "command",
        [
            "echo done claude",
            "git commit -m 'while claude runs'",
            "grep -rn 'then claude' .",
            "if grep -q claude file; then make; fi",
            "while read f; do echo claude; done",
            "for claude in a b; do echo $claude; done",
            "case claude in x) echo hi;; esac",
            "function claude { echo hi; }",
            "select claude in a b; do echo hi; done",
            "if true; then pytest -q; fi",
            "{ make build; pytest -q; }",
        ],
    )
    def test_grammar_around_a_mention_is_not_an_invocation(self, command):
        """``for claude in …`` names a loop VARIABLE and ``function claude``
        defines one; neither runs the CLI, which is why the word after a
        naming keyword is skipped rather than read as a head."""
        assert _blocked(command) is False


class TestDirectExecutorWrappers:
    """Commands that take ANOTHER command as data. Every shape here was found
    by adversarial testing to reach a host claude while the guard allowed it:
    the invocation is not in head position of the segment, it is an argument
    the executor later runs."""

    @pytest.mark.parametrize(
        "command",
        [
            'env -S "claude -p x"',
            "env -S 'claude -p x'",
            'env -S"claude -p x"',
            'env --split-string="claude -p x"',
            'env --split-string "claude -p x"',
            'env -S "make build && claude -p x"',
            'env -i -S "claude -p x"',
            'env -S "env FOO=1 claude"',
            '/usr/bin/env -S "claude -p x"',
            'sudo env -S "claude -p x"',
        ],
    )
    def test_gnu_env_split_string_is_a_whole_command_string(self, command):
        """``env -S`` exists precisely to run a command string from a single
        argument, so the invocation never appears as its own token."""
        assert _blocked(command) is True

    def test_env_ignore_environment_does_not_swallow_the_head(self):
        """``env -i`` takes no value; consuming the next token as one hid
        ``claude`` behind it."""
        assert _blocked("env -i claude -p x") is True
        assert _blocked("env -i -u FOO claude") is True

    @pytest.mark.parametrize(
        "command",
        [
            'busybox sh -c "claude -p x"',
            "busybox sh -c 'claude'",
            "busybox ash -c 'claude -p x'",
            "busybox env claude -p x",
            "busybox env FOO=1 claude",
            "busybox timeout 30 claude",
            "/bin/busybox sh -c 'claude -p x'",
            "sudo busybox sh -c 'claude -p x'",
        ],
    )
    def test_busybox_dispatches_on_its_applet_argument(self, command):
        """busybox's applet name sits where a shell's flags would be, so the
        ``-c`` scan never fired and the payload went unexamined."""
        assert _blocked(command) is True

    @pytest.mark.parametrize(
        "command",
        [
            "xargs claude -p x",
            "printf x | xargs claude -p",
            "echo x | xargs -n 1 claude -p",
            "xargs -0 claude",
            "xargs -I {} claude -p {}",
            "xargs -i claude -p",
            "xargs -p claude",
            "find . -name '*.py' | xargs claude -p",
            "cat list.txt | xargs -P 4 /usr/local/bin/claude -p",
            "xargs -a list.txt claude",
        ],
    )
    def test_xargs_runs_its_argument_list(self, command):
        assert _blocked(command) is True

    @pytest.mark.parametrize(
        "command",
        [
            "watch claude -p x",
            "watch -n 5 claude -p x",
            "watch -n1 claude",
            "watch -c claude",
            'watch "claude -p x"',
            "watch 'make build && claude -p x'",
            "parallel claude -p x ::: 1 2",
            "parallel -j 4 claude -p",
            "parallel -k claude",
            'parallel "claude -p {}" ::: a b',
        ],
    )
    def test_watch_and_parallel_run_their_argument_list(self, command):
        """Both hand the remaining arguments to a shell, so a single quoted
        argument is a command string in its own right."""
        assert _blocked(command) is True

    @pytest.mark.parametrize(
        "command",
        [
            "find . -exec claude -p x ;",
            "find . -exec claude -p x \\;",
            "find . -execdir /usr/local/bin/claude -p x ;",
            "find . -execdir claude -p x \\;",
            "find . -name '*.py' -exec claude -p {} +",
            "find . -ok claude -p x ;",
            "find . -okdir claude ;",
            "find . -type f -exec claude -p {} \\;",
            "find . -exec echo {} \\; -exec claude -p x \\;",
            "find . -exec sh -c 'claude -p x' \\;",
            "sudo find /repo -exec claude ;",
        ],
    )
    def test_find_exec_primaries_run_a_command(self, command):
        assert _blocked(command) is True

    def test_the_reason_names_the_executor_that_would_run_it(self):
        assert "env split-string" in terminal_guard.evaluate('env -S "claude -p x"')["reason"]
        assert "busybox applet" in terminal_guard.evaluate("busybox env claude")["reason"]
        assert "-exec" in terminal_guard.evaluate("find . -exec claude ;")["reason"]
        assert "watch-wrapped" in terminal_guard.evaluate('watch "claude -p x"')["reason"]


class TestExecutableSubstitutions:
    """The shell runs a substitution as its own command before (or alongside)
    the one containing it, so an innocent head is no defence: ``echo
    `claude -p x` `` runs claude and prints its output."""

    @pytest.mark.parametrize(
        "command",
        [
            "echo `claude -p x`",
            "echo `claude`",
            'echo "`claude -p x`"',
            "echo `sudo -n claude`",
            "echo `/usr/local/bin/claude -p x`",
            "echo $(claude -p x)",
            'echo "$(claude -p x)"',
            'git commit -m "$(claude -p x)"',
            "cat <(claude -p x)",
            "cat >(claude -p x)",
            "diff <(claude -p a) <(cat b)",
            "tee >(claude -p x) < in.txt",
            "make build && echo $(claude -p x)",
        ],
    )
    def test_substitutions_are_analysed_as_commands(self, command):
        assert _blocked(command) is True

    @pytest.mark.parametrize(
        "command",
        [
            "echo $(echo $(claude -p x))",
            "echo `echo $(claude -p x)`",
            'bash -c "echo $(claude -p x)"',
            "cat <(bash -c 'claude -p x')",
        ],
    )
    def test_nested_substitutions_are_followed_to_the_bounded_depth(self, command):
        assert _blocked(command) is True

    def test_an_unterminated_substitution_is_still_inspected(self):
        """The shell would reject it, but a shape this module cannot parse is
        looked at rather than waved through."""
        assert _blocked("echo `claude -p x") is True
        assert _blocked("echo $(claude -p x") is True

    def test_the_reason_records_that_it_was_a_substitution(self):
        verdict = terminal_guard.evaluate("cat <(claude -p x)")
        assert verdict["blocked"] is True
        assert "substitution" in verdict["reason"]

    @pytest.mark.parametrize(
        "command",
        [
            "echo '$(claude -p x)'",
            "echo '`claude -p x`'",
            "grep -r '$(claude)' .",
            "grep -rn '<(claude' .",
            'git commit -m "$(date) claude fix"',
            "echo $(date)",
            "echo $((1 + 2))",
        ],
    )
    def test_literal_and_harmless_substitutions_are_allowed(self, command):
        """Single quotes suppress substitution — ``echo '$(claude)'`` prints a
        string and runs nothing — and a substitution that does not invoke
        claude is ordinary work."""
        assert _blocked(command) is False

    def test_quoting_decides_which_substitutions_are_live(self):
        assert terminal_guard._substitution_payloads("echo '$(claude)'") == []
        assert terminal_guard._substitution_payloads('echo "$(claude)"') == ["claude"]
        assert terminal_guard._substitution_payloads("echo `claude -p x`") == ["claude -p x"]
        assert terminal_guard._substitution_payloads("cat <(claude) >(tee f)") == [
            "claude", "tee f",
        ]
        assert terminal_guard._substitution_payloads("echo hi > claude") == []


class TestExecutorLookalikesRemainAllowed:
    """The executor and substitution rules must not swallow the ordinary uses
    of the very same tools."""

    @pytest.mark.parametrize(
        "command",
        [
            "find . -name claude",
            "find . -name claude -delete",
            "find / -type f -name claude -print",
            "find . -exec grep claude {} \\;",
            "find . -type f -exec chmod 644 {} +",
            "xargs grep claude",
            "grep -rl claude . | xargs sed -i 's/claude/x/'",
            "printf x | xargs echo claude",
            "xargs -I {} echo claude {}",
            "watch date",
            "watch -n 5 'git status'",
            "parallel echo claude ::: a b",
            "busybox ls /usr/local/bin/claude",
            "busybox sh -c 'make build && pytest -q'",
            "env -S 'echo claude'",
            "env -i pytest -q",
            "env --split-string='grep -r claude .'",
        ],
    )
    def test_ordinary_use_of_an_executor_is_untouched(self, command):
        assert _blocked(command) is False


class TestDocumentedScopeLimits:
    """What this guard does NOT claim. It reads one command string
    lexically, so a command that BUILDS its target at runtime has no
    ``claude`` token for any parser here to find. Widening the guard to guess
    at that would block ordinary work without closing the hole; the sandbox
    and the credential boundary are what contain it. These assertions pin the
    boundary honestly rather than implying coverage that does not exist."""

    @pytest.mark.parametrize(
        "command",
        [
            "python3 -c 'import os; os.execvp(\"cla\" + \"ude\", [\"claude\"])'",
            "python3 -c 'import base64, os; os.system(base64.b64decode(\"Y2xhdWRl\"))'",
            "./run-the-agent.sh",
        ],
    )
    def test_runtime_constructed_invocations_are_out_of_lexical_reach(self, command):
        assert _blocked(command) is False

    @pytest.mark.parametrize(
        "command",
        [
            'echo "claude -p x" | bash',
            "printf 'claude -p x' | sh",
            "bash < script.sh",
            "bash script.sh",
        ],
    )
    def test_a_script_reaching_a_shell_by_pipe_or_file_is_not_analysed(self, command):
        """The here-string rule covers a script the shell is handed as TEXT in
        this command string. A script arriving on a pipe is not analysed —
        the module splits on ``|`` without modelling which side feeds which —
        and one arriving from a file has no command text here at all. Both
        are named in the README rather than implied to be covered."""
        assert _blocked(command) is False


class TestOrdinaryCommandsRemainAllowed:
    """Blocking too much would break the build/test/git work these sessions
    exist to do. Every command here MENTIONS claude and none of them runs
    it."""

    @pytest.mark.parametrize(
        "command",
        [
            'git commit -m "claude worker fix"',
            "git commit -m 'claude worker fix'",
            'git commit -m "wire up claude"',
            "git log --grep claude",
            "grep -r claude .",
            "grep -rn 'claude -p' plugins/",
            "rg claude plugins/claude-worker",
            "pytest tests/test_claude_worker_gate.py",
            "pytest tests/plugins/test_claude_worker_gate.py",
            "pytest -q tests/plugins -k claude",
            "printf 'claude'",
            "printf claude",
            "find . -name claude",
            "ls /usr/local/bin/claude",
            "ls /usr/bin/claude",
            "ls -la ~/.claude",
            "cat plugins/claude-worker/README.md",
            "echo claude",
            "echo 'claude -p x'",
            "which claude",
            "type claude",
            "file /usr/local/bin/claude",
            "stat /usr/local/bin/claude",
            "npm run claude",
            "make claude",
            "./scripts/claude-helper.sh",
            "python -c 'print(\"claude\")'",
            "python3 -c 'print(\"claude\")'",
            "docker build -t claude-worker-sandbox:2.1.237 .",
        ],
    )
    def test_mentioning_claude_is_not_invoking_it(self, command):
        assert _blocked(command) is False

    @pytest.mark.parametrize(
        "command",
        [
            "make build",
            "make build && make test",
            "pytest -q",
            "pytest -q && ruff check .",
            "npm ci && npm test",
            "cargo build --release",
            "git status --porcelain",
            "git add -A && git commit -m 'wip' && git push",
            "ls -la",
            "cd /repo && ./gradlew test",
            "bash -c 'make build && pytest -q'",
            "sudo -n systemctl restart hermes",
            "timeout 300 pytest -q",
            "env FOO=1 pytest -q",
        ],
    )
    def test_ordinary_build_test_and_git_commands_are_untouched(self, command):
        assert _blocked(command) is False

    @pytest.mark.parametrize(
        "token", ["myclaude", "claudex", "not-claude", "claude_worker", "clauded"],
    )
    def test_lookalike_command_names_are_not_the_cli(self, token):
        assert _blocked(f"{token} -p x") is False

    @pytest.mark.parametrize("command", ["", "   ", None, 7, [], {}])
    def test_empty_or_non_string_commands_are_not_invocations(self, command):
        assert _blocked(command) is False


class TestMalformedCommandsFailClosed:
    def test_unbalanced_quoting_hiding_a_claude_invocation_blocks(self):
        verdict = terminal_guard.evaluate('claude -p "unterminated')
        assert verdict["blocked"] is True
        assert verdict["malformed"] is True

    @pytest.mark.parametrize(
        "command",
        [
            "claude -p 'unterminated",
            'sudo claude -p "unterminated',
            '/usr/local/bin/claude "oops',
            'make && claude "oops',
        ],
    )
    def test_every_malformed_shape_with_a_visible_invocation_blocks(self, command):
        assert _blocked(command) is True

    def test_a_malformed_command_with_no_claude_is_left_alone(self):
        """The shell will reject it on its own; blocking here would be noise."""
        verdict = terminal_guard.evaluate('echo "unterminated')
        assert verdict["blocked"] is False
        assert verdict["malformed"] is True

    def test_an_absolute_path_invocation_is_visible_to_the_raw_scan(self):
        """The raw scan's path group can only consume ``dir/`` repetitions,
        so without an explicit leading-slash allowance a malformed
        ``/usr/local/bin/claude "oops`` scanned clean and sailed through."""
        assert terminal_guard._RAW_CLAUDE.search('/usr/local/bin/claude "oops') is not None
        assert terminal_guard._RAW_CLAUDE.search('sudo /usr/bin/claude "oops') is not None

    @pytest.mark.parametrize(
        "command",
        [
            'echo "myclaudething',
            'echo "/usr/bin/claudex',
            'echo "--claude-flag',
            'echo "claudette',
        ],
    )
    def test_the_raw_scan_does_not_match_a_substring(self, command):
        assert terminal_guard.evaluate(command)["blocked"] is False

    def test_internal_errors_degrade_to_the_fail_closed_scan(self, monkeypatch):
        def _boom(command, depth=0):
            raise RuntimeError("guard internals exploded")

        monkeypatch.setattr(terminal_guard, "_command_reason", _boom)
        assert terminal_guard.evaluate("claude -p x")["blocked"] is True
        assert terminal_guard.evaluate("make build")["blocked"] is False

    def test_evaluate_never_raises(self, monkeypatch):
        class _Hostile:
            def __str__(self):  # pragma: no cover - must never be reached
                raise RuntimeError("nope")

        for command in (_Hostile(), object(), b"claude", 3.14):
            assert terminal_guard.evaluate(command)["blocked"] is False


class TestTokenPredicate:
    @pytest.mark.parametrize(
        "token",
        ["claude", "/usr/local/bin/claude", "./claude", "../claude", "~/bin/claude", "a/b/c/claude"],
    )
    def test_tokens_that_name_the_cli(self, token):
        assert terminal_guard.is_claude_invocation_token(token) is True

    @pytest.mark.parametrize(
        "token",
        ["", None, 7, "claude-code", "myclaude", "claude.md", "/usr/bin/claudex",
         "--claude", "CLAUDE", "/usr/bin/CLAUDE"],
    )
    def test_tokens_that_do_not(self, token):
        assert terminal_guard.is_claude_invocation_token(token) is False

    def test_the_basename_comes_from_policy_not_a_local_literal(self):
        assert policy.CLAUDE_CLI_BASENAME == "claude"
        assert terminal_guard.is_claude_invocation_token(policy.CLAUDE_CLI_BASENAME) is True
