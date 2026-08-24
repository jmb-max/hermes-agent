"""Close the terminal bypass: a Discord-origin session must not reach a host
Claude CLI through the ``terminal`` tool.

The write gate (``gate.py``) makes ``patch``/``write_file``/``skill_manage``
go through ``claude_worker`` — but ``terminal`` takes a free-form command
string, so ``claude -p "edit the file"`` routed around the entire sandbox:
no image pin, no non-root uid, no empty MCP config, no tool allowlist, no
resource caps, and full host filesystem access. The canonical path to Claude
from a Discord session is the ``claude_worker`` tool; a direct host
invocation is not an alternative to it.

This module answers exactly one question about a command string: *does it
directly invoke the Claude CLI?* It is deliberately narrow. Blocking too
much would break the ordinary build/test/git work these sessions exist to
do, so a token only counts when it is in COMMAND HEAD position — the thing
being executed. ``git commit -m "claude worker fix"``, ``grep -r claude .``,
``find . -name claude``, ``pytest tests/test_claude_worker_gate.py`` and
``ls /usr/bin/claude`` all mention claude and are all fine; none of them
runs it.

Shapes that ARE a direct invocation:

* the bare command ``claude`` (with or without ``-p``/other flags);
* any path whose final component is ``claude`` — ``/usr/local/bin/claude``,
  ``./claude``, ``~/.local/bin/claude``;
* a wrapper followed by either of the above — ``env claude``,
  ``env FOO=1 claude``, ``env -i claude``, ``sudo -n claude``,
  ``command claude``, ``nohup claude``, ``timeout 300 claude``,
  ``exec claude``, ``nice -n 5 claude``, …;
* an EXECUTOR that takes a command in its argument list — ``xargs claude -p
  x``, ``watch claude -p x``, ``parallel claude -p x ::: 1``,
  ``find . -exec claude -p x \\;``, ``find . -execdir /usr/local/bin/claude
  -p x \\;``. ``watch``/``parallel`` hand their remaining arguments to a
  shell, so a single quoted argument (``watch "claude -p x"``) is analysed
  as a command string too;
* GNU ``env``'s split-string form, which is a command string in one
  argument — ``env -S "claude -p x"``, ``env --split-string="claude -p x"``;
* busybox applet dispatch, where the real command starts one token to the
  right — ``busybox sh -c "claude -p x"``, ``busybox env claude -p x``;
* any segment of a shell-chained command — ``make build && claude -p x``,
  ``x; claude``, ``a | claude``, ``printf x | xargs claude -p``, and a
  NEWLINE-separated command (``make build`` / newline / ``claude -p x``),
  which ends a command exactly as ``;`` does;
* a payload smuggled through an explicit shell — ``bash -c "claude -p x"``
  (recursively, to a bounded depth). The ``-c`` scan follows real shell
  option parsing rather than getopt-style guessing: the command string is
  the next argv WORD, so it is still found behind an option cluster
  (``bash -cx``, ``sh -cv``, ``bash -ce``) and behind options that take a
  value of their own (``bash -O extglob -c …``, ``bash --rcfile /dev/null
  -c …``, ``bash --noprofile --norc -c …``);
* a script handed to a shell on stdin when the script TEXT is right there in
  the command string — the here-string ``bash <<< "claude -p x"``, and a
  here-doc body, whose lines are ordinary command lines to this module
  because a newline already ends a segment;
* a segment whose head is shell GRAMMAR rather than a command, which used to
  stop the scan one token before claude — ``if true; then claude -p x; fi``,
  ``if claude -p x; then true; fi``, ``while true; do claude -p x; done``,
  ``until false; do claude -p x; done``, ``! claude -p x``,
  ``{ claude -p x; }``, ``f(){ claude -p x; }; f``,
  ``case x in x) claude -p x;; esac``. Reserved words are skipped only in
  COMMAND HEAD position, so ``echo "then claude"``, ``printf 'do claude'``
  and ``git commit -m 'if claude'`` stay arguments and stay allowed;
* an executable substitution, which the shell runs as its own command before
  the outer one ever starts — ``echo `claude -p x` ``, ``echo $(claude -p
  x)``, ``cat <(claude -p x)``, ``cat >(claude -p x)``. These are scanned
  quote-aware and recursively, also to a bounded depth: text inside SINGLE
  quotes is literal (``echo '$(claude)'`` prints a string, it runs nothing),
  while ``$(…)`` and backticks inside double quotes do execute.

Fail-closed on malformed shapes: a command string whose quoting does not
tokenize (an unbalanced quote) cannot be analysed segment by segment, so if
a raw scan still shows what looks like a head-position claude invocation, it
is blocked rather than passed through on the grounds that the parse failed.
A malformed command with no sign of claude in it is left alone — the shell
will reject it on its own, and blocking it here would just be noise.

Fail-closed at the NESTING BOUND, for the same reason. Recursion through
``-c`` payloads, ``env -S`` strings, busybox applets, ``find -exec`` argument
lists and substitutions stops at :data:`MAX_SHELL_DEPTH`, and stopping used
to mean allowing: ``bash -c "bash -c 'bash -c \\"bash -c claude\\"'"`` put the
invocation one level past the bound and was passed through. The bound is
unchanged — it is what keeps a hostile nest bounded — but reaching it now
runs the same raw scan, so a lexically visible head-position invocation
blocks. Text at the bound with no such invocation is still allowed, so depth
alone never blocks.

Every place that stops at the bound scans only the text that would actually
be EXECUTED there: the busybox applet tail rather than the tokens in front of
it, and each ``find`` exec primary's own argument slice up to its ``;``/``+``
rather than the surrounding predicates — ``find . -name claude`` is a
filename test at the bound exactly as it is at depth zero.

**Scope, stated honestly.** This is a lexical guard over one command string.
It blocks direct invocations and the common shell-native wrappers, executors
and substitutions enumerated above. It is NOT a sandbox and does not claim to
be one: a command that BUILDS its target at runtime — an interpreter decoding
an encoded string and calling ``exec``, a shell variable expanded to the
binary's name, a script file on disk that runs claude itself — contains no
lexical ``claude`` for any parser in this module to see, and no amount of
widening would find it without blocking ordinary work. Containment for that
case is the sandbox and the credential boundary, not this file.

Two neighbours of the stdin rule above are named rather than implied: a
shell reading a script from a FILE (``bash script.sh``, ``bash < script.sh``)
has no command text in the string at all, and a script arriving through a
PIPE (``echo "claude -p x" | bash``) is not analysed either — this module
splits on ``|`` without modelling which side feeds which, so the producer's
arguments are its own segment's arguments and nothing more. Neither is
claimed as covered.

:func:`evaluate` never raises: an internal error degrades to the same
fail-closed raw scan.
"""

from __future__ import annotations

import logging
import re
import shlex
from typing import Any, Dict, List, Optional, Tuple

from . import policy as _policy

logger = logging.getLogger(__name__)

#: Commands that run *another* command: the real head is further right. This
#: covers both plain wrappers (``env``, ``sudo``, ``nohup``, …) and executors
#: that take their command as ordinary arguments (``xargs``, ``watch``,
#: ``parallel``) — ``xargs claude -p x`` runs claude just as surely as
#: ``env claude -p x`` does.
WRAPPER_COMMANDS = frozenset({
    "env", "sudo", "doas", "command", "builtin", "exec", "nohup", "setsid",
    "stdbuf", "nice", "ionice", "time", "timeout", "chroot", "runuser",
    "xargs", "watch", "parallel",
})

#: Shells whose ``-c`` payload is itself a command string to analyse.
SHELL_COMMANDS = frozenset({"sh", "bash", "zsh", "dash", "ksh", "ash"})

#: Shell options that consume the NEXT word as their value, so that value is
#: not mistaken for the end of the option list. ``bash -O extglob -c "claude
#: -p x"`` and ``bash --rcfile /dev/null -c "claude -p x"`` both stopped the
#: ``-c`` scan dead at ``extglob``/``/dev/null`` and ran unexamined. Long
#: options with no value (``--noprofile``, ``--norc``, ``--posix``, …) need no
#: entry: any ``--`` token that is not here is skipped as a lone flag.
_SHELL_VALUE_TAKING_OPTIONS = frozenset({
    "-o", "+o", "-O", "+O", "--rcfile", "--init-file",
})

#: Shell reserved words that can HEAD a command segment without being the
#: command being run: ``then claude -p x`` runs claude, and so do ``do``,
#: ``else``, ``!`` and the rest. They are skipped only while scanning for the
#: head, so a reserved word appearing as an ARGUMENT (``echo "then claude"``,
#: ``printf 'do claude'``) is never reached and never reinterpreted.
_RESERVED_WORDS = frozenset({
    "if", "then", "elif", "else", "fi",
    "while", "until", "do", "done",
    "esac", "coproc", "!",
})

#: Reserved words followed by a NAME rather than by a command: the loop
#: variable of ``for``/``select``, the subject of ``case``, the name in
#: ``function``. Skipping the name too is what keeps ``for claude in a b``
#: and ``function claude { … }`` — a variable and a definition, not
#: invocations — from being read as a command head.
_RESERVED_WORDS_WITH_NAME = frozenset({"for", "select", "case", "function"})

#: Multi-call binaries that dispatch on their FIRST argument: ``busybox sh -c
#: '…'`` and ``busybox env claude`` are the shell and the wrapper, reached one
#: token to the right. Handled separately from :data:`SHELL_COMMANDS` because
#: the applet name sits where a shell's flags would be, so the ``-c`` scan
#: never fires and ``busybox sh -c "claude -p x"`` sailed through.
BUSYBOX_COMMANDS = frozenset({"busybox"})

#: Executors that hand their remaining arguments to a shell rather than
#: exec'ing them directly, so a single quoted argument is a whole command
#: string: ``watch "claude -p x"``, ``parallel "claude -p {}" ::: 1``.
SHELL_STRING_EXECUTORS = frozenset({"watch", "parallel"})

#: ``find`` primaries whose following arguments are a command to run, up to a
#: ``;`` or ``+`` terminator. ``-name claude`` is a filename test and stays
#: allowed; ``-exec claude`` is an invocation.
FIND_COMMANDS = frozenset({"find"})
FIND_EXEC_PRIMARIES = frozenset({"-exec", "-execdir", "-ok", "-okdir"})

#: Ends the argument list of a ``find`` exec primary. ``;`` is normally split
#: off as a segment separator before this is consulted; ``+`` is an ordinary
#: word and is not.
_FIND_EXEC_TERMINATORS = frozenset({";", "+"})

#: Tokens that end one command segment and begin another. ``{`` and ``}`` are
#: here as STANDALONE tokens only: a lone brace is shell grouping grammar
#: (``{ claude -p x; }``, ``f(){ claude -p x; }``), never a command and never
#: an argument in ordinary work — ``find … {} \;`` and ``xargs -I {}`` pass
#: ``{}``, which is one token and stays untouched.
_SEGMENT_SEPARATORS = frozenset({
    ";", "&&", "||", "|", "&", "|&", ";;", "(", ")", "\n", "{", "}",
})

#: Characters that only ever separate commands. A token made up ENTIRELY of
#: these is a separator even when it is a combination the set above does not
#: enumerate — ``&&\n`` and ``;\n`` are emitted as single punctuation runs
#: once newlines participate in tokenization (see :func:`_tokenize`), and
#: treating them as ordinary tokens would silently re-merge the segments they
#: divide. Redirections (``<``/``>``) are deliberately NOT here: ``echo hi >
#: claude`` writes a file, it does not run one, and splitting there would
#: block it for no reason.
_SEGMENT_SEPARATOR_CHARS = frozenset(";&|()\n")

#: Shell punctuation shlex should emit as its own token. This is shlex's own
#: ``punctuation_chars=True`` set plus ``\n``.
_PUNCTUATION_CHARS = "();<>|&\n"

_ENV_ASSIGNMENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
_NUMERIC = re.compile(r"^\d+(\.\d+)?[smhd]?$")

#: Wrapper flags that consume the NEXT token as their value, so that value is
#: not mistaken for the command being wrapped. Without this, ``sudo -u root
#: claude -p x`` would stop scanning at ``root`` and miss the invocation.
_VALUE_TAKING_FLAGS = frozenset({
    "-u", "-g", "-p", "-n", "-c", "-s", "-k", "-i", "-o", "-e", "-C", "-D",
    "-R", "-T", "--user", "--group", "--userspec", "--signal", "--kill-after",
    "--adjustment", "--class", "--classdata", "--chdir",
})

#: Per-wrapper additions to :data:`_VALUE_TAKING_FLAGS`. ``xargs -I {} cmd``
#: and ``xargs -P 4 cmd`` would otherwise stop the scan on the flag's value
#: and never reach the command. Only flags whose argument is MANDATORY belong
#: here: GNU's optional-argument forms (``-i[R]``, ``-l[N]``, ``-e[EOF]``)
#: attach their value to the flag token, so treating them as value-taking
#: would swallow the command instead.
_WRAPPER_FLAG_TAKES_VALUE = {
    "xargs": frozenset({"-I", "-L", "-P", "-d", "-a", "-E", "--max-procs",
                        "--max-lines", "--replace", "--delimiter", "--arg-file"}),
    "parallel": frozenset({"-j", "-N", "-L", "--jobs"}),
    "watch": frozenset({"--interval"}),
}

#: Flags in the value-taking sets that do NOT take a value for a specific
#: wrapper, so its next token must not be consumed as one. ``sudo -n`` means
#: "non-interactive" and takes no argument — unlike ``nice -n 5`` or
#: ``ionice -n 3``, where the same letter genuinely does. Without this,
#: ``sudo -n claude`` swallows ``claude`` as ``-n``'s value and the
#: invocation is never seen. Same story for ``env -i claude`` (ignore
#: environment), ``xargs -i claude`` (replace-string, optional argument),
#: ``xargs -p claude`` (prompt) and ``watch -c claude`` (colour).
_WRAPPER_FLAG_TAKES_NO_VALUE = {
    "sudo": frozenset({"-n", "-i", "-s", "-k", "-e"}),
    "doas": frozenset({"-n", "-s"}),
    "env": frozenset({"-i", "-0", "-v"}),
    "xargs": frozenset({"-0", "-p", "-r", "-t", "-x", "-i", "-e", "-l", "-o"}),
    "watch": frozenset({"-c", "-d", "-t", "-b", "-e", "-g", "-p", "-x", "-q"}),
    "parallel": frozenset({"-k", "-u", "-t", "-c", "-e", "-p", "-i"}),
    # busybox takes no option that consumes a value before its applet name,
    # so nothing here may swallow the applet — `busybox -c sh` must still
    # arrive at `sh`.
    "busybox": _VALUE_TAKING_FLAGS,
}

#: GNU ``env``'s "run a whole command string from one argument" option. The
#: string is a command in its own right and is analysed as one.
_ENV_SPLIT_STRING_LONG = "--split-string"

#: Last-resort scan used only when tokenizing fails. Requires the token to be
#: bare ``claude`` or to end in ``/claude``, and to sit at a plausible command
#: boundary — so ``myclaude`` and ``--claude-flag`` do not match. The leading
#: ``/?`` is what lets an ABSOLUTE path match: without it the repeated
#: ``[\w.~+-]+/`` group could never consume the very first ``/``, so a
#: malformed ``/usr/local/bin/claude "oops`` scanned clean and was allowed.
_RAW_CLAUDE = re.compile(
    r"(?:^|[\s;&|(=\"'`])/?(?:[\w.~+-]+/)*claude(?=$|[\s;&|)\"'`])"
)

#: How deep a nest of indirections — ``sh -c "sh -c '…'"``, ``$( $( … ) )``,
#: ``find -exec busybox sh -c …`` — is followed before giving up. One counter
#: covers every kind of nesting so a mixed chain cannot buy extra depth by
#: alternating between them.
MAX_SHELL_DEPTH = 3


def _basename(token: str) -> str:
    return token.rsplit("/", 1)[-1] if "/" in token else token


def is_claude_invocation_token(token: Any) -> bool:
    """True when *token* names the Claude CLI as something to execute:
    exactly ``claude``, or a path whose final component is ``claude``."""
    if not isinstance(token, str) or not token:
        return False
    if token == _policy.CLAUDE_CLI_BASENAME:
        return True
    return "/" in token and _basename(token) == _policy.CLAUDE_CLI_BASENAME


def _tokenize(command: str) -> List[str]:
    """Shell-aware tokenization that keeps operators as their own tokens.

    ``punctuation_chars`` is what makes ``a&&claude`` and ``a|claude`` split
    even without surrounding whitespace; plain ``shlex.split`` would hand
    back one opaque word and miss the chained invocation entirely. Raises
    ``ValueError`` on unbalanced quoting, which the caller treats as the
    malformed/fail-closed case.

    Two deliberate departures from shlex's defaults, both closing a way for a
    second command to be smuggled into the same string:

    * a NEWLINE is a command terminator, not whitespace. By default shlex
      discards it, so ``make build`` / newline / ``claude -p x`` tokenized as
      one segment headed by ``make`` and the invocation was never examined.
      Adding it to ``punctuation_chars`` (and removing it from ``whitespace``)
      makes it a token; a newline INSIDE quotes is untouched, so
      ``git commit -m "…\\nclaude"`` is still one word and still allowed.
    * ``#`` never starts a comment. shlex's comment handling consumes the
      rest of the LINE *including its newline*, which would splice the next
      command back onto this segment — ``make # note`` / newline /
      ``claude -p x`` would read as a single ``make`` segment again.
    """
    lexer = shlex.shlex(command, posix=True, punctuation_chars=_PUNCTUATION_CHARS)
    lexer.whitespace_split = True
    lexer.whitespace = " \t\r"
    lexer.commenters = ""
    return list(lexer)


def _is_segment_separator(token: str) -> bool:
    if token in _SEGMENT_SEPARATORS:
        return True
    return bool(token) and all(char in _SEGMENT_SEPARATOR_CHARS for char in token)


def _split_segments(tokens: List[str]) -> List[List[str]]:
    segments: List[List[str]] = [[]]
    for token in tokens:
        if _is_segment_separator(token):
            segments.append([])
        else:
            segments[-1].append(token)
    return [segment for segment in segments if segment]


def _nested_reason(text: str, depth: int) -> Optional[str]:
    """Analyse *text* as a command string one level deeper.

    Recursion still stops at :data:`MAX_SHELL_DEPTH` — the bound is what
    keeps a hostile nest from turning the guard into an unbounded recursion
    — but stopping used to mean returning ``None``, which ALLOWED whatever
    sat at the bottom of the nest. ``bash -c "bash -c 'bash -c \\"bash -c
    claude\\"'"`` reached the bound with ``claude`` as the last payload and
    was waved through, and so did any wrapper (``env -S``, ``busybox sh``)
    that spent a level of its own before the nested shells started.

    So the bound now ends the same way the malformed case does: the
    fail-closed raw scan of :func:`_fail_closed`. Text that still shows a
    lexically visible head-position invocation blocks; text that does not
    returns ``None``, so ordinary work is never blocked on depth alone. The
    scan is deliberately cruder than the parser it replaces — a deeply nested
    ``grep -r claude .`` matches it and blocks — which is the same trade this
    module already makes for a shape it cannot finish parsing.

    A nested string whose own quoting is broken cannot be tokenized, so it
    falls back to that same raw scan rather than being waved through.
    """
    if not text or not text.strip():
        return None
    if depth >= MAX_SHELL_DEPTH:
        if _RAW_CLAUDE.search(text):
            return "direct Claude CLI invocation (nesting depth exceeded)"
        return None
    try:
        return _command_reason(text, depth + 1)
    except ValueError:
        if _RAW_CLAUDE.search(text):
            return "direct Claude CLI invocation (unbalanced quoting)"
        return None


def _segment_reason(segment: List[str], depth: int) -> Optional[str]:
    """Reason this one command segment directly invokes claude, or ``None``."""
    index = 0
    while index < len(segment):
        token = segment[index]

        # `FOO=bar claude` — leading assignments are not the command.
        if _ENV_ASSIGNMENT.match(token):
            index += 1
            continue

        # Shell grammar in head position: `then claude`, `do claude`,
        # `! claude` are the same invocation with a reserved word in front.
        if token in _RESERVED_WORDS:
            index += 1
            continue
        if token in _RESERVED_WORDS_WITH_NAME:
            index += 2
            continue

        if is_claude_invocation_token(token):
            return f"direct Claude CLI invocation ({token!r})"

        # A second `-exec` in the same find heads its own segment, because
        # the `;` terminating the first one split the token stream there.
        if token in FIND_EXEC_PRIMARIES:
            return _find_exec_reason(segment[index:], depth)

        base = _basename(token)

        if base in BUSYBOX_COMMANDS:
            return _busybox_reason(segment, index, base, depth)

        if base in WRAPPER_COMMANDS:
            reason, index = _advance_past_wrapper(segment, index, base, depth)
            if reason is not None:
                return reason
            continue

        if base in SHELL_COMMANDS:
            for payload in _shell_payloads(segment[index + 1:]):
                nested = _nested_reason(payload, depth)
                if nested is not None:
                    return f"shell-wrapped {nested}"
            return None

        if base in FIND_COMMANDS:
            return _find_exec_reason(segment[index + 1:], depth)

        # An ordinary command: whatever follows are its arguments, not
        # another executable. `git commit -m "claude"` stops right here.
        return None
    return None


def _advance_past_wrapper(
    segment: List[str], index: int, base: str, depth: int,
) -> Tuple[Optional[str], int]:
    """Step over a wrapper's own flags to the command it wraps.

    Returns ``(reason, next_index)``: a reason when the wrapper itself
    carries the invocation (``env -S "claude -p x"``, ``watch "claude -p
    x"``), otherwise ``None`` and the index of the head candidate, which the
    caller re-examines. The index always advances by at least one, so the
    caller's loop cannot spin.
    """
    index += 1
    no_value = _WRAPPER_FLAG_TAKES_NO_VALUE.get(base, frozenset())
    extra_value = _WRAPPER_FLAG_TAKES_VALUE.get(base, frozenset())

    while index < len(segment):
        candidate = segment[index]

        if candidate.startswith("-"):
            if base == "env":
                split = _env_split_string(segment, index)
                if split is not None:
                    payload, consumed = split
                    reason = _nested_reason(payload, depth)
                    if reason is not None:
                        return f"env split-string {reason}", index
                    index += consumed
                    continue
            index += 1
            takes_value = (
                (candidate in _VALUE_TAKING_FLAGS or candidate in extra_value)
                and candidate not in no_value
            )
            if takes_value and index < len(segment):
                index += 1
            continue

        # `env FOO=1 claude`, `timeout 300 claude`, `nice 5 claude`.
        if _ENV_ASSIGNMENT.match(candidate) or _NUMERIC.match(candidate):
            index += 1
            continue
        break

    # `watch`/`parallel` re-assemble their arguments into a shell command, so
    # the whole tail is a command string — `watch "claude -p x"` never has
    # `claude` in head position as a token of THIS segment.
    if base in SHELL_STRING_EXECUTORS and index < len(segment):
        reason = _nested_reason(" ".join(segment[index:]), depth)
        if reason is not None:
            return f"{base}-wrapped {reason}", index

    return None, index


def _env_split_string(segment: List[str], index: int) -> Optional[Tuple[str, int]]:
    """GNU ``env``'s split-string payload at *index*, and how many tokens it
    spans: ``-S "cmd"``, ``-S"cmd"``, ``--split-string=cmd``,
    ``--split-string cmd``, and the ``-iS "cmd"`` cluster form."""
    token = segment[index]
    if token.startswith(_ENV_SPLIT_STRING_LONG + "="):
        return token[len(_ENV_SPLIT_STRING_LONG) + 1:], 1
    if token == _ENV_SPLIT_STRING_LONG:
        return (segment[index + 1], 2) if index + 1 < len(segment) else None
    if token.startswith("--"):
        return None
    if "S" in token[1:]:
        after = token[token.index("S") + 1:]
        if after:
            return after, 1
        return (segment[index + 1], 2) if index + 1 < len(segment) else None
    return None


def _busybox_reason(
    segment: List[str], index: int, base: str, depth: int,
) -> Optional[str]:
    """``busybox <applet> …`` — the real command starts at the applet.

    At :data:`MAX_SHELL_DEPTH` this ends the way :func:`_nested_reason` does,
    for the same reason: giving up used to mean ALLOW, so an attacker only had
    to spend the depth budget before dispatching — ``sh -c "sh -c 'sh -c
    \\"busybox env claude\\"'"`` reached the bound and was waved through. The
    bound is unchanged; reaching it runs the fail-closed raw scan instead.

    The scan covers the APPLET TAIL — what busybox would actually exec — and
    not the tokens before it, so an unrelated mention in the prefix (``env
    FOO=claude busybox echo hi``) is not evidence of an invocation. Locating
    that tail costs no recursion: :func:`_advance_past_wrapper` only recurses
    for ``env``/``watch``/``parallel``, never for busybox.
    """
    _, applet = _advance_past_wrapper(segment, index, base, depth)
    rest = segment[applet:]
    if not rest:
        return None
    if depth >= MAX_SHELL_DEPTH:
        if _RAW_CLAUDE.search(" ".join(rest)):
            return "direct Claude CLI invocation (nesting depth exceeded)"
        return None
    nested = _segment_reason(rest, depth + 1)
    return f"busybox applet {nested}" if nested is not None else None


def _find_exec_reason(rest: List[str], depth: int) -> Optional[str]:
    """Reason a ``find`` expression runs claude via ``-exec``/``-execdir``.

    Only the argument list of an exec primary is treated as a command, which
    is what keeps ``find . -name claude`` and ``find . -exec grep claude {}
    \\;`` allowed.

    At :data:`MAX_SHELL_DEPTH` this ends the way :func:`_nested_reason` does,
    for the same reason: giving up used to mean ALLOW, so an attacker only had
    to spend the depth budget before the ``-exec`` — ``sh -c "sh -c 'sh -c
    \\"find . -exec claude -p x ;\\"'"`` reached the bound and was waved
    through. The bound is unchanged; reaching it runs the fail-closed raw scan
    instead.

    The scan stays scoped to each exec primary's own argument slice, up to its
    ``;``/``+`` terminator, so the predicates around it are not swept in:
    ``find . -name claude`` is a filename test at every depth and stays
    allowed even here.
    """
    at_bound = depth >= MAX_SHELL_DEPTH
    for position, token in enumerate(rest):
        if token not in FIND_EXEC_PRIMARIES:
            continue
        argv: List[str] = []
        for candidate in rest[position + 1:]:
            if candidate in _FIND_EXEC_TERMINATORS:
                break
            argv.append(candidate)
        if not argv:
            continue
        if at_bound:
            if _RAW_CLAUDE.search(" ".join(argv)):
                return "direct Claude CLI invocation (nesting depth exceeded)"
            continue
        nested = _segment_reason(argv, depth + 1)
        if nested is not None:
            return f"find {token} {nested}"
    return None


def _shell_payloads(rest: List[str]) -> List[str]:
    """Every command string an explicit shell invocation would run.

    Two feeds, both returned so the caller can analyse each: the ``-c``
    operand and a here-string. Order does not matter — any one of them
    carrying an invocation blocks.
    """
    return _shell_command_strings(rest) + _here_string_payloads(rest)


def _shell_command_strings(rest: List[str]) -> List[str]:
    """The ``-c`` command string(s) of a shell invocation, if any.

    Real shells do NOT parse ``-c`` as a getopt option with an attached
    argument: ``c`` sets "read the command from the first operand", and the
    remaining letters of its cluster are *more options*. ``bash -cx "claude
    -p x"`` is ``-c`` plus ``-x`` (xtrace) and the command string is the next
    argv word. Reading the cluster tail as the payload — the previous
    behaviour — meant ``bash -cx``, ``sh -cv`` and ``bash -ce`` all reported
    a payload of ``"x"``/``"v"``/``"e"`` and the real one was never looked at.

    So the operand is always taken. Options before it are skipped along with
    their values (``-O extglob``, ``--rcfile /dev/null``), and a value is
    only consumed when it does not itself look like an option, so
    ``bash -O -c "claude -p x"`` cannot swallow the ``-c``.

    A cluster tail is *additionally* scanned rather than trusted as the whole
    story: getopt-style shells (ksh) do accept an attached ``-c'claude'``,
    and the tokenizer also collapses ``bash -c"claude -p x"`` into a single
    ``-cclaude -p x`` token. Scanning a genuine option cluster like ``x`` or
    ``vx`` costs nothing — no letter run can name the CLI.
    """
    payloads: List[str] = []
    index = 0
    saw_c = False

    while index < len(rest):
        token = rest[index]

        if token == "--":  # end of options; the operand is next
            index += 1
            break
        if len(token) < 2 or token[0] not in "-+":
            break  # the first operand

        if token in _SHELL_VALUE_TAKING_OPTIONS:
            index += 1
            if index < len(rest) and rest[index][:1] not in ("-", "+"):
                index += 1
            continue
        if token.startswith("--"):
            index += 1
            continue

        cluster = token[1:]
        if "c" in cluster:
            saw_c = True
            attached = cluster[cluster.index("c") + 1:]
            if attached:
                payloads.append(attached)
        index += 1

    if saw_c and index < len(rest):
        payloads.append(rest[index])
    return payloads


def _here_string_payloads(rest: List[str]) -> List[str]:
    """Here-strings feeding a shell: ``bash <<< "claude -p x"``.

    A here-string is the shell's script, so its text is a command string in
    exactly the way a ``-c`` operand is — and it wore no ``-c``, so the
    payload scan never fired and ``bash <<< "claude -p x"`` was allowed.
    Only the ``<`` run itself is matched (``<<<``, as its own token), and
    only for a segment already headed by a shell: ``cat <<< claude`` prints a
    word and stays allowed, and a quoted mention (``echo 'bash <<< "claude -p
    x"'``) is one argument token that never reaches here.

    Here-DOCS need no case: ``<<`` is followed by a delimiter word, and the
    body arrives on its own lines, which a newline has already split into
    ordinary segments for the head-position scan.
    """
    payloads: List[str] = []
    for index, token in enumerate(rest):
        if len(token) >= 3 and set(token) == {"<"} and index + 1 < len(rest):
            payloads.append(rest[index + 1])
    return payloads


def _closing_backtick(text: str, start: int) -> Optional[int]:
    index = start
    while index < len(text):
        char = text[index]
        if char == "\\":
            index += 2
            continue
        if char == "`":
            return index
        index += 1
    return None


def _matching_paren(text: str, start: int) -> Optional[int]:
    """Index of the ``)`` closing the ``(`` at *start*, honouring nesting and
    quotes so ``$(echo "a)b")`` is not cut short at the quoted paren."""
    depth = 0
    index = start
    in_single = False
    in_double = False
    while index < len(text):
        char = text[index]
        if in_single:
            if char == "'":
                in_single = False
        elif char == "\\":
            index += 1
        elif in_double:
            if char == '"':
                in_double = False
        elif char == "'":
            in_single = True
        elif char == '"':
            in_double = True
        elif char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return index
        index += 1
    return None


def _substitution_payloads(command: str) -> List[str]:
    """Every executable substitution in *command*, as command strings.

    The shell runs these BEFORE (or alongside) the command that contains
    them, so ``echo `claude -p x` `` invokes claude however innocent the
    outer head looks. Four forms qualify: backticks, ``$(…)``, and the
    ``<(…)``/``>(…)`` process substitutions.

    Quoting decides what is live, and this walk follows the shell's rules
    rather than blanket-matching: inside SINGLE quotes everything is literal,
    so ``echo '$(claude)'`` yields no payload and stays allowed; inside
    double quotes ``$(…)`` and backticks still execute, so they do. Process
    substitution is not performed inside double quotes at all.

    An UNTERMINATED substitution yields the rest of the string as its
    payload: the shell would reject the command, but this module's stance on
    a shape it cannot parse is to look anyway rather than pass it through.
    """
    payloads: List[str] = []
    index = 0
    length = len(command)
    in_single = False
    in_double = False

    while index < length:
        char = command[index]

        if in_single:
            if char == "'":
                in_single = False
            index += 1
            continue
        if char == "\\":
            index += 2
            continue
        if char == '"':
            in_double = not in_double
            index += 1
            continue
        if char == "'" and not in_double:
            in_single = True
            index += 1
            continue

        if char == "`":
            end = _closing_backtick(command, index + 1)
            if end is None:
                payloads.append(command[index + 1:])
                break
            payloads.append(command[index + 1:end])
            index = end + 1
            continue

        opens_substitution = (
            command.startswith("$(", index)
            or (not in_double and char in "<>" and command.startswith("(", index + 1))
        )
        if opens_substitution:
            end = _matching_paren(command, index + 1)
            if end is None:
                payloads.append(command[index + 2:])
                break
            payloads.append(command[index + 2:end])
            index = end + 1
            continue

        index += 1

    return payloads


def _command_reason(command: str, depth: int = 0) -> Optional[str]:
    for segment in _split_segments(_tokenize(command)):
        reason = _segment_reason(segment, depth)
        if reason is not None:
            return reason
    for payload in _substitution_payloads(command):
        nested = _nested_reason(payload, depth)
        if nested is not None:
            return f"command substitution with {nested}"
    return None


def evaluate(command: Any) -> Dict[str, Any]:
    """Decide whether *command* directly invokes the Claude CLI.

    Returns ``{"blocked": bool, "reason": str, "malformed": bool}`` and never
    raises — a guard that could throw would take tool dispatch down with it.
    A non-string/empty command is not an invocation of anything and is
    allowed; only an actual detected invocation blocks.
    """
    if not isinstance(command, str) or not command.strip():
        return {"blocked": False, "reason": "", "malformed": False}

    try:
        reason = _command_reason(command)
    except ValueError:
        return _fail_closed(command, "command could not be tokenized (unbalanced quoting)")
    except Exception:  # pragma: no cover - defensive
        logger.warning("claude-worker terminal guard: internal error", exc_info=True)
        return _fail_closed(command, "terminal guard internal error")

    if reason is None:
        return {"blocked": False, "reason": "", "malformed": False}
    return {"blocked": True, "reason": reason, "malformed": False}


def _fail_closed(command: str, detail: str) -> Dict[str, Any]:
    """Malformed shape: block only when a direct invocation is still visible."""
    if _RAW_CLAUDE.search(command):
        return {
            "blocked": True,
            "reason": f"malformed terminal command with a direct Claude CLI invocation ({detail})",
            "malformed": True,
        }
    return {"blocked": False, "reason": "", "malformed": True}
