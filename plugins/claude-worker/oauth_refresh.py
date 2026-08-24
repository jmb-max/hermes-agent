"""Isolated OAuth refresh probe — the ONE implementation of the seam
``oauth.set_refresh_probe`` has always exposed.

Why this lives outside ``oauth.py``
-----------------------------------
``oauth.py`` is tripwired at the source level against ever gaining a write
path, a child process, or network egress, and that tripwire is load-bearing:
it is what makes "this plugin never owns the privileged credential write" a
structural fact rather than a promise. A refresh therefore cannot be
implemented there — and it is not reimplemented here either. It is DELEGATED
to the one program that legitimately owns the privileged write and knows the
grant parameters: the host's own Claude CLI at :data:`HOST_CLAUDE_CLI`.

What this module does is narrow on purpose: run that fixed, trust-validated
binary exactly once, with a fixed argv that grants it no tool at all, a
minimal environment built from literals, an empty throwaway working directory
inside the root-owned staging root, and a bounded timeout — then believe
nothing but the credential the run leaves on disk. ``oauth.preflight`` remains
the final authority: after the probe returns it re-reads the credential and
believes only what it finds there.

An exit status is not a refresh
-------------------------------
``claude --print`` exits 0 for plenty of turns that never touched the
credential — a cached session, a refusal, a reply produced from an access
token that was valid for one more second. So a clean exit is only the
INVITATION to look: the credential is re-read while the lock is STILL held,
and the run is reported as a refresh only if it is now
``oauth.STATE_FRESH``. Releasing the lock first would let the next probe's
winner rewrite the file underneath this one's confirmation.

What never crosses this boundary
--------------------------------
* **No credential material, in either direction.** This module opens the
  credentials file for exactly one purpose — asking ``oauth.read_freshness``
  for the non-secret freshness summary — and never parses, persists, re-owns,
  or returns any part of it. The child's own output is sent straight to
  ``/dev/null``, so a diagnostic that quotes the credential the CLI just tried
  to use is discarded before it exists in this process at all.
* **No free text.** Every value this module can return comes from the closed,
  fixed :data:`REASONS` set. Nothing is interpolated into a reason, and no
  child output or exception text is ever logged, not even at DEBUG.
* **No second control plane.** The only knob is the validated
  ``plugins.entries.claude-worker.oauth`` block in config.yaml: whether the
  probe runs at all, and how long the host CLI may run. The binary, the lock,
  the argv, the environment, the working directory, and the prompt are
  literals here and are not reachable from configuration.
* **No shared state.** A failed refresh records nothing anywhere: an
  unrefreshable credential is not an observed rejection by the API, and this
  module deliberately has no way to say otherwise.

The stampede this closes
------------------------
Several sessions can hit the same expired credential at the same instant.
Each would otherwise spawn its own privileged CLI run against the same
credential. So the work happens under an isolated, root-owned lock at
:data:`REFRESH_LOCK_PATH`, and freshness is re-read INSIDE the lock: whoever
loses the race finds the credential already refreshed by the winner and
returns success having spawned nothing. Checking first and locking second
would be exactly the stampede — every waiter would have decided to spawn
before anyone won.

Nothing here is taken on faith twice
------------------------------------
Two windows sit between a check and the use it authorizes, and both are
closed. The trusted-path validation of the CLI returns an identity snapshot,
and that snapshot is re-checked immediately before the process is created —
otherwise the lock acquisition in between (up to a minute) would be an ample
time-of-check/time-of-use window over the one program we hand the host's
credential to. And the staging root is validated on every run rather than
believed from a ``makedirs(exist_ok=True)`` that would happily accept a
directory somebody else already made.

Every failure mode fails closed: a config knob turned off, an untrusted
binary, a binary that changed after validation, a staging directory that is
not private, a lock that cannot be taken, a timeout, a non-zero exit, an
unreadable exit status, a run that finished without renewing anything, and
any unexpected exception all return ``{"ok": False, "reason": <fixed
literal>}``. ``oauth.preflight`` then HOLDs the call, which is the safe
direction.
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import logging
import os
import shutil
import signal
import stat
import subprocess
import tempfile
import time
from typing import Any, Dict, List, Optional

from . import config as _config
from . import oauth as _oauth
from . import policy as _policy
from . import trust as _trust

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Fixed literals — none of these is reachable from configuration
# ---------------------------------------------------------------------------

#: The host Claude CLI, as a fixed ABSOLUTE path. Never the bare basename:
#: resolving ``claude`` off ``$PATH`` would let any earlier directory
#: substitute an attacker's binary for the program we are about to let touch
#: the host's credential. The path is additionally validated as a
#: root-owned, non-symlink, owner-executable regular file (with every parent
#: directory likewise) immediately before each run, so a host where the CLI
#: is a user-owned symlink simply fails closed with
#: :data:`REASON_UNTRUSTED_CLI` rather than running something unvetted.
#: This is the packaged executable inside the global npm install — basename
#: ``claude.exe``, not the ``claude`` wrapper name npm exposes on ``$PATH``,
#: which is a symlink and so is refused by that same validation.
HOST_CLAUDE_CLI = "/usr/lib/node_modules/@anthropic-ai/claude-code/bin/claude.exe"

#: The refresh lock, isolated inside the fixed root-owned mode-0700 staging
#: root. A lock in a world-writable directory is not a lock: any local user
#: could hold it forever (putting the stampede back) or replace the file.
REFRESH_LOCK_PATH = f"{_policy.CREDENTIAL_STAGING_ROOT.rstrip('/')}/oauth-refresh.lock"

#: Longest a probe will wait for the lock before reporting it busy. Bounded
#: so a stuck holder degrades into one HOLD, never into a hung caller.
REFRESH_LOCK_TIMEOUT_SECONDS = 60

_LOCK_DIR_MODE = 0o700
_LOCK_FILE_MODE = 0o600
_LOCK_POLL_SECONDS = 0.02

#: How long a timed-out refresh turn's process GROUP is given to exit after
#: its SIGTERM before the group is SIGKILLed. Fixed and short: this runs while
#: the isolated lock is still held, so a generous grace here is a stampede
#: everybody else waits through.
REFRESH_KILL_GRACE_SECONDS = 3.0

#: One cheap turn whose only job is to make the CLI use — and therefore
#: renew — its own credential. Nothing caller-shaped is ever in it.
REFRESH_PROMPT = "Reply with the single word ok and nothing else."

#: The cheap default model, never the escalation model: this turn exists to
#: touch a credential, not to do work.
REFRESH_MODEL = _policy.DEFAULT_MODEL

#: The child's ``$PATH``: a fixed list of absolute system directories.
#: :data:`HOST_CLAUDE_CLI` is absolute and never resolved through it, but a
#: fixed minimal ``$PATH`` means nothing the CLI itself execs keeps looking
#: in an attacker-controlled directory either.
REFRESH_PATH = "/usr/local/bin:/usr/bin:/bin"

#: A fixed locale, so the child's behaviour does not depend on the ambient
#: one. Like every other value here it is a literal, never inherited.
REFRESH_LANG = "C.UTF-8"

#: The permission mode the refresh turn runs in. Deliberately NOT
#: ``policy.PERMISSION_MODE`` (``acceptEdits`` auto-approves the very tools
#: this turn must never get) and never ``bypassPermissions`` (which would
#: ignore the deny rule below outright). ``dontAsk`` never prompts — there is
#: nobody to answer, the turn's stdin is ``/dev/null`` — and still enforces.
REFRESH_PERMISSION_MODE = "dontAsk"

#: The file-shaped half of the no-tool policy. Not redundant with the
#: ``--disallowed-tools`` flag: ``--settings`` is the control plane the CLI
#: merges its permission rules from, so a deny that lives only in argv is one
#: flag-name change away from silently granting the world. A bare-name glob
#: ``*`` in ``deny`` matches every tool; there is deliberately no ``allow``
#: and no ``ask`` list, because an allow list only ever ADDS a permission.
REFRESH_SETTINGS = {
    "permissions": {
        "deny": ["*"],
        "defaultMode": REFRESH_PERMISSION_MODE,
    },
}

_CWD_PREFIX = "claude-worker-refresh-"

# ---------------------------------------------------------------------------
# The closed set of reasons — fixed, non-secret, never interpolated
# ---------------------------------------------------------------------------

REASON_REFRESHED = "the host Claude CLI completed a refresh turn"
REASON_ALREADY_FRESH = "the credential was already refreshed by another run"
REASON_DISABLED = "automatic OAuth refresh is disabled by configuration"
REASON_UNTRUSTED_CLI = "the host Claude CLI failed trusted-path validation"
REASON_LOCK_BUSY = "another refresh currently holds the refresh lock"
REASON_TIMEOUT = "the host Claude CLI exceeded its refresh timeout"
REASON_CLI_FAILED = "the host Claude CLI exited non-zero"
REASON_MALFORMED = "the host Claude CLI produced no readable exit status"
#: The run finished cleanly and the credential on disk is still not fresh. Its
#: own literal rather than a fold into :data:`REASON_CLI_FAILED`, because the
#: two say different things to an operator: the CLI did not fail, it simply
#: never renewed anything.
REASON_NOT_REFRESHED = "the host Claude CLI finished without renewing the credential"
REASON_ERROR = "the refresh could not be completed"

#: Everything a probe result may ever carry. A caller can compare against
#: this set; nothing outside it is reachable.
REASONS = frozenset({
    REASON_REFRESHED,
    REASON_ALREADY_FRESH,
    REASON_DISABLED,
    REASON_UNTRUSTED_CLI,
    REASON_LOCK_BUSY,
    REASON_TIMEOUT,
    REASON_CLI_FAILED,
    REASON_MALFORMED,
    REASON_NOT_REFRESHED,
    REASON_ERROR,
})


class RefreshLockUnavailable(RuntimeError):
    """The isolated refresh lock could not be taken: another refresh holds
    it, or the lock file itself is not usable safely."""


def _outcome(ok: bool, reason: str) -> Dict[str, Any]:
    """The only result shape this module produces: a bool and a fixed
    literal, and nothing else that could carry text out of here."""
    return {"ok": bool(ok), "reason": reason}


# ---------------------------------------------------------------------------
# The one configuration knob
# ---------------------------------------------------------------------------


def _oauth_settings(cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """The validated ``oauth`` block.

    When *cfg* is a mapping it is believed as-is and configuration is NOT
    re-loaded (the plugin already loaded it once at registration). Otherwise
    the validating loader is consulted, which fails closed to its own
    defaults and never raises.
    """
    if isinstance(cfg, dict):
        section = cfg.get("oauth")
    else:
        try:
            section = _config.load_plugin_config().get("oauth")
        except Exception:  # pragma: no cover - the loader already fails closed
            section = None
    return section if isinstance(section, dict) else {}


def _auto_refresh_enabled(settings: Dict[str, Any]) -> bool:
    """Only a literal ``False`` turns the probe off. A truthy-looking string
    like ``"no"`` is a mistake, and here "off" means an expired credential
    HOLDs a session that could have been recovered."""
    return settings.get("auto_refresh") is not False


def _refresh_timeout_seconds(settings: Dict[str, Any]) -> int:
    """The bounded child timeout. The loader already clamps this; clamping
    again here means an unbounded (or zero) value cannot reach a privileged
    child process even if it arrives by some other route."""
    value = settings.get("refresh_timeout_seconds")
    if isinstance(value, bool) or not isinstance(value, int):
        value = _config.DEFAULT_REFRESH_TIMEOUT_SECONDS
    return max(
        _config.MIN_REFRESH_TIMEOUT_SECONDS,
        min(_config.MAX_REFRESH_TIMEOUT_SECONDS, value),
    )


# ---------------------------------------------------------------------------
# argv and env — built from literals, never from a caller or the ambient
# environment
# ---------------------------------------------------------------------------


def build_refresh_argv() -> List[str]:
    """The complete, fixed argv of the refresh turn.

    Takes no parameters BY CONSTRUCTION: with no task, no path, no model and
    no environment to pass in, there is nothing a caller could smuggle into
    the command line of a privileged host process. No shell is involved at
    any point, the turn is granted no tool at all, it is capped at one turn,
    its MCP configuration is empty and strict, and it is told to persist no
    settings or session for a later run to inherit.

    "No tool" is spelled the way the permission model actually reads. An
    empty ``--allowed-tools`` is NOT a denial: an allow list only ever adds a
    permission and never subtracts one, so an empty one grants nothing while
    also forbidding nothing — every default tool stays reachable behind a flag
    that reads like a guarantee. The enforcing halves are the wildcard deny
    (on the command line AND in the settings payload) and a permission mode
    that neither prompts nor bypasses the rules it just installed.
    """
    return [
        HOST_CLAUDE_CLI,
        "--print",
        "--model", REFRESH_MODEL,
        "--max-turns", "1",
        # A refresh turn needs no capability whatsoever: it exists to make
        # the CLI touch its own credential, not to do work. ``*`` is the
        # bare-name glob that matches — and so removes — every tool.
        "--disallowed-tools", "*",
        # And the tool set itself is empty: nothing is even loaded for a
        # permission rule to have to deny.
        "--tools", "",
        "--permission-mode", REFRESH_PERMISSION_MODE,
        "--settings", json.dumps(REFRESH_SETTINGS, sort_keys=True),
        "--strict-mcp-config",
        "--mcp-config", json.dumps({"mcpServers": {}}),
        # No settings sources, and no session/continue/resume flag anywhere:
        # the run leaves nothing behind.
        "--setting-sources", "",
        REFRESH_PROMPT,
    ]


def build_refresh_env() -> Dict[str, str]:
    """The child's complete environment, built from literals.

    Nothing is inherited. That is the point: an ambient ``ANTHROPIC_API_KEY``
    or ``CLAUDE_CODE_OAUTH_TOKEN`` would make the CLI authenticate as
    something other than the credential we are trying to renew, an ambient
    base-URL or proxy variable could redirect the exchange at an
    attacker-controlled endpoint, and the usual loader hazards
    (``LD_PRELOAD``, ``NODE_OPTIONS`` and friends) would let an unrelated
    variable inject code into a privileged child. None of them can be
    dropped, because none of them is ever picked up.

    ``HOME`` (and the CLI config directory) are derived from the fixed
    credentials path, so the CLI lands on exactly the credential
    ``oauth.preflight`` re-reads afterwards — refreshing some other profile
    would leave the preflight failing forever.
    """
    credentials_path = _policy.HOST_CREDENTIALS_PATH
    config_dir = os.path.dirname(credentials_path)
    return {
        "HOME": os.path.dirname(config_dir),
        "PATH": REFRESH_PATH,
        "CLAUDE_CONFIG_DIR": config_dir,
        "LANG": REFRESH_LANG,
    }


# ---------------------------------------------------------------------------
# The private staging tree the lock and the throwaway cwd both live in
# ---------------------------------------------------------------------------


def _validate_staging_dir(directory: str):
    """Validate one staging directory, chain and all, or raise
    ``trust.TrustViolation``.

    A throwaway directory is not automatically a directory it is safe to throw
    the host's credential into. Every parent must be a trusted directory in
    exactly the sense :mod:`trust` already means (real, non-symlink, root-owned,
    never writable by anyone else), and *directory* itself must additionally be
    private to its owner: ANY group or world bit is refused, including a merely
    readable one, since ``x`` on a directory is enough to walk into it. One
    group-writable parent would be enough to replace the directory underneath a
    check that only looked at its leaf, which is why the whole chain is walked.
    """
    path = os.path.normpath(str(directory))
    if not os.path.isabs(path):
        raise _trust.TrustViolation(
            f"staging directory is not an absolute path: {path!r}"
        )
    for parent in _trust.parent_dirs(path):
        _trust.check_dir_component(parent)
    st = _trust.check_dir_component(path)
    if st.st_mode & (stat.S_IRWXG | stat.S_IRWXO):
        raise _trust.TrustViolation(
            f"staging directory is not private to its owner: {path!r}"
        )
    return st


def _ensure_private_staging_dir(directory: str) -> str:
    """Create *directory* if it is absent, force it private, and validate it.

    The validation is the point: ``makedirs(exist_ok=True)`` succeeds just as
    happily on a directory somebody else already made — or symlinked — as on
    one we created, so the result is checked rather than assumed. The explicit
    mode is the other half: a bare ``mkdir`` obeys the umask, and the 0755
    directory that usually produces is exactly what the check must refuse.
    """
    path = os.path.normpath(str(directory))
    try:
        os.makedirs(path, mode=_LOCK_DIR_MODE, exist_ok=False)
    except FileExistsError:
        pass
    else:
        # Only ever on a directory this call just created, and never through a
        # symlink somebody else planted: an existing path is left untouched and
        # simply validated below.
        os.chmod(path, _LOCK_DIR_MODE)
    _validate_staging_dir(path)
    return path


# ---------------------------------------------------------------------------
# The isolated lock
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def refresh_lock():
    """Hold the isolated refresh lock, or raise
    :class:`RefreshLockUnavailable`.

    An advisory ``flock`` on a private mode-0600 file inside the root-owned
    staging root. Opened with ``O_NOFOLLOW`` so the final component can never
    be a symlink to somewhere else, and refused outright if the file that is
    there is anything but a regular file with no group or world bits — a lock
    another account can hold, replace, or observe is not a lock.

    The directory it lives in is validated BEFORE the lock file is created,
    and the lock is refused if that validation fails: a lock file made inside
    a directory that failed validation is a lock another account already
    controls, and creating it first would be creating exactly that.
    """
    path = REFRESH_LOCK_PATH
    directory = os.path.dirname(path)
    try:
        _ensure_private_staging_dir(directory)
    except Exception as exc:
        raise RefreshLockUnavailable(
            "the refresh lock directory is not usable"
        ) from exc

    flags = os.O_RDONLY | os.O_CREAT | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        fd = os.open(path, flags, _LOCK_FILE_MODE)
    except (OSError, ValueError) as exc:
        raise RefreshLockUnavailable("the refresh lock file is not usable") from exc

    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode) or st.st_mode & (stat.S_IRWXG | stat.S_IRWXO):
            raise RefreshLockUnavailable(
                "the refresh lock file is not private to its owner"
            )

        deadline = time.monotonic() + float(REFRESH_LOCK_TIMEOUT_SECONDS)
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError as exc:
                if time.monotonic() >= deadline:
                    raise RefreshLockUnavailable(
                        "another refresh holds the refresh lock"
                    ) from exc
                time.sleep(_LOCK_POLL_SECONDS)

        try:
            yield
        finally:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            except OSError:  # pragma: no cover - the fd is closed next anyway
                pass
    finally:
        os.close(fd)


# ---------------------------------------------------------------------------
# The one subprocess seam
# ---------------------------------------------------------------------------


def _reap_process_group(child) -> None:
    """Terminate everything the timed-out refresh turn left running.

    The child is its own session leader, so its pid IS its process-group id
    and the whole group can be signalled at once: SIGTERM, one fixed short
    grace, then SIGKILL for whatever ignored it — each followed by a wait, so
    the group is genuinely reaped rather than merely signalled. Signalling the
    direct child alone is what leaves the CLI's own descendants orphaned and
    running: still holding the credential, still able to write it, and now
    outliving the probe that was supposed to bound them.
    """
    pid = getattr(child, "pid", None)
    if not isinstance(pid, int):  # pragma: no cover - a real child always has one
        return
    for signum in (signal.SIGTERM, signal.SIGKILL):
        try:
            pgid = os.getpgid(pid)
        except OSError:
            return  # already gone
        try:
            os.killpg(pgid, signum)
        except OSError:
            return
        try:
            child.wait(timeout=REFRESH_KILL_GRACE_SECONDS)
            return
        except subprocess.TimeoutExpired:
            continue
        except OSError:  # pragma: no cover - nothing left to wait for
            return


def _run_refresh(argv, *, env, cwd, timeout_seconds):
    """Run the host CLI once. The ONE place this module spawns anything.

    The child's stdout and stderr go to ``/dev/null`` rather than into a
    buffer: a failing ``claude`` prints its own diagnostics, and those can
    quote the very credential it just tried to use. The exit status is the
    only thing this module ever wanted, and the only thing it gets. stdin is
    likewise closed, so the turn can never block waiting for input it will
    not receive.

    The child is created with ``start_new_session=True`` so that it leads its
    own process group and a timeout can reap that whole group. On a timeout
    the group is reaped here, before this function returns, so the throwaway
    working directory is never removed out from under something still running
    in it. The raised timeout is rebuilt from literals: the one the wait
    produces would be a fine place for child output to travel out of.
    """
    child = subprocess.Popen(
        argv,
        env=env,
        cwd=cwd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
        close_fds=True,
    )
    try:
        code = child.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        _reap_process_group(child)
        raise subprocess.TimeoutExpired(
            cmd=[HOST_CLAUDE_CLI], timeout=timeout_seconds,
        ) from None
    return subprocess.CompletedProcess(args=argv, returncode=code)


def _make_throwaway_cwd() -> str:
    """An empty, private, throwaway working directory inside the fixed
    root-owned staging root — never the ambient temp root.

    Running the refresh inside a repository would hand a fully-trusted host CLI
    a project to read, so it gets a scratch directory. But a scratch directory
    under the ambient temp root is world-traversable and attacker-influenceable
    — ``TMPDIR`` decides where it lands, and any local user can pre-create or
    watch the parent — so the location is a policy literal instead, in the same
    isolated tree the lock lives in, and both the root and the directory
    created inside it are validated rather than assumed.
    """
    root = _ensure_private_staging_dir(_policy.CREDENTIAL_STAGING_ROOT.rstrip("/"))
    workdir = tempfile.mkdtemp(prefix=_CWD_PREFIX, dir=root)
    try:
        _validate_staging_dir(workdir)
    except Exception:
        shutil.rmtree(workdir, ignore_errors=True)
        raise
    return workdir


def _spawn_refresh(timeout_seconds: int, snapshot) -> Dict[str, Any]:
    """One bounded run of the host CLI in a throwaway working directory.

    The retained trusted-path *snapshot* is re-checked immediately before the
    process is created: everything between the original validation and this
    moment — a lock acquisition that may have blocked for up to a minute —
    is time in which the binary could have been swapped, and the whole point
    of the snapshot is that the swap is caught rather than executed.
    """
    try:
        workdir = _make_throwaway_cwd()
    except Exception as exc:
        logger.warning(
            "claude_worker: the refresh staging directory failed validation (%s); "
            "not attempting an OAuth refresh",
            type(exc).__name__,
        )
        return _outcome(False, REASON_ERROR)

    try:
        try:
            _trust.revalidate_trusted_path_chain(snapshot)
        except Exception:
            # Same reasoning as the first validation: the path is a fixed
            # literal, so the violation text tells an operator nothing new.
            logger.warning(
                "claude_worker: the host Claude CLI changed after trusted-path "
                "validation; not attempting an OAuth refresh"
            )
            return _outcome(False, REASON_UNTRUSTED_CLI)

        try:
            completed = _run_refresh(
                build_refresh_argv(),
                env=build_refresh_env(),
                cwd=workdir,
                timeout_seconds=timeout_seconds,
            )
        except subprocess.TimeoutExpired:
            # Deliberately no exception text and no traceback: a timeout
            # carries whatever the child had already produced.
            logger.warning("claude_worker: the host Claude CLI refresh turn timed out")
            return _outcome(False, REASON_TIMEOUT)
        except Exception as exc:
            logger.warning(
                "claude_worker: the host Claude CLI refresh turn raised %s",
                type(exc).__name__,
            )
            return _outcome(False, REASON_ERROR)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    # An exit status, and nothing else, is believed — and even that only says
    # the run FINISHED. Anything that cannot be read as an exit status is a
    # failure, never a success.
    code = getattr(completed, "returncode", None)
    if isinstance(code, bool) or not isinstance(code, int):
        return _outcome(False, REASON_MALFORMED)
    if code != 0:
        return _outcome(False, REASON_CLI_FAILED)
    return _outcome(True, REASON_REFRESHED)


def _read_state(now: Optional[float]):
    """The non-secret freshness state of the host credential, and nothing
    else: no field of the credential itself is read, kept, or returned."""
    freshness = _oauth.read_freshness(_policy.HOST_CREDENTIALS_PATH, now=now)
    return freshness.get("state") if isinstance(freshness, dict) else None


def _refresh_under_lock(
    timeout_seconds: int, now: Optional[float], snapshot,
) -> Dict[str, Any]:
    """Re-read freshness INSIDE the lock, spawn only if it is still worth
    spawning, and confirm the result against the file the CLI itself owns —
    all before the lock is released."""
    state = _read_state(now)

    if state == _oauth.STATE_FRESH:
        # Another process won the race and already did the work.
        return _outcome(True, REASON_ALREADY_FRESH)
    if state != _oauth.STATE_REFRESHABLE_EXPIRED:
        # Missing, malformed, or expired beyond recovery: there is nothing a
        # refresh turn could restore, so nothing is spawned.
        return _outcome(False, REASON_ERROR)

    outcome = _spawn_refresh(timeout_seconds, snapshot)
    if not outcome["ok"]:
        return outcome

    # The run finished cleanly. That is a claim about the PROCESS, not about
    # the credential, so the credential is re-read — still under this lock,
    # because releasing it first would let the next probe's winner rewrite the
    # file underneath this confirmation. It is the same read the stampede check
    # above already does, so it costs nothing.
    if _read_state(now) == _oauth.STATE_FRESH:
        return _outcome(True, REASON_REFRESHED)
    return _outcome(False, REASON_NOT_REFRESHED)


def refresh_probe(now: Optional[float] = None) -> Dict[str, Any]:
    """The zero-argument-callable probe ``oauth.preflight`` invokes.

    Reads the config knob, trust-validates the fixed CLI path and RETAINS the
    identity snapshot, takes the isolated lock, re-reads freshness under it,
    re-checks the snapshot immediately before process creation, and — only
    then, and only for a credential that is expired but still refreshable —
    runs the host CLI exactly once, confirming the outcome against the
    credential itself while the lock is still held. Never raises:
    ``oauth.preflight`` guards itself, but a probe that raised into it would
    still blur "the refresh failed" into "the preflight itself broke".
    """
    try:
        settings = _oauth_settings()
        if not _auto_refresh_enabled(settings):
            return _outcome(False, REASON_DISABLED)
        timeout_seconds = _refresh_timeout_seconds(settings)

        try:
            snapshot = _trust.validate_trusted_path_chain(
                HOST_CLAUDE_CLI, executable=True,
            )
        except Exception:
            # The path itself is a fixed literal, so the violation text adds
            # nothing an operator cannot read off HOST_CLAUDE_CLI.
            logger.warning(
                "claude_worker: the host Claude CLI failed trusted-path validation; "
                "not attempting an OAuth refresh"
            )
            return _outcome(False, REASON_UNTRUSTED_CLI)

        try:
            with refresh_lock():
                return _refresh_under_lock(timeout_seconds, now, snapshot)
        except RefreshLockUnavailable:
            return _outcome(False, REASON_LOCK_BUSY)
    except Exception as exc:
        logger.warning(
            "claude_worker: the OAuth refresh probe failed with %s", type(exc).__name__,
        )
        return _outcome(False, REASON_ERROR)


# ---------------------------------------------------------------------------
# Installation — what ``register(ctx)`` calls
# ---------------------------------------------------------------------------


def install_refresh_probe(cfg: Optional[Dict[str, Any]] = None) -> bool:
    """Install :func:`refresh_probe` behind ``oauth.set_refresh_probe``, or
    clear the seam when the knob is off. Returns whether a probe is now
    installed.

    Deterministic rather than merely additive: whatever an earlier
    registration left behind, the answer after this call is the knob's, and
    the installed object is always this one function — repeated registration
    can neither stack probes nor leave a stale one behind.
    """
    enabled = _auto_refresh_enabled(_oauth_settings(cfg))
    _oauth.set_refresh_probe(refresh_probe if enabled else None)
    return enabled
