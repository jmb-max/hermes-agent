"""OAuth credential freshness preflight — run BEFORE the auth breaker would
ever be opened, and before any spawn.

The failure this closes: the worker spawned with an already-expired OAuth
access token, the container came back with a 401, ``breaker.classify_failure``
called it ``auth``, and the breaker slammed shut for a full hour — over a
token that a refresh would have fixed in a second. Every session in the
meantime sat on HOLD. The breaker is the right response to *observed*
authentication failure; it is the wrong response to a credential we could
have inspected first.

So: read the credential's FRESHNESS before committing to a spawn. This module
is deliberately the only place that knows the shape of the credentials file,
so the knowledge is centralized and verifiable rather than re-derived at each
call site.

Secrets never leave this module
-------------------------------
Nothing here ever returns, logs, or formats a token. :func:`read_freshness`
returns booleans, an expiry timestamp, a seconds-remaining integer, the
non-secret scope/subscription labels, and a state string — that is the entire
surface. The raw bytes are read into a local, parsed, and dropped; the parsed
object is never placed in the result and never logged, not even at DEBUG, and
:func:`redact` is applied to any free-text reason that reaches a result so a
token that somehow appeared in a probe's own error string cannot ride out on
it.

Refresh is deliberately NOT reimplemented here
----------------------------------------------
When the access token is expired but the credential still carries usable
refresh metadata, :func:`preflight` performs AT MOST ONE isolated probe —
structurally at most one, on one code path — and then re-reads freshness
from disk to see whether the credential actually recovered. The probe itself
is an injection point (:data:`REFRESH_PROBE`), not a hand-rolled refresh:
writing a refreshed token back to the root-owned
``policy.HOST_CREDENTIALS_PATH`` would mean duplicating exactly the
privileged credential-writing logic this plugin has been careful never to
own — it stages a read-only copy and mounts that, and nothing else. With no
probe installed the preflight simply reports ``refreshable_expired`` and
HOLDs, which is the safe direction.

Why a repeated ``refreshable_expired`` HOLD is not fixed by refreshing into
the staging directory
-------------------------------------------------------------------------
The obvious-looking fix for a run of repeated ``refreshable_expired`` HOLDs is
an "ephemeral" refresh: exchange the refresh token, put the new access token
only into the per-spawn staged credential copy (``runner._stage_credentials``
under ``policy.CREDENTIAL_STAGING_ROOT``, which is destroyed after the run),
and never touch the root-owned host file. That was evaluated and deliberately
NOT implemented. Three blockers, the first of which is decisive:

1. **Refresh-token rotation makes "ephemeral" actively destructive.** A
   refresh grant is entitled to return a NEW refresh token and invalidate the
   one presented, and nothing available here establishes that this credential's
   issuer does otherwise. If it rotates, an ephemeral refresh spends the host's
   only refresh token and leaves the host file holding a dead one — turning a
   recoverable ``refreshable_expired`` into an unrecoverable ``expired`` that
   needs a full interactive re-login, for every consumer of that credential and
   not just this plugin. Keeping the new token instead would mean writing it
   back to the host file, which is exactly the privileged credential write this
   plugin has never owned. There is no way to learn which behaviour applies
   except by spending the operator's live refresh token on the experiment, so
   the fail-closed HOLD is strictly safer than the "fix".
2. **The grant parameters are not in the credential.** The token endpoint and
   the client identity are not recorded in the credentials file and are not
   part of any contract this plugin can read. Hardcoding a guessed endpoint and
   client id would be inventing an authentication flow, not implementing one.
3. **It needs capabilities this module structurally does not have.** Calling an
   auth endpoint means network egress from the privileged host runner process,
   plus a write path for the staged file. This module is tripwired against
   precisely those — a source-level test asserts it can neither open a file for
   writing, spawn a child process, nor reach the network — and that tripwire is
   load-bearing, not incidental.

So the behaviour here is unchanged and stays fail-closed. The remedies are:

* **Re-authenticate the host Claude Code session.** That is what the HOLD's own
  operator text already says, and it leaves the credential owned by the tool
  that owns it.
* **Install a probe** via :func:`set_refresh_probe` if an operator wants this
  automated. The seam exists for this: the probe runs OUTSIDE this module, so
  an operator can drive the Claude Code CLI's own refresh logic — the code that
  legitimately owns the privileged write and knows the grant parameters —
  without this plugin ever hand-rolling a refresh or handling a token.
  :func:`preflight` then re-reads freshness from disk and proceeds on its own
  evidence. With no probe installed the HOLD stands, which is the current
  default.

What a failed preflight may and may not do
------------------------------------------
It may classify the problem as ``auth`` and HOLD the call (no spawn, session
stays gated). It must NOT clear or reset breaker state: a breaker opened by a
real observed failure stays open for its own cooldown, per class, and nothing
here resets anything on start. It also does not *open* the breaker — an
unreadable credentials file is not an observed rejection by the API, and
burning an hour of cooldown on it would be the same overreaction this module
exists to prevent.

Where this runs
---------------
``runner.run_worker`` calls :func:`preflight` after cwd validation and after
the already-open-breaker HOLD check, and BEFORE baseline evidence gathering
or any docker/Claude spawn. A failed preflight becomes
``runner._oauth_preflight_hold_payload`` — a HOLD with ``attempts`` 0, the
breaker reported exactly as it already was, and only the state name and the
``refresh_attempted`` boolean carried across (never a reason string, never a
token). Its failure class is ``runner.OAUTH_PREFLIGHT_FAILURE_CLASS``, which
is deliberately outside ``breaker.BREAKER_CLASSES`` so no "record the failure
class" path can convert it into a cooldown.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Callable, Dict, Optional

from . import policy as _policy

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# States
# ---------------------------------------------------------------------------

#: A usable access token with time left on it.
STATE_FRESH = "fresh"
#: Access token expired (or about to), but refresh metadata is present.
STATE_REFRESHABLE_EXPIRED = "refreshable_expired"
#: Access token expired and nothing to refresh with.
STATE_EXPIRED = "expired"
#: Parsed fine, but carries no usable OAuth material at all.
STATE_INVALID = "invalid"
#: Present but not parseable as the expected JSON object shape.
STATE_MALFORMED = "malformed"
#: No credentials file (or it cannot be opened safely).
STATE_MISSING = "missing"
#: The preflight itself failed unexpectedly.
STATE_ERROR = "error"

#: Treat a token with less than this left as already expired — a token that
#: expires mid-run is an auth failure we can still avoid.
FRESHNESS_MARGIN_SECONDS = 60

_REDACTED = "[redacted]"

#: Key names whose VALUES are secret. Used both to pick the metadata apart and
#: to scrub any of those values out of free text before it is returned.
_SECRET_KEYS = (
    "accessToken", "access_token", "refreshToken", "refresh_token",
    "idToken", "id_token", "token", "apiKey", "api_key",
)

_ACCESS_KEYS = ("accessToken", "access_token")
_REFRESH_KEYS = ("refreshToken", "refresh_token")
_EXPIRY_KEYS = ("expiresAt", "expires_at", "expiry", "expiresAtMs")
_OAUTH_SECTIONS = ("claudeAiOauth", "claude_ai_oauth", "oauth", "claudeAiOAuth")

#: Injection point for an operator-provided isolated refresh/probe. Called
#: with no arguments, at most once per :func:`preflight`, and expected to
#: return a mapping (``{"ok": bool, "reason": str}``); anything else is
#: treated as "the probe did not succeed". ``None`` means no probe is
#: installed, which is the default.
REFRESH_PROBE: Optional[Callable[[], Any]] = None


def set_refresh_probe(probe: Optional[Callable[[], Any]]) -> None:
    """Install (or clear) the isolated refresh/probe hook."""
    global REFRESH_PROBE
    REFRESH_PROBE = probe


# ---------------------------------------------------------------------------
# Reading freshness — non-secret metadata only
# ---------------------------------------------------------------------------


def redact(text: Any, secrets: Any = ()) -> str:
    """Return *text* as a string with every value in *secrets* replaced.

    Applied to every free-text reason that reaches a preflight result, so a
    token quoted back by a probe's own error message can never be surfaced.
    """
    out = text if isinstance(text, str) else ("" if text is None else str(text))
    for secret in secrets or ():
        if isinstance(secret, str) and len(secret) >= 4 and secret in out:
            out = out.replace(secret, _REDACTED)
    return out


def _read_bytes(path: str) -> Optional[bytes]:
    """Read the credentials file without following a symlink at the final
    component and without ever reading an unbounded amount."""
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        fd = os.open(path, flags)
    except (OSError, ValueError):
        # ValueError, not just OSError: an embedded NUL in the path is a
        # ValueError from ``os.open``, and ``read_freshness``'s "never
        # raises" contract has to hold for every rejection the open can
        # produce, not only the ones that arrive as OSError.
        return None
    try:
        return os.read(fd, _policy.MAX_CREDENTIAL_BYTES + 1)
    except OSError:
        return None
    finally:
        os.close(fd)


def _first(mapping: Dict[str, Any], keys) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def _oauth_section(parsed: Dict[str, Any]) -> Dict[str, Any]:
    for key in _OAUTH_SECTIONS:
        section = parsed.get(key)
        if isinstance(section, dict):
            return section
    return parsed


def _normalize_expiry(value: Any) -> Optional[float]:
    """Accept seconds or milliseconds since the epoch; reject anything else.

    Claude Code writes ``expiresAt`` in milliseconds; other tooling writes
    seconds. A value past the year ~5138 in seconds is certainly milliseconds.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if value <= 0:
        return None
    return float(value) / 1000.0 if value > 1e11 else float(value)


def _string_list(value: Any):
    if isinstance(value, (list, tuple)):
        return [item for item in value if isinstance(item, str)]
    if isinstance(value, str):
        return [value]
    return []


def read_freshness(
    path: Optional[str] = None, now: Optional[float] = None,
) -> Dict[str, Any]:
    """Non-secret freshness summary of the OAuth credentials file.

    Never raises, never returns or logs a token. Every field below is
    metadata: presence booleans, an expiry instant, seconds remaining, and
    the non-secret scope/subscription labels the CLI itself displays.
    """
    target = path or _policy.HOST_CREDENTIALS_PATH
    stamp = time.time() if now is None else float(now)
    result: Dict[str, Any] = {
        "state": STATE_MISSING,
        "present": False,
        "has_access_token": False,
        "has_refresh_token": False,
        "expires_at_epoch": None,
        "seconds_remaining": None,
        "expired": True,
        "scopes": [],
        "subscription_type": "",
        "reason": "no OAuth credentials file is present",
    }

    raw = _read_bytes(target)
    if raw is None:
        return result
    result["present"] = True

    if len(raw) > _policy.MAX_CREDENTIAL_BYTES:
        result["state"] = STATE_MALFORMED
        result["reason"] = "OAuth credentials file is larger than the permitted maximum"
        return result

    try:
        parsed = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        result["state"] = STATE_MALFORMED
        result["reason"] = "OAuth credentials file is not valid JSON"
        return result
    if not isinstance(parsed, dict):
        result["state"] = STATE_MALFORMED
        result["reason"] = "OAuth credentials file is not a JSON object"
        return result

    section = _oauth_section(parsed)
    if not isinstance(section, dict):
        result["state"] = STATE_MALFORMED
        result["reason"] = "OAuth credentials file has no OAuth object"
        return result

    access = _first(section, _ACCESS_KEYS)
    refresh = _first(section, _REFRESH_KEYS)
    expires_at = _normalize_expiry(_first(section, _EXPIRY_KEYS))

    result["has_access_token"] = isinstance(access, str) and bool(access.strip())
    result["has_refresh_token"] = isinstance(refresh, str) and bool(refresh.strip())
    result["expires_at_epoch"] = expires_at
    result["scopes"] = _string_list(section.get("scopes"))
    subscription = section.get("subscriptionType") or section.get("subscription_type")
    result["subscription_type"] = subscription if isinstance(subscription, str) else ""

    if not result["has_access_token"]:
        result["state"] = STATE_INVALID
        result["reason"] = "OAuth credentials carry no access token"
        return result

    if expires_at is None:
        # No expiry recorded at all: treat the token as unusable rather than
        # assuming it is eternal. An expiring credential with no expiry field
        # is exactly the case that produced surprise 401s.
        result["state"] = (
            STATE_REFRESHABLE_EXPIRED if result["has_refresh_token"] else STATE_INVALID
        )
        result["reason"] = "OAuth credentials carry no usable expiry timestamp"
        return result

    remaining = expires_at - stamp
    result["seconds_remaining"] = int(remaining)
    if remaining > FRESHNESS_MARGIN_SECONDS:
        result["expired"] = False
        result["state"] = STATE_FRESH
        result["reason"] = ""
        return result

    result["expired"] = True
    if result["has_refresh_token"]:
        result["state"] = STATE_REFRESHABLE_EXPIRED
        result["reason"] = "OAuth access token is expired but a refresh token is present"
    else:
        result["state"] = STATE_EXPIRED
        result["reason"] = "OAuth access token is expired and there is no refresh token"
    return result


def _secret_values(path: str):
    """The secret strings currently in the credentials file, for redaction
    only. Never returned to a caller, never logged."""
    raw = _read_bytes(path)
    if raw is None:
        return ()
    try:
        parsed = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return ()
    if not isinstance(parsed, dict):
        return ()
    section = _oauth_section(parsed)
    values = []
    for source in (parsed, section):
        if not isinstance(source, dict):
            continue
        for key in _SECRET_KEYS:
            value = source.get(key)
            if isinstance(value, str) and value:
                values.append(value)
    return tuple(values)


# ---------------------------------------------------------------------------
# The preflight itself
# ---------------------------------------------------------------------------


def _result(
    ok: bool,
    freshness: Dict[str, Any],
    reason: str,
    secrets,
    classification: Optional[str] = None,
    refresh_attempted: bool = False,
) -> Dict[str, Any]:
    return {
        "ok": ok,
        "state": freshness.get("state", STATE_ERROR),
        "classification": classification,
        "refresh_attempted": refresh_attempted,
        "reason": redact(reason, secrets),
        "freshness": freshness,
    }


def preflight(
    path: Optional[str] = None,
    now: Optional[float] = None,
    refresh_probe: Optional[Callable[[], Any]] = None,
) -> Dict[str, Any]:
    """Check the OAuth credential before the worker commits to a spawn.

    Returns ``{"ok", "state", "classification", "refresh_attempted",
    "reason", "freshness"}``. ``ok`` True means "the credential looks usable,
    proceed". ``ok`` False means HOLD: ``classification`` is ``"auth"`` so
    the caller can report it honestly, but the caller must NOT open or clear
    any breaker state on the strength of this — see the module docstring.

    A ``refreshable_expired`` credential gets AT MOST ONE isolated probe
    (:data:`REFRESH_PROBE`, or the *refresh_probe* argument for tests),
    followed by a fresh re-read of the credential from disk. Never raises.
    """
    target = path or _policy.HOST_CREDENTIALS_PATH
    try:
        secrets = _secret_values(target)
    except Exception:  # pragma: no cover - defensive
        secrets = ()

    try:
        freshness = read_freshness(target, now=now)
    except Exception:  # pragma: no cover - read_freshness already fails closed
        logger.warning("claude_worker: OAuth preflight could not read credentials", exc_info=True)
        return _result(
            False, {"state": STATE_ERROR}, "OAuth credential freshness could not be read",
            secrets, classification="auth",
        )

    state = freshness.get("state")
    if state == STATE_FRESH:
        return _result(True, freshness, "", secrets)

    if state != STATE_REFRESHABLE_EXPIRED:
        return _result(
            False, freshness, freshness.get("reason") or "OAuth credential is unusable",
            secrets, classification="auth",
        )

    probe = refresh_probe if refresh_probe is not None else REFRESH_PROBE
    if probe is None:
        return _result(
            False, freshness,
            "OAuth access token is expired and no isolated refresh probe is configured",
            secrets, classification="auth",
        )

    # Exactly one probe, on exactly this path. No retry loop lives here.
    try:
        outcome = probe()
    except Exception as exc:
        # Deliberately NOT ``exc_info=True``: a traceback embeds the raw
        # exception text, and an operator-installed probe talking to an auth
        # endpoint can easily quote the token it just sent back in its own
        # error message. The type plus a redacted message keeps the log
        # actionable without giving a token a route out through logging.
        logger.warning(
            "claude_worker: OAuth refresh probe raised %s: %s",
            type(exc).__name__, redact(exc, secrets),
        )
        outcome = {"ok": False, "reason": "refresh probe raised"}

    probe_reason = ""
    if isinstance(outcome, dict):
        probe_reason = redact(outcome.get("reason") or "", secrets)

    try:
        after = read_freshness(target, now=now)
    except Exception:  # pragma: no cover - defensive
        after = {"state": STATE_ERROR, "reason": "credential re-read failed"}

    if after.get("state") == STATE_FRESH:
        return _result(True, after, "", secrets, refresh_attempted=True)

    detail = probe_reason or after.get("reason") or "the credential is still expired"
    return _result(
        False, after,
        f"OAuth refresh probe did not restore a fresh credential: {detail}",
        secrets, classification="auth", refresh_attempted=True,
    )
