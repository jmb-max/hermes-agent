"""Config loading for the claude-worker plugin.

Everything lives under ``plugins.entries.claude-worker`` in config.yaml —
``hermes_cli/config.py`` has no strict top-level key whitelist, so no core
config change is required to add this namespace.

This is a *validating* loader, not a deep merge of arbitrary user data.
Terra's review showed why: a deep merge hands configuration a direct write
to whatever the code later reads, so the canary ids, the model identities,
the gated tool set, the escalation cap, and the Claude tool allowlist were
all reachable from config.yaml. Those values now live in ``policy.py`` and
are not part of this namespace at all; here, only the keys in ``_SCHEMA``
survive, each coerced to its expected type or dropped back to its default.
Anything unknown — including a key a future version of this plugin might
add — is discarded rather than passed through.

Operational knobs only:

* ``discord.enabled``       — the ONE global kill switch. Scope is now every
  Discord-origin session (guild channel, DM, and every thread under either),
  so the only thing configuration may do is turn the whole feature off for
  every session at once. It can never widen scope, and it can no longer
  narrow it to hand-picked channels either.
* ``isolation.timeout_seconds`` — clamped to sane bounds.
* ``review`` / ``verification`` / ``breaker.cooldown_seconds``.
* ``oauth.auto_refresh`` / ``oauth.refresh_timeout_seconds`` — whether the
  isolated host-CLI refresh probe (``oauth_refresh.py``) runs at all, and how
  long the host CLI may run, clamped to
  ``[MIN_REFRESH_TIMEOUT_SECONDS, MAX_REFRESH_TIMEOUT_SECONDS]``. Everything
  that probe actually executes — the binary, the lock path, the argv, the
  child environment, and the prompt — is a literal in ``oauth_refresh.py``
  and is deliberately NOT part of this namespace: a config-supplied command
  or credential path would be a direct route into a privileged host
  subprocess.

Three keys are still PARSED for backward compatibility and are otherwise
INERT — a stale config.yaml keeps loading, but none of them can change
behavior:

* ``canary.enabled`` — DEPRECATED alias for ``discord.enabled``. It still
  works (an operator who disabled the old two-channel canary keeps the
  feature off), and ``_validate`` folds it into ``discord.enabled`` so
  ``policy.discord_scope_enabled`` sees one effective answer.
* ``canary.channel_ids`` — DEPRECATED and fully inert. Scope is the PLATFORM
  now, so there is no channel allowlist for this to narrow (or widen);
  ``policy.py`` never reads it and neither does ``canary.py``.
* ``gate.repo_roots`` — DEPRECATED and fully inert. The gate and the runner
  both derive scope from the request via ``project.resolve_project_root*``
  (the canonical Git worktree root of the requested path), so a configured
  list can neither authorise a repository nor exclude one. Keeping it inert
  rather than authoritative is what removed the mount hazard: whatever this
  key named used to be what got bind-mounted, so an entry like
  ``/root/worktrees`` handed every container every sibling checkout.
* ``telemetry.path`` — DEPRECATED and inert: ``telemetry.append_record``
  always writes to the fixed ``get_hermes_home()/claude-worker/telemetry.jsonl``
  destination and ignores any ``configured_path`` it is passed, so a
  config-supplied value can never redirect telemetry to an attacker-chosen
  file.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List

PLUGIN_KEY = "claude-worker"

MIN_TIMEOUT_SECONDS = 30
MAX_TIMEOUT_SECONDS = 3600

#: Bounds on the host-CLI refresh turn (``oauth_refresh.refresh_probe``). The
#: default is generous enough for one cheap turn on a slow link; the floor
#: keeps a mistyped ``0`` from making every refresh fail before it starts, and
#: the ceiling keeps a privileged host subprocess from being told to run
#: effectively forever.
DEFAULT_REFRESH_TIMEOUT_SECONDS = 45
MIN_REFRESH_TIMEOUT_SECONDS = 10
MAX_REFRESH_TIMEOUT_SECONDS = 120

DEFAULT_CONFIG: Dict[str, Any] = {
    # The global kill switch. ``canary`` below is the deprecated alias kept
    # for old config files; ``_validate`` keeps the two in sync so nothing
    # downstream has to know which one an operator wrote.
    "discord": {
        "enabled": True,
    },
    "canary": {
        "enabled": True,
        # DEPRECATED, inert — see the module docstring.
        "channel_ids": [],
    },
    "gate": {
        # DEPRECATED, inert — see the module docstring.
        "repo_roots": [],
    },
    "breaker": {
        "cooldown_seconds": {"auth": 3600, "rate": 900, "extra_usage": 3600},
    },
    "isolation": {
        "timeout_seconds": 900,
    },
    "review": {
        "enabled": True,
        "min_changed_files": 3,
    },
    "verification": {
        "enabled": False,
        "command": [],
    },
    "telemetry": {
        "path": "",
    },
    "oauth": {
        # On by default: an expired credential that a refresh would fix must
        # not HOLD a session just because nobody set a knob.
        "auto_refresh": True,
        "refresh_timeout_seconds": DEFAULT_REFRESH_TIMEOUT_SECONDS,
    },
}

_COOLDOWN_CLASSES = ("auth", "rate", "extra_usage")


def _load_raw_config() -> Dict[str, Any]:
    """Load the full Hermes config.yaml. Isolated as its own function so
    tests can monkeypatch it without touching disk."""
    from hermes_cli.config import load_config

    return load_config() or {}


def _bool(value: Any, default: bool) -> bool:
    """Only a real bool counts. A truthy string like ``"no"`` is a mistake,
    and for a safety toggle the mistake must not read as "off"."""
    return value if isinstance(value, bool) else default


def _int(value: Any, default: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        return default
    return value


def _str_list(value: Any) -> List[str]:
    if not isinstance(value, (list, tuple)):
        return []
    return [entry for entry in value if isinstance(entry, str) and entry]


def _clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


def _validate(entry: Dict[str, Any]) -> Dict[str, Any]:
    cfg = copy.deepcopy(DEFAULT_CONFIG)

    # One effective kill-switch answer from two accepted spellings. The
    # deprecated ``canary.enabled`` is read first so an existing config that
    # turned the old two-channel canary off keeps the feature off; an
    # explicit ``discord.enabled`` bool wins when both are present.
    enabled = True
    canary = entry.get("canary")
    if isinstance(canary, dict):
        enabled = _bool(canary.get("enabled"), enabled)
        # Parsed, never read — kept only so a stale file still validates.
        cfg["canary"]["channel_ids"] = _str_list(canary.get("channel_ids"))

    discord = entry.get("discord")
    if isinstance(discord, dict):
        enabled = _bool(discord.get("enabled"), enabled)

    cfg["discord"]["enabled"] = enabled
    cfg["canary"]["enabled"] = enabled

    gate = entry.get("gate")
    if isinstance(gate, dict):
        # Parsed, never read — project scope is resolved dynamically from
        # the request by ``project.py``, shared by the gate and the runner.
        cfg["gate"]["repo_roots"] = _str_list(gate.get("repo_roots"))

    breaker = entry.get("breaker")
    if isinstance(breaker, dict):
        cooldowns = breaker.get("cooldown_seconds")
        if isinstance(cooldowns, dict):
            for cls in _COOLDOWN_CLASSES:
                cfg["breaker"]["cooldown_seconds"][cls] = _clamp(
                    _int(cooldowns.get(cls), DEFAULT_CONFIG["breaker"]["cooldown_seconds"][cls]),
                    0, 24 * 3600,
                )

    isolation = entry.get("isolation")
    if isinstance(isolation, dict):
        cfg["isolation"]["timeout_seconds"] = _clamp(
            _int(isolation.get("timeout_seconds"), DEFAULT_CONFIG["isolation"]["timeout_seconds"]),
            MIN_TIMEOUT_SECONDS, MAX_TIMEOUT_SECONDS,
        )

    review = entry.get("review")
    if isinstance(review, dict):
        cfg["review"]["enabled"] = _bool(review.get("enabled"), True)
        cfg["review"]["min_changed_files"] = _clamp(
            _int(review.get("min_changed_files"), DEFAULT_CONFIG["review"]["min_changed_files"]),
            1, 10_000,
        )

    verification = entry.get("verification")
    if isinstance(verification, dict):
        cfg["verification"]["enabled"] = _bool(verification.get("enabled"), False)
        cfg["verification"]["command"] = _str_list(verification.get("command"))

    telemetry = entry.get("telemetry")
    if isinstance(telemetry, dict):
        path = telemetry.get("path")
        cfg["telemetry"]["path"] = path if isinstance(path, str) else ""

    oauth = entry.get("oauth")
    if isinstance(oauth, dict):
        # Only a real ``False`` turns the probe off — a truthy-looking
        # ``"no"`` must not read as "off" when "off" means an expired
        # credential HOLDs a session that could have been recovered. Every
        # other key here is dropped on the floor: nothing configuration
        # supplies may reach the privileged host subprocess.
        cfg["oauth"]["auto_refresh"] = _bool(oauth.get("auto_refresh"), True)
        cfg["oauth"]["refresh_timeout_seconds"] = _clamp(
            _int(oauth.get("refresh_timeout_seconds"), DEFAULT_REFRESH_TIMEOUT_SECONDS),
            MIN_REFRESH_TIMEOUT_SECONDS, MAX_REFRESH_TIMEOUT_SECONDS,
        )

    return cfg


def load_plugin_config() -> Dict[str, Any]:
    """Return the plugin's effective config: validated operational knobs only.

    Fails closed to defaults (never raises) — a broken config.yaml must not
    crash tool dispatch or plugin hooks, and must never be able to widen
    policy (see ``policy.py``).
    """
    try:
        raw = _load_raw_config()
    except Exception:
        raw = {}
    entries: Any = {}
    if isinstance(raw, dict):
        plugins_section = raw.get("plugins")
        if isinstance(plugins_section, dict):
            entries = plugins_section.get("entries") or {}
    entry = entries.get(PLUGIN_KEY) if isinstance(entries, dict) else None
    if not isinstance(entry, dict):
        entry = {}
    try:
        return _validate(entry)
    except Exception:
        return copy.deepcopy(DEFAULT_CONFIG)
