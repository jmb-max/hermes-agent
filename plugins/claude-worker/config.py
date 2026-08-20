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

* ``canary.enabled`` / ``canary.channel_ids`` — disable, or select a subset
  of the policy channels (``policy.enabled_canary_channel_ids`` enforces
  that it can only ever narrow).
* ``gate.repo_roots``       — where the worker may run and the gate applies.
* ``isolation.timeout_seconds`` — clamped to sane bounds.
* ``review`` / ``verification`` / ``breaker.cooldown_seconds``.

``telemetry.path`` is still parsed here for backward compatibility but is
DEPRECATED and inert: ``telemetry.append_record`` always writes to the fixed
``get_hermes_home()/claude-worker/telemetry.jsonl`` destination and ignores
any ``configured_path`` it is passed, so a config-supplied value can never
redirect telemetry to an attacker-chosen file.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List

PLUGIN_KEY = "claude-worker"

MIN_TIMEOUT_SECONDS = 30
MAX_TIMEOUT_SECONDS = 3600

DEFAULT_CONFIG: Dict[str, Any] = {
    "canary": {
        "enabled": True,
        "channel_ids": [],
    },
    "gate": {
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

    canary = entry.get("canary")
    if isinstance(canary, dict):
        cfg["canary"]["enabled"] = _bool(canary.get("enabled"), True)
        cfg["canary"]["channel_ids"] = _str_list(canary.get("channel_ids"))

    gate = entry.get("gate")
    if isinstance(gate, dict):
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
