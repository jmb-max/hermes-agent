"""Immutable policy for claude_worker — the values configuration may never
widen.

Terra's review found every policy-critical value reachable from
``plugins.entries.claude-worker``: canary channel ids, the platform, the
gated tool set, the two model identities, and the escalation cap. A typo or
a hostile edit in config.yaml could therefore gate a non-canary channel,
swap Sonnet and Opus, or turn one escalation into an unbounded retry loop.

This module is the single source of truth for those values, as module-level
literals. It reads NO configuration of its own. The only thing configuration
is allowed to do is *narrow*: disable the canary, or select a subset of the
policy's channel ids (see :func:`enabled_canary_channel_ids`). Anything a
caller passes that is not already inside the policy is discarded, never
unioned in.

``canonical_repo_roots`` / ``resolve_within_roots`` live here for the same
reason: the gate and the runner must share ONE root resolution, so the gate
can never block a repository the worker is not permitted to run in (Terra's
deadlock finding).
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Canary scope — exactly two Discord channels, forever
# ---------------------------------------------------------------------------

#: The only chats claude_worker gating may ever apply to. Threads are covered
#: via their ``parent_chat_id`` (see ``canary.py``), not by adding thread ids.
CANARY_CHANNEL_IDS = frozenset({
    "1527706694665113670",
    "1501268569697026140",
})

#: The only platform. A matching chat id on any other platform is not a canary.
CANARY_PLATFORMS = frozenset({"discord"})


def is_canary_platform(platform: Any) -> bool:
    """True only for the exact literal platform value ``"discord"``."""
    return isinstance(platform, str) and platform in CANARY_PLATFORMS


def enabled_canary_channel_ids(cfg: Optional[Dict[str, Any]]) -> frozenset:
    """Return the policy channel ids currently in force.

    Configuration may *disable* (``canary.enabled: false``) or *select a
    subset* (``canary.channel_ids``); an id outside the policy is dropped
    rather than added, and an absent/malformed selection means "all policy
    channels" — the defaults activate the intended canary with no config at
    all.
    """
    canary_cfg = cfg.get("canary") if isinstance(cfg, dict) else None
    if not isinstance(canary_cfg, dict):
        return CANARY_CHANNEL_IDS

    if canary_cfg.get("enabled", True) is False:
        return frozenset()

    selection = canary_cfg.get("channel_ids")
    if not isinstance(selection, (list, tuple, set, frozenset)):
        return CANARY_CHANNEL_IDS
    if not selection:
        return CANARY_CHANNEL_IDS

    selected = {value for value in selection if isinstance(value, str)}
    # Intersection, never union: config narrows the policy or does nothing.
    return frozenset(selected & set(CANARY_CHANNEL_IDS))


# ---------------------------------------------------------------------------
# The mandatory write gate
# ---------------------------------------------------------------------------

#: Tools whose repo-scoped writes are gated in a confirmed canary session.
GATED_TOOLS = frozenset({"patch", "write_file", "skill_manage"})

#: Every ``skill_manage`` action that can mutate a repo file (the tool's own
#: schema enum — see ``tools/skill_manager_tool.py``). All of them are gated;
#: anything else is a read and is out of scope.
SKILL_MANAGE_MUTATING_ACTIONS = frozenset({
    "create", "edit", "patch", "delete", "write_file", "remove_file",
})


# ---------------------------------------------------------------------------
# Models and the structural attempt cap
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "claude-sonnet-5"
ESCALATION_MODEL = "claude-opus-5"

#: Defense in depth only — the identities above are the contract.
MODEL_ALLOWLIST = frozenset({"claude-sonnet-5", "claude-opus-5"})

#: Attempt 0 (Sonnet) plus at most one escalation (Opus). Never configurable.
MAX_ATTEMPTS = 2

#: Hard wall-clock budget shared by every model attempt in one tool call.
#: It stays five minutes below the production gateway's 1,800-second timeout,
#: leaving headroom for evidence collection, telemetry, and the tool response.
#: Configuration may shorten individual attempts but can never widen this total.
MAX_TOTAL_ATTEMPT_SECONDS = 1500


# ---------------------------------------------------------------------------
# Sandbox literals (consumed by the isolated runner)
# ---------------------------------------------------------------------------

#: A fixed, absolute path — never a bare command name resolved off the
#: caller's ``$PATH``. A bare ``"docker"`` would let a hostile PATH entry
#: (or any earlier-matching directory) substitute an attacker-controlled
#: binary for the real one; ``runner._validate_docker_binary_trust``
#: additionally verifies, at runtime, that this exact path and its parent
#: directories are non-symlink, root-owned, and never group/world-writable
#: before it is ever executed.
DOCKER_BIN = "/usr/bin/docker"

#: The only endpoint the docker CLI is ever told to talk to — a fixed literal,
#: never read from the ambient ``DOCKER_HOST``, so a hostile or accidental
#: environment variable can never redirect the sandbox at a remote or
#: attacker-controlled daemon.
DOCKER_HOST_ENDPOINT = "unix:///var/run/docker.sock"

CLAUDE_CLI_VERSION = "2.1.237"
SANDBOX_IMAGE_TAG = f"claude-worker-sandbox:{CLAUDE_CLI_VERSION}"

#: The tag above is a human-friendly label, not a security boundary — it can
#: be retagged onto a different image. Every actual container invocation
#: (run/help/verifier) pins the immutable content-addressed id instead;
#: preflight inspects the tag once, requires it to resolve to exactly this
#: id, and only then is the id considered trustworthy to run. This value is
#: generated from the locally built, Node-22-based sandbox image during
#: activation and is deliberately a policy literal rather than config.
SANDBOX_IMAGE_ID = "sha256:46dc23aaa53c845dacb081dbd71d865a1772fdb2af51fb28ee8f4bfa35f8dc80"
SANDBOX_IMAGE_ID_CONFIGURED = True

#: The worker edits files. It does not run commands, fetch URLs, or spawn
#: sub-agents — those are the capabilities that let a worker leave the repo.
CLAUDE_ALLOWED_TOOLS = ("Read", "Edit", "Write", "Glob", "Grep")
CLAUDE_DENIED_TOOLS = (
    "Bash", "BashOutput", "KillShell", "WebFetch", "WebSearch",
    "Agent", "Task", "NotebookEdit",
)

PERMISSION_MODE = "acceptEdits"

#: Flags the sandboxed CLI must support, or the runner refuses to spawn.
REQUIRED_CLI_FLAGS = ("--strict-mcp-config", "--settings", "--setting-sources")

#: Container resource bounds — every spawn, worker or verification, is
#: capped so a runaway or hostile process inside the sandbox cannot exhaust
#: the host.
PIDS_LIMIT = 256
MEMORY_LIMIT = "2g"
CPU_LIMIT = "2"

CONTAINER_WORKDIR = "/workspace"
CONTAINER_HOME = "/home/worker"
CONTAINER_CREDENTIALS_PATH = f"{CONTAINER_HOME}/.claude/.credentials.json"
CONTAINER_CLAUDE_CONFIG_DIR = f"{CONTAINER_HOME}/.claude"

#: The one host path ever mounted into the sandbox besides the repo itself —
#: a fixed literal, not derived from the caller's environment, so the mount
#: source can never be redirected by anything task- or config-controlled.
HOST_CREDENTIALS_PATH = "/root/.claude/.credentials.json"

#: The fixed, non-root identity every sandbox container — the worker spawn
#: AND the verifier alike — always runs as, matching the ``USER worker``
#: (uid 10001) baked into ``sandbox/Dockerfile``. The host runner process
#: itself must run as root (it is the only identity that can read the
#: root-owned ``HOST_CREDENTIALS_PATH``), so mirroring the *caller's* uid —
#: the previous behavior — actually ran the container as root. Neither
#: value is ever configurable: an operator changing this would silently
#: hand every worker/verifier container root inside its own namespace.
SANDBOX_UID = 10001
SANDBOX_GID = 10001

#: Fixed, root-owned, mode-0700 staging root for the OAuth credentials file
#: copy made before every spawn (see ``runner._stage_credentials``). The
#: original ``HOST_CREDENTIALS_PATH`` is root-owned and unreadable by
#: ``SANDBOX_UID``, so it is never mounted directly; only a staged copy
#: re-owned to ``SANDBOX_UID``/``SANDBOX_GID`` ever is. A fixed literal, not
#: derived from the caller's environment, for the same reason
#: ``HOST_CREDENTIALS_PATH`` is.
CREDENTIAL_STAGING_ROOT = "/run/hermes-claude-worker"

#: Refuse to stage (and therefore refuse to spawn) an OAuth credentials file
#: larger than this. A legitimate credentials file is a few KB of JSON; a
#: bound this generous is purely a sanity/DoS backstop, not a functional
#: limit.
MAX_CREDENTIAL_BYTES = 1024 * 1024


# ---------------------------------------------------------------------------
# The one canonical repo-root resolution, shared by gate and runner
# ---------------------------------------------------------------------------


def canonical_repo_roots(cfg: Optional[Dict[str, Any]]) -> List[str]:
    """Resolve ``gate.repo_roots`` into a deduplicated list of real paths.

    Symlinked, relative, trailing-slash, and duplicate entries all collapse
    to one canonical form; entries that are not existing directories are
    dropped. There is deliberately no ``.git``-ancestor fallback: scope is
    exactly what the operator configured.
    """
    gate_cfg = cfg.get("gate") if isinstance(cfg, dict) else None
    raw = gate_cfg.get("repo_roots") if isinstance(gate_cfg, dict) else None
    if not isinstance(raw, (list, tuple)):
        return []

    roots: List[str] = []
    for entry in raw:
        if not isinstance(entry, str) or not entry.strip():
            continue
        try:
            real = os.path.realpath(entry)
        except (OSError, ValueError):
            continue
        if not os.path.isdir(real):
            continue
        if real not in roots:
            roots.append(real)
    return roots


def resolve_within_roots(path: Any, roots: List[str]) -> Optional[str]:
    """Return the canonical root *path* lives under, or ``None``.

    ``realpath`` runs BEFORE the containment check, so a symlink planted
    inside a root that points outside it resolves out of scope instead of
    smuggling access in. Containment is segment-wise, so ``/repo-evil`` is
    not "inside" ``/repo``.
    """
    if not isinstance(path, str) or not path:
        return None
    try:
        real = os.path.realpath(path)
    except (OSError, ValueError):
        return None
    for root in roots or []:
        if not isinstance(root, str) or not root:
            continue
        if real == root or real.startswith(root.rstrip(os.sep) + os.sep):
            return root
    return None
