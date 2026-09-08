"""Immutable policy for claude_worker — the values configuration may never
widen.

Terra's review found every policy-critical value reachable from
``plugins.entries.claude-worker``: the in-scope chats, the platform, the
gated tool set, the two model identities, and the escalation cap. A typo or
a hostile edit in config.yaml could therefore gate an out-of-scope chat,
swap Sonnet and Opus, or turn one escalation into an unbounded retry loop.

This module is the single source of truth for those values, as module-level
literals. It reads NO configuration of its own. The only thing configuration
is allowed to do is turn the whole feature OFF (see
:func:`discord_scope_enabled`); it can never widen scope, and — since the
policy is now "every Discord-origin session" — it can no longer narrow scope
to a hand-picked list of channel ids either. The two hardcoded canary channel
ids are gone: scope is the PLATFORM, not a channel allowlist.

Project scope is likewise no longer a static, configured list of repo roots.
It is resolved dynamically, per request, from the canonical Git worktree root
of the requested cwd — see ``project.py``, which the gate and the runner both
call, so the gate can never block a repository the worker is not permitted to
run in (Terra's deadlock finding). The literals that resolution depends on
(the trusted git binary, its timeout, the system-sensitive roots that may
never be a project root) live here.
"""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# Discord scope — EVERY Discord-origin session, no channel allowlist
# ---------------------------------------------------------------------------

#: The only platform claude_worker gating ever applies to. Every chat on it —
#: guild channel, DM, and every thread under either — is in scope; threads are
#: covered because the platform, not the chat id, is the test. A chat id on
#: any other platform is never in scope.
DISCORD_PLATFORMS = frozenset({"discord"})


def is_discord_platform(platform: Any) -> bool:
    """True only for the exact literal platform value ``"discord"``."""
    return isinstance(platform, str) and platform in DISCORD_PLATFORMS


def discord_scope_enabled(cfg: Any) -> bool:
    """Whether the Discord write gate is active at all.

    This is a GLOBAL kill switch and nothing more. Configuration may turn the
    feature off entirely (``discord.enabled: false``, or the deprecated
    ``canary.enabled: false``), which is safe because it can only ever remove
    a restriction from *every* session at once — an operator cannot use it to
    quietly exempt one channel while leaving the rest gated. There is
    deliberately no channel selection: policy is ALL Discord, so any
    ``channel_ids`` a stale config still carries is inert data that this
    function never reads.

    Only the literal ``False`` disables. A missing/malformed section, or a
    truthy-looking string like ``"no"``, leaves the gate ON.
    """
    if not isinstance(cfg, dict):
        return True
    section = cfg.get("discord")
    if not isinstance(section, dict):
        # Deprecated alias, accepted for backward compatibility with configs
        # written while this feature was still a two-channel canary.
        section = cfg.get("canary")
    if not isinstance(section, dict):
        return True
    return section.get("enabled", True) is not False


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
#
# These two literals ARE the canonical claude_worker runtime — the model pair
# the sandbox image and the routing rules were proven against.
# ``claude-sonnet-5`` is the default for every task, unconditionally.
# ``claude-opus-5`` is selected ONLY when BOTH an allowed complexity
# classification (architecture/security/hard-debugging) AND explicit caller
# authorization (``allow_opus=True``) are present — ``complexity`` alone
# never authorizes it (see ``routing.choose_model``). There is exactly one
# spawn per ``claude_worker`` call: no failure-driven retry, no escalation,
# and the task text itself never influences model selection. Nothing in this
# hardening pass changes either model id, and neither is reachable from
# configuration. (Hermes' own Sol/Terra/Grok routing is a separate system and
# is untouched by this plugin.)
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "claude-sonnet-5"
#: Selected only via ``routing.choose_model``'s dual-authorization gate — an
#: allowed ``complexity`` value AND ``allow_opus=True``, both required. The
#: name is historical (this plugin no longer escalates on failure); it is
#: kept to avoid churn in every call site and test that already reads it.
ESCALATION_MODEL = "claude-opus-5"

#: Defense in depth only — the identities above are the contract.
MODEL_ALLOWLIST = frozenset({"claude-sonnet-5", "claude-opus-5"})

#: Exactly one spawn per call. Never configurable, and never widened by a
#: failure: there is no retry and no post-failure escalation.
MAX_ATTEMPTS = 1

#: Hard wall-clock budget for the one model attempt in a tool call. It stays
#: five minutes below the production gateway's 1,800-second timeout, leaving
#: headroom for evidence collection, telemetry, and the tool response.
#: Configuration may shorten the attempt but can never widen this total.
MAX_TOTAL_ATTEMPT_SECONDS = 1500

#: Bound on the operator-configured verification command (``verification.
#: command``), run once after a successful spawn in the same network-
#: isolated, credential-less container as any other verification command.
#: Fixed and non-configurable, like every other sandbox resource bound.
VERIFICATION_TIMEOUT_SECONDS = 300


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
# Dynamic project-root resolution literals (consumed by ``project.py``)
#
# There is no configured repo allowlist anymore. ``project.py`` resolves the
# canonical Git worktree root of whatever cwd was requested and hands the
# gate and the runner the SAME answer; these literals bound what that
# resolution is allowed to accept and how it is allowed to ask git.
# ---------------------------------------------------------------------------

#: A fixed, absolute path — never a bare ``"git"`` resolved off the caller's
#: ``$PATH``. ``project._git_toplevel`` additionally validates this exact
#: path and every parent directory as non-symlink, root-owned, and never
#: group/world-writable (``trust.validate_trusted_path_chain``) before it is
#: ever executed, and refuses to consult git at all if that fails.
GIT_BIN = "/usr/bin/git"

#: Hard wall-clock bound on the one ``git rev-parse`` argv the resolver ever
#: runs. Short on purpose: this is a local metadata read, not a network op.
GIT_RESOLVE_TIMEOUT_SECONDS = 10

#: Longest ancestor walk the resolver will do looking for a ``.git`` marker.
#: A real checkout is a handful of levels deep; this is a runaway backstop.
MAX_PROJECT_WALK_DEPTH = 64

#: Directories that may never themselves BE a project root, even if someone
#: puts a ``.git`` in them. Mounting any of these would hand the sandbox a
#: system directory, a whole home directory, or a parent holding many
#: unrelated projects, instead of one repository.
UNSAFE_PROJECT_ROOTS = frozenset({
    "/", "/bin", "/boot", "/dev", "/etc", "/home", "/lib", "/lib32", "/lib64",
    "/libx32", "/media", "/mnt", "/opt", "/proc", "/root", "/run", "/sbin",
    "/srv", "/sys", "/tmp", "/usr", "/var",
    # The worktree CONTAINER, never a project: mounting it would expose every
    # sibling project to a worker asked to touch exactly one of them.
    "/root/worktrees",
})

#: Prefixes under which nothing may ever be a project root, at any depth.
UNSAFE_PROJECT_PREFIXES = (
    "/bin", "/boot", "/dev", "/etc", "/lib", "/lib32", "/lib64", "/libx32",
    "/proc", "/run", "/sbin", "/sys", "/usr", "/var",
)


# ---------------------------------------------------------------------------
# Terminal bypass — the canonical path to Claude is the claude_worker tool
# ---------------------------------------------------------------------------

#: Tools whose free-form command string could otherwise reach a host Claude
#: CLI and route around the whole sandbox (see ``terminal_guard.py``).
TERMINAL_TOOLS = frozenset({"terminal"})

#: The executable basename that IS the Claude CLI. A token equal to this, or
#: any path whose final component is this, is a direct invocation.
CLAUDE_CLI_BASENAME = "claude"
