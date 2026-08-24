"""claude_worker — delegate coding tasks to an isolated Claude Code CLI
subprocess, authenticated via the installed Claude Code OAuth session
(never an API key).

Wires together the modules in this package:

* ``policy.py``    — the immutable layer: the in-scope PLATFORM (Discord,
                      every session on it — there is no channel allowlist),
                      the mandatory gated tool set, the two model identities,
                      the structural attempt cap, the sandbox literals, and
                      the literals the dynamic project-root resolver is bound
                      by. Reads no configuration; configuration can only turn
                      the whole feature off.
* ``project.py``   — the ONE dynamic, safe Git-worktree-root resolver the
                      gate and the runner share, so they can never disagree
                      about a request's repository and deadlock a session.
                      Rejects relative/non-existent paths, anything outside a
                      real (non-bare) Git worktree, symlink escapes, and
                      system-sensitive or multi-project roots. The resolved
                      root is the only directory ever bind-mounted.
* ``routing.py``   — Sonnet-default auto routing; Opus for architecture /
                      security / hard-debugging, or exactly one controlled
                      escalation after a Sonnet non-breaker failure.
* ``breaker.py``    — a per-failure-class (auth/rate/extra_usage) circuit
                       breaker with cooldown; open means no spawn, no retry.
* ``trust.py``      — root-owned/non-symlink/non-world-writable path-chain
                       validation, shared by the docker binary, the git
                       binary, and the OAuth credentials file.
* ``oauth.py``      — the pre-spawn OAuth credential freshness preflight, run
                       BEFORE the auth breaker could ever be opened. Returns
                       non-secret metadata only (never a token), performs at
                       most one isolated refresh probe, and HOLDs rather than
                       opening, clearing, or resetting any breaker class.
* ``oauth_refresh.py`` — the ONE implementation of that refresh probe, kept
                       out of ``oauth.py`` so that module keeps its
                       no-write/no-subprocess tripwire. It delegates the
                       refresh to the trusted absolute host Claude CLI (fixed
                       argv, no tools, empty strict MCP config, one turn,
                       minimal literal env, isolated lock, throwaway cwd,
                       bounded timeout) and never reads or writes a token
                       itself. ``register`` installs it — or clears the seam
                       when ``oauth.auto_refresh`` is off.
* ``runner.py``     — the isolated spawn itself: explicit-allowlist child
                       env, ``--strict-mcp-config`` with an empty generated
                       MCP config, a scoped ``--allowed-tools`` list, and cwd
                       pinned to the dynamically resolved Git worktree root,
                       which is also the single writable mount.
* ``canary.py``     — Discord-origin session eligibility. Scope is the
                       PLATFORM: every guild channel, every DM, and every
                       thread under either. Tri-state and fail-closed —
                       ``None`` means "could not confirm", never "not
                       Discord".
* ``terminal_guard.py`` — detects a ``terminal`` command that would invoke
                       the Claude CLI directly on the host and bypass the
                       sandbox entirely.
* ``gate.py``       — blocks direct ``patch``/``write_file``/``skill_manage``
                       edits in a Discord-origin session, scoped to the
                       dynamically resolved Git root, for the whole session:
                       the worker is the coder, and a successful run releases
                       nothing. The one release is an explicit Terra fallback
                       that actually returned a result. Host Claude CLI
                       invocations through ``terminal`` are blocked
                       unconditionally.
* ``telemetry.py``  — one redacted JSONL record per invocation.
* ``review.py``     — advisory Terra review via the existing auxiliary-task
                       system for substantial successful changes.

See ``README.md`` in this directory for the operator-facing summary of the
global Discord scope and the dynamic single-repository mount. This plugin
never registers a model provider and never writes ``model.default`` —
routing is a closed two-model allowlist enforced in ``routing.py``.
"""

from __future__ import annotations

from typing import Any, Dict

from . import (
    breaker,
    canary,
    config,
    gate,
    oauth,
    oauth_refresh,
    policy,
    project,
    review,
    routing,
    runner,
    telemetry,
    terminal_guard,
    trust,
)

TOOL_SCHEMA: Dict[str, Any] = {
    "name": "claude_worker",
    "description": (
        "Delegate a coding task to an isolated Claude Code CLI subprocess, "
        "authenticated via the installed Claude Code OAuth session (never "
        "an API key). Auto-routes between claude-sonnet-5 (default) and "
        "claude-opus-5 (architecture/security/hard-debugging tasks, or one "
        "controlled escalation after a Sonnet failure). Runs isolated: an "
        "empty, strict MCP config, a scoped tool allowlist, and a cwd "
        "pinned to the canonical Git worktree root resolved from the "
        "requested cwd — that one repository is the only directory mounted "
        "into the sandbox. If the host OAuth credential is missing or "
        "expired the call HOLDs before spawning anything "
        "(failure_class auth_preflight) and the host session must be "
        "re-authenticated. Use this instead of direct file edits: in a "
        "Discord-origin session (any channel, DM, or thread) direct "
        "patch/write_file/skill_manage edits inside a Git repository are "
        "blocked for the whole session and every code change goes through "
        "this tool. A successful run does NOT hand back direct edit access — "
        "send the next change to the worker too."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "task": {
                "type": "string",
                "description": "The coding task to delegate to the worker.",
            },
            "cwd": {
                "type": "string",
                "description": (
                    "Absolute path to work in. Must be an existing "
                    "directory inside a real (non-bare) Git worktree; the "
                    "worker resolves that worktree's canonical root itself "
                    "and mounts exactly that one repository. Relative "
                    "paths, non-Git directories, bare repositories, and "
                    "system-sensitive or multi-project roots are refused."
                ),
            },
            "complexity": {
                "type": "string",
                "enum": ["architecture", "security", "hard_debugging"],
                "description": (
                    "Optional explicit complexity hint that forces "
                    "claude-opus-5 routing regardless of keyword detection."
                ),
            },
            "allow_terra_fallback": {
                "type": "boolean",
                "description": (
                    "Opt in to the explicit Terra fallback when the worker is "
                    "unavailable (breaker open). Default false: the session "
                    "stays on HOLD. When true and Terra actually returns "
                    "guidance, the result carries fallback_ready and direct "
                    "edits in this repo are released for the session; if Terra "
                    "does not answer, the HOLD stands."
                ),
            },
        },
        "required": ["task", "cwd"],
    },
}


def register(ctx) -> None:
    ctx.register_tool(
        name="claude_worker",
        toolset="claude_worker",
        schema=TOOL_SCHEMA,
        handler=runner.run_worker,
        description=TOOL_SCHEMA["description"],
        emoji="🤖",
    )

    # The isolated host-CLI refresh probe. Deterministic on every call: it
    # installs exactly this one probe when ``oauth.auto_refresh`` is on and
    # clears the seam when it is off, so a reload can never stack probes or
    # leave a stale one behind. Not wrapped in a try/except: the loader it
    # consults already fails closed to its defaults, and silently swallowing
    # a failure here would leave whatever was installed before in place.
    oauth_refresh.install_refresh_probe()

    ctx.register_hook("pre_gateway_dispatch", canary.compute_and_cache_eligibility)
    ctx.register_hook("pre_tool_call", gate.on_pre_tool_call)
    ctx.register_hook("post_tool_call", gate.on_post_tool_call)

    try:
        ctx.register_auxiliary_task(
            key=review.AUX_TASK_KEY,
            display_name="Claude Worker Review (Terra)",
            description=(
                "Advisory secondary review of claude_worker's changes, "
                "run via the auxiliary-task system (no new provider)."
            ),
            defaults=dict(review.AUX_TASK_DEFAULTS),
        )
    except Exception:
        # Auxiliary task registration is a nice-to-have (advisory review
        # config surface); never let it block tool/hook registration.
        pass
