"""claude_worker — delegate coding tasks to an isolated Claude Code CLI
subprocess, authenticated via the installed Claude Code OAuth session
(never an API key).

Wires together the modules in this package:

* ``policy.py``    — the immutable layer: the two canary Discord channel
                      ids, the platform, the mandatory gated tool set, the
                      two model identities, the structural attempt cap, the
                      sandbox literals, and the one canonical repo-root
                      resolution shared by the gate and the runner. Reads no
                      configuration; configuration can only narrow it.
* ``routing.py``   — Sonnet-default auto routing; Opus for architecture /
                      security / hard-debugging, or exactly one controlled
                      escalation after a Sonnet non-breaker failure.
* ``breaker.py``    — a per-failure-class (auth/rate/extra_usage) circuit
                       breaker with cooldown; open means no spawn, no retry.
* ``runner.py``     — the isolated spawn itself: explicit-allowlist child
                       env, ``--strict-mcp-config`` with an empty generated
                       MCP config, a scoped ``--allowed-tools`` list, and
                       cwd pinned to an allowlisted repo root.
* ``canary.py``     — Discord canary-channel eligibility (including
                       threads, via ``pre_gateway_dispatch``).
* ``gate.py``       — blocks direct ``patch``/``write_file``/``skill_manage``
                       edits in canary sessions, scoped to repo roots, until
                       a claude_worker run has succeeded there.
* ``telemetry.py``  — one redacted JSONL record per invocation.
* ``review.py``     — advisory Terra review via the existing auxiliary-task
                       system for substantial successful changes.

See the implementation plan for the full requirements this satisfies. This
plugin never registers a model provider and never writes ``model.default``
— routing is a closed two-model allowlist enforced in ``routing.py``.
"""

from __future__ import annotations

from typing import Any, Dict

from . import breaker, canary, config, gate, policy, review, routing, runner, telemetry

TOOL_SCHEMA: Dict[str, Any] = {
    "name": "claude_worker",
    "description": (
        "Delegate a coding task to an isolated Claude Code CLI subprocess, "
        "authenticated via the installed Claude Code OAuth session (never "
        "an API key). Auto-routes between claude-sonnet-5 (default) and "
        "claude-opus-5 (architecture/security/hard-debugging tasks, or one "
        "controlled escalation after a Sonnet failure). Runs isolated: an "
        "empty, strict MCP config, a scoped tool allowlist, and a cwd "
        "pinned to an allowlisted repo root. Use this instead of direct "
        "file edits when a canary session requires it."
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
                    "Absolute path to the repo root to work in. Must "
                    "resolve inside a configured claude-worker repo root."
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
