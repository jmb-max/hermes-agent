"""Auto model routing for the claude_worker tool (requirement 2).

Sonnet is ALWAYS the default. Opus is used ONLY when the caller passes an
explicit, non-null ``complexity`` argument that exactly matches one of the
values the tool schema already declares (``architecture`` / ``security`` /
``hard_debugging``).

Nothing here is configurable. Terra's review found the contract mutable in
two directions at once: ``models.default``/``models.escalated`` could be
swapped, and ``routing.escalate_after_failures`` fed straight into the
runner's spawn loop, so a value of 2 bought a third spawn. The model
identities are now literals from ``policy.py``.

There is no failure-driven escalation of any kind, and there never will be a
second spawn to escalate on: a Sonnet run that fails — for ANY reason
(timeout, nonzero exit, isolation refusal, an auth/preflight problem, a
quota/breaker-class failure, or anything else) — returns that failure as-is.
``choose_model`` therefore takes no ``attempt`` counter and no
``previous_failure_class`` input; there is nothing for either to feed.

There is also no keyword/heuristic scan of ``task`` text. That used to exist
and could silently escalate a run to Opus based on vocabulary alone (a task
that merely mentioned "security" or "race condition"); the content of
``task`` now never influences routing at all. ``task`` is accepted only
because it is part of the caller-facing contract and may be useful context
for future callers — it is read nowhere in this module.

Deliberately NOT an LLM call: the whole classification is one dict lookup
against a fixed 3-value set, so routing stays fast, testable, and
side-effect free.
"""

from __future__ import annotations

from typing import Optional, Tuple

from . import policy as _policy

MODEL_ALLOWLIST = _policy.MODEL_ALLOWLIST

_COMPLEXITY_CATEGORIES = frozenset({"architecture", "security", "hard_debugging"})


class ModelNotAllowedError(ValueError):
    """Raised when a model id outside the exact allowlist is requested."""


def validate_model(model: str) -> str:
    """Return *model* unchanged if allowlisted, else raise."""
    if model not in MODEL_ALLOWLIST:
        raise ModelNotAllowedError(
            f"Model {model!r} is not in the claude_worker allowlist "
            f"{sorted(MODEL_ALLOWLIST)}"
        )
    return model


def choose_model(task: str, complexity: Optional[str] = None) -> Tuple[str, str]:
    """Return ``(model, route_reason)`` for the one claude_worker spawn.

    Sonnet unless *complexity* is an EXACT match (no case-folding, no
    whitespace trimming) against one of ``_COMPLEXITY_CATEGORIES`` — the same
    three values the tool schema's ``complexity`` enum declares. Anything
    else — ``None``, an empty string, an unrecognized value, or a value that
    only fuzzily resembles an allowed one — is the plain default. *task* is
    accepted for interface compatibility only; its content is never read.
    """
    if complexity in _COMPLEXITY_CATEGORIES:
        return validate_model(_policy.ESCALATION_MODEL), f"category:{complexity}"
    return validate_model(_policy.DEFAULT_MODEL), "default"
