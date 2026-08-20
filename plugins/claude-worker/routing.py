"""Auto model routing for the claude_worker tool (requirement 2).

Sonnet is the default. Opus is used for a classifier hit on architecture /
security / hard-debugging tasks, or for exactly one controlled escalation
after a Sonnet attempt fails for a non-breaker reason.

Nothing here is configurable. Terra's review found the contract mutable in
two directions at once: ``models.default``/``models.escalated`` could be
swapped, and ``routing.escalate_after_failures`` fed straight into the
runner's spawn loop, so a value of 2 bought a third spawn. The model
identities are now literals from ``policy.py`` and the cap is structural —
``choose_model`` raises for any attempt other than 0 or 1, and refuses
attempt 1 unless it follows a genuine non-breaker failure of a Sonnet run.
A task that starts on Opus (a category hit) has nothing to escalate to, so
it gets no second attempt either.

Deliberately NOT an LLM call: classification is a small deterministic
keyword scan (plus an explicit ``complexity`` override), so routing stays
fast, testable, and side-effect free.
"""

from __future__ import annotations

from typing import Optional, Tuple

from . import policy as _policy

MODEL_ALLOWLIST = _policy.MODEL_ALLOWLIST

_BREAKER_FAILURE_CLASSES = frozenset({"auth", "rate", "extra_usage"})

_COMPLEXITY_CATEGORIES = frozenset({"architecture", "security", "hard_debugging"})

_ARCHITECTURE_KEYWORDS = (
    "architecture", "system design", "design review", "refactor plan",
    "design doc", "high-level design",
)
_SECURITY_KEYWORDS = (
    "security", "vulnerability", "vuln", "cve", "exploit", "auth bypass",
    "injection", "privilege escalation", "penetration test", "pentest",
)
_HARD_DEBUGGING_KEYWORDS = (
    "segfault", "race condition", "heisenbug", "flaky", "deadlock",
    "memory corruption", "data corruption", "intermittent",
)


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


def _classify_complexity(task: str, complexity: Optional[str]) -> Optional[str]:
    if complexity:
        normalized = complexity.strip().lower()
        if normalized in _COMPLEXITY_CATEGORIES:
            return normalized
    lower_task = (task or "").lower()
    for keyword in _SECURITY_KEYWORDS:
        if keyword in lower_task:
            return "security"
    for keyword in _ARCHITECTURE_KEYWORDS:
        if keyword in lower_task:
            return "architecture"
    for keyword in _HARD_DEBUGGING_KEYWORDS:
        if keyword in lower_task:
            return "hard_debugging"
    return None


def choose_model(
    task: str,
    attempt: int = 0,
    complexity: Optional[str] = None,
    previous_failure_class: Optional[str] = None,
) -> Tuple[str, str]:
    """Return ``(model, route_reason)`` for a claude_worker spawn attempt.

    ``attempt`` is 0 for the first spawn and 1 for the single permitted
    escalation. Every other case raises ``RuntimeError`` — the hard "no
    third spawn" guarantee, enforced here rather than trusted to the caller:

    * ``attempt`` outside ``range(policy.MAX_ATTEMPTS)``;
    * ``attempt == 1`` with no previous failure, or with a breaker-class
      failure (auth/rate/extra_usage — a bigger model is never the right
      answer to a credentials or quota problem);
    * ``attempt == 1`` for a task that already started on Opus.
    """
    if not isinstance(attempt, int) or isinstance(attempt, bool):
        raise RuntimeError(f"claude_worker: invalid attempt {attempt!r}")
    if attempt < 0 or attempt >= _policy.MAX_ATTEMPTS:
        raise RuntimeError(
            f"claude_worker: attempt {attempt} is outside the structural cap "
            f"of {_policy.MAX_ATTEMPTS} attempts — no further spawn permitted"
        )

    category = _classify_complexity(task, complexity)

    if attempt == 0:
        if category is not None:
            return validate_model(_policy.ESCALATION_MODEL), f"category:{category}"
        return validate_model(_policy.DEFAULT_MODEL), "default"

    # attempt == 1: the one escalation, and only from a real Sonnet failure.
    if category is not None:
        raise RuntimeError(
            f"claude_worker: task already routed to {_policy.ESCALATION_MODEL} "
            f"(category:{category}) — there is nothing to escalate to"
        )
    if previous_failure_class is None:
        raise RuntimeError(
            "claude_worker: escalation requires a previous failed attempt"
        )
    if previous_failure_class in _BREAKER_FAILURE_CLASSES:
        raise RuntimeError(
            f"claude_worker: {previous_failure_class} failures never escalate — "
            "the breaker, not a bigger model, is the response to auth/rate/quota errors"
        )
    return validate_model(_policy.ESCALATION_MODEL), "escalation:sonnet-failure"
