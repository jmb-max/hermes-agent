"""Model routing for the claude_worker tool (requirement 2).

Sonnet is the default for every task, unconditionally. Opus is used ONLY
when the caller supplies BOTH an allowed ``complexity`` classification
(architecture / security / hard-debugging) AND explicit authorization
(``allow_opus=True``) — either one alone selects Sonnet. There is exactly
one spawn per ``claude_worker`` call: no retry, no post-failure escalation,
and the task text itself never influences model selection.

This replaces an earlier design that classified free-form task text with a
keyword scan and allowed one automatic escalation to Opus after a Sonnet
failure. Both were removed deliberately:

* a keyword scan over task text is a covert, caller-invisible way to select
  a more expensive/more capable model — the caller must ask for Opus
  explicitly, in a dedicated field, not by phrasing a task a certain way;
* ``complexity`` alone used to be sufficient to route to Opus. That let any
  caller who could set one string field opt itself into Opus with no
  additional authorization. ``allow_opus`` closes that: selecting Opus now
  requires the caller to affirmatively ask for it, not merely describe the
  task as hard;
* a failure-driven second spawn on a bigger model hides cost and latency
  behind a single tool call and makes "no third spawn" harder to reason
  about. There is now no failure path that spawns again at all — a caller
  who wants to retry calls ``claude_worker`` again themselves.

Nothing here is configurable. The model identities are literals from
``policy.py``; there is no ``models.default``/``models.escalated`` config
key and no ``routing.escalate_after_failures`` — the whole point of this
module is that neither the model chosen nor the number of spawns can be
changed by configuration or by task phrasing.

Deliberately NOT an LLM call: this is a small, pure, side-effect-free
function, so routing stays fast and testable.
"""

from __future__ import annotations

from typing import Optional, Tuple

from . import policy as _policy

MODEL_ALLOWLIST = _policy.MODEL_ALLOWLIST

#: The only ``complexity`` values that are even eligible for Opus — still
#: insufficient on their own; ``allow_opus=True`` is also required.
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


def _normalized_category(complexity: Optional[str]) -> Optional[str]:
    if not isinstance(complexity, str):
        return None
    normalized = complexity.strip().lower()
    return normalized if normalized in _COMPLEXITY_CATEGORIES else None


def choose_model(
    task: str,
    complexity: Optional[str] = None,
    allow_opus: bool = False,
) -> Tuple[str, str]:
    """Return ``(model, route_reason)`` for the one claude_worker spawn.

    *task* is accepted (and otherwise ignored) only so callers do not need to
    special-case this signature — it is never inspected. Opus requires BOTH:

    * ``complexity`` to be one of ``architecture``/``security``/
      ``hard_debugging`` (case-insensitive, whitespace-trimmed); and
    * ``allow_opus`` to be the literal ``True`` — anything else (missing,
      ``False``, a truthy-looking non-bool) leaves the model on Sonnet.

    Every other combination — no complexity, an unrecognized complexity,
    ``allow_opus`` not True, or both together — returns
    ``policy.DEFAULT_MODEL`` with route reason ``"default"``.
    """
    del task  # never inspected; kept for call-site symmetry only

    category = _normalized_category(complexity)
    if category is not None and allow_opus is True:
        return validate_model(_policy.ESCALATION_MODEL), f"authorized:{category}"
    return validate_model(_policy.DEFAULT_MODEL), "default"
