"""A thin wrapper around litellm.Router that owns the fallback decision.

litellm's built-in fallbacks fire on almost any error. This turns them off
per call (`disable_fallbacks=True`), lets the Router do its normal retries
and cooldowns on the primary group, and only then decides, using the policy
and the team's budget, whether to try the next group.
"""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from typing import Any

import litellm

from .budget import FallbackBudget
from .policy import Decision, Reason, classify

log = logging.getLogger("fallback_guard")


@dataclass(frozen=True)
class Event:
    """One fallback decision. Send these to your metrics."""

    team: str
    model_group: str
    reason: Reason
    fell_back: bool  # did we actually call the next group
    budget_blocked: bool  # policy said yes, team budget said no


class FallbackRefused(Exception):
    """Raised when the policy allowed a fallback but the team's budget didn't.

    Carries the original error so callers still see the real status code.
    """

    def __init__(self, team: str, original: BaseException) -> None:
        super().__init__(f"fallback budget exhausted for team {team!r}: {original}")
        self.team = team
        self.original = original
        self.status_code = getattr(original, "status_code", 429)


class GuardedRouter:
    def __init__(
        self,
        router: litellm.Router,
        fallbacks: dict[str, list[str]],
        budget: FallbackBudget | None = None,
        classify_fn: Callable[[BaseException], Decision] = classify,
        on_event: Callable[[Event], None] | None = None,
    ) -> None:
        self.router = router
        self.fallbacks = {k: list(v) for k, v in fallbacks.items()}
        self.budget = budget or FallbackBudget()
        self.classify = classify_fn
        self.on_event = on_event
        self.counts: Counter[tuple[str, Reason, bool]] = Counter()

    def _emit(self, ev: Event) -> None:
        self.counts[(ev.team, ev.reason, ev.fell_back)] += 1
        if self.on_event:
            try:
                self.on_event(ev)
            except Exception:  # metrics must never break a request
                log.exception("on_event hook failed")

    def _should_fall_back(self, team: str, group: str, exc: BaseException) -> bool:
        decision = self.classify(exc)
        if not decision.fallback:
            self._emit(Event(team, group, decision.reason, fell_back=False, budget_blocked=False))
            return False
        if not self.budget.try_take(team):
            self._emit(Event(team, group, decision.reason, fell_back=False, budget_blocked=True))
            raise FallbackRefused(team, exc) from exc
        self._emit(Event(team, group, decision.reason, fell_back=True, budget_blocked=False))
        log.warning("falling back from %s for team=%s reason=%s", group, team, decision.reason.value)
        return True

    async def acompletion(self, *, model: str, team: str, **kwargs: Any) -> Any:
        """Like Router.acompletion, plus a `team` the budget is keyed on.

        With stream=True you get back an async iterator. Fallback can only
        happen before the first chunk; after that a failure is raised to you.
        """
        if kwargs.get("stream"):
            return self._stream(model=model, team=team, **kwargs)

        chain = [model, *self.fallbacks.get(model, [])]
        for i, group in enumerate(chain):
            try:
                return await self.router.acompletion(model=group, disable_fallbacks=True, **kwargs)
            except Exception as exc:
                last = i == len(chain) - 1
                if last or not self._should_fall_back(team, group, exc):
                    raise
        raise AssertionError("unreachable")

    async def _stream(self, *, model: str, team: str, **kwargs: Any) -> AsyncIterator[Any]:
        chain = [model, *self.fallbacks.get(model, [])]
        for i, group in enumerate(chain):
            started = False
            try:
                stream = await self.router.acompletion(model=group, disable_fallbacks=True, **kwargs)
                async for chunk in stream:
                    started = True
                    yield chunk
                return
            except Exception as exc:
                if started:
                    # The client already has partial output. Switching providers
                    # now would splice two different answers together.
                    self._emit(Event(team, group, Reason.MID_STREAM, fell_back=False, budget_blocked=False))
                    raise
                last = i == len(chain) - 1
                if last or not self._should_fall_back(team, group, exc):
                    raise
