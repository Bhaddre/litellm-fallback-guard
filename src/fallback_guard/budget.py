"""Per-team token bucket for fallback calls.

The backup provider is shared insurance. During a real outage every team
falls back a little and stays under its budget. When one team's runaway loop
is the thing causing 429s, that team burns through its own budget in seconds
and gets the 429 back, instead of moving the flood onto the backup provider
that everyone else is counting on.

You don't have to figure out whether a 429 was "their fault". The budget
sorts it out.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass


@dataclass
class _Bucket:
    tokens: float
    updated: float


class FallbackBudget:
    """Token bucket per team. Thread safe.

    capacity: burst size, i.e. how many fallbacks a team can do back to back.
    per_minute: steady-state refill rate.
    overrides: per-team (capacity, per_minute) for teams that need more.
    """

    def __init__(
        self,
        capacity: int = 20,
        per_minute: float = 60,
        overrides: dict[str, tuple[int, float]] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if capacity <= 0 or per_minute <= 0:
            raise ValueError("capacity and per_minute must be positive")
        self._default = (capacity, per_minute)
        self._overrides = dict(overrides or {})
        self._clock = clock
        self._buckets: dict[str, _Bucket] = {}
        self._lock = threading.Lock()

    def _limits(self, team: str) -> tuple[int, float]:
        return self._overrides.get(team, self._default)

    def try_take(self, team: str) -> bool:
        """Spend one fallback for `team`. False means the team is out of budget."""
        capacity, per_minute = self._limits(team)
        now = self._clock()
        with self._lock:
            b = self._buckets.get(team)
            if b is None:
                b = self._buckets[team] = _Bucket(tokens=float(capacity), updated=now)
            else:
                b.tokens = min(capacity, b.tokens + (now - b.updated) * per_minute / 60)
                b.updated = now
            if b.tokens >= 1:
                b.tokens -= 1
                return True
            return False

    def remaining(self, team: str) -> float:
        """Tokens left right now, without spending any. Handy for metrics."""
        capacity, per_minute = self._limits(team)
        with self._lock:
            b = self._buckets.get(team)
            if b is None:
                return float(capacity)
            return min(capacity, b.tokens + (self._clock() - b.updated) * per_minute / 60)
