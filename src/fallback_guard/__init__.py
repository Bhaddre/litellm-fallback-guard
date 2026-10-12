"""Fallback rules for LiteLLM: fall back when the provider is sick, not when a team is."""

from .budget import FallbackBudget
from .policy import Decision, Reason, classify
from .router import Event, FallbackRefused, GuardedRouter

__all__ = [
    "Decision",
    "Event",
    "FallbackBudget",
    "FallbackRefused",
    "GuardedRouter",
    "Reason",
    "classify",
]
__version__ = "0.1.0"
