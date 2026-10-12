"""Decide whether an error from the primary provider should trigger a fallback.

Rule of thumb: fall back when the *provider* is sick, not when the *request*
is bad. A bad request gets the same answer from every provider, just slower.

Unknown errors don't fall back. Doubling load on the backup provider because
of something we don't understand is how one outage becomes two.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from litellm import exceptions as lx


class Reason(str, Enum):
    # fallback ok
    PROVIDER_DOWN = "provider_down"  # 5xx, 502, 503, connection errors
    TIMEOUT = "timeout"
    RATE_LIMITED = "rate_limited"  # provider 429. still subject to the team budget
    PRIMARY_CREDENTIALS = "primary_credentials"  # our creds or model access on the primary, not the caller's
    # no fallback
    BAD_REQUEST = "bad_request"
    CONTEXT_WINDOW = "context_window"  # different problem, use a bigger-context model group
    CONTENT_POLICY = "content_policy"
    TEAM_BUDGET = "team_budget"  # the caller's own spend limit. working as intended
    MID_STREAM = "mid_stream"  # tokens already went out, can't switch now
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class Decision:
    fallback: bool
    reason: Reason


# Order matters: litellm's ContextWindowExceededError and ContentPolicyViolationError
# subclass BadRequestError, so the specific ones have to be checked first.
_RULES: list[tuple[type[BaseException], Decision]] = [
    (lx.ContextWindowExceededError, Decision(False, Reason.CONTEXT_WINDOW)),
    (lx.ContentPolicyViolationError, Decision(False, Reason.CONTENT_POLICY)),
    (lx.BudgetExceededError, Decision(False, Reason.TEAM_BUDGET)),
    (lx.BadRequestError, Decision(False, Reason.BAD_REQUEST)),
    (lx.UnprocessableEntityError, Decision(False, Reason.BAD_REQUEST)),
    (lx.RateLimitError, Decision(True, Reason.RATE_LIMITED)),
    (lx.Timeout, Decision(True, Reason.TIMEOUT)),
    # The gateway holds the provider credentials, not the caller. A 401/403/404
    # from the primary means our side is broken (expired key, model not enabled
    # in that region), and the backup provider might be fine.
    (lx.AuthenticationError, Decision(True, Reason.PRIMARY_CREDENTIALS)),
    (lx.PermissionDeniedError, Decision(True, Reason.PRIMARY_CREDENTIALS)),
    (lx.NotFoundError, Decision(True, Reason.PRIMARY_CREDENTIALS)),
    (lx.ServiceUnavailableError, Decision(True, Reason.PROVIDER_DOWN)),
    (lx.BadGatewayError, Decision(True, Reason.PROVIDER_DOWN)),
    (lx.InternalServerError, Decision(True, Reason.PROVIDER_DOWN)),
    (lx.APIConnectionError, Decision(True, Reason.PROVIDER_DOWN)),
]

# Newer litellm raises this when a stream dies after it started. It subclasses
# ServiceUnavailableError, so it has to go first. Older versions don't have it.
if hasattr(lx, "MidStreamFallbackError"):
    _RULES.insert(0, (lx.MidStreamFallbackError, Decision(False, Reason.MID_STREAM)))


def classify(exc: BaseException) -> Decision:
    """Map an exception from the primary call to a fallback decision."""
    for exc_type, decision in _RULES:
        if isinstance(exc, exc_type):
            return decision
    # Generic APIError: trust the status code if there is one.
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        if status == 429:
            return Decision(True, Reason.RATE_LIMITED)
        if status in (408, 504):
            return Decision(True, Reason.TIMEOUT)
        if status >= 500:
            return Decision(True, Reason.PROVIDER_DOWN)
        if 400 <= status < 500:
            return Decision(False, Reason.BAD_REQUEST)
    return Decision(False, Reason.UNKNOWN)
