import litellm
import pytest

from fallback_guard import Reason, classify

KW = {"model": "m", "llm_provider": "vertex_ai"}


@pytest.mark.parametrize(
    "exc, fallback, reason",
    [
        (litellm.RateLimitError(message="429", **KW), True, Reason.RATE_LIMITED),
        (litellm.Timeout(message="slow", **KW), True, Reason.TIMEOUT),
        (litellm.ServiceUnavailableError(message="503", **KW), True, Reason.PROVIDER_DOWN),
        (litellm.InternalServerError(message="500", **KW), True, Reason.PROVIDER_DOWN),
        (litellm.APIConnectionError(message="reset", **KW), True, Reason.PROVIDER_DOWN),
        (litellm.AuthenticationError(message="expired key", **KW), True, Reason.PRIMARY_CREDENTIALS),
        (litellm.BadRequestError(message="bad", **KW), False, Reason.BAD_REQUEST),
        (litellm.ContextWindowExceededError(message="too long", **KW), False, Reason.CONTEXT_WINDOW),
        (litellm.ContentPolicyViolationError(message="nope", **KW), False, Reason.CONTENT_POLICY),
        (litellm.BudgetExceededError(current_cost=10, max_budget=5), False, Reason.TEAM_BUDGET),
        (ValueError("something weird"), False, Reason.UNKNOWN),
    ],
)
def test_classify(exc, fallback, reason):
    d = classify(exc)
    assert (d.fallback, d.reason) == (fallback, reason)


def test_subclasses_are_checked_before_bad_request():
    # ContextWindowExceededError is a BadRequestError. Make sure it doesn't
    # get swallowed by the generic rule.
    exc = litellm.ContextWindowExceededError(message="too long", **KW)
    assert isinstance(exc, litellm.BadRequestError)
    assert classify(exc).reason is Reason.CONTEXT_WINDOW


@pytest.mark.parametrize(
    "status, fallback, reason",
    [
        (429, True, Reason.RATE_LIMITED),
        (504, True, Reason.TIMEOUT),
        (529, True, Reason.PROVIDER_DOWN),
        (418, False, Reason.BAD_REQUEST),
    ],
)
def test_generic_api_error_uses_status_code(status, fallback, reason):
    d = classify(litellm.APIError(status_code=status, message="x", **KW))
    assert (d.fallback, d.reason) == (fallback, reason)
