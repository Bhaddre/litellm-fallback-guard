"""End-to-end against a real litellm.Router, using litellm's mock_response so
no provider is called. This also catches us drifting from litellm's API."""

import asyncio

import litellm
import pytest

from fallback_guard import FallbackBudget, FallbackRefused, GuardedRouter, Reason

KW = {"model": "m", "llm_provider": "vertex_ai"}
MSG = [{"role": "user", "content": "hi"}]


def make_router(primary_mock, backup_mock="from backup"):
    r = litellm.Router(
        model_list=[
            {
                "model_name": "sonnet",
                "litellm_params": {"model": "openai/primary", "api_key": "x", "mock_response": "placeholder"},
            },
            {
                "model_name": "sonnet-backup",
                "litellm_params": {"model": "openai/backup", "api_key": "x", "mock_response": "placeholder"},
            },
        ],
        num_retries=0,
        # router-level fallbacks are on here on purpose. if GuardedRouter didn't
        # disable them per call, the "no fallback" tests would fail.
        fallbacks=[{"sonnet": ["sonnet-backup"]}],
    )
    # Router deep-copies model_list on init and exceptions don't survive that,
    # so set the mocks afterwards.
    mocks = {"sonnet": primary_mock, "sonnet-backup": backup_mock}
    for d in r.model_list:
        d["litellm_params"]["mock_response"] = mocks[d["model_name"]]
    return r


def guarded(primary_mock, budget=None, events=None):
    return GuardedRouter(
        make_router(primary_mock),
        fallbacks={"sonnet": ["sonnet-backup"]},
        budget=budget or FallbackBudget(capacity=10, per_minute=60),
        on_event=events.append if events is not None else None,
    )


def run(coro):
    return asyncio.run(coro)


def text(resp):
    return resp.choices[0].message.content


def test_happy_path_never_touches_backup():
    events = []
    g = guarded("from primary", events=events)
    assert text(run(g.acompletion(model="sonnet", team="a", messages=MSG))) == "from primary"
    assert events == []


def test_provider_outage_falls_back():
    events = []
    g = guarded(litellm.ServiceUnavailableError(message="503", **KW), events=events)
    assert text(run(g.acompletion(model="sonnet", team="a", messages=MSG))) == "from backup"
    assert [(e.reason, e.fell_back) for e in events] == [(Reason.PROVIDER_DOWN, True)]


def test_bad_request_does_not_fall_back_even_though_router_has_fallbacks():
    g = guarded(litellm.BadRequestError(message="bad schema", **KW))
    with pytest.raises(litellm.BadRequestError):
        run(g.acompletion(model="sonnet", team="a", messages=MSG))


def test_runaway_team_gets_its_429_back_once_budget_is_spent():
    budget = FallbackBudget(capacity=2, per_minute=0.001)
    g = guarded(litellm.RateLimitError(message="429", **KW), budget=budget)

    for _ in range(2):
        assert text(run(g.acompletion(model="sonnet", team="noisy", messages=MSG))) == "from backup"

    with pytest.raises(FallbackRefused) as ei:
        run(g.acompletion(model="sonnet", team="noisy", messages=MSG))
    assert ei.value.status_code == 429
    assert isinstance(ei.value.original, litellm.RateLimitError)

    # other teams still get insurance
    assert text(run(g.acompletion(model="sonnet", team="quiet", messages=MSG))) == "from backup"


def test_last_group_error_is_raised_as_is():
    r = make_router(
        litellm.ServiceUnavailableError(message="primary down", **KW),
        backup_mock=litellm.ServiceUnavailableError(message="backup down too", **KW),
    )
    g = GuardedRouter(r, fallbacks={"sonnet": ["sonnet-backup"]})
    with pytest.raises(litellm.ServiceUnavailableError, match="backup down too"):
        run(g.acompletion(model="sonnet", team="a", messages=MSG))


def test_broken_metrics_hook_does_not_break_requests():
    def boom(_):
        raise RuntimeError("statsd is down")

    g = GuardedRouter(
        make_router(litellm.ServiceUnavailableError(message="503", **KW)),
        fallbacks={"sonnet": ["sonnet-backup"]},
        on_event=boom,
    )
    assert text(run(g.acompletion(model="sonnet", team="a", messages=MSG))) == "from backup"


# --- streaming -------------------------------------------------------------


async def collect(agen):
    return "".join([c.choices[0].delta.content or "" async for c in agen])


def test_stream_falls_back_before_first_chunk():
    g = guarded(litellm.ServiceUnavailableError(message="503", **KW))

    async def go():
        return await collect(await g.acompletion(model="sonnet", team="a", messages=MSG, stream=True))

    assert run(go()) == "from backup"


class _FailingStream:
    """Yields one chunk, then dies. What a provider dropping mid-response looks like."""

    def __init__(self, first):
        self.first = first
        self.sent = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self.sent:
            self.sent = True
            return self.first
        raise litellm.ServiceUnavailableError(message="dropped", **KW)


class _MidStreamRouter:
    def __init__(self, first_chunk):
        self.first_chunk = first_chunk
        self.calls = []

    async def acompletion(self, model, **kwargs):
        self.calls.append(model)
        return _FailingStream(self.first_chunk)


def test_stream_does_not_fall_back_after_first_chunk():
    async def first_chunk():
        r = make_router("partial")
        s = await r.acompletion(model="sonnet", messages=MSG, stream=True)
        async for c in s:
            return c

    chunk = run(first_chunk())
    fake = _MidStreamRouter(chunk)
    events = []
    g = GuardedRouter(fake, fallbacks={"sonnet": ["sonnet-backup"]}, on_event=events.append)

    async def go():
        got = []
        stream = await g.acompletion(model="sonnet", team="a", messages=MSG, stream=True)
        with pytest.raises(litellm.ServiceUnavailableError):
            async for c in stream:
                got.append(c)
        return got

    got = run(go())
    assert len(got) == 1
    assert fake.calls == ["sonnet"]  # never tried the backup
    assert [e.reason for e in events] == [Reason.MID_STREAM]
