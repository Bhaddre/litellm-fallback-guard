import threading

import pytest

from fallback_guard import FallbackBudget


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def test_burst_then_empty():
    b = FallbackBudget(capacity=3, per_minute=60, clock=FakeClock())
    assert [b.try_take("a") for _ in range(4)] == [True, True, True, False]


def test_refills_over_time():
    clock = FakeClock()
    b = FallbackBudget(capacity=2, per_minute=60, clock=clock)  # 1 token / sec
    assert b.try_take("a") and b.try_take("a")
    assert not b.try_take("a")
    clock.t = 1.0
    assert b.try_take("a")
    assert not b.try_take("a")


def test_refill_caps_at_capacity():
    clock = FakeClock()
    b = FallbackBudget(capacity=2, per_minute=60, clock=clock)
    b.try_take("a")
    clock.t = 3600
    assert b.remaining("a") == 2


def test_teams_are_isolated():
    b = FallbackBudget(capacity=1, per_minute=1, clock=FakeClock())
    assert b.try_take("noisy")
    assert not b.try_take("noisy")
    assert b.try_take("quiet")


def test_override_for_one_team():
    b = FallbackBudget(capacity=1, per_minute=1, overrides={"big": (5, 60)}, clock=FakeClock())
    assert sum(b.try_take("big") for _ in range(10)) == 5
    assert sum(b.try_take("small") for _ in range(10)) == 1


def test_rejects_bad_config():
    with pytest.raises(ValueError):
        FallbackBudget(capacity=0)


def test_thread_safe_no_overspend():
    b = FallbackBudget(capacity=100, per_minute=0.0001)
    taken = []

    def grab():
        for _ in range(50):
            if b.try_take("t"):
                taken.append(1)

    threads = [threading.Thread(target=grab) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(taken) == 100
