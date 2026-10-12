# litellm-fallback-guard

Fallback rules for [LiteLLM](https://github.com/BerriAI/litellm): fall back when the **provider** is sick, not when a **team** is.

```
pip install git+https://github.com/Bhaddre/litellm-fallback-guard
```

## the problem

"If Vertex is down, fail over to Bedrock." Add a fallback list, done. Except:

- One team's agent gets stuck in a loop and hits the provider's rate limit. Every 429 gets rerouted to the backup provider. Now two providers are getting hammered, the backup is degraded for every other team, and the team that caused it never notices because their calls "worked."
- A malformed request gets a 400 from the primary. The router tries it on the backup. Another 400, just slower.
- A stream dies halfway through. The client already has half an answer. There's nothing safe to fall back to at that point, but you'd like to know it happened.

LiteLLM's built-in fallbacks are a good mechanism. This adds the policy on top.

Longer writeup: [Your LLM fallback is probably making outages worse](https://bhaddre.github.io/2026/10/llm-gateway-fallback-rules/)

## what it does

`GuardedRouter` wraps a normal `litellm.Router`. On every call it:

1. Calls the primary model group with router-level fallbacks turned off for that call. Your `num_retries`, cooldowns and load balancing still work as usual.
2. If that fails, classifies the error:

   | error | fall back? | why |
   |---|---|---|
   | 5xx, 502, 503, connection reset | yes | provider is sick |
   | timeout | yes | |
   | 429 from the provider | yes, **if the team has budget** | see below |
   | 401 / 403 / 404 from the provider | yes | the gateway holds the creds, so this is our side broken (expired key, model not enabled in that region). the backup may be fine |
   | 400, 422 | no | bad request. same answer everywhere |
   | context window exceeded | no | different problem. route to a bigger-context model instead |
   | content policy | no | |
   | team spend budget exceeded | no | that's the limit working |
   | anything unrecognized | no | don't double load on something you don't understand |

3. If the error is fallback-eligible, spends one token from that **team's fallback budget**. No token left means the team gets its original error back (as `FallbackRefused`, with the original status code).
4. Streams only fall back before the first chunk.

## why a budget instead of "detect whose fault the 429 was"

You can't reliably tell from a single 429 whether the provider is overloaded or one team is flooding it. You don't need to.

During a real outage every team falls back a little and stays well under budget. When one team's runaway loop is causing the 429s, that team burns its own budget in seconds and gets its 429s back. Everyone else keeps their insurance.

It's a token bucket per team: a burst size plus a refill rate, with per-team overrides for teams that legitimately need more.

## usage

```python
import litellm
from fallback_guard import FallbackBudget, GuardedRouter

router = litellm.Router(
    model_list=[...],          # "claude-sonnet" on Vertex, "claude-sonnet-bedrock" on Bedrock
    num_retries=2,
    allowed_fails=3,
    cooldown_time=60,
)

guarded = GuardedRouter(
    router,
    fallbacks={"claude-sonnet": ["claude-sonnet-bedrock"]},
    budget=FallbackBudget(capacity=20, per_minute=60),
    on_event=lambda ev: metrics.incr("llm.fallback", tags=[f"team:{ev.team}", f"reason:{ev.reason.value}"]),
)

resp = await guarded.acompletion(
    model="claude-sonnet",
    team="search-agents",          # what the budget is keyed on
    messages=[{"role": "user", "content": "hi"}],
)
```

Streaming works the same way with `stream=True`. You get an async iterator back.

Full example with Vertex AI and Bedrock: [`examples/vertex_to_bedrock.py`](examples/vertex_to_bedrock.py).

## what to alert on

Every decision emits an `Event(team, model_group, reason, fell_back, budget_blocked)`.

- **fallbacks with `fell_back=True`.** Should be close to zero on a normal day. If it never quite gets to zero, something in the primary path is flaky and fallback is hiding it.
- **`budget_blocked=True` by team.** That's a team looping or a team that needs a bigger budget. Either way someone should talk to them.
- **`reason=mid_stream`.** Fallback can't help here. If it's frequent, the agents need their own retry.

## scope

- This is the SDK `Router`. It doesn't plug into the LiteLLM proxy server as a config option yet. If you run the proxy, the same policy can be applied in a custom handler, or you can put this in your agent framework's client layer.
- Same model family on both sides, please. If the fallback is a different model, every outage also becomes a quality incident.
- Prompt caching lives at the provider, so the first calls after a fallback are slower and cost more. Expected during an outage. If you're flapping between providers, raise `cooldown_time`.

## development

```bash
pip install -e ".[dev]"
pytest -q
ruff check . && ruff format --check .
```

Tests run against a real `litellm.Router` using LiteLLM's `mock_response`, so nothing calls a provider. They'd also catch a LiteLLM upgrade that changes how per-call fallback disabling works.

## license

MIT
