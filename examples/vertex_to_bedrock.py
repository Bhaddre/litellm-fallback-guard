"""Vertex AI primary, Bedrock fallback, per-team fallback budget.

Needs real credentials for both providers:
  VERTEX_PROJECT, plus gcloud application-default credentials
  AWS credentials for Bedrock in us-east-1

Model IDs change. Check the current ones for your regions.
"""

import asyncio
import logging
import os

import litellm

from fallback_guard import Event, FallbackBudget, GuardedRouter

logging.basicConfig(level=logging.INFO)

router = litellm.Router(
    model_list=[
        {
            "model_name": "claude-sonnet",
            "litellm_params": {
                "model": "vertex_ai/claude-sonnet-4-5@20250929",
                "vertex_project": os.environ["VERTEX_PROJECT"],
                "vertex_location": "us-east5",
            },
        },
        {
            "model_name": "claude-sonnet-bedrock",
            "litellm_params": {
                "model": "bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0",
                "aws_region_name": "us-east-1",
            },
        },
    ],
    num_retries=2,  # retry the primary before anyone thinks about falling back
    timeout=60,
    allowed_fails=3,
    cooldown_time=60,
)


def log_event(ev: Event) -> None:
    # swap for statsd / prometheus / otel in real life
    print(
        f"fallback team={ev.team} reason={ev.reason.value} fell_back={ev.fell_back} "
        f"budget_blocked={ev.budget_blocked}"
    )


guarded = GuardedRouter(
    router,
    fallbacks={"claude-sonnet": ["claude-sonnet-bedrock"]},
    budget=FallbackBudget(
        capacity=20,  # burst
        per_minute=60,  # steady state
        overrides={"support-agents": (100, 300)},  # a big team that's earned more
    ),
    on_event=log_event,
)


async def main() -> None:
    resp = await guarded.acompletion(
        model="claude-sonnet",
        team="search-agents",
        messages=[{"role": "user", "content": "one sentence on why fallbacks need budgets"}],
        max_tokens=100,
    )
    print(resp.choices[0].message.content)


if __name__ == "__main__":
    asyncio.run(main())
