from __future__ import annotations

import json
import os
from typing import Any

from openai import OpenAI

from models import SecurityAction
from server.environment import SecurityDefenseEnvironment

SYSTEM_PROMPT = """
You are an autonomous SOC analyst. You must return exactly one JSON object with keys:
- action_type: one of BlockIP, IsolateHost, WatchLogs, ResolveAlert
- ip (optional)
- hostname (optional)
- alert_id (optional)
Choose the safest mitigation action based on active alerts and logs.
""".strip()


def _emit(tag: str, payload: dict[str, Any]) -> None:
    print(f"[{tag}] {json.dumps(payload, separators=(',', ':'), ensure_ascii=False)}")


def decide_action(client: OpenAI, observation: dict) -> SecurityAction:
    response = client.responses.create(
        model=os.environ["MODEL_NAME"],
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(observation)},
        ],
        temperature=0,
    )

    text = response.output_text.strip()
    payload = json.loads(text)
    return SecurityAction.model_validate(payload)


def run_episode(client: OpenAI, tier: int, max_steps: int = 12) -> tuple[float, dict[str, Any]]:
    env = SecurityDefenseEnvironment(seed=17 + tier)
    obs = env.reset(tier=tier)
    total_reward = 0.0

    for step in range(max_steps):
        action = decide_action(client, obs.model_dump())
        result = env.step(action)
        total_reward += float(result.reward)
        _emit(
            "STEP",
            {
                "tier": tier,
                "step": step,
                "action": action.model_dump(),
                "reward": round(float(result.reward), 4),
                "done": bool(result.done),
            },
        )
        obs = result.observation
        if result.done:
            break

    final_state = env.state()
    score = max(0.0, min(1.0, (total_reward + 5.0) / 10.0))
    return round(score, 4), final_state


def main() -> None:
    api_base_url = os.environ["API_BASE_URL"]
    model_name = os.environ["MODEL_NAME"]
    hf_token = os.environ["HF_TOKEN"]

    client = OpenAI(api_key=hf_token, base_url=api_base_url)

    _emit(
        "START",
        {
            "model": model_name,
            "api_base_url": api_base_url,
            "tasks": ["easy_tier1", "medium_tier2", "hard_tier3"],
        },
    )

    task_scores: dict[str, float] = {}
    for tier, name in ((1, "easy_tier1"), (2, "medium_tier2"), (3, "hard_tier3")):
        score, final_state = run_episode(client=client, tier=tier)
        task_scores[name] = score
        _emit(
            "STEP",
            {
                "task": name,
                "tier": tier,
                "score": score,
                "attack_stopped": bool(final_state.get("attack", {}).get("stopped")),
            },
        )

    baseline_score = round(sum(task_scores.values()) / 3.0, 4)
    _emit("END", {"task_scores": task_scores, "baseline_score": baseline_score})


if __name__ == "__main__":
    main()
