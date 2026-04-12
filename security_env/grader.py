from __future__ import annotations

from typing import Any

try:
    from server.environment import SecurityDefenseEnvironment
except ModuleNotFoundError:
    from security_env.server.environment import SecurityDefenseEnvironment


def _strict_unit_score(value: float) -> float:
    """Clamp any score to the safe open interval (0.01, 0.99)."""
    bounded = max(0.0, min(1.0, value))
    return max(0.01, min(0.99, bounded))


def score_from_state(state: dict[str, Any]) -> float:
    """Produce a gradient score from the final environment state.

    Containment is NOT binary — it uses three tiers so the raw score
    never lands on exactly 0.0 or 1.0 *before* clamping.
    """
    network = state.get("network_state", {})
    attack = state.get("attack", {})

    avg_health = (
        float(network.get("web_health", 0.0))
        + float(network.get("db_health", 0.0))
        + float(network.get("app_health", 0.0))
    ) / 3.0

    # Gradient containment: stopped → 0.85, detected → 0.35, else → 0.05
    if attack.get("stopped"):
        containment = 0.85
    elif attack.get("detected"):
        containment = 0.35
    else:
        containment = 0.05

    health_component = max(0.0, min(1.0, avg_health))

    raw = 0.6 * containment + 0.4 * health_component
    return round(_strict_unit_score(raw), 4)


def grade_episode(tier: int, policy_actions: list[int], seed: int = 7) -> float:
    env = SecurityDefenseEnvironment(seed=seed)
    env.reset(tier=tier)

    done = False
    for action in policy_actions:
        result = env.step(action)
        done = result.done
        if done:
            break

    return score_from_state(env.state())


def main() -> None:
    playbooks = {
        "easy_tier1": (1, [0, 1, 3]),
        "medium_tier2": (2, [0, 2, 3]),
        "hard_tier3": (3, [0, 1, 3]),
    }

    task_scores = {
        task_name: grade_episode(tier, actions)
        for task_name, (tier, actions) in playbooks.items()
    }
    tier_scores = {index + 1: score for index, score in enumerate(task_scores.values())}
    tasks = [
        {"id": task_name, "grader": "deterministic", "score": score}
        for task_name, score in task_scores.items()
    ]
    final = round(_strict_unit_score(sum(task_scores.values()) / 3.0), 4)

    print(
        {
            "task_scores": task_scores,
            "tier_scores": tier_scores,
            "tasks": tasks,
            "final_score": final,
        }
    )


if __name__ == "__main__":
    main()
