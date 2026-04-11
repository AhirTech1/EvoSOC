from __future__ import annotations

from typing import Any

from server.environment import SecurityDefenseEnvironment


def _strict_unit_score(value: float) -> float:
    bounded = max(0.0, min(1.0, value))
    return max(0.0001, min(0.9999, bounded))


def score_from_state(state: dict[str, Any]) -> float:
    network = state.get("network_state", {})
    attack = state.get("attack", {})

    avg_health = (
        float(network.get("web_health", 0.0))
        + float(network.get("db_health", 0.0))
        + float(network.get("app_health", 0.0))
    ) / 3.0

    containment = 1.0 if attack.get("stopped") else 0.0
    health_component = max(0.0, min(1.0, avg_health))

    raw = 0.7 * containment + 0.3 * health_component
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
        1: [0, 1, 3],
        2: [0, 2, 3],
        3: [0, 1, 3],
    }

    tier_scores = {tier: grade_episode(tier, actions) for tier, actions in playbooks.items()}
    final = round(_strict_unit_score(sum(tier_scores.values()) / 3.0), 4)

    print({"tier_scores": tier_scores, "final_score": final})


if __name__ == "__main__":
    main()
