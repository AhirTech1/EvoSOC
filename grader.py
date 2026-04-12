"""Root grader wrapper — re-exports the grade function so the validator
can import it as ``grader:grade`` from the repo root."""
from __future__ import annotations

import json
from typing import Any

from security_env.grader import grade, grade_episode, score_from_state, _strict_unit_score


def main() -> None:
    task_ids = ["easy_tier1", "medium_tier2", "hard_tier3"]

    task_scores: dict[str, float] = {}
    for task_id in task_ids:
        task_scores[task_id] = grade(task_id)

    tier_scores = {str(i + 1): score for i, score in enumerate(task_scores.values())}
    tasks = [
        {"id": tid, "grader": "grader:grade", "score": score}
        for tid, score in task_scores.items()
    ]
    final = round(_strict_unit_score(sum(task_scores.values()) / len(task_scores)), 4)

    output = {
        "task_scores": task_scores,
        "tier_scores": tier_scores,
        "tasks": tasks,
        "final_score": final,
    }
    print(json.dumps(output))


if __name__ == "__main__":
    main()
