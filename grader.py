"""Root grader wrapper — re-exports the grade function so the validator
can import it as ``grader:grade`` from the repo root."""
from __future__ import annotations

import json
from typing import Any

from security_env.grader import (
    grade,
    grade_easy_tier1,
    grade_episode,
    grade_hard_tier3,
    grade_medium_tier2,
    score_from_state,
    _strict_unit_score,
)


def main() -> None:
    task_scores: dict[str, float] = {
        "easy_tier1": grade_easy_tier1(),
        "medium_tier2": grade_medium_tier2(),
        "hard_tier3": grade_hard_tier3(),
    }

    tier_scores = {str(i + 1): score for i, score in enumerate(task_scores.values())}
    tasks = [
        {"id": "easy_tier1", "grader": "grader:grade_easy_tier1", "score": task_scores["easy_tier1"]},
        {"id": "medium_tier2", "grader": "grader:grade_medium_tier2", "score": task_scores["medium_tier2"]},
        {"id": "hard_tier3", "grader": "grader:grade_hard_tier3", "score": task_scores["hard_tier3"]},
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
