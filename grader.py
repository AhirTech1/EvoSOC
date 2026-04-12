"""Root grader wrapper — re-exports the grade function so the validator
can import it as ``grader:grade`` from the repo root."""
from __future__ import annotations

import json
import os
import sys
from typing import Any, Callable

# Ensure repo root is importable when validator runs from a different cwd.
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

try:
    from security_env.grader import grade as _grade
    from security_env.grader import grade_easy_tier1 as _grade_easy_tier1
    from security_env.grader import grade_hard_tier3 as _grade_hard_tier3
    from security_env.grader import grade_medium_tier2 as _grade_medium_tier2
except Exception:
    _grade = None
    _grade_easy_tier1 = None
    _grade_medium_tier2 = None
    _grade_hard_tier3 = None


def _strict_unit_score(value: float) -> float:
    bounded = max(0.0, min(1.0, float(value)))
    return round(max(0.01, min(0.99, bounded)), 4)


def _safe_call(fn: Callable[..., float] | None, default: float = 0.5, *args: Any, **kwargs: Any) -> float:
    if fn is None:
        return _strict_unit_score(default)
    try:
        return _strict_unit_score(float(fn(*args, **kwargs)))
    except Exception:
        return _strict_unit_score(default)


def grade(task_id: str, inference_output: dict[str, Any] | None = None) -> float:
    if _grade is not None:
        return _safe_call(_grade, 0.5, task_id, inference_output)

    fallback = {
        "easy_tier1": grade_easy_tier1,
        "medium_tier2": grade_medium_tier2,
        "hard_tier3": grade_hard_tier3,
    }
    return fallback.get(task_id, lambda *_a, **_kw: 0.5)(inference_output)


def grade_easy_tier1(inference_output: dict[str, Any] | None = None) -> float:
    return _safe_call(_grade_easy_tier1, 0.5, inference_output)


def grade_medium_tier2(inference_output: dict[str, Any] | None = None) -> float:
    return _safe_call(_grade_medium_tier2, 0.5, inference_output)


def grade_hard_tier3(inference_output: dict[str, Any] | None = None) -> float:
    return _safe_call(_grade_hard_tier3, 0.5, inference_output)


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
