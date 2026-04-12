#!/usr/bin/env python3
"""Dry-run validation: ensure all task scores fall strictly in (0, 1)."""
from __future__ import annotations

import sys
import os

# Ensure the project root is on sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "security_env"))
sys.path.insert(0, os.path.dirname(__file__))

from security_env.grader import grade_episode, _strict_unit_score
from security_env.inference import run_episode


def main() -> None:
    LOWER, UPPER = 0.01, 0.99
    all_pass = True

    # ── Grader playbook scores ──────────────────────────────────
    playbooks = {
        "easy_tier1":   (1, [0, 1, 3]),
        "medium_tier2": (2, [0, 2, 3]),
        "hard_tier3":   (3, [0, 1, 3]),
    }

    print("=" * 55)
    print("  GRADER (deterministic playbook) scores")
    print("=" * 55)
    for name, (tier, actions) in playbooks.items():
        score = grade_episode(tier, actions)
        ok = LOWER <= score <= UPPER
        status = "✓ PASS" if ok else "✗ FAIL"
        print(f"  {name:20s}  score={score:.4f}  {status}")
        if not ok:
            all_pass = False

    # ── Inference default-policy scores (no LLM) ───────────────
    print()
    print("=" * 55)
    print("  INFERENCE (default policy, no LLM) scores")
    print("=" * 55)
    for tier, name in ((1, "easy_tier1"), (2, "medium_tier2"), (3, "hard_tier3")):
        score, _ = run_episode(client=None, model_name="", tier=tier)
        ok = LOWER <= score <= UPPER
        status = "✓ PASS" if ok else "✗ FAIL"
        print(f"  {name:20s}  score={score:.4f}  {status}")
        if not ok:
            all_pass = False

    # ── Final aggregate ────────────────────────────────────────
    grader_scores = [grade_episode(t, a) for _, (t, a) in playbooks.items()]
    final = round(_strict_unit_score(sum(grader_scores) / 3.0), 4)
    print()
    print(f"  Aggregate final_score = {final:.4f}")
    ok = LOWER <= final <= UPPER
    print(f"  Aggregate check: {'✓ PASS' if ok else '✗ FAIL'}")
    if not ok:
        all_pass = False

    print()
    if all_pass:
        print("══ ALL CHECKS PASSED ══")
    else:
        print("══ SOME CHECKS FAILED ══")
        sys.exit(1)


if __name__ == "__main__":
    main()
