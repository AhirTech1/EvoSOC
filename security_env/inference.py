from __future__ import annotations

import json
import os
import re
from typing import Any

from openai import OpenAI

try:
    from models import ActionType, SecurityAction
    from server.environment import SecurityDefenseEnvironment
except ModuleNotFoundError:
    from security_env.models import ActionType, SecurityAction
    from security_env.server.environment import SecurityDefenseEnvironment

SYSTEM_PROMPT = """
You are an autonomous SOC analyst. You must return exactly one JSON object with keys:
- action_type: one of BlockIP, IsolateHost, WatchLogs, ResolveAlert
- ip (optional)
- hostname (optional)
- alert_id (optional)
Choose the safest mitigation action based on active alerts and logs.
""".strip()


def _emit(tag: str, payload: dict[str, Any]) -> None:
    print(f"[{tag}] {json.dumps(payload, separators=(',', ':'), ensure_ascii=False)}", flush=True)


def _emit_kv(tag: str, **fields: Any) -> None:
    ordered = [f"{key}={fields[key]}" for key in fields]
    print(f"[{tag}] {' '.join(ordered)}", flush=True)


def _alert_id(alert: dict[str, Any]) -> str | None:
    value = alert.get("id") or alert.get("alert_id")
    return str(value) if value else None


def _alert_source_ip(alert: dict[str, Any]) -> str | None:
    value = alert.get("source_ip")
    return str(value) if value else None


def _alert_host(alert: dict[str, Any]) -> str | None:
    value = alert.get("hostname") or alert.get("target_host")
    return str(value) if value else None


def _default_action(observation: dict[str, Any]) -> SecurityAction:
    active_alerts = observation.get("active_alerts") or []
    blocked_ips = set(observation.get("blocked_ips") or [])
    isolated_hosts = set(observation.get("isolated_hosts") or [])

    for alert in active_alerts:
        source_ip = _alert_source_ip(alert)
        if source_ip and source_ip not in blocked_ips:
            return SecurityAction(action_type="BlockIP", ip=source_ip)

    for alert in active_alerts:
        target_host = _alert_host(alert)
        if target_host and target_host not in isolated_hosts:
            return SecurityAction(action_type="IsolateHost", hostname=target_host)

    for alert in active_alerts:
        alert_id = _alert_id(alert)
        if alert_id:
            return SecurityAction(action_type="ResolveAlert", alert_id=alert_id)

    return SecurityAction(action_type="WatchLogs")


def _severity_rank(level: str | None) -> int:
    return {"low": 1, "medium": 2, "high": 3, "critical": 4}.get((level or "").lower(), 0)


def _apply_policy_guardrail(observation: dict[str, Any], proposed: SecurityAction) -> SecurityAction:
    active_alerts = observation.get("active_alerts") or []
    if not active_alerts:
        return proposed

    highest_severity = max((_severity_rank(alert.get("severity")) for alert in active_alerts), default=0)
    if highest_severity < 3:
        return proposed

    blocked_ips = set(observation.get("blocked_ips") or [])
    isolated_hosts = set(observation.get("isolated_hosts") or [])
    fallback = _default_action(observation)

    if proposed.action_type == ActionType.WATCH_LOGS:
        return fallback

    if proposed.action_type == ActionType.BLOCK_IP:
        source_ips = {_alert_source_ip(alert) for alert in active_alerts if _alert_source_ip(alert)}
        if proposed.ip and proposed.ip in source_ips:
            unresolved_hosts = [
                _alert_host(alert)
                for alert in active_alerts
                if _alert_host(alert) and _alert_host(alert) not in isolated_hosts
            ]
            if unresolved_hosts and proposed.ip in blocked_ips:
                return SecurityAction(action_type="IsolateHost", hostname=unresolved_hosts[0])

    return proposed


def _parse_action_payload(text: str) -> dict[str, Any] | None:
    content = (text or "").strip()
    if not content:
        return None

    candidates: list[str] = [content]
    code_blocks = re.findall(r"```(?:json)?\s*(.*?)\s*```", content, flags=re.DOTALL | re.IGNORECASE)
    candidates.extend(code_blocks)

    start = content.find("{")
    end = content.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidates.append(content[start : end + 1])

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed

    return None


def _strict_unit_score(value: float) -> float:
    """Clamp any score to the safe open interval (0.01, 0.99)."""
    bounded = max(0.0, min(1.0, value))
    return max(0.01, min(0.99, bounded))


def _score_from_state(state: dict[str, Any], total_reward: float, max_steps: int) -> float:
    network = state.get("network_state", {})
    attack = state.get("attack", {})

    avg_health = (
        float(network.get("web_health", 0.0))
        + float(network.get("db_health", 0.0))
        + float(network.get("app_health", 0.0))
    ) / 3.0

    if attack.get("stopped"):
        containment = 0.9
    elif attack.get("detected"):
        containment = 0.5
    else:
        containment = 0.2

    trajectory = (total_reward + float(max_steps)) / (2.0 * float(max_steps))
    raw = 0.5 * containment + 0.3 * max(0.0, min(1.0, avg_health)) + 0.2 * max(0.0, min(1.0, trajectory))
    return _strict_unit_score(raw)


def decide_action(client: OpenAI | None, model_name: str, observation: dict) -> SecurityAction:
    if client is None:
        return _default_action(observation)

    try:
        response = client.responses.create(
            model=model_name,
            input=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(observation)},
            ],
            temperature=0,
        )
    except Exception:
        return _default_action(observation)

    payload = _parse_action_payload(response.output_text)
    if payload is None:
        return _default_action(observation)

    try:
        action = SecurityAction.model_validate(payload)
        return _apply_policy_guardrail(observation, action)
    except Exception:
        return _default_action(observation)


def run_episode(client: OpenAI | None, model_name: str, tier: int, max_steps: int = 12) -> tuple[float, dict[str, Any]]:
    env = SecurityDefenseEnvironment(seed=17 + tier)
    obs = env.reset(tier=tier)
    total_reward = 0.0

    for step in range(max_steps):
        action = decide_action(client, model_name, obs.model_dump())
        result = env.step(action)
        total_reward += float(result.reward)
        _emit(
            "STEP",
            {
                "tier": tier,
                "step": step,
                "action": action.model_dump(exclude_none=True),
                "reward": round(float(result.reward), 4),
                "done": bool(result.done),
            },
        )
        obs = result.observation
        if result.done:
            break

    final_state = env.state()
    score = _score_from_state(final_state, total_reward, max_steps)
    return round(score, 4), final_state


def main() -> None:
    api_base_url = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
    model_name = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct-1M")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    hf_token = os.getenv("HF_TOKEN")
    api_key = openai_api_key or hf_token

    client: OpenAI | None = None
    if api_key:
        client = OpenAI(api_key=api_key, base_url=api_base_url)

    _emit(
        "START",
        {
            "model": model_name,
            "api_base_url": api_base_url,
            "llm_enabled": bool(api_key),
        },
    )

    task_scores: dict[str, float] = {}
    for tier, name in ((1, "easy_tier1"), (2, "medium_tier2"), (3, "hard_tier3")):
        _emit_kv("START", task=name, tier=tier)
        score, final_state = run_episode(client=client, model_name=model_name, tier=tier)
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
        _emit_kv(
            "END",
            task=name,
            score=score,
            steps=max(1, int(final_state.get("step_count", 1) or 1)),
        )

    baseline_score = round(_strict_unit_score(sum(task_scores.values()) / 3.0), 4)
    tasks = [
        {"id": "easy_tier1", "grader": "grader:grade_easy_tier1", "score": task_scores["easy_tier1"]},
        {"id": "medium_tier2", "grader": "grader:grade_medium_tier2", "score": task_scores["medium_tier2"]},
        {"id": "hard_tier3", "grader": "grader:grade_hard_tier3", "score": task_scores["hard_tier3"]},
    ]
    tier_scores = {index + 1: score for index, score in enumerate(task_scores.values())}
    final_payload = {
        "task_scores": task_scores,
        "tier_scores": tier_scores,
        "tasks": tasks,
        "baseline_score": baseline_score,
    }
    _emit("END", final_payload)
    print(json.dumps(final_payload, separators=(",", ":"), ensure_ascii=False))


if __name__ == "__main__":
    main()
