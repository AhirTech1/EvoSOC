from __future__ import annotations

from dataclasses import dataclass
import json
from random import Random
from typing import Any

try:
    from models import ActionType, SecurityAction, SecurityAlert, SecurityLogEvent, SecurityObservation
except ModuleNotFoundError:
    from security_env.models import ActionType, SecurityAction, SecurityAlert, SecurityLogEvent, SecurityObservation

try:
    from openenv_core.env import Environment, StepResult
except Exception:  # Local fallback for development without openenv-core
    @dataclass
    class StepResult:
        observation: SecurityObservation
        reward: float
        done: bool
        info: dict[str, Any] | None = None

    class Environment:  # type: ignore[override]
        pass


class SecurityDefenseEnvironment(Environment):
    def __init__(self, seed: int | None = None, max_steps: int = 50) -> None:
        self.rng = Random(seed)
        self.max_steps = max_steps
        self.episode_id = 0
        self.step_count = 0
        self.network_state: dict[str, Any] = {}
        self.log_buffer: list[SecurityLogEvent] = []
        self.active_alerts: list[SecurityAlert] = []
        self.scenario_tier = 1
        self.current_attack: dict[str, Any] = {}

    def state(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "step_count": self.step_count,
            "tier": self.scenario_tier,
            "network_state": self.network_state,
            "attack": self.current_attack,
        }

    def _append_log(self, event: SecurityLogEvent) -> None:
        self.log_buffer.append(event)
        self.log_buffer = self.log_buffer[-10:]

    def _open_alert(self, alert: SecurityAlert) -> None:
        for existing in self.active_alerts:
            if existing.id == alert.id:
                return
        self.active_alerts.append(alert)

    def _generate_tier_baseline(self) -> None:
        attack = self.current_attack
        ts = self.step_count

        if self.scenario_tier == 1:
            self._append_log(
                SecurityLogEvent(
                    ts=ts,
                    event_type="auth_failed",
                    severity="high",
                    source_ip=attack["attacker_ip"],
                    destination="Web-Server",
                    message="Multiple failed admin login attempts detected.",
                    alert_id="ALERT-BRUTE-1",
                )
            )
            self._open_alert(
                SecurityAlert(
                    id="ALERT-BRUTE-1",
                    title="Brute Force Attack",
                    severity="high",
                    hostname="Web-Server",
                    source_ip=attack["attacker_ip"],
                )
            )
            if not attack["stopped"]:
                self.network_state["web_health"] = max(0.0, self.network_state["web_health"] - 0.06)

        elif self.scenario_tier == 2:
            self._append_log(
                SecurityLogEvent(
                    ts=ts,
                    event_type="lateral_scan",
                    severity="critical",
                    source_ip=attack["source_ip"],
                    destination="DB-Server",
                    message="HR-Workstation scanning DB server ports.",
                    alert_id="ALERT-LAT-2",
                )
            )
            self._open_alert(
                SecurityAlert(
                    id="ALERT-LAT-2",
                    title="Lateral Movement",
                    severity="critical",
                    hostname="HR-Workstation",
                    source_ip=attack["source_ip"],
                )
            )
            if not attack["stopped"]:
                self.network_state["db_health"] = max(0.0, self.network_state["db_health"] - 0.08)

        else:
            self._append_log(
                SecurityLogEvent(
                    ts=ts,
                    event_type="data_egress",
                    severity="medium",
                    source_ip="App-Server",
                    destination=attack["exfil_ip"],
                    message="Low-volume outbound transfer to unclassified external IP.",
                    alert_id="ALERT-EXFIL-3",
                )
            )
            self._open_alert(
                SecurityAlert(
                    id="ALERT-EXFIL-3",
                    title="Possible Data Exfiltration",
                    severity="high",
                    hostname="App-Server",
                    source_ip=attack["exfil_ip"],
                )
            )
            if not attack["stopped"]:
                self.network_state["app_health"] = max(0.0, self.network_state["app_health"] - 0.05)

        self._append_log(
            SecurityLogEvent(
                ts=ts,
                event_type="normal_traffic",
                severity="low",
                source_ip="203.0.113.10",
                destination="Web-Server",
                message="Legitimate customer browsing session.",
                legitimate=True,
            )
        )

    def _sample_observation(self) -> SecurityObservation:
        return SecurityObservation(
            tier=int(self.scenario_tier),
            web_health=float(self.network_state["web_health"]),
            db_health=float(self.network_state["db_health"]),
            app_health=float(self.network_state["app_health"]),
            blocked_ips=list(self.network_state["blocked_ips"]),
            isolated_hosts=list(self.network_state["isolated_hosts"]),
            active_alerts=[item for item in self.active_alerts if item.status == "open"],
            log_buffer_json=json.dumps([entry.model_dump() for entry in self.log_buffer]),
        )

    def _scenario(self, tier: int) -> dict[str, Any]:
        if tier == 1:
            return {
                "name": "Brute Force",
                "attacker_ip": "198.51.100.24",
                "detected": False,
                "stopped": False,
            }
        if tier == 2:
            return {
                "name": "Lateral Movement",
                "source_ip": "10.10.4.23",
                "source_host": "HR-Workstation",
                "detected": False,
                "stopped": False,
            }
        return {
            "name": "Data Exfiltration",
            "exfil_ip": "203.0.113.66",
            "detected": False,
            "stopped": False,
        }

    def reset(self, tier: int | None = None) -> SecurityObservation:
        self.episode_id += 1
        self.step_count = 0

        if tier is None:
            self.scenario_tier = ((self.episode_id - 1) % 3) + 1
        else:
            self.scenario_tier = max(1, min(3, int(tier)))

        self.network_state = {
            "web_health": 1.0,
            "db_health": 1.0,
            "app_health": 1.0,
            "blocked_ips": set(),
            "isolated_hosts": set(),
            "firewall_rules": {},
        }
        self.log_buffer = []
        self.active_alerts = []
        self.current_attack = self._scenario(self.scenario_tier)

        self._generate_tier_baseline()
        return self._sample_observation()

    def discrete_action_to_model(self, action_idx: int) -> SecurityAction:
        open_alert = next((item for item in self.active_alerts if item.status == "open"), None)

        if action_idx == 0:
            return SecurityAction(action_type=ActionType.WATCH_LOGS)
        if action_idx == 1:
            target_ip = self.current_attack.get("attacker_ip") or self.current_attack.get("source_ip") or self.current_attack.get("exfil_ip")
            return SecurityAction(action_type=ActionType.BLOCK_IP, ip=str(target_ip))
        if action_idx == 2:
            target_host = self.current_attack.get("source_host") or "App-Server"
            return SecurityAction(action_type=ActionType.ISOLATE_HOST, hostname=str(target_host))

        alert_id = open_alert.id if open_alert else "ALERT-UNKNOWN"
        return SecurityAction(action_type=ActionType.RESOLVE_ALERT, alert_id=alert_id)

    def step(self, action: int | SecurityAction | dict[str, Any]) -> StepResult:
        if isinstance(action, int):
            action_model = self.discrete_action_to_model(action)
        elif isinstance(action, dict):
            action_model = SecurityAction.model_validate(action)
        else:
            action_model = action

        self.step_count += 1
        reward = 0.0
        attack = self.current_attack

        if action_model.action_type == ActionType.WATCH_LOGS:
            if not attack["detected"]:
                attack["detected"] = True
                reward += 0.1
                self._append_log(
                    SecurityLogEvent(
                        ts=self.step_count,
                        event_type="analyst_watch",
                        severity="low",
                        source_ip="SOC-Agent",
                        destination="SIEM",
                        message="Agent reviewed log stream and marked IoCs for follow-up.",
                    )
                )

        elif action_model.action_type == ActionType.BLOCK_IP:
            ip = action_model.ip or ""
            attack_ip = attack.get("attacker_ip") or attack.get("source_ip") or attack.get("exfil_ip")
            self.network_state["blocked_ips"].add(ip)
            self.network_state["firewall_rules"][ip] = "deny"
            if ip == attack_ip:
                reward += 0.1
                if self.scenario_tier in {1, 3}:
                    attack["stopped"] = True
                    reward += 1.0
            else:
                reward -= 0.5

        elif action_model.action_type == ActionType.ISOLATE_HOST:
            hostname = action_model.hostname or ""
            self.network_state["isolated_hosts"].add(hostname)
            if self.scenario_tier == 2 and hostname == "HR-Workstation":
                reward += 0.1
                attack["stopped"] = True
                reward += 1.0
            elif hostname != "HR-Workstation":
                reward -= 0.5

        elif action_model.action_type == ActionType.RESOLVE_ALERT:
            matched = False
            for alert in self.active_alerts:
                if alert.id == action_model.alert_id and alert.status == "open":
                    alert.status = "resolved"
                    matched = True
                    if attack["detected"]:
                        reward += 0.1
                    break
            if not matched:
                reward -= 0.2

        self._generate_tier_baseline()

        avg_health = (
            float(self.network_state["web_health"])
            + float(self.network_state["db_health"])
            + float(self.network_state["app_health"])
        ) / 3.0

        done = (
            self.step_count >= self.max_steps
            or avg_health <= 0.25
            or bool(attack.get("stopped"))
        )

        info = {
            "attack_name": attack.get("name"),
            "detected": bool(attack.get("detected")),
            "stopped": bool(attack.get("stopped")),
            "avg_health": round(avg_health, 4),
        }
        try:
            return StepResult(
                observation=self._sample_observation(),
                reward=reward,
                done=done,
                info=info,
            )
        except TypeError:
            return StepResult(
                observation=self._sample_observation(),
                reward=reward,
                done=done,
            )
