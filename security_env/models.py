from __future__ import annotations

import json
from enum import Enum
from typing import Any
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class ActionType(str, Enum):
    BLOCK_IP = "BlockIP"
    ISOLATE_HOST = "IsolateHost"
    WATCH_LOGS = "WatchLogs"
    RESOLVE_ALERT = "ResolveAlert"


RewardLiteral = float


class SecurityLogEvent(BaseModel):
    ts: int
    event_type: str
    severity: Literal["low", "medium", "high", "critical"]
    source_ip: str
    destination: str
    message: str
    alert_id: str | None = None
    legitimate: bool = False


class SecurityAlert(BaseModel):
    id: str
    title: str
    severity: Literal["low", "medium", "high", "critical"]
    status: Literal["open", "resolved"] = "open"
    hostname: str | None = None
    source_ip: str | None = None


class SecurityAction(BaseModel):
    action_type: ActionType
    ip: str | None = None
    hostname: str | None = None
    alert_id: str | None = None

    @model_validator(mode="after")
    def validate_action_payload(self) -> "SecurityAction":
        if self.action_type == ActionType.BLOCK_IP and not self.ip:
            raise ValueError("BlockIP requires 'ip'.")
        if self.action_type == ActionType.ISOLATE_HOST and not self.hostname:
            raise ValueError("IsolateHost requires 'hostname'.")
        if self.action_type == ActionType.RESOLVE_ALERT and not self.alert_id:
            raise ValueError("ResolveAlert requires 'alert_id'.")
        return self


class SecurityObservation(BaseModel):
    tier: int = Field(ge=1, le=3)
    web_health: float = Field(ge=0.0, le=1.0)
    db_health: float = Field(ge=0.0, le=1.0)
    app_health: float = Field(ge=0.0, le=1.0)
    blocked_ips: list[str]
    isolated_hosts: list[str]
    active_alerts: list[SecurityAlert]
    log_buffer_json: str

    def to_vector(self) -> list[float]:
        events = self.log_events()
        critical_logs = sum(1 for item in events if item.severity in {"high", "critical"})
        suspicious_logs = sum(1 for item in events if not item.legitimate)

        return [
            float(self.tier),
            float(self.web_health),
            float(self.db_health),
            float(self.app_health),
            float(len(self.blocked_ips)),
            float(len(self.isolated_hosts)),
            float(len(self.active_alerts)),
            float(critical_logs),
            float(suspicious_logs),
        ]

    def to_tensor(self, *, dtype: Any = None) -> Any:
        try:
            import torch  # type: ignore
        except Exception as exc:
            raise RuntimeError("torch is required for tensor conversion. Install with: pip install -e .[train]") from exc

        tensor_dtype = dtype if dtype is not None else torch.float32
        return torch.tensor(self.to_vector(), dtype=tensor_dtype)

    def log_events(self) -> list[SecurityLogEvent]:
        raw = json.loads(self.log_buffer_json)
        return [SecurityLogEvent.model_validate(item) for item in raw]


class SecurityReward(BaseModel):
    value: float
