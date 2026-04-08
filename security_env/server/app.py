from __future__ import annotations

from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel

from models import SecurityAction, SecurityObservation
from server.environment import SecurityDefenseEnvironment

app = FastAPI(title="Security Defense OpenEnv")
env = SecurityDefenseEnvironment()


class StepRequest(BaseModel):
    action: SecurityAction | int


class ResetRequest(BaseModel):
    tier: int | None = None


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/reset", response_model=SecurityObservation)
def reset(request: ResetRequest) -> SecurityObservation:
    return env.reset(tier=request.tier)


@app.post("/step")
def step(request: StepRequest) -> dict:
    result = env.step(request.action)
    return {
        "observation": result.observation.model_dump(),
        "reward": result.reward,
        "done": result.done,
        "info": getattr(result, "info", None),
    }


@app.get("/state")
def state() -> dict:
    return env.state()


def main() -> None:
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
