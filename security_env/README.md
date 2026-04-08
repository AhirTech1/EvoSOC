---
title: EvoSOC Security Defense v1
emoji: 🛡️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# EvoSOC — Autonomous SOC Analyst Environment (OpenEnv)

EvoSOC is a real-world reinforcement learning environment for Security Operations Center (SOC) automation. An agent monitors security logs, triages alerts, and executes mitigation actions against active attacks in a simulated enterprise network.

## Why this environment matters

SOC analysts perform repetitive but critical tasks: log review, incident triage, host isolation, and firewall containment. EvoSOC captures this workflow with realistic attack patterns, making it useful for evaluating autonomous agents in cybersecurity operations.

## OpenEnv compliance

Implemented assets:

- Typed Pydantic models in `models.py`:
	- `SecurityObservation`
	- `SecurityAction`
	- `SecurityReward`
- Environment interface in `server/environment.py`:
	- `reset()`
	- `step(action)`
	- `state()`
- OpenEnv metadata in `openenv.yaml`

Validated using:

```bash
openenv validate
```

## Observation space

`SecurityObservation` includes:

- Tier indicator (1-3)
- Per-server health (`Web`, `DB`, `App`)
- Blocked IP list
- Isolated host list
- Active alerts (structured)
- `log_buffer_json` containing the last 10 log events

The observation can be converted to tensor features via:

- `to_vector()`
- `to_tensor()`

## Action space

`SecurityAction` supports:

- `BlockIP(ip)`
- `IsolateHost(hostname)`
- `WatchLogs()`
- `ResolveAlert(id)`

## Task suite and difficulty

EvoSOC includes 3 tasks with increasing difficulty and deterministic grading:

1. Easy (Tier 1): Brute force from one noisy external IP — primary mitigation `BlockIP`
2. Medium (Tier 2): Lateral movement from `HR-Workstation` to `DB-Server` — primary mitigation `IsolateHost`
3. Hard (Tier 3): Subtle low-volume data exfiltration to unknown external IP — requires deeper log analysis and containment actions

## Reward design

Reward shaping provides dense trajectory feedback:

- `+1.0` for fully stopping an attack
- `+0.1` for valid intermediate mitigation steps (e.g., detection/log triage, alert resolution)
- `-0.5` for false positives (e.g., blocking legitimate traffic, isolating wrong host)
- Additional small penalties for clearly unproductive actions

## Deterministic graders

`grader.py` evaluates all 3 tiers and emits deterministic scores in `[0.0, 1.0]`.

Run:

```bash
python grader.py
```

## Baseline inference (`inference.py`)

The baseline script uses the OpenAI client and is configured via required environment variables:

- `API_BASE_URL`
- `MODEL_NAME`
- `HF_TOKEN`

### Environment variables (local + Space)

Required:

- `API_BASE_URL` → OpenAI-compatible base URL (for Hugging Face routing: `https://router.huggingface.co/v1`)
- `MODEL_NAME` → model identifier (prefer exact string from HF Playground snippet, e.g. `Qwen/Qwen2.5-7B-Instruct-1M` or provider-suffixed form)
- `HF_TOKEN` → Hugging Face token with Inference permissions

Optional (safe defaults):

- `MAX_STEPS` → max steps per task episode (default in code: 12)
- `PYTHONUNBUFFERED` → set to `1` for real-time logs in containers/Spaces

Local shell example (fish):

```bash
set -xg API_BASE_URL "https://router.huggingface.co/v1"
set -xg MODEL_NAME "Qwen/Qwen2.5-7B-Instruct-1M"
set -xg HF_TOKEN "hf_..."
set -xg PYTHONUNBUFFERED "1"
```

Hugging Face Space Secrets (Settings → Variables and secrets):

- `API_BASE_URL`
- `MODEL_NAME`
- `HF_TOKEN`

It emits structured logs in strict tag format:

- `[START] ...`
- `[STEP] ...`
- `[END] ...`

Run:

```bash
python inference.py
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

For RL training with PyTorch:

```bash
pip install -e .[train]
```

## Local API usage

Start server:

```bash
uvicorn --app-dir . server.app:app --host 0.0.0.0 --port 7860
```

Quick smoke test:

```bash
curl -sS http://127.0.0.1:7860/health
curl -sS -X POST http://127.0.0.1:7860/reset -H 'Content-Type: application/json' -d '{"tier":2}'
curl -sS -X POST http://127.0.0.1:7860/step -H 'Content-Type: application/json' -d '{"action":{"action_type":"WatchLogs"}}'
curl -sS http://127.0.0.1:7860/state
```

## Docker

Build and run:

```bash
docker build -t evosoc:latest .
docker run --rm -p 7860:7860 evosoc:latest
```

## Hugging Face Spaces deployment

Push with OpenEnv CLI:

```bash
openenv push --repo-id 7h0St/security-defense-v1
```

Space should be configured as Docker SDK and tagged for OpenEnv usage.

## Suggested pre-submission validation

```bash
python -m py_compile models.py server/environment.py server/app.py grader.py inference.py
openenv validate
python grader.py
```

## Baseline score expectations

With deterministic policies and seeds, grader outputs reproducible task scores in `[0.0, 1.0]` and a reproducible mean final score.
