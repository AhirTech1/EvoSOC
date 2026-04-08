from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
import random
import sys

import torch
import torch.nn as nn
import torch.optim as optim

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from server.environment import SecurityDefenseEnvironment


class DQN(nn.Module):
    def __init__(self, input_size: int = 9, hidden_size: int = 64, output_size: int = 4) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


@dataclass
class Transition:
    state: torch.Tensor
    action: int
    reward: float
    next_state: torch.Tensor
    done: float


def select_action(policy: DQN, obs: torch.Tensor, epsilon: float) -> int:
    if random.random() < epsilon:
        return random.randint(0, 3)
    with torch.no_grad():
        q_values = policy(obs.unsqueeze(0))
        return int(torch.argmax(q_values, dim=1).item())


def train(episodes: int = 1000, gamma: float = 0.99, batch_size: int = 32) -> None:
    env = SecurityDefenseEnvironment()
    policy = DQN()
    target = DQN()
    target.load_state_dict(policy.state_dict())
    optimizer = optim.Adam(policy.parameters(), lr=1e-3)

    replay: deque[Transition] = deque(maxlen=5000)
    epsilon = 1.0
    epsilon_min = 0.05
    epsilon_decay = 0.995

    for episode in range(episodes):
        obs = env.reset().to_tensor()
        total_reward = 0.0

        for _ in range(env.max_steps):
            action = select_action(policy, obs, epsilon)
            step_result = env.step(action)
            next_obs = step_result.observation.to_tensor()

            replay.append(
                Transition(
                    state=obs,
                    action=action,
                    reward=float(step_result.reward),
                    next_state=next_obs,
                    done=float(step_result.done),
                )
            )

            obs = next_obs
            total_reward += float(step_result.reward)

            if len(replay) >= batch_size:
                batch = random.sample(replay, batch_size)

                states = torch.stack([item.state for item in batch])
                actions = torch.tensor([item.action for item in batch], dtype=torch.int64).unsqueeze(1)
                rewards = torch.tensor([item.reward for item in batch], dtype=torch.float32)
                next_states = torch.stack([item.next_state for item in batch])
                dones = torch.tensor([item.done for item in batch], dtype=torch.float32)

                q_values = policy(states).gather(1, actions).squeeze(1)
                with torch.no_grad():
                    next_q_values = target(next_states).max(dim=1).values
                    targets = rewards + gamma * next_q_values * (1.0 - dones)

                loss = nn.functional.mse_loss(q_values, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if step_result.done:
                break

        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        if episode % 20 == 0:
            target.load_state_dict(policy.state_dict())
            print(f"episode={episode} total_reward={total_reward:.2f} epsilon={epsilon:.3f}")

    torch.save(policy.state_dict(), "agent/policy.pt")


if __name__ == "__main__":
    train()
