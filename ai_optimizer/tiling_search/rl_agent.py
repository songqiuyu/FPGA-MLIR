"""
Tiling Search - Reinforcement Learning Agent (Phase 3-A, Method 2)
Implements a lightweight DQN that learns a tiling policy from repeated
interaction with TilingEnv.  After training, the policy is evaluated and
compared against the greedy heuristic baseline.

Requirements: torch, numpy (optuna for baseline comparison)
Install: pip install torch numpy optuna
"""

import os
import sys
import json
import random
import math
from collections import deque
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'old_version', 'tools'))
from assign_addr import get_tile
from env import TilingEnv

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False
    print("[rl_agent] PyTorch not found. Install: pip install torch")


# ---- Hyperparameters ----
GAMMA        = 0.99
LR           = 1e-3
BATCH_SIZE   = 64
BUFFER_SIZE  = 10_000
EPS_START    = 1.0
EPS_END      = 0.05
EPS_DECAY    = 500
TARGET_SYNC  = 50   # steps between target network updates
TRAIN_STEPS  = 5_000


class QNetwork(nn.Module):
    """Lightweight 3-layer MLP Q-network."""

    def __init__(self, state_dim: int, n_actions: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),   nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buf = deque(maxlen=capacity)

    def push(self, s, a, r, s_next, done):
        self.buf.append((s, a, r, s_next, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buf, batch_size)
        s, a, r, s_next, d = zip(*batch)
        return (np.array(s), np.array(a), np.array(r, dtype=np.float32),
                np.array(s_next), np.array(d, dtype=np.float32))

    def __len__(self):
        return len(self.buf)


class DQNAgent:
    """DQN agent that learns a tiling policy over a distribution of layer shapes."""

    def __init__(self, state_dim: int = TilingEnv.STATE_DIM,
                 n_actions: int = TilingEnv.N_ACTIONS,
                 device: str = "cpu"):
        self.device = torch.device(device) if _HAS_TORCH else None
        self.n_actions = n_actions
        self.steps = 0

        if _HAS_TORCH:
            self.q_net  = QNetwork(state_dim, n_actions).to(self.device)
            self.target = QNetwork(state_dim, n_actions).to(self.device)
            self.target.load_state_dict(self.q_net.state_dict())
            self.optimizer = optim.Adam(self.q_net.parameters(), lr=LR)
            self.loss_fn = nn.SmoothL1Loss()

        self.buf = ReplayBuffer(BUFFER_SIZE)

    def _epsilon(self):
        return EPS_END + (EPS_START - EPS_END) * math.exp(-self.steps / EPS_DECAY)

    def select_action(self, state: np.ndarray) -> int:
        if not _HAS_TORCH or random.random() < self._epsilon():
            return random.randrange(self.n_actions)
        with torch.no_grad():
            t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            return int(self.q_net(t).argmax(dim=1).item())

    def learn(self):
        if not _HAS_TORCH or len(self.buf) < BATCH_SIZE:
            return
        s, a, r, s_next, d = self.buf.sample(BATCH_SIZE)
        s      = torch.FloatTensor(s).to(self.device)
        a      = torch.LongTensor(a).unsqueeze(1).to(self.device)
        r      = torch.FloatTensor(r).unsqueeze(1).to(self.device)
        s_next = torch.FloatTensor(s_next).to(self.device)
        d      = torch.FloatTensor(d).unsqueeze(1).to(self.device)

        q_vals   = self.q_net(s).gather(1, a)
        with torch.no_grad():
            q_next = self.target(s_next).max(1, keepdim=True)[0]
            target = r + GAMMA * q_next * (1 - d)

        loss = self.loss_fn(q_vals, target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
        self.optimizer.step()

        self.steps += 1
        if self.steps % TARGET_SYNC == 0:
            self.target.load_state_dict(self.q_net.state_dict())

    def save(self, path: str):
        if _HAS_TORCH:
            torch.save(self.q_net.state_dict(), path)

    def load(self, path: str):
        if _HAS_TORCH and os.path.exists(path):
            self.q_net.load_state_dict(torch.load(path, map_location=self.device))
            self.target.load_state_dict(self.q_net.state_dict())


def _sample_layer_params() -> Dict:
    """Sample a random layer configuration for training diversity."""
    M = random.choice([16, 32, 64, 128, 256, 512])
    N = random.choice([16, 32, 64, 128, 256, 512])
    R = random.choice([1, 7, 14, 28, 56, 112])
    C = random.choice([1, 7, 14, 28, 56, 112])
    k = random.choice([1, 3, 5, 7])
    s = random.choice([1, 2])
    d = random.choice([1, 2])
    return dict(N=N, M=M, R=R, C=C, k=k, s=s, d=d)


def train(n_steps: int = TRAIN_STEPS, save_path: str = "dqn_tiling.pth") -> DQNAgent:
    """Train DQN agent on a distribution of random layer shapes."""
    if not _HAS_TORCH:
        print("[rl_agent] Skipping training: PyTorch not available.")
        return None

    agent = DQNAgent()
    episode_rewards = []

    params = _sample_layer_params()
    env = TilingEnv(**params)
    state = env.reset()
    ep_reward = 0.0
    ep = 0

    for step in range(n_steps):
        action = agent.select_action(state)
        next_state, reward, done, _ = env.step(action)
        agent.buf.push(state, action, reward, next_state, float(done))
        state = next_state
        ep_reward += reward

        agent.learn()

        if done:
            episode_rewards.append(ep_reward)
            ep += 1
            if ep % 100 == 0:
                mean_r = sum(episode_rewards[-100:]) / 100
                print(f"  Step {step:5d} | Ep {ep:4d} | "
                      f"eps={agent._epsilon():.3f} | mean_r={mean_r:.4f}")
            # New episode with a freshly sampled layer
            params = _sample_layer_params()
            env = TilingEnv(**params)
            state = env.reset()
            ep_reward = 0.0

    agent.save(save_path)
    print(f"\nTraining done. Model saved to {save_path}")
    return agent


def evaluate(agent: DQNAgent, layer_list: List[Dict]) -> Dict:
    """
    Evaluate trained agent vs greedy baseline on a fixed layer list.
    Returns a comparison table.
    """
    results = []
    for layer in layer_list:
        env = TilingEnv(**layer)
        state = env.reset()
        done = False
        # Greedy epsilon=0 policy
        if agent is not None and _HAS_TORCH:
            orig_steps = agent.steps
            agent.steps = 1_000_000  # force epsilon=0
            while not done:
                a = agent.select_action(state)
                state, _, done, _ = env.step(a)
            agent.steps = orig_steps
            rl_tile = env.best_tile
        else:
            rl_tile = None

        greedy_tile = get_tile(
            layer["N"], layer["M"], layer["R"], layer["C"],
            layer["k"], layer["s"], 0, layer["d"])

        results.append({
            "layer": layer,
            "greedy": greedy_tile,
            "rl": rl_tile,
        })
        print(f"  Layer {layer}: greedy={greedy_tile}  rl={rl_tile}")
    return results


if __name__ == "__main__":
    print("=== DQN Tiling Agent Training ===\n")
    agent = train(n_steps=TRAIN_STEPS, save_path="dqn_tiling.pth")

    eval_layers = [
        dict(N=64,  M=64,  R=56, C=56, k=3, s=1, d=1),
        dict(N=256, M=512, R=28, C=28, k=1, s=1, d=1),
        dict(N=512, M=128, R=28, C=28, k=3, s=1, d=1),
    ]
    print("\n=== Evaluation ===")
    evaluate(agent, eval_layers)
