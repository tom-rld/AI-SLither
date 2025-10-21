#!/usr/bin/env python3
"""
Training harness to learn an autonomous Snake.io agent from raw screen pixels.

This script leaves the original game code untouched and instead spins up
"MainGame" instances in-process, capturing only what a human player would see
(the rendered frame) and providing actions through virtual mouse inputs.  The
agent is rewarded solely for eliminating opponents and for immediately
consuming the pellets that spawn afterwards, in line with the requested rules.

Usage examples
--------------
Run a quick sanity-check session with rendering enabled for the first
environment:

    ./iasnake.io --episodes 2 --render-preview

Launch a longer multi-environment training job (headless) where the agent is
trained across eight parallel simulations:

    ./iasnake.io --total-steps 200000 --num-envs 8 --batch-size 256

You can interrupt the process at any time; checkpoints will be written to the
folder specified by ``--checkpoint-dir``.
"""

from __future__ import annotations

import argparse
import math
import os
import random
import time
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pygame
import torch
import torch.nn as nn
import torch.optim as optim

# --- Import the untouched game modules ---------------------------------------------------------
from MainGame import MainGame


# ----------------------------------------------------------------------------------------------------------------------
# Utility helpers
# ----------------------------------------------------------------------------------------------------------------------

def ensure_dummy_driver() -> None:
    """Force SDL to run in headless mode when no display is available."""
    if os.environ.get("DISPLAY") or os.environ.get("SDL_VIDEODRIVER"):
        return
    os.environ["SDL_VIDEODRIVER"] = "dummy"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@dataclass
class StepResult:
    observation: np.ndarray
    reward: float
    done: bool
    info: Dict[str, float]


class ScreenPreprocessor:
    """Converts the pygame surface into a stacked grayscale tensor."""

    def __init__(self, stack_size: int = 4, downsample_factor: int = 8) -> None:
        self.stack_size = stack_size
        self.downsample_factor = downsample_factor
        self.frames: Deque[np.ndarray] = deque(maxlen=stack_size)

    def reset(self) -> np.ndarray:
        self.frames.clear()
        blank = np.zeros((88, 125), dtype=np.float32)
        for _ in range(self.stack_size):
            self.frames.append(blank)
        return self._stack()

    def push(self, surface: pygame.Surface) -> np.ndarray:
        arr = pygame.surfarray.array3d(surface)  # (width, height, channel)
        arr = np.transpose(arr, (1, 0, 2))  # -> (height, width, channel)
        arr = arr[:: self.downsample_factor, :: self.downsample_factor]

        # ensure deterministic frame size (crop/pad to 88x125)
        arr = arr[:88, :125]
        if arr.shape[0] < 88:
            pad_rows = 88 - arr.shape[0]
            arr = np.pad(arr, ((0, pad_rows), (0, 0), (0, 0)), mode="constant")
        if arr.shape[1] < 125:
            pad_cols = 125 - arr.shape[1]
            arr = np.pad(arr, ((0, 0), (0, pad_cols), (0, 0)), mode="constant")

        gray = np.dot(arr[..., :3], np.array([0.299, 0.587, 0.114], dtype=np.float32))
        gray /= 255.0
        self.frames.append(gray.astype(np.float32))
        return self._stack()

    def _stack(self) -> np.ndarray:
        assert len(self.frames) == self.stack_size
        return np.stack(self.frames, axis=0)


class SnakeIOEnv:
    """Environment wrapper that exposes MainGame via discrete pixel-based control."""

    def __init__(
        self,
        render: bool = False,
        kill_reward: float = 1.0,
        eat_bonus: float = 0.2,
        eat_window_frames: int = 120,
        frame_skip: int = 2,
        seed: int = 0,
    ) -> None:
        ensure_dummy_driver()
        seed_everything(seed)

        self.render_enabled = render
        self.kill_reward = kill_reward
        self.eat_bonus = eat_bonus
        self.eat_window_frames = eat_window_frames
        self.frame_skip = frame_skip
        self.random = random.Random(seed)

        self.game = MainGame()
        self.game.play = lambda: None  # prevent the built-in game loop from running
        self.game.init()
        self.player = self.game.player

        self.preprocessor = ScreenPreprocessor()
        self.last_score = float(self.player.score)
        self.pending_eat_window = 0
        self.previous_enemy_heads: Dict[int, pygame.Rect] = {}

        self.action_angles = [i * (2 * math.pi / 16) for i in range(16)]

    # ----------------------------------------------------------------------------------
    # Core environment API
    # ----------------------------------------------------------------------------------
    def reset(self) -> np.ndarray:
        self.game.reset_game()
        self.last_score = float(self.player.score)
        self.pending_eat_window = 0
        self.previous_enemy_heads.clear()
        self._sync_enemies()
        pygame.mouse.set_pos(self.game.winDims[0] // 2, self.game.winDims[1] // 2)
        return self.preprocessor.reset()

    def step(self, action: int, boost: bool = False) -> StepResult:
        self._apply_action(action, boost)
        reward = 0.0
        info: Dict[str, float] = {}

        for _ in range(self.frame_skip):
            self._sync_enemies()
            self._advance_frame()
            if self.game.game_over or self.game.quit:
                break

        reward += self._collect_rewards(info)
        observation = self.preprocessor.push(self.game.window)
        done = bool(self.game.game_over or self.game.quit)

        return StepResult(observation=observation, reward=reward, done=done, info=info)

    # ----------------------------------------------------------------------------------
    # Internal helpers
    # ----------------------------------------------------------------------------------
    def _apply_action(self, action: int, boost: bool) -> None:
        angle = self.action_angles[action % len(self.action_angles)]
        radius = min(self.game.winDims) // 2 - 10
        center_x = self.game.winDims[0] // 2
        center_y = self.game.winDims[1] // 2
        target_x = int(center_x + radius * math.cos(angle))
        target_y = int(center_y + radius * math.sin(angle))

        pygame.mouse.set_pos(target_x, target_y)

        if boost:
            pygame.event.post(pygame.event.Event(pygame.MOUSEBUTTONDOWN, {"button": 1, "pos": (target_x, target_y)}))
            pygame.event.post(pygame.event.Event(pygame.MOUSEBUTTONUP, {"button": 1, "pos": (target_x, target_y)}))

    def _advance_frame(self) -> None:
        pygame.event.pump()
        self.game.update()
        if self.render_enabled:
            self.game.render()

    def _sync_enemies(self) -> None:
        self.previous_enemy_heads = {id(s): s.rect.copy() for s in self.game.snakes if s is not self.player}

    def _collect_rewards(self, info: Dict[str, float]) -> float:
        reward = 0.0
        current_enemies = {id(s): s for s in self.game.snakes if s is not self.player}
        removed = set(self.previous_enemy_heads.keys()) - set(current_enemies.keys())

        if removed:
            killed = 0
            for enemy_id in removed:
                head_rect = self.previous_enemy_heads[enemy_id]
                if self._collided_with_player_body(head_rect):
                    reward += self.kill_reward
                    killed += 1
                    self.pending_eat_window = self.eat_window_frames
            if killed:
                info["kills"] = killed

        score_delta = float(self.player.score) - self.last_score
        self.last_score = float(self.player.score)

        if self.pending_eat_window > 0:
            self.pending_eat_window -= 1
            if score_delta > 0:
                reward += self.eat_bonus * score_delta
                info["eat_bonus"] = info.get("eat_bonus", 0.0) + score_delta
                self.pending_eat_window = max(self.pending_eat_window - 5, 0)
        return reward

    def _collided_with_player_body(self, enemy_head: pygame.Rect) -> bool:
        if enemy_head.colliderect(self.player.rect):
            return True
        for segment in self.player.segments:
            if enemy_head.colliderect(segment.rect):
                return True
        return False


# ----------------------------------------------------------------------------------------------------------------------
# DQN agent implementation
# ----------------------------------------------------------------------------------------------------------------------


class ConvDQN(nn.Module):
    def __init__(self, num_actions: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((6, 8)),
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 6 * 8, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.buffer: Deque[Tuple[np.ndarray, int, float, np.ndarray, bool]] = deque(maxlen=capacity)

    def push(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.stack(states, axis=0),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.stack(next_states, axis=0),
            np.array(dones, dtype=np.bool_),
        )

    def __len__(self) -> int:
        return len(self.buffer)


@dataclass
class TrainingConfig:
    total_steps: int = 250_000
    warmup_steps: int = 5_000
    gamma: float = 0.99
    batch_size: int = 128
    lr: float = 1e-4
    update_target_every: int = 2_000
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: int = 150_000
    replay_capacity: int = 200_000


class DQNTrainer:
    def __init__(
        self,
        config: TrainingConfig,
        num_envs: int = 1,
        seed: int = 0,
        render_preview: bool = False,
        checkpoint_dir: Optional[Path] = None,
    ) -> None:
        self.config = config
        self.seed = seed
        self.checkpoint_dir = checkpoint_dir
        ensure_dummy_driver()
        seed_everything(seed)

        self.envs = [
            SnakeIOEnv(
                render=(render_preview and idx == 0),
                seed=seed + idx,
            )
            for idx in range(num_envs)
        ]
        self.num_envs = num_envs

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = ConvDQN(num_actions=16).to(self.device)
        self.target_net = ConvDQN(num_actions=16).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.config.lr)
        self.replay = ReplayBuffer(self.config.replay_capacity)

        self.global_step = 0
        self.last_checkpoint_time = time.time()

    def train(self) -> None:
        states = [env.reset() for env in self.envs]
        episode_rewards = Counter()
        episode_lengths = Counter()
        episode_ids = Counter()

        while self.global_step < self.config.total_steps:
            epsilon = self._compute_epsilon(self.global_step)
            actions, boosts = zip(*(self._act(state, epsilon) for state in states))

            step_results = [env.step(action, boost) for env, action, boost in zip(self.envs, actions, boosts)]

            for idx, (state, action, result) in enumerate(zip(states, actions, step_results)):
                self.replay.push(state.copy(), action, result.reward, result.observation.copy(), result.done)
                episode_rewards[idx] += result.reward
                episode_lengths[idx] += 1

            states = [result.observation for result in step_results]

            for idx, result in enumerate(step_results):
                if result.done:
                    episode_ids[idx] += 1
                    print(
                        f"[env {idx}] episode {episode_ids[idx]} finished: reward={episode_rewards[idx]:.2f}, "
                        f"length={episode_lengths[idx]}"
                    )
                    states[idx] = self.envs[idx].reset()
                    episode_rewards[idx] = 0.0
                    episode_lengths[idx] = 0

            if self.global_step >= self.config.warmup_steps and len(self.replay) >= self.config.batch_size:
                self._learn()

            if self.global_step % self.config.update_target_every == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

            if self.checkpoint_dir and time.time() - self.last_checkpoint_time > 600:
                self._save_checkpoint()

            self.global_step += self.num_envs

        if self.checkpoint_dir:
            self._save_checkpoint(force=True)

    # ----------------------------------------------------------------------------------
    # Internal methods
    # ----------------------------------------------------------------------------------
    def _act(self, state: np.ndarray, epsilon: float) -> Tuple[int, bool]:
        if random.random() < epsilon:
            return random.randrange(16), False
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
        action = int(q_values.argmax(dim=1).item())
        boost = False
        return action, boost

    def _learn(self) -> None:
        states, actions, rewards, next_states, dones = self.replay.sample(self.config.batch_size)

        state_tensor = torch.from_numpy(states).float().to(self.device)
        next_state_tensor = torch.from_numpy(next_states).float().to(self.device)
        action_tensor = torch.from_numpy(actions).long().to(self.device)
        reward_tensor = torch.from_numpy(rewards).float().to(self.device)
        done_tensor = torch.from_numpy(dones.astype(np.float32)).float().to(self.device)

        q_values = self.policy_net(state_tensor).gather(1, action_tensor.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q_values = self.target_net(next_state_tensor).max(1)[0]
            targets = reward_tensor + (1.0 - done_tensor) * self.config.gamma * next_q_values

        loss = nn.MSELoss()(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
        self.optimizer.step()

    def _compute_epsilon(self, step: int) -> float:
        fraction = min(1.0, step / self.config.epsilon_decay)
        return self.config.epsilon_start + fraction * (self.config.epsilon_end - self.config.epsilon_start)

    def _save_checkpoint(self, force: bool = False) -> None:
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        path = self.checkpoint_dir / f"snakeio_step_{self.global_step}.pt"
        torch.save({
            "step": self.global_step,
            "policy_state_dict": self.policy_net.state_dict(),
            "target_state_dict": self.target_net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }, path)
        self.last_checkpoint_time = time.time()
        if force:
            print(f"Checkpoint saved to {path}")


# ----------------------------------------------------------------------------------------------------------------------
# Command-line interface
# ----------------------------------------------------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    defaults = TrainingConfig()

    parser = argparse.ArgumentParser(description="Train a Snake.io DQN agent from pixels.")
    parser.add_argument("--total-steps", type=int, default=defaults.total_steps)
    parser.add_argument("--warmup-steps", type=int, default=defaults.warmup_steps)
    parser.add_argument("--gamma", type=float, default=defaults.gamma)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--lr", type=float, default=defaults.lr)
    parser.add_argument("--update-target", type=int, default=defaults.update_target_every)
    parser.add_argument("--epsilon-start", type=float, default=defaults.epsilon_start)
    parser.add_argument("--epsilon-end", type=float, default=defaults.epsilon_end)
    parser.add_argument("--epsilon-decay", type=int, default=defaults.epsilon_decay)
    parser.add_argument("--replay-capacity", type=int, default=defaults.replay_capacity)
    parser.add_argument("--num-envs", type=int, default=1, help="Number of parallel simulations to run.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--render-preview", action="store_true", help="Render the first environment for monitoring.")
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--episodes", type=int, default=None, help="Optional: limit to N episodes instead of total steps.")
    return parser


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = build_arg_parser().parse_args(argv)

    config = TrainingConfig(
        total_steps=args.total_steps,
        warmup_steps=args.warmup_steps,
        gamma=args.gamma,
        batch_size=args.batch_size,
        lr=args.lr,
        update_target_every=args.update_target,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay,
        replay_capacity=args.replay_capacity,
    )

    trainer = DQNTrainer(
        config=config,
        num_envs=args.num_envs,
        seed=args.seed,
        render_preview=args.render_preview,
        checkpoint_dir=args.checkpoint_dir,
    )

    if args.episodes is not None:
        # Run in episode-bound mode (useful for smoke tests)
        for episode in range(args.episodes):
            states = [env.reset() for env in trainer.envs]
            done_envs = [False] * trainer.num_envs
            episode_reward = [0.0] * trainer.num_envs
            while not all(done_envs):
                epsilon = trainer._compute_epsilon(trainer.global_step)
                actions, boosts = zip(*(trainer._act(state, epsilon) for state in states))
                results = [env.step(action, boost) for env, action, boost in zip(trainer.envs, actions, boosts)]
                states = [result.observation for result in results]
                for idx, result in enumerate(results):
                    episode_reward[idx] += result.reward
                    done_envs[idx] = result.done
                    if result.done:
                        trainer.envs[idx].reset()
                trainer.global_step += trainer.num_envs
            print(f"Episode {episode + 1} finished with rewards: {episode_reward}")
    else:
        trainer.train()


if __name__ == "__main__":
    main()
