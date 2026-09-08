"""PPO training over the spatial action map.

    python -m atc.arcade.train --steps 2000000

The observation and the action are the same grid, so the policy is a small CNN:
convolutions preserve the spatial layout, which is what makes "avoid the
aircraft two cells north-east" transfer to every position on the map rather
than being memorised per location.

SB3's stock NatureCNN assumes Atari-sized frames (>=36x36) and will not accept a
20x14 grid, so the feature extractor here is a small padded-conv stack that keeps
resolution instead of downsampling it away.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch as th
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor

from atc.arcade.gym_env import ATCArcadeEnv

OUT = Path(__file__).resolve().parents[3] / "runs"


def _tb_dir():
    """Only log to tensorboard if it is actually installed."""
    try:
        import tensorboard  # noqa: F401
    except ImportError:
        return None
    return str(OUT / "tb")


class GridCNN(BaseFeaturesExtractor):
    """Small fully-padded CNN: 20x14 in, 20x14 feature map, then a dense head."""

    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        c = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(c, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.Flatten(),
        )
        with th.no_grad():
            n = self.cnn(th.zeros(1, *observation_space.shape)).shape[1]
        self.linear = nn.Sequential(nn.Linear(n, features_dim), nn.ReLU())

    def forward(self, x: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(x))


def make_env(rank: int, delay_max: int, shaping: float):
    def _f():
        import random
        # Randomise the action delay per episode so the policy cannot overfit one
        # latency. This is what makes it survive a real network round trip later,
        # without ever training over the wire.
        d = random.randint(0, delay_max) if delay_max else 0
        return ATCArcadeEnv(action_delay=d, shaping=shaping)
    return _f


def main() -> int:
    ap = argparse.ArgumentParser(prog="atc.arcade.train")
    ap.add_argument("--steps", type=int, default=1_000_000)
    ap.add_argument("--envs", type=int, default=8)
    ap.add_argument("--delay-max", type=int, default=0, help="max action delay in ticks (20 = 1s)")
    ap.add_argument("--shaping", type=float, default=0.0)
    ap.add_argument("--name", default="ppo_grid")
    ap.add_argument("--subproc", action="store_true", help="separate processes per env")
    a = ap.parse_args()

    OUT.mkdir(exist_ok=True)
    cls = SubprocVecEnv if a.subproc else DummyVecEnv
    venv = VecMonitor(cls([make_env(i, a.delay_max, a.shaping) for i in range(a.envs)]))

    model = PPO(
        "CnnPolicy", venv,
        policy_kwargs=dict(features_extractor_class=GridCNN,
                           features_extractor_kwargs=dict(features_dim=256)),
        n_steps=256, batch_size=512, n_epochs=4,
        learning_rate=3e-4, gamma=0.995, gae_lambda=0.95,
        ent_coef=0.01, clip_range=0.2,
        verbose=1, tensorboard_log=_tb_dir(),
        device="cpu",   # a 20x14x8 grid is far too small to beat CPU via GPU transfer
    )
    model.learn(total_timesteps=a.steps, progress_bar=False,
                callback=CheckpointCallback(save_freq=max(1, 100_000 // a.envs),
                                            save_path=str(OUT), name_prefix=a.name))
    model.save(OUT / a.name)
    print(f"saved {OUT / a.name}.zip")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
