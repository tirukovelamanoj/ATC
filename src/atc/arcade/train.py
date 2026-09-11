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
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor

from atc.arcade.gym_env import ATCArcadeEnv

OUT = Path(__file__).resolve().parents[3] / "runs"


def _device(choice: str) -> str:
    """Pick a device. The convolutions run at full grid resolution, so their
    cost scales with grid area: at 20x14 the CPU is fine, but at 40x28 a single
    update is ~750ms on CPU versus ~90ms on MPS. Measured, not assumed."""
    if choice != "auto":
        return choice
    if th.cuda.is_available():
        return "cuda"
    if th.backends.mps.is_available():
        return "mps"
    return "cpu"


def _tb_dir():
    """Only log to tensorboard if it is actually installed."""
    try:
        import tensorboard  # noqa: F401
    except ImportError:
        return None
    return str(OUT / "tb")


class SpatialCNN(BaseFeaturesExtractor):
    """Fully convolutional: emits ONE LOGIT PER GRID CELL, no dense layer.

    Flattening the feature map into a dense head costs 64*H*W*256 weights — 18M
    at 40x28 — which dominates training time and makes the network anything but
    resolution-independent. A 1x1 convolution produces the same H*W logits from
    ~58k weights that do not grow with the grid at all, and it keeps the policy
    translation-equivariant, which is the entire reason for a spatial action map.
    """

    def __init__(self, observation_space: spaces.Box):
        h, w = observation_space.shape[1], observation_space.shape[2]
        super().__init__(observation_space, h * w)
        c = observation_space.shape[0]
        self.body = nn.Sequential(
            nn.Conv2d(c, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
        )
        self.head = nn.Conv2d(64, 1, 1)          # one logit per cell

    def forward(self, x: th.Tensor) -> th.Tensor:
        return self.head(self.body(x)).flatten(1)


class SpatialPolicy(ActorCriticPolicy):
    """The extractor already produces the action logits, so the usual dense
    action head would just be a redundant H*W x H*W layer. Replace it with an
    identity and rebuild the optimiser so it does not track discarded weights."""

    def _build(self, lr_schedule) -> None:
        super()._build(lr_schedule)
        self.action_net = nn.Identity()
        self.optimizer = self.optimizer_class(
            self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)


def make_env(rank: int, delay_max: int, shaping: float, gw: int, gh: int,
             conflict_penalty: float):
    def _f():
        import random
        # Randomise the action delay per episode so the policy cannot overfit one
        # latency. This is what makes it survive a real network round trip later,
        # without ever training over the wire.
        d = random.randint(0, delay_max) if delay_max else 0
        return ATCArcadeEnv(grid_w=gw, grid_h=gh, action_delay=d, shaping=shaping,
                            conflict_penalty=conflict_penalty)
    return _f


def main() -> int:
    ap = argparse.ArgumentParser(prog="atc.arcade.train")
    ap.add_argument("--steps", type=int, default=1_000_000)
    ap.add_argument("--envs", type=int, default=8)
    ap.add_argument("--delay-max", type=int, default=0, help="max action delay in ticks (20 = 1s)")
    ap.add_argument("--shaping", type=float, default=0.0)
    ap.add_argument("--conflict-penalty", type=float, default=0.0,
                    help="cost per tick spent inside another aircraft's warning radius")
    ap.add_argument("--grid-w", type=int, default=20)
    ap.add_argument("--grid-h", type=int, default=14)
    ap.add_argument("--n-steps", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    ap.add_argument("--name", default="ppo_grid")
    ap.add_argument("--subproc", action="store_true", help="separate processes per env")
    ap.add_argument("--resume", type=Path, default=None,
                    help="continue from a checkpoint instead of starting over")
    a = ap.parse_args()

    OUT.mkdir(exist_ok=True)
    cls = SubprocVecEnv if a.subproc else DummyVecEnv
    venv = VecMonitor(cls([make_env(i, a.delay_max, a.shaping, a.grid_w, a.grid_h, a.conflict_penalty)
                            for i in range(a.envs)]))

    if a.resume:
        # A long run can be killed by the OS; checkpoints make that survivable
        # rather than a total loss. reset_num_timesteps=False keeps the step
        # counter and the tensorboard curve continuous.
        model = PPO.load(a.resume, env=venv, device=_device(a.device))
        model.learn(total_timesteps=a.steps, reset_num_timesteps=False,
                    callback=CheckpointCallback(save_freq=max(1, 500_000 // a.envs),
                                                save_path=str(OUT), name_prefix=a.name))
        model.save(OUT / a.name)
        print(f"saved {OUT / a.name}.zip")
        return 0

    model = PPO(
        SpatialPolicy, venv,
        policy_kwargs=dict(features_extractor_class=SpatialCNN,
                           net_arch=dict(pi=[], vf=[256, 256])),
        n_steps=a.n_steps, batch_size=a.batch_size, n_epochs=4,
        learning_rate=3e-4, gamma=0.995, gae_lambda=0.95,
        ent_coef=0.01, clip_range=0.2,
        verbose=1, tensorboard_log=_tb_dir(),
        device=_device(a.device),
    )
    model.learn(total_timesteps=a.steps, progress_bar=False,
                callback=CheckpointCallback(save_freq=max(1, 500_000 // a.envs),
                                            save_path=str(OUT), name_prefix=a.name))
    model.save(OUT / a.name)
    print(f"saved {OUT / a.name}.zip")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
