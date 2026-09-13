"""Score a trained policy against the scripted bars.

    python -m atc.arcade.evaluate runs/ppo_grid.zip --episodes 25
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

from atc.arcade.gym_env import ATCArcadeEnv

# One table, in configs/, measured under a single protocol. These used to be
# literals here AND in server.py, and the two drifted apart.
_B = json.loads((Path(__file__).resolve().parents[3] / "configs" / "baselines.json").read_text())
BARS = {_B["labels"][k]: v for k, v in _B["scores"].items() if k != "this_policy"}
PROTOCOL = _B["protocol"]


def aligned_policy(env, obs):
    """The hand-coded bar: aim at the zone, via the near threshold for runways."""
    aid = env._needs_route()
    if aid is None:
        return 0
    ac = env.engine.aircraft[aid]
    z = env.cfg.zone(env.cfg.types[ac.type].zone)
    if z is None:
        return 0
    if z.kind == "pad":
        cx, cy = env._cell(z.pos[0], z.pos[1])
        return int(cy * env.gw + cx)
    h = math.radians(z.heading_deg)
    ax, ay = math.sin(h), -math.cos(h)
    reach = z.length / 2 + 110
    gates = [(z.pos[0] + ax*reach, z.pos[1] + ay*reach),
             (z.pos[0] - ax*reach, z.pos[1] - ay*reach)]
    if math.hypot(ac.x - z.pos[0], ac.y - z.pos[1]) > reach * 1.2:
        g = min(gates, key=lambda p: math.hypot(p[0]-ac.x, p[1]-ac.y))
        cx, cy = env._cell(*g)
    else:
        cx, cy = env._cell(z.pos[0], z.pos[1])
    return int(cy * env.gw + cx)


def run(policy, episodes: int, delay: int, gw: int = 20, gh: int = 14, seed0: int = 500):
    env = ATCArcadeEnv(grid_w=gw, grid_h=gh, action_delay=delay)
    out = []
    for i in range(episodes):
        obs, _ = env.reset(seed=seed0 + i)
        done = False
        while not done:
            obs, _, term, trunc, info = env.step(policy(env, obs))
            done = term or trunc
        out.append((info["landed"], info["time_s"]))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(prog="atc.arcade.evaluate")
    ap.add_argument("model", nargs="?", default=None)
    ap.add_argument("--episodes", type=int, default=25)
    ap.add_argument("--delay", type=int, default=0)
    ap.add_argument("--grid-w", type=int, default=20)
    ap.add_argument("--grid-h", type=int, default=14)
    a = ap.parse_args()

    rows = []
    if a.model:
        from stable_baselines3 import PPO
        m = PPO.load(a.model, device="cpu")
        rows.append(("trained policy",
                     run(lambda e, o: int(m.predict(o, deterministic=True)[0]),
                         a.episodes, a.delay, a.grid_w, a.grid_h)))
    rows.append(("aim + align (bar)", run(aligned_policy, a.episodes, a.delay,
                                          a.grid_w, a.grid_h)))

    print(f"{'policy':<22}{'landed':>9}{'survived':>11}   (delay {a.delay} ticks)")
    for name, res in rows:
        print(f"{name:<22}{statistics.mean(r[0] for r in res):>9.2f}"
              f"{statistics.mean(r[1] for r in res):>10.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
