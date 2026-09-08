"""Headless baseline sweep: `python -m atc.arcade.bench [--episodes N]`."""
from __future__ import annotations

import argparse
import statistics
from pathlib import Path

from atc.arcade.bots import GreedyRouter, play
from atc.arcade.engine import ArcadeEngine, load_arcade_config

CONFIG = Path(__file__).resolve().parents[3] / "configs" / "arcade_m1.json"


def main() -> int:
    ap = argparse.ArgumentParser(prog="atc.arcade.bench")
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--config", type=Path, default=CONFIG)
    a = ap.parse_args()

    base = load_arcade_config(a.config).raw
    runs = []
    for i in range(a.episodes):
        cfg = load_arcade_config({**base, "seed": 4000 + i})
        runs.append(play(ArcadeEngine(cfg), GreedyRouter(cfg.raw)))
    m = lambda k: statistics.mean(r[k] for r in runs)
    print(f"greedy baseline over {len(runs)} seeds")
    print(f"  landed    mean {m('landed'):6.2f}  median {statistics.median(r['landed'] for r in runs):5.1f}"
          f"  best {max(r['landed'] for r in runs)}")
    print(f"  survived  mean {m('survived_s'):6.1f}s best {max(r['survived_s'] for r in runs):.0f}s")
    print(f"  score     mean {m('score'):7.1f}")
    print(f"  crashed   {sum(r['crashed'] for r in runs)}/{len(runs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
