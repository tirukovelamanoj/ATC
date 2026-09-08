"""Thin CLI over the core. Deliberately dumb — no game logic lives here."""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from atc.bots import GreedyBot, play
from atc.config import load_config
from atc.engine import Engine
from atc.observation import to_diagnostic

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "m1_single_runway.json"


def _episode(cfg_path: Path, seed: int | None) -> dict:
    cfg = load_config(cfg_path)
    if seed is not None:
        cfg = load_config({**cfg.raw, "seed": seed})
    e = Engine(cfg)
    play(e, GreedyBot(altitude_bands=cfg.airspace.altitude_bands,
                      runway=cfg.runways[0].id))
    return to_diagnostic(e)


def main() -> int:
    ap = argparse.ArgumentParser(prog="atc")
    ap.add_argument("command", choices=["run", "sweep"])
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--episodes", type=int, default=20)
    a = ap.parse_args()

    if a.command == "run":
        print(json.dumps(_episode(a.config, a.seed), indent=2))
        return 0

    base = load_config(a.config).seed
    runs = [_episode(a.config, base + i) for i in range(a.episodes)]
    scores = [r["score"] for r in runs]
    print(f"greedy baseline over {len(runs)} seeds")
    print(f"  score      mean {statistics.mean(scores):8.1f}  "
          f"median {statistics.median(scores):7.1f}  "
          f"min {min(scores)}  max {max(scores)}")
    for k in ("landings", "separation_violations", "go_arounds",
              "lost_to_fuel", "runway_idle_pct"):
        print(f"  {k:<22} mean {statistics.mean(r[k] for r in runs):6.2f}")
    return 0
