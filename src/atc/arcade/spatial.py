"""The grid encoding — the one and only copy.

Everything here works from **plain data**: the observation dict the server sends
over the wire, and the config dict it hands out when a game starts. That is
deliberate. The training env, the server's own pilot and a remote agent on
someone else's laptop all call the same function with the same inputs, so a
remote agent cannot drift from what the policy was trained on. An
engine-object-shaped encoder would have forced remote agents to reimplement it,
and a second implementation is a bug waiting to happen.
"""
from __future__ import annotations

import math

import numpy as np

CH_SELF, CH_SELF_SIN, CH_SELF_COS, CH_OTHERS, CH_PATHS, CH_TARGET, CH_ZONES, CH_CONFLICT = range(8)
N_CHANNELS = 8

CHANNELS = (
    "self", "self_heading_sin", "self_heading_cos", "other_traffic",
    "committed_paths", "target_zone", "all_zones", "conflicts",
)


def cell_of(x: float, y: float, map_w: float, map_h: float, gw: int, gh: int) -> tuple[int, int]:
    cx = min(gw - 1, max(0, int(x / map_w * gw)))
    cy = min(gh - 1, max(0, int(y / map_h * gh)))
    return cx, cy


def cell_centre(idx: int, map_w: float, map_h: float, gw: int, gh: int) -> tuple[float, float]:
    cy, cx = divmod(int(idx), gw)
    return ((cx + 0.5) / gw * map_w, (cy + 0.5) / gh * map_h)


def needs_route(state: dict, pending=()) -> str | None:
    """Next aircraft awaiting a route — lowest id, so every client agrees."""
    for a in sorted(state["aircraft"], key=lambda a: a["id"]):
        if not a["path"] and a["id"] not in pending:
            return a["id"]
    return None


def leg(x0: float, y0: float, x1: float, y1: float, step: float = 6.0):
    """A straight run sampled at mouse-drag density."""
    n = max(1, int(math.hypot(x1 - x0, y1 - y0) / step))
    return [(x0 + (x1 - x0) * i / n, y0 + (y1 - y0) * i / n) for i in range(1, n + 1)]


def build_obs(state: dict, config: dict, gw: int, gh: int, selected: str | None) -> np.ndarray:
    """(observation, config) -> float32 array of shape (8, gh, gw)."""
    mw, mh = config["map"]["w"], config["map"]["h"]
    zones = {z["id"]: z for z in config["zones"]}
    g = np.zeros((N_CHANNELS, gh, gw), dtype=np.float32)

    for z in config["zones"]:
        cx, cy = cell_of(z["pos"][0], z["pos"][1], mw, mh, gw, gh)
        g[CH_ZONES, cy, cx] = 1.0

    for a in state["aircraft"]:
        cx, cy = cell_of(a["x"], a["y"], mw, mh, gw, gh)
        if a["id"] == selected:
            g[CH_SELF, cy, cx] = 1.0
            r = math.radians(a["heading"])
            g[CH_SELF_SIN, cy, cx] = (math.sin(r) + 1) / 2
            g[CH_SELF_COS, cy, cx] = (math.cos(r) + 1) / 2
            z = zones.get(a["zone"])
            if z:
                zx, zy = cell_of(z["pos"][0], z["pos"][1], mw, mh, gw, gh)
                g[CH_TARGET, zy, zx] = 1.0
        else:
            g[CH_OTHERS, cy, cx] = min(1.0, g[CH_OTHERS, cy, cx] + 1.0)
        for (px, py) in a["path"][::3]:
            pcx, pcy = cell_of(px, py, mw, mh, gw, gh)
            g[CH_PATHS, pcy, pcx] = 1.0
        if a.get("conflict"):
            g[CH_CONFLICT, cy, cx] = 1.0
    return g
