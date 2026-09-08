"""Scripted baselines for the arcade engine.

These exist so an LLM's score means something. A number with nothing to compare
against says nothing about whether a model is reasoning well — the useful result
on the graph engine was the GAP between a naive bot and a better one, not either
score alone.

Bots consume ONLY the observation JSON and emit only paths, which is exactly the
surface an agent gets over HTTP. That keeps the agent-facing projection provably
sufficient to play, rather than something kept in sync by review.
"""
from __future__ import annotations

import math

STEP = 6.0          # sample paths at mouse-drag density


def _sample(a: tuple[float, float], b: tuple[float, float]) -> list[list[float]]:
    n = max(1, int(math.hypot(b[0]-a[0], b[1]-a[1]) / STEP))
    return [[a[0] + (b[0]-a[0])*i/n, a[1] + (b[1]-a[1])*i/n] for i in range(1, n+1)]


class GreedyRouter:
    """Send every aircraft straight to its matching zone. No conflict avoidance.

    This is the floor: it does the one thing the game obviously requires (match
    aircraft type to landing zone) and nothing else, so whatever it scores is
    what a controller earns without thinking about traffic at all.
    """

    name = "greedy"

    def __init__(self, config: dict):
        self.zones = {z["id"]: z for z in config["zones"]}

    def _approach(self, ac: dict, z: dict) -> list[list[float]]:
        pos = (ac["x"], ac["y"])
        if z["kind"] == "pad":
            return _sample(pos, (z["pos"][0], z["pos"][1]))
        # line up with the runway axis: pick whichever threshold is nearer and
        # run in along the strip, otherwise the approach arrives across it and
        # fails the alignment check
        h = math.radians(z.get("heading_deg", 0.0))
        ax, ay = math.sin(h), -math.cos(h)
        reach = z["length"] / 2 + 110
        gates = [(z["pos"][0] + ax*reach, z["pos"][1] + ay*reach),
                 (z["pos"][0] - ax*reach, z["pos"][1] - ay*reach)]
        gate = min(gates, key=lambda g: math.hypot(g[0]-pos[0], g[1]-pos[1]))
        return _sample(pos, gate) + _sample(gate, (z["pos"][0], z["pos"][1]))

    def act(self, obs: dict) -> list[tuple[str, list[list[float]]]]:
        out = []
        for ac in obs["aircraft"]:
            if ac["path"]:
                continue                      # already routed; leave it alone
            z = self.zones.get(ac["zone"])
            if z:
                out.append((ac["id"], self._approach(ac, z)))
        return out


def play(engine, bot, max_ticks: int = 24000) -> dict:
    """Drive a headless game to its first crash (or the tick cap)."""
    for _ in range(max_ticks):
        for aid, path in bot.act(engine.observation()):
            engine.set_path(aid, [(p[0], p[1]) for p in path])
        engine.step()
        if engine.game_over:
            break
    return {"score": engine.score, "landed": engine.landed,
            "survived_s": engine.time_s, "crashed": engine.game_over,
            "reason": engine.over_reason}
