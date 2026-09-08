"""Gymnasium wrapper — spatial action map over a coarse grid.

Action space is one categorical over grid cells: "where should this aircraft go
next". Painting a whole route as a multi-binary grid would make almost every
action an invalid, disconnected blob, so the agent would spend millions of steps
learning connectivity before it ever learned traffic. One cell at a time is
always a legal action, and multi-leg routes emerge from repeated decisions.

Observation is the SAME grid as the action, stacked as channels, so a
fully-convolutional policy is translation-equivariant: "avoid the aircraft at
this cell" generalises across the map instead of being memorised per location.
"""
from __future__ import annotations

import math
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from atc.arcade.engine import ArcadeEngine, load_arcade_config

CONFIG = Path(__file__).resolve().parents[3] / "configs" / "arcade_m1.json"

CH_SELF, CH_SELF_SIN, CH_SELF_COS, CH_OTHERS, CH_PATHS, CH_TARGET, CH_ZONES, CH_CONFLICT = range(8)
N_CHANNELS = 8


class ATCArcadeEnv(gym.Env):
    """One env step = one routing decision for one aircraft."""

    metadata = {"render_modes": []}

    def __init__(self, grid_w: int = 20, grid_h: int = 14, config=CONFIG,
                 seed: int | None = None, max_decisions: int = 600,
                 action_delay: int = 0, shaping: float = 0.0):
        super().__init__()
        self.gw, self.gh = grid_w, grid_h
        self.base = load_arcade_config(config).raw
        self.max_decisions = max_decisions
        # Ticks between choosing an action and it taking effect. Training with a
        # realistic delay is how a policy becomes robust to network latency —
        # see README; training over real HTTP is far too slow to be practical.
        self.action_delay = action_delay
        self.shaping = shaping
        self._seed = seed

        self.action_space = spaces.Discrete(grid_w * grid_h)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(N_CHANNELS, grid_h, grid_w), dtype=np.float32)

    # ── grid helpers ────────────────────────────────────────────────────────
    def _cell(self, x: float, y: float) -> tuple[int, int]:
        cx = min(self.gw - 1, max(0, int(x / self.cfg.map_w * self.gw)))
        cy = min(self.gh - 1, max(0, int(y / self.cfg.map_h * self.gh)))
        return cx, cy

    def _cell_centre(self, idx: int) -> tuple[float, float]:
        cy, cx = divmod(idx, self.gw)
        return ((cx + 0.5) / self.gw * self.cfg.map_w,
                (cy + 0.5) / self.gh * self.cfg.map_h)

    # ── lifecycle ───────────────────────────────────────────────────────────
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        s = seed if seed is not None else (self._seed if self._seed is not None
                                           else int(self.np_random.integers(1 << 30)))
        self.cfg = load_arcade_config({**self.base, "seed": int(s)})
        self.engine = ArcadeEngine(self.cfg)
        self.decisions = 0
        self._pending: list[tuple[int, str, int]] = []      # (fire_tick, ac_id, cell)
        self._advance_to_decision()
        return self._obs(), {}

    def _needs_route(self) -> str | None:
        """Next aircraft awaiting a route — lowest id, so it is deterministic."""
        for aid in sorted(self.engine.aircraft):
            ac = self.engine.aircraft[aid]
            if ac.state == "flying" and not ac.path:
                if not any(p[1] == aid for p in self._pending):
                    return aid
        return None

    def _advance_to_decision(self, cap: int = 4000) -> None:
        """Run the sim until someone needs a route, or the game ends."""
        for _ in range(cap):
            self._fire_pending()
            if self.engine.game_over or self._needs_route() is not None:
                return
            self.engine.step()

    def _fire_pending(self) -> None:
        due = [p for p in self._pending if p[0] <= self.engine.tick]
        for _, aid, cell in due:
            ac = self.engine.aircraft.get(aid)
            if ac is not None and ac.state == "flying":
                tx, ty = self._cell_centre(cell)
                self.engine.set_path(aid, self._leg(ac.x, ac.y, tx, ty))
        if due:
            self._pending = [p for p in self._pending if p[0] > self.engine.tick]

    @staticmethod
    def _leg(x0, y0, x1, y1, step: float = 6.0):
        n = max(1, int(math.hypot(x1-x0, y1-y0) / step))
        return [(x0 + (x1-x0)*i/n, y0 + (y1-y0)*i/n) for i in range(1, n + 1)]

    # ── step ────────────────────────────────────────────────────────────────
    def step(self, action: int):
        aid = self._needs_route()
        landed_before = self.engine.landed
        dist_before = self._dist_to_zone(aid)

        if aid is not None:
            if self.action_delay > 0:
                self._pending.append((self.engine.tick + self.action_delay, aid, int(action)))
            else:
                ac = self.engine.aircraft[aid]
                tx, ty = self._cell_centre(int(action))
                self.engine.set_path(aid, self._leg(ac.x, ac.y, tx, ty))

        self.decisions += 1
        self._advance_to_decision()

        reward = float(self.engine.landed - landed_before)          # +1 per landing
        if self.shaping and aid is not None:
            after = self._dist_to_zone(aid)
            if dist_before is not None and after is not None:
                reward += self.shaping * (dist_before - after) / max(1.0, self.cfg.map_w)

        terminated = self.engine.game_over
        if terminated:
            reward -= 1.0                                           # collision ends the run
        truncated = self.decisions >= self.max_decisions
        return self._obs(), reward, terminated, truncated, self._info()

    def _dist_to_zone(self, aid: str | None) -> float | None:
        if aid is None:
            return None
        ac = self.engine.aircraft.get(aid)
        if ac is None:
            return None
        z = self.cfg.zone(self.cfg.types[ac.type].zone)
        return math.hypot(ac.x - z.pos[0], ac.y - z.pos[1]) if z else None

    def _info(self) -> dict:
        return {"landed": self.engine.landed, "score": self.engine.score,
                "time_s": self.engine.time_s, "decisions": self.decisions}

    # ── observation ─────────────────────────────────────────────────────────
    def _obs(self) -> np.ndarray:
        g = np.zeros((N_CHANNELS, self.gh, self.gw), dtype=np.float32)
        sel = self._needs_route()
        for z in self.cfg.zones:
            cx, cy = self._cell(z.pos[0], z.pos[1])
            g[CH_ZONES, cy, cx] = 1.0
        for aid in sorted(self.engine.aircraft):
            ac = self.engine.aircraft[aid]
            if ac.state != "flying":
                continue
            cx, cy = self._cell(ac.x, ac.y)
            if aid == sel:
                g[CH_SELF, cy, cx] = 1.0
                r = math.radians(ac.heading)
                g[CH_SELF_SIN, cy, cx] = (math.sin(r) + 1) / 2
                g[CH_SELF_COS, cy, cx] = (math.cos(r) + 1) / 2
                z = self.cfg.zone(self.cfg.types[ac.type].zone)
                if z:
                    zx, zy = self._cell(z.pos[0], z.pos[1])
                    g[CH_TARGET, zy, zx] = 1.0
            else:
                g[CH_OTHERS, cy, cx] = min(1.0, g[CH_OTHERS, cy, cx] + 1.0)
            for (px, py) in ac.path[::3]:
                pcx, pcy = self._cell(px, py)
                g[CH_PATHS, pcy, pcx] = 1.0
        for a in self.engine.observation()["aircraft"]:
            if a["conflict"]:
                cx, cy = self._cell(a["x"], a["y"])
                g[CH_CONFLICT, cy, cx] = 1.0
        return g
