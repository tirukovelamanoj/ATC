"""Episode configuration — §7.

Every rule constant arrives here from JSON; the engine hardcodes nothing.
Seconds are converted to integer ticks at load time so the simulation core
never touches a float for anything that accumulates.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

TICK_MS = 100
TICKS_PER_S = 1000 // TICK_MS
# Fuel is stored as centi-ticks: one tick of normal burn costs exactly 100.
# Burn multipliers are integer percents, so endurance arithmetic stays exact.
CENTITICKS_PER_MIN = 60 * TICKS_PER_S * 100


def _s_to_ticks(seconds: float) -> int:
    return int(round(seconds * TICKS_PER_S))


@dataclass(frozen=True)
class Edge:
    to: str
    traverse_ticks: int


@dataclass(frozen=True)
class Airspace:
    altitude_bands: int
    entry_fixes: tuple[str, ...]
    approach_fix: str
    altitude_change_ticks: int
    edges: dict[str, tuple[Edge, ...]]

    def edge(self, frm: str, to: str) -> Edge | None:
        return next((e for e in self.edges.get(frm, ()) if e.to == to), None)


@dataclass(frozen=True)
class Runway:
    id: str
    landing_occupancy_ticks: int
    trailing_block_ticks: int
    takeoff_occupancy_ticks: int
    final_approach_ticks: int
    dependent_with: tuple[str, ...]


@dataclass(frozen=True)
class Fuel:
    burn_pct: dict[str, int]
    low_fuel_threshold_centiticks: int
    spawn_fuel_centiticks: int
    spawn_fuel_jitter_centiticks: int


@dataclass(frozen=True)
class Scoring:
    landing: int
    fuel_bonus_per_min: int
    separation_violation: int
    go_around: int
    departure_timeout: int
    fuel_exhaustion: int
    collision: int


@dataclass(frozen=True)
class EpisodeConfig:
    config_id: str
    seed: int
    episode_length_ticks: int
    max_concurrent_aircraft: int
    airspace: Airspace
    runways: tuple[Runway, ...]
    separation_ticks: dict[str, int]
    fuel: Fuel
    speed_factor_pct: dict[str, int]
    arrival_rate_per_min: float
    initial_aircraft: int
    wake_weights: dict[str, int]
    scoring: Scoring
    raw: dict = field(repr=False, default_factory=dict)

    def runway(self, rid: str) -> Runway | None:
        return next((r for r in self.runways if r.id == rid), None)

    def wake_separation_ticks(self, leader: str, follower: str) -> int:
        return self.separation_ticks.get(f"{leader}_then_{follower}", 0)


def load_config(source: str | Path | dict) -> EpisodeConfig:
    raw = json.loads(Path(source).read_text()) if not isinstance(source, dict) else source
    a, f, s = raw["airspace"], raw["fuel"], raw["scoring"]

    edges = {
        fix["id"]: tuple(Edge(e["to"], _s_to_ticks(e["traverse_s"])) for e in fix["edges"])
        for fix in a["fixes"]
    }
    return EpisodeConfig(
        config_id=raw["config_id"],
        seed=raw["seed"],
        episode_length_ticks=_s_to_ticks(raw["episode_length_s"]),
        max_concurrent_aircraft=raw["max_concurrent_aircraft"],
        airspace=Airspace(
            altitude_bands=a["altitude_bands"],
            entry_fixes=tuple(a["entry_fixes"]),
            approach_fix=a["approach_fix"],
            altitude_change_ticks=_s_to_ticks(a["altitude_change_s"]),
            edges=edges,
        ),
        runways=tuple(
            Runway(
                id=r["id"],
                landing_occupancy_ticks=_s_to_ticks(r["landing_occupancy_s"]),
                trailing_block_ticks=_s_to_ticks(r["trailing_block_s"]),
                takeoff_occupancy_ticks=_s_to_ticks(r["takeoff_occupancy_s"]),
                final_approach_ticks=_s_to_ticks(r["final_approach_s"]),
                dependent_with=tuple(r.get("dependent_with", ())),
            )
            for r in raw["runways"]
        ),
        separation_ticks={k: _s_to_ticks(v) for k, v in raw["separation_matrix"].items()},
        fuel=Fuel(
            burn_pct={
                "slow": f["burn_slow_pct"], "normal": f["burn_normal_pct"],
                "fast": f["burn_fast_pct"], "hold": f["burn_hold_pct"],
            },
            low_fuel_threshold_centiticks=f["low_fuel_threshold_min"] * CENTITICKS_PER_MIN,
            spawn_fuel_centiticks=f["spawn_fuel_min"] * CENTITICKS_PER_MIN,
            spawn_fuel_jitter_centiticks=f["spawn_fuel_jitter_min"] * CENTITICKS_PER_MIN,
        ),
        speed_factor_pct=dict(raw["speed_factor_pct"]),
        arrival_rate_per_min=raw["arrivals"]["rate_per_min"],
        initial_aircraft=raw["arrivals"].get("initial_aircraft", 0),
        wake_weights=dict(raw["arrivals"]["wake_weights"]),
        scoring=Scoring(
            landing=s["landing"], fuel_bonus_per_min=s["fuel_bonus_per_min"],
            separation_violation=s["separation_violation"], go_around=s["go_around"],
            departure_timeout=s["departure_timeout"], fuel_exhaustion=s["fuel_exhaustion"],
            collision=s["collision"],
        ),
        raw=raw,
    )
