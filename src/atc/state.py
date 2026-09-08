"""World entities. Plain mutable dataclasses owned solely by the engine.

Nothing here reads a clock or an RNG — the engine drives every field.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class AcState(StrEnum):
    ENROUTE = "enroute"    # traversing an edge between fixes
    AT_FIX = "at_fix"      # parked at a fix, awaiting route or clearance
    HOLDING = "holding"    # orbiting at a fix, burning hold fuel
    FINAL = "final"        # cleared and committed: only lands or goes around
    LANDED = "landed"
    LOST = "lost"


ACTIVE_STATES = (AcState.ENROUTE, AcState.AT_FIX, AcState.HOLDING, AcState.FINAL)


@dataclass
class Aircraft:
    id: str
    callsign: str
    wake: str
    fuel_centiticks: int
    at_fix: str                      # current fix, or the fix departed if ENROUTE
    next_fix: str | None = None
    progress_centi: int = 0          # progress along the current edge, in centi-ticks
    altitude_band: int = 2
    speed: str = "normal"
    state: AcState = AcState.AT_FIX
    cleared_runway: str | None = None
    final_ticks_left: int = 0
    go_arounds: int = 0
    spawn_tick: int = 0
    pending_route: str | None = None   # fix chosen for the next leg
    hold_at: str | None = None         # orbit on reaching this fix
    pending_band: int | None = None    # climb/descend in progress
    alt_ticks_left: int = 0
    pending_clearance: str | None = None  # cleared before reaching the approach fix

    @property
    def committed(self) -> bool:
        return self.state is AcState.FINAL


@dataclass
class RunwayState:
    id: str
    occupied_until: int = 0          # tick until touchdown roll-out completes
    blocked_until: int = 0           # tick until the next arrival may touch down
    last_wake: str | None = None
    cleared_aircraft: str | None = None
    closed: bool = False


@dataclass
class Diagnostic:
    """§13 — the breakdown an LLM can actually reason about, not just a score."""
    landings: int = 0
    lost_to_fuel: int = 0
    separation_violations: int = 0
    go_arounds: int = 0
    departures_timed_out: int = 0
    collisions: int = 0
    runway_busy_ticks: int = 0
    fuel_at_touchdown_centiticks: int = 0
    detail: list[str] = field(default_factory=list)

    def note(self, msg: str, cap: int = 20) -> None:
        if len(self.detail) < cap:
            self.detail.append(msg)


@dataclass
class Rejection:
    """§10 — illegal entries are rejected individually with a reason."""
    cmd: str
    aircraft: str | None
    reason: str


@dataclass
class Command:
    at_tick: int
    cmd: str
    aircraft: str
    value: str | None = None         # speed value / altitude band / fix / runway

    def as_log(self) -> dict:
        return {"at_tick": self.at_tick, "cmd": self.cmd,
                "aircraft": self.aircraft, "value": self.value}

    @staticmethod
    def from_log(d: dict) -> "Command":
        return Command(d["at_tick"], d["cmd"], d["aircraft"], d.get("value"))
