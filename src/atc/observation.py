"""Projections of engine state — §9 observation and §13 diagnostic.

The hard invariant from §1: this JSON and the pixels the human sees are two
projections of the same state. Nothing may appear here that the human has no
path to, and nothing may be rendered that is absent here.
"""
from __future__ import annotations

from atc.config import CENTITICKS_PER_MIN, EpisodeConfig
from atc.state import ACTIVE_STATES, AcState, Aircraft


def _fuel_min(centiticks: int) -> float:
    return round(max(0, centiticks) / CENTITICKS_PER_MIN, 1)


def _eta_next_fix_s(ac: Aircraft, cfg: EpisodeConfig) -> int | None:
    if ac.state is not AcState.ENROUTE or ac.next_fix is None:
        return None
    edge = cfg.airspace.edge(ac.at_fix, ac.next_fix)
    if edge is None:
        return None
    remaining_centi = edge.traverse_ticks * 100 - ac.progress_centi
    per_tick = cfg.speed_factor_pct[ac.speed]
    return int(round(remaining_centi / per_tick / 10))


def _eta_touchdown_s(ac: Aircraft, cfg: EpisodeConfig) -> int | None:
    """Sum of remaining legs along the default route to the runway."""
    if ac.state is AcState.FINAL:
        return int(round(ac.final_ticks_left / 10))
    leg = _eta_next_fix_s(ac, cfg)
    fix = ac.next_fix if ac.state is AcState.ENROUTE else ac.at_fix
    total = leg or 0
    seen = set()
    while fix and fix != cfg.airspace.approach_fix and fix not in seen:
        seen.add(fix)
        edges = cfg.airspace.edges.get(fix, ())
        if not edges:
            return None
        total += int(round(edges[0].traverse_ticks / cfg.speed_factor_pct[ac.speed] * 100 / 10))
        fix = edges[0].to
    if fix == cfg.airspace.approach_fix:
        total += int(round(cfg.runways[0].final_approach_ticks / 10))
    return total


def to_observation(engine) -> dict:
    cfg = engine.cfg
    aircraft = []
    for ac in (engine.aircraft[i] for i in sorted(engine.aircraft)):
        if ac.state not in ACTIVE_STATES:
            continue
        aircraft.append({
            "id": ac.id,
            "callsign": ac.callsign,
            "type": "arrival",
            "wake": ac.wake,
            "fuel_min": _fuel_min(ac.fuel_centiticks),
            "low_fuel": ac.fuel_centiticks < cfg.fuel.low_fuel_threshold_centiticks,
            "at_fix": ac.at_fix,
            "next_fix": ac.next_fix,
            "eta_next_fix_s": _eta_next_fix_s(ac, cfg),
            "altitude_band": ac.altitude_band,
            "changing_to_band": ac.pending_band,
            "speed": ac.speed,
            "state": str(ac.state),
            "committed": ac.committed,
            "cleared_for": ac.cleared_runway or ac.pending_clearance,
            "eta_touchdown_s": _eta_touchdown_s(ac, cfg),
        })
    runways = [{
        "id": rs.id,
        "occupied_until_s": round(max(0, rs.occupied_until - engine.tick) / 10, 1),
        "blocked_until_s": round(max(0, rs.blocked_until - engine.tick) / 10, 1),
        "last_wake": rs.last_wake,
        "cleared_aircraft": rs.cleared_aircraft,
        "closed": rs.closed,
    } for rs in (engine.runways[k] for k in sorted(engine.runways))]

    return {
        "tick": engine.tick,
        "game_time_s": engine.game_time_s,
        "score": engine.score,
        "aircraft": aircraft,
        "runways": runways,
        "departures_waiting": [],
        "weather": [],
        "events_since_last": list(engine.events),
    }


def to_diagnostic(engine) -> dict:
    """§13 — a breakdown the LLM can reason about, not a bare score."""
    d, cfg = engine.diag, engine.cfg
    elapsed = max(1, engine.tick)
    avg_fuel = (d.fuel_at_touchdown_centiticks / d.landings / CENTITICKS_PER_MIN) if d.landings else 0.0
    return {
        "score": engine.score,
        "landings": d.landings,
        "lost_to_fuel": d.lost_to_fuel,
        "separation_violations": d.separation_violations,
        "go_arounds": d.go_arounds,
        "departures_timed_out": d.departures_timed_out,
        "runway_idle_pct": round(100 * (1 - d.runway_busy_ticks / (elapsed * len(engine.runways))), 1),
        "avg_fuel_at_touchdown_min": round(avg_fuel, 1),
        "detail": list(d.detail),
    }
