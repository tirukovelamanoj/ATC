"""The deterministic core — §8.

Pure function of (config, seed, action_log). No wall-clock reads, no I/O, no
rendering. Every other surface (REST, WebSocket, React, Gymnasium) is a shell
over this.

Determinism rules observed here:
  * time is an integer tick counter; nothing accumulates in float
  * fuel and edge progress are integer centi-ticks
  * aircraft are always iterated in sorted id order, never dict/set order
  * each stochastic concern owns a separate RNG stream, so adding weather
    later cannot shift the arrival sequence of an existing seed
"""
from __future__ import annotations

import math
import random

from atc.config import CENTITICKS_PER_MIN, EpisodeConfig
from atc.state import (ACTIVE_STATES, AcState, Aircraft, Command, Diagnostic,
                       Rejection, RunwayState)

SPEEDS = ("slow", "normal", "fast")
_CALLSIGN_PREFIXES = ("SKY", "JET", "NOR", "ATL", "VIR", "CAP", "RED", "BLU")


class Engine:
    def __init__(self, config: EpisodeConfig):
        self.cfg = config
        self.tick = 0
        self.score = 0
        self.diag = Diagnostic()
        self.aircraft: dict[str, Aircraft] = {}
        self.runways: dict[str, RunwayState] = {r.id: RunwayState(r.id) for r in config.runways}
        self.events: list[dict] = []
        self.action_log: list[Command] = []

        # Separate streams per concern — see module docstring.
        self._rng_arrivals = random.Random(config.seed)
        self._rng_wake = random.Random(config.seed ^ 0x5EED_0001)
        self._rng_fuel = random.Random(config.seed ^ 0x5EED_0002)
        self._rng_callsign = random.Random(config.seed ^ 0x5EED_0003)

        self._pending: list[Command] = []
        self._seq = 0
        self._active_conflicts: set[tuple[str, str]] = set()
        self._next_spawn_tick = self._draw_spawn_gap()
        self._seed_initial_traffic(config.initial_aircraft)

    # ── time helpers ────────────────────────────────────────────────────────
    @property
    def game_time_s(self) -> float:
        return round(self.tick / 10, 1)

    @property
    def done(self) -> bool:
        return self.tick >= self.cfg.episode_length_ticks

    def _emit(self, type_: str, **kw) -> None:
        self.events.append({"t": self.game_time_s, "type": type_, **kw})

    # ── command intake — §10 ────────────────────────────────────────────────
    def submit(self, commands: list[Command]) -> tuple[list[Command], list[Rejection]]:
        """Queue a timed plan. Entries are validated at execution, not here;
        only structurally impossible ones are refused up front."""
        accepted, rejected = [], []
        for c in commands:
            if c.at_tick < self.tick:
                rejected.append(Rejection(c.cmd, c.aircraft, f"stale: at_tick {c.at_tick} < now {self.tick}"))
            else:
                self._pending.append(c)
                accepted.append(c)
        self._pending.sort(key=lambda c: (c.at_tick, c.aircraft, c.cmd))
        return accepted, rejected

    def _apply(self, c: Command) -> Rejection | None:
        cfg, ac = self.cfg, self.aircraft.get(c.aircraft)
        if ac is None or ac.state not in ACTIVE_STATES:
            return Rejection(c.cmd, c.aircraft, "unknown or inactive aircraft")
        if ac.committed and c.cmd != "go_around":
            return Rejection(c.cmd, c.aircraft, "committed to final; only go_around is legal")

        match c.cmd:
            case "speed":
                if c.value not in SPEEDS:
                    return Rejection(c.cmd, c.aircraft, f"speed must be one of {SPEEDS}")
                ac.speed = c.value

            case "altitude":
                band = int(c.value)
                if not 1 <= band <= cfg.airspace.altitude_bands:
                    return Rejection(c.cmd, c.aircraft, f"band out of range 1..{cfg.airspace.altitude_bands}")
                if band == ac.altitude_band and ac.pending_band is None:
                    return Rejection(c.cmd, c.aircraft, "already at that band")
                ac.pending_band = band
                ac.alt_ticks_left = cfg.airspace.altitude_change_ticks

            case "route":
                base = ac.next_fix if ac.state is AcState.ENROUTE else ac.at_fix
                if cfg.airspace.edge(base, c.value) is None:
                    return Rejection(c.cmd, c.aircraft, f"{c.value} is not adjacent to {base}")
                ac.pending_route = c.value
                ac.hold_at = None
                if ac.state is AcState.HOLDING:
                    self._depart_fix(ac)

            case "hold":
                if c.value not in cfg.airspace.edges:
                    return Rejection(c.cmd, c.aircraft, f"unknown fix {c.value}")
                reachable = ac.next_fix if ac.state is AcState.ENROUTE else ac.at_fix
                if c.value != reachable:
                    return Rejection(c.cmd, c.aircraft,
                                     f"can only hold at {reachable}, not {c.value}")
                ac.hold_at = c.value
                if ac.state in (AcState.AT_FIX, AcState.HOLDING) and ac.at_fix == c.value:
                    ac.state = AcState.HOLDING

            case "clear_land":
                rw = cfg.runway(c.value)
                if rw is None:
                    return Rejection(c.cmd, c.aircraft, f"unknown runway {c.value}")
                rs = self.runways[rw.id]
                if rs.closed:
                    return Rejection(c.cmd, c.aircraft, f"runway {rw.id} is closed")
                at_approach = (ac.at_fix == cfg.airspace.approach_fix
                               and ac.state in (AcState.AT_FIX, AcState.HOLDING))
                if not at_approach:
                    # pre-clearance: honoured the moment it reaches the approach fix
                    if not self._heads_for_approach(ac):
                        return Rejection(c.cmd, c.aircraft, "aircraft is not routed to the approach")
                    ac.pending_clearance, ac.hold_at = rw.id, None
                    self._emit("cleared_early", aircraft=ac.id, runway=rw.id)
                    return None
                if not self._enter_final(ac, rw):
                    short = (self._free_at(rw, ac) - (self.tick + rw.final_approach_ticks)) / 10
                    return Rejection(c.cmd, c.aircraft,
                                     f"runway {rw.id} not free in time (short by {short:.1f}s)")

            case "go_around":
                if ac.state is not AcState.FINAL:
                    return Rejection(c.cmd, c.aircraft, "aircraft is not on final")
                self._go_around(ac, "commanded")

            case _:
                return Rejection(c.cmd, c.aircraft, f"unknown command {c.cmd}")
        return None

    # ── clearance helpers ───────────────────────────────────────────────────
    def _free_at(self, rw, ac: Aircraft) -> int:
        rs = self.runways[rw.id]
        return rs.blocked_until + self.cfg.wake_separation_ticks(rs.last_wake or "light", ac.wake)

    def _heads_for_approach(self, ac: Aircraft) -> bool:
        """Does the default route from here still reach the approach fix?"""
        fix = ac.next_fix if ac.state is AcState.ENROUTE else ac.at_fix
        seen = set()
        while fix and fix not in seen:
            if fix == self.cfg.airspace.approach_fix:
                return True
            seen.add(fix)
            edges = self.cfg.airspace.edges.get(fix, ())
            fix = edges[0].to if edges else None
        return False

    def _enter_final(self, ac: Aircraft, rw) -> bool:
        """Commit to final if the runway will be free on arrival."""
        if self.tick + rw.final_approach_ticks < self._free_at(rw, ac):
            return False
        ac.state, ac.cleared_runway = AcState.FINAL, rw.id
        ac.final_ticks_left, ac.hold_at, ac.pending_clearance = rw.final_approach_ticks, None, None
        self.runways[rw.id].cleared_aircraft = ac.id
        return True

    # ── transitions ─────────────────────────────────────────────────────────
    def _default_next(self, fix: str) -> str | None:
        edges = self.cfg.airspace.edges.get(fix, ())
        return edges[0].to if edges else None

    def _depart_fix(self, ac: Aircraft) -> None:
        """Send an aircraft down its next edge, honouring a pending route."""
        target = ac.pending_route or self._default_next(ac.at_fix)
        ac.pending_route = None
        if target is None or self.cfg.airspace.edge(ac.at_fix, target) is None:
            ac.state = AcState.AT_FIX
            return
        ac.next_fix, ac.progress_centi, ac.state = target, 0, AcState.ENROUTE

    def _go_around(self, ac: Aircraft, reason: str) -> None:
        self.score += self.cfg.scoring.go_around
        self.diag.go_arounds += 1
        ac.go_arounds += 1
        if ac.cleared_runway:
            rs = self.runways[ac.cleared_runway]
            if rs.cleared_aircraft == ac.id:
                rs.cleared_aircraft = None
        ac.cleared_runway, ac.final_ticks_left = None, 0
        # Re-enter the pattern one fix upstream of the approach.
        upstream = next((f for f, es in sorted(self.cfg.airspace.edges.items())
                         if any(e.to == self.cfg.airspace.approach_fix for e in es)), ac.at_fix)
        ac.at_fix, ac.state = upstream, AcState.AT_FIX
        ac.progress_centi, ac.next_fix = 0, None
        self._emit("go_around", aircraft=ac.id, reason=reason)
        self.diag.note(f"go_around {ac.id} at t={self.game_time_s} ({reason})")

    def _lose(self, ac: Aircraft, reason: str) -> None:
        ac.state = AcState.LOST
        if ac.cleared_runway:
            rs = self.runways[ac.cleared_runway]
            if rs.cleared_aircraft == ac.id:
                rs.cleared_aircraft = None
        self.score += self.cfg.scoring.fuel_exhaustion
        self.diag.lost_to_fuel += 1
        self._emit("lost", aircraft=ac.id, reason=reason)
        self.diag.note(f"{ac.id} lost to fuel at {ac.at_fix} t={self.game_time_s}")

    # ── spawning ────────────────────────────────────────────────────────────
    def _draw_spawn_gap(self) -> int:
        u = self._rng_arrivals.random()
        minutes = -math.log(1.0 - u) / self.cfg.arrival_rate_per_min
        return max(1, int(round(minutes * 60 * 10)))

    def _spawn(self) -> None:
        cfg = self.cfg
        self._seq += 1
        wakes = sorted(cfg.wake_weights)
        wake = self._rng_wake.choices(wakes, weights=[cfg.wake_weights[w] for w in wakes])[0]
        jitter = self._rng_fuel.randint(-cfg.fuel.spawn_fuel_jitter_centiticks,
                                        cfg.fuel.spawn_fuel_jitter_centiticks)
        entry = cfg.airspace.entry_fixes[self._rng_arrivals.randrange(len(cfg.airspace.entry_fixes))]
        ac = Aircraft(
            id=f"AC{self._seq:03d}",
            callsign=f"{self._rng_callsign.choice(_CALLSIGN_PREFIXES)}{self._rng_callsign.randrange(100, 999)}",
            wake=wake,
            fuel_centiticks=cfg.fuel.spawn_fuel_centiticks + jitter,
            at_fix=entry,
            altitude_band=cfg.airspace.altitude_bands,
            spawn_tick=self.tick,
        )
        self.aircraft[ac.id] = ac
        self._emit("spawn", aircraft=ac.id, callsign=ac.callsign, wake=wake, fix=entry)

    def _seed_initial_traffic(self, n: int) -> None:
        """Start the episode mid-stream.

        An empty ladder is a terrible opening, and it is not a rare case: the
        inter-arrival gap is exponential, so at 0.9/min roughly 7% of episodes
        would legitimately show nothing for three minutes. Pre-place a few
        arrivals, staggered down the approach rather than clumped at the entry.
        """
        for i in range(n):
            self._spawn()
            ac = self.aircraft[f"AC{self._seq:03d}"]
            self._depart_fix(ac)
            edge = self.cfg.airspace.edge(ac.at_fix, ac.next_fix) if ac.next_fix else None
            if edge is not None:
                frac = (n - i) / (n + 1)
                ac.progress_centi = min(edge.traverse_ticks * 100 - 1,
                                        int(edge.traverse_ticks * 100 * frac))

    # ── the tick loop ───────────────────────────────────────────────────────
    def step(self) -> None:
        cfg = self.cfg
        self.events.clear()

        # 1. drain the command queue
        while self._pending and self._pending[0].at_tick <= self.tick:
            c = self._pending.pop(0)
            self.action_log.append(c)
            if (rej := self._apply(c)) is not None:
                self._emit("rejected", cmd=rej.cmd, aircraft=rej.aircraft, reason=rej.reason)

        active = [self.aircraft[i] for i in sorted(self.aircraft)
                  if self.aircraft[i].state in ACTIVE_STATES]

        # 2. advance along edges + altitude transitions
        for ac in active:
            if ac.alt_ticks_left > 0:
                ac.alt_ticks_left -= 1
                if ac.alt_ticks_left == 0 and ac.pending_band is not None:
                    ac.altitude_band, ac.pending_band = ac.pending_band, None
            if ac.state is AcState.ENROUTE:
                ac.progress_centi += cfg.speed_factor_pct[ac.speed]
            elif ac.state is AcState.FINAL:
                ac.final_ticks_left -= 1

        # 3. burn fuel
        for ac in active:
            mode = "hold" if ac.state in (AcState.HOLDING, AcState.AT_FIX) else ac.speed
            ac.fuel_centiticks -= cfg.fuel.burn_pct[mode]

        # 4. runway timers (nothing to decrement — absolute ticks — but track idleness)
        for rs in self.runways.values():
            if rs.occupied_until > self.tick:
                self.diag.runway_busy_ticks += 1

        # 5. resolve arrivals at fixes / approach handling
        for ac in active:
            if ac.state is not AcState.ENROUTE or ac.next_fix is None:
                continue
            edge = cfg.airspace.edge(ac.at_fix, ac.next_fix)
            if edge is not None and ac.progress_centi >= edge.traverse_ticks * 100:
                ac.at_fix, ac.next_fix, ac.progress_centi = ac.next_fix, None, 0
                ac.state = AcState.AT_FIX
                self._emit("at_fix", aircraft=ac.id, fix=ac.at_fix)
                if ac.at_fix == cfg.airspace.approach_fix:
                    ac.state = AcState.HOLDING          # awaits clear_land
                    if ac.pending_clearance:
                        rw = cfg.runway(ac.pending_clearance)
                        if rw and not self._enter_final(ac, rw):
                            self._emit("clearance_held", aircraft=ac.id, runway=rw.id,
                                       reason="runway not free on arrival")
                elif ac.hold_at == ac.at_fix:
                    ac.state = AcState.HOLDING
                else:
                    self._depart_fix(ac)

        # Anyone parked at a fix and free to move on: newly spawned arrivals,
        # aircraft re-entering the pattern after a go-around, holds released.
        for ac in active:
            if ac.state is AcState.AT_FIX and ac.at_fix != cfg.airspace.approach_fix:
                if ac.hold_at == ac.at_fix:
                    ac.state = AcState.HOLDING
                else:
                    self._depart_fix(ac)

        # 6. touchdowns and fuel exhaustion
        for ac in active:
            if ac.state is AcState.FINAL and ac.final_ticks_left <= 0:
                rw, rs = cfg.runway(ac.cleared_runway), self.runways[ac.cleared_runway]
                ac.state = AcState.LANDED
                rs.occupied_until = self.tick + rw.landing_occupancy_ticks
                rs.blocked_until = rs.occupied_until + rw.trailing_block_ticks
                rs.last_wake, rs.cleared_aircraft = ac.wake, None
                fuel_min = max(0, ac.fuel_centiticks) / CENTITICKS_PER_MIN
                bonus = (max(0, ac.fuel_centiticks) * cfg.scoring.fuel_bonus_per_min) // CENTITICKS_PER_MIN
                self.score += cfg.scoring.landing + bonus
                self.diag.landings += 1
                self.diag.fuel_at_touchdown_centiticks += max(0, ac.fuel_centiticks)
                self._emit("landed", aircraft=ac.id, runway=rs.id,
                           fuel_remaining_min=round(fuel_min, 1))
            elif ac.fuel_centiticks <= 0:
                self._lose(ac, "fuel exhaustion")

        # 7. separation — same fix, same altitude band, both stationary
        parked: dict[tuple[str, int], list[str]] = {}
        for ac in active:
            if ac.state in (AcState.AT_FIX, AcState.HOLDING):
                parked.setdefault((ac.at_fix, ac.altitude_band), []).append(ac.id)
        now_conflicting: set[tuple[str, str]] = set()
        for (fix, band), ids in sorted(parked.items()):
            if len(ids) < 2:
                continue
            for i in range(len(ids)):
                for j in range(i + 1, len(ids)):
                    pair = (ids[i], ids[j])
                    now_conflicting.add(pair)
                    if pair not in self._active_conflicts:
                        self.score += cfg.scoring.separation_violation
                        self.diag.separation_violations += 1
                        self._emit("separation", aircraft=ids[i], other=ids[j], fix=fix, band=band)
                        self.diag.note(f"separation {ids[i]}/{ids[j]} at {fix} band {band} t={self.game_time_s}")
        self._active_conflicts = now_conflicting

        # 8. spawn
        if (self.tick >= self._next_spawn_tick
                and len([a for a in self.aircraft.values() if a.state in ACTIVE_STATES])
                < cfg.max_concurrent_aircraft):
            self._spawn()
            self._next_spawn_tick = self.tick + self._draw_spawn_gap()

        self.tick += 1

    def run(self) -> None:
        while not self.done:
            self.step()
