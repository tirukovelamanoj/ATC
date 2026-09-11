"""Arcade engine — free-vector Flight Control style.

This deliberately replaces the waypoint-graph model: aircraft hold continuous
positions and headings and follow a hand-drawn polyline. That is the arcade feel
the reference game has, and it is why LLM agents can no longer compete here (see
README) — the skill being tested is mouse precision under time pressure.

Kept from the graph engine: integer tick counter, seeded RNG, and an action log,
so a run still replays. Positions are floats, so replay is exact on the same
machine but not guaranteed bit-identical across platforms.
"""
from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path

def _mod2pi(a: float) -> float:
    return a - 2*math.pi*math.floor(a/(2*math.pi))


def _dubins(x0, y0, p0, x1, y1, p1, r, step):
    """Shortest curvature-limited path between two poses, sampled every `step`.

    Standard CSC construction (Shkel & Lumelsky): four candidate word types,
    each a turn, a straight and a turn. Returns [] if none is valid.
    """
    D = math.hypot(x1 - x0, y1 - y0)
    if D < 1e-9 or r < 1e-9:
        return []
    d = D / r
    th = math.atan2(y1 - y0, x1 - x0)
    a = _mod2pi(p0 - th)
    b = _mod2pi(p1 - th)
    sa, ca, sb, cb = math.sin(a), math.cos(a), math.sin(b), math.cos(b)
    cab = math.cos(a - b)
    words = []

    # LSL
    tmp = math.atan2(cb - ca, d + sa - sb)
    q2 = 2 + d*d - 2*cab + 2*d*(sa - sb)
    if q2 >= 0:
        words.append((_mod2pi(-a + tmp), math.sqrt(q2), _mod2pi(b - tmp), "LSL"))
    # RSR
    tmp = math.atan2(ca - cb, d - sa + sb)
    q2 = 2 + d*d - 2*cab + 2*d*(sb - sa)
    if q2 >= 0:
        words.append((_mod2pi(a - tmp), math.sqrt(q2), _mod2pi(-b + tmp), "RSR"))
    # LSR
    q2 = -2 + d*d + 2*cab + 2*d*(sa + sb)
    if q2 >= 0:
        pp = math.sqrt(q2)
        tmp = math.atan2(-ca - cb, d + sa + sb) - math.atan2(-2.0, pp)
        words.append((_mod2pi(-a + tmp), pp, _mod2pi(-_mod2pi(b) + tmp), "LSR"))
    # RSL
    q2 = d*d - 2 + 2*cab - 2*d*(sa + sb)
    if q2 >= 0:
        pp = math.sqrt(q2)
        tmp = math.atan2(ca + cb, d - sa - sb) - math.atan2(2.0, pp)
        words.append((_mod2pi(a - tmp), pp, _mod2pi(b - tmp), "RSL"))
    if not words:
        return []

    t1, t2, t3, word = min(words, key=lambda w: w[0] + w[1] + w[2])
    segs = [(word[0], t1*r), (word[1], t2*r), (word[2], t3*r)]

    # Sample each segment in an integer number of EQUAL sub-steps. Using a
    # fixed step plus a leftover remainder makes the path end a fraction past
    # or short of the goal, and trimming that ragged tail is what produced a
    # kink at the handover to the drawn path.
    out, x, y, ph = [], x0, y0, p0
    for kind, length in segs:
        if length <= 1e-9:
            continue
        n = max(1, int(round(length / step)))
        ds = length / n
        for _ in range(n):
            # midpoint integration: turn half, translate, turn the other half.
            # Turning the full amount BEFORE translating biases every arc
            # outward by ~ds^2/2r per step, which over a 180-degree turn drifts
            # the endpoint by several units and wrecks the handover.
            half = ds / (2*r)
            if kind == "L":
                ph += half
            elif kind == "R":
                ph -= half
            x += math.cos(ph) * ds
            y += math.sin(ph) * ds
            if kind == "L":
                ph += half
            elif kind == "R":
                ph -= half
            out.append((x, y))
    return out


_CALLSIGNS = ("SKY", "JET", "NOR", "ATL", "VIR", "CAP", "RED", "BLU", "GLD", "ION")


def _norm(deg: float) -> float:
    """Wrap to (-180, 180]."""
    return (deg + 180.0) % 360.0 - 180.0


@dataclass
class Zone:
    id: str
    kind: str
    colour: str
    pos: tuple[float, float]
    accepts: tuple[str, ...]
    heading_deg: float = 0.0
    length: float = 0.0
    width: float = 0.0
    radius: float = 0.0

    def captures(self, x: float, y: float, heading: float, cfg: "ArcadeConfig") -> bool:
        dx, dy = x - self.pos[0], y - self.pos[1]
        if self.kind == "pad":
            return math.hypot(dx, dy) <= self.radius + cfg.capture_margin
        # rotate into runway frame: +along is the landing direction
        a = math.radians(self.heading_deg)
        along = dx * math.sin(a) - dy * math.cos(a)
        across = dx * math.cos(a) + dy * math.sin(a)
        if abs(along) > self.length / 2 or abs(across) > self.width / 2 + cfg.capture_margin:
            return False
        # runways are bidirectional (09L is also 27R), so accept an approach
        # aligned with the strip in either direction
        off = min(abs(_norm(heading - self.heading_deg)),
                  abs(_norm(heading - self.heading_deg - 180.0)))
        return off <= cfg.align_tolerance_deg


@dataclass
class AcType:
    key: str
    label: str
    speed: float
    turn_rate_deg: float
    radius: float
    zone: str


@dataclass
class ArcadeConfig:
    config_id: str
    seed: int
    tick_ms: int
    map_w: float
    map_h: float
    types: dict[str, AcType]
    zones: tuple[Zone, ...]
    align_tolerance_deg: float
    capture_margin: float
    first_delay_s: float
    interval_s: float
    interval_min_s: float
    ramp_per_landing_s: float
    max_concurrent: int
    type_weights: dict[str, int]
    warn_dist: float
    join_style: str
    speed_multiplier: float
    heading_slew_deg_s: float
    score_landing: int
    score_wrong_zone: int
    raw: dict = field(repr=False, default_factory=dict)

    @property
    def dt(self) -> float:
        return self.tick_ms / 1000.0

    def zone(self, zid: str) -> Zone | None:
        return next((z for z in self.zones if z.id == zid), None)


def load_arcade_config(src: str | Path | dict) -> ArcadeConfig:
    raw = json.loads(Path(src).read_text()) if not isinstance(src, dict) else src
    sp, sc = raw["spawn"], raw["scoring"]
    return ArcadeConfig(
        config_id=raw["config_id"], seed=raw["seed"], tick_ms=raw["tick_ms"],
        map_w=raw["map"]["w"], map_h=raw["map"]["h"],
        types={k: AcType(k, v["label"], v["speed"], v["turn_rate_deg"], v["radius"], v["zone"])
               for k, v in raw["aircraft_types"].items()},
        zones=tuple(Zone(z["id"], z["kind"], z["colour"], tuple(z["pos"]),
                         tuple(z["accepts"]), z.get("heading_deg", 0.0),
                         z.get("length", 0.0), z.get("width", 0.0), z.get("radius", 0.0))
                    for z in raw["zones"]),
        align_tolerance_deg=raw["landing"]["align_tolerance_deg"],
        capture_margin=raw["landing"]["capture_margin"],
        first_delay_s=sp["first_delay_s"], interval_s=sp["interval_s"],
        interval_min_s=sp["interval_min_s"], ramp_per_landing_s=sp["ramp_per_landing_s"],
        max_concurrent=sp["max_concurrent"], type_weights=dict(sp["type_weights"]),
        warn_dist=raw["collision"]["warn_dist"],
        join_style=raw.get("join_style", "immediate"),
        speed_multiplier=float(raw.get("speed_multiplier", 1.0)),
        heading_slew_deg_s=raw.get("heading_slew_deg_s", 540.0),
        score_landing=sc["landing"], score_wrong_zone=sc["wrong_zone"], raw=raw,
    )


@dataclass
class Aircraft:
    id: str
    callsign: str
    type: str
    x: float
    y: float
    heading: float               # degrees, 0 = north (screen up), clockwise
    path: list[tuple[float, float]] = field(default_factory=list)
    state: str = "flying"        # flying | landed | crashed
    spawn_tick: int = 0


class ArcadeEngine:
    def __init__(self, cfg: ArcadeConfig):
        self.cfg = cfg
        self.tick = 0
        self.score = 0
        self.landed = 0
        self.game_over = False
        self.conflict_ticks = 0
        self.over_reason = ""
        self.aircraft: dict[str, Aircraft] = {}
        self.events: list[dict] = []
        self.action_log: list[dict] = []
        self._rng_spawn = random.Random(cfg.seed)
        self._rng_type = random.Random(cfg.seed ^ 0xA1)
        self._rng_call = random.Random(cfg.seed ^ 0xB2)
        self._seq = 0
        self._in_zone: dict[str, str] = {}   # aircraft -> zone it is currently over
        self._next_spawn = int(cfg.first_delay_s / cfg.dt)

    @property
    def time_s(self) -> float:
        return round(self.tick * self.cfg.dt, 2)

    def _emit(self, type_: str, **kw) -> None:
        self.events.append({"t": self.time_s, "type": type_, **kw})

    # ── commands ────────────────────────────────────────────────────────────
    def _lead_in(self, ac: Aircraft, drawn: list) -> list[tuple[float, float]]:
        """A flyable join from the aircraft's nose into the start of the drawn path.

        The drawn line is followed exactly, so a stroke starting behind the
        aircraft would otherwise flip its heading 180 degrees in one tick.

        This is the shortest curvature-constrained path between two POSES
        (position + heading) — a Dubins path. Ad-hoc pursuit cannot solve it:
        when the aircraft is already level with the path start but facing the
        wrong way, no approach line is long enough, because the room to
        intercept is fixed by geometry. Dubins handles that case by construction
        and is curvature-limited throughout, so every heading change is within
        the aircraft's turn rate by definition.
        """
        t = self.cfg.types[ac.type]
        step = t.speed * self.cfg.dt
        r = t.speed / max(1e-6, math.radians(t.turn_rate_deg))     # turn radius

        bearing = math.degrees(math.atan2(drawn[0][0] - ac.x, -(drawn[0][1] - ac.y)))
        # A generous tolerance here is itself a snap: skipping the join permits
        # an instant heading change of exactly this size. Keep it below one
        # tick's worth of turn for the nimblest aircraft.
        if abs(_norm(bearing - ac.heading)) <= 5.0:
            return []                                              # already joining

        k = min(len(drawn) - 1, 6)
        dx, dy = (drawn[k][0] - drawn[0][0], drawn[k][1] - drawn[0][1]) if k > 0 else (0.0, 0.0)
        if math.hypot(dx, dy) < 1e-6:
            dx, dy = drawn[0][0] - ac.x, drawn[0][1] - ac.y
        # screen-space bearing: movement is (cos p, sin p) with y pointing down
        p0 = math.radians(ac.heading) - math.pi/2
        p1 = math.atan2(dy, dx)
        pts = _dubins(ac.x, ac.y, p0, drawn[0][0], drawn[0][1], p1, r, step)
        if not pts:
            return []
        return pts

    def set_path(self, aircraft_id: str, points: list[tuple[float, float]]) -> str | None:
        """Assign a hand-drawn path. Returns a rejection reason, or None."""
        ac = self.aircraft.get(aircraft_id)
        if ac is None or ac.state != "flying":
            return "unknown or inactive aircraft"
        if self.game_over:
            return "episode is over"
        # quantise to whole pixels so a replayed run gets identical inputs
        pts = [(round(float(x), 1), round(float(y), 1)) for x, y in points][:400]
        drawn = [p for i, p in enumerate(pts)
                 if i == 0 or math.hypot(p[0]-pts[i-1][0], p[1]-pts[i-1][1]) >= 4]
        # A stroke takes a second or more to draw and the aircraft keeps flying
        # the whole time, so the head of the stroke records where it WAS. Pick
        # the line up at whichever early point is nearest the aircraft NOW,
        # otherwise it doubles back to a stale start before setting off.
        # Only the first half is searched: a stroke that loops back near the
        # aircraft later must not be mistaken for the join point.
        # Consume only the head the aircraft has actually flown past: walk
        # forward while the stroke keeps getting closer and stop at that first
        # local minimum. Scoring entry points by total remaining travel instead
        # made it skip most of the stroke and fly a straight line to the end —
        # the drawn route has to survive, only its stale start is dropped.
        while len(drawn) > 2 and (
                math.hypot(drawn[1][0]-ac.x, drawn[1][1]-ac.y)
                < math.hypot(drawn[0][0]-ac.x, drawn[0][1]-ac.y)):
            drawn.pop(0)
        while len(drawn) > 1 and math.hypot(drawn[0][0]-ac.x, drawn[0][1]-ac.y) < 6:
            drawn.pop(0)
        if not drawn:
            return "path too short"
        # "immediate": begin traversing the stroke at once, which is what the
        # arcade feel wants. "procedure_turn" prepends a Dubins join so the
        # aircraft banks around into the line first — realistic, but it flies a
        # circuit before starting.
        ac.path = (self._lead_in(ac, drawn) + drawn
                   if self.cfg.join_style == "procedure_turn" else drawn)
        self.action_log.append({"tick": self.tick, "aircraft": aircraft_id,
                                "path": list(ac.path)})   # snapshot: ac.path is consumed in flight
        return None

    # ── spawning ────────────────────────────────────────────────────────────
    def _spawn_interval_ticks(self) -> int:
        gap = max(self.cfg.interval_min_s,
                  self.cfg.interval_s - self.landed * self.cfg.ramp_per_landing_s)
        jitter = self._rng_spawn.uniform(0.75, 1.25)
        return max(1, int(gap * jitter / self.cfg.dt))

    def _spawn(self) -> None:
        cfg = self.cfg
        keys = sorted(cfg.type_weights)
        kind = self._rng_type.choices(keys, weights=[cfg.type_weights[k] for k in keys])[0]
        edge = self._rng_spawn.randrange(4)
        m = 12.0
        if edge == 0:    x, y, h = self._rng_spawn.uniform(0, cfg.map_w), m, 180.0
        elif edge == 1:  x, y, h = cfg.map_w - m, self._rng_spawn.uniform(0, cfg.map_h), 270.0
        elif edge == 2:  x, y, h = self._rng_spawn.uniform(0, cfg.map_w), cfg.map_h - m, 0.0
        else:            x, y, h = m, self._rng_spawn.uniform(0, cfg.map_h), 90.0
        self._seq += 1
        ac = Aircraft(id=f"AC{self._seq:03d}",
                      callsign=f"{self._rng_call.choice(_CALLSIGNS)}{self._rng_call.randrange(100, 999)}",
                      type=kind, x=x, y=y, heading=h, spawn_tick=self.tick)
        self.aircraft[ac.id] = ac
        self._emit("spawn", aircraft=ac.id, callsign=ac.callsign, kind=kind)

    # ── tick ────────────────────────────────────────────────────────────────
    def step(self) -> None:
        cfg = self.cfg
        self.events.clear()
        if self.game_over:
            self.tick += 1
            return

        live = [self.aircraft[i] for i in sorted(self.aircraft)
                if self.aircraft[i].state == "flying"]

        # 1. ride the drawn path exactly, then advance
        for ac in live:
            t = cfg.types[ac.type]
            ox, oy = ac.x, ac.y
            if ac.path:
                # The drawn polyline IS the trajectory. Walk speed*dt of arc
                # length along it and sit at exactly that point. The previous
                # model steered toward each waypoint at a limited turn rate,
                # which made the aircraft bank in a circle to line itself up
                # with every point instead of following the line the player drew.
                rem = t.speed * cfg.dt
                while rem > 0 and ac.path:
                    tx, ty = ac.path[0]
                    d = math.hypot(tx - ac.x, ty - ac.y)
                    if d <= 1e-9:
                        ac.path.pop(0)
                        continue
                    if d <= rem:
                        ac.x, ac.y = tx, ty
                        rem -= d
                        ac.path.pop(0)
                    else:
                        ac.x += (tx - ac.x) / d * rem
                        ac.y += (ty - ac.y) / d * rem
                        rem = 0.0
            else:
                # no path: hold heading, but turn back rather than leave the field
                if not (0 < ac.x < cfg.map_w and 0 < ac.y < cfg.map_h):
                    want = math.degrees(math.atan2(cfg.map_w/2 - ac.x, -(cfg.map_h/2 - ac.y)))
                    delta = _norm(want - ac.heading)
                    turn = t.turn_rate_deg * cfg.dt
                    ac.heading = _norm(ac.heading + max(-turn, min(turn, delta)))
                a = math.radians(ac.heading)
                ac.x += math.sin(a) * t.speed * cfg.dt
                ac.y -= math.cos(a) * t.speed * cfg.dt
            # heading always follows actual travel, so the sprite points along
            # the drawn line rather than at some waypoint it is chasing
            dx, dy = ac.x - ox, ac.y - oy
            if dx*dx + dy*dy > 1e-12:
                want = math.degrees(math.atan2(dx, -dy))
                # Snapping straight to the travel direction is what made a
                # backwards stroke flip the aircraft in a single frame. Position
                # still follows the drawn line exactly; only the sprite swings,
                # fast enough to keep up but never instantly.
                slew = cfg.heading_slew_deg_s * cfg.dt
                ac.heading = _norm(ac.heading + max(-slew, min(slew, _norm(want - ac.heading))))

        # 2. landings. A wrong-zone overflight is charged ONCE on entry, not on
        # every tick the aircraft happens to be inside the box — at 20Hz that
        # billed -1000/second and swamped the score.
        for ac in live:
            t = cfg.types[ac.type]
            over = next((z for z in cfg.zones
                         if z.captures(ac.x, ac.y, ac.heading, cfg)), None)
            was = self._in_zone.get(ac.id)
            self._in_zone[ac.id] = over.id if over else ""
            if over is None:
                continue
            if ac.type in over.accepts:
                ac.state, ac.path = "landed", []
                self._in_zone.pop(ac.id, None)
                self.landed += 1
                self.score += cfg.score_landing
                self._emit("landed", aircraft=ac.id, callsign=ac.callsign, zone=over.id)
            elif was != over.id:
                self.score += cfg.score_wrong_zone
                self._emit("wrong_zone", aircraft=ac.id, callsign=ac.callsign,
                           zone=over.id, needs=t.zone)

        # 3. collisions — the loss condition
        live = [a for a in live if a.state == "flying"]
        for i in range(len(live)):
            for j in range(i + 1, len(live)):
                a, b = live[i], live[j]
                lim = cfg.types[a.type].radius + cfg.types[b.type].radius
                d = math.hypot(a.x - b.x, a.y - b.y)
                # Near-miss accounting, computed in the loop we already run so it
                # costs nothing. A crash gives one -1 at the very end, far too
                # sparse to teach avoidance; this exposes the danger continuously.
                if d <= cfg.warn_dist:
                    self.conflict_ticks += 1
                if d <= lim:
                    a.state = b.state = "crashed"
                    self.game_over = True
                    self.over_reason = f"{a.callsign} and {b.callsign} collided"
                    self._emit("crash", aircraft=a.id, other=b.id,
                               callsigns=[a.callsign, b.callsign])
                    break
            if self.game_over:
                break

        # 4. spawn
        if not self.game_over and self.tick >= self._next_spawn and len(
                [a for a in self.aircraft.values() if a.state == "flying"]) < cfg.max_concurrent:
            self._spawn()
            self._next_spawn = self.tick + self._spawn_interval_ticks()

        self.tick += 1

    def observation(self) -> dict:
        cfg = self.cfg
        live = [self.aircraft[i] for i in sorted(self.aircraft)
                if self.aircraft[i].state == "flying"]
        warn = set()
        for i in range(len(live)):
            for j in range(i + 1, len(live)):
                a, b = live[i], live[j]
                if math.hypot(a.x - b.x, a.y - b.y) <= cfg.warn_dist:
                    warn.add(a.id); warn.add(b.id)
        return {
            "tick": self.tick, "time_s": self.time_s,
            "score": self.score, "landed": self.landed,
            "game_over": self.game_over, "over_reason": self.over_reason,
            "aircraft": [{
                "id": a.id, "callsign": a.callsign, "type": a.type,
                "x": round(a.x, 1), "y": round(a.y, 1), "heading": round(a.heading, 1),
                "path": a.path, "conflict": a.id in warn,
                "zone": cfg.types[a.type].zone,
            } for a in live],
            "events_since_last": list(self.events),
        }
