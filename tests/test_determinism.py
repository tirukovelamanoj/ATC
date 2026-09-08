"""§8 determinism: re-simulating (config, seed, action_log) reproduces a run exactly.

This is the load-bearing test of the whole project. Replay, verification and
debugging are all free consequences of it, and League 3/4 scores are worthless
without it. Run directly: `.venv/bin/python tests/test_determinism.py`
"""
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from atc.bots import GreedyBot, play
from atc.config import load_config
from atc.engine import Engine
from atc.observation import to_diagnostic, to_observation
from atc.state import ACTIVE_STATES

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "m1_single_runway.json"


def state_hash(e: Engine) -> str:
    """Fingerprint every field that a divergence could show up in."""
    parts = [f"{e.tick}|{e.score}"]
    for i in sorted(e.aircraft):
        a = e.aircraft[i]
        parts.append(f"{a.id},{a.state},{a.at_fix},{a.next_fix},{a.progress_centi},"
                     f"{a.fuel_centiticks},{a.altitude_band},{a.speed},{a.final_ticks_left}")
    for k in sorted(e.runways):
        r = e.runways[k]
        parts.append(f"{r.id},{r.occupied_until},{r.blocked_until},{r.last_wake},{r.cleared_aircraft}")
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


def record() -> tuple[Engine, list, list[str]]:
    e = Engine(load_config(CONFIG))
    bot, trace = GreedyBot(), []
    while not e.done:
        e.submit(bot.act(to_observation(e)))
        e.step()
        trace.append(state_hash(e))
    return e, [c.as_log() for c in e.action_log], trace


def replay(action_log: list) -> tuple[Engine, list[str]]:
    from atc.state import Command
    e = Engine(load_config(CONFIG))
    e.submit([Command.from_log(d) for d in action_log])
    trace = []
    while not e.done:
        e.step()
        trace.append(state_hash(e))
    return e, trace


def main() -> int:
    live, log, live_trace = record()
    rerun, rerun_trace = replay(log)

    # 1. tick-by-tick identity — pinpoints the first divergent tick, not just a bad total
    assert len(live_trace) == len(rerun_trace), "trace lengths differ"
    for t, (a, b) in enumerate(zip(live_trace, rerun_trace)):
        assert a == b, f"diverged at tick {t}: {a} != {b}"

    # 2. observable outcomes identical
    assert live.score == rerun.score, f"score {live.score} != {rerun.score}"
    assert to_diagnostic(live) == to_diagnostic(rerun), "diagnostic differs"

    # 3. a different seed must actually produce a different run (guards against
    #    a trivially-passing test where the sim ignores the RNG entirely)
    cfg2 = load_config(CONFIG)
    other = Engine(load_config({**cfg2.raw, "seed": cfg2.seed + 1}))
    play(other, GreedyBot())
    assert state_hash(other) != state_hash(live), "different seeds produced identical runs"

    d = to_diagnostic(live)
    print(f"OK  determinism holds over {len(live_trace)} ticks, {len(log)} logged commands")
    print(f"    score={d['score']} landings={d['landings']} "
          f"sep={d['separation_violations']} lost={d['lost_to_fuel']} "
          f"idle={d['runway_idle_pct']}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
