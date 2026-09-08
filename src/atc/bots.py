"""Scripted baseline controllers — §14 wants one on day one.

The greedy bot is the depth oracle for milestone 1: if a first-come-first-served
controller scores near the ceiling, the game is not deep enough yet and
departures need to come forward.

Bots read ONLY the observation JSON (§9). That is deliberate — it continuously
proves the agent-facing projection is sufficient to actually play the game,
which is the §1 parity invariant enforced by construction rather than by review.
"""
from __future__ import annotations

from atc.state import Command


class GreedyBot:
    """Clear the hungriest aircraft whenever the runway will take it, and keep
    the approach stack on distinct altitude bands to avoid separation loss."""

    def __init__(self, altitude_bands: int = 3, runway: str = "09L"):
        self.bands = altitude_bands
        self.runway = runway
        self._assigned: dict[str, int] = {}

    def act(self, obs: dict) -> list[Command]:
        tick = obs["tick"]
        cmds: list[Command] = []
        rw = next((r for r in obs["runways"] if r["id"] == self.runway), None)
        if rw is None:
            return cmds

        holding = [a for a in obs["aircraft"]
                   if a["state"] == "holding" and a["at_fix"].startswith("APPR")]

        # Spread the approach stack across altitude bands, lowest fuel lowest band.
        for slot, ac in enumerate(sorted(holding, key=lambda a: a["fuel_min"])):
            want = (slot % self.bands) + 1
            if ac["altitude_band"] != want and ac["changing_to_band"] != want:
                if self._assigned.get(ac["id"]) != want:
                    self._assigned[ac["id"]] = want
                    cmds.append(Command(tick, "altitude", ac["id"], str(want)))

        # Clear the hungriest aircraft the moment the runway can accept it.
        if rw["blocked_until_s"] <= 0.0 and not rw["cleared_aircraft"] and holding:
            target = min(holding, key=lambda a: a["fuel_min"])
            cmds.append(Command(tick, "clear_land", target["id"], self.runway))
        return cmds


def play(engine, bot) -> None:
    """Drive an episode to completion with a scripted controller."""
    from atc.observation import to_observation
    while not engine.done:
        engine.submit(bot.act(to_observation(engine)))
        engine.step()


class LookaheadBot(GreedyBot):
    """Plays the two skills GreedyBot ignores, and exists to prove they matter.

    1. Clears with LEAD TIME — an aircraft cleared now touches down
       final_approach_s later, so clearing while the runway is still blocked
       (but will be free on arrival) reclaims that dead time every cycle.
    2. Spreads the approach stack by ROUTE, not just altitude — once the stack
       is full, send the next arrival the long way round instead of parking it
       on top of someone.
    """

    def __init__(self, final_approach_s: float = 45.0, long_route: str = "DELTA", **kw):
        super().__init__(**kw)
        self.final_s = final_approach_s
        self.long_route = long_route

    def act(self, obs):
        tick = obs["tick"]
        cmds = super().act(obs)
        cmds = [c for c in cmds if c.cmd != "clear_land"]      # re-decide clearance
        rw = next((r for r in obs["runways"] if r["id"] == self.runway), None)
        if rw is None:
            return cmds

        holding = [a for a in obs["aircraft"]
                   if a["state"] == "holding" and a["at_fix"].startswith("APPR")]

        # 1. clear as soon as the aircraft would ARRIVE after the block clears
        if not rw["cleared_aircraft"] and holding and rw["blocked_until_s"] <= self.final_s:
            target = min(holding, key=lambda a: a["fuel_min"])
            cmds.append(Command(tick, "clear_land", target["id"], self.runway))

        # 2. stack is full — divert the next inbound the long way instead
        if len(holding) >= self.bands:
            for a in obs["aircraft"]:
                if a["state"] == "enroute" and a["next_fix"] == self.long_route:
                    continue
                if a["state"] == "at_fix" and self.long_route in (a["at_fix"], a["next_fix"] or ""):
                    continue
                if a["state"] == "enroute" and a["next_fix"] == "KILO" and a["at_fix"] == "ALPHA":
                    cmds.append(Command(tick, "route", a["id"], self.long_route))
                    break
        return cmds
