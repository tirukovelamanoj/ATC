# ATC Arena — Project Specification

A real-time air traffic control game playable by **humans** and **LLM agents** on
identical state, designed so that each side's cognitive strengths and weaknesses
are tested rather than papered over.

---

## 1. Core thesis

| Player | Strength | Weakness |
|---|---|---|
| Human | ~250ms reaction, pre-attentive perception, improvisation | tunnel vision, forgets, fatigues |
| LLM | absorbs full structured state, never forgets, reasons over horizon | 1–3s latency, token cost |

The game must be tunable along **two independent dials**:

- **Dial A — information volume** (aircraft count, rule complexity, weather). Crank it and the *human* drowns.
- **Dial B — time pressure** (world speed multiplier, arrival rate). Crank it and the *LLM* drowns.

Balance is the coordinate where both plateau at a similar score. Find it
empirically, not by argument.

**Hard invariant:** the pixels shown to the human and the JSON sent to the agent
are projections of *identical* state. Never render something that is absent from
the JSON, and never expose a JSON field that the human has no path to. The moment
this breaks, cross-league scores become meaningless.

The split is by **depth, not availability**:

| Information | Human | Agent |
|---|---|---|
| Position on ladder, fuel band, runway blocks | glanceable, free | in JSON |
| Exact fuel minutes, wake class, hold cost | one click, ~1s | in JSON, free |
| Full pairwise conflict projection | not rendered | derivable from JSON |

Everything is *available* to the human; some of it costs seconds they don't have.

---

## 2. Build order

Each stage is independently shippable. Do not start stage N+1 until stage N runs.

1. **Headless deterministic core** — pure Python, seeded RNG, no I/O, no rendering.
2. **Playable single-player web game** — must be genuinely fun on its own.
3. **HTTP/WebSocket endpoints** for live agents (League 2).
4. **Synthesis league** — agents submit controllers (League 3).
5. **Weight-tuning league** — LLM-in-the-loop parameter optimization (League 4).

> The core must be a pure function of `(config, seed, action_log)`. Every other
> surface — React UI, REST API, Gym adapter — is a thin shell over it. If game
> logic leaks into the render loop, everything downstream has to be rewritten.

---

## 3. Airspace model

**Waypoint graph, not free vectors.** Aircraft travel between named fixes on a
directed graph. Do *not* implement continuous heading control — it turns the game
into a mouse-precision test, which kills the LLM and isn't the interesting skill.

- Fixes: `ALPHA`, `BRAVO`, `KILO`, `DELTA`, `ECHO`, … each with edges and traversal times.
- Altitude bands: 3 discrete levels. Aircraft at different bands never conflict.
- Approach: a fix adjacent to a runway threshold. Entering it commits the aircraft to final.
- Departures spawn on the ground at a gate with a timeout clock.

### Commands (exactly six)

| Command | Args | Effect |
|---|---|---|
| `route` | aircraft_id, fix_id | set next fix (must be graph-adjacent) |
| `altitude` | aircraft_id, band | climb/descend, takes time |
| `speed` | aircraft_id, `slow`\|`normal`\|`fast` | changes ETA and fuel burn |
| `hold` | aircraft_id, fix_id | orbit at fix, burns fuel |
| `clear_land` | aircraft_id, runway_id | authorize final approach |
| `go_around` | aircraft_id | abandon approach, re-enter pattern |

Departures use `clear_takeoff` (runway_id) — treat as a seventh command or a
variant of `clear_land`; be consistent in the schema.

---

## 4. Fuel

Fuel is denominated in **minutes**, not percent, so it is directly comparable to
travel time by both human and agent.

| State | Burn multiplier |
|---|---|
| `slow` | 0.7× |
| `normal` | 1.0× |
| `fast` | 2.0× |
| `hold` | 1.5× |

- Below **8 minutes**: aircraft declares low fuel, flagged in JSON and rendered urgently.
- At **0**: aircraft lost (see scoring).

Fuel is what converts every decision into a deadline. It is the reason "hold
everything and think" is not a winning strategy.

---

## 5. Runways

A runway is a **mutex with occupancy time plus trailing separation**.

| Event | Runway occupied | Blocked after |
|---|---|---|
| Landing | 50s (touchdown → exit) | 40s |
| Takeoff | 35s (roll → airborne) | 60s if heavy departs |
| Arrival behind heavy | — | +40s wake separation |

So a single landing costs ~90s of runway. Three back-to-back arrivals = 4.5
minutes with zero departures launched, while a departure that has waited 8
minutes is timing out.

**Two runways must be dependent, not independent.** Independent runways decompose
into two easy problems. Use a partial lock (parallel-close or crossing):

- Simultaneous *landings*: illegal.
- Departure on one during landing rollout on the other: legal.

**Commitment point.** Once an aircraft is inside final (~45s / 2 miles), it can
only land or go around. No re-sequencing. This is the human's moment: if a
departure is slow to roll, there are ~20 seconds to choose between a go-around
penalty and a gamble.

> The game in one sentence: **build a schedule that fits, then react when reality doesn't.**

---

## 6. Scoring

One number, so all leagues compare directly.

```
+100   per landing
+100   per departure launched before timeout
  +2   per minute of fuel remaining at touchdown
 -50   separation violation
-200   go-around
-500   departure timeout
-1000  fuel exhaustion
-1000  collision / runway incursion
```

The fuel bonus is what prevents "hold everything, land slowly" from being optimal.

---

## 7. Config block — parameterize everything

**Every rule constant must arrive in the episode config JSON, never hardcoded in
the simulator.** This is what makes the synthesis league a real generalization
test rather than a curve fit: a controller that reads the config generalizes; one
that assumes 90-second occupancy dies on the crossing-runway config.

```json
{
  "config_id": "train_02",
  "seed": 44711,
  "speed_multiplier": 1.0,
  "airspace": {
    "fixes": [{"id": "KILO", "edges": ["APPR_09L"], "traverse_s": 60}],
    "altitude_bands": 3
  },
  "runways": [
    {"id": "09L", "landing_occupancy_s": 50, "trailing_block_s": 40,
     "takeoff_occupancy_s": 35, "dependent_with": ["09R"]}
  ],
  "separation_matrix": {
    "heavy_then_light": 40, "light_then_light": 0, "super_then_any": 90
  },
  "fuel": {"burn_slow": 0.7, "burn_normal": 1.0, "burn_fast": 2.0,
           "burn_hold": 1.5, "low_fuel_threshold_min": 8},
  "arrivals": {"distribution": "poisson", "rate_per_min": 1.4},
  "weather": {"cells": []},
  "episode_length_s": 900
}
```

### Train / test split (League 3+)

| Dimension | Train (agent may observe) | Held-out test |
|---|---|---|
| Runways | 1 runway; 2 parallel | 2 crossing |
| Wake classes | light, heavy | + super |
| Arrivals | steady Poisson | bursty |
| Weather | none, light | storm cell closes a runway mid-episode |
| Speed | 1.0× | 0.5× – 3.0× |

Budget observation in **episodes, not seconds** — 10 training episodes, unlimited
thinking time between them, then the controller is frozen and evaluated on
held-out configs.

---

## 8. Tick loop

```
SIM_TICK = 100ms (game time)
wall_dt  = SIM_TICK / speed_multiplier

loop:
    1. drain command queue (validate; reject illegal, log rejection with reason)
    2. advance aircraft along edges by dt
    3. burn fuel by state multiplier
    4. tick runway mutex timers
    5. resolve arrivals at fixes / approach commitment
    6. resolve touchdowns, takeoffs, timeouts, fuel exhaustion
    7. detect separation violations and collisions
    8. spawn new arrivals/departures per config distribution
    9. emit snapshot
```

- Server is **authoritative**. Clients send commands only, never positions.
- Snapshots to human UI at 10Hz; the UI interpolates between them.
- Agents may poll or subscribe at any rate; snapshots carry `tick` and `game_time_s`.
- **Determinism:** seeded RNG, integer tick counter, no wall-clock reads inside the core.
  Re-simulating `(config, seed, action_log)` must reproduce the run exactly. This gives
  replay, verification, and debugging for free.

---

## 9. Observation JSON (agent view)

```json
{
  "tick": 1204,
  "game_time_s": 120.4,
  "score": 830,
  "aircraft": [
    {
      "id": "AC017",
      "callsign": "SKY441",
      "type": "arrival",
      "wake": "heavy",
      "fuel_min": 6.2,
      "low_fuel": true,
      "at_fix": "KILO",
      "next_fix": "APPR_09L",
      "eta_next_fix_s": 34,
      "altitude_band": 2,
      "speed": "normal",
      "state": "enroute",
      "committed": false,
      "eta_touchdown_s": 96
    }
  ],
  "runways": [
    {"id": "09L", "occupied_until_s": 14.0, "blocked_until_s": 54.0,
     "cleared_aircraft": "AC012", "closed": false}
  ],
  "departures_waiting": [
    {"id": "AC023", "callsign": "JET220", "wake": "light", "timeout_in_s": 210}
  ],
  "weather": [],
  "events_since_last": [
    {"t": 118.2, "type": "landed", "aircraft": "AC009", "fuel_remaining_min": 12.4}
  ]
}
```

## 10. Action submission

Agents submit a **timed plan**, not a single action. The client executes it
tick-accurately; the next plan supersedes the current one. This makes agent
latency irrelevant — the agent always plans the window ahead of execution.

```json
{
  "episode_id": "ep_8831",
  "issued_at_tick": 1204,
  "plan": [
    {"at_game_time_s": 121.0, "cmd": "speed",      "aircraft": "AC017", "value": "slow"},
    {"at_game_time_s": 124.5, "cmd": "clear_land", "aircraft": "AC012", "runway": "09L"},
    {"at_game_time_s": 130.0, "cmd": "hold",       "aircraft": "AC019", "fix": "BRAVO"}
  ]
}
```

Illegal or stale entries are rejected individually with a reason, not silently
dropped. Rejections appear in the next snapshot's `events_since_last`.

## 11. HTTP / WS API

```
POST   /v1/episodes                  {config_id | config}  -> {episode_id, config, observation}
GET    /v1/episodes/{id}/observation                       -> observation
POST   /v1/episodes/{id}/plan        {plan}                -> {accepted[], rejected[]}
POST   /v1/episodes/{id}/abort
GET    /v1/episodes/{id}/result                            -> score + diagnostic
WS     /v1/episodes/{id}/stream                            -> snapshot stream
```

- Version the schema `v1` from day one. Changing it later invalidates every stored run.
- API key per agent; rate limit per key.
- Episodes in Redis with a TTL.
- **The human web client uses these same endpoints.** No private path into the
  engine — that is what makes parity structural rather than a maintenance task.

---

## 12. Human UI — the ladder view

**Distance from the runway = time to touchdown, not geographic position.** One
vertical lane per runway; blips slide down toward the threshold at the bottom.

Why this beats a radar map:

- Two blips at equal height in one lane = a conflict, visible with zero mental math.
- Issuing `slow` makes the blip visibly slide *backward* up the ladder — command
  effect is legible immediately.
- Runway occupancy renders as grey bars in the lane; you can see the gap a
  departure must fit into.

Rules:

- **Do not encode fuel by hue alone** (~1 in 12 men cannot reliably distinguish
  red/green). Use a depleting bar, plus a numeric countdown when critical.
- **Reserve motion and flashing for imminence only.** It is the one pre-attentive
  channel; spending it on non-urgent state wastes the best signal available.
- Departures queue in a side rail with visible timeout bars.
- Mouse: click an aircraft, then click a fix/runway, or drag onto a lane.

---

## 13. Leagues

The world speed multiplier scales *everything* — fuel burn, arrival rate, runway
occupancy. The game is identical in game-time; only wall-clock changes. This makes
it a fairness-preserving difficulty dial and gives a better leaderboard axis than
raw score:

> **Not "what did you score" but "what is the fastest speed at which you can still clear 2000 points?"**

| League | Controller | Latency | Tests |
|---|---|---|---|
| 1. Human | person at keyboard | ~250ms | perception, triage under load |
| 2. Live LLM | model in the loop per plan | 1–3s | in-context reasoning |
| 3. Synthesis | agent writes a controller after 10 observation episodes | ~0 | generalization from few episodes |
| 4. Weight-tuning | LLM adjusts a small parameter vector across rounds | ~0 | optimization from diagnostic feedback |

### League 3 — synthesis

Agent observes 10 training episodes, then submits a controller evaluated on
held-out configs (§7). Overfitting to training seeds craters the score.

**Do not execute submitted code.** Participants run their own controller against
your endpoints. Executing untrusted Python means building sandboxing, resource
limits and network isolation instead of building the game.

### League 4 — LLM-in-the-loop weight tuning

Publish a fixed feature extractor (~15 features per aircraft: fuel minutes, ETA,
runway conflict count, wake class, hold cost, …). The agent submits a small weight
matrix (~50–90 numbers) scoring candidate actions. At that size every coefficient
has a defensible meaning — "fuel urgency should dominate, +3.0" — so the LLM is
reasoning, not guessing floats.

Loop rules:

1. **Weights change between episodes only.** Mid-episode changes make the return
   correspond to no single policy.
2. **Average over 5 fixed seeds per candidate.** With one seed, run-to-run noise
   exceeds the effect of small weight changes and the LLM chases phantoms.
3. **Keep the full `(weights, diagnostic, score)` history in context.** Without it
   there is no optimization, only repeated guessing.
4. **Return a diagnostic, not a score.** This is the make-or-break detail:

```json
{
  "score": 1450,
  "lost_to_fuel": 2,
  "lost_to_fuel_detail": "both holding >4min at KILO",
  "separation_violations": 1,
  "separation_detail": "parallel runways, t=203",
  "runway_idle_pct": 38,
  "avg_fuel_at_touchdown_min": 11.0,
  "departures_timed_out": 3,
  "go_arounds": 1
}
```

`score: 1450` alone gives the model nothing to reason about. The breakdown lets it
diagnose — "38% idle with departures timing out means my departure priority weight
is too low" — which is the LLM's actual edge over a numeric optimizer.

### The measurement worth publishing

| Optimizer | Same 50-episode budget |
|---|---|
| Random search | baseline |
| Hill climbing / CMA-ES | strong baseline |
| LLM-in-the-loop | ? |

Plot score vs episodes. If the LLM beats CMA-ES on sample efficiency, that says
something real about diagnostic feedback. If it loses, that is also a genuine
result.

---

## 14. Non-goals and cautions

- **Not a pixel/vision benchmark.** State is structured; no frame scraping.
- **No continuous vector control.** Graph routing only.
- **No untrusted code execution** on the server.
- **Balance is empirical.** Build a scripted dumb bot as a third player type on
  day one so thousands of matches can be run before any human or LLM plays.
  Asymmetric multiplayer is notoriously hard to balance and there will be no
  playtesters at first.
- **Scope risk is the main risk.** This is four products stacked on one game.
  The single-player game must be fun on its own or nothing above it has a
  foundation.

---

## 15. Suggested stack

| Layer | Choice |
|---|---|
| Core sim | Python, pure functions, seeded RNG, no I/O |
| API | FastAPI + WebSockets |
| Episode state | Redis with TTL |
| Web client | React + canvas, calls the same public API |
| Packaging | Docker; optional Gymnasium adapter wrapping the core locally |

---

## 16. First milestone

One runway. Max 8 aircraft. No weather. No departures. Ladder UI. Playable by a
human at 1.0× and scoring correctly, with the core re-simulating an action log
deterministically.

Everything in this document is layered on top of that.
