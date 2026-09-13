# ATC Arena

A real-time air traffic control game you can play with a mouse — and train a
reinforcement learning agent to play on the exact same interface.

**Play it: [game.manojtirukovela.com](https://game.manojtirukovela.com)** — press
**WATCH AI** to see the trained policy fly, or **TRY WITH YOUR AI** for the
protocol and a downloadable agent.

![ATC Arena gameplay](docs/images/gameplay.jpg)

Draw a flight path from an aircraft to its matching landing zone. Jets take the
blue runway, props the green one, helicopters the amber pad. Two aircraft
touching is game over. The spawn rate ramps up with every landing, so the run
ends when the traffic beats you.

The agent and the human drive the **same public API** — there is no private path
into the engine, so a policy is playing the game you play, not a simplified
version of it.

---

## Quick start

Requires Python 3.13 and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/tirukovelamanoj/ATC.git && cd ATC
uv venv --python 3.13
uv pip install -e .
.venv/bin/python -m uvicorn atc.arcade.server:app --port 8099
```

Open <http://127.0.0.1:8099> and press **NEW GAME**. Click and drag from an
aircraft to draw its route — the aircraft follows the line you drew exactly.

![Drawing a flight path](docs/images/drawing.jpg)

The map is generated per game: simplex-noise elevation banded into biomes and
hillshaded, with a flat plateau forced under every landing zone so a runway can
never appear in the sea.

---

## Train an agent

The game doubles as a Gymnasium environment. The policy sees the map as a small
grid and picks one cell per decision: *where should this aircraft go next.*

```bash
uv pip install -e ".[train]"          # adds gymnasium, stable-baselines3, torch (~2.5GB)

python -m atc.arcade.bench            # scripted baseline, ~25 landings
python -m atc.arcade.train --steps 4000000 --envs 8 --grid-w 40 --grid-h 28 --shaping 0.05
python -m atc.arcade.evaluate runs/ppo_grid.zip --grid-w 40 --grid-h 28
```

The policy is **fully convolutional**: a 1x1 conv emits one logit per grid cell,
so it has ~58k weights *at any resolution* — a 20x14 grid and a 40x28 grid use
the identical network. Flattening the feature map into a dense head instead
would cost 18M weights at 40x28 and would not be resolution-independent at all.

`--device` defaults to `auto`. The convolutions run at full grid resolution, so
their cost scales with grid area: at 20x14 the CPU is fine, but at 40x28 one
update is ~750ms on CPU versus ~90ms on Apple MPS, which is 181 vs 747
steps/s end to end. The simulator itself is never the bottleneck — it runs at
~156,000 steps/s and the Gymnasium wrapper at ~3,700 decisions/s.

### Scores to beat

All measured the same way — 25 episodes from seed 500, 40x28 grid, no action
delay — because comparing across grid sizes is meaningless. The numbers live in
`configs/baselines.json`, which the server, `evaluate.py` and the docs page all
read; they used to be three hardcoded copies and two of them had drifted.

| controller | landings | notes |
|---|---|---|
| random | 0.80 | picks cells at random |
| aim at the zone | 5.64 | flies straight at it, ignores runway heading |
| `GreedyRouter` | 24.87 | scripted, routes once per aircraft |
| aim + runway alignment | 29.08 | the bar — hand-coded, re-decides continuously |
| **trained policy** | **33.68** | `models/policy.onnx`, 800k steps |

The network clears the hand-coded bar, but **not** by understanding traffic:
blank every other aircraft out of its observation and only 0.7% of its decisions
change. That is the open headroom — a policy that actually spaces traffic has a
lot of room above 33.68.

### How the environment is shaped

```
  OBSERVATION  Box(8, 28, 40)              ACTION  Discrete(1120)
  8 channels over the same grid            one cell = the next waypoint
   selected aircraft · heading sin/cos      the env expands it into the same
   other traffic · their paths              polyline a mouse would draw
   target zone · all zones · conflicts
```

Input and output share a grid on purpose: a convolutional policy is then
translation-equivariant, so *"avoid the aircraft two cells north-east"*
generalises across the whole map instead of being memorised per location.

Emitting one cell at a time — rather than painting a whole route — keeps every
action valid. Multi-leg routes emerge from repeated decisions, and the training
signal is far denser.

### Latency

A deployed agent talks over a network. Train with the delay simulated:

```bash
python -m atc.arcade.train --steps 4000000 --delay-max 40   # up to 2s, randomised
```

Measured cost of latency for the hand-coded bar:

| action delay | landings |
|---|---|
| 0 | 27.6 |
| 1.0s | 17.2 |
| 2.0s | 10.6 |

Do **not** train against a live server. The game advances at 20 ticks per *real*
second there, versus 156,000 steps/s locally — about 7,800x slower, and network
jitter makes runs non-reproducible. Train locally with simulated delay; use the
live server to evaluate.

---

## Two engines

| | `src/atc/` (graph) | `src/atc/arcade/` (active) |
|---|---|---|
| Control | route / clear-to-land / hold | free-form drawn paths |
| Loss condition | fuel exhaustion, separation | mid-air collision |
| Agents | LLM leagues, timed plans | RL over a spatial action map |
| Run | `uvicorn atc.server:app` | `uvicorn atc.arcade.server:app` |

The arcade engine is the game. The graph engine came first, is kept intact, and
is the only one where human-vs-agent scores are strictly comparable — a human
draws with a mouse while an agent emits exact coordinates, which is different
input bandwidth on the same task. Agent-vs-agent comparison is sound on both.

`atc-arena-spec.md` is the original design document.

## Layout

```
src/atc/arcade/
├── engine.py      the simulation. the only place game rules live
├── server.py      FastAPI + WebSocket; also serves the UI
├── web/           canvas client (no framework, no WebGL)
├── spatial.py     THE observation encoder. one copy, shared by all three
│                  callers: trainer, hosted pilot, remote agents
├── gym_env.py     Gymnasium wrapper, spatial action map
├── train.py       PPO + small CNN
├── export_onnx.py torch checkpoint -> single self-contained .onnx
├── evaluate.py    policy vs the scripted bars
├── bots.py        scripted baselines
└── bench.py       baseline sweep
configs/           every rule constant, plus baselines.json (published scores)
examples/agent.py  remote client; runs standalone from downloaded files
models/policy.onnx the served policy, 234KB
tests/             determinism invariant for the graph engine
```

## Tuning without touching code

Everything in `configs/arcade_m1.json` is live: aircraft speeds and turn rates,
spawn ramp, zone positions and colours, collision radii, scoring, map size.

```jsonc
"speed_multiplier": 1.0,      // wall-clock only; game time is unchanged
"join_style": "immediate",    // or "procedure_turn" for a banked Dubins join
"heading_slew_deg_s": 540     // how fast a sprite swings to face its travel
```

## Watch a trained policy play

![The trained policy flying](docs/images/ai-playing.jpg)

Export a checkpoint to ONNX and the server will fly a game itself — press
**WATCH AI** in the browser.

```bash
python -m atc.arcade.export_onnx runs/ppo_v2.zip --out models/policy.onnx
uv pip install -e ".[ai]"          # onnxruntime, ~50MB
.venv/bin/python -m uvicorn atc.arcade.server:app --port 8099
```

`models/policy.onnx` is a single self-contained file (~230KB) and inference is
~0.2ms on CPU, so serving needs no GPU and no torch. The export verifies that
the ONNX graph picks the same action as torch on 200 random observations —
without that check a deployed agent can silently play a different policy than
the one you evaluated.

## Bring your own agent

Press **TRY WITH YOUR AI** in the running game for the full protocol, or:

```bash
python examples/agent.py --server https://your-server
```

It creates a game, plays it, and prints a link like `/?watch=g_7f3a` that anyone
can open to watch your agent fly live.

**Your model runs on your machine and is never uploaded.** The simulation stays
server-side, so scores and seeds are the server's — which is what makes them
comparable. A 1-second round trip costs about 38% of landings; a local one is
10-50ms, so the network is not a factor.

| | |
|---|---|
| `GET /v1/spec` | the whole contract as JSON |
| `GET /v1/policy.onnx` | reference weights (234KB) |
| `GET /v1/agent.py` | the example client |
| `GET /v1/spatial.py` | the observation encoder, so the client runs without this repo |
| `GET /docs` | Swagger for the HTTP endpoints |
| `WS /v1/games/{id}/stream` | state every tick |
| `POST /v1/games/{id}/path` | your action: a polyline |

No clone needed to play — three files and three packages:

```bash
curl -O https://game.manojtirukovela.com/v1/agent.py
curl -O https://game.manojtirukovela.com/v1/spatial.py
curl -o policy.onnx https://game.manojtirukovela.com/v1/policy.onnx
pip install numpy websockets onnxruntime
python agent.py --server https://game.manojtirukovela.com
```

Nothing requires ONNX. Replace `decide()` in the example with torch, JAX or a
heuristic — it only has to turn an observation into a target cell. Build the
observation with `build_obs(...)` from `spatial.py`: the trainer, this server and
your agent all call that one function, so your encoding cannot drift from the one
the policy was trained on. `/v1/spatial.py` serves the *same file the server
imports*, not a copy — a second copy is the drift that module exists to prevent.

## Deploying

```bash
docker build -t atc-arena .
docker run -p 8000:8000 atc-arena      # then open localhost:8000
```

The image carries the game, the config and the exported policy — but not torch,
which is why it stays small enough for a modest instance: 449MB on disk, ~60MB
resident whatever the load.

**Sizing is a CPU decision, not a memory one.** Every game is a live 20Hz task,
so capacity is set by vCPU. Measured as sim-seconds per wall-second (1.0 = the
game is keeping real time) using AI games, the worst case:

| instance | 2 games | 4 games | 8 games |
|---|---|---|---|
| 0.10 vCPU (free tier) | 0.94x | 0.71x | 0.24x |
| 0.25 vCPU (eco-micro) | 1.00x | 1.00x | 0.33x |
| 0.50 vCPU (eco-small) | 1.00x | 1.00x | 0.52x |

Roughly 0.06 vCPU per game. An overloaded event loop does not shed load, it runs
the simulation in slow motion for *everyone*, so `ATC_MAX_GAMES` (default 4)
must match real capacity. Past the cap the server returns 429 with `Retry-After`.
Abandoned games free themselves after 60s of no contact — a WebSocket or any
HTTP call counts, so an agent that polls is not mistaken for a closed tab.

Avoid the free tier for anything public: 0.1 vCPU degrades at two concurrent AI
games, and it scales to zero after an hour idle, which is most of the time for a
portfolio link.

On Koyeb: point a service at the repo (it builds the Dockerfile), set the health
check to `/health`, and attach your domain. WebSockets and TLS work out of the
box; the client picks `wss://` automatically when served over HTTPS.

Keep it to a single instance — games live in process memory, so a second replica
would strand players whose WebSocket lands on the wrong one. `min = max = 1`.

| env var | default | |
|---|---|---|
| `ATC_MAX_GAMES` | 4 | concurrent games before 429 |
| `ATC_LOG_LEVEL` | INFO | `DEBUG` adds a line per drawn path |

## Not built yet

- Submitting a trained policy to a hosted instance for live evaluation
- Persisting games outside process memory (needed before scaling past one replica)
- A policy that actually avoids conflicts — the current one essentially ignores
  the traffic channels

## License

MIT — see `LICENSE`.

## Credits

`web/vendor/simplex-noise.js` — simplex-noise 2.4.0, MIT, (c) 2018 Jonas Wagner.
Vendored rather than CDN-linked so the game runs offline. Everything else is
first-party: the terrain, aircraft, runways and effects are drawn with plain
Canvas 2D.
