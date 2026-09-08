"""HTTP / WebSocket surface — §11.

A thin shell over the engine. No game logic lives here; if a rule needs to
change, it changes in engine.py or the config JSON, never in a route.

The human web client uses these exact endpoints. There is no private path into
the engine — that is what makes the §1 parity invariant structural rather than
a maintenance task.
"""
from __future__ import annotations

import asyncio
import contextlib
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from atc.config import TICK_MS, EpisodeConfig, load_config
from atc.engine import Engine
from atc.observation import to_diagnostic, to_observation
from atc.state import Command

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"
WEB_DIR = Path(__file__).resolve().parent / "web"
EPISODE_TTL_S = 3600

# ponytail: in-process episode store. §11 wants Redis with a TTL; that matters
# only once there is more than one server process. Swap when we scale out.
_episodes: dict[str, "EpisodeRunner"] = {}


class EpisodeRunner:
    """Owns one Engine and advances it in real time at the config's speed."""

    def __init__(self, config: EpisodeConfig, speed_multiplier: float):
        self.id = f"ep_{uuid.uuid4().hex[:8]}"
        self.engine = Engine(config)
        self.speed = speed_multiplier or 1.0
        self.created_at = time.monotonic()
        self.subscribers: set[WebSocket] = set()
        self.task: asyncio.Task | None = None
        self.aborted = False

    def start(self) -> None:
        self.task = asyncio.create_task(self._loop())

    async def _loop(self) -> None:
        # Absolute deadlines rather than sleep(dt): sleep drift would otherwise
        # accumulate and make wall-clock pacing diverge from game time.
        wall_dt = (TICK_MS / 1000) / self.speed
        next_at = time.monotonic()
        try:
            while not self.engine.done and not self.aborted:
                self.engine.step()
                await self._broadcast(to_observation(self.engine))
                next_at += wall_dt
                await asyncio.sleep(max(0.0, next_at - time.monotonic()))
            await self._broadcast({"type": "finished", **self.result()})
        except asyncio.CancelledError:
            raise

    async def _broadcast(self, payload: dict) -> None:
        for ws in list(self.subscribers):
            try:
                await ws.send_json(payload)
            except Exception:
                self.subscribers.discard(ws)

    def result(self) -> dict:
        return {"episode_id": self.id, "finished": self.engine.done or self.aborted,
                "diagnostic": to_diagnostic(self.engine)}

    def stop(self) -> None:
        self.aborted = True
        if self.task:
            self.task.cancel()


def _reap() -> None:
    now = time.monotonic()
    for eid in [k for k, r in _episodes.items() if now - r.created_at > EPISODE_TTL_S]:
        _episodes.pop(eid).stop()


def _get(episode_id: str) -> EpisodeRunner:
    runner = _episodes.get(episode_id)
    if runner is None:
        raise HTTPException(404, f"unknown episode {episode_id}")
    return runner


def _to_command(entry: dict, engine: Engine) -> tuple[Command | None, str | None]:
    """Map a §10 plan entry onto an engine Command.

    An entry with no time means 'now' (a human clicking a button); an explicit
    past time is stale and gets rejected, per §10.
    """
    cmd = entry.get("cmd")
    aircraft = entry.get("aircraft")
    if not cmd or not aircraft:
        return None, "entry needs both 'cmd' and 'aircraft'"
    value = next((entry[k] for k in ("value", "runway", "fix", "band") if k in entry), None)
    if "at_game_time_s" in entry and entry["at_game_time_s"] is not None:
        at_tick = int(round(float(entry["at_game_time_s"]) * (1000 // TICK_MS)))
        if at_tick < engine.tick:
            return None, f"stale: at_game_time_s {entry['at_game_time_s']} is in the past"
    else:
        at_tick = engine.tick + 1
    return Command(at_tick, cmd, aircraft, None if value is None else str(value)), None


app = FastAPI(title="ATC Arena", version="1.0")


@app.post("/v1/episodes")
async def create_episode(body: dict | None = None) -> dict:
    _reap()
    body = body or {}
    if "config" in body:
        cfg = load_config(body["config"])
    else:
        path = CONFIG_DIR / f"{body.get('config_id', 'm1_single_runway')}.json"
        if not path.exists():
            raise HTTPException(400, f"unknown config_id {body.get('config_id')}")
        cfg = load_config(path)
    if "seed" in body:
        cfg = load_config({**cfg.raw, "seed": int(body["seed"])})

    runner = EpisodeRunner(cfg, body.get("speed_multiplier", cfg.raw.get("speed_multiplier", 1.0)))
    _episodes[runner.id] = runner
    runner.start()
    return {"episode_id": runner.id, "config": cfg.raw,
            "observation": to_observation(runner.engine)}


@app.get("/v1/episodes/{episode_id}/observation")
async def get_observation(episode_id: str) -> dict:
    return to_observation(_get(episode_id).engine)


@app.post("/v1/episodes/{episode_id}/plan")
async def submit_plan(episode_id: str, body: dict) -> dict:
    runner = _get(episode_id)
    accepted, rejected = [], []
    for entry in body.get("plan", []):
        cmd, err = _to_command(entry, runner.engine)
        if err:
            rejected.append({**entry, "reason": err})
            continue
        ok, bad = runner.engine.submit([cmd])
        accepted.extend(c.as_log() for c in ok)
        rejected.extend({**entry, "reason": r.reason} for r in bad)
    return {"accepted": accepted, "rejected": rejected}


@app.post("/v1/episodes/{episode_id}/abort")
async def abort_episode(episode_id: str) -> dict:
    runner = _get(episode_id)
    runner.stop()
    return runner.result()


@app.get("/v1/episodes/{episode_id}/result")
async def get_result(episode_id: str) -> dict:
    return _get(episode_id).result()


@app.websocket("/v1/episodes/{episode_id}/stream")
async def stream(ws: WebSocket, episode_id: str) -> None:
    runner = _episodes.get(episode_id)
    await ws.accept()
    if runner is None:
        await ws.close(code=4404)
        return
    runner.subscribers.add(ws)
    await ws.send_json(to_observation(runner.engine))
    try:
        while True:                      # client sends nothing; hold the socket open
            await ws.receive_text()
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        runner.subscribers.discard(ws)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "episodes": len(_episodes)}


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


@app.on_event("shutdown")
async def _shutdown() -> None:
    for runner in list(_episodes.values()):
        runner.stop()
        with contextlib.suppress(Exception):
            await runner.task
