"""Arcade server — FastAPI + WebSocket over the free-vector engine."""
from __future__ import annotations

import asyncio
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from atc.arcade.engine import ArcadeEngine, load_arcade_config

CONFIG = Path(__file__).resolve().parents[3] / "configs" / "arcade_m1.json"
WEB = Path(__file__).resolve().parent / "web"
TTL_S = 1800
_games: dict[str, "Runner"] = {}


class Runner:
    def __init__(self, seed: int | None, speed: float | None = None):
        cfg = load_arcade_config(CONFIG)
        if seed is not None:
            cfg = load_arcade_config({**cfg.raw, "seed": seed})
        if speed is not None:
            cfg = load_arcade_config({**cfg.raw, "speed_multiplier": float(speed)})
        self.id = f"g_{uuid.uuid4().hex[:8]}"
        self.engine = ArcadeEngine(cfg)
        self.created = time.monotonic()
        self.subs: set[WebSocket] = set()
        self.task: asyncio.Task | None = None
        self.stopped = False

    def start(self) -> None:
        self.task = asyncio.create_task(self._loop())

    async def _loop(self) -> None:
        # speed_multiplier scales WALL-CLOCK only. Game time, physics and
        # scoring are identical, so a run at 0.4x is directly comparable to one
        # at 1x — it just gives a slow thinker room to think. Same fairness dial
        # the graph engine used for its agent leagues.
        dt = self.engine.cfg.dt / max(0.05, self.engine.cfg.speed_multiplier)
        nxt = time.monotonic()
        while not self.stopped:
            self.engine.step()
            snap = self.engine.observation()
            for ws in list(self.subs):
                try:
                    await ws.send_json(snap)
                except Exception:
                    self.subs.discard(ws)
            if self.engine.game_over:
                break
            nxt += dt
            await asyncio.sleep(max(0.0, nxt - time.monotonic()))

    def stop(self) -> None:
        self.stopped = True
        if self.task:
            self.task.cancel()


def _get(gid: str) -> Runner:
    r = _games.get(gid)
    if r is None:
        raise HTTPException(404, f"unknown game {gid}")
    return r


app = FastAPI(title="ATC Arena — Arcade", version="1.0")


@app.post("/v1/games")
async def new_game(body: dict | None = None) -> dict:
    now = time.monotonic()
    for gid in [k for k, r in _games.items() if now - r.created > TTL_S]:
        _games.pop(gid).stop()
    b = body or {}
    r = Runner(b.get("seed"), b.get("speed_multiplier"))
    _games[r.id] = r
    r.start()
    return {"game_id": r.id, "config": r.engine.cfg.raw, "state": r.engine.observation()}


@app.post("/v1/games/{gid}/path")
async def set_path(gid: str, body: dict) -> dict:
    r = _get(gid)
    reason = r.engine.set_path(body.get("aircraft", ""),
                               [(p[0], p[1]) for p in body.get("path", [])])
    return {"accepted": reason is None, "reason": reason}


@app.get("/v1/games/{gid}/state")
async def state(gid: str) -> dict:
    return _get(gid).engine.observation()


@app.post("/v1/games/{gid}/abort")
async def abort(gid: str) -> dict:
    r = _get(gid); r.stop()
    return r.engine.observation()


@app.websocket("/v1/games/{gid}/stream")
async def stream(ws: WebSocket, gid: str) -> None:
    r = _games.get(gid)
    await ws.accept()
    if r is None:
        await ws.close(code=4404); return
    r.subs.add(ws)
    await ws.send_json(r.engine.observation())
    try:
        while True:
            await ws.receive_text()
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        r.subs.discard(ws)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "games": len(_games)}


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(WEB / "index.html")


app.mount("/vendor", StaticFiles(directory=WEB / "vendor"), name="vendor")
