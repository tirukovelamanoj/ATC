"""Arcade server — FastAPI + WebSocket over the free-vector engine."""
from __future__ import annotations

import asyncio
import os
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from atc.arcade.engine import ArcadeEngine, load_arcade_config
from atc.arcade.spatial import (CHANNELS, build_obs, cell_centre, cell_of, leg,
                                needs_route)

# parents[3] is the repo root when running from a source checkout, but points
# into site-packages once pip-installed — so the container would not find its
# own config. The env vars make the location explicit wherever it runs.
_ROOT = Path(__file__).resolve().parents[3]
CONFIG = Path(os.environ.get("ATC_CONFIG", _ROOT / "configs" / "arcade_m1.json"))
MODEL = Path(os.environ.get("ATC_MODEL", _ROOT / "models" / "policy.onnx"))
EXAMPLE = Path(os.environ.get("ATC_EXAMPLE", _ROOT / "examples" / "agent.py"))
WEB = Path(__file__).resolve().parent / "web"
TTL_S = 1800
_games: dict[str, "Runner"] = {}


class AIPilot:
    """Serves a trained policy. Uses onnxruntime (~50MB) rather than torch
    (~2.5GB) so the whole thing fits on a small instance.

    The grid size is read out of the model's own input shape instead of being
    configured separately — a mismatch there would feed the policy a differently
    shaped world and look like bad play rather than a bug.
    """

    def __init__(self, path: Path):
        import onnxruntime as ort
        self.sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        shape = self.sess.get_inputs()[0].shape          # (batch, C, H, W)
        self.gh, self.gw = int(shape[2]), int(shape[3])

    def act(self, engine, cfg) -> None:
        state = engine.observation()
        aid = needs_route(state)
        if aid is None:
            return
        obs = build_obs(state, cfg.raw, self.gw, self.gh, aid)[None]
        logits = self.sess.run(None, {"obs": obs})[0][0]
        tx, ty = cell_centre(int(logits.argmax()), cfg.map_w, cfg.map_h, self.gw, self.gh)
        ac = engine.aircraft[aid]
        engine.set_path(aid, leg(ac.x, ac.y, tx, ty))


def load_pilot() -> "AIPilot | None":
    if not MODEL.exists():
        return None
    try:
        return AIPilot(MODEL)
    except Exception as exc:                              # missing extra, bad file
        print(f"AI unavailable: {type(exc).__name__}: {exc}", flush=True)
        return None


class Runner:
    def __init__(self, seed: int | None, speed: float | None = None,
                 pilot: "AIPilot | None" = None):
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
        self.pilot = pilot

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
            if self.pilot is not None:
                self.pilot.act(self.engine, self.engine.cfg)
            self.engine.step()
            snap = self.engine.observation()
            snap["ai"] = self.pilot is not None
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
_PILOT = load_pilot()   # loaded once at import; None if absent


@app.post("/v1/games")
async def new_game(body: dict | None = None) -> dict:
    now = time.monotonic()
    for gid in [k for k, r in _games.items() if now - r.created > TTL_S]:
        _games.pop(gid).stop()
    b = body or {}
    pilot = _PILOT if b.get("ai") else None
    if b.get("ai") and pilot is None:
        raise HTTPException(503, "no trained policy available on this server")
    r = Runner(b.get("seed"), b.get("speed_multiplier"), pilot)
    _games[r.id] = r
    r.start()
    st = r.engine.observation(); st["ai"] = pilot is not None
    return {"game_id": r.id, "config": r.engine.cfg.raw, "state": st, "ai": pilot is not None}


@app.post("/v1/games/{gid}/path")
async def set_path(gid: str, body: dict) -> dict:
    r = _get(gid)
    reason = r.engine.set_path(body.get("aircraft", ""),
                               [(p[0], p[1]) for p in body.get("path", [])])
    return {"accepted": reason is None, "reason": reason}


@app.get("/v1/games/{gid}/config")
async def game_config(gid: str) -> dict:
    """Map geometry and rules for a running game — a spectator needs this to
    draw the same world, and a remote agent needs it to encode observations."""
    return _get(gid).engine.cfg.raw


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


@app.get("/v1/spec")
async def spec() -> dict:
    """The agent contract, served from the same constants the engine uses.

    Hand-written protocol docs drift from the code; this cannot, because the
    channel list comes straight out of the encoder both sides import.
    """
    return {
        "observation": {
            "shape": ["8", "grid_h", "grid_w"],
            "dtype": "float32",
            "channels": list(CHANNELS),
            "how": "build_obs(state, config, grid_w, grid_h, aircraft_id) in atc.arcade.spatial",
        },
        "action": {
            "endpoint": "POST /v1/games/{game_id}/path",
            "body": {"aircraft": "AC017", "path": [[520.0, 300.0], "..."]},
            "note": "A path is a polyline in map coordinates. The aircraft flies it exactly.",
        },
        "flow": [
            "POST /v1/games                     -> {game_id, config, state}",
            "connect WS /v1/games/{id}/stream   -> a state snapshot every tick",
            "when an aircraft has an empty path, choose where it should go",
            "POST /v1/games/{id}/path           -> it flies your polyline",
            "open /?watch={game_id}             -> anyone can watch it live",
        ],
        "endpoints": {
            "create": "POST /v1/games",
            "state": "GET /v1/games/{id}/state",
            "config": "GET /v1/games/{id}/config",
            "path": "POST /v1/games/{id}/path",
            "abort": "POST /v1/games/{id}/abort",
            "stream": "WS /v1/games/{id}/stream",
            "http_docs": "/docs",
        },
        "downloads": {"policy": "/v1/policy.onnx", "example_agent": "/v1/agent.py"},
        "reference_policy": {
            "available": _PILOT is not None,
            "grid": [_PILOT.gw, _PILOT.gh] if _PILOT else None,
            "scores": {"random": 0.88, "scripted_greedy": 24.87,
                       "hand_coded_bar": 30.65, "this_policy": 33.5},
        },
    }


@app.get("/v1/sample")
async def sample_observation() -> dict:
    """A REAL observation, plus what the bundled policy makes of it.

    Generated live from a throwaway game rather than hand-written, so the
    documentation cannot describe an encoding the code no longer produces.
    Sent sparsely: a typical observation has under a dozen non-zero cells out of
    8960, so this is ~1KB instead of 35KB.
    """
    import numpy as np

    cfg = load_arcade_config(CONFIG)
    eng = ArcadeEngine(cfg)
    for _ in range(3000):                      # run until there is real traffic
        eng.step()
        st = eng.observation()
        if len(st["aircraft"]) >= 3 and needs_route(st):
            break
    state = eng.observation()
    gw, gh = (_PILOT.gw, _PILOT.gh) if _PILOT else (40, 28)
    sel = needs_route(state)
    obs = build_obs(state, cfg.raw, gw, gh, sel)

    channels = []
    for i, name in enumerate(CHANNELS):
        ys, xs = np.nonzero(obs[i])
        channels.append({"name": name,
                         "cells": [[int(x), int(y), round(float(obs[i, y, x]), 3)]
                                   for x, y in zip(xs, ys)]})

    # world geometry too, so the docs can draw the map beside the matrix and
    # show which cell a given position lands in
    out = {"grid": [gw, gh], "channels": channels, "selected": sel,
           "map": {"w": cfg.map_w, "h": cfg.map_h,
                   "cell_w": round(cfg.map_w / gw, 1),
                   "cell_h": round(cfg.map_h / gh, 1)},
           "zones": [{"id": z.id, "kind": z.kind, "colour": z.colour,
                      "pos": list(z.pos), "accepts": list(z.accepts),
                      "length": z.length, "width": z.width, "radius": z.radius,
                      "heading_deg": z.heading_deg} for z in cfg.zones],
           "types": {k: {"speed": t.speed, "radius": t.radius, "zone": t.zone}
                     for k, t in cfg.types.items()},
           "collision_dist": round(2 * max(t.radius for t in cfg.types.values()), 1),
           "aircraft": [{"callsign": a["callsign"], "type": a["type"],
                         "zone": a["zone"], "selected": a["id"] == sel,
                         "x": a["x"], "y": a["y"], "heading": a["heading"],
                         "cell": list(cell_of(a["x"], a["y"], cfg.map_w, cfg.map_h, gw, gh))}
                        for a in state["aircraft"]]}

    if _PILOT is not None and sel:
        logits = _PILOT.sess.run(None, {"obs": obs[None]})[0][0]
        e = np.exp(logits - logits.max())
        probs = e / e.sum()
        top = int(probs.argmax())
        out["policy"] = {
            "chosen_cell": top,
            "chosen_xy": [top % gw, top // gw],
            # only the meaningful tail: most cells carry ~zero probability
            # int(): argsort yields numpy.int64, which pydantic cannot serialise
            "heatmap": [[int(i) % gw, int(i) // gw, round(float(probs[i]), 5)]
                        for i in np.argsort(probs)[-60:][::-1]],
        }
    return out


@app.get("/v1/policy.onnx")
async def download_policy() -> FileResponse:
    if not MODEL.exists():
        raise HTTPException(404, "no policy on this server")
    return FileResponse(MODEL, media_type="application/octet-stream",
                        filename="policy.onnx")


@app.get("/v1/agent.py")
async def download_agent() -> FileResponse:
    if not EXAMPLE.exists():
        raise HTTPException(404, "example not bundled")
    return FileResponse(EXAMPLE, media_type="text/x-python", filename="agent.py")


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "games": len(_games), "ai": _PILOT is not None}


@app.get("/")
async def index() -> FileResponse:
    # No Cache-Control means browsers fall back to HEURISTIC caching and can
    # serve a stale page without revalidating — which shows up as "the new
    # button does nothing" after a deploy. no-cache still allows a 304 via the
    # ETag, so this costs a round trip, not a re-download.
    return FileResponse(WEB / "index.html",
                        headers={"Cache-Control": "no-cache"})


app.mount("/vendor", StaticFiles(directory=WEB / "vendor"), name="vendor")
