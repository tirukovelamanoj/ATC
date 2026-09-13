"""Play ATC Arena from your own machine, against a hosted (or local) server.

    python examples/agent.py                                  # uses the bundled policy
    python examples/agent.py --model runs/my_policy.onnx
    python examples/agent.py --server https://game.example.com

Without the repo, three files are enough -- grab them from any running server:
    curl -O https://game.example.com/v1/agent.py
    curl -O https://game.example.com/v1/spatial.py
    curl -o policy.onnx https://game.example.com/v1/policy.onnx
    pip install numpy websockets onnxruntime

It prints a watch link. Open it and anyone can see your agent flying, live —
the simulation runs on the server, so the score is the server's, not yours.

The model here is ONNX only because that keeps the example dependency-light.
Nothing in the protocol requires it: replace `decide()` with torch, JAX, a
hand-written heuristic, or anything else that turns an observation into a cell
index. The server neither knows nor cares.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import urllib.request

import numpy as np
import websockets

try:                     # installed alongside the repo
    from atc.arcade.spatial import build_obs, cell_centre, leg, needs_route
except ModuleNotFoundError:   # downloaded loose from a running server
    from spatial import build_obs, cell_centre, leg, needs_route


def post(base: str, path: str, body: dict) -> dict:
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(),
                                 headers={"content-type": "application/json"},
                                 method="POST")
    return json.load(urllib.request.urlopen(req))


class OnnxBrain:
    """Turns an observation grid into a target cell. Swap this out for your own."""

    def __init__(self, path: str):
        import onnxruntime as ort
        self.sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        shape = self.sess.get_inputs()[0].shape       # (batch, channels, gh, gw)
        self.gh, self.gw = int(shape[2]), int(shape[3])

    def decide(self, obs: np.ndarray) -> int:
        logits = self.sess.run(None, {"obs": obs[None]})[0][0]
        return int(logits.argmax())


async def play(server: str, ws_url: str, brain, quiet: bool, seed: int | None) -> dict:
    # Without a seed the server reuses the one in its config, so every run is
    # the SAME game — fine for a reproducible comparison, misleading if you
    # think you are sampling. Random by default; pin it with --seed.
    game = post(server, "/v1/games", {} if seed is None else {"seed": seed})
    gid, cfg = game["game_id"], game["config"]
    watch = f"{server}/?watch={gid}"
    # flush: stdout is block-buffered when piped, and a watch link that only
    # appears after the run has finished is no use to anyone.
    print(f"game {gid}\nWATCH LIVE: {watch}\n", flush=True)

    last = {}
    async with websockets.connect(f"{ws_url}/v1/games/{gid}/stream") as ws:
        async for raw in ws:
            state = json.loads(raw)
            if state.get("type") == "finished" or state.get("game_over"):
                last = state
                break
            aid = needs_route(state)
            if aid is None:
                continue
            obs = build_obs(state, cfg, brain.gw, brain.gh, aid)
            cell = brain.decide(obs)
            tx, ty = cell_centre(cell, cfg["map"]["w"], cfg["map"]["h"], brain.gw, brain.gh)
            ac = next(a for a in state["aircraft"] if a["id"] == aid)
            post(server, f"/v1/games/{gid}/path",
                 {"aircraft": aid, "path": leg(ac["x"], ac["y"], tx, ty)})
            if not quiet and state["landed"] != last.get("landed"):
                print(f"  t={state['time_s']:6.1f}s  landed {state['landed']}  "
                      f"score {state['score']}", flush=True)
            last = state
    return last


def main() -> int:
    ap = argparse.ArgumentParser(prog="agent")
    ap.add_argument("--server", default="http://127.0.0.1:8099")
    ap.add_argument("--model", default="models/policy.onnx")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--seed", type=int, default=None,
                    help="pin the episode; omit for a random one")
    a = ap.parse_args()

    ws_url = a.server.replace("https://", "wss://").replace("http://", "ws://")
    brain = OnnxBrain(a.model)
    print(f"model expects an {brain.gw}x{brain.gh} grid", flush=True)
    seed = a.seed if a.seed is not None else random.randrange(1 << 30)
    print(f"seed {seed}", flush=True)
    final = asyncio.run(play(a.server, ws_url, brain, a.quiet, seed))
    print(f"\nFINAL  landed {final.get('landed')}  score {final.get('score')}  "
          f"survived {final.get('time_s', 0):.0f}s")
    if final.get("over_reason"):
        print(f"       {final['over_reason']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
