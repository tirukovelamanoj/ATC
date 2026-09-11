"""Export a trained policy to ONNX.

    python -m atc.arcade.export_onnx runs/ppo_v2.zip --out runs/policy.onnx

Production loads the ONNX file with onnxruntime (~50MB) instead of torch
(~2.5GB), which is the difference between fitting on a small instance and not.
Inference is a single forward pass over an 8x28x40 grid — about a millisecond on
CPU, and the game only asks a few times per second.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch as th
import torch.nn as nn
from stable_baselines3 import PPO


class LogitsOnly(nn.Module):
    """obs -> action logits. Drops the value head, which serving never needs."""

    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, obs: th.Tensor) -> th.Tensor:
        feats = self.policy.extract_features(obs)
        if not self.policy.share_features_extractor:
            feats = feats[0]
        return self.policy.action_net(self.policy.mlp_extractor.forward_actor(feats))


def main() -> int:
    ap = argparse.ArgumentParser(prog="atc.arcade.export_onnx")
    ap.add_argument("model")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--grid-w", type=int, default=40)
    ap.add_argument("--grid-h", type=int, default=28)
    a = ap.parse_args()
    out = a.out or Path(a.model).with_suffix(".onnx")

    model = PPO.load(a.model, device="cpu")
    net = LogitsOnly(model.policy).eval()
    dummy = th.zeros(1, 8, a.grid_h, a.grid_w)

    th.onnx.export(net, (dummy,), str(out), input_names=["obs"], output_names=["logits"],
                   dynamic_axes={"obs": {0: "batch"}, "logits": {0: "batch"}}, opset_version=17)

    # The exporter splits weights into a sibling .onnx.data file by default.
    # Shipping only the .onnx then fails to load with a confusing error, so fold
    # everything back into ONE self-contained file.
    import onnx
    graph = onnx.load(str(out))                     # resolves the external data
    onnx.save_model(graph, str(out), save_as_external_data=False)
    sidecar = out.with_name(out.name + ".data")
    if sidecar.exists():
        sidecar.unlink()

    # parity check: the exported graph must pick the same cells as torch, or the
    # deployed agent silently plays a different policy than the one evaluated
    import onnxruntime as ort
    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    agree = 0
    for _ in range(200):
        x = (rng.random((1, 8, a.grid_h, a.grid_w)) < 0.06).astype(np.float32)
        with th.no_grad():
            t = net(th.from_numpy(x)).numpy()
        o = sess.run(None, {"obs": x})[0]
        agree += int(t.argmax() == o.argmax())
    size_mb = out.stat().st_size / 1e6
    print(f"wrote {out}  ({size_mb:.1f} MB)")
    print(f"argmax parity with torch: {agree}/200")
    return 0 if agree == 200 else 1


if __name__ == "__main__":
    raise SystemExit(main())
