"""pose_server_peacock.py  --  Phase-2 GPU pose server (peacock05).

Mirrors the repo's cloud inference pattern: a ZMQ REP server on the GPU box;
the robot (pasteur) reaches it over an SSH tunnel (add one alongside the
existing obs/action tunnels):

    ssh -f -N -L 15558:localhost:8558 peacock05

Robot request  : {"rgb": HxWx3 uint8, "depth": HxW float [m], "K": 3x3,
                  "items": [{"label","container","mask"(opt)}, ...]}
Server reply   : {label: {"t": (3,), "R": (3,3), "conf": float}}

FoundationPose plugs in at FoundationPoseBackend.estimate(); meshes are the
ones referenced by the canonical MJCF (scene_graph.MESH_FILES), so the same
CAD assets serve verify, sim, and pose. Until the GPU backend is wired, the
server falls back to an identity-pose stub so the full loop is testable.

Run on peacock05:
    python -m bench_verify.pose_server_peacock --port 8558 --mesh-dir assets/
"""
from __future__ import annotations

import argparse

import numpy as np


class FoundationPoseBackend:
    """Wrap FoundationPose. Loads one mesh per container from mesh_dir and runs
    model-based 6-DoF pose from RGB-D + mask. Replace estimate() body with the
    NVlabs/FoundationPose call; signature is already what the server expects.
    """
    def __init__(self, mesh_dir: str, device: str = "cuda"):
        self.mesh_dir = mesh_dir
        self.device = device
        # PLUG: self.estimator = FoundationPose(...); preload meshes per container

    def estimate(self, rgb, depth, K, item) -> dict:
        # PLUG: pose = self.estimator.register(rgb, depth, K, mesh[item.container],
        #                                      mask=item.get("mask"))
        # return {"t": pose[:3,3], "R": pose[:3,:3], "conf": score}
        return {"t": np.zeros(3), "R": np.eye(3), "conf": 0.0}  # identity stub


def serve(port: int, mesh_dir: str, device: str = "cuda") -> None:
    import zmq
    backend = FoundationPoseBackend(mesh_dir, device=device)
    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.REP)
    sock.bind(f"tcp://*:{port}")
    print(f"FoundationPose server on tcp://*:{port}  device={device}  "
          f"mesh_dir={mesh_dir}", flush=True)
    while True:
        try:
            req = sock.recv_pyobj()
            rgb, depth, K = req["rgb"], req["depth"], req["K"]
            reply = {}
            for item in req.get("items", []):
                p = backend.estimate(rgb, depth, K, item)
                reply[item["label"]] = {"t": np.asarray(p["t"]),
                                        "R": np.asarray(p["R"]),
                                        "conf": float(p["conf"])}
            sock.send_pyobj(reply)
        except KeyboardInterrupt:
            break
        except Exception as e:  # keep the server alive
            print(f"pose server error: {e}", flush=True)
            try:
                sock.send_pyobj({"error": str(e)})
            except Exception:
                pass
    print("pose server stopped")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8558)
    ap.add_argument("--mesh-dir", default="assets")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    serve(args.port, args.mesh_dir, device=args.device)


if __name__ == "__main__":
    main()
