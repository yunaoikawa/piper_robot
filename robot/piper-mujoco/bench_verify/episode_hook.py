"""episode_hook.py  --  wire SceneVerifier into rollout/episode.py.

The robot repo already has a primitive EN-module: EpisodeManager auto-starts,
resets arms to home, checks manipulability, and times out. This adds the two
missing pieces ENPIRE's EN-module needs: metric success verification and a
reset target, both grounded in the canonical MJCF instead of a vision
heuristic.

Perception is injected as a callable, so the same hook runs in three modes
without code change:
  * MJCFOracleBackend   - phase 1, sim/oracle ground truth (debug/CI).
  * RemotePoseBackend   - phase 2, FoundationPose on peacock05 over the tunnel.
  * any  fn(rgb, depth, K) -> BenchState.

Wiring into rollout/episode.py (EpisodeManager.end_episode):

    from bench_verify.episode_hook import EpisodeSceneVerifier, RemotePoseBackend
    from bench_verify.scene_graph import from_mjcf

    # in PolicyController.__init__:
    self.scene_check = EpisodeSceneVerifier(
        canonical=from_mjcf("bench_canonical.xml"),
        perception=RemotePoseBackend(host="localhost", port=15558),
        frame_offset="fail")          # single bolted bench -> reject big shifts

    # in EpisodeManager.end_episode(...), after recording stops:
    rgb, ts, depth = self.controller.camera.get_latest_frame()
    K = self.controller.camera_intrinsics            # from Record3D
    res = self.scene_check.check(rgb, depth, K)
    self.recorder.tag_episode(success=res.success, reason=res.reason())
    if not res.success and self.autonomous_mode:
        goals = self.scene_check.reset_goals(res)    # canonical in obs frame
        self.robot_rpc.execute_reset(goals)          # your reset policy

`check` returns the same VerifyResult used in validate.py, so the hardware
loop logs exactly the false_success / false_reset rates measured in Phase 0.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from .scene_graph import BenchState, reground, transfer_diff
from .verify import SceneVerifier, VerifyResult

PerceptionFn = Callable[[np.ndarray, np.ndarray, np.ndarray], BenchState]


class EpisodeSceneVerifier:
    """Glue: perception -> observed BenchState -> verify -> success + reset."""

    def __init__(self, canonical: BenchState, perception: PerceptionFn,
                 **verifier_kw):
        self.canonical = canonical
        self.perception = perception
        self.verifier = SceneVerifier(canonical, **verifier_kw)
        self.last_observed: BenchState | None = None

    def check(self, rgb, depth, K) -> VerifyResult:
        self.last_observed = self.perception(rgb, depth, K)
        return self.verifier.verify(self.last_observed)

    def reset_goals(self, result: VerifyResult | None = None) -> BenchState:
        """Per-object 6-DoF goals (canonical mapped into the current frame).
        For "moved"/"missing" the affected items are restored to canonical."""
        obs = self.last_observed
        if obs is None:
            return self.canonical
        d = transfer_diff(self.canonical, obs)
        return reground(self.canonical, d)


# --------------------------------------------------------------------------- #
# Perception backends
# --------------------------------------------------------------------------- #
class MJCFOracleBackend:
    """Phase-1 backend: returns ground truth from a (possibly perturbed) MJCF.
    Use for CI / debugging the loop without cameras. rgb/depth/K are ignored.
    """
    def __init__(self, mjcf_path: str, name_map: dict | None = None):
        from .mujoco_oracle import ground_truth_state
        self._state = ground_truth_state(mjcf_path, name_map=name_map)

    def __call__(self, rgb=None, depth=None, K=None) -> BenchState:
        return self._state


class RemotePoseBackend:
    """Phase-2 backend: send RGB-D + intrinsics to the FoundationPose server on
    peacock05 (via the SSH tunnel), receive per-object 6-DoF, build BenchState.
    Mirrors the repo's existing ZMQ obs/action client pattern.
    """
    def __init__(self, host: str = "localhost", port: int = 15558,
                 timeout_ms: int = 4000, items_meta: list[dict] | None = None):
        import zmq
        self.items_meta = items_meta or []  # [{label,kind,container,mask?}, ...]
        ctx = zmq.Context.instance()
        self.sock = ctx.socket(zmq.REQ)
        self.sock.setsockopt(zmq.RCVTIMEO, timeout_ms)
        self.sock.setsockopt(zmq.SNDTIMEO, timeout_ms)
        self.sock.setsockopt(zmq.LINGER, 0)
        self.sock.connect(f"tcp://{host}:{port}")

    def __call__(self, rgb, depth, K) -> BenchState:
        from .scene_graph import Item
        self.sock.send_pyobj({"rgb": rgb, "depth": depth, "K": K,
                              "items": self.items_meta})
        rep = self.sock.recv_pyobj()  # {label: {"t":(3,), "R":(3,3), "conf":f}}
        meta = {m["label"]: m for m in self.items_meta}
        items = []
        for label, p in rep.items():
            m = meta.get(label, {})
            items.append(Item(m.get("item_id", label), label,
                              m.get("kind", "Labware"),
                              m.get("container", "unknown"),
                              np.asarray(p["t"]), np.asarray(p["R"]),
                              float(p.get("conf", 1.0))))
        return BenchState("observed", items, captured_by="foundationpose")
