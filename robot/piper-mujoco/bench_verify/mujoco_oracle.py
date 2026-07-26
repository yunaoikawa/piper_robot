"""mujoco_oracle.py  --  Phase 1 bridge: your lab MuJoCo as the oracle.

Uses the already-existing lab MuJoCo scene as (a) the ground-truth scene graph
and (b) a paired RGB-D renderer, so you can drive REAL perception from the
twin and score ADD against exact ground truth. Requires `mujoco` (already a
dependency of the robot repo: mujoco>=3.2.7); everything else in this package
runs without it.

Why this matters for the MJCF-vs-CAD decision: the canonical bench is an MJCF,
so the same file is (1) the verifier's ground truth, (2) the mesh source for
FoundationPose, and (3) the renderer for controlled tests. One artifact, three
uses.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as Rot

from .scene_graph import Item, BenchState, _SP, _dec_name

try:
    import mujoco  # noqa: F401
    _HAVE_MJ = True
except Exception:  # pragma: no cover
    _HAVE_MJ = False


def _require():
    if not _HAVE_MJ:
        raise RuntimeError(
            "mujoco not available. `pip install mujoco` (the robot repo already "
            "pins mujoco>=3.2.7). Only mujoco_oracle needs it.")


def ground_truth_state(mjcf_path: str | Path, name_map: dict | None = None,
                       frame: str = "robot_base",
                       keyframe: int | str | None = None) -> BenchState:
    """World-frame ground-truth scene graph from a MuJoCo model.

    Unlike scene_graph.from_mjcf (flat XML parse), this composes the full
    kinematic tree, so it also works on nested scenes (your real lab.xml).
    Item bodies are those named "item__..." or listed in name_map.
    `keyframe` (name or index, e.g. "lab_home") is reset before reading, so
    freejoint items such as the flask report their staged pose.
    """
    _require()
    model = mujoco.MjModel.from_xml_path(str(mjcf_path))
    data = mujoco.MjData(model)
    if keyframe is not None:
        kid = (keyframe if isinstance(keyframe, int)
               else mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, keyframe))
        mujoco.mj_resetDataKeyframe(model, data, kid)
    mujoco.mj_forward(model, data)

    items: list[Item] = []
    for b in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b)
        if not name:
            continue
        if name_map and name in name_map:
            item_id, label, kind, container = name_map[name]
        elif name.startswith("item" + _SP):
            item_id, label, kind, container = _dec_name(name)
        else:
            continue
        t = np.array(data.xpos[b])               # world position
        R = np.array(data.xmat[b]).reshape(3, 3)  # world rotation
        items.append(Item(item_id, label, kind, container, t, R, 1.0))
    return BenchState(Path(mjcf_path).stem, items, frame=frame,
                      captured_by="mujoco_oracle")


def render_rgbd(mjcf_path: str | Path, camera: str | int = -1,
                width: int = 640, height: int = 480
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Render (rgb uint8 HxWx3, depth float HxW [m], K 3x3, cam_pose 4x4 world).

    Feed rgb/depth/K to a real perception backend to obtain an observed
    BenchState, then SceneVerifier.verify(...) against ground_truth_state.
    """
    _require()
    model = mujoco.MjModel.from_xml_path(str(mjcf_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    renderer = mujoco.Renderer(model, height=height, width=width)
    renderer.update_scene(data, camera=camera)
    rgb = renderer.render().copy()
    renderer.enable_depth_rendering()
    renderer.update_scene(data, camera=camera)
    depth = renderer.render().copy()
    renderer.disable_depth_rendering()

    # Intrinsics from vertical FOV
    cam_id = (camera if isinstance(camera, int)
              else mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera))
    fovy = np.deg2rad(model.cam_fovy[cam_id]) if cam_id >= 0 else np.deg2rad(45.0)
    f = (height / 2) / np.tan(fovy / 2)
    K = np.array([[f, 0, width / 2], [0, f, height / 2], [0, 0, 1]], float)

    cam_pose = np.eye(4)
    if cam_id >= 0:
        cam_pose[:3, 3] = data.cam_xpos[cam_id]
        cam_pose[:3, :3] = np.array(data.cam_xmat[cam_id]).reshape(3, 3)
    return rgb, depth, K, cam_pose


def perturb_to_mjcf(state: BenchState, out_path: str | Path, *,
                    frame_R=None, frame_t=None, move=None,
                    drop: list[str] | None = None) -> Path:
    """Write a controlled variant of a scene to MJCF for real-camera tests:
    apply a global (frame_R, frame_t), displace specific labels (move={label:
    dxyz}), and/or drop labels. Lets you stage ok/frame/moved/missing cases on
    the real bench with known ground truth."""
    from .scene_graph import to_mjcf
    frame_R = np.eye(3) if frame_R is None else np.asarray(frame_R)
    frame_t = np.zeros(3) if frame_t is None else np.asarray(frame_t)
    move = move or {}
    drop = set(drop or [])
    items = []
    for it in state.items:
        if it.label in drop:
            continue
        t = frame_R @ it.t + frame_t + np.asarray(move.get(it.label, 0.0))
        items.append(Item(it.item_id, it.label, it.kind, it.container,
                          t, frame_R @ it.R, it.confidence))
    return to_mjcf(BenchState(state.bench_id + "_perturbed", items), out_path)
