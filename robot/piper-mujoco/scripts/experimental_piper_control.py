#!/usr/bin/env python3
"""
experimental_piper_control.py

MuJoCo 上で Piper を「まず表示して少し動かす」ための実験コード。
mujoco_menagerie/agilex_piper と Piper_mujoco の両方を意識した最小構成。

使い方例:
  # mujoco_menagerie 側
  python experimental_piper_control.py \
      --xml /path/to/mujoco_menagerie/agilex_piper/scene.xml \
      --mode sin

  # Piper_mujoco 側
  python experimental_piper_control.py \
      --xml /path/to/Piper_mujoco/assets/Piper/scene.xml \
      --mode hold

macOS で launch_passive を使う場合は mjpython 推奨:
  mjpython experimental_piper_control.py --xml ...
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import mujoco
import numpy as np


@dataclass
class RobotSpec:
    joint_ids: List[int]
    actuator_ids: List[int]
    joint_names: List[str]
    actuator_names: List[str]


def find_existing_names(
    model: mujoco.MjModel,
    candidates: Sequence[str],
    kind: str,
) -> List[Tuple[str, int]]:
    """候補名のうち model に存在するものを返す。"""
    found: List[Tuple[str, int]] = []

    for name in candidates:
        try:
            if kind == "joint":
                idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            elif kind == "actuator":
                idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            else:
                raise ValueError(f"unknown kind: {kind}")

            if idx != -1:
                found.append((name, idx))
        except Exception:
            pass

    return found


def discover_piper_spec(model: mujoco.MjModel) -> RobotSpec:
    """
    PiPER っぽい joint / actuator を推定する。
    まず既知の名前を試し、足りなければ hinge joint / actuator を前から拾う。
    """
    joint_candidates = [
        "joint 1", "joint 2", "joint 3", "joint 4", "joint 5", "joint 6",
        "joint1", "joint2", "joint3", "joint4", "joint5", "joint6",
        "piper_joint_1", "piper_joint_2", "piper_joint_3",
        "piper_joint_4", "piper_joint_5", "piper_joint_6",
    ]
    actuator_candidates = [
        "actuator 1", "actuator 2", "actuator 3", "actuator 4", "actuator 5", "actuator 6",
        "actuator1", "actuator2", "actuator3", "actuator4", "actuator5", "actuator6",
        "piper_actuator_1", "piper_actuator_2", "piper_actuator_3",
        "piper_actuator_4", "piper_actuator_5", "piper_actuator_6",
    ]

    found_joints = find_existing_names(model, joint_candidates, "joint")
    found_actuators = find_existing_names(model, actuator_candidates, "actuator")

    if len(found_joints) >= 6 and len(found_actuators) >= 6:
        found_joints = found_joints[:6]
        found_actuators = found_actuators[:6]
        return RobotSpec(
            joint_ids=[idx for _, idx in found_joints],
            actuator_ids=[idx for _, idx in found_actuators],
            joint_names=[name for name, _ in found_joints],
            actuator_names=[name for name, _ in found_actuators],
        )

    # fallback: model 全体から拾う
    hinge_joint_ids: List[int] = []
    hinge_joint_names: List[str] = []
    for j in range(model.njnt):
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE:
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
            if name is not None:
                hinge_joint_ids.append(j)
                hinge_joint_names.append(name)

    actuator_ids: List[int] = []
    actuator_names: List[str] = []
    for a in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a)
        if name is not None:
            actuator_ids.append(a)
            actuator_names.append(name)

    if len(hinge_joint_ids) < 6 or len(actuator_ids) < 6:
        raise RuntimeError(
            "6軸アームとして扱える joint / actuator を自動発見できませんでした。"
        )

    return RobotSpec(
        joint_ids=hinge_joint_ids[:6],
        actuator_ids=actuator_ids[:6],
        joint_names=hinge_joint_names[:6],
        actuator_names=actuator_names[:6],
    )


def joint_qpos_indices(model: mujoco.MjModel, joint_ids: Sequence[int]) -> List[int]:
    """各 joint が対応する qpos の先頭 index を返す。"""
    return [int(model.jnt_qposadr[j]) for j in joint_ids]


def joint_range_center_and_halfwidth(
    model: mujoco.MjModel,
    joint_ids: Sequence[int],
) -> Tuple[np.ndarray, np.ndarray]:
    centers = []
    halfwidths = []

    for j in joint_ids:
        low, high = model.jnt_range[j]
        if low < high:
            centers.append(0.5 * (low + high))
            halfwidths.append(0.45 * (high - low))
        else:
            # range 未設定なら控えめな既定値
            centers.append(0.0)
            halfwidths.append(0.6)

    return np.array(centers, dtype=float), np.array(halfwidths, dtype=float)


def clamp_to_joint_ranges(
    model: mujoco.MjModel,
    joint_ids: Sequence[int],
    q: np.ndarray,
) -> np.ndarray:
    out = q.copy()
    for i, j in enumerate(joint_ids):
        low, high = model.jnt_range[j]
        if low < high:
            out[i] = np.clip(out[i], low, high)
    return out


def set_position_targets(data: mujoco.MjData, actuator_ids: Sequence[int], targets: np.ndarray) -> None:
    for a_id, value in zip(actuator_ids, targets):
        data.ctrl[a_id] = float(value)


def make_mode_target(
    mode: str,
    t: float,
    q_home: np.ndarray,
    q_center: np.ndarray,
    q_halfwidth: np.ndarray,
) -> np.ndarray:
    if mode == "hold":
        return q_home.copy()

    if mode == "sin":
        # ゆっくり全関節を少しずつ動かす
        target = q_center.copy()
        freqs = np.array([0.20, 0.17, 0.23, 0.27, 0.19, 0.31], dtype=float)
        gains = np.array([0.25, 0.20, 0.18, 0.25, 0.35, 0.40], dtype=float)
        target += q_halfwidth * gains * np.sin(2.0 * np.pi * freqs * t)
        return target

    if mode == "ee-circle":
        # 厳密 IK ではなく、手先が円を描くっぽい joint-space パターン
        target = q_home.copy()
        target[0] += 0.35 * math.sin(0.6 * t)
        target[1] += 0.25 * math.sin(0.6 * t + math.pi / 2.0)
        target[2] += 0.20 * math.sin(0.6 * t)
        target[3] += 0.15 * math.cos(0.6 * t)
        target[4] += 0.20 * math.sin(0.6 * t + math.pi / 3.0)
        target[5] += 0.25 * math.cos(0.6 * t)
        return target

    raise ValueError(f"unknown mode: {mode}")


def print_robot_summary(model: mujoco.MjModel, spec: RobotSpec, qpos_ids: Sequence[int]) -> None:
    print("=== Piper candidate summary ===")
    print("joints:")
    for name, j, qidx in zip(spec.joint_names, spec.joint_ids, qpos_ids):
        rng = model.jnt_range[j]
        print(f"  - {name:20s} id={j:2d} qpos={qidx:2d} range=({rng[0]: .3f}, {rng[1]: .3f})")
    print("actuators:")
    for name, a in zip(spec.actuator_names, spec.actuator_ids):
        print(f"  - {name:20s} id={a:2d}")
    print("===============================")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Experimental MuJoCo controller for PiPER"
    )
    parser.add_argument("--xml", type=str, required=True, help="scene.xml or piper.xml path")
    parser.add_argument(
        "--mode",
        type=str,
        default="sin",
        choices=["hold", "sin", "ee-circle"],
        help="control mode",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="0以下なら無限ループ。秒数を指定するとその時間で終了",
    )
    parser.add_argument(
        "--print-only",
        action="store_true",
        help="モデルの joint / actuator を表示して終了",
    )
    parser.add_argument(
        "--save-traj",
        type=str,
        default=None,
        metavar="PATH",
        help="軌跡を .npz 形式で保存するパス（例: traj.npz）。指定しない場合は保存しない",
    )
    parser.add_argument(
        "--traj-every",
        type=int,
        default=1,
        metavar="N",
        help="N ステップごとに軌跡を記録する（デフォルト: 1 = 毎ステップ）",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    xml_path = Path(args.xml)

    if not xml_path.exists():
        print(f"[ERROR] XML not found: {xml_path}", file=sys.stderr)
        return 1

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)

    spec = discover_piper_spec(model)
    qpos_ids = joint_qpos_indices(model, spec.joint_ids)
    q_center, q_halfwidth = joint_range_center_and_halfwidth(model, spec.joint_ids)

    # 初期姿勢をホーム姿勢として採用
    mujoco.mj_resetData(model, data)
    q_home = np.array([data.qpos[qidx] for qidx in qpos_ids], dtype=float)

    print_robot_summary(model, spec, qpos_ids)

    if args.print_only:
        return 0

    # 一度 forward しておく
    mujoco.mj_forward(model, data)

    # 軌跡バッファ
    traj_time: List[float] = []
    traj_qpos: List[np.ndarray] = []
    traj_ctrl: List[np.ndarray] = []

    # ヘッドレスシミュレーションループ
    # duration が 0 以下の場合は 10 秒をデフォルトとする
    sim_duration = args.duration if args.duration > 0.0 else 10.0
    total_steps = int(sim_duration / model.opt.timestep)
    print(f"[sim] headless mode: {sim_duration:.1f}s / {total_steps} steps (dt={model.opt.timestep:.4f})")

    for traj_step in range(total_steps):
        t = data.time

        target = make_mode_target(
            mode=args.mode,
            t=t,
            q_home=q_home,
            q_center=q_center,
            q_halfwidth=q_halfwidth,
        )
        target = clamp_to_joint_ranges(model, spec.joint_ids, target)

        set_position_targets(data, spec.actuator_ids, target)
        mujoco.mj_step(model, data)

        if args.save_traj is not None and traj_step % args.traj_every == 0:
            traj_time.append(data.time)
            traj_qpos.append(np.array([data.qpos[i] for i in qpos_ids], dtype=np.float32))
            traj_ctrl.append(np.array([data.ctrl[a] for a in spec.actuator_ids], dtype=np.float32))

    print(f"[sim] done.")

    # 軌跡を保存
    if args.save_traj is not None and len(traj_time) > 0:
        save_path = Path(args.save_traj)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            save_path,
            time=np.array(traj_time, dtype=np.float32),
            qpos=np.stack(traj_qpos),   # shape: (T, 6)
            ctrl=np.stack(traj_ctrl),   # shape: (T, 6)
            joint_names=np.array(spec.joint_names),
            actuator_names=np.array(spec.actuator_names),
        )
        print(f"[traj] saved {len(traj_time)} steps -> {save_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())