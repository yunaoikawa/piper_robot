"""LabEnv — headless MuJoCo twin of the wet-lab bench.

Wraps robot/piper-mujoco/xml/lab-scene.xml (two 5-DOF Piper arms with coupled-jaw
grippers, a table, an incubator, and a freejointed flask) and exposes the motor
primitives WetRobo's skills are built from:

    reset()                          -> lab_home keyframe
    move_ee(arm, pos, quat)          -> Cartesian move via ArmIK + position actuators
    set_gripper(arm, open_ratio)     -> 1.0 fully open .. 0.0 fully closed
    get_object_pose(body)            -> (pos, quat_wxyz) of any body, live
    render(camera)                   -> RGB (+ optional depth) for perception

Arms discovered from the model: right = joint1..5 / site "ee" / actuator "gripper";
left = left_joint1..5 / site "left_ee" / actuator "left_gripper". The gripper actuator
range is 0..0.99 with 0.99 = fully open (per the scene's lab_home ctrl), so open_ratio
maps as ctrl = open_ratio * jaw_open.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import mujoco
import mink

from wetrobo._paths import LAB_SCENE_XML
from wetrobo.sim.ik import ArmIK


# Gripper approach orientation for every reach.
#
# A top-down (vertical) wrist is NOT achievable by this 5-DOF arm at the bench's reach
# distances — the 6th wrist joint is welded, and commanding straight-down makes the free
# wrist drift to an inconsistent 60-79° tilt that varies pose-to-pose (looks like it's
# flailing). Instead we command ONE achievable orientation: top-down pitched -50° about
# world Y — a shallow side-approach whose jaws close along world-Y, straddling the
# vertical flask neck. The arm holds this consistently at both the pick and the place
# (measured Δtilt ~1°), so the grasp reads as a deliberate, repeatable side-pinch rather
# than a cocked wrist. Documented sim reality, not a tuned constant (see CLAUDE.md).
_APPROACH_PITCH = np.radians(-50.0)
APPROACH_QUAT = mink.SO3.from_rpy_radians(0.0, _APPROACH_PITCH, 0.0).multiply(
    mink.SO3(np.array([0.0, 1.0, 0.0, 0.0]))).wxyz


@dataclass
class ArmSpec:
    name: str
    joint_names: list[str]
    ee_site: str
    arm_actuators: list[str]
    gripper_actuator: str
    pad_sites: tuple = ()          # (upper, lower) inner sponge contact sites


def _right_spec() -> "ArmSpec":
    return ArmSpec(
        "right",
        [f"joint{i}" for i in range(1, 6)],
        "grasp",
        [f"joint{i}" for i in range(1, 6)],
        "gripper",
        ("pad_upper", "pad_lower"),
    )


def _left_spec() -> "ArmSpec":
    return ArmSpec(
        "left",
        [f"left_joint{i}" for i in range(1, 6)],
        "left_grasp",
        [f"left_joint{i}" for i in range(1, 6)],
        "left_gripper",
        ("left_pad_upper", "left_pad_lower"),
    )


class LabEnv:
    def __init__(self, xml_path: str | None = None, keyframe: str = "lab_home"):
        self.xml_path = str(xml_path or LAB_SCENE_XML)
        self.model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.data = mujoco.MjData(self.model)
        self.keyframe = keyframe
        self._key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, keyframe)

        self.arms: dict[str, ArmSpec] = {"right": _right_spec(), "left": _left_spec()}
        self.ik: dict[str, ArmIK] = {
            a.name: ArmIK(self.model, a.joint_names, a.ee_site, a.arm_actuators)
            for a in self.arms.values()
        }
        self._grip_act = {
            a.name: self.model.actuator(a.gripper_actuator).id for a in self.arms.values()
        }
        # jaw-open ctrl value = upper bound of the gripper actuator ctrlrange.
        self._jaw_open = {
            a.name: float(self.model.actuator(a.gripper_actuator).ctrlrange[1])
            for a in self.arms.values()
        }
        self._grasp_site_id = {a.name: self.model.site(a.ee_site).id for a in self.arms.values()}
        # Inner sponge-pad site ids per arm (only the pads grip); empty if the arm's
        # gripper defines none, in which case grasp() falls back to the midpoint gate.
        self._pad_site_id: dict[str, tuple] = {}
        for a in self.arms.values():
            ids = tuple(self.model.site(s).id for s in a.pad_sites
                        if mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, s) >= 0)
            self._pad_site_id[a.name] = ids if len(ids) == 2 else ()
        self._held: dict[str, dict] = {}   # arm -> {body, T_rel, qadr, vadr}
        self._frozen: dict[str, dict] = {}  # body -> {qpos7, qadr, vadr}
        self._stiffen_arms()
        self._renderer: mujoco.Renderer | None = None
        self.reset()

    def _stiffen_arms(self, kp: float = 900.0, kv: float = 90.0) -> None:
        """Raise the arm position-actuator gains so IK joint targets are tracked with
        little steady-state error. The scene ships low gains (kp 10-80) tuned to the
        real robot's compliance; for kinematic sim control we want stiff tracking. Only
        the arm actuators are touched (grippers keep their gains)."""
        for spec in self.arms.values():
            for name in spec.arm_actuators:
                aid = self.model.actuator(name).id
                self.model.actuator_gainprm[aid, 0] = kp
                self.model.actuator_biasprm[aid, 1] = -kp
                self.model.actuator_biasprm[aid, 2] = -kv

    # --- pose helpers ------------------------------------------------------ #
    def _site_T(self, arm: str) -> np.ndarray:
        sid = self._grasp_site_id[arm]
        T = np.eye(4)
        T[:3, :3] = self.data.site_xmat[sid].reshape(3, 3)
        T[:3, 3] = self.data.site_xpos[sid]
        return T

    def _body_T(self, body: str) -> np.ndarray:
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body)
        T = np.eye(4)
        T[:3, :3] = self.data.xmat[bid].reshape(3, 3)
        T[:3, 3] = self.data.xpos[bid]
        return T

    def _freejoint_addr(self, body: str) -> tuple[int, int]:
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body)
        jid = self.model.body_jntadr[bid]
        return int(self.model.jnt_qposadr[jid]), int(self.model.jnt_dofadr[jid])

    # --- lifecycle --------------------------------------------------------- #
    def reset(self) -> None:
        mujoco.mj_resetDataKeyframe(self.model, self.data, self._key_id)
        mujoco.mj_forward(self.model, self.data)

    def home_arm(self, arm: str, settle_steps: int = 120) -> None:
        """Return one arm's joints to the lab_home keyframe pose (leaves other bodies,
        e.g. the flask, untouched) — a clean starting pose between retry attempts."""
        key = self.model.key(self.keyframe)
        dof = self.ik[arm].dof_ids
        adr = self.ik[arm].qpos_adr
        self.data.qpos[adr] = key.qpos[adr]
        self.data.ctrl[self.ik[arm].actuator_ids] = key.qpos[adr]
        mujoco.mj_forward(self.model, self.data)
        self.step(settle_steps)

    def step(self, n: int = 1) -> None:
        for _ in range(n):
            mujoco.mj_step(self.model, self.data)
            self._enforce()

    def _enforce(self) -> None:
        """Pin frozen and held objects after each physics step.

        Frozen: a bench object stays exactly where it is (models "objects don't move
        until grasped") so a clumsy reach cannot shove the target before the grasp
        gate is evaluated. Held: the object rides the gripper's grasp frame at the
        relative transform captured at grasp time (kinematic carry, not a friction
        grasp). Release/unfreeze stops enforcement and physics resumes."""
        for body, f in self._frozen.items():
            qa, va = f["qadr"], f["vadr"]
            self.data.qpos[qa:qa + 7] = f["qpos7"]
            self.data.qvel[va:va + 6] = 0.0
        for arm, h in self._held.items():
            gs = self.data.site_xpos[self._grasp_site_id[arm]]
            qa, va = h["qadr"], h["vadr"]
            # Carry the object upright, directly beneath the grasp site by the vertical
            # offset captured at grasp time -> placement is predictable regardless of
            # the wrist's (uncontrollable, 5-DOF) orientation.
            self.data.qpos[qa:qa + 3] = [gs[0], gs[1], gs[2] - h["carry_dz"]]
            self.data.qpos[qa + 3:qa + 7] = [1.0, 0.0, 0.0, 0.0]
            self.data.qvel[va:va + 6] = 0.0

    def freeze(self, body: str) -> None:
        qa, va = self._freejoint_addr(body)
        self._frozen[body] = {"qpos7": self.data.qpos[qa:qa + 7].copy(), "qadr": qa, "vadr": va}

    def unfreeze(self, body: str) -> None:
        self._frozen.pop(body, None)

    # --- observation ------------------------------------------------------- #
    def get_object_pose(self, body: str) -> tuple[np.ndarray, np.ndarray]:
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body)
        if bid < 0:
            raise KeyError(f"no body named {body!r}")
        pos = self.data.xpos[bid].copy()
        quat = self.data.xquat[bid].copy()  # wxyz
        return pos, quat

    def get_ee_pose(self, arm: str) -> mink.SE3:
        return self.ik[arm].forward(self.data.qpos)

    def gripper_opening(self, arm: str) -> float:
        """Current commanded open ratio (0 closed .. 1 open)."""
        return float(self.data.ctrl[self._grip_act[arm]] / self._jaw_open[arm])

    # --- primitives -------------------------------------------------------- #
    def set_gripper(self, arm: str, open_ratio: float, settle_steps: int = 60) -> None:
        open_ratio = float(np.clip(open_ratio, 0.0, 1.0))
        self.data.ctrl[self._grip_act[arm]] = open_ratio * self._jaw_open[arm]
        self.step(settle_steps)

    def grasp(self, arm: str, body: str, grasp_local=(0.0, 0.0, 0.0), tol: float = 0.035,
              pad_len: float = 0.018) -> dict:
        """Close the gripper and attach the object IFF its true grasp point actually
        lies between the two inner sponge pads (position-gated kinematic grasp).

        Only the sponge pads grip: after the jaws close, the object's grasp point must
        project onto the segment between the pad sites (``|along| <= tol``, so it is
        centred in the jaw gap, not caught on a tip) and sit within ``pad_len`` of that
        axis (so the pad, not the housing, contacts it). Because the gate is measured in
        the *gripper's* frame, an off-axis or badly-oriented approach — e.g. a tilted
        wrist whose horizontal pinch misses the vertical neck — fails, while an accurate
        square approach grips. Falls back to a midpoint-distance gate if the arm defines
        no pad sites. ``grasp_local`` is the grasp point in the object body frame."""
        T_obj = self._body_T(body)
        gp_world = (T_obj @ np.array([*grasp_local, 1.0]))[:3]
        self.set_gripper(arm, 0.0)
        pads = self._pad_site_id.get(arm, ())
        if pads:
            pu = self.data.site_xpos[pads[0]].copy()
            pl = self.data.site_xpos[pads[1]].copy()
            mid = 0.5 * (pu + pl)
            axis = pu - pl
            n = np.linalg.norm(axis)
            axis = axis / n if n > 1e-9 else np.array([0.0, 1.0, 0.0])
            v = gp_world - mid
            along = float(abs(np.dot(v, axis)))          # offset along the closing axis
            perp = float(np.linalg.norm(v - np.dot(v, axis) * axis))  # off the pad face
            dist = float(np.linalg.norm(v))
            grasped = along <= tol and perp <= pad_len
            gate = {"along_m": along, "perp_m": perp}
        else:
            gs = self.data.site_xpos[self._grasp_site_id[arm]].copy()
            dist = float(np.linalg.norm(gs - gp_world))
            grasped = dist <= tol
            gate = {}
        if grasped:
            gs = self.data.site_xpos[self._grasp_site_id[arm]].copy()
            qa, va = self._freejoint_addr(body)
            carry_dz = float(gs[2] - T_obj[:3, 3][2])  # grasp-site height above object origin
            geoms = self._body_geoms(body)
            saved = [(self.model.geom_contype[g], self.model.geom_conaffinity[g]) for g in geoms]
            for g in geoms:                          # kinematic carry -> no contacts
                self.model.geom_contype[g] = 0
                self.model.geom_conaffinity[g] = 0
            self._held[arm] = {"body": body, "carry_dz": carry_dz, "qadr": qa, "vadr": va,
                               "geoms": geoms, "saved_col": saved}
            return {"grasped": True, "dist_m": dist, **gate}
        return {"grasped": False, "dist_m": dist, **gate}

    def _body_geoms(self, body: str) -> list[int]:
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body)
        return [g for g in range(self.model.ngeom) if self.model.geom_bodyid[g] == bid]

    def release(self, arm: str, settle_steps: int = 200) -> None:
        h = self._held.pop(arm, None)
        if h is not None:                            # restore the object's collisions
            for g, (ct, ca) in zip(h["geoms"], h["saved_col"]):
                self.model.geom_contype[g] = ct
                self.model.geom_conaffinity[g] = ca
        self.set_gripper(arm, 1.0, settle_steps=0)
        self.step(settle_steps)

    def is_holding(self, arm: str) -> bool:
        return arm in self._held

    def move_ee(
        self,
        arm: str,
        pos,
        quat=None,
        pos_tol: float = 0.012,
        step_len: float = 0.01,
        steps_per_wp: int = 25,
        settle_steps: int = 100,
        max_iter_ik: int = 120,
        n_correct: int = 6,
    ) -> dict:
        """Drive the arm's grasp point to a world pose.

        Two mechanisms make this reliable on the 5-DOF arm with low-gain position
        actuators: (1) the Cartesian path is walked in ``step_len`` increments so the
        arm tracks smoothly instead of jumping (a jump sweeps objects aside); (2) an
        integral outer loop cancels the steady-state error from the weak wrist
        actuator sagging under load — after settling we measure the grasp-point error
        and add it to the commanded target, so the arm converges onto ``pos`` without
        modifying the model's physics gains. Holds the current gripper command."""
        pos = np.asarray(pos, float)
        quat = APPROACH_QUAT if quat is None else np.asarray(quat, float)
        rot = mink.SO3(quat)
        act_ids = self.ik[arm].actuator_ids

        # Phase 1: smooth interpolated approach toward the raw target.
        start = self.get_ee_pose(arm).translation()
        dist = float(np.linalg.norm(pos - start))
        n_wp = max(1, int(np.ceil(dist / step_len)))
        last = None
        for i in range(1, n_wp + 1):
            wp = start + (pos - start) * (i / n_wp)
            last = self.ik[arm].solve(
                mink.SE3.from_rotation_and_translation(rot, wp),
                self.data.qpos, max_iter=max_iter_ik)
            self.data.ctrl[act_ids] = last.q
            self.step(steps_per_wp)
        self.step(settle_steps)

        # Phase 2: integral correction of steady-state error, with divergence guards
        # (clamp each correction, keep the best config, stop if the error grows) so a
        # near-boundary target cannot drive the arm into a flailing config.
        cmd = pos.copy()
        ee = self.get_ee_pose(arm).translation()
        best_err = float(np.linalg.norm(pos - ee))
        best_q = self.data.qpos[self.ik[arm].qpos_adr].copy()
        for _ in range(n_correct):
            err = pos - ee
            if np.linalg.norm(err) <= pos_tol:
                break
            step = np.clip(err, -0.05, 0.05)          # bound each correction to 5 cm
            cmd = cmd + step
            last = self.ik[arm].solve(
                mink.SE3.from_rotation_and_translation(rot, cmd),
                self.data.qpos, max_iter=max_iter_ik)
            self.data.ctrl[act_ids] = last.q
            self.step(steps_per_wp + settle_steps)
            ee = self.get_ee_pose(arm).translation()
            e = float(np.linalg.norm(pos - ee))
            if e < best_err:
                best_err, best_q = e, last.q.copy()
            elif e > best_err + 0.01:                 # diverging -> revert to best
                self.data.ctrl[act_ids] = best_q
                self.step(settle_steps)
                ee = self.get_ee_pose(arm).translation()
                break

        return {
            "arm": arm,
            "target": pos,
            "ik_pos_err_m": last.pos_err_m if last else float("nan"),
            "ik_rot_err_deg": last.rot_err_deg if last else float("nan"),
            "ik_solved": last.solved if last else False,
            "reached": bool(np.linalg.norm(ee - pos) <= pos_tol),
            "ee_pos": ee,
            "ee_err_m": float(np.linalg.norm(ee - pos)),
        }

    # --- rendering (for the no-CAD perception baseline) -------------------- #
    def render(
        self, camera: str = "topdown", width: int = 640, height: int = 480, depth: bool = False
    ):
        if self._renderer is None or self._renderer.width != width or self._renderer.height != height:
            self._renderer = mujoco.Renderer(self.model, height=height, width=width)
        self._renderer.disable_depth_rendering()
        self._renderer.update_scene(self.data, camera=camera)
        rgb = self._renderer.render().copy()
        if not depth:
            return rgb
        self._renderer.enable_depth_rendering()
        self._renderer.update_scene(self.data, camera=camera)
        d = self._renderer.render().copy()
        self._renderer.disable_depth_rendering()
        return rgb, d
