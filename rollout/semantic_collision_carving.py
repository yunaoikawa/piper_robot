"""Carve robot-contaminated semantic collision voxels at a proven home pose.

SAM can label a background object and a foreground robot as one region when
their silhouettes touch.  The RGB-D alignment stage separates their depth
ordering, but an older semantic collision mesh may still contain voxels copied
from the robot surface.  This module removes only voxel bodies that:

1. belong to an explicitly allow-listed observed semantic object,
2. penetrate exact robot CAD at the accepted home keyframe, and
3. are backed by an accepted depth-aware alignment report.

The observed visual mesh is preserved.  Inferred solid volumes, supports, and
unlisted objects are never carved.
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET


SCHEMA = "piper_robot.semantic_robot_contamination_carve/v1"


def _contacts(
    model_path: Path,
    keyframe: str,
    *,
    qpos_by_joint: dict[str, float] | None = None,
    robot_body_prefixes: tuple[str, ...] = ("left/", "right/"),
    robot_clearance_margin_m: float = 0.0,
) -> list[dict]:
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    if robot_clearance_margin_m > 0:
        for geom_id in range(model.ngeom):
            body_name = model.body(model.geom_bodyid[geom_id]).name
            if body_name.startswith(robot_body_prefixes):
                model.geom_margin[geom_id] = max(
                    float(model.geom_margin[geom_id]),
                    float(robot_clearance_margin_m),
                )
    key_id = int(model.key(keyframe).id)
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    for joint_name, value in (qpos_by_joint or {}).items():
        data.qpos[int(model.joint(joint_name).qposadr[0])] = float(value)
    mujoco.mj_forward(model, data)
    records = []
    for contact in data.contact:
        first_body = model.body(model.geom_bodyid[contact.geom1]).name
        second_body = model.body(model.geom_bodyid[contact.geom2]).name
        first_robot = first_body.startswith(robot_body_prefixes)
        second_robot = second_body.startswith(robot_body_prefixes)
        if first_robot == second_robot:
            continue
        robot_body = first_body if first_robot else second_body
        environment_body = second_body if first_robot else first_body
        records.append(
            {
                "robot_body": robot_body,
                "environment_body": environment_body,
                "penetration_depth_m": max(0.0, -float(contact.dist)),
                "position_xyz_m": [float(value) for value in contact.pos],
            }
        )
    return sorted(
        records,
        key=lambda item: item["penetration_depth_m"],
        reverse=True,
    )


def _allowed_body(name: str, prefixes: tuple[str, ...]) -> bool:
    return any(name.startswith(prefix) for prefix in prefixes)


def carve_robot_contamination(
    source_model: str | Path,
    alignment_report_path: str | Path,
    output_model: str | Path,
    report_path: str | Path,
    *,
    allowed_body_prefixes: tuple[str, ...],
    keyframe: str = "home",
    maximum_removed_fraction: float = 0.30,
    robot_body_prefixes: tuple[str, ...] = ("left/", "right/"),
    verified_qpos_by_name: dict[str, dict[str, float]] | None = None,
    robot_clearance_margin_m: float = 0.0,
) -> dict:
    """Create a derived scene with home-overlapping semantic voxels removed."""

    source_model = Path(source_model).resolve()
    alignment_report_path = Path(alignment_report_path).resolve()
    output_model = Path(output_model).resolve()
    report_path = Path(report_path).resolve()
    if not allowed_body_prefixes:
        raise ValueError("at least one semantic voxel body prefix is required")
    alignment = json.loads(alignment_report_path.read_text())
    prerequisites = {
        "alignment_accepted": alignment.get("accepted") is True,
        "depth_persistence_accepted": alignment.get(
            "persistent_depth_robot_fit", {}
        ).get("accepted") is True,
        "home_pose_accepted": alignment.get(
            "home_pose_provenance", {}
        ).get("accepted") is True,
        "commands_sent_false": alignment.get("commands_sent") is False,
    }
    if not all(prerequisites.values()):
        failed = [name for name, value in prerequisites.items() if not value]
        raise ValueError(
            "collision carving prerequisites failed: " + ", ".join(failed)
        )
    verified_qpos_by_name = verified_qpos_by_name or {}
    pose_inputs = {"home": None, **verified_qpos_by_name}
    before_by_pose = {
        name: _contacts(
            source_model,
            keyframe,
            qpos_by_joint=values,
            robot_body_prefixes=robot_body_prefixes,
            robot_clearance_margin_m=robot_clearance_margin_m,
        )
        for name, values in pose_inputs.items()
    }
    before = [
        {**item, "verified_pose": name}
        for name, contacts in before_by_pose.items()
        for item in contacts
    ]
    disallowed_contacts = [
        item for item in before
        if not _allowed_body(
            item["environment_body"],
            allowed_body_prefixes,
        )
    ]
    tree = ET.parse(source_model)
    root = tree.getroot()
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("scene model has no worldbody")
    direct_bodies = {
        body.get("name"): body
        for body in worldbody.findall("body")
        if body.get("name")
    }
    eligible_bodies = {
        name for name in direct_bodies
        if _allowed_body(name, allowed_body_prefixes)
    }
    contacted_bodies = {
        item["environment_body"] for item in before
        if _allowed_body(
            item["environment_body"],
            allowed_body_prefixes,
        )
    }
    missing = sorted(contacted_bodies - set(direct_bodies))
    if missing:
        raise ValueError(
            "contacted semantic voxels are not direct scene bodies: "
            + ", ".join(missing)
        )
    removed = []
    for name in sorted(contacted_bodies):
        body = direct_bodies[name]
        geom = body.find("geom")
        if geom is None or geom.get("type") != "box":
            raise ValueError(f"refusing to carve non-box semantic body {name}")
        removed.append(
            {
                "body": name,
                "position_xyz_m": [
                    float(value)
                    for value in body.get("pos", "0 0 0").split()
                ],
                "size_half_xyz_m": [
                    float(value)
                    for value in geom.get("size", "0 0 0").split()
                ],
            }
        )
        worldbody.remove(body)
    removed_fraction = (
        len(removed) / len(eligible_bodies)
        if eligible_bodies
        else float("inf")
    )
    output_model.parent.mkdir(parents=True, exist_ok=True)
    if removed_fraction > maximum_removed_fraction:
        shutil.copy2(source_model, output_model)
        after_by_pose = before_by_pose
        after = before
        accepted = False
        reasons = ["removed_semantic_fraction_exceeds_limit"]
    else:
        tree.write(output_model, encoding="unicode")
        after_by_pose = {
            name: _contacts(
                output_model,
                keyframe,
                qpos_by_joint=values,
                robot_body_prefixes=robot_body_prefixes,
                robot_clearance_margin_m=0.0,
            )
            for name, values in pose_inputs.items()
        }
        after = [
            {**item, "verified_pose": name}
            for name, contacts in after_by_pose.items()
            for item in contacts
        ]
        reasons = []
        if disallowed_contacts:
            reasons.append("non_allowlisted_home_contacts_present")
        if after:
            reasons.append("home_robot_environment_contacts_remain")
        accepted = not reasons
    report = {
        "schema": SCHEMA,
        "accepted": accepted,
        "commands_sent": False,
        "observation_only": True,
        "source_model": str(source_model),
        "output_model": str(output_model),
        "alignment_report": str(alignment_report_path),
        "keyframe": keyframe,
        "robot_body_prefixes": list(robot_body_prefixes),
        "robot_clearance_margin_m": robot_clearance_margin_m,
        "verified_pose_names": list(pose_inputs),
        "allowed_body_prefixes": list(allowed_body_prefixes),
        "prerequisites": prerequisites,
        "before_contacts": before,
        "before_contacts_by_pose": before_by_pose,
        "after_contacts": after,
        "after_contacts_by_pose": after_by_pose,
        "removed_voxel_bodies": removed,
        "eligible_voxel_body_count": len(eligible_bodies),
        "removed_fraction": removed_fraction,
        "maximum_removed_fraction": maximum_removed_fraction,
        "visual_mesh_preserved": True,
        "inferred_solid_volumes_modified": False,
        "support_geometry_modified": False,
        "reasons": reasons,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    )
    return report
