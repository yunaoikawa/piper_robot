import json
import subprocess

import numpy as np
from scipy.spatial.transform import Rotation

from rollout.appliance_frame import enroll_local_tag
from src.run_incubator_door_autonomy import _residual_door_yaw_deg


def _transform(x=0, y=0, z=0, yaw=0):
    value = np.eye(4)
    value[:3, :3] = Rotation.from_euler("z", yaw, degrees=True).as_matrix()
    value[:3, 3] = [x, y, z]
    return value


def _enrollment(appliance, size=(0.4, 0.3, 0.5)):
    return {
        "schema": "piper_robot.appliance_frame_enrollment/v1",
        "accepted": True,
        "motion_authority": True,
        "appliance_semantic_name": "incubator",
        "local_tag": None,
        "T_robot_appliance_at_enrollment": appliance.tolist(),
        "evidence": {"size_xyz_m": list(size)},
    }


def test_registration_cli_uses_arbitrary_current_lab_tag_placement(tmp_path):
    reference_pose = _transform(0.1, 0.2, 0.3, 5)
    current_pose = _transform(0.2, 0.1, 0.3, 15)
    reference = _enrollment(reference_pose)
    current = _enrollment(current_pose)
    local_tag_pose = _transform(-0.13, 0.07, 0.18, 71)
    current.update(
        enroll_local_tag(
            current_pose,
            current_pose @ local_tag_pose,
            tag_id=88,
            appliance_semantic_name="incubator",
        )
    )
    current["motion_authority"] = True
    current["evidence"] = {"size_xyz_m": [0.4, 0.3, 0.5]}
    reference_path = tmp_path / "reference.json"
    current_path = tmp_path / "current.json"
    observation_path = tmp_path / "tag.json"
    output_path = tmp_path / "registration.json"
    reference_path.write_text(json.dumps(reference))
    current_path.write_text(json.dumps(current))
    observation_path.write_text(
        json.dumps(
            {
                "tag_id": 88,
                "T_robot_tag": (current_pose @ local_tag_pose).tolist(),
            }
        )
    )
    result = subprocess.run(
        [
            "/home/admin/miniforge3/envs/robot-test/bin/python",
            "src/prepare_appliance_registration.py",
            "--reference-enrollment",
            str(reference_path),
            "--current-enrollment",
            str(current_path),
            "--current-tag-observation",
            str(observation_path),
            "--output",
            str(output_path),
        ],
        check=False,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(output_path.read_text())
    assert payload["accepted"] is True
    assert payload["pose_source"] == "lab_local_tag_tracking"
    np.testing.assert_allclose(
        np.asarray(payload["T_registration"]) @ reference_pose,
        current_pose,
        atol=1e-8,
    )


def test_live_plane_yaw_is_only_residual_after_full_registration():
    assert _residual_door_yaw_deg(-5.0, -10.0, None) == 5.0
    assert _residual_door_yaw_deg(-5.0, -10.0, {"yaw_deg": 4.25}) == 0.75
