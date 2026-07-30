from pathlib import Path

import cv2
import numpy as np

from rollout.wrist_rgbd_target import detect_tool_relative_blue_cross


def test_tool_relative_cross_chooses_nearest_cross_without_pixel_roi():
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    depth = np.ones((480, 640), dtype=float)
    # Largest blue component is the gripper/tool reference.
    cv2.rectangle(image, (330, 300), (560, 410), (255, 0, 0), -1)
    # Two cross-shaped candidates; the nearer one is the intended marker.
    cv2.rectangle(image, (270, 246), (310, 258), (255, 0, 0), -1)
    cv2.rectangle(image, (284, 230), (296, 274), (255, 0, 0), -1)
    cv2.rectangle(image, (50, 86), (90, 98), (255, 0, 0), -1)
    cv2.rectangle(image, (64, 70), (76, 114), (255, 0, 0), -1)
    camera_matrix = np.array(
        [[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]]
    )
    profile = {
        "minimum_tool_area_fraction": 0.02,
        "minimum_marker_area_fraction": 0.0001,
        "maximum_marker_area_fraction": 0.01,
        "minimum_cross_score": 0.65,
        "maximum_normalized_tool_distance": 0.5,
        "minimum_depth_sample_fraction": 0.00001,
    }

    selected, overlay = detect_tool_relative_blue_cross(
        image,
        depth,
        camera_matrix,
        profile,
    )

    np.testing.assert_allclose(selected["center_px"], [290, 252], atol=2)
    np.testing.assert_allclose(
        selected["point_camera_m"],
        [-0.06, 0.024, 1.0],
        atol=0.006,
    )
    assert overlay.shape == image.shape
    assert "roi" not in selected


def test_wrist_pipeline_source_is_observation_only():
    root = Path(__file__).resolve().parents[1]
    source = (root / "rollout/wrist_rgbd_target.py").read_text()
    forbidden = ("PiperClient", "robot_rpc", "send_joint", "move_arm")
    assert not any(token in source for token in forbidden)
