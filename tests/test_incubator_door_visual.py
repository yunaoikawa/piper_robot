import cv2
import numpy as np

from rollout.incubator_door_visual import extract_feature, predict_local_delta


SETTINGS = {
    "hsv_ranges": [[[0, 35, 35], [16, 255, 255]]],
    "normalized_x_range": [0.2, 0.9],
    "normalized_y_range": [0.1, 0.9],
    "minimum_area_px": 20,
    "minimum_width_px": 4,
    "minimum_height_px": 4,
}


def test_extract_feature_is_resolution_normalized():
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    cv2.rectangle(image, (80, 40), (119, 59), (0, 0, 255), -1)
    feature, report = extract_feature(image, SETTINGS)
    assert np.allclose(feature[:2], [0.4975, 0.495])
    assert report["area_px"] == 800


def test_predict_uses_goal_minus_live_feature():
    model = {
        "goal_feature_mean": [0.5, 0.4, -6.0],
        "coefficients": np.asarray(
            [[1, 0, 0], [0, 2, 0], [0, 0, 3], [0, 0, 0]], dtype=float
        ).tolist(),
    }
    value = predict_local_delta(model, [0.4, 0.5, -6.2])
    assert np.allclose(value, [0.1, -0.2, 0.6])
