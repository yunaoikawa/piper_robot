import numpy as np

from teleop_collect_example import apply_teleop_position_safety


class _RejectingSafety:
    def __init__(self):
        self.calls = []

    def check(self, arm, target):
        self.calls.append((arm, target))
        return None


def test_normal_step_recording_bypasses_recovery_position_gate():
    safety = _RejectingSafety()
    target = np.array([0.2, -0.1, 0.4])

    result = apply_teleop_position_safety(
        mode="steps", safety=safety, arm="right", target=target
    )

    assert result is target
    assert safety.calls == []


def test_parity_recording_bypasses_recovery_position_gate():
    safety = _RejectingSafety()
    target = np.array([0.2, -0.1, 0.4])

    result = apply_teleop_position_safety(
        mode="parity", safety=safety, arm="left", target=target
    )

    assert result is target
    assert safety.calls == []


def test_recovery_mode_keeps_position_gate():
    safety = _RejectingSafety()
    target = np.array([0.2, -0.1, 0.4])

    result = apply_teleop_position_safety(
        mode="recovery", safety=safety, arm="right", target=target
    )

    assert result is None
    assert safety.calls == [("right", target)]
