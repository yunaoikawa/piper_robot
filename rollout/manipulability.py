"""Manipulability calculation using Jacobian."""

import numpy as np


class ManipulabilityCalculator:
    """Calculates robot manipulability from Jacobian."""

    def __init__(self, robot_rpc, rpc_lock):
        """
        Initialize manipulability calculator.

        Args:
            robot_rpc: RPC client for robot communication
            rpc_lock: Threading lock to protect RPC access
        """
        self.robot_rpc = robot_rpc
        self.rpc_lock = rpc_lock

    def calculate(self, arm='right'):
        """
        Calculate manipulability score using the volume of the manipulability ellipsoid.
        Manipulability = sqrt(det(J * J^T)) where J is the Jacobian matrix.

        Args:
            arm: 'left' or 'right' arm to calculate manipulability for

        Returns:
            float: Manipulability measure (higher is better, units: m³ for 6-DOF arm)
        """
        try:
            # Get Jacobian from robot via RPC (thread-safe)
            with self.rpc_lock:
                if arm == 'right':
                    J = self.robot_rpc.get_right_jacobian()
                else:
                    J = self.robot_rpc.get_left_jacobian()

            # Calculate manipulability from Jacobian: sqrt(det(J * J^T))
            # Use only position part (first 3 rows) for translational manipulability
            J_pos = J[:3, :]  # 3 x n_joints
            JJT = J_pos @ J_pos.T  # 3 x 3
            det_JJT = np.linalg.det(JJT)

            # Handle numerical issues
            if det_JJT < 0:
                det_JJT = 0

            manipulability = np.sqrt(det_JJT)

            return manipulability

        except Exception as e:
            # On error, return moderate score to avoid crashing
            print(f"Warning: Error calculating manipulability: {e}")
            return 0.05