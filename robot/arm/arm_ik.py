import os
import numpy as np
import math
import time

import casadi
import meshcat.geometry as mg
import pinocchio as pin
from pinocchio import casadi as cpin
from pinocchio.visualize import MeshcatVisualizer

from robot.arm.tools import (
    quaternion_from_matrix,
    create_transformation_matrix,
    matrix_to_xyzrpy,
    quaternion_from_euler,
)

VISUALIZE = False


class ArmIK:
    def __init__(self, urdf_file):
        np.set_printoptions(precision=5, suppress=True, linewidth=200)

        urdf_file = os.path.join(os.path.dirname(__file__), "models", urdf_file)
        mesh_dir = os.path.join(os.path.dirname(__file__), "models", "meshes")
        print(f"Loading URDF from {urdf_file} with mesh directory {mesh_dir}")
        self.robot = pin.RobotWrapper.BuildFromURDF(urdf_file, package_dirs=[mesh_dir])
        self.reduced_robot = self.robot.buildReducedRobot(
            list_of_joints_to_lock=["joint7", "joint8"],
            reference_configuration=np.array([0] * self.robot.model.nq),
        )

        self.last_matrix = np.dot(
            create_transformation_matrix(0, 0, 0, 0, -1.57, 0),
            create_transformation_matrix(0.13, 0.0, 0.0, 0, 0, 0),
        )
        q = quaternion_from_matrix(self.last_matrix)
        self.reduced_robot.model.addFrame(
            pin.Frame(
                "ee",
                self.reduced_robot.model.getJointId("joint6"),
                pin.SE3(
                    pin.Quaternion(q[3], q[0], q[1], q[2]),
                    np.array([
                        self.last_matrix[0, 3],
                        self.last_matrix[1, 3],
                        self.last_matrix[2, 3],
                    ]),  # -y
                ),
                pin.FrameType.OP_FRAME,
            )
        )

        self.geom_model = pin.buildGeomFromUrdf(
            self.robot.model, urdf_file, pin.GeometryType.COLLISION, package_dirs=[mesh_dir]
        )
        for i in range(4, 9):
            for j in range(0, 3):
                self.geom_model.addCollisionPair(pin.CollisionPair(i, j))
        self.geometry_data = pin.GeometryData(self.geom_model)

        self.init_data = np.zeros(self.reduced_robot.model.nq)
        self.history_data = np.zeros(self.reduced_robot.model.nq)

        # Initialize the Meshcat visualizer  for visualization
        if VISUALIZE:
            self.vis = MeshcatVisualizer(
                self.reduced_robot.model,
                self.reduced_robot.collision_model,
                self.reduced_robot.visual_model,
            )
            self.vis.initViewer(open=True)
            self.vis.loadViewerModel("pinocchio")
            self.vis.displayFrames(True, frame_ids=[113, 114], axis_length=0.15, axis_width=5)
            self.vis.display(pin.neutral(self.reduced_robot.model))

        # Enable the display of end effector target frames with short axis lengths and greater width.
        frame_viz_names = ["ee_target"]
        FRAME_AXIS_POSITIONS = (
            np.array([[0, 0, 0], [1, 0, 0], [0, 0, 0], [0, 1, 0], [0, 0, 0], [0, 0, 1]]).astype(np.float32).T
        )
        FRAME_AXIS_COLORS = (
            np.array([
                [1, 0, 0],
                [1, 0.6, 0],
                [0, 1, 0],
                [0.6, 1, 0],
                [0, 0, 1],
                [0, 0.6, 1],
            ])
            .astype(np.float32)
            .T
        )
        axis_length = 0.1
        axis_width = 10
        if VISUALIZE:
            for frame_viz_name in frame_viz_names:
                self.vis.viewer[frame_viz_name].set_object(
                    mg.LineSegments(
                        mg.PointsGeometry(
                            position=axis_length * FRAME_AXIS_POSITIONS,
                            color=FRAME_AXIS_COLORS,
                        ),
                        mg.LineBasicMaterial(
                            linewidth=axis_width,
                            vertexColors=True,
                        ),
                    )
                )

        # Creating Casadi models and data for symbolic computing
        self.cmodel = cpin.Model(self.reduced_robot.model)
        self.cdata = self.cmodel.createData()

        # Creating symbolic variables
        self.cq = casadi.SX.sym("q", self.reduced_robot.model.nq, 1)
        self.cTf = casadi.SX.sym("tf", 4, 4)
        cpin.framesForwardKinematics(self.cmodel, self.cdata, self.cq)

        # # Get the hand joint ID and define the error function
        self.gripper_id = self.reduced_robot.model.getFrameId("ee")
        self.error = casadi.Function(
            "error",
            [self.cq, self.cTf],
            [
                casadi.vertcat(
                    cpin.log6(self.cdata.oMf[self.gripper_id].inverse() * cpin.SE3(self.cTf)).vector,
                )
            ],
        )

        # Defining the optimization problem
        self.opti = casadi.Opti()
        self.var_q = self.opti.variable(self.reduced_robot.model.nq)
        # self.var_q_last = self.opti.parameter(self.reduced_robot.model.nq)   # for smooth
        self.param_tf = self.opti.parameter(4, 4)

        error_vec = self.error(self.var_q, self.param_tf)
        pos_error = error_vec[:3]
        ori_error = error_vec[3:]
        weight_position = 1.0
        weight_orientation = 0.1
        self.totalcost = casadi.sumsqr(weight_position * pos_error) + casadi.sumsqr(weight_orientation * ori_error)
        self.regularization = casadi.sumsqr(self.var_q)

        # Setting optimization constraints and goals
        self.opti.subject_to(
            self.opti.bounded(
                self.reduced_robot.model.lowerPositionLimit,
                self.var_q,
                self.reduced_robot.model.upperPositionLimit,
            )
        )
        self.opti.minimize(20 * self.totalcost + 0.01 * self.regularization)
        # self.opti.minimize(20 * self.totalcost + 0.01 * self.regularization + 0.1 * self.smooth_cost) # for smooth

        opts = {
            "ipopt": {"print_level": 0, "max_iter": 50, "tol": 1e-4},
            "print_time": False,
        }
        self.opti.solver("ipopt", opts)

    def ik_fun(self, target_pose, gripper=0, motorstate=None, motorV=None):
        gripper = np.array([gripper / 2.0, -gripper / 2.0])
        if motorstate is not None:
            self.init_data = motorstate
        self.opti.set_initial(self.var_q, self.init_data)

        if VISUALIZE:
            self.vis.viewer["ee_target"].set_transform(target_pose)  # for visualization

        self.opti.set_value(self.param_tf, target_pose)
        # self.opti.set_value(self.var_q_last, self.init_data) # for smooth

        try:
            # sol = self.opti.solve()
            _ = self.opti.solve_limited()
            sol_q = self.opti.value(self.var_q)

            if self.init_data is not None:
                max_diff = max(abs(self.history_data - sol_q))
                # print("max_diff:", max_diff)
                self.init_data = sol_q
                if max_diff > 30.0 / 180.0 * 3.1415:
                    # print("Excessive changes in joint angle:", max_diff)
                    self.init_data = np.zeros(self.reduced_robot.model.nq)
            else:
                self.init_data = sol_q
            self.history_data = sol_q

            if motorV is not None:
                v = motorV * 0.0
            else:
                v = (sol_q - self.init_data) * 0.0

            tau_ff = pin.rnea(
                self.reduced_robot.model,
                self.reduced_robot.data,
                sol_q,
                v,
                np.zeros(self.reduced_robot.model.nv),
            )

            is_collision = self.check_self_collision(sol_q, gripper)
            if VISUALIZE and not is_collision:
                self.vis.display(sol_q)

            _ = self.get_dist(sol_q, target_pose[:3, 3])
            return sol_q, tau_ff, is_collision

        except Exception as e:
            print(f"ERROR in convergence, plotting debug info.{e}")
            # sol_q = self.opti.debug.value(self.var_q)   # return original value
            return None, "", False

    def check_self_collision(self, q, gripper=np.array([0, 0])):
        pin.forwardKinematics(self.robot.model, self.robot.data, np.concatenate([q, gripper], axis=0))
        pin.updateGeometryPlacements(self.robot.model, self.robot.data, self.geom_model, self.geometry_data)
        collision = pin.computeCollisions(self.geom_model, self.geometry_data, False)
        return collision

    def get_dist(self, q, xyz):
        # print("q:", q)
        pin.forwardKinematics(
            self.reduced_robot.model,
            self.reduced_robot.data,
            np.concatenate([q], axis=0),
        )
        dist = math.sqrt(
            pow((xyz[0] - self.reduced_robot.data.oMi[6].translation[0]), 2)
            + pow((xyz[1] - self.reduced_robot.data.oMi[6].translation[1]), 2)
            + pow((xyz[2] - self.reduced_robot.data.oMi[6].translation[2]), 2)
        )
        return dist

    def get_pose(self, q):
        index = 6
        pin.forwardKinematics(
            self.reduced_robot.model,
            self.reduced_robot.data,
            np.concatenate([q], axis=0),
        )
        end_pose = create_transformation_matrix(
            self.reduced_robot.data.oMi[index].translation[0],
            self.reduced_robot.data.oMi[index].translation[1],
            self.reduced_robot.data.oMi[index].translation[2],
            math.atan2(
                self.reduced_robot.data.oMi[index].rotation[2, 1],
                self.reduced_robot.data.oMi[index].rotation[2, 2],
            ),
            math.asin(-self.reduced_robot.data.oMi[index].rotation[2, 0]),
            math.atan2(
                self.reduced_robot.data.oMi[index].rotation[1, 0],
                self.reduced_robot.data.oMi[index].rotation[0, 0],
            ),
        )
        end_pose = np.dot(end_pose, self.last_matrix)
        return matrix_to_xyzrpy(end_pose)


if __name__ == "__main__":
    arm_ik = ArmIK("piper-left.urdf")

    def solve_ik(x, y, z, roll, pitch, yaw):
        q = quaternion_from_euler(roll, pitch, yaw)
        target = pin.SE3(
            pin.Quaternion(q[3], q[0], q[1], q[2]),
            np.array([x, y, z]),
        )
        sol_q, _, is_collision = arm_ik.ik_fun(target.homogeneous, 0)
        return sol_q

    sol_q = solve_ik(0.29, 0, 0.2, 0, 0, 0)

    input("Press Enter to continue...")

    sol_q = solve_ik(0.19, 0, 0.2, 0, 0, 0)

    input("show gripper pose")

    arm_ik.vis.display(np.array([0.78, 0, 0, 0, 0, 0]))

    while True:
        time.sleep(1)
