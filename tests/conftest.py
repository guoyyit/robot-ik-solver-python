import numpy as np
import pytest

from robot_ik import ArmConfig, JointSpec, KinematicModel, SolverOptions


def mat_to_quat_wxyz(R: np.ndarray):
    """Rotation matrix -> quaternion (w, x, y, z), deterministic branch."""
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        return (
            0.25 * s,
            (R[2, 1] - R[1, 2]) / s,
            (R[0, 2] - R[2, 0]) / s,
            (R[1, 0] - R[0, 1]) / s,
        )
    i = int(np.argmax(np.diag(R)))
    if i == 0:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        return (
            (R[2, 1] - R[1, 2]) / s,
            0.25 * s,
            (R[0, 1] + R[1, 0]) / s,
            (R[0, 2] + R[2, 0]) / s,
        )
    if i == 1:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        return (
            (R[0, 2] - R[2, 0]) / s,
            (R[0, 1] + R[1, 0]) / s,
            0.25 * s,
            (R[1, 2] + R[2, 1]) / s,
        )
    s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
    return (
        (R[1, 0] - R[0, 1]) / s,
        (R[0, 2] + R[2, 0]) / s,
        (R[1, 2] + R[2, 1]) / s,
        0.25 * s,
    )


def pose_from_transform(T: np.ndarray):
    from robot_ik import Pose

    return Pose(
        position=tuple(float(x) for x in T[:3, 3]),
        orientation_wxyz=tuple(float(x) for x in mat_to_quat_wxyz(T[:3, :3])),
    )


ARM6_JOINTS = (
    JointSpec(name="j1", axis=(0, 0, 1), translation=(0, 0, 0.2)),
    JointSpec(name="j2", axis=(0, 1, 0), translation=(0, 0, 0.1)),
    JointSpec(name="j3", axis=(0, 1, 0), translation=(0.3, 0, 0)),
    JointSpec(name="j4", axis=(1, 0, 0), translation=(0.2, 0, 0)),
    JointSpec(name="j5", axis=(0, 1, 0), translation=(0.1, 0, 0)),
    JointSpec(name="j6", axis=(1, 0, 0), translation=(0.05, 0, 0)),
)


@pytest.fixture()
def arm6_config() -> ArmConfig:
    return ArmConfig(name="arm6", version="v1", joints=ARM6_JOINTS)


@pytest.fixture()
def arm6_model(arm6_config) -> KinematicModel:
    return KinematicModel(arm6_config)


@pytest.fixture()
def options() -> SolverOptions:
    return SolverOptions(seed=7, num_restarts=8)


@pytest.fixture()
def planar2_config() -> ArmConfig:
    return ArmConfig(
        name="planar2",
        version="v1",
        joints=(
            JointSpec(name="j1", axis=(0, 0, 1)),
            JointSpec(name="j2", axis=(0, 0, 1), translation=(1.0, 0, 0)),
        ),
    )
