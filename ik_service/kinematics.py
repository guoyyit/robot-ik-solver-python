"""Forward kinematics and geometric Jacobian.

Conventions (URDF/ROS):

* each joint carries an ``origin`` transform (xyz + fixed-axis RPY) from the
  previous link, then a screw motion about ``axis`` by the joint angle;
* fixed joints contribute no DOF, only their origin transform;
* the end-effector frame is the frame after the last joint.
"""

from __future__ import annotations

import math

import numpy as np

from .models import JointType, RobotConfig

_EYE4: Final = np.eye(4, dtype=np.float64)


def rpy_matrix(rpy: tuple[float, float, float]) -> np.ndarray:
    """Fixed-axis roll-pitch-yaw rotation ``Rz(y) Ry(p) Rx(r)``."""
    r, p, y = (float(v) for v in rpy)
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
    return rz @ ry @ rx


def origin_transform(
    xyz: tuple[float, float, float], rpy: tuple[float, float, float]
) -> np.ndarray:
    t = np.eye(4, dtype=np.float64)
    t[:3, :3] = rpy_matrix(rpy)
    t[:3, 3] = np.asarray(xyz, dtype=np.float64)
    return t


def axis_rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues rotation of ``angle`` radians about unit ``axis``."""
    x, y, z = (float(v) for v in axis)
    c, s = math.cos(angle), math.sin(angle)
    c1 = 1.0 - c
    return np.array(
        [
            [c + x * x * c1, x * y * c1 - z * s, x * z * c1 + y * s],
            [y * x * c1 + z * s, c + y * y * c1, y * z * c1 - x * s],
            [z * x * c1 - y * s, z * y * c1 + x * s, c + z * z * c1],
        ],
        dtype=np.float64,
    )


def quaternion_to_rotation(q: tuple[float, float, float, float]) -> np.ndarray:
    """Quaternion ``(w, x, y, z)`` to a 3x3 rotation matrix."""
    w, x, y, z = (float(v) for v in q)
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n == 0.0:
        raise ValueError("quaternion must not be zero-length")
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _chain_frames(
    config: RobotConfig, q: np.ndarray
) -> tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
    """Walk the chain.

    Returns ``(T_ee, joints)`` where ``joints`` holds, for every revolute
    joint in DOF order, ``(axis_in_base, origin_in_base)`` measured at the
    joint frame *after* its origin transform and *before* its own rotation.
    """
    q = np.asarray(q, dtype=np.float64)
    dof_index = 0
    t_base = np.eye(4, dtype=np.float64)
    joints: list[tuple[np.ndarray, np.ndarray]] = []
    for spec in config.joints:
        t_base = t_base @ origin_transform(spec.origin_xyz, spec.origin_rpy)
        if spec.joint_type is JointType.FIXED:
            continue
        axis = np.asarray(spec.axis, dtype=np.float64)
        axis = axis / np.linalg.norm(axis)
        axis_base = t_base[:3, :3] @ axis
        origin_base = t_base[:3, 3].copy()
        joints.append((axis_base, origin_base))
        screw = np.eye(4, dtype=np.float64)
        screw[:3, :3] = axis_rotation(axis, float(q[dof_index]))
        t_base = t_base @ screw
        dof_index += 1
    return t_base, joints


def forward_kinematics(config: RobotConfig, q: np.ndarray) -> np.ndarray:
    """End-effector pose as a 4x4 homogeneous transform in the base frame."""
    t_ee, _ = _chain_frames(config, q)
    return t_ee


def geometric_jacobian(config: RobotConfig, q: np.ndarray) -> np.ndarray:
    """Base-frame geometric Jacobian ``6 x n`` (rows: omega, v)."""
    t_ee, joints = _chain_frames(config, q)
    p_ee = t_ee[:3, 3]
    jac = np.zeros((6, len(joints)), dtype=np.float64)
    for i, (axis, origin) in enumerate(joints):
        jac[:3, i] = axis
        jac[3:, i] = np.cross(axis, p_ee - origin)
    return jac


def pose_error(
    t_current: np.ndarray, target_rotation: np.ndarray, target_position: np.ndarray
) -> np.ndarray:
    """6D pose error ``[orientation(3); position(3)]`` expressed in base frame.

    Orientation error is the canonical
    ``0.5 * vee(Rd Rc^T - Rc Rd^T)``; it vanishes exactly at ``Rc == Rd``
    and is small-angle locally.
    """
    rc = t_current[:3, :3]
    pc = t_current[:3, 3]
    skew = target_rotation @ rc.T - rc @ target_rotation.T
    e_rot = 0.5 * np.array(
        [skew[2, 1] - skew[1, 2], skew[0, 2] - skew[2, 0], skew[1, 0] - skew[0, 1]],
        dtype=np.float64,
    )
    e_pos = np.asarray(target_position, dtype=np.float64) - pc
    return np.concatenate([e_rot, e_pos])
