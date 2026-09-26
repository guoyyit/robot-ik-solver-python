"""Forward kinematics and geometric Jacobian for serial revolute chains.

Chain convention: for joints i = 0..n-1,

    T(q) = T_fixed_0 @ Rot(axis_0, q_0) @ ... @ T_fixed_n-1 @ Rot(axis_n-1, q_n-1) @ T_tool

where T_fixed_i combines the joint's constant translation and rotation, and
Rot(axis, q) is a rotation of angle q about the joint axis expressed in the
joint frame.
"""

from __future__ import annotations

import numpy as np

from .types import ArmConfig


def rotation_about_axis(axis: np.ndarray, angle: float) -> np.ndarray:
    """3x3 rotation matrix from a unit axis and an angle (Rodrigues)."""
    x, y, z = axis
    c = np.cos(angle)
    s = np.sin(angle)
    C = 1.0 - c
    return np.array(
        [
            [x * x * C + c, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, y * y * C + c, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, z * z * C + c],
        ]
    )


def quaternion_to_matrix(q_wxyz: np.ndarray) -> np.ndarray:
    """Convert a unit quaternion (w, x, y, z) to a 3x3 rotation matrix."""
    w, x, y, z = q_wxyz / np.linalg.norm(q_wxyz)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def pose_error(current: np.ndarray, target: np.ndarray) -> np.ndarray:
    """6D pose error (linear xyz, angular xyz) driving current -> target.

    The angular part is the rotation vector of R_err = R_target @ R_current^T
    expressed in the base frame.
    """
    dp = target[:3, 3] - current[:3, 3]
    r_err = target[:3, :3] @ current[:3, :3].T
    # Rotation vector via the numerically stable log map.
    cos_angle = (np.trace(r_err) - 1.0) / 2.0
    cos_angle = min(1.0, max(-1.0, cos_angle))
    angle = np.arccos(cos_angle)
    if angle < 1e-9:
        drot = np.array(
            [r_err[2, 1] - r_err[1, 2], r_err[0, 2] - r_err[2, 0], r_err[1, 0] - r_err[0, 1]]
        ) * 0.5
    elif angle > np.pi - 1e-6:
        # Near 180 degrees: extract axis from the symmetric part.
        sym = (r_err + np.eye(3)) * 0.5
        axis = np.array([sym[0, 0], sym[1, 1], sym[2, 2]])
        axis = np.sqrt(np.maximum(axis, 0.0))
        # Fix signs using off-diagonal terms.
        if r_err[2, 1] - r_err[1, 2] < 0:
            axis[0] = -axis[0]
        if r_err[0, 2] - r_err[2, 0] < 0:
            axis[1] = -axis[1]
        if r_err[1, 0] - r_err[0, 1] < 0:
            axis[2] = -axis[2]
        n = np.linalg.norm(axis)
        axis = axis / n if n > 1e-12 else np.array([1.0, 0.0, 0.0])
        drot = axis * angle
    else:
        drot = (
            angle
            / (2.0 * np.sin(angle))
            * np.array(
                [
                    r_err[2, 1] - r_err[1, 2],
                    r_err[0, 2] - r_err[2, 0],
                    r_err[1, 0] - r_err[0, 1],
                ]
            )
        )
    err = np.empty(6)
    err[:3] = dp
    err[3:] = drot
    return err


class KinematicModel:
    """Precomputed, immutable kinematic model built from an ArmConfig."""

    def __init__(self, config: ArmConfig) -> None:
        self._dof = len(config.joints)
        self._axes = []
        self._fixed = []
        for j in config.joints:
            axis = np.asarray(j.axis, dtype=float)
            self._axes.append(axis / np.linalg.norm(axis))
            t = np.eye(4)
            t[:3, :3] = np.asarray(j.rotation, dtype=float)
            t[:3, 3] = np.asarray(j.translation, dtype=float)
            self._fixed.append(t)
        self._axes = tuple(self._axes)
        self._fixed = tuple(self._fixed)
        self._tool = np.asarray(config.tool_transform, dtype=float)
        self.lower_limits = np.array([j.lower_limit for j in config.joints])
        self.upper_limits = np.array([j.upper_limit for j in config.joints])
        reach = sum(float(np.linalg.norm(t[:3, 3])) for t in self._fixed)
        self._max_reach = reach + float(np.linalg.norm(self._tool[:3, 3]))

    @property
    def dof(self) -> int:
        return self._dof

    def max_reach(self) -> float:
        return self._max_reach

    def forward_kinematics(self, q: np.ndarray) -> np.ndarray:
        """Return the 4x4 end-effector transform for joint vector ``q``."""
        t = np.eye(4)
        for i in range(self._dof):
            rot = np.eye(4)
            rot[:3, :3] = rotation_about_axis(self._axes[i], q[i])
            t = t @ self._fixed[i] @ rot
        return t @ self._tool

    def jacobian(self, q: np.ndarray) -> np.ndarray:
        """Return the 6 x n geometric Jacobian (linear xyz over angular xyz)."""
        # Forward pass: record each joint frame (before its rotation).
        origins = []
        axes_base = []
        t = np.eye(4)
        for i in range(self._dof):
            t = t @ self._fixed[i]
            origins.append(t[:3, 3].copy())
            axes_base.append(t[:3, :3] @ self._axes[i])
            rot = np.eye(4)
            rot[:3, :3] = rotation_about_axis(self._axes[i], q[i])
            t = t @ rot
        t = t @ self._tool
        p_ee = t[:3, 3]

        jac = np.zeros((6, self._dof))
        for i in range(self._dof):
            a = axes_base[i]
            jac[:3, i] = np.cross(a, p_ee - origins[i])
            jac[3:, i] = a
        return jac
