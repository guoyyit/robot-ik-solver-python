"""Domain types for the robot inverse-kinematics service.

All types are immutable data containers. Validation lives in
:mod:`ik_service.config`; the numerical solver consumes these types only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Final


class JointType(str, Enum):
    """Supported joint kinds. Only revolute joints are solved numerically."""

    REVOLUTE = "revolute"
    FIXED = "fixed"


class FailureReason(str, Enum):
    """Distinguishable failure diagnoses returned to callers.

    No approximate/partial joint answer is ever returned together with one
    of these reasons.
    """

    INVALID_CONFIG = "invalid_config"
    INVALID_REQUEST = "invalid_request"
    UNREACHABLE = "unreachable"
    JOINT_LIMIT_CONFLICT = "joint_limit_conflict"
    SINGULAR = "singular"
    DID_NOT_CONVERGE = "did_not_converge"
    TIMEOUT = "timeout"
    VERSION_MISMATCH = "version_mismatch"


@dataclass(frozen=True)
class JointSpec:
    """One joint/link of the kinematic chain.

    The link transform ``T(q) = T(origin) * exp(axis_screw * q)`` follows the
    URDF/ROS convention: ``axis`` is expressed in the joint's own (post-origin)
    frame and ``origin`` is the transform from the previous link.
    """

    name: str
    joint_type: JointType = JointType.REVOLUTE
    axis: tuple[float, float, float] = (0.0, 0.0, 1.0)
    origin_rpy: tuple[float, float, float] = (0.0, 0.0, 0.0)
    origin_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    lower_limit: float = -3.141592653589793
    upper_limit: float = 3.141592653589793


@dataclass(frozen=True)
class RobotConfig:
    """Immutable, validated robot description plus solver settings."""

    name: str
    joints: tuple[JointSpec, ...]
    singular_threshold: float = 1e-4
    near_singular_threshold: float = 1e-2
    position_tolerance: float = 1e-5
    orientation_tolerance: float = 1e-5
    max_iterations: int = 200
    timeout_seconds: float = 5.0
    restarts: int = 8
    damping: float = 1e-4
    random_seed: int = 0
    version: int = 1


@dataclass(frozen=True)
class Pose:
    """Target/current end-effector pose: 3D position + quaternion (w, x, y, z)."""

    position: tuple[float, float, float]
    quaternion_wxyz: tuple[float, float, float, float]


@dataclass(frozen=True)
class IKRequest:
    target: Pose
    previous_joints: tuple[float, ...] | None = None
    expected_version: int | None = None
    seed_override: int | None = None
    timeout_override: float | None = None


@dataclass(frozen=True)
class SingularityReport:
    """Result of the manipulability analysis at a candidate."""

    is_singular: bool
    is_near_singular: bool
    smallest_singular_value: float
    manipulability: float
    affected_direction: tuple[float, float, float, float, float, float] = (
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    )


@dataclass(frozen=True)
class IKResult:
    """Solver outcome. On success ``joints`` is populated; on failure it is None."""

    success: bool
    joints: tuple[float, ...] | None
    reason: FailureReason | None
    message: str
    singularity: SingularityReport | None = None
    iterations: int = 0
    restarts_used: int = 0
    config_version: int = 0
    residual_position: float = 0.0
    residual_orientation: float = 0.0
