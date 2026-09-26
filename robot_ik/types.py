"""Core data types for the robot IK solver service.

All types are immutable (frozen dataclasses) so that a configuration snapshot
can be shared safely across concurrent solves without any locking.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# Configuration model
# ---------------------------------------------------------------------------

class JointType(str, enum.Enum):
    REVOLUTE = "revolute"


@dataclass(frozen=True)
class JointSpec:
    """One revolute joint.

    The kinematic chain is defined as an ordered list of joints.  Each joint
    contributes ``T_fixed_i @ Rot(axis_i, q_i)`` to the chain, where
    ``T_fixed_i`` is the constant transform from the previous joint frame to
    this joint frame (expressed as translation + fixed rotation matrix), and
    ``axis_i`` is the rotation axis expressed in this joint's frame.
    """

    name: str
    joint_type: JointType = JointType.REVOLUTE
    axis: Tuple[float, float, float] = (0.0, 0.0, 1.0)
    translation: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    # Fixed rotation (row-major 3x3) of this joint frame relative to parent.
    rotation: Tuple[Tuple[float, float, float], ...] = (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    )
    lower_limit: float = -3.141592653589793
    upper_limit: float = 3.141592653589793


@dataclass(frozen=True)
class ArmConfig:
    """Complete kinematic configuration of one arm."""

    name: str
    version: str
    joints: Tuple[JointSpec, ...]
    # Constant tool transform applied after the last joint (row-major 4x4).
    tool_transform: Tuple[Tuple[float, float, float, float], ...] = (
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )


# ---------------------------------------------------------------------------
# Solver options
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SolverOptions:
    """Tunable numerical parameters.  All have deterministic defaults."""

    max_iterations: int = 200
    tolerance: float = 1e-6
    damping: float = 1e-3
    step_size: float = 1.0
    num_restarts: int = 8
    seed: int = 0
    timeout_ms: Optional[float] = None
    singularity_threshold: float = 1e-3


# ---------------------------------------------------------------------------
# Requests and results
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Pose:
    """Target pose: position (metres) + unit quaternion (w, x, y, z)."""

    position: Tuple[float, float, float]
    orientation_wxyz: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)


@dataclass(frozen=True)
class IKRequest:
    target: Pose
    # Optional previous solution used for continuity-aware selection.
    previous_joints: Optional[Tuple[float, ...]] = None
    # Optional per-request override of solver options (None -> service default).
    options: Optional[SolverOptions] = None


class IKStatus(str, enum.Enum):
    SUCCESS = "success"
    UNREACHABLE = "unreachable"
    JOINT_LIMIT_CONFLICT = "joint_limit_conflict"
    NOT_CONVERGED = "not_converged"
    TIMEOUT = "timeout"
    INVALID_CONFIG = "invalid_config"
    INVALID_REQUEST = "invalid_request"


@dataclass(frozen=True)
class SingularityReport:
    is_singular: bool
    is_near_singular: bool
    min_singular_value: float
    manipulability: float
    # Task-space directions (6D: linear xyz + angular xyz) whose singular
    # value falls below the configured threshold.
    degenerate_directions: Tuple[Tuple[float, ...], ...] = ()


@dataclass(frozen=True)
class IKResult:
    status: IKStatus
    joint_angles: Optional[Tuple[float, ...]]
    pose_error: Optional[float]
    iterations: int
    restarts_used: int
    singularity: Optional[SingularityReport]
    config_version: str
    message: str = ""
