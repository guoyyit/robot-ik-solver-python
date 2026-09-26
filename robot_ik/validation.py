"""Configuration and request validation.

All checks are pure functions of their inputs: no state, no randomness, so
validation itself is deterministic and safe to run from any thread.
"""

from __future__ import annotations

import math

import numpy as np

from .types import ArmConfig, IKRequest, JointType, SolverOptions

_ROT_TOL = 1e-6


class ConfigError(ValueError):
    """Raised when an arm configuration is inconsistent or invalid."""


class RequestError(ValueError):
    """Raised when an IK request is malformed."""


def _check_finite_vec(values, what: str) -> None:
    arr = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(arr)):
        raise ConfigError(f"{what} must contain only finite numbers")


def _check_rotation_matrix(rot, what: str) -> None:
    r = np.asarray(rot, dtype=float)
    if r.shape != (3, 3):
        raise ConfigError(f"{what} must be a 3x3 matrix, got shape {r.shape}")
    if not np.all(np.isfinite(r)):
        raise ConfigError(f"{what} must contain only finite numbers")
    if not np.allclose(r @ r.T, np.eye(3), atol=_ROT_TOL):
        raise ConfigError(f"{what} is not orthonormal")
    if abs(np.linalg.det(r) - 1.0) > _ROT_TOL:
        raise ConfigError(f"{what} has determinant != +1 (not a rotation)")


def validate_config(config: ArmConfig) -> None:
    """Validate an arm configuration, raising ConfigError on any problem."""
    if not isinstance(config.name, str) or not config.name:
        raise ConfigError("config name must be a non-empty string")
    if not isinstance(config.version, str) or not config.version:
        raise ConfigError("config version must be a non-empty string")
    if len(config.joints) == 0:
        raise ConfigError("config must contain at least one joint")

    seen = set()
    for i, j in enumerate(config.joints):
        what = f"joint[{i}] ({j.name!r})"
        if not j.name:
            raise ConfigError(f"joint[{i}] has an empty name")
        if j.name in seen:
            raise ConfigError(f"duplicate joint name {j.name!r}")
        seen.add(j.name)
        if j.joint_type is not JointType.REVOLUTE:
            raise ConfigError(f"{what}: unsupported joint type {j.joint_type!r}")

        axis = np.asarray(j.axis, dtype=float)
        if axis.shape != (3,):
            raise ConfigError(f"{what}: axis must have 3 components")
        _check_finite_vec(axis, f"{what}: axis")
        if np.linalg.norm(axis) < 1e-9:
            raise ConfigError(f"{what}: axis must be non-zero")

        t = np.asarray(j.translation, dtype=float)
        if t.shape != (3,):
            raise ConfigError(f"{what}: translation must have 3 components")
        _check_finite_vec(t, f"{what}: translation")

        _check_rotation_matrix(j.rotation, f"{what}: rotation")

        if not (math.isfinite(j.lower_limit) and math.isfinite(j.upper_limit)):
            raise ConfigError(f"{what}: joint limits must be finite")
        if not j.lower_limit < j.upper_limit:
            raise ConfigError(
                f"{what}: lower limit {j.lower_limit} must be < upper limit {j.upper_limit}"
            )

    tool = np.asarray(config.tool_transform, dtype=float)
    if tool.shape != (4, 4):
        raise ConfigError(f"tool_transform must be 4x4, got shape {tool.shape}")
    if not np.all(np.isfinite(tool)):
        raise ConfigError("tool_transform must contain only finite numbers")
    if not np.allclose(tool[3, :], [0.0, 0.0, 0.0, 1.0], atol=_ROT_TOL):
        raise ConfigError("tool_transform last row must be [0, 0, 0, 1]")
    _check_rotation_matrix(tool[:3, :3], "tool_transform rotation")


def validate_options(options: SolverOptions) -> None:
    """Validate solver options, raising RequestError on any problem."""
    if options.max_iterations <= 0:
        raise RequestError("max_iterations must be positive")
    if not (options.tolerance > 0.0) or not math.isfinite(options.tolerance):
        raise RequestError("tolerance must be a positive finite number")
    if not (options.damping >= 0.0) or not math.isfinite(options.damping):
        raise RequestError("damping must be a non-negative finite number")
    if not (0.0 < options.step_size <= 1.0):
        raise RequestError("step_size must be in (0, 1]")
    if options.num_restarts < 0:
        raise RequestError("num_restarts must be >= 0")
    if options.timeout_ms is not None and not (
        math.isfinite(options.timeout_ms) and options.timeout_ms > 0.0
    ):
        raise RequestError("timeout_ms must be a positive finite number or None")
    if not (options.singularity_threshold > 0.0) or not math.isfinite(
        options.singularity_threshold
    ):
        raise RequestError("singularity_threshold must be a positive finite number")


def validate_request(request: IKRequest, dof: int) -> None:
    """Validate a solve request against the arm's degrees of freedom."""
    pos = np.asarray(request.target.position, dtype=float)
    if pos.shape != (3,) or not np.all(np.isfinite(pos)):
        raise RequestError("target position must be 3 finite numbers")
    quat = np.asarray(request.target.orientation_wxyz, dtype=float)
    if quat.shape != (4,) or not np.all(np.isfinite(quat)):
        raise RequestError("target orientation must be 4 finite numbers (w, x, y, z)")
    if np.linalg.norm(quat) < 1e-9:
        raise RequestError("target orientation quaternion must be non-zero")

    if request.previous_joints is not None:
        prev = np.asarray(request.previous_joints, dtype=float)
        if prev.shape != (dof,):
            raise RequestError(
                f"previous_joints must have {dof} entries, got shape {prev.shape}"
            )
        if not np.all(np.isfinite(prev)):
            raise RequestError("previous_joints must contain only finite numbers")

    if request.options is not None:
        validate_options(request.options)
