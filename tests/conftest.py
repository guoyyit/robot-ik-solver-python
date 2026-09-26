"""Shared fixtures: deterministic planar-3R and Puma-like 6R robots."""

from __future__ import annotations

import math

import numpy as np
import pytest

from ik_service.config import ConfigStore
from ik_service.kinematics import forward_kinematics
from ik_service.models import JointSpec, JointType, Pose, RobotConfig
from ik_service.service import IKService


def planar_3r(limits: tuple[float, float] = (-math.pi, math.pi), **overrides) -> RobotConfig:
    lo, hi = limits
    joints = (
        JointSpec(name="j0", origin_xyz=(0.0, 0.0, 0.0), lower_limit=lo, upper_limit=hi),
        JointSpec(name="j1", origin_xyz=(1.0, 0.0, 0.0), lower_limit=lo, upper_limit=hi),
        JointSpec(name="j2", origin_xyz=(1.0, 0.0, 0.0), lower_limit=lo, upper_limit=hi),
    )
    defaults = dict(name="planar-3r", restarts=8, timeout_seconds=10.0, random_seed=0)
    defaults.update(overrides)
    return RobotConfig(joints=joints, **defaults)


def puma_6r(**overrides) -> RobotConfig:
    joints = (
        JointSpec(name="base", origin_xyz=(0.0, 0.0, 0.0)),
        JointSpec(name="shoulder", origin_rpy=(0.0, math.pi / 2, 0.0)),
        JointSpec(name="elbow", origin_xyz=(0.3, 0.0, 0.0)),
        JointSpec(name="wrist1", origin_xyz=(0.3, 0.0, 0.0), origin_rpy=(0.0, -math.pi / 2, 0.0)),
        JointSpec(name="wrist2", origin_xyz=(0.0, 0.0, 0.1), origin_rpy=(0.0, -math.pi / 2, 0.0)),
        JointSpec(name="wrist3", origin_xyz=(0.0, 0.0, 0.05), origin_rpy=(0.0, math.pi / 2, 0.0)),
    )
    defaults = dict(
        name="puma-6r",
        restarts=12,
        timeout_seconds=20.0,
        random_seed=0,
        singular_threshold=1e-3,
        near_singular_threshold=5e-2,
    )
    defaults.update(overrides)
    return RobotConfig(joints=joints, **defaults)


def rot_to_quat(rotation: np.ndarray) -> tuple[float, float, float, float]:
    """Robust rotation-matrix -> (w, x, y, z) for test targets."""
    ang = math.acos(float(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0)))
    if ang < 1e-12:
        return (1.0, 0.0, 0.0, 0.0)
    axis = np.array(
        [
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ]
    ) / (2.0 * math.sin(ang))
    half = ang / 2.0
    return (math.cos(half), *(float(v) * math.sin(half) for v in axis))


def pose_of(config: RobotConfig, q: np.ndarray) -> Pose:
    t = forward_kinematics(config, q)
    return Pose(
        position=tuple(float(v) for v in t[:3, 3]),
        quaternion_wxyz=rot_to_quat(t[:3, :3]),
    )


def planar_pose(x: float, y: float, z: float = 0.0) -> Pose:
    return Pose(position=(float(x), float(y), float(z)), quaternion_wxyz=(1.0, 0.0, 0.0, 0.0))


@pytest.fixture
def planar_config() -> RobotConfig:
    return planar_3r()


@pytest.fixture
def service(planar_config: RobotConfig) -> IKService:
    return IKService(ConfigStore(planar_config))
