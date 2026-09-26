"""HTTP API for the robot IK solver service.

The heavy lifting lives in :class:`robot_ik.IKService`; this module is only a
thin JSON adapter.  All computation is local and deterministic for a fixed
seed.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from robot_ik import (
    ArmConfig,
    IKRequest,
    IKService,
    JointSpec,
    JointType,
    Pose,
    SolverOptions,
)
from robot_ik.service import VersionConflictError
from robot_ik.validation import ConfigError

# ---------------------------------------------------------------------------
# JSON schemas
# ---------------------------------------------------------------------------


class JointIn(BaseModel):
    name: str
    joint_type: str = "revolute"
    axis: Tuple[float, float, float] = (0.0, 0.0, 1.0)
    translation: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation: Tuple[Tuple[float, float, float], ...] = (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    )
    lower_limit: float = -3.141592653589793
    upper_limit: float = 3.141592653589793


class ConfigIn(BaseModel):
    name: str
    version: str
    joints: List[JointIn]
    tool_transform: Tuple[Tuple[float, float, float, float], ...] = (
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )
    expected_version: Optional[str] = None


class OptionsIn(BaseModel):
    max_iterations: int = 200
    tolerance: float = 1e-6
    damping: float = 1e-3
    step_size: float = 1.0
    num_restarts: int = 8
    seed: int = 0
    timeout_ms: Optional[float] = None
    singularity_threshold: float = 1e-3


class PoseIn(BaseModel):
    position: Tuple[float, float, float]
    orientation_wxyz: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)


class SolveIn(BaseModel):
    target: PoseIn
    previous_joints: Optional[List[float]] = None
    options: Optional[OptionsIn] = None


# ---------------------------------------------------------------------------
# conversion helpers
# ---------------------------------------------------------------------------


def _to_config(c: ConfigIn) -> ArmConfig:
    return ArmConfig(
        name=c.name,
        version=c.version,
        joints=tuple(
            JointSpec(
                name=j.name,
                joint_type=JointType(j.joint_type),
                axis=tuple(j.axis),
                translation=tuple(j.translation),
                rotation=tuple(tuple(r) for r in j.rotation),
                lower_limit=j.lower_limit,
                upper_limit=j.upper_limit,
            )
            for j in c.joints
        ),
        tool_transform=tuple(tuple(r) for r in c.tool_transform),
    )


def _to_options(o: OptionsIn) -> SolverOptions:
    return SolverOptions(
        max_iterations=o.max_iterations,
        tolerance=o.tolerance,
        damping=o.damping,
        step_size=o.step_size,
        num_restarts=o.num_restarts,
        seed=o.seed,
        timeout_ms=o.timeout_ms,
        singularity_threshold=o.singularity_threshold,
    )


# ---------------------------------------------------------------------------
# application
# ---------------------------------------------------------------------------


def _default_config() -> ArmConfig:
    joints = (
        JointSpec(name="j1", axis=(0, 0, 1), translation=(0, 0, 0.2)),
        JointSpec(name="j2", axis=(0, 1, 0), translation=(0, 0, 0.1)),
        JointSpec(name="j3", axis=(0, 1, 0), translation=(0.3, 0, 0)),
        JointSpec(name="j4", axis=(1, 0, 0), translation=(0.2, 0, 0)),
        JointSpec(name="j5", axis=(0, 1, 0), translation=(0.1, 0, 0)),
        JointSpec(name="j6", axis=(1, 0, 0), translation=(0.05, 0, 0)),
    )
    return ArmConfig(name="arm6", version="v1", joints=joints)


app = FastAPI(title="robot-ik-solver", version="0.1.0")
service = IKService(_default_config(), SolverOptions(seed=0))


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/config")
def get_config() -> dict:
    cfg = service.current_config()
    return {
        "name": cfg.name,
        "version": cfg.version,
        "dof": len(cfg.joints),
        "joints": [
            {
                "name": j.name,
                "axis": j.axis,
                "translation": j.translation,
                "lower_limit": j.lower_limit,
                "upper_limit": j.upper_limit,
            }
            for j in cfg.joints
        ],
        "options": service.current_options().__dict__,
    }


@app.put("/config")
def put_config(body: ConfigIn) -> dict:
    try:
        new_version = service.update_config(
            _to_config(body), expected_version=body.expected_version
        )
    except ConfigError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except VersionConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    return {"version": new_version}


@app.put("/options")
def put_options(body: OptionsIn) -> dict:
    service.update_options(_to_options(body))
    return {"status": "ok"}


@app.post("/solve")
def solve(body: SolveIn) -> dict:
    request = IKRequest(
        target=Pose(
            position=tuple(body.target.position),
            orientation_wxyz=tuple(body.target.orientation_wxyz),
        ),
        previous_joints=(
            None if body.previous_joints is None else tuple(body.previous_joints)
        ),
        options=None if body.options is None else _to_options(body.options),
    )
    result = service.solve(request)
    return {
        "status": result.status.value,
        "joint_angles": result.joint_angles,
        "pose_error": result.pose_error,
        "iterations": result.iterations,
        "restarts_used": result.restarts_used,
        "config_version": result.config_version,
        "message": result.message,
        "singularity": None
        if result.singularity is None
        else {
            "is_singular": result.singularity.is_singular,
            "is_near_singular": result.singularity.is_near_singular,
            "min_singular_value": result.singularity.min_singular_value,
            "manipulability": result.singularity.manipulability,
            "degenerate_directions": result.singularity.degenerate_directions,
        },
    }
