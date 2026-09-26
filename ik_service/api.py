"""FastAPI HTTP layer exposing the IK service."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .config import ConfigStore, ConfigValidationError
from .models import IKRequest, IKResult, JointSpec, JointType, Pose, RobotConfig
from .service import IKService


class JointModel(BaseModel):
    name: str
    joint_type: JointType = JointType.REVOLUTE
    axis: tuple[float, float, float] = (0.0, 0.0, 1.0)
    origin_rpy: tuple[float, float, float] = (0.0, 0.0, 0.0)
    origin_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    lower_limit: float = -3.141592653589793
    upper_limit: float = 3.141592653589793


class RobotConfigModel(BaseModel):
    name: str
    joints: list[JointModel]
    singular_threshold: float = 1e-4
    near_singular_threshold: float = 1e-2
    position_tolerance: float = 1e-5
    orientation_tolerance: float = 1e-5
    max_iterations: int = 200
    timeout_seconds: float = 5.0
    restarts: int = 8
    damping: float = 1e-4
    random_seed: int = 0


class PoseModel(BaseModel):
    position: tuple[float, float, float]
    quaternion_wxyz: tuple[float, float, float, float]


class IKRequestModel(BaseModel):
    target: PoseModel
    previous_joints: list[float] | None = None
    expected_version: int | None = None
    seed_override: int | None = None
    timeout_override: float | None = Field(default=None, gt=0)


def _to_domain(model: RobotConfigModel) -> RobotConfig:
    return RobotConfig(
        name=model.name,
        joints=tuple(
            JointSpec(
                name=j.name,
                joint_type=j.joint_type,
                axis=j.axis,
                origin_rpy=j.origin_rpy,
                origin_xyz=j.origin_xyz,
                lower_limit=j.lower_limit,
                upper_limit=j.upper_limit,
            )
            for j in model.joints
        ),
        singular_threshold=model.singular_threshold,
        near_singular_threshold=model.near_singular_threshold,
        position_tolerance=model.position_tolerance,
        orientation_tolerance=model.orientation_tolerance,
        max_iterations=model.max_iterations,
        timeout_seconds=model.timeout_seconds,
        restarts=model.restarts,
        damping=model.damping,
        random_seed=model.random_seed,
    )


def _result_to_dict(result: IKResult) -> dict:
    return {
        "success": result.success,
        "joints": list(result.joints) if result.joints is not None else None,
        "reason": result.reason.value if result.reason is not None else None,
        "message": result.message,
        "singularity": None
        if result.singularity is None
        else {
            "is_singular": result.singularity.is_singular,
            "is_near_singular": result.singularity.is_near_singular,
            "smallest_singular_value": result.singularity.smallest_singular_value,
            "manipulability": result.singularity.manipulability,
            "affected_direction": list(result.singularity.affected_direction),
        },
        "iterations": result.iterations,
        "restarts_used": result.restarts_used,
        "config_version": result.config_version,
        "residual_position": result.residual_position,
        "residual_orientation": result.residual_orientation,
    }


def create_app(service: IKService | None = None) -> FastAPI:
    """Build the ASGI app; a default planar-3R config is used when none given."""
    if service is None:
        default = RobotConfig(
            name="default-planar-3r",
            joints=(
                JointSpec(name="joint_0", origin_xyz=(0.0, 0.0, 0.0)),
                JointSpec(name="joint_1", origin_xyz=(1.0, 0.0, 0.0)),
                JointSpec(name="joint_2", origin_xyz=(1.0, 0.0, 0.0)),
            ),
        )
        service = IKService(ConfigStore(default))

    app = FastAPI(title="Robot IK Solver", version="1.0.0")
    app.state.service = service

    @app.get("/health")
    def health() -> dict:
        cfg = service.store.current()
        return {"status": "ok", "config": cfg.name, "version": cfg.version}

    @app.get("/config")
    def get_config() -> dict:
        cfg = service.store.current()
        return {
            "name": cfg.name,
            "version": cfg.version,
            "joints": [
                {
                    "name": j.name,
                    "joint_type": j.joint_type.value,
                    "axis": list(j.axis),
                    "origin_rpy": list(j.origin_rpy),
                    "origin_xyz": list(j.origin_xyz),
                    "lower_limit": j.lower_limit,
                    "upper_limit": j.upper_limit,
                }
                for j in cfg.joints
            ],
            "singular_threshold": cfg.singular_threshold,
            "near_singular_threshold": cfg.near_singular_threshold,
            "position_tolerance": cfg.position_tolerance,
            "orientation_tolerance": cfg.orientation_tolerance,
            "max_iterations": cfg.max_iterations,
            "timeout_seconds": cfg.timeout_seconds,
            "restarts": cfg.restarts,
            "damping": cfg.damping,
            "random_seed": cfg.random_seed,
        }

    @app.put("/config")
    def replace_config(model: RobotConfigModel) -> JSONResponse:
        try:
            installed = service.store.update(_to_domain(model))
        except ConfigValidationError as exc:
            return JSONResponse(
                status_code=422,
                content={"success": False, "reason": "invalid_config", "message": str(exc)},
            )
        return JSONResponse(
            status_code=200,
            content={"success": True, "name": installed.name, "version": installed.version},
        )

    @app.post("/ik")
    def solve(model: IKRequestModel) -> JSONResponse:
        request = IKRequest(
            target=Pose(
                position=tuple(model.target.position),
                quaternion_wxyz=tuple(model.target.quaternion_wxyz),
            ),
            previous_joints=(
                tuple(model.previous_joints) if model.previous_joints is not None else None
            ),
            expected_version=model.expected_version,
            seed_override=model.seed_override,
            timeout_override=model.timeout_override,
        )
        result = service.solve(request)
        status = 200 if result.success else 422
        return JSONResponse(status_code=status, content=_result_to_dict(result))

    return app
