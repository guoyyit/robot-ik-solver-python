"""Robot multi-joint inverse kinematics solver and singularity diagnostics."""

from .types import (
    ArmConfig,
    IKRequest,
    IKResult,
    IKStatus,
    JointSpec,
    JointType,
    Pose,
    SingularityReport,
    SolverOptions,
)
from .validation import ConfigError, RequestError, validate_config, validate_request
from .kinematics import KinematicModel
from .singularity import analyze_singularity
from .solver import IKSolver
from .service import IKService

__all__ = [
    "ArmConfig",
    "IKRequest",
    "IKResult",
    "IKStatus",
    "JointSpec",
    "JointType",
    "Pose",
    "SingularityReport",
    "SolverOptions",
    "ConfigError",
    "RequestError",
    "validate_config",
    "validate_request",
    "KinematicModel",
    "analyze_singularity",
    "IKSolver",
    "IKService",
]
