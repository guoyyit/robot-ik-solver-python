import numpy as np
import pytest

from robot_ik import (
    ArmConfig,
    IKRequest,
    JointSpec,
    Pose,
    SolverOptions,
    validate_config,
    validate_request,
)
from robot_ik.validation import ConfigError, RequestError


def _cfg(**kw):
    base = dict(name="a", version="v1", joints=(JointSpec(name="j1"),))
    base.update(kw)
    return ArmConfig(**base)


def test_valid_config_accepted():
    validate_config(_cfg())


def test_empty_joints_rejected():
    with pytest.raises(ConfigError):
        validate_config(_cfg(joints=()))


def test_duplicate_joint_names_rejected():
    with pytest.raises(ConfigError):
        validate_config(_cfg(joints=(JointSpec(name="j"), JointSpec(name="j"))))


def test_zero_axis_rejected():
    with pytest.raises(ConfigError):
        validate_config(_cfg(joints=(JointSpec(name="j", axis=(0, 0, 0)),)))


def test_non_finite_axis_rejected():
    with pytest.raises(ConfigError):
        validate_config(_cfg(joints=(JointSpec(name="j", axis=(0, 0, float("nan"))),)))


def test_inverted_limits_rejected():
    with pytest.raises(ConfigError):
        validate_config(
            _cfg(joints=(JointSpec(name="j", lower_limit=1.0, upper_limit=-1.0),))
        )


def test_non_rotation_matrix_rejected():
    bad = ((2.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    with pytest.raises(ConfigError):
        validate_config(_cfg(joints=(JointSpec(name="j", rotation=bad),)))


def test_bad_tool_transform_rejected():
    bad_tool = (
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0),
        (0.0, 0.0, 0.0, 2.0),
    )
    with pytest.raises(ConfigError):
        validate_config(_cfg(tool_transform=bad_tool))


def test_empty_version_rejected():
    with pytest.raises(ConfigError):
        validate_config(_cfg(version=""))


def test_request_nan_target_rejected():
    req = IKRequest(target=Pose(position=(float("nan"), 0.0, 0.0)))
    with pytest.raises(RequestError):
        validate_request(req, dof=6)


def test_request_zero_quaternion_rejected():
    req = IKRequest(target=Pose(position=(0, 0, 0), orientation_wxyz=(0, 0, 0, 0)))
    with pytest.raises(RequestError):
        validate_request(req, dof=6)


def test_request_previous_joints_wrong_dof_rejected():
    req = IKRequest(
        target=Pose(position=(0, 0, 0)), previous_joints=(0.0, 0.0)
    )
    with pytest.raises(RequestError):
        validate_request(req, dof=6)


def test_request_bad_options_rejected():
    req = IKRequest(
        target=Pose(position=(0, 0, 0)),
        options=SolverOptions(max_iterations=0),
    )
    with pytest.raises(RequestError):
        validate_request(req, dof=6)
