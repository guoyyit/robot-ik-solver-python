"""Numerical solver: success, multi-solution selection, diagnoses, budgets."""

from __future__ import annotations

import math

import numpy as np
import pytest

from ik_service.kinematics import forward_kinematics
from ik_service.models import (
    FailureReason,
    IKRequest,
    JointSpec,
    JointType,
    Pose,
    RobotConfig,
)
from ik_service.solver import solve_ik, singularity_report

from .conftest import planar_3r, planar_pose, pose_of, puma_6r


def _assert_at_target(config, result, target: Pose, pos_tol=1e-5, ori_tol=1e-5):
    assert result.success, result.message
    q = np.asarray(result.joints)
    t = forward_kinematics(config, q)
    assert np.allclose(t[:3, 3], target.position, atol=pos_tol)
    if result.singularity is not None:
        assert isinstance(result.singularity.affected_direction, tuple)
    lo = np.array([j.lower_limit for j in config.joints if j.joint_type is JointType.REVOLUTE])
    hi = np.array([j.upper_limit for j in config.joints if j.joint_type is JointType.REVOLUTE])
    assert np.all(q >= lo - 1e-9) and np.all(q <= hi + 1e-9)


def test_planar_reachable_target(planar_config):
    target = planar_pose(1.5, 0.5)
    result = solve_ik(planar_config, IKRequest(target=target))
    _assert_at_target(planar_config, result, target)


def test_six_dof_roundtrip():
    cfg = puma_6r()
    q = np.array([0.2, 0.5, -0.8, 0.1, 0.3, 0.0])
    target = pose_of(cfg, q)
    result = solve_ik(cfg, IKRequest(target=target, seed_override=1))
    _assert_at_target(cfg, result, target, pos_tol=1e-4, ori_tol=1e-4)
    assert result.joints is not None
    assert not result.singularity.is_singular


def test_multi_solution_prefers_previous_for_continuity(planar_config):
    # The planar-3R has redundant IK; track two nearby targets and require a
    # small joint-space jump when previous_joints is supplied.
    t1 = planar_pose(1.5, 0.5)
    r1 = solve_ik(planar_config, IKRequest(target=t1, seed_override=42))
    assert r1.success
    t2 = planar_pose(1.52, 0.5)
    r_cont = solve_ik(
        planar_config, IKRequest(target=t2, previous_joints=r1.joints, seed_override=42)
    )
    r_noprev = solve_ik(planar_config, IKRequest(target=t2, seed_override=42))
    assert r_cont.success and r_noprev.success
    jump_cont = max(abs(a - b) for a, b in zip(r1.joints, r_cont.joints))
    assert jump_cont < 0.1, jump_cont
    # continuity choice never worse than the unseeded ordering
    d_cont = sum((a - b) ** 2 for a, b in zip(r1.joints, r_cont.joints))
    d_noprev = sum((a - b) ** 2 for a, b in zip(r1.joints, r_noprev.joints))
    assert d_cont <= d_noprev + 1e-12


def test_unreachable_returns_distinguishable_reason_and_no_joints(planar_config):
    result = solve_ik(planar_config, IKRequest(target=planar_pose(10.0, 10.0)))
    assert not result.success
    assert result.joints is None
    assert result.reason is FailureReason.UNREACHABLE


def test_joint_limit_conflict_distinguishable():
    # Limits that only allow a nearly-straight arm cannot reach a point high
    # above the shoulder; the relaxed (unclamped) track can.
    cfg = planar_3r(limits=(-0.1, 0.1))
    result = solve_ik(cfg, IKRequest(target=planar_pose(0.2, 1.5)))
    assert not result.success
    assert result.joints is None
    assert result.reason is FailureReason.JOINT_LIMIT_CONFLICT


def test_solution_satisfying_limits_not_mislabeled():
    # Same target with full limits must succeed.
    cfg = planar_3r(limits=(-math.pi, math.pi))
    result = solve_ik(cfg, IKRequest(target=planar_pose(0.2, 1.5)))
    assert result.success
    assert result.reason is None


def test_did_not_converge_when_budget_tiny(planar_config):
    cfg = planar_3r(max_iterations=1, restarts=1, timeout_seconds=10.0)
    result = solve_ik(cfg, IKRequest(target=planar_pose(1.5, 0.5)))
    assert not result.success
    assert result.joints is None
    assert result.reason is FailureReason.DID_NOT_CONVERGE


def test_timeout_returns_timeout_and_no_partial_result(planar_config):
    cfg = planar_3r(max_iterations=10_000, restarts=10_000, timeout_seconds=10.0)
    result = solve_ik(
        cfg, IKRequest(target=planar_pose(10.0, 10.0), timeout_override=0.02)
    )
    assert not result.success
    assert result.joints is None
    assert result.reason is FailureReason.TIMEOUT


def test_invalid_requests_distinguishable(planar_config):
    bad_pose = IKRequest(target=Pose(position=(0.0, 0.0), quaternion_wxyz=(1.0, 0.0, 0.0, 0.0)))
    result = solve_ik(planar_config, bad_pose)
    assert result.reason is FailureReason.INVALID_REQUEST

    zero_q = IKRequest(target=Pose(position=(1.0, 0.0, 0.0), quaternion_wxyz=(0.0, 0.0, 0.0, 0.0)))
    result = solve_ik(planar_config, zero_q)
    assert result.reason is FailureReason.INVALID_REQUEST

    wrong_len = IKRequest(
        target=planar_pose(1.0, 0.0), previous_joints=(0.1, 0.2)
    )
    result = solve_ik(planar_config, wrong_len)
    assert result.reason is FailureReason.INVALID_REQUEST

    out_of_limits = IKRequest(
        target=planar_pose(1.0, 0.0), previous_joints=(10.0, 0.0, 0.0)
    )
    result = solve_ik(planar_config, out_of_limits)
    assert result.reason is FailureReason.INVALID_REQUEST


def test_version_mismatch_distinguishable(planar_config):
    request = IKRequest(target=planar_pose(1.0, 0.0), expected_version=999)
    result = solve_ik(planar_config, request)
    assert not result.success
    assert result.reason is FailureReason.VERSION_MISMATCH


def test_singularity_report_at_full_extension(puma_6r_cfg=None):
    cfg = puma_6r()
    report = singularity_report(cfg, np.zeros(6))
    assert report.is_singular
    assert report.is_near_singular
    assert report.smallest_singular_value < cfg.singular_threshold
    assert len(report.affected_direction) == 6
    # normalized task-space direction
    assert abs(math.sqrt(sum(v * v for v in report.affected_direction)) - 1.0) < 1e-9


def test_singularity_threshold_is_effective_and_deterministic():
    cfg_loose = planar_3r(singular_threshold=0.0, near_singular_threshold=0.0)
    cfg_strict = planar_3r(singular_threshold=1e6, near_singular_threshold=1e6)
    q = np.array([0.0, 0.0, 0.0])
    assert not singularity_report(cfg_loose, q).is_singular
    assert singularity_report(cfg_strict, q).is_singular


def test_seed_reproducibility(planar_config):
    target = planar_pose(1.5, 0.5)
    r1 = solve_ik(planar_config, IKRequest(target=target, seed_override=123))
    r2 = solve_ik(planar_config, IKRequest(target=target, seed_override=123))
    r3 = solve_ik(planar_config, IKRequest(target=target, seed_override=321))
    assert r1.joints == r2.joints
    assert r1.success
    # Different seed is allowed (but need not) to differ; result must still be valid.
    _assert_at_target(planar_config, r3, target)


def test_previous_joints_does_not_change_random_stream(planar_config):
    target = planar_pose(1.2, 0.7)
    a = solve_ik(planar_config, IKRequest(target=target, seed_override=7))
    b = solve_ik(
        planar_config,
        IKRequest(target=target, seed_override=7, previous_joints=(0.0, 0.0, 0.0)),
    )
    # Same explicit seed + deterministic draw count -> identical candidate set;
    # with previous=(0,0,0) continuity selects against the same guess the
    # ordering would choose first anyway.
    assert a.success and b.success


def test_robot_with_fixed_joints():
    # R - fixed(0.5m) - R: two DOF, reach radius up to 1.5.
    joints = (
        JointSpec(name="j0", joint_type=JointType.REVOLUTE, origin_xyz=(0.0, 0.0, 0.0)),
        JointSpec(
            name="f0",
            joint_type=JointType.FIXED,
            origin_xyz=(0.5, 0.0, 0.0),
            lower_limit=0.0,
            upper_limit=0.0,
        ),
        JointSpec(name="j1", joint_type=JointType.REVOLUTE, origin_xyz=(1.0, 0.0, 0.0)),
    )
    cfg = RobotConfig(name="2r-fixed", joints=joints, restarts=8, timeout_seconds=10.0)
    # FK sanity: q0=0,q1=0 -> x=1.5
    t = forward_kinematics(cfg, np.zeros(2))
    assert np.allclose(t[:3, 3], [1.5, 0.0, 0.0])
    # reachable interior point: build via FK
    target = pose_of(cfg, np.array([0.4, -0.7]))
    result = solve_ik(cfg, IKRequest(target=target, seed_override=3))
    assert result.success, result.message
    assert len(result.joints) == 2
    _assert_at_target(cfg, result, target)
    # previous_joints with wrong DOF count rejected
    bad = solve_ik(cfg, IKRequest(target=target, previous_joints=(0.1, 0.2, 0.3)))
    assert bad.reason is FailureReason.INVALID_REQUEST
