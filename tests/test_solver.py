import numpy as np
import pytest

from robot_ik import (
    ArmConfig,
    IKRequest,
    IKSolver,
    IKStatus,
    JointSpec,
    KinematicModel,
    Pose,
    SolverOptions,
    analyze_singularity,
)
from conftest import pose_from_transform


def _solve(model, options, target, previous=None):
    solver = IKSolver(model, options)
    return solver.solve(IKRequest(target=target, previous_joints=previous))


# ---------------------------------------------------------------------------
# success paths
# ---------------------------------------------------------------------------


def test_round_trip_success(arm6_model, options):
    q_true = np.array([0.3, -0.5, 0.8, 0.2, -0.4, 0.6])
    target = pose_from_transform(arm6_model.forward_kinematics(q_true))
    res = _solve(arm6_model, options, target)
    assert res.status == IKStatus.SUCCESS
    assert res.joint_angles is not None
    assert res.pose_error < options.tolerance * 10
    # solution really reaches the target
    T_sol = arm6_model.forward_kinematics(np.array(res.joint_angles))
    T_ref = arm6_model.forward_kinematics(q_true)
    np.testing.assert_allclose(T_sol, T_ref, atol=1e-4)


def test_solution_respects_joint_limits(options):
    joints = (
        JointSpec(name="j1", axis=(0, 0, 1), translation=(0, 0, 0.2),
                  lower_limit=-1.0, upper_limit=1.0),
        JointSpec(name="j2", axis=(0, 1, 0), translation=(0, 0, 0.1),
                  lower_limit=-0.8, upper_limit=0.8),
        JointSpec(name="j3", axis=(0, 1, 0), translation=(0.3, 0, 0),
                  lower_limit=-1.2, upper_limit=1.2),
        JointSpec(name="j4", axis=(1, 0, 0), translation=(0.2, 0, 0),
                  lower_limit=-1.0, upper_limit=1.0),
        JointSpec(name="j5", axis=(0, 1, 0), translation=(0.1, 0, 0),
                  lower_limit=-1.0, upper_limit=1.0),
        JointSpec(name="j6", axis=(1, 0, 0), translation=(0.05, 0, 0),
                  lower_limit=-1.5, upper_limit=1.5),
    )
    cfg = ArmConfig(name="limited", version="v1", joints=joints)
    model = KinematicModel(cfg)
    q_true = np.array([0.3, -0.4, 0.6, 0.2, -0.3, 0.5])
    target = pose_from_transform(model.forward_kinematics(q_true))
    res = _solve(model, options, target)
    assert res.status == IKStatus.SUCCESS
    q = np.array(res.joint_angles)
    assert np.all(q >= model.lower_limits - 1e-9)
    assert np.all(q <= model.upper_limits + 1e-9)


# ---------------------------------------------------------------------------
# multi-solution selection / continuity
# ---------------------------------------------------------------------------


def test_multiple_solutions_selected_by_previous(arm6_model, options):
    q_true = np.array([0.3, -0.5, 0.8, 0.2, -0.4, 0.6])
    target = pose_from_transform(arm6_model.forward_kinematics(q_true))

    res_near = _solve(arm6_model, options, target, previous=tuple(q_true))
    assert res_near.status == IKStatus.SUCCESS
    np.testing.assert_allclose(res_near.joint_angles, q_true, atol=1e-3)

    # A distant previous solution pulls selection to a different branch.
    q_far = q_true + 1.5
    res_far = _solve(arm6_model, options, target, previous=tuple(q_far))
    assert res_far.status == IKStatus.SUCCESS
    d_near = np.linalg.norm(np.array(res_far.joint_angles) - q_far)
    d_other = np.linalg.norm(np.array(res_near.joint_angles) - q_far)
    assert d_near <= d_other + 1e-9


def test_continuity_across_trajectory(arm6_model, options):
    """Walking a trajectory with previous-solution seeding must not jump."""
    q = np.array([0.3, -0.5, 0.8, 0.2, -0.4, 0.6])
    prev = tuple(q)
    for step in range(1, 10):
        q_next = q + 0.05 * step
        target = pose_from_transform(arm6_model.forward_kinematics(q_next))
        res = _solve(arm6_model, options, target, previous=prev)
        assert res.status == IKStatus.SUCCESS
        jump = np.linalg.norm(np.array(res.joint_angles) - np.array(prev))
        assert jump < 0.5, f"step {step}: discontinuity of {jump} rad"
        prev = res.joint_angles


# ---------------------------------------------------------------------------
# failure classification
# ---------------------------------------------------------------------------


def test_unreachable_beyond_max_reach(arm6_model, options):
    res = _solve(arm6_model, options, Pose(position=(10.0, 0.0, 0.0)))
    assert res.status == IKStatus.UNREACHABLE
    assert res.joint_angles is None


def test_unreachable_inside_reach_sphere():
    """2-link planar arm: target inside the dead zone around the base."""
    cfg = ArmConfig(
        name="planar2link",
        version="v1",
        joints=(
            JointSpec(name="j1", axis=(0, 0, 1)),
            JointSpec(name="j2", axis=(0, 0, 1), translation=(1.0, 0, 0)),
        ),
        tool_transform=(
            (1.0, 0.0, 0.0, 0.7),
            (0.0, 1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        ),
    )
    model = KinematicModel(cfg)
    # |L1 - L2| = 0.3 > 0.1: cannot reach 0.1m from the base.
    res = _solve(model, SolverOptions(seed=1), Pose(position=(0.1, 0.0, 0.0)))
    assert res.status == IKStatus.UNREACHABLE
    assert res.joint_angles is None


def test_joint_limit_conflict(arm6_model, options):
    q_true = np.array([0.3, -0.5, 0.8, 0.2, -0.4, 0.6])
    target = pose_from_transform(arm6_model.forward_kinematics(q_true))
    tight = ArmConfig(
        name="tight",
        version="v1",
        joints=tuple(
            JointSpec(name=j.name, axis=j.axis, translation=j.translation,
                      lower_limit=-0.1, upper_limit=0.1)
            for j in arm6_model_config_joints()
        ),
    )
    tight_model = KinematicModel(tight)
    res = _solve(tight_model, options, target)
    assert res.status == IKStatus.JOINT_LIMIT_CONFLICT
    assert res.joint_angles is None


def arm6_model_config_joints():
    from conftest import ARM6_JOINTS

    return ARM6_JOINTS


def test_not_converged_with_tiny_budget(arm6_model):
    q_true = np.array([0.3, -0.5, 0.8, 0.2, -0.4, 0.6])
    target = pose_from_transform(arm6_model.forward_kinematics(q_true))
    opt = SolverOptions(max_iterations=1, num_restarts=0, seed=1)
    res = _solve(arm6_model, opt, target)
    assert res.status == IKStatus.NOT_CONVERGED
    assert res.joint_angles is None


def test_timeout_returns_no_partial_result(arm6_model):
    q_true = np.array([0.3, -0.5, 0.8, 0.2, -0.4, 0.6])
    target = pose_from_transform(arm6_model.forward_kinematics(q_true))
    opt = SolverOptions(
        max_iterations=100000, num_restarts=50, seed=1, timeout_ms=1e-3
    )
    res = _solve(arm6_model, opt, target)
    assert res.status == IKStatus.TIMEOUT
    assert res.joint_angles is None
    assert res.pose_error is None


def test_failure_statuses_are_distinguishable():
    """The four failure kinds map to four distinct enum values."""
    values = {
        IKStatus.UNREACHABLE,
        IKStatus.JOINT_LIMIT_CONFLICT,
        IKStatus.NOT_CONVERGED,
        IKStatus.TIMEOUT,
    }
    assert len(values) == 4


# ---------------------------------------------------------------------------
# singularity
# ---------------------------------------------------------------------------


def test_singularity_detected_when_stretched(arm6_model, options):
    target = pose_from_transform(arm6_model.forward_kinematics(np.zeros(6)))
    res = _solve(arm6_model, options, target)
    assert res.status == IKStatus.SUCCESS
    assert res.singularity is not None
    assert res.singularity.is_near_singular
    assert len(res.singularity.degenerate_directions) >= 1


def test_no_singularity_when_well_conditioned(arm6_model, options):
    q = np.array([0.3, -0.5, 0.8, 0.2, -0.4, 0.6])
    target = pose_from_transform(arm6_model.forward_kinematics(q))
    res = _solve(arm6_model, options, target)
    assert res.status == IKStatus.SUCCESS
    assert not res.singularity.is_singular


def test_singularity_threshold_configurable(arm6_model):
    q = np.array([0.3, -0.5, 0.8, 0.2, -0.4, 0.6])
    J = arm6_model.jacobian(q)
    sv = np.linalg.svd(J, compute_uv=False)
    min_sv = float(np.min(sv))
    assert min_sv > 0
    below = analyze_singularity(J, threshold=min_sv / 2)
    above = analyze_singularity(J, threshold=min_sv * 2)
    assert not below.is_near_singular
    assert above.is_near_singular
    assert len(above.degenerate_directions) >= 1


def test_degenerate_directions_point_along_lost_motion(planar2_config):
    model = KinematicModel(planar2_config)
    J = model.jacobian(np.zeros(2))  # fully stretched along x
    report = analyze_singularity(J, threshold=1e-3)
    # stretched planar arm loses the x-linear direction (among others, since
    # a 2-dof arm never spans 6d task space)
    linear_x = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    found = any(
        abs(np.dot(np.array(d), linear_x)) > 0.99
        for d in report.degenerate_directions
    )
    assert found


# ---------------------------------------------------------------------------
# reproducibility
# ---------------------------------------------------------------------------


def test_same_seed_same_result(arm6_model):
    q_true = np.array([0.3, -0.5, 0.8, 0.2, -0.4, 0.6])
    target = pose_from_transform(arm6_model.forward_kinematics(q_true))
    r1 = _solve(arm6_model, SolverOptions(seed=11), target)
    r2 = _solve(arm6_model, SolverOptions(seed=11), target)
    assert r1.status == r2.status
    assert r1.joint_angles == r2.joint_angles
    assert r1.iterations == r2.iterations
