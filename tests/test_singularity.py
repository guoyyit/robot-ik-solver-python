"""Singularity detection semantics and configurability."""

from __future__ import annotations

import math

import numpy as np

from ik_service.models import IKRequest
from ik_service.solver import singularity_report, solve_ik

from .conftest import planar_3r, planar_pose, pose_of, puma_6r


def test_full_rank_config_not_singular():
    cfg = puma_6r()
    report = singularity_report(cfg, np.array([0.2, 0.5, -0.8, 0.1, 0.3, 0.0]))
    assert not report.is_singular
    assert report.smallest_singular_value > cfg.singular_threshold
    assert report.manipulability > 0.0
    direction = np.asarray(report.affected_direction)
    assert np.isclose(np.linalg.norm(direction), 1.0)


def test_full_extension_is_singular_with_direction():
    cfg = puma_6r()
    report = singularity_report(cfg, np.zeros(6))
    assert report.is_singular and report.is_near_singular
    assert report.smallest_singular_value < cfg.singular_threshold
    # Weak direction must be a unit twist in base-frame task space.
    assert len(report.affected_direction) == 6
    assert math.isclose(math.sqrt(sum(v * v for v in report.affected_direction)), 1.0, abs_tol=1e-9)


def test_threshold_configuration_changes_classification():
    q = np.array([0.3, -0.2, 0.4])
    permissive = planar_3r(singular_threshold=0.0, near_singular_threshold=0.0)
    strict = planar_3r(singular_threshold=1e6, near_singular_threshold=1e6)
    assert not singularity_report(permissive, q).is_singular
    assert singularity_report(strict, q).is_singular


def test_report_is_deterministic():
    cfg = puma_6r()
    q = np.zeros(6)
    a = singularity_report(cfg, q)
    b = singularity_report(cfg, q)
    assert a.affected_direction == b.affected_direction
    assert a.smallest_singular_value == b.smallest_singular_value


def test_success_result_carries_singularity_report(planar_config):
    # The planar-3R Jacobian always has 3 zero task-space rows, so the planar
    # robot is structurally singular in 6D; the success path must still attach
    # a well-formed report rather than hiding the condition.
    result = solve_ik(planar_config, IKRequest(target=planar_pose(1.5, 0.5)))
    assert result.success
    assert result.singularity is not None
    assert result.singularity.is_singular
    assert len(result.singularity.affected_direction) == 6


def test_six_dof_generic_pose_not_flagged_singular():
    cfg = puma_6r()
    q = np.array([0.2, 0.5, -0.8, 0.1, 0.3, 0.0])
    target = pose_of(cfg, q)
    result = solve_ik(cfg, IKRequest(target=target, seed_override=1))
    assert result.success
    assert not result.singularity.is_singular
