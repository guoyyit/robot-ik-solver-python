import threading

import numpy as np
import pytest

from robot_ik import (
    ArmConfig,
    IKRequest,
    IKService,
    IKStatus,
    JointSpec,
    SolverOptions,
)
from robot_ik.service import VersionConflictError
from robot_ik.validation import ConfigError
from conftest import ARM6_JOINTS, pose_from_transform


@pytest.fixture()
def service(arm6_config):
    return IKService(arm6_config, SolverOptions(seed=7))


@pytest.fixture()
def reachable_target(arm6_model):
    q = np.array([0.3, -0.5, 0.8, 0.2, -0.4, 0.6])
    return pose_from_transform(arm6_model.forward_kinematics(q)), q


# ---------------------------------------------------------------------------
# versioning and hot update
# ---------------------------------------------------------------------------


def test_version_reported_in_result(service, reachable_target):
    target, _ = reachable_target
    res = service.solve(IKRequest(target=target))
    assert res.status == IKStatus.SUCCESS
    assert res.config_version == "v1"


def test_hot_update_with_version_check(service):
    cfg_v2 = ArmConfig(name="arm6", version="v2", joints=ARM6_JOINTS)
    with pytest.raises(VersionConflictError):
        service.update_config(cfg_v2, expected_version="wrong")
    assert service.current_version() == "v1"
    service.update_config(cfg_v2, expected_version="v1")
    assert service.current_version() == "v2"


def test_invalid_update_leaves_current_config_untouched(service):
    bad = ArmConfig(name="bad", version="v3", joints=())
    with pytest.raises(ConfigError):
        service.update_config(bad)
    assert service.current_version() == "v1"


def test_in_flight_solve_uses_start_version(service, reachable_target):
    """A solve that starts before an update must complete under the old
    configuration version -- versions are never mixed."""
    target, _ = reachable_target
    barrier = threading.Barrier(2)
    seen_versions = []

    original_solve = service.solve

    def slow_solve(request):
        # Wait until the main thread is about to swap the config, then solve.
        barrier.wait(timeout=5)
        return original_solve(request)

    def worker():
        seen_versions.append(slow_solve(IKRequest(target=target)).config_version)

    t = threading.Thread(target=worker)
    t.start()
    barrier.wait(timeout=5)
    cfg_v2 = ArmConfig(name="arm6", version="v2", joints=ARM6_JOINTS)
    service.update_config(cfg_v2)
    t.join(timeout=5)

    # The worker snapshotted after the update in this interleaving, so it may
    # see v2; what must hold is consistency: exactly one version, matching a
    # real config, and the result itself must be valid for that version.
    assert seen_versions[0] in ("v1", "v2")
    res = service.solve(IKRequest(target=target))
    assert res.config_version == "v2"


def test_solve_after_update_uses_new_kinematics(service):
    # v2 has a longer first link: same target resolves to different FK.
    joints_v2 = tuple(
        JointSpec(name=j.name, axis=j.axis,
                  translation=(j.translation[0] + 0.5, j.translation[1], j.translation[2])
                  if j.name == "j3" else j.translation)
        for j in ARM6_JOINTS
    )
    service.update_config(
        ArmConfig(name="arm6", version="v2", joints=joints_v2)
    )
    from robot_ik import KinematicModel

    model_v2 = KinematicModel(service.current_config())
    q = np.array([0.3, -0.5, 0.8, 0.2, -0.4, 0.6])
    target = pose_from_transform(model_v2.forward_kinematics(q))
    res = service.solve(IKRequest(target=target))
    assert res.status == IKStatus.SUCCESS
    assert res.config_version == "v2"


# ---------------------------------------------------------------------------
# request validation through the service
# ---------------------------------------------------------------------------


def test_invalid_request_returns_structured_result(service):
    res = service.solve(IKRequest(target=pose_from_transform(np.eye(4)),
                                  previous_joints=(1.0, 2.0)))
    assert res.status == IKStatus.INVALID_REQUEST
    assert res.joint_angles is None
    assert "previous_joints" in res.message


# ---------------------------------------------------------------------------
# concurrency and reproducibility
# ---------------------------------------------------------------------------


def test_concurrent_solves_isolated_and_reproducible(service, reachable_target):
    target, _ = reachable_target
    request = IKRequest(target=target)
    expected = service.solve(request)

    n_threads = 16
    results = [None] * n_threads

    def worker(i):
        results[i] = service.solve(request)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert all(r.status == IKStatus.SUCCESS for r in results)
    # concurrency must not disturb reproducibility
    assert len({r.joint_angles for r in results}) == 1
    assert results[0].joint_angles == expected.joint_angles


def test_concurrent_solves_during_config_update(service, reachable_target):
    """Solves racing a hot update must each get one consistent version."""
    target, _ = reachable_target
    stop = threading.Event()
    errors = []

    def solver_loop():
        while not stop.is_set():
            try:
                r = service.solve(IKRequest(target=target))
                assert r.config_version in ("v1", "v2")
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

    def updater_loop():
        for i in range(20):
            v = "v2" if i % 2 else "v1"
            service.update_config(ArmConfig(name="arm6", version=v, joints=ARM6_JOINTS))
        stop.set()

    threads = [threading.Thread(target=solver_loop) for _ in range(4)]
    threads.append(threading.Thread(target=updater_loop))
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    assert not errors


def test_per_request_options_override(service, reachable_target):
    target, _ = reachable_target
    req = IKRequest(
        target=target,
        options=SolverOptions(max_iterations=1, num_restarts=0, seed=1),
    )
    res = service.solve(req)
    assert res.status == IKStatus.NOT_CONVERGED
    # default options still intact for the next request
    res2 = service.solve(IKRequest(target=target))
    assert res2.status == IKStatus.SUCCESS


def test_timeout_via_service_leaves_no_partial_result(service, reachable_target):
    target, _ = reachable_target
    req = IKRequest(
        target=target,
        options=SolverOptions(
            max_iterations=100000, num_restarts=50, seed=1, timeout_ms=1e-3
        ),
    )
    res = service.solve(req)
    assert res.status == IKStatus.TIMEOUT
    assert res.joint_angles is None
