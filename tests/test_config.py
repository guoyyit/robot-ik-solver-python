"""Configuration validation and hot-reload semantics."""

from __future__ import annotations

import threading

import pytest

from ik_service.config import ConfigStore, ConfigValidationError, validate_config
from ik_service.models import JointSpec, RobotConfig

from .conftest import planar_3r


def test_valid_config_accepted():
    validate_config(planar_3r())


@pytest.mark.parametrize(
    "mutate",
    [
        lambda c: RobotConfig(name="", joints=c.joints),
        lambda c: RobotConfig(name=c.name, joints=()),
        lambda c: RobotConfig(
            name=c.name,
            joints=(JointSpec("a"), JointSpec("a")),
        ),
        lambda c: RobotConfig(
            name=c.name,
            joints=(JointSpec("a", lower_limit=1.0, upper_limit=0.0),),
        ),
        lambda c: RobotConfig(
            name=c.name,
            joints=(JointSpec("a", axis=(0.0, 0.0, 0.0)),),
        ),
        lambda c: RobotConfig(name=c.name, joints=c.joints, singular_threshold=0.1,
                              near_singular_threshold=0.01),
        lambda c: RobotConfig(name=c.name, joints=c.joints, max_iterations=0),
        lambda c: RobotConfig(name=c.name, joints=c.joints, restarts=0),
        lambda c: RobotConfig(name=c.name, joints=c.joints, timeout_seconds=0.0),
        lambda c: RobotConfig(name=c.name, joints=c.joints, damping=-1.0),
        lambda c: RobotConfig(name=c.name, joints=c.joints, version=0),
    ],
)
def test_invalid_configs_rejected(mutate):
    with pytest.raises(ConfigValidationError):
        validate_config(mutate(planar_3r()))


def test_store_version_increments_and_old_snapshot_immutable():
    store = ConfigStore(planar_3r())
    v1 = store.current()
    assert v1.version == 1
    installed = store.update(planar_3r(random_seed=99))
    assert installed.version == 2
    assert store.current().version == 2
    assert store.current().random_seed == 99
    # A pinned reference never observes the reload.
    assert v1.version == 1 and v1.random_seed == 0


def test_update_rejects_invalid_config_and_keeps_old():
    store = ConfigStore(planar_3r())
    with pytest.raises(ConfigValidationError):
        store.update(RobotConfig(name="bad", joints=()))
    assert store.current().version == 1


def test_concurrent_reads_see_consistent_snapshot():
    store = ConfigStore(planar_3r())
    seen: list[int] = []

    def reader() -> None:
        for _ in range(200):
            seen.append(store.current().version)

    threads = [threading.Thread(target=reader) for _ in range(4)]
    for t in threads:
        t.start()
    for i in range(50):
        store.update(planar_3r(random_seed=i + 1))
    for t in threads:
        t.join()
    assert all(isinstance(v, int) for v in seen)
