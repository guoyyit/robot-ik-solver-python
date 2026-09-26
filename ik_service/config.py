"""Configuration validation, immutable snapshots and hot-reload store.

Configurations are plain frozen dataclasses (see :mod:`ik_service.models`).
:func:`validate_config` rejects internally inconsistent descriptions; the
:class:`ConfigStore` hands out a single immutable object per generation so an
in-flight solve can pin a version and never see a mixed new/old configuration.
"""

from __future__ import annotations

import math
import threading
from dataclasses import replace

from .models import JointType, RobotConfig


class ConfigValidationError(ValueError):
    """Raised when a robot configuration is internally inconsistent."""


def _is_finite_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _finite_vector3(value: object) -> bool:
    return (
        isinstance(value, tuple)
        and len(value) == 3
        and all(_is_finite_number(v) for v in value)
    )


def validate_config(config: RobotConfig) -> None:
    """Validate a configuration. Raises :class:`ConfigValidationError`."""
    if not isinstance(config, RobotConfig):
        raise ConfigValidationError("config must be a RobotConfig instance")
    if not isinstance(config.joints, tuple) or len(config.joints) == 0:
        raise ConfigValidationError("joints must be a non-empty tuple")

    seen_names: set[str] = set()
    revolute_count = 0
    for i, spec in enumerate(config.joints):
        if not spec.name:
            raise ConfigValidationError(f"joint at index {i} has an empty name")
        if spec.name in seen_names:
            raise ConfigValidationError(f"duplicate joint name: {spec.name!r}")
        seen_names.add(spec.name)
        if spec.joint_type not in (JointType.REVOLUTE, JointType.FIXED):
            raise ConfigValidationError(f"joint {spec.name!r}: unsupported joint type")
        if not _finite_vector3(spec.axis):
            raise ConfigValidationError(f"joint {spec.name!r}: axis must be 3 finite numbers")
        norm = math.sqrt(sum(float(v) ** 2 for v in spec.axis))
        if spec.joint_type is JointType.REVOLUTE and norm == 0.0:
            raise ConfigValidationError(f"joint {spec.name!r}: revolute axis must not be zero")
        if not _finite_vector3(spec.origin_xyz) or not _finite_vector3(spec.origin_rpy):
            raise ConfigValidationError(f"joint {spec.name!r}: origin xyz/rpy must be finite")
        if not (
            _is_finite_number(spec.lower_limit)
            and _is_finite_number(spec.upper_limit)
        ):
            raise ConfigValidationError(f"joint {spec.name!r}: limits must be finite")
        if spec.lower_limit >= spec.upper_limit:
            raise ConfigValidationError(
                f"joint {spec.name!r}: lower_limit ({spec.lower_limit}) must be "
                f"strictly smaller than upper_limit ({spec.upper_limit})"
            )
        if spec.joint_type is JointType.REVOLUTE:
            revolute_count += 1

    if revolute_count == 0:
        raise ConfigValidationError("configuration must contain at least one revolute joint")

    positive_settings = {
        "singular_threshold": config.singular_threshold,
        "near_singular_threshold": config.near_singular_threshold,
        "position_tolerance": config.position_tolerance,
        "orientation_tolerance": config.orientation_tolerance,
        "damping": config.damping,
        "timeout_seconds": config.timeout_seconds,
    }
    for key, value in positive_settings.items():
        if not _is_finite_number(value) or float(value) <= 0.0:
            raise ConfigValidationError(f"{key} must be a positive finite number")
    if config.singular_threshold > config.near_singular_threshold:
        raise ConfigValidationError(
            "singular_threshold must not exceed near_singular_threshold"
        )
    if not isinstance(config.max_iterations, int) or config.max_iterations <= 0:
        raise ConfigValidationError("max_iterations must be a positive integer")
    if not isinstance(config.restarts, int) or config.restarts <= 0:
        raise ConfigValidationError("restarts must be a positive integer")
    if not isinstance(config.random_seed, int):
        raise ConfigValidationError("random_seed must be an integer")
    if not isinstance(config.version, int) or config.version < 1:
        raise ConfigValidationError("version must be a positive integer")
    if not isinstance(config.name, str) or not config.name:
        raise ConfigValidationError("robot name must be a non-empty string")


class ConfigStore:
    """Holds the active configuration snapshot; supports atomic hot reload.

    Every :meth:`current` call returns the same immutable object until
    :meth:`update` installs a new generation, whose version is the previous
    version plus one. In-flight solves keep their own reference, so reloads
    never corrupt running requests.
    """

    def __init__(self, config: RobotConfig) -> None:
        validate_config(config)
        self._lock = threading.RLock()
        self._current = config
        self._next_version = config.version + 1

    def current(self) -> RobotConfig:
        with self._lock:
            return self._current

    def update(self, config: RobotConfig) -> RobotConfig:
        """Validate and atomically install ``config`` with a new version."""
        validate_config(config)
        with self._lock:
            versioned = replace(config, version=self._next_version)
            self._next_version += 1
            self._current = versioned
            return versioned
