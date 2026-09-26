"""Thread-safe IK service with config versioning and hot update.

Concurrency model:

* The service holds one immutable ``(ArmConfig, KinematicModel, SolverOptions)``
  snapshot behind a lock.  Readers only copy the reference, never mutate.
* Every ``solve`` builds its own :class:`IKSolver` (own RNG, own buffers), so
  concurrent requests are fully isolated and reproducibility is unaffected
  by thread interleaving.
* ``update_config`` validates the new configuration *before* swapping, and
  supports optimistic version checks.  A solve that already started keeps
  using the snapshot it took at entry -- versions are never mixed.
"""

from __future__ import annotations

import threading
from dataclasses import replace
from typing import Optional, Tuple

from .kinematics import KinematicModel
from .solver import IKSolver
from .types import ArmConfig, IKRequest, IKResult, IKStatus, SolverOptions
from .validation import (
    ConfigError,
    RequestError,
    validate_config,
    validate_options,
    validate_request,
)

_DEFAULT_OPTIONS = SolverOptions()


class VersionConflictError(RuntimeError):
    """Raised when an update's expected version does not match the current one."""


class IKService:
    """Owns the current configuration; each solve snapshots it at entry."""

    def __init__(self, config: ArmConfig, options: Optional[SolverOptions] = None) -> None:
        validate_config(config)
        opts = options if options is not None else _DEFAULT_OPTIONS
        validate_options(opts)
        self._lock = threading.Lock()
        self._snapshot: Tuple[ArmConfig, KinematicModel, SolverOptions] = (
            config,
            KinematicModel(config),
            opts,
        )

    # ------------------------------------------------------------------

    def _current(self) -> Tuple[ArmConfig, KinematicModel, SolverOptions]:
        with self._lock:
            return self._snapshot

    def current_version(self) -> str:
        return self._current()[0].version

    def current_config(self) -> ArmConfig:
        return self._current()[0]

    def current_options(self) -> SolverOptions:
        return self._current()[2]

    # ------------------------------------------------------------------

    def update_config(self, config: ArmConfig, expected_version: Optional[str] = None) -> str:
        """Atomically swap the configuration (with optional version check).

        The new configuration is fully validated before the swap; on any
        error the current configuration is left untouched.
        """
        validate_config(config)
        model = KinematicModel(config)  # build outside the lock
        with self._lock:
            current = self._snapshot[0]
            if expected_version is not None and expected_version != current.version:
                raise VersionConflictError(
                    f"expected version {expected_version!r} but current is "
                    f"{current.version!r}; update rejected"
                )
            self._snapshot = (config, model, self._snapshot[2])
            return config.version

    def update_options(self, options: SolverOptions) -> None:
        """Atomically swap the default solver options."""
        validate_options(options)
        with self._lock:
            cfg, model, _ = self._snapshot
            self._snapshot = (cfg, model, options)

    # ------------------------------------------------------------------

    def solve(self, request: IKRequest) -> IKResult:
        config, model, default_opts = self._current()  # snapshot at entry
        options = request.options if request.options is not None else default_opts
        try:
            validate_request(request, model.dof)
        except RequestError as exc:
            return IKResult(
                status=IKStatus.INVALID_REQUEST,
                joint_angles=None,
                pose_error=None,
                iterations=0,
                restarts_used=0,
                singularity=None,
                config_version=config.version,
                message=str(exc),
            )
        solver = IKSolver(model, options)
        result = solver.solve(request)
        return replace(result, config_version=config.version)
