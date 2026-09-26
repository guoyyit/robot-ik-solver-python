"""Numerical inverse kinematics solver.

Method: damped least-squares (Levenberg-Marquardt style) iterations on the
6D pose error, with joint-limit clamping after every step and deterministic
multi-start restarts drawn from a per-solver seeded RNG.

Failure classification (never a silent wrong answer):

* ``UNREACHABLE``            – the target pose is outside the workspace: a
  diagnostic pass *without* joint limits also stalls with residual error.
* ``JOINT_LIMIT_CONFLICT``   – the limit-free diagnostic pass converges but
  the clamped pass does not: the kinematics could reach the pose, the joint
  limits forbid it.
* ``NOT_CONVERGED``          – iterations were still making progress when the
  iteration budget ran out (raise ``max_iterations`` to confirm).
* ``TIMEOUT``                – the wall-clock budget expired; no partial
  result is returned.
"""

from __future__ import annotations

import time
from typing import List, Optional, Tuple

import numpy as np

from .kinematics import KinematicModel, pose_error, quaternion_to_matrix
from .singularity import analyze_singularity
from .types import IKRequest, IKResult, IKStatus, SolverOptions

# Relative improvement below this over the tail of a run counts as "stalled".
_STALL_RATIO = 1e-3
_STALL_WINDOW = 10


class _Timeout(Exception):
    pass


class IKSolver:
    """Single-use solver bound to one immutable kinematic model.

    A fresh solver is created per request, so concurrent solves never share
    mutable state; each solver owns its RNG, making results reproducible for
    a fixed seed regardless of interleaving.
    """

    def __init__(self, model: KinematicModel, options: SolverOptions) -> None:
        self._model = model
        self._opt = options
        self._rng = np.random.default_rng(options.seed)

    # ------------------------------------------------------------------
    # public entry point
    # ------------------------------------------------------------------

    def solve(self, request: IKRequest) -> IKResult:
        start = time.monotonic()
        deadline = (
            None
            if self._opt.timeout_ms is None
            else start + self._opt.timeout_ms / 1000.0
        )
        target = np.eye(4)
        target[:3, :3] = quaternion_to_matrix(
            np.asarray(request.target.orientation_wxyz, dtype=float)
        )
        target[:3, 3] = np.asarray(request.target.position, dtype=float)

        # Fast analytic reject: clearly outside the maximum reach.
        if float(np.linalg.norm(target[:3, 3])) > self._model.max_reach() + 1e-9:
            return self._finish(
                IKStatus.UNREACHABLE, None, None, 0, 0, start,
                "target position beyond maximum reach of the arm",
            )

        seeds = self._make_seeds(request.previous_joints)
        try:
            solutions, best_q, best_err, total_iters, restarts_used, stalled = (
                self._run_passes(seeds, target, clamp=True, deadline=deadline)
            )
        except _Timeout:
            return self._finish(
                IKStatus.TIMEOUT, None, None, 0, 0, start,
                "solver exceeded the configured time budget",
            )

        if solutions:
            chosen = self._select(solutions, request.previous_joints)
            report = analyze_singularity(
                self._model.jacobian(chosen), self._opt.singularity_threshold
            )
            err = float(np.linalg.norm(pose_error(self._model.forward_kinematics(chosen), target)))
            return IKResult(
                status=IKStatus.SUCCESS,
                joint_angles=tuple(float(x) for x in chosen),
                pose_error=err,
                iterations=total_iters,
                restarts_used=restarts_used,
                singularity=report,
                config_version="",
                message="converged within joint limits",
            )

        # Failure: run a limit-free diagnostic pass to classify the cause.
        status, message = self._classify(
            seeds, target, best_q, best_err, stalled, deadline
        )
        return self._finish(status, best_q, best_err, total_iters, restarts_used, start, message)

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _finish(self, status, best_q, best_err, iters, restarts, start, message):
        report = None
        if best_q is not None:
            report = analyze_singularity(
                self._model.jacobian(best_q), self._opt.singularity_threshold
            )
        return IKResult(
            status=status,
            joint_angles=None,  # failures never expose partial joint angles
            pose_error=None if best_err is None else float(best_err),
            iterations=iters,
            restarts_used=restarts,
            singularity=report,
            config_version="",
            message=message,
        )

    def _make_seeds(self, previous_joints: Optional[Tuple[float, ...]]) -> List[np.ndarray]:
        lo, hi = self._model.lower_limits, self._model.upper_limits
        seeds: List[np.ndarray] = []
        if previous_joints is not None:
            seeds.append(np.clip(np.asarray(previous_joints, dtype=float), lo, hi))
        seeds.append(np.clip(np.zeros(self._model.dof), lo, hi))
        seeds.append((lo + hi) / 2.0)
        for _ in range(self._opt.num_restarts):
            seeds.append(self._rng.uniform(lo, hi))
        return seeds

    def _iterate(self, q0, target, clamp, deadline):
        """Run DLS from one seed.  Returns (q, err_norm, stalled, iterations)."""
        opt = self._opt
        q = q0.copy()
        lo, hi = self._model.lower_limits, self._model.upper_limits
        history: List[float] = []
        err_norm = np.inf
        for it in range(opt.max_iterations):
            if deadline is not None and time.monotonic() > deadline:
                raise _Timeout
            err = pose_error(self._model.forward_kinematics(q), target)
            err_norm = float(np.linalg.norm(err))
            history.append(err_norm)
            if err_norm < opt.tolerance:
                return q, err_norm, False, it + 1
            jac = self._model.jacobian(q)
            # Damped least squares: dq = J^T (J J^T + lambda^2 I)^{-1} err
            jj_t = jac @ jac.T
            dq = jac.T @ np.linalg.solve(
                jj_t + (opt.damping ** 2) * np.eye(6), err
            )
            q = q + opt.step_size * dq
            if clamp:
                q = np.clip(q, lo, hi)
        stalled = False
        if len(history) > _STALL_WINDOW:
            head = history[-_STALL_WINDOW]
            tail = history[-1]
            stalled = head - tail <= _STALL_RATIO * max(head, 1e-12)
        return q, err_norm, stalled, opt.max_iterations

    def _run_passes(self, seeds, target, clamp, deadline):
        solutions: List[Tuple[np.ndarray, float, int]] = []
        best_q, best_err = None, np.inf
        total_iters = 0
        any_stalled = False
        for idx, seed in enumerate(seeds):
            q, err, stalled, iters = self._iterate(seed, target, clamp, deadline)
            total_iters += iters
            any_stalled = any_stalled or stalled
            if err < best_err:
                best_q, best_err = q.copy(), err
            if err < self._opt.tolerance:
                solutions.append((q, err, idx))
        return solutions, best_q, best_err, total_iters, len(seeds), any_stalled

    def _select(self, solutions, previous_joints) -> np.ndarray:
        """Pick the continuous solution: closest to the previous joints.

        Without a previous solution, the one closest to the zero
        configuration is chosen; ties break by residual error, then seed
        order, so selection is fully deterministic.
        """
        reference = (
            np.zeros(self._model.dof)
            if previous_joints is None
            else np.asarray(previous_joints, dtype=float)
        )

        def key(item):
            q, err, idx = item
            return (float(np.linalg.norm(q - reference)), err, idx)

        solutions.sort(key=key)
        return solutions[0][0]

    def _classify(self, seeds, target, best_q, best_err, stalled, deadline):
        """Distinguish UNREACHABLE / JOINT_LIMIT_CONFLICT / NOT_CONVERGED."""
        try:
            free_solutions, _, free_err, _, _, _ = self._run_passes(
                seeds, target, clamp=False, deadline=deadline
            )
        except _Timeout:
            return IKStatus.TIMEOUT, "solver exceeded the configured time budget"
        if free_solutions:
            return (
                IKStatus.JOINT_LIMIT_CONFLICT,
                "pose is reachable by the kinematics but violates joint limits",
            )
        if stalled:
            return (
                IKStatus.UNREACHABLE,
                "target pose is outside the reachable workspace",
            )
        return (
            IKStatus.NOT_CONVERGED,
            "iteration budget exhausted while still making progress",
        )
