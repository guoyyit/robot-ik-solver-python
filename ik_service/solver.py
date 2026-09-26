"""Numerical inverse kinematics.

Method
------
* Damped least squares (Levenberg-Marquardt):
  ``dq = Js^T (Js Js^T + lambda^2 I)^{-1} e`` where ``Js`` is the geometric
  Jacobian scaled by the position/orientation tolerances so that convergence
  is a component-wise check against configured tolerances.
* Joint limits are handled by projection (clipping) with a short
  deterministic backtracking line search.
* Multiple deterministic starts are tried: the previous solution (continuity),
  the clipped zero pose, the limit midpoint, then ``restarts`` draws from a
  seeded ``numpy.random.Generator``. The number of random draws is fixed, so
  presence/absence of ``previous_joints`` never perturbs the random stream.
* Each start is also solved *without* limit projection; convergence only in
  that relaxed track distinguishes a joint-limit conflict from an unreachable
  target.
* Timeout/iteration budget are checked at every iteration boundary; on
  timeout no partial joint vector is returned.
"""

from __future__ import annotations

import math
import time

import numpy as np

from .kinematics import (
    forward_kinematics,
    geometric_jacobian,
    pose_error,
    quaternion_to_rotation,
)
from .models import FailureReason, IKRequest, IKResult, Pose, RobotConfig, SingularityReport

_ALPHAS: Final = (1.0, 0.5, 0.25, 0.125)
_STALL_LIMIT: Final = 8
_NEAR_CONVERGED_FACTOR: Final = 1.0e1
_MAX_STEP_RADIANS: Final = 0.5


def _failure(
    config: RobotConfig,
    reason: FailureReason,
    message: str,
    *,
    iterations: int = 0,
    restarts: int = 0,
) -> IKResult:
    return IKResult(
        success=False,
        joints=None,
        reason=reason,
        message=message,
        singularity=None,
        iterations=iterations,
        restarts_used=restarts,
        config_version=config.version,
    )


def _validate_request(
    config: RobotConfig, request: IKRequest
) -> tuple[np.ndarray, np.ndarray, float, int, str | None]:
    n = sum(1 for j in config.joints if j.joint_type.value == "revolute")
    target: Pose = request.target
    p = target.position
    qw = target.quaternion_wxyz
    if not (isinstance(p, tuple) and len(p) == 3 and all(math.isfinite(float(v)) for v in p)):
        return _bad("target.position must be 3 finite numbers")
    if not (isinstance(qw, tuple) and len(qw) == 4 and all(math.isfinite(float(v)) for v in qw)):
        return _bad("target.quaternion_wxyz must be 4 finite numbers")
    norm = math.sqrt(sum(float(v) ** 2 for v in qw))
    if norm == 0.0:
        return _bad("target quaternion must not be zero-length")
    if request.previous_joints is not None:
        prev = request.previous_joints
        if not (isinstance(prev, tuple) and len(prev) == n):
            return _bad(f"previous_joints must have length {n}")
        if not all(isinstance(v, (int, float)) and math.isfinite(float(v)) for v in prev):
            return _bad("previous_joints must contain finite numbers")
        lo = np.array([j.lower_limit for j in config.joints if j.joint_type.value == "revolute"])
        hi = np.array([j.upper_limit for j in config.joints if j.joint_type.value == "revolute"])
        arr = np.asarray(prev, dtype=np.float64)
        if np.any(arr < lo - 1e-9) or np.any(arr > hi + 1e-9):
            return _bad("previous_joints must lie within the configured joint limits")
    if request.expected_version is not None and request.expected_version != config.version:
        return _bad(
            f"expected config version {request.expected_version}, active is {config.version}"
        )
    timeout = config.timeout_seconds
    if request.timeout_override is not None:
        t = request.timeout_override
        if not isinstance(t, (int, float)) or isinstance(t, bool) or not math.isfinite(float(t)) or t <= 0:
            return _bad("timeout_override must be a positive finite number of seconds")
        timeout = float(t)
    seed = config.random_seed if request.seed_override is None else request.seed_override
    if not isinstance(seed, int):
        return _bad("seed must be an integer")
    return (
        quaternion_to_rotation(qw),
        np.asarray(p, dtype=np.float64),
        float(timeout),
        int(seed),
        None,
    )


def _bad(message: str):
    return None, None, 0.0, 0, message


def _limits(config: RobotConfig) -> tuple[np.ndarray, np.ndarray]:
    lo = np.array(
        [j.lower_limit for j in config.joints if j.joint_type.value == "revolute"],
        dtype=np.float64,
    )
    hi = np.array(
        [j.upper_limit for j in config.joints if j.joint_type.value == "revolute"],
        dtype=np.float64,
    )
    return lo, hi


def _initial_guesses(
    config: RobotConfig, request: IKRequest, rng: np.random.Generator
) -> list[np.ndarray]:
    lo, hi = _limits(config)
    zero = np.clip(np.zeros_like(lo), lo, hi)
    midpoint = 0.5 * (lo + hi)
    guesses: list[np.ndarray] = []
    if request.previous_joints is not None:
        guesses.append(np.asarray(request.previous_joints, dtype=np.float64))
    guesses.extend([zero, midpoint])
    # Fixed number of draws regardless of other inputs -> stable stream.
    for _ in range(config.restarts):
        guesses.append(rng.uniform(lo, hi))
    return guesses


def _converged(config: RobotConfig, e: np.ndarray) -> bool:
    e_rot = e[:3]
    e_pos = e[3:]
    return (
        bool(np.all(np.abs(e_rot) <= config.orientation_tolerance))
        and bool(np.all(np.abs(e_pos) <= config.position_tolerance))
    )


def _scaled(config: RobotConfig, e: np.ndarray) -> np.ndarray:
    scale = np.concatenate(
        [
            np.full(3, 1.0 / config.orientation_tolerance),
            np.full(3, 1.0 / config.position_tolerance),
        ]
    )
    return e * scale


def _run_track(
    config: RobotConfig,
    q0: np.ndarray,
    target_r: np.ndarray,
    target_p: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    *,
    clamp: bool,
    deadline: float,
) -> tuple[np.ndarray, bool, int, float, bool, bool]:
    """One DLS track.

    Returns ``(q, converged, iterations, best_scaled_norm, timed_out,
    budget_exhausted)``; ``budget_exhausted`` is True when the track spent its
    full iteration allowance (the budget terminated the search, not a stall).
    """
    q = q0.copy()
    if clamp:
        q = np.clip(q, lo, hi)
    best_norm = math.inf
    best_q = q.copy()
    stall = 0
    lam2 = config.damping * config.damping
    it = 0
    for it in range(1, config.max_iterations + 1):
        if time.monotonic() >= deadline:
            return best_q, False, it - 1, best_norm, True, False
        t_cur = forward_kinematics(config, q)
        e = pose_error(t_cur, target_r, target_p)
        norm = float(np.linalg.norm(_scaled(config, e)))
        if norm < best_norm:
            if best_norm - norm < 1e-13:
                stall += 1
            else:
                stall = 0
            best_norm = norm
            best_q = q.copy()
        else:
            stall += 1
        if _converged(config, e):
            return q, True, it, norm, False, False
        if stall >= _STALL_LIMIT:
            return best_q, False, it, best_norm, False, False
        jac = geometric_jacobian(config, q)
        # DLS on the raw Jacobian/error, n x n damped normal equations.
        # The scaled error drives only line-search/metrics/convergence, so
        # tolerances stay component-wise while the linear system is well
        # conditioned even when whole task-space rows of J are zero.
        n_dof = q.shape[0]
        lhs = jac.T @ jac + lam2 * np.eye(n_dof)
        dq = np.linalg.solve(lhs, jac.T @ e)
        # Trust-region step clipping: near singularities a small damping still
        # admits huge pseudo-inverse jumps. Line search alone can accept an
        # overshoot that lands on joint limits and kills subsequent progress,
        # so cap the joint-space step to a fixed radius.
        step_norm = float(np.linalg.norm(dq))
        if step_norm > _MAX_STEP_RADIANS:
            dq = dq * (_MAX_STEP_RADIANS / step_norm)
        accepted_q = q
        accepted_norm = norm
        moved = False
        for alpha in _ALPHAS:
            cand = q + alpha * dq
            if clamp:
                cand = np.clip(cand, lo, hi)
            if np.allclose(cand, q, atol=0.0, rtol=0.0):
                continue
            t_cand = forward_kinematics(config, cand)
            cand_norm = float(np.linalg.norm(_scaled(config, pose_error(t_cand, target_r, target_p))))
            if cand_norm < accepted_norm:
                accepted_q = cand
                accepted_norm = cand_norm
                moved = True
        if not moved:
            return best_q, False, it, best_norm, False, False
        q = accepted_q
    # Loop completed normally -> the iteration budget was fully spent.
    return best_q, False, it, best_norm, False, True


def _canonicalize(vec: np.ndarray) -> np.ndarray:
    """Deterministic sign: largest-magnitude component is non-negative."""
    k = int(np.argmax(np.abs(vec)))
    if vec[k] < 0.0:
        vec = -vec
    return vec


def singularity_report(
    config: RobotConfig, q: np.ndarray
) -> SingularityReport:
    """SVD analysis of the base-frame geometric Jacobian at ``q``."""
    jac = geometric_jacobian(config, q)
    # J is 6 x n; singular values of (J J^T) describe task-space mobility.
    gram = jac @ jac.T
    eig = np.linalg.eigvalsh(gram)
    eig = np.clip(eig, 0.0, None)
    sigmas = np.sqrt(eig)
    sigma_min = float(sigmas[0])
    manipulability = float(np.prod(sigmas))
    u, _, _ = np.linalg.svd(jac, full_matrices=False)
    direction = _canonicalize(np.asarray(u[:, -1], dtype=np.float64))
    near = sigma_min < config.near_singular_threshold
    singular = sigma_min < config.singular_threshold
    return SingularityReport(
        is_singular=singular,
        is_near_singular=near,
        smallest_singular_value=sigma_min,
        manipulability=manipulability,
        affected_direction=tuple(float(v) for v in direction),
    )


def solve_ik(config: RobotConfig, request: IKRequest) -> IKResult:
    """Solve one request. Pure function of its immutable arguments."""
    target_r, target_p, timeout, seed, err = _validate_request(config, request)
    if err is not None:
        reason = (
            FailureReason.VERSION_MISMATCH
            if err.startswith("expected config version")
            else FailureReason.INVALID_REQUEST
        )
        return _failure(config, reason, err)

    rng = np.random.default_rng(seed)
    lo, hi = _limits(config)
    guesses = _initial_guesses(config, request, rng)
    deadline = time.monotonic() + timeout

    successes: list[tuple[np.ndarray, SingularityReport, int, int, float, float]] = []
    # Evidence of a genuine limit conflict: a relaxed start converged to the
    # target (within tolerance) but its joint vector violates the limits,
    # while the corresponding clamped start stayed far from the target.
    relaxed_converged_violating: list[tuple[np.ndarray, float]] = []
    best_clamped_norm = math.inf
    best_any: tuple[np.ndarray, float] | None = None
    total_iter = 0
    timed_out = False
    restarts_used = 0
    budget_exhausted = False

    for idx, q0 in enumerate(guesses):
        restarts_used = idx + 1
        q_rel, conv_rel, it_rel, norm_rel, to_rel, budget_rel = _run_track(
            config, q0, target_r, target_p, lo, hi, clamp=False, deadline=deadline
        )
        total_iter += it_rel
        if to_rel:
            timed_out = True
            break
        budget_exhausted = budget_exhausted or budget_rel
        if best_any is None or norm_rel < best_any[1]:
            best_any = (q_rel, norm_rel)
        if conv_rel and (np.any(q_rel < lo - 1e-9) or np.any(q_rel > hi + 1e-9)):
            relaxed_converged_violating.append((q_rel, norm_rel))

        q_cl, conv_cl, it_cl, norm_cl, to_cl, budget_cl = _run_track(
            config, q0, target_r, target_p, lo, hi, clamp=True, deadline=deadline
        )
        total_iter += it_cl
        if to_cl:
            timed_out = True
            break
        budget_exhausted = budget_exhausted or budget_cl
        best_clamped_norm = min(best_clamped_norm, norm_cl)
        if best_any is None or norm_cl < best_any[1]:
            best_any = (q_cl, norm_cl)
        if conv_cl and np.all(q_cl >= lo - 1e-12) and np.all(q_cl <= hi + 1e-12):
            report = singularity_report(config, q_cl)
            e = pose_error(forward_kinematics(config, q_cl), target_r, target_p)
            successes.append(
                (
                    q_cl,
                    report,
                    total_iter,
                    restarts_used,
                    float(np.linalg.norm(e[3:])),
                    float(np.linalg.norm(e[:3])),
                )
            )

    if timed_out:
        return _failure(
            config,
            FailureReason.TIMEOUT,
            f"solver exceeded the {timeout:.6g}s time budget before converging",
            iterations=total_iter,
            restarts=restarts_used,
        )

    if successes:
        if request.previous_joints is not None:
            prev = np.asarray(request.previous_joints, dtype=np.float64)
            best = min(successes, key=lambda s: float(np.sum((s[0] - prev) ** 2)))
        else:
            best = successes[0]
        q, report, iters, used, res_p, res_r = best
        return IKResult(
            success=True,
            joints=tuple(float(v) for v in q),
            reason=None,
            message="converged",
            singularity=report,
            iterations=iters,
            restarts_used=used,
            config_version=config.version,
            residual_position=res_p,
            residual_orientation=res_r,
        )

    tol_norm = math.sqrt(3.0 * config.position_tolerance**2 + 3.0 * config.orientation_tolerance**2)
    scale_norm = math.sqrt(6.0)
    best_overall = best_any[1] if best_any is not None else math.inf

    if relaxed_converged_violating and best_clamped_norm > scale_norm:
        return _failure(
            config,
            FailureReason.JOINT_LIMIT_CONFLICT,
            "the target is reachable only by violating one or more joint "
            "limits; every limit-satisfying start stayed outside tolerance",
            iterations=total_iter,
            restarts=restarts_used,
        )

    near_q = best_any[0] if best_any is not None else np.zeros(len(lo))
    near_report = singularity_report(config, near_q)
    if near_report.is_singular and best_overall <= _NEAR_CONVERGED_FACTOR * scale_norm:
        return _failure(
            config,
            FailureReason.SINGULAR,
            "iteration stalled at a singular configuration",
            iterations=total_iter,
            restarts=restarts_used,
        )
    if budget_exhausted:
        # At least one track spent its full iteration budget while still
        # progressing: the numerical search ran out of budget, so calling the
        # target "unreachable" would silently misreport an unfinished solve.
        return _failure(
            config,
            FailureReason.DID_NOT_CONVERGE,
            f"iteration budget exhausted while residual was still decreasing; "
            f"best scaled residual {best_overall:.6g}",
            iterations=total_iter,
            restarts=restarts_used,
        )
    if best_overall <= _NEAR_CONVERGED_FACTOR * scale_norm:
        return _failure(
            config,
            FailureReason.DID_NOT_CONVERGE,
            f"iteration budget exhausted; best scaled residual {best_overall:.6g}",
            iterations=total_iter,
            restarts=restarts_used,
        )
    return _failure(
        config,
        FailureReason.UNREACHABLE,
        f"target outside reachable workspace; best scaled residual {best_overall:.6g} "
        f"(tolerance norm {tol_norm:.6g})",
        iterations=total_iter,
        restarts=restarts_used,
    )
