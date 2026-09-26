# Robot Multi-Joint Inverse-Kinematics & Singularity Diagnosis Service

A local, deterministic, concurrency-safe inverse-kinematics (IK) service for
serial revolute arms. Given an end-effector target pose and a kinematic
description it returns a joint solution within limits, choosing the
continuity-closest solution when several exist, and — when it cannot solve —
returns a **distinguishable failure reason** instead of an approximate answer.

The configuration interface is in-memory (an injectable store); all
computation is local and reproducible.

## Features

1. **Arbitrary DOF / kinematic parameters** — revolute and fixed joints with
   configurable screw axes, link origins (xyz + fixed-axis RPY) and per-joint
   limits. Inconsistent configurations are rejected at load/reload time.
2. **Numerical IK with continuity selection** — damped least squares from
   multiple deterministic starts; all feasible converged solutions are
   candidates and the one minimizing squared joint distance to
   `previous_joints` is returned.
3. **Distinguishable failure diagnoses** — `unreachable`,
   `joint_limit_conflict`, `singular`, `did_not_converge`, `timeout`,
   `invalid_request`, `invalid_config`, `version_mismatch`. Failed responses
   never carry joint values.
4. **Singularity analysis** — SVD of the base-frame geometric Jacobian:
   smallest singular value, manipulability, and the weakest task-space
   direction. Singular / near-singular thresholds are configurable and the
   direction sign is canonicalized for deterministic output.
5. **Concurrent isolation** — each solve snapshots an immutable configuration
   generation; solver state is per-request. Timeouts are checked at every
   iteration boundary and never leave partial results.
6. **Determinism / reproducibility** — a fixed seed (config or per request)
   drives a local `numpy.random.Generator`. The number of random draws is
   fixed regardless of other inputs, and SVD directions are sign-normalized,
   so identical inputs yield identical tuples, including across threads.
7. **Hot reload with versioning** — validated installs get a monotonic
   version; in-flight solves pin the generation they started on and never mix
   old/new parameters. Clients may pin `expected_version`.

## Project layout

```
ik_service/
  models.py      immutable domain types (joints, config, pose, request, result)
  kinematics.py  URDF-convention FK, geometric Jacobian, pose error
  config.py      validation + atomic, versioned ConfigStore
  solver.py      damped least-squares IK, multi-start, singularity analysis
  service.py     facade: version pinning, per-request isolation
  api.py         FastAPI app (/health /config GET/PUT /ik POST)
tests/           50 automated cases covering all required scenarios
main.py          uvicorn entry point
```

## Running

```bash
uv sync
uv run uvicorn main:app --reload        # HTTP service
uv run pytest                           # tests
```

## Kinematic model

Each joint defines, following the URDF/ROS convention:

* `origin_xyz`, `origin_rpy` — fixed transform from the previous link,
  RPY as fixed-axis `Rz(y) Ry(p) Rx(r)`;
* `axis` — unit screw axis in the joint's post-origin frame (normalized
  internally for revolute joints);
* joint transform `T(q) = T(origin) · exp(axis_screw · q)`;
* `lower_limit`, `upper_limit` — hard joint limits.

Fixed joints add only their origin transform and no DOF. The geometric
Jacobian `J(q) ∈ R^{6×n}` is assembled in the base frame
(`[ω; v]` rows, revolute columns `[z; z × (p_ee − p_joint)]`).

## Numerical method

* **Error.** Position error `p_d − p`; orientation error
  `0.5·vee(R_d R^T − R R_d^T)` (vanishes exactly at the target).
* **Update.** Damped least squares (Levenberg–Marquardt), n×n damped normal
  equations: `Δq = (JᵀJ + λ²I)^{−1} Jᵀ e`. Tolerance scaling
  (`1/pos_tol`, `1/ori_tol`) is applied only to the scalar merit used for
  line search, metric and diagnosis — never to the linear system, which keeps
  the system well-conditioned even for robots with structurally zero
  task-space rows (e.g. planar arms).
* **Trust region + line search.** Each step is clipped to a maximum joint
  radius of 0.5 rad to prevent pseudo-inverse overshoots near singularities,
  then accepted via deterministic backtracking
  `α ∈ {1, 0.5, 0.25, 0.125}` requiring strict merit decrease.
* **Limits.** Every start is solved on two tracks: one projected onto the
  box limits (`clip`) and one unconstrained (`relaxed`). Projection keeps
  iterates feasible; the relaxed track only feeds failure classification.
* **Starts (fixed order).** `previous_joints` (if given), clipped zero,
  limit midpoint, then exactly `restarts` uniform draws in
  `[lower, upper]` from the seeded RNG. Track iteration count and result
  recording are independent of whether `previous_joints` was supplied.
* **Convergence.** All three position components ≤ `position_tolerance` and
  all three orientation components ≤ `orientation_tolerance`
  (radians). Before returning, the joint vector is re-checked against FK and
  the limits.
* **Budgets.** `max_iterations` per track and `timeout_seconds` wall-clock
  budget (overridable per request). The deadline is checked at the top of
  every iteration; on expiry the solver returns immediately with no joint
  vector.

## Failure diagnosis semantics

| Reason | Meaning | Evidence required |
| --- | --- | --- |
| `invalid_request` | malformed pose, wrong DOF count, out-of-limit seed, non-finite values | validation only |
| `version_mismatch` | `expected_version` differs from the active generation | validation only |
| `joint_limit_conflict` | a relaxed track converged **within tolerance** but violates a limit, while every limit-projected track stayed outside tolerance | relaxed converged + clamped not converged |
| `singular` | iteration stalled at a rank-deficient Jacobian (`σ_min < singular_threshold`) with residual still near the solution scale | SVD at the best candidate |
| `did_not_converge` | the iteration budget was fully spent while residual was still decreasing, or tracks stalled within 10× the tolerance norm | bookkeeping |
| `unreachable` | every track stalled naturally far outside tolerance (best scaled residual > 10× the tolerance norm) | residual gap |
| `timeout` | the wall-clock budget expired before convergence | deadline check |

Scaled residual norms are in tolerance units, so a norm of `√6` means "every
component is exactly at its tolerance"; the table's factors are scale-free.

## Singularity report

For every successful solution (and near-miss candidate):

* `smallest_singular_value` — `σ_min(J)` from the eigenvalues of `JJᵀ`;
* `manipulability` — product of singular values (Yoshikawa measure);
* `is_singular` — `σ_min < singular_threshold`;
* `is_near_singular` — `σ_min < near_singular_threshold`;
* `affected_direction` — unit left singular vector for `σ_min`, a base-frame
  twist `[ω_x, ω_y, ω_z, v_x, v_y, v_z]` identifying the task direction that
  cannot be produced. Its sign is canonicalized (largest-magnitude component
  non-negative) so results are bit-for-bit reproducible.

Structurally deficient robots (e.g. a planar arm asked about full 6D pose)
report `is_singular=true` with three exact zero singular values; their
solutions still succeed on the achievable subspace.

## HTTP contract

* `GET /health` — service status, active robot and version.
* `GET /config` — active configuration.
* `PUT /config` — validate + atomically install a new generation
  (`422 {"reason":"invalid_config"}` on rejection; the old generation stays
  active).
* `POST /ik` — body:

```json
{
  "target": {"position": [x, y, z], "quaternion_wxyz": [w, x, y, z]},
  "previous_joints": [0.1, -0.2, 0.0],
  "expected_version": 2,
  "seed_override": 0,
  "timeout_override": 2.0
}
```

Success → `200` with `joints`, residuals, `iterations`, `restarts_used`,
`config_version` and the singularity report. Any diagnosed failure → `422`
with `joints: null` and a machine-readable `reason`
(`unreachable` | `joint_limit_conflict` | `singular` | `did_not_converge` |
`timeout` | `invalid_request` | `version_mismatch`). Schema errors also
return `422`.

## Determinism guarantees

* `random_seed` in config; `seed_override` per request wins.
* Fixed number of RNG draws (`restarts`) in a fixed start order.
* Deterministic backtracking and candidate ordering.
* SVD weak-direction signs canonicalized; `eigvalsh` for scalar metrics.
* Wall-clock budget can only change `success`/`timeout`, never the chosen
  solution for a run that completes (timeout is checked, not used in
  arithmetic).
