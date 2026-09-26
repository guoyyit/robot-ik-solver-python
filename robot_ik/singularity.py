"""Singularity detection and proximity evaluation via Jacobian SVD.

Semantics (deterministic, driven by one configurable threshold ``t``):

* ``min_singular_value``  – smallest singular value of the 6 x n Jacobian
  (zero-padded when the arm has fewer than 6 joints).
* ``manipulability``      – product of all singular values (Yoshikawa).
* ``is_near_singular``    – ``min_singular_value < t``.
* ``is_singular``         – ``min_singular_value < t * 1e-3`` (numerically
  rank-deficient).  ``is_singular`` implies ``is_near_singular``.
* ``degenerate_directions`` – left singular vectors (task-space 6D
  directions, linear xyz then angular xyz) whose singular value is below
  ``t``; these are the directions the arm cannot (or can barely) move along.
"""

from __future__ import annotations

import numpy as np

from .types import SingularityReport

_SINGULAR_RATIO = 1e-3


def analyze_singularity(jacobian: np.ndarray, threshold: float) -> SingularityReport:
    """Classify a Jacobian as singular / near-singular and report directions."""
    j = np.asarray(jacobian, dtype=float)
    # full_matrices=True so U always spans the 6D task space; directions
    # beyond the number of joints have an implicit singular value of zero.
    u, s, _ = np.linalg.svd(j, full_matrices=True)
    n_dirs = u.shape[1]
    padded = np.zeros(n_dirs)
    padded[: len(s)] = s
    min_sv = float(np.min(padded)) if n_dirs > 0 else 0.0
    manipulability = float(np.prod(s)) if len(s) > 0 else 0.0

    degenerate = tuple(
        tuple(float(x) for x in u[:, k]) for k in range(n_dirs) if padded[k] < threshold
    )
    is_singular = min_sv < threshold * _SINGULAR_RATIO
    return SingularityReport(
        is_singular=is_singular,
        is_near_singular=is_singular or min_sv < threshold,
        min_singular_value=min_sv,
        manipulability=manipulability,
        degenerate_directions=degenerate,
    )
