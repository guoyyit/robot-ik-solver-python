"""Service facade around the stateless numerical solver.

Responsibilities beyond :func:`ik_service.solver.solve_ik`:

* pin the configuration generation for the whole request: the snapshot is
  read once up front and passed to the solver, so a concurrent hot reload can
  never mix old and new parameters mid-solve;
* give every concurrent request its own :class:`~ik_service.models.IKRequest`
  data and solver-local RNG/arrays (the solver holds no shared mutable state);
* apply the per-request timeout/iteration budget configured at request start.
"""

from __future__ import annotations

from .config import ConfigStore
from .models import IKRequest, IKResult
from .solver import solve_ik


class IKService:
    """Thin thread-safe facade. Stateless apart from the config store."""

    def __init__(self, store: ConfigStore) -> None:
        self._store = store

    @property
    def store(self) -> ConfigStore:
        return self._store

    def solve(self, request: IKRequest) -> IKResult:
        # Snapshot once: version pinning for the entire solve. A reload after
        # this line only affects subsequent requests.
        config = self._store.current()
        return solve_ik(config, request)
