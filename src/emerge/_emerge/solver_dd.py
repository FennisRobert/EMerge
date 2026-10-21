# solver_dd.py
"""
Domain-decomposition solver support.

Provides a `Factorization` object that factorizes a (sub)domain matrix once
and can then be reused for repeated solves against new right-hand-side
vectors, plus a lightweight `DDSolver` manager that keeps a dict of these
per subdomain.

Currently only EMSolver.SUPERLU is supported. Support for solvers backed by
persistent external contexts (PARDISO, MUMPS, AASDS, cuDSS) is deliberately
left out until their interfaces are confirmed safe to run as multiple
concurrent, independent factorizations.
"""
from __future__ import annotations

import time
from typing import Hashable

import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu, SuperLU  # type: ignore
from loguru import logger

from .solver import EMSolver, SolveReport, _SKSP_AVAILABLE, _pfx  # adjust to your actual module name

if _SKSP_AVAILABLE:
    from sksparse.cholmod import metis  # type: ignore


_SUPPORTED_DD_SOLVERS = (EMSolver.SUPERLU,)


class Factorization:
    """Holds a persistent factorization of a matrix A, so it can be solved
    against many right-hand-side vectors b without refactorizing.

    Typical use:
        f = Factorization(EMSolver.SUPERLU)
        f.factorize(A, solve_ids=free_dofs)
        x1, report1 = f.solve(b1)
        x2, report2 = f.solve(b2)   # reuses the same LU factors
    """

    def __init__(
        self,
        solver_type: EMSolver = EMSolver.SUPERLU,
        pre: str = "",
        pivoting_threshold: float = 0.001,
    ):
        if solver_type not in _SUPPORTED_DD_SOLVERS:
            raise NotImplementedError(
                f"Factorization currently only supports {[s.name for s in _SUPPORTED_DD_SOLVERS]}, "
                f"got {solver_type.name}. Extend solver_dd.py once the corresponding solver "
                f"interface is confirmed safe for multiple independent, concurrent factorizations."
            )
        self.solver_type: EMSolver = solver_type
        self.pre: str = pre
        self._pivoting_threshold: float = pivoting_threshold
        self.options: dict[str, str] = dict(
            SymmetricMode=True, Equil=False, IterRefine="SINGLE"
        )

        # Persistent factorization state
        self.lu: SuperLU | None = None
        self._perm: np.ndarray | None = None

        # Problem bookkeeping needed to map full <-> reduced DOF sets
        self.solve_ids: np.ndarray | None = None
        self.n_full: int | None = None
        self.dtype = None

        self.factor_report: SolveReport | None = None

    def __str__(self) -> str:
        state = "factorized" if self.is_factorized else "empty"
        return f"Factorization({self.solver_type.name}, {state})"

    @property
    def is_factorized(self) -> bool:
        return self.lu is not None

    def set_pivoting_threshold(self, pivoting_threshold: float) -> None:
        """Change the diagonal pivoting threshold used on the next factorize()."""
        self._pivoting_threshold = pivoting_threshold

    def factorize(
        self,
        A: csc_matrix,
        solve_ids: np.ndarray | None = None,
    ) -> SolveReport:
        """Factorize matrix A (optionally restricted to `solve_ids` rows/cols)
        and store the factorization for repeated solves.

        Args:
            A: The full sparse system matrix.
            solve_ids: Optional array of free DOF indices (e.g. non-PEC nodes).
                If given, A is reduced to A[solve_ids, :][:, solve_ids] before
                factorization, and solve() will scatter results back into the
                full-size solution vector.

        Returns:
            SolveReport describing the factorization step.
        """
        logger.info(f"{_pfx(self.pre)} Factorizing (SuperLU, DD).")
        start = time.time()

        self.n_full = A.shape[0]
        self.solve_ids = solve_ids
        Asel = A[:, solve_ids][solve_ids, :] if solve_ids is not None else A
        self.dtype = Asel.dtype

        if _SKSP_AVAILABLE:
            logger.trace(f"{_pfx(self.pre)} Computing METIS permutation.")
            self._perm = metis(Asel)
            Aordered = Asel[self._perm][:, self._perm]
            permc = "NATURAL"
        else:
            self._perm = None
            Aordered = Asel
            permc = "MMD_AT_PLUS_A"

        self.lu = splu(
            Aordered,
            permc_spec=permc,
            relax=0,
            diag_pivot_thresh=self._pivoting_threshold,
            options=self.options,
        )

        simtime = time.time() - start
        logger.info(f"{_pfx(self.pre)} Factorization complete in {simtime:.3f}s.")

        report = SolveReport(
            solver="SUPERLU-DD",
            simtime=simtime,
            ndof=self.n_full,
            nnz=A.nnz,
            ndof_solve=Asel.shape[0],
            nnz_solve=Asel.nnz,
            worker_name=self.pre,
            aux={"pivoting threshold": str(self._pivoting_threshold)},
        )
        self.factor_report = report
        return report

    def refactorize(self, A: csc_matrix, solve_ids: np.ndarray | None = None) -> SolveReport:
        """Discard the current factorization and factorize a new matrix
        (e.g. after the subdomain's material/mesh properties change)."""
        self.reset()
        return self.factorize(A, solve_ids)

    def solve(self, b: np.ndarray) -> tuple[np.ndarray, SolveReport]:
        """Solve A x = b (or the reduced system) against the stored
        factorization. Accepts b as a 1D vector or a 2D (ndof, nrhs) block.
        """
        if not self.is_factorized:
            raise RuntimeError(
                f"{_pfx(self.pre)} Factorization.solve() called before factorize(). "
                f"Call .factorize(A) first."
            )

        start = time.time()
        b = np.asarray(b)
        was_1d = b.ndim == 1
        if was_1d:
            b = b.reshape(-1, 1)

        bsel = b[self.solve_ids, :] if self.solve_ids is not None else b

        if self._perm is not None:
            x = np.empty_like(bsel)
            for i in range(bsel.shape[1]):
                bp = bsel[self._perm, i]
                xp = self.lu.solve(bp)
                x[self._perm, i] = xp
        else:
            x = self.lu.solve(bsel)

        if self.solve_ids is not None:
            solution = np.zeros((self.n_full, b.shape[1]), dtype=self.dtype)
            solution[self.solve_ids, :] = x
        else:
            solution = x

        if was_1d:
            solution = solution.ravel()

        simtime = time.time() - start
        report = SolveReport(
            solver="SUPERLU-DD",
            simtime=simtime,
            ndof=self.n_full,
            ndof_solve=bsel.shape[0],
            worker_name=self.pre,
        )
        return solution, report

    def reset(self) -> None:
        """Drop the stored factorization, freeing the underlying SuperLU object."""
        self.lu = None
        self._perm = None
        self.solve_ids = None
        self.n_full = None
        self.dtype = None
        self.factor_report = None


class DDSolver:
    """Manages a collection of independent Factorization objects, one per
    subdomain, keyed by whatever hashable id you use for subdomains
    (int, str, tuple, ...).

    Example:
        dd = DDSolver()
        dd.factorize("dom1", A1, solve_ids=free1)
        dd.factorize("dom2", A2, solve_ids=free2)
        x1 = dd.solve("dom1", b1)
        x1_new = dd.solve("dom1", b1_updated)   # no refactorization
    """

    def __init__(self, solver_type: EMSolver = EMSolver.SUPERLU):
        self.solver_type: EMSolver = solver_type
        self.factorizations: dict[Hashable, Factorization] = {}

    def __contains__(self, key: Hashable) -> bool:
        return key in self.factorizations

    def __len__(self) -> int:
        return len(self.factorizations)

    def factorize(
        self,
        key: Hashable,
        A: csc_matrix,
        solve_ids: np.ndarray | None = None,
        pivoting_threshold: float = 0.001,
    ) -> SolveReport:
        """Factorize (or refactorize) the subdomain identified by `key`."""
        f = self.factorizations.get(key)
        if f is None:
            f = Factorization(self.solver_type, pre=f"dd-{key}", pivoting_threshold=pivoting_threshold)
            self.factorizations[key] = f
        else:
            f.set_pivoting_threshold(pivoting_threshold)
        return f.factorize(A, solve_ids)

    def solve(self, key: Hashable, b: np.ndarray) -> np.ndarray:
        """Solve for subdomain `key` against its stored factorization, return x only."""
        return self.factorizations[key].solve(b)[0]

    def solve_report(self, key: Hashable, b: np.ndarray) -> tuple[np.ndarray, SolveReport]:
        """Same as solve(), but also returns the SolveReport."""
        return self.factorizations[key].solve(b)

    def release(self, key: Hashable) -> None:
        """Drop the factorization for a single subdomain."""
        f = self.factorizations.pop(key, None)
        if f is not None:
            f.reset()

    def release_all(self) -> None:
        """Drop all stored factorizations."""
        for f in self.factorizations.values():
            f.reset()
        self.factorizations.clear()