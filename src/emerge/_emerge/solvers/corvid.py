# EMerge is an open source Python based FEM EM simulation module.
# Copyright (C) 2025  Robert Fennis.

# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 2
# of the License, or (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program; if not, see
# <https://www.gnu.org/licenses/>.

"""The CORVID iterative solver for complex symmetric systems."""
from __future__ import annotations
from typing import Literal
import numpy as np
from scipy.sparse import csc_matrix  # type: ignore
from loguru import logger
from ..precon_data import PreconData
from .solver_base import Solver, SolverType, SolveReport, SimulationError, _pfx

try:
    import corvid  # type: ignore
    AVAILABLE = True
    # [Written by Claude] corvid.Solver(threads=...) exists from CORVID 0.4.2 on
    _HAS_THREADS = hasattr(corvid, "set_num_threads")
    # [Written by Claude] From CORVID 0.5.0 one Solver is kept across a sweep: factorize() per frequency
    # refactors in place (or keeps the factors of a nearby frequency) and solve() starts from the
    # previous frequencies' solutions
    import inspect as _inspect
    _HAS_SWEEP = "warm_start" in _inspect.signature(corvid.Solver).parameters
except ModuleNotFoundError:
    logger.trace("CORVID Interface not found. (not harmful)")
    AVAILABLE = False
    _HAS_THREADS = False
    _HAS_SWEEP = False


class SolverCorvid(Solver):
    """Implements the CORVID preconditioned iterative solver.

    CORVID combines vertex-block Schwarz smoothing with a Whitney-level coarse solve inside a GMRES or
    COCG Krylov loop. Next to A and b it needs the mesh topology of the solved DoFs, which the SolveRoutine
    hands over through set_precon_data() before every solve.

    Requires the corvid package.
    """

    available = AVAILABLE
    requires = "the corvid package"

    req_sorter = False  # rows must stay in solve_ids order to match the topology
    real_only = False
    stype = SolverType.PARALLEL  # CORVID runs its own thread pool and keeps the GIL
    name = "CORVID"
    needs_precon_data = True

    def __init__(self, pre: str):
        super().__init__(pre)
        self.own_preconditioner = True
        self.method: Literal["gmres", "cocg"] = "cocg"
        self.order: int = 3  # [Written by Claude] order 3: fewer (serial) coarse solves per solve
        self.rtol: float = 1e-6
        self.maxiter: int = 500
        self.restart: int = 100
        self.nthreads: int | None = None
        self._topology = None
        self._topology_key: tuple | None = None
        self._k0: float | None = None
        # [Written by Claude] the CORVID solver kept across the frequencies of a sweep, and its settings
        self._sol = None
        self._sol_key: tuple | None = None

    def set_options(
        self,
        order: int | None = None,
        rtol: float | None = None,
        maxiter: int | None = None,
        restart: int | None = None,
    ) -> SolverCorvid:
        """Set the CORVID solver options. Options that are None are left unchanged.

        Args:
            order (int | None, optional): The Chebyshev order of the smoother. Defaults to None.
            rtol (float | None, optional): The relative residual tolerance. Defaults to None.
            maxiter (int | None, optional): The maximum number of Krylov iterations. Defaults to None.
            restart (int | None, optional): The GMRES restart length. Defaults to None.
        """
        if order is not None:
            self.order = order
        if rtol is not None:
            self.rtol = rtol
        if maxiter is not None:
            self.maxiter = maxiter
        if restart is not None:
            self.restart = restart
        return self

    # [Written by Claude]
    def set_nthreads(self, nthreads: int | None) -> SolverCorvid:
        """Install the number of CPU threads CORVID uses for its factorization (vertex blocks and the exact
        level's LDL^T) and its Krylov solves. None leaves it to CORVID (corvid.get_num_threads():
        RAYON_NUM_THREADS, otherwise the number of cores up to 8).

        Args:
            nthreads (int | None): The number of threads.
        """
        if not _HAS_THREADS:
            return super().set_nthreads(nthreads)
        if nthreads is not None and int(nthreads) < 1:
            raise ValueError(f"CORVID needs at least one thread, got {nthreads}")
        self.nthreads = None if nthreads is None else int(nthreads)
        return self

    def set_symmetry(self, complex_symmetric: bool) -> None:
        # COCG stores no Krylov basis but requires a complex symmetric matrix
        self.method = "cocg" if complex_symmetric else "gmres"

    def set_precon_data(self, precon_dataset: PreconData, solve_ids: np.ndarray, k0: float | None = None) -> None:
        """Build (or reuse) the CORVID topology of the solved DoFs.

        With the node coordinates and k0, CORVID also solves the unknowns next to coarsely meshed metal walls
        exactly (its wall rule), which strongly reduces the iterations on coarse meshes.

        Args:
            precon_dataset (PreconData): The mesh topology per global DoF
            solve_ids (np.ndarray): The solved global DoFs, in the row order of the solved system
            k0 (float | None, optional): The free space wave number of this solve. Defaults to None.
        """
        self._k0 = k0
        key = (id(precon_dataset.dof_entity), solve_ids.shape[0], hash(solve_ids.tobytes()))
        if key == self._topology_key:
            return
        logger.debug(self.pre + " Building CORVID topology")
        self._topology = corvid.Topology(
            precon_dataset.dof_entity[solve_ids],
            precon_dataset.dof_kind[solve_ids],
            precon_dataset.edges,
            precon_dataset.tris,
            wall_vertices=precon_dataset.wall_vertices(solve_ids),
            coords=precon_dataset.nodes,
        )
        self._topology_key = key

    def reset(self) -> None:
        self._topology = None
        self._topology_key = None
        self._k0 = None
        self._sol = None
        self._sol_key = None

    def duplicate(self) -> Solver:
        new_solver = self.__class__(self.pre)
        new_solver.set_options(self.order, self.rtol, self.maxiter, self.restart)
        # [Written by Claude] Worker copies keep the thread count
        return new_solver.set_nthreads(self.nthreads)

    def solve(self, A: csc_matrix, b: np.ndarray, precon, id: int = -1) -> tuple[np.ndarray, SolveReport]:
        if self._topology is None:
            raise SimulationError("CORVID needs the mesh topology. Call set_precon_data() before solve().")
        # [Written by Claude] threads only exists from CORVID 0.4.2 on
        thread_kw = {"threads": self.nthreads} if _HAS_THREADS else {}
        threads = self.nthreads if self.nthreads is not None else (corvid.get_num_threads() if _HAS_THREADS else "default")
        logger.info(f"{_pfx(self.pre, id)} Calling CORVID solver ({self.method}, {threads} threads).")
        # [Written by Claude] keep one CORVID solver per topology and settings across the sweep
        key = (self._topology_key, self.order, self.method, self.rtol, self.maxiter, self.restart, self.nthreads)
        if self._sol is None or self._sol_key != key or not _HAS_SWEEP:
            self._sol = None
            self._sol = corvid.Solver(
                self._topology,
                order=self.order,
                method=self.method,
                rtol=self.rtol,
                maxiter=self.maxiter,
                restart=self.restart,
                **thread_kw,
            )
            self._sol_key = key
        sol = self._sol.factorize(A, k0=self._k0)
        x = sol.solve(b)
        iterations = [info["iterations"] for info in sol.info]
        residuals = [info["residual"] for info in sol.info]
        converged = all(info["converged"] for info in sol.info)
        state = ("kept factors" if sol.reused else "refactored" if sol.refactored else "factorized") if _HAS_SWEEP else "factorized"
        logger.info(
            f"{_pfx(self.pre, id)} CORVID {self.method}: iterations {iterations} ({state}, {sol.n_exact} extra exact rows)"
        )
        return x, SolveReport(
            solver=str(self),
            exit_code=0 if converged else 1,
            aux={
                "iterations": iterations,
                "residuals": residuals,
                "n_exact": sol.n_exact,
                "memory_bytes": sol.memory_bytes,
            },
        )
