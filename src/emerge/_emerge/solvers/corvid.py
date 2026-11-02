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
except ModuleNotFoundError:
    logger.trace("CORVID Interface not found. (not harmful)")
    AVAILABLE = False


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
        self.order: int = 1
        self.rtol: float = 1e-6
        self.maxiter: int = 500
        self.restart: int = 100
        self._topology = None
        self._topology_key: tuple | None = None
        self._k0: float | None = None

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

    def duplicate(self) -> Solver:
        new_solver = self.__class__(self.pre)
        new_solver.set_options(self.order, self.rtol, self.maxiter, self.restart)
        return new_solver

    def solve(self, A: csc_matrix, b: np.ndarray, precon, id: int = -1) -> tuple[np.ndarray, SolveReport]:
        if self._topology is None:
            raise SimulationError("CORVID needs the mesh topology. Call set_precon_data() before solve().")
        logger.info(f"{_pfx(self.pre, id)} Calling CORVID solver ({self.method}).")
        sol = corvid.Solver(
            self._topology,
            order=self.order,
            method=self.method,
            rtol=self.rtol,
            maxiter=self.maxiter,
            restart=self.restart,
        ).factorize(A, k0=self._k0)
        x = sol.solve(b)
        iterations = [info["iterations"] for info in sol.info]
        residuals = [info["residual"] for info in sol.info]
        converged = all(info["converged"] for info in sol.info)
        logger.info(
            f"{_pfx(self.pre, id)} CORVID {self.method}: iterations {iterations} ({sol.n_exact} extra exact rows)"
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
