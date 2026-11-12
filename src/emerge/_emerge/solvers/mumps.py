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

"""The MUMPS direct solver: the interface to mumps_emerge and the Solver."""
from __future__ import annotations
import numpy as np
from scipy import sparse  # type: ignore
from loguru import logger
from .solver_base import Solver, SolverType, SolveReport, _pfx

try:
    from mumps_emerge import ZMumpsContext  # type: ignore
    AVAILABLE = True
except ModuleNotFoundError:
    logger.trace("MUMPSInterface not found. (not harmful)")
    AVAILABLE = False


# Reuse factorizations by running `job=3` with new right hand sides
# or analyses by supplying new values and running `job=2` to repeat
# the factorization process.

class MUMPSInterface:
    def __init__(self):
        self.ctx = ZMumpsContext(1,0)
        self.ctx.set_silent()
        self._analysed = False
        self._factorized = False
        self.ctx.id.icntl[6]  = 5      # METIS
        self.ctx.id.icntl[27] = 1      # ICNTL(28)
        self.ctx.id.icntl[28] = 0      # ICNTL(29) automatic (safe)
        self.ctx.id.icntl[47] = 0
        
    def analyse_matrix(self, A: sparse.csr_matrix):
        # Convert to COO
        A = A.tocoo()
        n = A.shape[0]

        if self.ctx.myid == 0:
            # 1-based indices
            irn = A.row.astype(np.int32) + 1
            jcn = A.col.astype(np.int32) + 1
            a = A.data.astype(np.complex128)

            self.ctx.set_shape(n)
            self.ctx.set_centralized_assembled(irn, jcn, a)

        # Analysis
        self.ctx.run(job=1)
        self._analysed = True
        self._factorized = False

    def factorize(self, A: sparse.csr_matrix):
        if not self._analysed:
            raise RuntimeError("Matrix must be analysed before factorization")
        A = A.tocoo()

        n = A.shape[0]

        if self.ctx.myid == 0:
            # 1-based indices
            irn = A.row.astype(np.int32) + 1
            jcn = A.col.astype(np.int32) + 1
            a = A.data.astype(np.complex128)

            self.ctx.set_shape(n)
            self.ctx.set_centralized_assembled(irn, jcn, a)
            
        self.ctx.run(job=2)
        self._factorized = True

    def solve(self, b):
        if not self._factorized:
            raise RuntimeError("Matrix must be factorized before solve")
        if self.ctx.myid == 0:
            x = np.asarray(b, dtype=np.complex128).copy()
            self.ctx.set_rhs(x)
        else:
            x = None

        self.ctx.run(job=3)

        return x, 0

    def destroy(self):
        self.ctx.destroy()


class SolverMUMPS(Solver):
    """Implements the MUMPS Sparse SP solver.

    Requires the mumps_emerge package.
    """

    available = AVAILABLE
    requires = "the mumps_emerge package"

    req_sorter = False
    real_only = False
    stype = SolverType.SINGLE_MP

    def __init__(self, pre: str):
        super().__init__(pre)
        logger.trace(self.pre + "Creating MUMPS solver")
        self.A: np.ndarray = None
        self.b: np.ndarray = None

        self.mumps: MUMPSInterface | None = None

        # SETTINGS
        self._pivoting_threshold: float = 0.001

        self.fact_symb: bool = False
        self.initalized: bool = False

    def initialize(self):
        if self.initalized:
            return
        logger.trace(self.pre + "Initializing MUMPS Solver")
        self.mumps = MUMPSInterface()
        self.initalized = True

    def reset(self) -> None:
        logger.trace(self.pre + "Resetting MUMPS solver state")
        self.fact_symb = False

    def duplicate(self) -> Solver:
        new_solver = self.__class__(self.pre)
        return new_solver

    def solve(
        self, A, b, precon, reuse_factorization: bool = False, id: int = -1
    ) -> tuple[np.ndarray, SolveReport]:
        logger.info(f"{_pfx(self.pre, id)} Calling MUMPS Solver.")
        if self.fact_symb is False:
            logger.trace(f"{_pfx(self.pre, id)} Executing symbollic factorization.")
            self.mumps.analyse_matrix(A)
            self.fact_symb = True
        if not reuse_factorization:
            logger.trace(f"{_pfx(self.pre, id)} Executing numeric factorization.")
            self.mumps.factorize(A)
            self.A = A
        logger.trace(f"{_pfx(self.pre, id)} Solving linear system.")
        x = np.zeros_like(b)
        for i in range(x.shape[1]):
            logger.trace(f"{_pfx(self.pre, id)} Solve RHS {i}.")
            x[:, i], _ = self.mumps.solve(b[:, i])  # ty: ignore
        return x, SolveReport(solver=str(self), exit_code=0)
