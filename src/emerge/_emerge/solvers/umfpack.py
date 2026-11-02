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

"""The UMFPACK direct solver (scikit-umfpack)."""
from __future__ import annotations
import numpy as np
from loguru import logger
from .solver_base import Solver, SolverType, SolveReport, _pfx

try:
    import scikits.umfpack as um  # type: ignore
    AVAILABLE = True
except ModuleNotFoundError:
    logger.trace("UMFPACK Interface not found. (not harmful)")
    AVAILABLE = False


class SolverUMFPACK(Solver):
    """Implements the UMFPACK Sparse SP solver.

    Requires scikit-umfpack: pip install emerge[umfpack] (conda install conda-forge::scikit-umfpack on Windows).
    """

    available = AVAILABLE
    requires = "scikit-umfpack: pip install emerge[umfpack] (conda install conda-forge::scikit-umfpack on Windows)"

    req_sorter = False
    real_only = False
    stype = SolverType.SINGLE_MP
    name = "UMFPACK"

    def __init__(self, pre: str):
        super().__init__(pre)
        logger.trace(self.pre + "Creating UMFPACK solver")
        self.A: np.ndarray = None
        self.b: np.ndarray = None

        self.umfpack: um.UmfpackContext | None = None

        # SETTINGS
        self._pivoting_threshold: float = 0.001

        self.fact_symb: bool = False
        self.initalized: bool = False

    def initialize(self):
        if self.initalized:
            return
        logger.trace(self.pre + "Initializing UMFPACK Solver")
        self.umfpack = um.UmfpackContext("zl")
        self.umfpack.control[um.UMFPACK_PRL] = 0  # ty: ignore
        self.umfpack.control[um.UMFPACK_IRSTEP] = 2  # ty: ignore
        self.umfpack.control[um.UMFPACK_STRATEGY] = um.UMFPACK_STRATEGY_SYMMETRIC  # ty: ignore
        self.umfpack.control[um.UMFPACK_ORDERING] = 3  # ty: ignore
        self.umfpack.control[um.UMFPACK_PIVOT_TOLERANCE] = 0.001  # ty: ignore
        self.umfpack.control[um.UMFPACK_SYM_PIVOT_TOLERANCE] = 0.001  # ty: ignore
        self.umfpack.control[um.UMFPACK_BLOCK_SIZE] = 64  # ty: ignore
        self.umfpack.control[um.UMFPACK_FIXQ] = -1  # ty: ignore
        self.initalized = True

    def reset(self) -> None:
        logger.trace(self.pre + "Resetting UMFPACK solver state")
        self.fact_symb = False

    def set_options(self, pivoting_threshold: float | None = None) -> SolverUMFPACK:
        self.initialize()
        if pivoting_threshold is not None:
            self.umfpack.control[um.UMFPACK_PIVOT_TOLERANCE] = pivoting_threshold  # ty: ignore
            self.umfpack.control[um.UMFPACK_SYM_PIVOT_TOLERANCE] = pivoting_threshold  # ty: ignore
            self._pivoting_threshold = pivoting_threshold
        return self

    def duplicate(self) -> Solver:
        new_solver = self.__class__(self.pre)
        new_solver.set_options(pivoting_threshold=self._pivoting_threshold)
        return new_solver

    def solve(self, A, b, precon, id: int = -1) -> tuple[np.ndarray, SolveReport]:
        logger.info(f"{_pfx(self.pre, id)} Calling UMFPACK Solver.")
        A.indptr = A.indptr.astype(np.int64)
        A.indices = A.indices.astype(np.int64)
        if self.fact_symb is False:
            logger.trace(f"{_pfx(self.pre, id)} Executing symbollic factorization.")
            self.umfpack.symbolic(A)
            self.fact_symb = True

        logger.trace(f"{_pfx(self.pre, id)} Executing numeric factorization.")
        self.umfpack.numeric(A)
        logger.trace(f"{_pfx(self.pre, id)} Solving linear system.")
        x = np.zeros_like(b)
        for i in range(b.shape[1]):
            logger.trace(f"{_pfx(self.pre, id)} Solving RHS {i}")
            x[:, i] = self.umfpack.solve(um.UMFPACK_A, A, b[:, i], autoTranspose=False)  # ty: ignore
        aux = {"Pivoting Threshold": str(self._pivoting_threshold)}
        return x, SolveReport(solver=str(self), exit_code=0, aux=aux)
