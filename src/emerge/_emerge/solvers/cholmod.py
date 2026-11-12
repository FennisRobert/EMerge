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

"""CHOLMOD (scikit-sparse) Cholesky solver for symmetric positive definite systems."""
from __future__ import annotations
from loguru import logger
from .solver_base import Solver, SolverType, SolveReport, _pfx

try:
    from sksparse.cholmod import cho_factor, metis  # type: ignore
    AVAILABLE = True
except ModuleNotFoundError:
    logger.trace("Scikit Sparse Interface not found. (not harmful)")
    cho_factor = metis = None
    AVAILABLE = False


class SolverCHOLMOD(Solver):
    """CHOLMOD direct solver with nested dissection ordering for SPD systems.

    Requires scikit-sparse: pip install scikit-sparse.
    """

    available = AVAILABLE
    requires = "scikit-sparse: pip install scikit-sparse"

    req_sorter = False
    stype = SolverType.SINGLE_MT
    name = "CHOLMOD_ND"

    def __init__(self, pre: str):
        super().__init__(pre)
        self.factor = None
        self.perm = None

    def reset(self):
        self.factor = None
        self.perm = None

    def duplicate(self):
        return self.__class__(self.pre)

    def solve(self, A, b, precon, reuse_factorization=False, id=-1):
        logger.info(f"{_pfx(self.pre, id)} Calling CHOLMOD (Nested Dissection) Solver.")

        if b.ndim == 1:
            b = b.reshape(-1, 1)

        if not reuse_factorization or self.factor is None:
            logger.trace(f"{_pfx(self.pre, id)} Computing Cholesky factorization.")
            self.factor = cho_factor(
                A, order="colamd", sym_kind="sym", supernodal_mode="supernodal"
            )

        x = self.factor.solve(b)

        return x, SolveReport(solver=str(self), exit_code=0)
