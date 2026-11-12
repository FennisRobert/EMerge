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

"""The RS Lab direct solver."""
from __future__ import annotations
import os
import numpy as np
from scipy import sparse  # type: ignore
from loguru import logger
from .solver_base import Solver, SolverType, SolveReport, _pfx

try:
    import rslab  # type: ignore
    AVAILABLE = True
except ModuleNotFoundError:
    logger.trace("RSLAB Interface not found. (not harmful)")
    AVAILABLE = False


class SolverRSLAB(Solver):
    """Implements the RS Lab sparse direct solver.

    Requires the rslab package.
    """

    available = AVAILABLE
    requires = "the rslab package"

    req_sorter = False
    real_only = False
    stype = SolverType.SINGLE_MP
    name = "RS Lab"

    def __init__(self, pre: str):
        super().__init__(pre)
        logger.trace(self.pre + "Creating RSLAB solver")
        self.A: np.ndarray = None
        self.b: np.ndarray = None
        self._csym: bool = True
        self.factor = None
        self.sym = None
        self.initalized = False

    def initialize(self):
        if self.initalized:
            return
        logger.trace(self.pre + "Initializing RSLab Solver")
        self.initalized = True

    def reset(self) -> None:
        logger.trace(self.pre + "Resetting RSLab solver state")
        self.fact_symb = False

    def set_symmetry(self, complex_symmetric: bool) -> None:
        self._csym = complex_symmetric

    def duplicate(self) -> Solver:
        new_solver = self.__class__(self.pre)
        return new_solver

    def solve(self, A, b, precon, id: int = -1) -> tuple[np.ndarray, SolveReport]:
        logger.info(f"{_pfx(self.pre, id)} Calling RSLAB Solver.")
        logger.trace(f"{_pfx(self.pre, id)} Executing numeric factorization.")
        nthreads = int(os.getenv('RSLAB_NUM_THREADS', '4'))

        path = 'lu'
        extrdata = lambda x: x.data
        kwargs = dict(method='left_looking', nemin=16, panel_nb=64, threads=nthreads)
        if self._csym:
            path = 'ldlt'
            extrdata = lambda x: sparse.tril(A).data
        if self.sym is None:
            self.sym = rslab.analyze(A, path=path, ordering='metis', **kwargs)
        self.factor = self.sym.factor(extrdata(A), **kwargs)


        logger.trace(f"{_pfx(self.pre, id)} Solving linear system.")
        x = self.factor.solve_many(b)
        return x, SolveReport(solver=str(self), exit_code=0, aux=None)
