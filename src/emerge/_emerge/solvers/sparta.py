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

"""The SPARTA sparse direct solver (Metal GPU and CPU)."""
from __future__ import annotations
from typing import Literal
import numpy as np
from scipy.sparse import csc_matrix  # type: ignore
from loguru import logger
from packaging.version import Version
from .solver_base import Solver, SolverType, SolveReport, _pfx

try:
    import sparta  # type: ignore
    AVAILABLE = True
    if Version(sparta.__version__) < Version("0.2.0"):
        logger.error('Sparta version is outdated, install the latest (at least 0.2.0) from github: pip install git+https://github.com/FennisRobert/SPARTA')
        AVAILABLE = False
except ModuleNotFoundError:
    logger.trace("SPARTA Interface not found. (not harmful)")
    AVAILABLE = False

# [Written by Claude] If SPARTA can use a Metal GPU (Apple silicon). Without one only its CPU path works.
GPU = False
if AVAILABLE:
    try:
        GPU = bool(sparta.gpu_available())
    except Exception:
        GPU = False


class SolverSparta(Solver):
    """Implements the SPARTA Sparse Metal based GPU solver.

    Requires SPARTA 0.2.0 or newer: pip install git+https://github.com/FennisRobert/SPARTA.
    """

    available = AVAILABLE
    gpu = GPU
    requires = "SPARTA 0.2.0 or newer: pip install git+https://github.com/FennisRobert/SPARTA"

    req_sorter = False
    real_only = False
    stype = SolverType.SINGLE_MP
    name = "Sparta"

    def __init__(self, pre: str):
        super().__init__(pre)
        logger.trace(self.pre + "Creating Sparta solver")
        self.A: np.ndarray = None
        self.b: np.ndarray = None
        self._csym: bool = True
        self.factor = None
        self.sym = None
        self.initalized = False
        self.mode: Literal['CPU','GPU','HYBRID'] = 'GPU'
        self.nthreads: int | None = None
        self.settings: dict = dict(
            ordering=sparta.Ordering.SCALPEL,
            refinement_steps=0,
        )

    def initialize(self):
        if self.initalized:
            return
        logger.trace(self.pre + "Initializing SPARTA Solver")
        self.initalized = True

    def reset(self) -> None:
        logger.trace(self.pre + "Resetting SPARTA solver state")
        if self.factor is not None:
            self.factor.destroy()
            self.factor = None
        if self.sym is not None:
            self.sym.destroy()
            self.sym = None
        self.fact_symb = False

    def set_options(self, mode: Literal['GPU','CPU','HYBRID']) -> SolverSparta:
        """Set the factorization mode for the factorization. Options are:
         - 'GPU': Use the GPU only
         - 'CPU': Use the CPU only
         - 'HYBRID': Use a Hybrid GPU + CPU solver (if available)
        """
        self.mode = mode.strip().upper()
        return self
    
    # [Written by Claude]
    def set_nthreads(self, nthreads: int | None) -> SolverSparta:
        """Install the number of CPU threads SPARTA uses for its analysis, CPU factorization, solves and the
        hybrid's CPU share. None leaves it to SPARTA (the process-wide pool sized by RAYON_NUM_THREADS).

        SPARTA sizes its thread pool in analyze(), so a different count discards the current analysis.

        Args:
            nthreads (int | None): The number of threads.
        """
        if Version(sparta.__version__) < Version("0.3.0"):
            return super().set_nthreads(nthreads)
        if nthreads != self.nthreads and self.sym is not None:
            logger.trace(f"{self.pre}SPARTA thread count changed to {nthreads}, analyzing again.")
            self.reset()
        self.nthreads = nthreads
        return self
    
    def set_symmetry(self, complex_symmetric: bool) -> None:
        self._csym = complex_symmetric

    def duplicate(self) -> Solver:
        # [Written by Claude] Worker copies keep the mode and thread count
        return self.__class__(self.pre).set_options(self.mode).set_nthreads(self.nthreads)

    def solve(self, A: csc_matrix, b: np.ndarray, precon, id: int = -1) -> tuple[np.ndarray, SolveReport]:

        modemap = {
            'GPU': (False, False),
            'CPU': (True, False),
            'HYBRID': (False, True),
        }
        cpu, hybrid = modemap.get(self.mode, (False, False))
        
        logger.info(f"{_pfx(self.pre, id)} Calling SPARTA Solver ({self.mode}, {sparta.__version__}).")
        logger.trace(f"{_pfx(self.pre, id)} Executing numeric factorization.")
        
        mtype = sparta.MatrixType.COMPLEX_STRUCTURALLY_SYMMETRIC

        if self._csym:
            mtype = sparta.MatrixType.COMPLEX_SYMMETRIC
            
        if self.sym is None:
            # threads is an analysis setting only; factorize() does not accept it
            threads = {} if self.nthreads is None else dict(threads=self.nthreads)
            self.sym = sparta.analyze(A, force_cpu=cpu, hybrid=hybrid, mtype=mtype, **threads, **self.settings)
        
        self.factor = self.sym.factorize(A, force_cpu=cpu, hybrid=hybrid, mtype=mtype, **self.settings)
        if not self.factor.quality.usable:
            logger.warning('Matrix is so singular that it is broken on single precision. Use a double-precision CPU solver instead.')

        logger.trace(f"{_pfx(self.pre, id)} Solving linear system.")
        x = self.factor.solve_many(b)
        
        return x, SolveReport(solver=str(self), exit_code=0, aux=None)
