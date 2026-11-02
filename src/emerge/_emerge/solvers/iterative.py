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

"""Iterative Krylov solvers from SciPy: CG, BiCGSTAB, GCROT(m,k) and GMRES."""
from __future__ import annotations
import numpy as np
from scipy.sparse import csc_matrix  # type: ignore
from scipy.sparse.linalg import bicgstab, cg, gmres, gcrotmk  # type: ignore
from loguru import logger
from .solver_base import Solver, SolverType, SolveReport, Preconditioner, _pfx


############################################################
#                     ITERATIVE SOLVERS                    #
############################################################


class SolverCG(Solver):
    """Implements the Conjugate Gradient method for SPD systems"""

    req_sorter = True
    name = "CG"
    stype = SolverType.SINGLE_ONLY

    def __init__(self, pre: str):
        super().__init__(pre)
        self.atol = 1e-10

        self.A: np.ndarray = None
        self.b: np.ndarray = None

    def callback(self, xk):
        self._iter += 1
        if self._iter % 50 == 0:
            res = np.linalg.norm(self.A @ xk - self.b.ravel())
            logger.info(f"{self.pre}: Iteration {self._iter}, |r| = {res:.2e}")

    def solve(self, A, b, precon, reuse_factorization=False, id=-1):
        logger.info(f"{_pfx(self.pre, id)} Calling CG.")
        self._iter = 0
        self.A = A
        self.b = b
        if precon.M is not None:
            x, info = cg(A, b, M=precon.M, atol=self.atol, callback=self.callback)
        else:
            x, info = cg(A, b, atol=self.atol, callback=self.callback)
        logger.info(f"{self.pre}: Converged in {self._iter} iterations.")
        x = x.reshape(b.shape)
        return x, SolveReport(solver=str(self), exit_code=info)


class SolverBicgstab(Solver):
    """Implements the Bi-Conjugate Gradient Stabilized method"""

    req_sorter = True
    name = "BISCGSTAB"
    stype = SolverType.SINGLE_ONLY

    def __init__(self, pre: str):
        super().__init__(pre)
        self.atol = 1e-5

        self.A: np.ndarray = None
        self.b: np.ndarray = None

    def callback(self, xk):
        """Iteration callback that logs the progress.

        Args:
            xk (np.ndarray): The current solution iterate
        """
        self._iter += 1
        if self._iter % 10 == 0:
            logger.info(f"{self.pre}: Iteration {self._iter}")

    def solve(
        self, A, b, precon, reuse_factorization: bool = False, id: int = -1
    ) -> tuple[np.ndarray, SolveReport]:
        logger.info(f"{_pfx(self.pre, id)} Calling BiCGStab.")
        self._iter = 0
        self.A = A
        self.b = b
        if precon.M is not None:
            x, info = bicgstab(A, b, M=precon.M, atol=self.atol, callback=self.callback)
        else:
            x, info = bicgstab(A, b, atol=self.atol, callback=self.callback)
        logger.info(f"{self.pre}: Converged in {self._iter} iterations.")
        x = x.reshape(b.shape)
        return x, SolveReport(solver=str(self), exit_code=info)


class SolverGCROTMK(Solver):
    stype = SolverType.SINGLE_ONLY
    name: str = "GCROTMK"
    """ Implements the GCRO-T(m,k) Iterative solver. """

    def __init__(self):
        super().__init__()
        self.atol = 1e-5
        self.A: np.ndarray = None
        self.b: np.ndarray = None

    def callback(self, xk):
        convergence = np.linalg.norm((self.A @ xk - self.b))
        logger.info(self.pre + f"Iteration {convergence:.4f}")

    def solve(
        self,
        A: csc_matrix,
        b: np.ndarray,
        precon: Preconditioner,
        reuse_factorization: bool = False,
        id: int = -1,
    ) -> tuple[np.ndarray, SolveReport]:
        logger.info(f"{_pfx(self.pre, id)} Calling GCRO-T(m,k) algorithm")
        self.A = A
        self.b = b
        if precon.M is not None:
            x, info = gcrotmk(A, b, M=precon.M, atol=self.atol, callback=self.callback)
        else:
            x, info = gcrotmk(A, b, atol=self.atol, callback=self.callback)
        return x, SolveReport(solver=str(self), exit_code=info)


class SolverGMRES(Solver):
    """Implements the GMRES solver."""

    stype = SolverType.SINGLE_ONLY
    real_only = False
    req_sorter = True
    name = "GMRES"

    def __init__(self):
        super().__init__()
        self.atol = 1e-5

        self.A: np.ndarray = None
        self.b: np.ndarray = None

    def callback(self, norm):
        # convergence = np.linalg.norm((self.A @ xk - self.b))
        """Iteration callback that logs the progress.

        Args:
            norm (float): The preconditioned residual norm
        """
        logger.info(self.pre + f"Iteration {norm:.4f}")

    def solve(
        self, A, b, precon, reuse_factorization: bool = False, id: int = -1
    ) -> tuple[np.ndarray, SolveReport]:
        logger.info(f"{_pfx(self.pre, id)} Calling GMRES Function.")
        self.A = A
        self.b = b
        if precon.M is not None:
            x, info = gmres(
                A,
                b,
                M=precon.M,
                atol=self.atol,
                callback=self.callback,
                callback_type="pr_norm",
            )
        else:
            x, info = gmres(
                A,
                b,
                atol=self.atol,
                callback=self.callback,
                restart=500,
                callback_type="pr_norm",
            )
        return x, SolveReport(solver=str(self), exit_code=info)
