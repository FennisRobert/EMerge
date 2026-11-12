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

"""Sorters (reorderings) and preconditioners of the iterative solvers."""
from __future__ import annotations
import numpy as np
from scipy import sparse  # type: ignore
from scipy.sparse.csgraph import reverse_cuthill_mckee  # type: ignore
from loguru import logger
from .solver_base import Sorter, Preconditioner
from .cholmod import cho_factor


############################################################
#                          SORTERS                         #
############################################################


class ReverseCuthillMckee(Sorter):
    """Implements the Reverse Cuthill-Mckee sorting."""

    def __init__(self):
        super().__init__()

    def sort(self, A, b, reuse_sorting: bool = False):
        if not reuse_sorting:
            logger.debug("Generating Reverse Cuthill-Mckee sorting.")
            self.perm = reverse_cuthill_mckee(A)
            self.inv_perm = np.argsort(self.perm)
        logger.debug("Applying Reverse Cuthill-Mckee sorting.")
        Asorted = A[self.perm, :][:, self.perm]
        bsorted = b[self.perm]
        return Asorted, bsorted

    def unsort(self, x: np.ndarray):
        logger.debug("Reversing Reverse Cuthill-Mckee sorting.")
        return x[self.inv_perm]


############################################################
#                      PRECONDITIONERS                     #
############################################################


class ILUPrecon(Preconditioner):
    """Implements the incomplete LU preconditioner on matrix A."""

    def __init__(self):
        super().__init__()
        self.M = None
        self.fill_factor = 20
        self.options: dict[str, str] = dict(SymmetricMode=True)

    def init(self, A, b):
        logger.info("Generating ILU Preconditioner")
        self.ilu = sparse.linalg.spilu(
            A,
            drop_tol=1e-3,
            drop_rule="basic,area,interp",
            fill_factor=self.fill_factor,  # ty: ignore
            permc_spec="MMD_AT_PLUS_A",
            diag_pivot_thresh=0.001,
            options=self.options,
        )
        self.M = sparse.linalg.LinearOperator(A.shape, self.ilu.solve)  # ty: ignore


class JacobiPrecon(Preconditioner):
    """Implements the diagonal (Jacobi) preconditioner on matrix A."""

    def __init__(self):
        super().__init__()
        self.M = None

    def init(self, A, b):
        logger.info("Generating Jacobi Preconditioner")
        diag_inv = 1.0 / A.diagonal()
        self.M = sparse.linalg.LinearOperator(A.shape, lambda x: diag_inv * x)


class ICCPrecon(Preconditioner):
    def __init__(self):
        super().__init__()
        self.M = None
        self.beta = 0.0

    def init(self, A, b):
        logger.info(f"Generating ICC Preconditioner (beta={self.beta})")
        factor = cho_factor(A, order="colamd", beta=self.beta)
        self.M = sparse.linalg.LinearOperator(A.shape, factor.solve)
