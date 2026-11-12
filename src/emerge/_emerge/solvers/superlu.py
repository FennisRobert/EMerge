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

"""SciPy's SuperLU direct solver, with an optional METIS ordering from scikit-sparse."""
from __future__ import annotations
import hashlib
import numpy as np
from scipy.sparse import csc_matrix  # type: ignore
from scipy.sparse.linalg import splu  # type: ignore
from loguru import logger
from .solver_base import Solver, SolverType, SolveReport, _pfx
from .cholmod import AVAILABLE as _SKSP_AVAILABLE, metis


class SolverSuperLU(Solver):
    """Implements Scipy's direct SuperLU solver, with optional METIS preordering."""

    req_sorter: bool = False
    real_only: bool = False
    released_gil: bool = True
    stype = SolverType.SINGLE_MT
    name = "SUPERLU"

    def __init__(self, pre: str):
        super().__init__(pre)
        self.atol = 1e-5
        self.A = None
        self.b = None
        self.options: dict[str, str] = dict(
            SymmetricMode=True, Equil=False, IterRefine="SINGLE"
        )
        self._pivoting_threshold: float = 0.001
        self.lu = None
        self._perm = None
        self._a_hash: bytes = None

    def hash_A(self, A: csc_matrix) -> bytes:
        """Hashes the A matrix

        Args:
            A (csc_matrix): The system matrix

        Returns:
            bytes: The digest of the matrix
        """
        hasher = hashlib.sha256()
        hasher.update(A.indptr.tobytes())
        hasher.update(A.indices.tobytes())
        return hasher.digest()

    def reset(self):
        self.A = None
        self.b = None
        self.lu = None
        self._perm = None
        self._a_hash = None
        self.options = dict(SymmetricMode=True, Equil=False, IterRefine="SINGLE")
        self._pivoting_threshold = 0.001

    def duplicate(self) -> Solver:
        new_solver = self.__class__(self.pre)
        new_solver._pivoting_threshold = self._pivoting_threshold
        return new_solver

    def set_options(self, pivoting_threshold: float | None = None) -> SolverSuperLU:
        if pivoting_threshold is not None:
            self._pivoting_threshold = pivoting_threshold
        return self

    def _metis(self, A: csc_matrix, hash_value: bytes, idf: int) -> None:
        """Executes METIS column preordering algorithm from Scikit-Sparse

        Args:
            A (csc_matrix): The A-matrix
            hash_value (bytes): _description_
        """
        logger.debug(f"{_pfx(self.pre, idf)} Computing METIS permutation.")
        self._perm = metis(A)
        self._a_hash = hash_value

    def _metis_if_needed(self, A: csc_matrix, idf: int) -> None:
        """Executes a new METIS column preordering only if there is no
        ordering or the ordering is of a different matrix.

        Args:
            A (csc_matrix): _description_
            idf (int): _description_
        """
        hash_A = self.hash_A(A)
        if self._perm is None:
            logger.trace("No self._perm, requiring new METIS ordering.")
            self._metis(A, hash_A, idf)
            return
        if hash_A != self._a_hash:
            logger.trace("Different hash detected, requiring new METIS ordering.")
            self._metis(A, hash_A, idf)

    def solve(
        self, A: csc_matrix, b: np.ndarray, precon, id: int = -1
    ) -> tuple[np.ndarray, SolveReport]:
        logger.info(f"{_pfx(self.pre, id)} Calling SuperLU Solver.")

        if _SKSP_AVAILABLE:
            self._metis_if_needed(A, id)
            A_ordered = A[self._perm][:, self._perm]
            permc = "NATURAL"
        else:
            self._perm = None
            A_ordered = A
            permc = "MMD_AT_PLUS_A"

        logger.trace(f"{_pfx(self.pre, id)} Computing LU-Decomposition.")
        self.lu = splu(
            A_ordered,
            permc_spec=permc,
            relax=0,
            diag_pivot_thresh=self._pivoting_threshold,
            options=self.options,
        )

        if self._perm is not None:
            if b.ndim == 1:
                bp = b[self._perm]
                xp = self.lu.solve(bp)
                x = np.empty_like(xp)
                x[self._perm] = xp
            else:
                bp = b[self._perm, :]
                xp = self.lu.solve(bp)
                x = np.empty_like(xp)
                x[self._perm, :] = xp
        else:
            x = self.lu.solve(b)

        aux = {"pivoting threshold": str(self._pivoting_threshold)}
        return x, SolveReport(solver=str(self), exit_code=0, aux=aux)
