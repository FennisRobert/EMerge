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

# Last Cleanup: 2025-01-01

# This specific function is written by Claude Code and optimized manually for memory reduction.
# The new version is optimized by Gemini Pro. No copyright on this specific function.

from __future__ import annotations
import numpy as np
from emr.mth import precompute_csc_pattern, scatter_to_csc_c16, scatter_to_csc_f8
from scipy.sparse import csc_matrix
from dataclasses import dataclass

@dataclass
class CSCMapping:
    indptr: np.ndarray | None
    indices: np.ndarray | None
    target_csc: np.ndarray
    col_offsets: np.ndarray
    perm: np.ndarray
    N: int
    nnz: int = 0
    nnz_coo: int = 0

    def __post_init__(self):
        self.nnz = self.indices.shape[0]
        self.nnz_coo = self.perm.shape[0]

    @staticmethod
    def from_rowcol(rows, cols, N) -> CSCMapping:
        return CSCMapping(*precompute_csc_pattern(rows, cols, N), N)

    def release_pattern(self) -> None:
        """[Written by Claude] Frees indices and indptr (nnz and N+1 long) for an owner that keeps its own copy of
        the pattern. scatter() still works, to_csc() no longer does."""
        self.indices = None
        self.indptr = None

    def scatter(self, data: np.ndarray) -> np.ndarray:
        """Sums the COO values into a CSC data array aligned with self.indices/self.indptr."""
        # [Written by Claude] The gather buffer (COO length) only lives during the call: kept on the mapping it held
        # 24 bytes per COO entry for the whole run.
        if np.iscomplexobj(data):
            data_csc = np.zeros(self.nnz, dtype=np.complex128)
            buffer = np.empty(self.nnz_coo, dtype=np.complex128)
            scatter_to_csc_c16(data, self.perm, self.target_csc, self.col_offsets, buffer, data_csc)
        else:
            data_csc = np.zeros(self.nnz, dtype=np.float64)
            buffer = np.empty(self.nnz_coo, dtype=np.float64)
            scatter_to_csc_f8(data, self.perm, self.target_csc, self.col_offsets, buffer, data_csc)
        return data_csc

    def to_csc(self, data: np.ndarray) -> csc_matrix:
        if self.indices is None:
            raise RuntimeError("This CSCMapping released its pattern (release_pattern) and can only scatter.")
        return csc_matrix((self.scatter(data), self.indices, self.indptr), shape=(self.N, self.N))
