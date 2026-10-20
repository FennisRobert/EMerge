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
from dataclasses import dataclass, field

@dataclass
class CSCMapping:
    indptr: np.ndarray
    indices: np.ndarray
    target_csc: np.ndarray
    col_offsets: np.ndarray
    perm: np.ndarray
    N: int
    nnz: int = 0
    nnz_coo: int = 0
    
    # Pre-allocated 1D buffers to hold perfectly sorted memory.
    # Because it is 1D (size of COO), it scales to any number of threads with zero overhead.
    _gather_buffer_c16: np.ndarray = field(init=False, repr=False)
    _gather_buffer_f8: np.ndarray = field(init=False, repr=False)
    
    def __post_init__(self):
        self.nnz = self.indices.shape[0]
        self.nnz_coo = self.perm.shape[0]
        self._gather_buffer_c16 = np.empty(self.nnz_coo, dtype=np.complex128)
        self._gather_buffer_f8 = np.empty(self.nnz_coo, dtype=np.float64)

    @staticmethod
    def from_rowcol(rows, cols, N) -> CSCMapping:
        return CSCMapping(*precompute_csc_pattern(rows, cols, N), N)
    
    def scatter(self, data: np.ndarray) -> np.ndarray:
        """Sums the COO values into a CSC data array aligned with self.indices/self.indptr."""
        if np.iscomplexobj(data):
            data_csc = np.zeros(self.nnz, dtype=np.complex128)
            scatter_to_csc_c16(data, self.perm, self.target_csc, self.col_offsets, self._gather_buffer_c16, data_csc)
        else:
            data_csc = np.zeros(self.nnz, dtype=np.float64)
            scatter_to_csc_f8(data, self.perm, self.target_csc, self.col_offsets, self._gather_buffer_f8, data_csc)
        return data_csc

    def to_csc(self, data: np.ndarray) -> csc_matrix:
        return csc_matrix((self.scatter(data), self.indices, self.indptr), shape=(self.N, self.N))
