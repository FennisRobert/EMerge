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
from numba import njit, prange, i8, c16, f8, types
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
    
    def to_csc(self, data: np.ndarray) -> csc_matrix:
        if np.iscomplexobj(data):
            data_csc = np.zeros(self.nnz, dtype=np.complex128)
            scatter_to_csc_c16(data, self.perm, self.target_csc, self.col_offsets, self._gather_buffer_c16, data_csc)
        else:
            data_csc = np.zeros(self.nnz, dtype=np.float64)
            scatter_to_csc_f8(data, self.perm, self.target_csc, self.col_offsets, self._gather_buffer_f8, data_csc)
            
        return csc_matrix((data_csc, self.indices, self.indptr), shape=(self.N, self.N))

@njit(types.Tuple((i8[:], i8[:], i8[:], i8[:], i8[:]))(i8[:], i8[:], i8), nogil=True, cache=True, parallel=True)
def precompute_csc_pattern(rows, cols, N) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Massively parallel precomputation for CSC assembly."""
    nnz_coo = cols.shape[0]
    
    # --- Phase 1: Sequential Count and Group (Memory bound, very fast) ---
    col_counts = np.zeros(N, dtype=np.int64)
    for k in range(nnz_coo):
        col_counts[cols[k]] += 1
        
    col_offsets = np.zeros(N + 1, dtype=np.int64)
    for i in range(N):
        col_offsets[i + 1] = col_offsets[i] + col_counts[i]
        
    grouped_perm = np.empty(nnz_coo, dtype=np.int64)
    cursor = col_offsets[:-1].copy()
    for k in range(nnz_coo):
        c = cols[k]
        grouped_perm[cursor[c]] = k
        cursor[c] += 1

    # --- Phase 2: PARALLEL Sort and Count Unique Rows ---
    unique_counts = np.zeros(N, dtype=np.int64)
    fully_sorted_perm = np.empty(nnz_coo, dtype=np.int64)
    
    for i in prange(N):
        start = col_offsets[i]
        count = col_offsets[i + 1] - start
        
        if count == 0:
            continue
            
        # Local row extraction
        local_rows = np.empty(count, dtype=np.int64)
        for j in range(count):
            local_rows[j] = rows[grouped_perm[start + j]]
            
        # The heaviest operation is now running in parallel
        sort_idx = np.argsort(local_rows)
        
        uniques = 1
        first_idx = sort_idx[0]
        prev_row = local_rows[first_idx]
        fully_sorted_perm[start] = grouped_perm[start + first_idx]
        
        for j in range(1, count):
            curr_idx = sort_idx[j]
            curr_row = local_rows[curr_idx]
            
            if curr_row != prev_row:
                uniques += 1
                prev_row = curr_row
                
            fully_sorted_perm[start + j] = grouped_perm[start + curr_idx]
            
        unique_counts[i] = uniques

    # --- Phase 3: Sequential Memory Boundary Calculation ---
    indptr = np.zeros(N + 1, dtype=np.int64)
    for i in range(N):
        indptr[i + 1] = indptr[i] + unique_counts[i]
        
    nnz_csc = indptr[N]
    indices = np.empty(nnz_csc, dtype=np.int64)
    target_csc = np.empty(nnz_coo, dtype=np.int64)

    # --- Phase 4: PARALLEL Populate CSC Arrays ---
    for i in prange(N):
        start = col_offsets[i]
        count = col_offsets[i + 1] - start
        
        if count == 0:
            continue
            
        # Each thread starts writing at its perfectly safe memory offset
        csc_pos = indptr[i]
        
        # Process the first element
        coo_idx = fully_sorted_perm[start]
        prev_row = rows[coo_idx]
        indices[csc_pos] = prev_row
        target_csc[start] = csc_pos
        
        # Process the rest
        for j in range(1, count):
            coo_idx = fully_sorted_perm[start + j]
            curr_row = rows[coo_idx]
            
            if curr_row != prev_row:
                csc_pos += 1
                indices[csc_pos] = curr_row
                prev_row = curr_row
                
            target_csc[start + j] = csc_pos
            
    return indptr, indices, target_csc, col_offsets, fully_sorted_perm

@njit(nogil=True, cache=True, parallel=True, fastmath=True)
def scatter_to_csc_c16(data_coo, perm, target_csc, col_offsets, sorted_data, data_csc):
    n = data_coo.shape[0]
    
    # 1. PARALLEL GATHER: Pull random memory into perfectly contiguous order.
    for k in prange(n):
        sorted_data[k] = data_coo[perm[k]]
        
    # 2. PARALLEL COMPUTE: Both sorted_data and target_csc are accessed strictly sequentially.
    # No race conditions because threads are isolated by columns.
    N = col_offsets.shape[0] - 1
    for i in prange(N):
        start = col_offsets[i]
        end = col_offsets[i + 1]
        for k in range(start, end):
            data_csc[target_csc[k]] += sorted_data[k]


@njit(nogil=True, cache=True, parallel=True, fastmath=True)
def scatter_to_csc_f8(data_coo, perm, target_csc, col_offsets, sorted_data, data_csc):
    n = data_coo.shape[0]
    
    # 1. PARALLEL GATHER
    for k in prange(n):
        sorted_data[k] = data_coo[perm[k]]
        
    # 2. PARALLEL COMPUTE
    N = col_offsets.shape[0] - 1
    for i in prange(N):
        start = col_offsets[i]
        end = col_offsets[i + 1]
        for k in range(start, end):
            data_csc[target_csc[k]] += sorted_data[k]