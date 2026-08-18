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
import numpy as np
from numba import njit, f8, c16, i8, types, prange
from ....mth.optimized import cross
from ....elements import Nedelec2
from ....compiled.ccbf import (
    _eval_f_2d, _eval_curl_f_2d, parse_dofcode
)
from scipy.sparse import csc_matrix
from typing import Callable
from loguru import logger
import functools


@njit(cache=True)
def _generate_rows_cols(index_in, index_out):
  # index_in: all subdomain global indices (sorted, length Nin)
  # index_out: interface global indices (sorted, length Nout, subset of index_in)

  Nin = index_in.shape[0]
  Nout = index_out.shape[0]

  # Maximum possible non-zeros is Nout (since each interface DoF maps to one subdomain DoF)
  rows = np.empty((Nout,), dtype=np.int64)
  cols = np.empty((Nout,), dtype=np.int64)

  i = 0  # pointer for index_in
  j = 0  # pointer for index_out
  count = 0

  while i < Nin and j < Nout:
    val_in = index_in[i]
    val_out = index_out[j]

    if val_in == val_out:
      # Found a match!
      # row corresponds to interface index (0 to Nout-1)
      # col corresponds to subdomain index (0 to Nin-1)
      rows[count] = j
      cols[count] = i
      count += 1
      i += 1
      j += 1
    elif val_in < val_out:
      i += 1
    else:
      # val_in > val_out (interface index not found in this subdomain)
      j += 1

  return rows[:count], cols[:count]

def generate_reduction_matrix(index_in: np.ndarray, index_out: np.ndarray) -> csc_matrix:
    """Generates a reduction matrix Rs to map some matrix or vector with
    global indices index_in into a sub-matrix with global indices index_out.

    Args:
        index_in (np.ndarray): _description_
        index_out (np.ndarray): _description_

    Returns:
        csc_matrix: _description_
    """

    rows, cols = _generate_rows_cols(index_in, index_out)
    Nout = index_out.shape[0]
    Nin = index_in.shape[0]

    return csc_matrix((np.ones(rows.shape[0], dtype=np.complex128), (rows, cols)), shape=(Nout, Nin))
