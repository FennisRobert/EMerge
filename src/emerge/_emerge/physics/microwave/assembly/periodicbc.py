# EMerge is an open source Python based FEM EM simulation module.
# Copyright (C) 2025  Robert Fennis.

# This program is free software; you can redistribute it and/or
# modify it under the TERMS of the GNU General Public License
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
from numba import njit, types, i8, c16
from scipy.sparse import csc_matrix
from ....compiled.ccbf import (
    _eval_f_2d, _eval_curl_f_2d, parse_dofcode
)
############################################################
#                      NUMBA COMPILED                     #
############################################################
@njit(
    types.Tuple((i8[:], i8[:], c16[:], c16[:]))(
        i8[:, :], i8[:, :], i8[:, :], i8[:, :], i8[:], i8
    ),
    cache=True,
    nogil=True,
)
def _fill_periodic_matrix(
    tris: np.ndarray,
    edges: np.ndarray,
    tri_to_field: np.ndarray,
    edge_to_field: np.ndarray,
    dofcodes: np.ndarray,
    Nfield: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    typearray, _ = parse_dofcode(dofcodes)
    
    nedof = edge_to_field.shape[0]
    NT = tris.shape[1]
    NE = edges.shape[1]

    # Count active face DoFs on a 2D triangle
    nfdof = 0
    for t in typearray:
        if t == 1:
            nfdof += 1

    N_entries = NT * nfdof + NE * nedof

    DIAGONAL = np.ones((Nfield,), dtype=np.complex128)
    ROWS = np.zeros((N_entries,), dtype=np.int64)
    COLS = np.zeros((N_entries,), dtype=np.int64)
    TERMS = np.zeros((N_entries,), dtype=np.complex128)

    i = 0

    # 1. Map Face DoFs for linked triangles
    if nfdof > 0:
        for it in range(NT):
            t1 = tris[0, it]
            t2 = tris[1, it]
            
            # Row index k in tri_to_field corresponds to dofcodes2d[k]
            for k in range(typearray.shape[0]):
                if typearray[k] == 1:  # Only link Face DoFs
                    f1 = tri_to_field[k, t1]
                    f2 = tri_to_field[k, t2]

                    DIAGONAL[f2] = 0.0
                    ROWS[i] = f2
                    COLS[i] = f1
                    TERMS[i] = 1.0
                    i += 1

    # 2. Map Edge DoFs for linked edges
    if nedof > 0:
        for ie in range(NE):
            e1 = edges[0, ie]
            e2 = edges[1, ie]
            for k in range(nedof):
                f1 = edge_to_field[k, e1]
                f2 = edge_to_field[k, e2]

                DIAGONAL[f2] = 0.0
                ROWS[i] = f2
                COLS[i] = f1
                TERMS[i] = 1.0
                i += 1

    return ROWS, COLS, TERMS, DIAGONAL


############################################################
#                     PYTHON INTERFACE                    #
############################################################

def gen_periodic_matrix(tris: np.ndarray, 
                        edges: np.ndarray, 
                        tri_to_field: np.ndarray, 
                        edge_to_field: np.ndarray, 
                        linked_tris: dict[int, int], 
                        linked_edges: dict[int, int], 
                        dofcodes: np.ndarray,
                        Nfield: int, 
                        phi: complex) -> tuple[csc_matrix, np.ndarray]:
    """This function constructs the periodic boundary matrix

    Args:
        tris (np.ndarray): _description_
        edges (np.ndarray): _description_
        tri_to_field (np.ndarray): _description_
        edge_to_field (np.ndarray): _description_
        linked_tris (dict[int, int]): _description_
        linked_edges (dict[int, int]): _description_
        Nfield (int): _description_
        phi (complex): _description_

    Returns:
        tuple[csr_matrix, np.ndarray]: _description_
    """

    tris_array = np.array([(tri, linked_tris[tri]) for tri in tris]).T
    edges_array = np.array([(edge, linked_edges[edge]) for edge in edges]).T
    ROWS, COLS, TERMS, diagonal = _fill_periodic_matrix(tris_array, edges_array, tri_to_field, edge_to_field, dofcodes, Nfield)
    matrix = csc_matrix((TERMS, (ROWS, COLS)), [Nfield, Nfield], dtype=np.complex128)
    matrix.data.fill(phi)
    matrix.setdiag(diagonal)
    
    return matrix, ROWS