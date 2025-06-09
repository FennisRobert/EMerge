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

import numpy as np
from ...elements import Nedelec2
#from ...elements.nedelec2 import ned2_tet_stiff_mass
from scipy.sparse import csr_matrix, coo_matrix
from numba_progress import ProgressBar, ProgressBarType
from ...mth.optimized import local_mapping, matinv
from ...mth.tet import ned2_tet_stiff_mass
from numba import c16, types, f8, i8, njit, prange

@njit(i8[:,:](i8[:,:], i8[:,:], i8[:,:], i8, i8), cache=True, nogil=True)
def local_tet_to_triid(tet_to_field, tets, tris, itet, nedges) -> np.ndarray:
    tri_ids = tet_to_field[6:10, itet] - nedges
    global_tri_map = tris[:, tri_ids]
    return local_mapping(tets[:, itet], global_tri_map)

@njit(i8[:,:](i8[:,:], i8[:,:], i8[:,:], i8), cache=True, nogil=True)
def local_tet_to_edgeid(tets, edges, tet_to_field, itet) -> np.ndarray:
    global_edge_map = edges[:, tet_to_field[:6,itet]]
    return local_mapping(tets[:, itet], global_edge_map)

@njit(i8[:,:](i8[:,:], i8[:,:], i8[:,:], i8), cache=True, nogil=True)
def local_tri_to_edgeid(tris, edges, tri_to_field, itri: int) -> np.ndarray:
    global_edge_map = edges[:, tri_to_field[:3,itri]]
    return local_mapping(tris[:, itri], global_edge_map)

def tet_mass_stiffness_matrices(field: Nedelec2,
                           er: np.ndarray, 
                           ur: np.ndarray) -> tuple[csr_matrix, csr_matrix]:
    
    tets = field.mesh.tets
    tris = field.mesh.tris
    edges = field.mesh.edges
    nodes = field.mesh.nodes

    nT = tets.shape[1]
    tet_to_field = field.tet_to_field
    tet_to_edge = field.mesh.tet_to_edge
    nE = edges.shape[1]
    nTri = tris.shape[1]

    with ProgressBar(total=nT, ncols=100, dynamic_ncols=False) as pgb:
        dataE, dataB, rows, cols = _matrix_builder(nodes, tets, tris, edges, field.mesh.edge_lengths, tet_to_field, tet_to_edge, ur, er, pgb)
        
    E = coo_matrix((dataE, (rows, cols)), shape=(nE*2 + nTri*2, nE*2 + nTri*2)).tocsr()
    B = coo_matrix((dataB, (rows, cols)), shape=(nE*2 + nTri*2, nE*2 + nTri*2)).tocsr()

    return E, B

@njit(types.Tuple((c16[:], c16[:], i8[:], i8[:]))(f8[:,:], 
                                                      i8[:,:], 
                                                      i8[:,:], 
                                                      i8[:,:], 
                                                      f8[:], 
                                                      i8[:,:], 
                                                      i8[:,:], 
                                                      c16[:,:,:], 
                                                      c16[:,:,:], 
                                                      ProgressBarType), cache=True, nogil=True, parallel=True)
def _matrix_builder(nodes, tets, tris, edges, all_edge_lengths, tet_to_field, tet_to_edge, ur, er, pgb: ProgressBar):
    nT = tets.shape[1]
    nedges = edges.shape[1]

    nnz = nT*400

    rows = np.empty(nnz, dtype=np.int64)
    cols = np.empty_like(rows)
    dataE = np.empty_like(rows, dtype=np.complex128)
    dataB = np.empty_like(rows, dtype=np.complex128)

    
    for itet in prange(nT):
        p = itet*400
        if np.mod(itet,10)==0:
            pgb.update(10)
        urt = ur[:,:,itet]
        ert = er[:,:,itet]

        # Construct a local mapping to global triangle orientations
        
        local_tri_map = local_tet_to_triid(tet_to_field, tets, tris, itet, nedges)
        local_edge_map = local_tet_to_edgeid(tets, edges, tet_to_field, itet)
        edge_lengths = all_edge_lengths[tet_to_edge[:,itet]]

        # Construct the local edge map

        Esub, Bsub = ned2_tet_stiff_mass(nodes[:,tets[:,itet]], 
                                                edge_lengths, 
                                                local_edge_map, 
                                                local_tri_map, 
                                                matinv(urt), ert)
        
        indices = tet_to_field[:, itet]
        for ii in range(20):
            rows[p+20*ii:p+20*(ii+1)] = indices[ii]
            cols[p+ii:p+400:20] = indices[ii]

        dataE[p:p+400] = Esub.ravel()
        dataB[p:p+400] = Bsub.ravel()
    return dataE, dataB, rows, cols