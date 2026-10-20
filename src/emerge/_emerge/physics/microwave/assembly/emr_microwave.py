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

# Last Cleanup: 2026-10-17
"""Python interface of the EMR (Rust) microwave assembly kernels."""
from __future__ import annotations
from typing import Callable
import numpy as np
from scipy.sparse import csc_matrix, csr_matrix
from emr import curlcurl, generalized_eigen_hb, robinbc, robin_abc_order2, periodicbc
from ....elements import Nedelec2
from ....elements.nedleg2 import NedelecLegrange2
from ....mth.csc_cast import CSCMapping
from ....mth.optimized import gaus_quad_tri

# Quadrature points of the Robin boundary integrals (must match strang(6) in emr/src/microwave/robinbc.rs)
ROBIN_DPTS = gaus_quad_tri(6)


############################################################
#                CURL-CURL VOLUME MATRICES                 #
############################################################


def tet_mass_stiffness_matrices(
    field: Nedelec2,
    er: np.ndarray,
    ur: np.ndarray,
    conductor_tets: np.ndarray,
    cscmap: CSCMapping | None = None,
) -> tuple[np.ndarray, np.ndarray, CSCMapping]:
    """Computes the curl-curl Nedelec-2 mass and stiffness matrices

    Args:
        field (Nedelec2): The Nedelec2 Field object
        er (np.ndarray): a 3x3xN array with permittivity tensors
        ur (np.ndarray): a 3x3xN array with permeability tensors

    Returns:
        tuple[csr_matrix, csr_matrix]: The stiffness and mass matrix.
    """
    tets = field.mesh.tets
    tris = field.mesh.tris
    edges = field.mesh.edges
    nodes = field.mesh.nodes

    tet_to_field = field.get_tet_to_field()
    tet_to_edge = field.mesh.tet_to_edge
    tet_to_tri = field.mesh.tet_to_tri

    tet_assy = np.arange(tets.shape[1])
    if conductor_tets.shape[0] > 0:
        tet_assy = np.delete(tet_assy, conductor_tets)
    #t0 = time.time()
    dataE, dataB, rows, cols = curlcurl._matrix_builder(
        nodes,
        tets,
        tris,
        edges,
        tet_to_field,
        tet_to_edge,
        tet_to_tri,
        ur,
        er,
        tet_assy,
        field.dofcodes3d,
    )
    #logger.debug(f' - Assembly speed: {(field.ntets -len(conductor_tets))/(t1-t0):.1f} tets/s')

    if cscmap is None:
        cscmap = CSCMapping.from_rowcol(rows, cols, field.n_field)
    return dataE, dataB, cscmap


def tet_mass_stiffness_matrices_subdom(
    field: Nedelec2,
    er: np.ndarray,
    ur: np.ndarray,
    domain_tets: np.ndarray,
    conductor_tets: np.ndarray,
    cscmap: CSCMapping | None = None,
) -> tuple[np.ndarray, np.ndarray, CSCMapping]:
    """Computes the curl-curl Nedelec-2 mass and stiffness matrices

    Args:
        field (Nedelec2): The Nedelec2 Field object
        er (np.ndarray): a 3x3xN array with permittivity tensors
        ur (np.ndarray): a 3x3xN array with permeability tensors

    Returns:
        tuple[csr_matrix, csr_matrix]: The stiffness and mass matrix.
    """
    tets = field.mesh.tets
    tris = field.mesh.tris
    edges = field.mesh.edges
    nodes = field.mesh.nodes

    tet_to_field = field.get_tet_to_field()
    tet_to_edge = field.mesh.tet_to_edge
    tet_to_tri = field.mesh.tet_to_tri

    keep = np.zeros((tets.shape[1],), dtype=np.bool_)
    keep[domain_tets] = 1
    keep[conductor_tets] = 0
    tet_assy = np.arange(tets.shape[1])[keep==True]

    dataE, dataB, rows, cols = curlcurl._matrix_builder(
        nodes,
        tets,
        tris,
        edges,
        tet_to_field,
        tet_to_edge,
        tet_to_tri,
        ur,
        er,
        tet_assy,
        field.dofcodes3d,
    )
    #logger.debug(f' - Assembly speed: {(field.ntets -len(conductor_tets))/(t1-t0):.1f} tets/s')

    if cscmap is None:
        cscmap = CSCMapping.from_rowcol(rows, cols, field.n_field)
    return dataE, dataB, cscmap

############################################################
#              PORT MODE EIGENVALUE MATRICES               #
############################################################


def generelized_eigenvalue_matrix(
    field: NedelecLegrange2,
    er: np.ndarray,
    ur: np.ndarray,
    inward_normal: np.ndarray,
    k0: float,
) -> tuple[csr_matrix, csr_matrix]:
    dofcodes = field.dofcodes
    tris = field.mesh.tris
    edges = field.mesh.edges

    nT = tris.shape[1]
    tri_to_field = field.tri_to_field

    nodes = field.local_nodes
    tri_to_edge = field.mesh.tri_to_edge
    dataE, dataB, rows, cols = generalized_eigen_hb._matrix_builder(
        nodes, tris, edges, tri_to_edge, tri_to_field, ur, er, k0, dofcodes,
    )

    nfield = field.n_field

    E = csr_matrix((dataE, (rows, cols)), shape=(nfield, nfield))
    B = csr_matrix((dataB, (rows, cols)), shape=(nfield, nfield))

    return E, B

############################################################
#                ROBIN BOUNDARY CONDITIONS                 #
############################################################


def assemble_robin_bc_bvec(
    field: Nedelec2,
    surf_triangle_indices: np.ndarray,
    Ufunc: Callable,
):  
    Bvec = np.zeros((field.n_field,), dtype=np.complex128)
    vertices = field.mesh.nodes

    xflat, yflat, zflat = robinbc.generate_points_3d(
        vertices, field.mesh.tris, ROBIN_DPTS, surf_triangle_indices
    )
    U_global = Ufunc(xflat, yflat, zflat)
    U_global_all = U_global.reshape((3, ROBIN_DPTS.shape[1], surf_triangle_indices.shape[0]))
    Bvec = robinbc.compute_force_entries(
        vertices,
        field.mesh.tris,
        Bvec,
        surf_triangle_indices,
        U_global_all,
        field.tri_to_field,
        field.dofcodes2d
    )
    return Bvec


def assemble_robin_bc(
    field: Nedelec2,
    surf_triangle_indices: np.ndarray,
    gamma: np.ndarray,
    other_side: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compact (data, rows, cols) triplet for a Robin BC, sized to just the
    given triangles instead of the whole mesh.
    """
    vertices = field.mesh.nodes
    N = field.n_tri_dofs ** 2
    Bmat = np.zeros(surf_triangle_indices.shape[0] * N, dtype=np.complex128)
    Bmat = robinbc.compute_bc_entries(
        vertices, field.mesh.tris, Bmat, surf_triangle_indices, gamma, field.dofcodes2d
    )
    rows, cols = field.tri_rowcol(surf_triangle_indices, other_side=other_side)
    return Bmat, rows, cols


def assemble_robin_bc_bvec_scat(
    field: Nedelec2,
    surf_triangle_indices: np.ndarray,
    Ufunc: Callable,
    UfuncCurl: Callable,
    normals: np.ndarray,
):

    Bvec = np.zeros((field.n_field,), dtype=np.complex128)

    vertices = field.mesh.nodes

    xflat, yflat, zflat = robinbc.generate_points_3d(
        vertices, field.mesh.tris, ROBIN_DPTS, surf_triangle_indices
    )

    U_global = Ufunc(xflat, yflat, zflat)
    U_global_curl = UfuncCurl(xflat, yflat, zflat)

    U_global_all = U_global.reshape((3, ROBIN_DPTS.shape[1], surf_triangle_indices.shape[0]))
    U_global_all_curl = U_global_curl.reshape(
        (3, ROBIN_DPTS.shape[1], surf_triangle_indices.shape[0])
    )

    Bvec = robinbc.compute_force_entries_scat(
        vertices,
        field.mesh.tris,
        Bvec,
        surf_triangle_indices,
        U_global_all,
        U_global_all_curl,
        field.tri_to_field,
        normals,
        field.dofcodes2d
    )
    return Bvec

############################################################
#                     SECOND ORDER ABC                     #
############################################################


def abc_order_2_matrix(
    field: Nedelec2, surf_triangle_indices: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Computes the second order absorbing boundary condition correction terms.

    The TE (surface curl) and TM (surface divergence) parts are returned separately
    at unit coefficient so that each can be scaled by its own coefficient:

        A = c_curl * Curl + c_div * Div

    with (c_curl, c_div) from RobinBC.get_abccorr(k0).

    Args:
        field (Nedelec2): The Basis function object
        surf_triangle_indices (np.ndarray): The surface triangle indices to add

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Compact (curl, div, rows, cols)
        data, sized to just the given triangles.
    """
    Curl, Div = robin_abc_order2._matrix_builder(
        field.mesh.nodes,
        field.mesh.tris,
        surf_triangle_indices,
        field.dofcodes2d
    )
    rows, cols = field.tri_rowcol(surf_triangle_indices)
    return Curl, Div, rows, cols

############################################################
#               PERIODIC BOUNDARY CONDITIONS               #
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
    ROWS, COLS, TERMS, diagonal = periodicbc._fill_periodic_matrix(tris_array, edges_array, tri_to_field, edge_to_field, dofcodes, Nfield)
    matrix = csc_matrix((TERMS, (ROWS, COLS)), [Nfield, Nfield], dtype=np.complex128)
    matrix.data.fill(phi)
    matrix.setdiag(diagonal)
    
    return matrix, ROWS

############################################################
#                   DOMAIN DECOMPOSITION                   #
############################################################


def generate_reduction_matrix(index_in: np.ndarray, index_out: np.ndarray) -> csc_matrix:
    """Generates a reduction matrix Rs to map some matrix or vector with
    global indices index_in into a sub-matrix with global indices index_out.

    Args:
        index_in (np.ndarray): _description_
        index_out (np.ndarray): _description_

    Returns:
        csc_matrix: _description_
    """

    # Positions of the common (sorted, unique) global indices: row in index_out, col in index_in
    _, cols, rows = np.intersect1d(index_in, index_out, assume_unique=True, return_indices=True)
    Nout = index_out.shape[0]
    Nin = index_in.shape[0]

    return csc_matrix((np.ones(rows.shape[0], dtype=np.complex128), (rows, cols)), shape=(Nout, Nin))
