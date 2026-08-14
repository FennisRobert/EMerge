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
from __future__ import annotations
import numpy as np
from ....elements import Nedelec2
from ....mth.optimized import local_mapping, matinv
from ....mth.csc_cast import CSCMapping
from ....mth.csr_cast import CSRMapping
from ....compiled.ccbf import (
    _eval_f_3d, _eval_curl_f_3d, parse_dofcode
)
from numba import c16, types, f8, i8, njit, prange, void
from time import time
from loguru import logger
import time

# Toggle this to True when you want to use standard Python breakpoints
# DEBUG_MODE = False
# #
# import functools

# def njit(*args, **kwargs):
#     """
#     Drop-in replacement for numba.njit.
#     If DEBUG_MODE is True, it turns into a transparent 'do-nothing' wrapper.
#     If DEBUG_MODE is False, it forwards everything to the real Numba compiler.
#     """
#     if DEBUG_MODE:
#         # Case A: Used without parentheses -> @njit
#         if len(args) == 1 and callable(args[0]):
#             return args[0]

#         # Case B: Used with signatures/kwargs -> @njit(cache=True)
#         def decorator(func):
#             @functools.wraps(func)
#             def wrapper(*func_args, **func_kwargs):
#                 return func(*func_args, **func_kwargs)

#             return wrapper

#         return decorator
#     else:
#         # Import Numba lazily only when debugging is turned off
#         import numba

#         return numba.njit(*args, **kwargs)


############################################################
#                  INDEX MAPPING FUNCTIONS                 #
############################################################

# These mapping functions return edge and face coordinates in the appropriate order.


@njit(i8[:, :](i8[:, :], i8[:, :], i8[:, :], i8), cache=True, nogil=True)
def local_tet_to_triid(tet_to_tri, tets, tris, itet) -> np.ndarray:
    """Returns the triangle node indices in the right order given a tet-index"""
    tri_ids = tet_to_tri[:, itet]
    global_tri_map = tris[:, tri_ids]
    out = local_mapping(tets[:, itet], global_tri_map)
    return out


@njit(i8[:, :](i8[:, :], i8[:, :], i8[:, :], i8), cache=True, nogil=True)
def local_tet_to_edgeid(tet_to_edge, tets, edges, itet) -> np.ndarray:
    """Returns the edge node indices in the right order given a tet-index"""
    global_edge_map = edges[:, tet_to_edge[:, itet]]
    return local_mapping(tets[:, itet], global_edge_map)


@njit(i8[:, :](i8[:, :], i8[:, :], i8[:, :], i8), cache=True, nogil=True)
def local_tri_to_edgeid(tri_to_edge, tris, edges, itri: int) -> np.ndarray:
    """Returns the edge node indices in the right order given a triangle-index"""
    global_edge_map = edges[:, tri_to_edge[:, itri]]
    return local_mapping(tris[:, itri], global_edge_map)


@njit(c16[:, :](c16[:, :], c16[:, :]), cache=True, nogil=True)
def matmul(Mat, Vec):
    ## Matrix multiplication of a 3D vector
    Vout = np.empty((3, Vec.shape[1]), dtype=np.complex128)
    Vout[0, :] = Mat[0, 0] * Vec[0, :] + Mat[0, 1] * Vec[1, :] + Mat[0, 2] * Vec[2, :]
    Vout[1, :] = Mat[1, 0] * Vec[0, :] + Mat[1, 1] * Vec[1, :] + Mat[1, 2] * Vec[2, :]
    Vout[2, :] = Mat[2, 0] * Vec[0, :] + Mat[2, 1] * Vec[1, :] + Mat[2, 2] * Vec[2, :]
    return Vout


@njit(c16[:](c16[:, :], c16[:, :]), cache=True, nogil=True)
def dot(MatA, MatB):
    return MatA[0, :] * MatB[0, :] + MatA[1, :] * MatB[1, :] + MatA[2, :] * MatB[2, :]


@njit(void(c16[:, :], c16[:, :], c16[:, :]), cache=True, nogil=True, fastmath=True)
def matmul_inplace(Mat, Vec, Vout):
    m00, m01, m02 = Mat[0, 0], Mat[0, 1], Mat[0, 2]
    m10, m11, m12 = Mat[1, 0], Mat[1, 1], Mat[1, 2]
    m20, m21, m22 = Mat[2, 0], Mat[2, 1], Mat[2, 2]
    for n in range(Vec.shape[1]):
        v0, v1, v2 = Vec[0, n], Vec[1, n], Vec[2, n]
        Vout[0, n] = m00 * v0 + m01 * v1 + m02 * v2
        Vout[1, n] = m10 * v0 + m11 * v1 + m12 * v2
        Vout[2, n] = m20 * v0 + m21 * v1 + m22 * v2


############################################################
#  COMPUTATION OF THE BARYCENTRIC COORDINATE COEFFICIENTS #
############################################################


@njit(
    types.Tuple((f8[:], f8[:], f8[:], f8[:], f8))(f8[:], f8[:], f8[:]),
    cache=True,
    nogil=True,
)
def tet_coefficients(xs, ys, zs):
    x1, x2, x3, x4 = xs
    y1, y2, y3, y4 = ys
    z1, z2, z3, z4 = zs

    aas = np.empty((4,), dtype=np.float64)
    bbs = np.empty((4,), dtype=np.float64)
    ccs = np.empty((4,), dtype=np.float64)
    dds = np.empty((4,), dtype=np.float64)

    V = np.abs(
        -x1 * y2 * z3 
        + x1 * y2 * z4
        + x1 * y3 * z2
        - x1 * y3 * z4
        - x1 * y4 * z2
        + x1 * y4 * z3
        + x2 * y1 * z3
        - x2 * y1 * z4
        - x2 * y3 * z1
        + x2 * y3 * z4
        + x2 * y4 * z1
        - x2 * y4 * z3
        - x3 * y1 * z2
        + x3 * y1 * z4
        + x3 * y2 * z1
        - x3 * y2 * z4
        - x3 * y4 * z1
        + x3 * y4 * z2
        + x4 * y1 * z2
        - x4 * y1 * z3
        - x4 * y2 * z1
        + x4 * y2 * z3
        + x4 * y3 * z1
        - x4 * y3 * z2
    ) / 6

    aas[0] = (
        x2 * y3 * z4
        - x2 * y4 * z3
        - x3 * y2 * z4
        + x3 * y4 * z2
        + x4 * y2 * z3
        - x4 * y3 * z2
    )
    aas[1] = (
        -x1 * y3 * z4
        + x1 * y4 * z3
        + x3 * y1 * z4
        - x3 * y4 * z1
        - x4 * y1 * z3
        + x4 * y3 * z1
    )
    aas[2] = (
        x1 * y2 * z4
        - x1 * y4 * z2
        - x2 * y1 * z4
        + x2 * y4 * z1
        + x4 * y1 * z2
        - x4 * y2 * z1
    )
    aas[3] = (
        -x1 * y2 * z3
        + x1 * y3 * z2
        + x2 * y1 * z3
        - x2 * y3 * z1
        - x3 * y1 * z2
        + x3 * y2 * z1
    )
    bbs[0] = -y2 * z3 + y2 * z4 + y3 * z2 - y3 * z4 - y4 * z2 + y4 * z3
    bbs[1] = y1 * z3 - y1 * z4 - y3 * z1 + y3 * z4 + y4 * z1 - y4 * z3
    bbs[2] = -y1 * z2 + y1 * z4 + y2 * z1 - y2 * z4 - y4 * z1 + y4 * z2
    bbs[3] = y1 * z2 - y1 * z3 - y2 * z1 + y2 * z3 + y3 * z1 - y3 * z2
    ccs[0] = x2 * z3 - x2 * z4 - x3 * z2 + x3 * z4 + x4 * z2 - x4 * z3
    ccs[1] = -x1 * z3 + x1 * z4 + x3 * z1 - x3 * z4 - x4 * z1 + x4 * z3
    ccs[2] = x1 * z2 - x1 * z4 - x2 * z1 + x2 * z4 + x4 * z1 - x4 * z2
    ccs[3] = -x1 * z2 + x1 * z3 + x2 * z1 - x2 * z3 - x3 * z1 + x3 * z2
    dds[0] = -x2 * y3 + x2 * y4 + x3 * y2 - x3 * y4 - x4 * y2 + x4 * y3
    dds[1] = x1 * y3 - x1 * y4 - x3 * y1 + x3 * y4 + x4 * y1 - x4 * y3
    dds[2] = -x1 * y2 + x1 * y4 + x2 * y1 - x2 * y4 - x4 * y1 + x4 * y2
    dds[3] = x1 * y2 - x1 * y3 - x2 * y1 + x2 * y3 + x3 * y1 - x3 * y2

    return aas, bbs, ccs, dds, V


############################################################
#              MAIN CURL-CURL MATRIX ASSEMBLY             #
############################################################


def tet_mass_stiffness_matrices(
    field: Nedelec2,
    er: np.ndarray,
    ur: np.ndarray,
    conductor_tets: np.ndarray,
    cscmap: CSRMapping | None = None,
) -> tuple[np.ndarray, np.ndarray, CSRMapping]:
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
    dataE, dataB, rows, cols = _matrix_builder(
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
    t1 = time.time()
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
    cscmap: CSRMapping | None = None,
) -> tuple[np.ndarray, np.ndarray, CSRMapping]:
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

    dataE, dataB, rows, cols = _matrix_builder(
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
    t1 = time.time()
    #logger.debug(f' - Assembly speed: {(field.ntets -len(conductor_tets))/(t1-t0):.1f} tets/s')

    if cscmap is None:
        cscmap = CSCMapping.from_rowcol(rows, cols, field.n_field)
    return dataE, dataB, cscmap


############################################################
#           NUMBA ACCELLERATE SUB-MATRIX ASSEMBLY          #
############################################################

_DPTS = np.array([
    [-0.07893333333333329982, 0.04573333333333329948, 0.04573333333333329948, 0.04573333333333329948, 0.04573333333333329948, 0.14933333333333329018, 0.14933333333333329018, 0.14933333333333329018, 0.14933333333333329018, 0.14933333333333329018, 0.14933333333333329018],
    [0.25000000000000000000, 0.78571428571428569843, 0.07142857142857139685, 0.07142857142857139685, 0.07142857142857139685, 0.10059642383320080428, 0.39940357616679922348, 0.39940357616679922348, 0.39940357616679922348, 0.10059642383320080428, 0.10059642383320080428],
    [0.25000000000000000000, 0.07142857142857139685, 0.07142857142857139685, 0.07142857142857139685, 0.78571428571428569843, 0.39940357616679922348, 0.10059642383320080428, 0.39940357616679922348, 0.10059642383320080428, 0.39940357616679922348, 0.10059642383320080428],
    [0.25000000000000000000, 0.07142857142857139685, 0.07142857142857139685, 0.78571428571428569843, 0.07142857142857139685, 0.39940357616679922348, 0.39940357616679922348, 0.10059642383320080428, 0.10059642383320080428, 0.10059642383320080428, 0.39940357616679922348],
    [0.25000000000000000000, 0.07142857142857150787, 0.78571428571428580945, 0.07142857142857150787, 0.07142857142857150787, 0.10059642383320077652, 0.10059642383320077652, 0.10059642383320074877, 0.39940357616679922348, 0.39940357616679922348, 0.39940357616679922348],
], dtype=np.float64)

@njit(
    types.Tuple((c16[:, :], c16[:, :]))(
        f8[:, :], i8[:, :], i8[:, :], c16[:, :], c16[:, :], i8[:], i8[:], i8[:]
    ),
    nogil=True,
    cache=True,
    parallel=False,
    fastmath=True,
)
def ned2_tet_stiff_mass(tet_vertices, local_edge_map, local_tri_map, Ms, Mm, dofcodes, typearry, indexarry):
    DPTS = _DPTS
    
    ndof = dofcodes.shape[0]

    MatStiff = np.empty((ndof, ndof), dtype=np.complex128)
    MatMass = np.empty((ndof, ndof), dtype=np.complex128)

    txs = tet_vertices[0, :]
    tys = tet_vertices[1, :]
    tzs = tet_vertices[2, :]

    aas, bbs, ccs, dds, V = tet_coefficients(txs, tys, tzs)
    coeff = np.empty((4, 4), dtype=np.float64)
    coeff[0, :] = aas / (6 * V)
    coeff[1, :] = bbs / (6 * V)
    coeff[2, :] = ccs / (6 * V)
    coeff[3, :] = dds / (6 * V)

    WEIGHTS = DPTS[0, :]
    DPTS1 = DPTS[1, :]
    DPTS2 = DPTS[2, :]
    DPTS3 = DPTS[3, :]
    DPTS4 = DPTS[4, :]

    xs = txs[0] * DPTS1 + txs[1] * DPTS2 + txs[2] * DPTS3 + txs[3] * DPTS4
    ys = tys[0] * DPTS1 + tys[1] * DPTS2 + tys[2] * DPTS3 + tys[3] * DPTS4
    zs = tzs[0] * DPTS1 + tzs[1] * DPTS2 + tzs[2] * DPTS3 + tzs[3] * DPTS4

    coords = np.empty((3, xs.shape[0]), dtype=np.float64)
    coords[0, :] = xs
    coords[1, :] = ys
    coords[2, :] = zs

    N = coords.shape[1]

    all_fdof = np.empty((ndof, 3, N), dtype=np.complex128)
    all_fdof_curl = np.empty((ndof, 3, N), dtype=np.complex128)
    all_Ms_fdof_curl_w = np.empty((ndof, 3, N), dtype=np.complex128)
    all_Mm_fdof_w = np.empty((ndof, 3, N), dtype=np.complex128)

    for idof in range(ndof):
        i_type = typearry[idof]
        i_index = indexarry[idof]
        if i_type == 0:
            i1 = local_edge_map[0, i_index]
            j1 = local_edge_map[1, i_index]
            k1 = 0
        else:
            i1 = local_tri_map[0, i_index]
            j1 = local_tri_map[1, i_index]
            k1 = local_tri_map[2, i_index]

        _eval_f_3d(coeff, coords, i1, j1, k1, dofcodes[idof], all_fdof[idof, :, :])
        _eval_curl_f_3d(coeff, coords, i1, j1, k1, dofcodes[idof], all_fdof_curl[idof, :, :])

        matmul_inplace(Ms, all_fdof_curl[idof, :, :], all_Ms_fdof_curl_w[idof, :, :])
        matmul_inplace(Mm, all_fdof[idof, :, :], all_Mm_fdof_w[idof, :, :])

        for c in range(3):
            for n in range(N):
                all_Ms_fdof_curl_w[idof, c, n] *= WEIGHTS[n]
                all_Mm_fdof_w[idof, c, n] *= WEIGHTS[n]

    for idof1 in range(ndof):
        for idof2 in range(idof1, ndof):
            accS = 0.0 + 0.0j
            accM = 0.0 + 0.0j
            for n in range(N):
                accS += (
                    all_Ms_fdof_curl_w[idof1, 0, n] * all_fdof_curl[idof2, 0, n]
                    + all_Ms_fdof_curl_w[idof1, 1, n] * all_fdof_curl[idof2, 1, n]
                    + all_Ms_fdof_curl_w[idof1, 2, n] * all_fdof_curl[idof2, 2, n]
                )
                accM += (
                    all_Mm_fdof_w[idof1, 0, n] * all_fdof[idof2, 0, n]
                    + all_Mm_fdof_w[idof1, 1, n] * all_fdof[idof2, 1, n]
                    + all_Mm_fdof_w[idof1, 2, n] * all_fdof[idof2, 2, n]
                )
            MatStiff[idof1, idof2] = accS
            MatMass[idof1, idof2] = accM
            MatStiff[idof2, idof1] = accS
            MatMass[idof2, idof1] = accM

    MatStiff = MatStiff * V
    MatMass = MatMass * V
    return MatStiff, MatMass

############################################################
#             NUMBA ACCELLERATED MATRIX BUILDER            #
############################################################


@njit(
    types.Tuple((c16[:], c16[:], i8[:], i8[:]))(
        f8[:, :],
        i8[:, :],
        i8[:, :],
        i8[:, :],
        i8[:, :],
        i8[:, :],
        i8[:, :],
        c16[:, :, :],
        c16[:, :, :],
        i8[:],
        i8[:],
    ),
    cache=True,
    nogil=True,
    parallel=True,
)
def _matrix_builder(
    nodes,
    tets,
    tris,
    edges,
    tet_to_field,
    tet_to_edge,
    tet_to_tri,
    ur,
    er,
    tetids,
    dofcodes,
):
    ndof = dofcodes.shape[0]
    nT = tets.shape[1]
    ntets_assy = tetids.shape[0]
    nnz = ntets_assy * dofcodes.shape[0]**2

    rows = np.empty(nnz, dtype=np.int64)
    cols = np.empty_like(rows)
    dataE = np.empty_like(rows, dtype=np.complex128)
    dataB = np.empty_like(rows, dtype=np.complex128)
    typearry, indexarry = parse_dofcode(dofcodes)
    for iassy in prange(ntets_assy):  # ty: ignore
        itet = tetids[iassy]
        p = iassy * ndof**2
        urt = ur[:, :, itet]
        ert = er[:, :, itet]

        # Construct a local mapping to global triangle orientations
        local_tri_map = local_tet_to_triid(tet_to_tri, tets, tris, itet)
        local_edge_map = local_tet_to_edgeid(tet_to_edge, tets, edges, itet)

        # Construct the local edge map
        Esub, Bsub = ned2_tet_stiff_mass(
            nodes[:, tets[:, itet]],
            local_edge_map,
            local_tri_map,
            matinv(urt),
            ert,
            dofcodes,
            typearry, 
            indexarry
        )
        indices = tet_to_field[:, itet]
        for ii in range(ndof):
            rows[p + ndof * ii : p + ndof * (ii + 1)] = indices[ii]
            cols[p + ii : p + ndof**2 + ii : ndof] = indices[ii]

        dataE[p : p + ndof**2] = Esub.ravel()
        dataB[p : p + ndof**2] = Bsub.ravel()
    return dataE, dataB, rows, cols
