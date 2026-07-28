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
from ....elements import Nedelec2
from ....mth.optimized import local_mapping
from numba import c16, types, f8, i8, njit, prange
from ....compiled.ccbf import (
    _eval_curl_f_2d, _eval_div_f_2d, parse_dofcode
)

#
from .robinbc import construct_local_vertices

############################################################
#                      FIELD MAPPING                      #
############################################################


@njit(i8[:, :](i8, i8[:, :], i8[:, :], i8[:, :]), cache=True, nogil=True)
def local_tri_to_edgeid(itri: int, tris, edges, tri_to_edge) -> np.ndarray:
    global_edge_map = edges[:, tri_to_edge[:, itri]]
    return local_mapping(tris[:, itri], global_edge_map)


@njit(cache=True, fastmath=True, nogil=True)
def optim_matmul(B: np.ndarray, data: np.ndarray):
    dnew = np.zeros_like(data)
    dnew[0, :] = B[0, 0] * data[0, :] + B[0, 1] * data[1, :] + B[0, 2] * data[2, :]
    dnew[1, :] = B[1, 0] * data[0, :] + B[1, 1] * data[1, :] + B[1, 2] * data[2, :]
    dnew[2, :] = B[2, 0] * data[0, :] + B[2, 1] * data[1, :] + B[2, 2] * data[2, :]
    return dnew


@njit(f8[:](f8[:], f8[:]), cache=True, fastmath=True, nogil=True)
def cross(a: np.ndarray, b: np.ndarray):
    crossv = np.empty((3,), dtype=np.float64)
    crossv[0] = a[1] * b[2] - a[2] * b[1]
    crossv[1] = a[2] * b[0] - a[0] * b[2]
    crossv[2] = a[0] * b[1] - a[1] * b[0]
    return crossv


@njit(cache=True, nogil=True)
def normalize(a: np.ndarray):
    return a / ((a[0] ** 2 + a[1] ** 2 + a[2] ** 2) ** 0.5)


@njit(f8[:, :](f8[:], f8[:]), cache=True, nogil=True, fastmath=True)
def compute_distances(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    N = xs.shape[0]
    Ds = np.empty((N, N), dtype=np.float64)
    for i in range(N):
        for j in range(i, N):
            Ds[i, j] = np.sqrt((xs[i] - xs[j]) ** 2 + (ys[i] - ys[j]) ** 2)
            Ds[j, i] = Ds[i, j]
    return Ds


############################################################
#              GAUSS QUADRATURE IMPLEMENTATION             #
############################################################


@njit(
    c16(c16[:], c16[:], types.Array(types.float64, 1, "A", readonly=True)),
    cache=True,
    nogil=True,
)
def _gqi(v1, v2, W):
    return np.sum(v1 * v2 * W)


############################################################
#                BASIS FUNCTION DERIVATIVES               #
############################################################


############################################################
#     TRIANGLE BARYCENTRIC COORDINATE LIN. COEFFICIENTS    #
############################################################


@njit(types.Tuple((f8[:], f8[:], f8[:], f8))(f8[:], f8[:]), cache=True, nogil=True)
def tri_coefficients(vxs, vys):

    x1, x2, x3 = vxs
    y1, y2, y3 = vys

    a1 = x2 * y3 - y2 * x3
    a2 = x3 * y1 - y3 * x1
    a3 = x1 * y2 - y1 * x2
    b1 = y2 - y3
    b2 = y3 - y1
    b3 = y1 - y2
    c1 = x3 - x2
    c2 = x1 - x3
    c3 = x2 - x1

    sA = 0.5 * ((x1 - x3) * (y2 - y1) - (x1 - x2) * (y3 - y1))
    sign = np.sign(sA)
    A = np.abs(sA)
    As = np.array([a1, a2, a3]) * sign
    Bs = np.array([b1, b2, b3]) * sign
    Cs = np.array([c1, c2, c3]) * sign
    return As, Bs, Cs, A


############################################################
#                  GAUSS QUADRATURE POINTS                 #
############################################################

# fmt: off
DPTS = np.array([
    [-0.56250000000000000, 0.52083333333333337, 0.52083333333333337, 0.52083333333333337],  # weights
    [0.33333333333333331, 0.59999999999999998, 0.20000000000000001, 0.20000000000000001],  # L1
    [0.33333333333333331, 0.20000000000000001, 0.59999999999999998, 0.20000000000000001],  # L2
    [0.33333333333333343, 0.20000000000000001, 0.20000000000000007, 0.60000000000000009],  # L3
], dtype=np.float64)
# fmt: on

############################################################
#                 NUMBA OPTIMIZED ASSEMBLER                #
############################################################


@njit(c16[:, :](f8[:, :], c16, i8[:]), cache=True, nogil=True)
def _abc_order_2_terms(tri_vertices, cf, dofcodes):
    """ABC order 2 tangent gradient term"""
    typearry, indexarry = parse_dofcode(dofcodes)
    ndof = dofcodes.shape[0]

    basis, local_vertices = construct_local_vertices(tri_vertices)

    xpts = local_vertices[0, :]
    ypts = local_vertices[1, :]

    aas, bbs, ccs, Area = tri_coefficients(xpts, ypts)
    Area = np.abs(Area)
    bary_coeff = np.empty((3, 3), dtype=np.float64)
    bary_coeff[0, :] = aas / (2 * Area)
    bary_coeff[1, :] = bbs / (2 * Area)
    bary_coeff[2, :] = ccs / (2 * Area)

    WEIGHTS = DPTS[0, :]
    xs = xpts[0] * DPTS[1, :] + xpts[1] * DPTS[2, :] + xpts[2] * DPTS[3, :]
    ys = ypts[0] * DPTS[1, :] + ypts[1] * DPTS[2, :] + ypts[2] * DPTS[3, :]

    coords = np.empty((2, xs.shape[0]), dtype=np.float64)
    coords[0, :] = xs
    coords[1, :] = ys

    ivec = np.array([0, 1, 0])
    jvec = np.array([1, 2, 2])
    kvec = np.array([0, 0, 0])

    CurlMatrix = np.zeros((ndof, ndof), dtype=np.complex128)
    DivMatrix = np.zeros((ndof, ndof), dtype=np.complex128)

    for idof1 in range(ndof):
        i_type = typearry[idof1]
        i_index = indexarry[idof1]

        i1 = ivec[i_index]
        j1 = jvec[i_index]
        k1 = kvec[i_index]

        FC1 = np.zeros((coords.shape[1],), dtype=np.complex128)
        FD1 = np.zeros((coords.shape[1],), dtype=np.complex128)
        FC2 = np.zeros((coords.shape[1],), dtype=np.complex128)
        FD2 = np.zeros((coords.shape[1],), dtype=np.complex128)
        if i_type==0:
            # Edge mode
            _eval_curl_f_2d(bary_coeff, coords, i1, j1, k1, dofcodes[idof1], FC1)
            _eval_div_f_2d(bary_coeff, coords, i1, j1, k1, dofcodes[idof1], FC2)
        else:
            _eval_curl_f_2d(bary_coeff, coords, 0, 1, 2, dofcodes[idof1], FC1)
            _eval_div_f_2d(bary_coeff, coords, 0, 1, 2, dofcodes[idof1], FD2)

        for idof2 in range(ndof):
            i_type = typearry[idof2]
            i_index2 = indexarry[idof2]

            i2 = ivec[i_index2]
            j2 = jvec[i_index2]
            k2 = kvec[i_index2]

            if i_type==0:
                # Edge mode
                _eval_curl_f_2d(bary_coeff, coords, i2, j2, k2, dofcodes[idof2], FC2)
                _eval_div_f_2d(bary_coeff, coords, i2, j2, k2, dofcodes[idof2], FD2)
            else:
                _eval_curl_f_2d(bary_coeff, coords, 0, 1, 2, dofcodes[idof2], FC2)
                _eval_div_f_2d(bary_coeff, coords, 0, 1, 2, dofcodes[idof2], FD2)

            CurlMatrix[idof1, idof2] = np.sum(FC1 * FC2 * WEIGHTS)
            DivMatrix[idof1, idof2] = np.sum(FD1 * FD2 * WEIGHTS)

    out = cf * (CurlMatrix - DivMatrix) * Area

    return out


############################################################
#            NUMBA OPTIMIZED INTEGRAL OVER TETS            #
############################################################


@njit(
    (c16[:])(f8[:, :], i8[:, :], i8[:, :], i8[:, :], i8[:], c16, i8[:]),
    cache=True,
    nogil=True,
    parallel=True,
)
def _matrix_builder(nodes, tris, edges, tri_to_field, tri_ids, coeff, dofcodes):
    """Numba optimized loop over each face triangle."""
    ntritot = tris.shape[1]
    n = dofcodes.shape[0]
    nsq = (n**2)
    nnz = ntritot * nsq

    Mat = np.zeros(nnz, dtype=np.complex128)

    Ntris = tri_ids.shape[0]
    for itri_sub in prange(Ntris):  # type: ignore
        itri = tri_ids[itri_sub]
        p = itri * nsq

        # Construct the local edge map
        tri_nodes = nodes[:, tris[:, itri]]
        subMat = _abc_order_2_terms(tri_nodes, coeff, dofcodes)

        Mat[p : p + nsq] += subMat.ravel()

    return Mat


############################################################
#                     PYTHON INTERFACE                     #
############################################################


def abc_order_2_matrix(
    field: Nedelec2, surf_triangle_indices: np.ndarray, coeff: complex
) -> np.ndarray:
    """Computes the second order absorbing boundary condition correction terms.

    Args:
        field (Nedelec2): The Basis function object
        surf_triangle_indices (np.ndarray): The surface triangle indices to add
        coeff (complex): The integral coefficient jp2/k0

    Returns:
        np.ndarray: The resultant matrix items
    """
    Mat = _matrix_builder(
        field.mesh.nodes,
        field.mesh.tris,
        field.mesh.edges,
        field.tri_to_field,
        surf_triangle_indices,
        coeff,
        field.dofcodes2d
    )
    return Mat
