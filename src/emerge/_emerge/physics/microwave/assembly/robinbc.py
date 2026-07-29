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
from typing import Callable
from loguru import logger
import functools

#
# Toggle this to True when you want to use standard Python breakpoints
DEBUG_MODE = False


def njit(*args, **kwargs):
    """
    Drop-in replacement for numba.njit.
    If DEBUG_MODE is True, it turns into a transparent 'do-nothing' wrapper.
    If DEBUG_MODE is False, it forwards everything to the real Numba compiler.
    """
    if DEBUG_MODE:
        # Case A: Used without parentheses -> @njit
        if len(args) == 1 and callable(args[0]):
            return args[0]

        # Case B: Used with signatures/kwargs -> @njit(cache=True)
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*func_args, **func_kwargs):
                return func(*func_args, **func_kwargs)

            return wrapper

        return decorator
    else:
        # Import Numba lazily only when debugging is turned off
        import numba

        return numba.njit(*args, **kwargs)


@njit(cache=True, fastmath=True, nogil=True)
def optim_matmul(B: np.ndarray, data: np.ndarray):
    dnew = np.zeros_like(data)
    dnew[0, :] = B[0, 0] * data[0, :] + B[0, 1] * data[1, :] + B[0, 2] * data[2, :]
    dnew[1, :] = B[1, 0] * data[0, :] + B[1, 1] * data[1, :] + B[1, 2] * data[2, :]
    dnew[2, :] = B[2, 0] * data[0, :] + B[2, 1] * data[1, :] + B[2, 2] * data[2, :]
    return dnew


@njit(cache=True, fastmath=True, nogil=True)
def optim_matmul_vec(B: np.ndarray, data: np.ndarray):
    dnew = np.zeros((3,), dtype=data.dtype)
    dnew[0] = B[0, 0] * data[0] + B[0, 1] * data[1] + B[0, 2] * data[2]
    dnew[1] = B[1, 0] * data[0] + B[1, 1] * data[1] + B[1, 2] * data[2]
    dnew[2] = B[2, 0] * data[0] + B[2, 1] * data[1] + B[2, 2] * data[2]
    return dnew


@njit(c16[:](c16[:, :], c16[:, :]), cache=True, fastmath=True, nogil=True)
def dot(a: np.ndarray, b: np.ndarray):
    return a[0, :] * b[0, :] + a[1, :] * b[1, :]


@njit(
    types.Tuple((f8[:], f8[:]))(f8[:, :], i8[:, :], f8[:, :], i8[:]),
    cache=True,
    nogil=True,
)
def generate_points(vertices_local, tris, DPTs, surf_triangle_indices):
    NS = surf_triangle_indices.shape[0]
    xall = np.zeros((DPTs.shape[1], NS))
    yall = np.zeros((DPTs.shape[1], NS))

    for i in range(NS):
        itri = surf_triangle_indices[i]
        vertex_ids = tris[:, itri]

        x1, x2, x3 = vertices_local[0, vertex_ids]
        y1, y2, y3 = vertices_local[1, vertex_ids]

        xall[:, i] = x1 * DPTs[1, :] + x2 * DPTs[2, :] + x3 * DPTs[3, :]
        yall[:, i] = y1 * DPTs[1, :] + y2 * DPTs[2, :] + y3 * DPTs[3, :]

    xflat = xall.flatten()
    yflat = yall.flatten()
    return xflat, yflat


@njit(
    types.Tuple((f8[:], f8[:], f8[:]))(f8[:, :], i8[:, :], f8[:, :], i8[:]),
    cache=True,
    nogil=True,
)
def generate_points_3d(vertices, tris, DPTs, surf_triangle_indices):
    NS = surf_triangle_indices.shape[0]
    xall = np.zeros((DPTs.shape[1], NS))
    yall = np.zeros((DPTs.shape[1], NS))
    zall = np.zeros((DPTs.shape[1], NS))
    for i in range(NS):
        itri = surf_triangle_indices[i]
        vertex_ids = tris[:, itri]

        x1, x2, x3 = vertices[0, vertex_ids]
        y1, y2, y3 = vertices[1, vertex_ids]
        z1, z2, z3 = vertices[2, vertex_ids]

        xall[:, i] = x1 * DPTs[1, :] + x2 * DPTs[2, :] + x3 * DPTs[3, :]
        yall[:, i] = y1 * DPTs[1, :] + y2 * DPTs[2, :] + y3 * DPTs[3, :]
        zall[:, i] = z1 * DPTs[1, :] + z2 * DPTs[2, :] + z3 * DPTs[3, :]
    xflat = xall.flatten()
    yflat = yall.flatten()
    zflat = zall.flatten()
    return xflat, yflat, zflat


@njit(f8[:, :](f8[:], f8[:]), cache=True, nogil=True, fastmath=True)
def compute_distances(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    N = xs.shape[0]
    Ds = np.empty((N, N), dtype=np.float64)
    for i in range(N):
        for j in range(i, N):
            Ds[i, j] = np.sqrt((xs[i] - xs[j]) ** 2 + (ys[i] - ys[j]) ** 2)
            Ds[j, i] = Ds[i, j]
    return Ds


@njit(cache=True, nogil=True)
def normalize(a: np.ndarray):
    return a / ((a[0] ** 2 + a[1] ** 2 + a[2] ** 2) ** 0.5)


@njit(c16[:, :](c16[:, :], c16[:, :]), cache=True, nogil=True)
def matmul(Mat, Vec):
    ## Matrix multiplication of a 2x2 Matrix with a Vector
    Vout = np.empty((2, Vec.shape[1]), dtype=np.complex128)
    Vout[0, :] = Mat[0, 0] * Vec[0, :] + Mat[0, 1] * Vec[1, :]
    Vout[1, :] = Mat[1, 0] * Vec[0, :] + Mat[1, 1] * Vec[1, :]
    return Vout


@njit(types.Tuple((f8[:, :], f8[:, :]))(f8[:, :]), cache=True, nogil=True)
def construct_local_vertices(glob_vertices):
    origin = glob_vertices[:, 0]
    vertex_2 = glob_vertices[:, 1]
    vertex_3 = glob_vertices[:, 2]

    edge_1 = vertex_2 - origin
    edge_2 = vertex_3 - origin

    zhat = normalize(cross(edge_1, edge_2))
    xhat = normalize(edge_1)
    yhat = normalize(cross(zhat, xhat))

    basis = np.zeros((3, 3), dtype=np.float64)
    basis[0, :] = xhat
    basis[1, :] = yhat
    basis[2, :] = zhat

    return basis, optim_matmul(basis, glob_vertices - origin[:, np.newaxis])


############################################################
#                         ASSEMBLY                        #
############################################################
# fmt: off
DPTS = np.array([
    [0.10995174365532200, 0.10995174365532200, 0.10995174365532200, 0.22338158967801100, 0.22338158967801100, 0.22338158967801100],  # weights
    [0.81684757298045896, 0.09157621350977101, 0.09157621350977101, 0.10810301816807000, 0.44594849091596500, 0.44594849091596500],  # L1
    [0.09157621350977101, 0.81684757298045896, 0.09157621350977101, 0.44594849091596500, 0.10810301816807000, 0.44594849091596500],  # L2
    [0.09157621350977004, 0.09157621350977008, 0.81684757298045807, 0.44594849091596500, 0.44594849091596500, 0.10810301816807000],  # L3
], dtype=np.float64)
# fmt: on


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

    sA = 0.5 * (b1 * c2 - b2 * c1)
    As = np.array([a1, a2, a3]) / (2 * sA)
    Bs = np.array([b1, b2, b3]) / (2 * sA)
    Cs = np.array([c1, c2, c3]) / (2 * sA)
    return As, Bs, Cs, np.abs(sA)


@njit(c16[:](f8[:, :], c16[:, :], i8[:]), cache=True, nogil=True, parallel=False)
def ned2_tri_force(glob_vertices, glob_Uinc, dofcodes):
    """Nedelec-2 Triangle forcing vector (For Boundary Condition of the Third Kind)"""
    typearry, indexarry = parse_dofcode(dofcodes)
    ndof = dofcodes.shape[0]
    bvec = np.zeros((ndof,), dtype=np.complex128)

    basis, local_vertices = construct_local_vertices(glob_vertices)
    txs = local_vertices[0, :]
    tys = local_vertices[1, :]
    
    aas, bbs, ccs, A = tri_coefficients(txs, tys)
    coeff = np.empty((3, 3), dtype=np.float64)
    coeff[0, :] = aas
    coeff[1, :] = bbs
    coeff[2, :] = ccs

    lcs_Uinc = optim_matmul(basis, glob_Uinc)

    WEIGHTS = DPTS[0, :]
    DPTS1 = DPTS[1, :]
    DPTS2 = DPTS[2, :]
    DPTS3 = DPTS[3, :]

    xs = txs[0] * DPTS1 + txs[1] * DPTS2 + txs[2] * DPTS3
    ys = tys[0] * DPTS1 + tys[1] * DPTS2 + tys[2] * DPTS3

    coords = np.empty((2, xs.shape[0]), dtype=np.float64)
    coords[0, :] = xs
    coords[1, :] = ys

    Ux = lcs_Uinc[0, :]
    Uy = lcs_Uinc[1, :]
    Uinc_2d = np.empty((2, xs.shape[0]), dtype=np.complex128)
    Uinc_2d[0, :] = Ux
    Uinc_2d[1, :] = Uy
    
    ivec = np.array([0, 1, 0])
    jvec = np.array([1, 2, 2])
    kvec = np.array([0, 0, 0])

    fdof = np.zeros((2,coords.shape[1]), dtype=np.complex128)
    for idof in range(ndof):
        i_type = typearry[idof]
        i_index = indexarry[idof]
        if i_type==0:
            i1 = ivec[i_index]
            j1 = jvec[i_index]
            k1 = kvec[i_index]
        else:
            i1 = 0
            j1 = 1
            k1 = 2
        
        _eval_f_2d(coeff, coords, i1, j1, k1, dofcodes[idof], fdof)
        
        bvec[idof] = -A * np.sum(WEIGHTS * (fdof[0, :] * Ux + fdof[1, :] * Uy))
    #print(bvec)
    return bvec


@njit(
    c16[:](f8[:, :], i8[:, :], c16[:], i8[:], c16[:, :, :], i8[:, :], i8[:]),
    cache=True,
    nogil=True,
    parallel=True,
)
def compute_force_entries(
    vertices_global, tris, Bvec, surf_triangle_indices, Uglobal_all, tri_to_field, dofcodes
):
    Niter = surf_triangle_indices.shape[0]
    n_threads = 20
    
    Bvec_private = np.zeros((n_threads, Bvec.shape[0]), dtype=np.complex128)
    chunk_size = (Niter + n_threads - 1) // n_threads

    for t in prange(n_threads):
        start_idx = t * chunk_size
        end_idx = min(start_idx + chunk_size, Niter)
        
        for i in range(start_idx, end_idx):
            itri = surf_triangle_indices[i]
            vertex_ids = tris[:, itri]
            Ulocal = Uglobal_all[:, :, i]
            bvec = ned2_tri_force(vertices_global[:, vertex_ids], Ulocal, dofcodes)
            indices = tri_to_field[:, itri]
            
            Bvec_private[t, indices] += bvec

    for idx in prange(Bvec.shape[0]):
        for t in range(n_threads):
            Bvec[idx] += Bvec_private[t, idx]
            
    return Bvec


@njit(c16[:, :](f8[:, :], c16, i8[:]), cache=True, nogil=True, parallel=False)
def ned2_tri_stiff(glob_vertices, gamma, dofcodes):
    """Nedelec-2 Triangle Stiffness matrix and forcing vector (For Boundary Condition of the Third Kind)"""
    typearry, indexarry = parse_dofcode(dofcodes)
    ndof = dofcodes.shape[0]
    
    Bmat = np.zeros((ndof, ndof), dtype=np.complex128)

    basis, local_vertices = construct_local_vertices(glob_vertices)
    txs = local_vertices[0, :]
    tys = local_vertices[1, :]

    aas, bbs, ccs, A = tri_coefficients(txs, tys)
    A = np.abs(A)
    coeff = np.empty((3, 3), dtype=np.float64)
    coeff[0, :] = aas  # / (2 * A)
    coeff[1, :] = bbs  # / (2 * A)
    coeff[2, :] = ccs  # / (2 * A)

    WEIGHTS = DPTS[0, :]
    DPTS1 = DPTS[1, :]
    DPTS2 = DPTS[2, :]
    DPTS3 = DPTS[3, :]

    xs = txs[0] * DPTS1 + txs[1] * DPTS2 + txs[2] * DPTS3
    ys = tys[0] * DPTS1 + tys[1] * DPTS2 + tys[2] * DPTS3

    coords = np.empty((2, xs.shape[0]), dtype=np.float64)
    coords[0, :] = xs
    coords[1, :] = ys

    ivec = np.array([0, 1, 0])
    jvec = np.array([1, 2, 2])
    kvec = np.array([0, 0, 0])

    fdof1 = np.zeros((2, coords.shape[1]), dtype=np.complex128)
    fdof2 = np.zeros((2, coords.shape[1]), dtype=np.complex128)
    for idof1 in range(ndof):
        i_type1 = typearry[idof1]
        i_index1 = indexarry[idof1]
        if i_type1==0:
            i1 = ivec[i_index1]
            j1 = jvec[i_index1]
            k1 = kvec[i_index1]
        else:
            i1 = 0
            j1 = 1
            k1 = 2

        _eval_f_2d(coeff, coords, i1, j1, k1, dofcodes[idof1], fdof1)

        for idof2 in range(ndof):
            i_type2 = typearry[idof2]
            i_index2 = indexarry[idof2]
            if i_type2==0:
                i2 = ivec[i_index2]
                j2 = jvec[i_index2]
                k2 = kvec[i_index2]
            else:
                i2 = 0
                j2 = 1
                k2 = 2

            _eval_f_2d(coeff, coords, i2, j2, k2, dofcodes[idof2], fdof2)

            Bmat[idof1, idof2] = gamma * np.sum(dot(fdof1, fdof2) * WEIGHTS)
    return Bmat * A


@njit(
    c16[:](f8[:, :], i8[:, :], c16[:], i8[:], c16, i8[:]),
    cache=True,
    nogil=True,
    parallel=True,
)
def compute_bc_entries(vertices, tris, Bmat, surf_triangle_indices, gamma, dofcodes):

    N = dofcodes.shape[0]**2
    Niter = surf_triangle_indices.shape[0]
    for i in prange(Niter):  # type: ignore
        itri = surf_triangle_indices[i]

        vertex_ids = tris[:, itri]
        Bsub = ned2_tri_stiff(vertices[:, vertex_ids], gamma, dofcodes)

        Bmat[itri * N : (itri + 1) * N] = Bmat[itri * N : (itri + 1) * N] + Bsub.ravel()
    return Bmat


def assemble_robin_bc_bvec(
    field: Nedelec2,
    surf_triangle_indices: np.ndarray,
    Ufunc: Callable,
):
    Bvec = np.zeros((field.n_field,), dtype=np.complex128)

    vertices = field.mesh.nodes

    xflat, yflat, zflat = generate_points_3d(
        vertices, field.mesh.tris, DPTS, surf_triangle_indices
    )

    U_global = Ufunc(xflat, yflat, zflat)

    U_global_all = U_global.reshape((3, DPTS.shape[1], surf_triangle_indices.shape[0]))

    Bvec = compute_force_entries(
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
    Bmat: np.ndarray,
    surf_triangle_indices: np.ndarray,
    gamma: np.ndarray,
):

    vertices = field.mesh.nodes
    Bmat = compute_bc_entries(
        vertices, field.mesh.tris, Bmat, surf_triangle_indices, gamma, field.dofcodes2d
    )

    return Bmat


############################################################
#                      SCATTERED FIELD                     #
############################################################


@njit(
    c16[:](f8[:, :], c16[:, :], c16[:, :], f8[:], i8[:]),
    cache=True,
    nogil=True,
    parallel=False,
)
def ned2_tri_force_scat(glob_vertices, glob_Uinc, glob_Uinc_curl, nhat, dofcodes):
    """Nedelec-2 Triangle forcing vector (scattered field, Robin BC)"""
    typearry, indexarry = parse_dofcode(dofcodes)
    ndof = dofcodes.shape[0]
    bvec = np.zeros((ndof,), dtype=np.complex128)

    basis, local_vertices = construct_local_vertices(glob_vertices)
    txs = local_vertices[0, :]
    tys = local_vertices[1, :]

    aas, bbs, ccs, A = tri_coefficients(txs, tys)
    coeff = np.empty((3, 3), dtype=np.float64)
    coeff[0, :] = aas
    coeff[1, :] = bbs
    coeff[2, :] = ccs

    lcs_Uinc = optim_matmul(basis, glob_Uinc)
    lcs_Uinc_curl = optim_matmul(basis, glob_Uinc_curl)
    lcs_nhat = optim_matmul_vec(basis, nhat)
    sgn = np.sign(lcs_nhat[2])

    WEIGHTS = DPTS[0, :]
    xs = txs[0] * DPTS[1, :] + txs[1] * DPTS[2, :] + txs[2] * DPTS[3, :]
    ys = tys[0] * DPTS[1, :] + tys[1] * DPTS[2, :] + tys[2] * DPTS[3, :]

    coords = np.empty((2, xs.shape[0]), dtype=np.float64)
    coords[0, :] = xs
    coords[1, :] = ys

    Ux = lcs_Uinc[0, :] + lcs_Uinc_curl[1, :] * sgn
    Uy = lcs_Uinc[1, :] - lcs_Uinc_curl[0, :] * sgn

    ivec = np.array([0, 1, 0])
    jvec = np.array([1, 2, 2])
    kvec = np.array([0, 0, 0])

    fdof = np.zeros((2, coords.shape[1]), dtype=np.complex128)
    for idof in range(ndof):
        i_type = typearry[idof]
        i_index = indexarry[idof]
        
        if i_type==0:
            i1 = ivec[i_index]
            j1 = jvec[i_index]
            k1 = kvec[i_index]
        else:
            i1 = 0
            j1 = 1
            k1 = 2
        
        _eval_f_2d(coeff, coords, i1, j1, k1, dofcodes[idof], fdof)

        bvec[idof] = -A * np.sum(WEIGHTS * (fdof[0, :] * Ux + fdof[1, :] * Uy))

    return bvec


@njit(
    c16[:](
        f8[:, :],
        i8[:, :],
        c16[:],
        i8[:],
        c16[:, :, :],
        c16[:, :, :],
        i8[:, :],
        f8[:, :],
        i8[:]
    ),
    cache=True,
    nogil=True,
    parallel=True,
)
def compute_force_entries_scat(
    vertices_global,
    tris,
    Bvec,
    surf_triangle_indices,
    Uglobal_all,
    Uglobal_all_curl,
    tri_to_field,
    normals,
    dofcodes,
):
    Niter = surf_triangle_indices.shape[0]
    n_threads = 20
    
    Bvec_private = np.zeros((n_threads, Bvec.shape[0]), dtype=np.complex128)
    chunk_size = (Niter + n_threads - 1) // n_threads

    for t in prange(n_threads):
        start_idx = t * chunk_size
        end_idx = min(start_idx + chunk_size, Niter)
        
        for i in range(start_idx, end_idx):
            itri = surf_triangle_indices[i]
            vertex_ids = tris[:, itri]
            Uglobal = Uglobal_all[:, :, i]
            UglobalCurl = Uglobal_all_curl[:, :, i]

            bvec = ned2_tri_force_scat(
                vertices_global[:, vertex_ids], Uglobal, UglobalCurl, normals[:, i], dofcodes
            )
            indices = tri_to_field[:, itri]

            for k in range(indices.shape[0]):
                dof = indices[k]
                if dof < 0:
                    continue
                Bvec_private[t, dof] += bvec[k]
    for idx in prange(Bvec.shape[0]):
        for t in range(n_threads):
            Bvec[idx] += Bvec_private[t, idx]
            
    return Bvec


def assemble_robin_bc_bvec_scat(
    field: Nedelec2,
    surf_triangle_indices: np.ndarray,
    Ufunc: Callable,
    UfuncCurl: Callable,
    normals: np.ndarray,
):

    Bvec = np.zeros((field.n_field,), dtype=np.complex128)

    vertices = field.mesh.nodes

    xflat, yflat, zflat = generate_points_3d(
        vertices, field.mesh.tris, DPTS, surf_triangle_indices
    )

    U_global = Ufunc(xflat, yflat, zflat)
    U_global_curl = UfuncCurl(xflat, yflat, zflat)

    U_global_all = U_global.reshape((3, DPTS.shape[1], surf_triangle_indices.shape[0]))
    U_global_all_curl = U_global_curl.reshape(
        (3, DPTS.shape[1], surf_triangle_indices.shape[0])
    )

    Bvec = compute_force_entries_scat(
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
