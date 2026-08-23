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

# Last Cleanup: 2026-03-XX -- updated for the generic dofcode-driven basis
# system. Every place that used to slice Etet as
#   Em1s = Etet[0:6]; Ef1s = Etet[6:10]; Em2s = Etet[10:16]; Ef2s = Etet[16:20]
# assumed a fixed 20-DOF (6 edge + 4 face, two orders each) Nedelec-2 layout.
# That assumption no longer holds under the variable-order DoFSet system --
# these now dispatch per-DOF via dofcodes, exactly like ned2_tet_interp does
# in nedelec2.py.
#
# NOTE: verify this relative import path against where adaptive_mesh.py
# actually sits versus ccbf.py -- nedelec2.py uses `from ..ccbf import ...`
# but adaptive_mesh.py may be at a different package depth.
from ...compiled.ccbf import (
    _eval_f_3d, _eval_curl_f_3d, _eval_div_f_3d, _eval_curlcurl_f_3d, parse_dofcode
)

from .microwave_data import MWField
import numpy as np
from ...mth.optimized import matmul, calc_inward_normal
from numba import njit, f8, c16, i8, types, prange, b1  # type: ignore, p
from loguru import logger
from ...const import C0, MU0, EPS0


DEPS = 1e-20
############################################################
#                    OPTIMIZED FUNCTIONS                   #
############################################################


@njit(
    types.Tuple((f8[:, :], f8[:]))(f8[:, :], i8[:, :], f8[:], b1[:]),
    nogil=True,
    cache=True,
)
def tet_to_node(nodes, tets, sizes, included):
    """
    Parameters
    ----------
    nodes : (3, N) float64
        Node coordinates.
    tets : (4, M) int64
        Node indices for each tetrahedron.
    sizes : (M,) float64
        Requested mesh sizes per tetrahedron.
    included : (M,) boolean
        Whether a tetrahedron imposes a size constraint.

    Returns
    -------
    coords_out : (K, 3) float64
        Coordinates of nodes that received a constraint.
    sizes_out : (K,) float64
        Minimum size constraint per returned node.
    """
    N = nodes.shape[1]
    M = tets.shape[1]

    big = 1e300
    node_sizes = np.full(N, big)
    constrained = np.zeros(N, dtype=np.uint8)  # 0/1 flag

    # Accumulate min size per node from included tets
    for e in range(M):
        if included[e]:
            s = sizes[e]
            # four vertices per tet
            n0 = tets[0, e]
            n1 = tets[1, e]
            n2 = tets[2, e]
            n3 = tets[3, e]

            if constrained[n0] == 0:
                node_sizes[n0] = s
                constrained[n0] = 1
            else:
                if s < node_sizes[n0]:
                    node_sizes[n0] = s

            if constrained[n1] == 0:
                node_sizes[n1] = s
                constrained[n1] = 1
            else:
                if s < node_sizes[n1]:
                    node_sizes[n1] = s

            if constrained[n2] == 0:
                node_sizes[n2] = s
                constrained[n2] = 1
            else:
                if s < node_sizes[n2]:
                    node_sizes[n2] = s

            if constrained[n3] == 0:
                node_sizes[n3] = s
                constrained[n3] = 1
            else:
                if s < node_sizes[n3]:
                    node_sizes[n3] = s

    # Count constrained nodes
    K = 0
    for i in range(N):
        if constrained[i] == 1:
            K += 1

    # Pack outputs
    coords_out = np.empty((3, K), dtype=nodes.dtype)
    sizes_out = np.empty(K, dtype=sizes.dtype)

    j = 0
    for i in range(N):
        if constrained[i] == 1:
            coords_out[0, j] = nodes[0, i]
            coords_out[1, j] = nodes[1, i]
            coords_out[2, j] = nodes[2, i]
            sizes_out[j] = node_sizes[i]
            j += 1

    return coords_out, sizes_out

@njit(
    types.Tuple((f8[:, :], f8[:], i8[:]))(f8[:, :], i8[:, :], f8[:], b1[:]),
    nogil=True,
    cache=True,
)
def tet_to_node_new(nodes, tets, sizes, included):
    """
    Same as before, plus a third return value: the node index each
    returned (coord, size) pair belongs to, so callers can scatter the
    result into a full-length node array without a coordinate lookup.

    Parameters
    ----------
    nodes : (3, N) float64
    tets : (4, M) int64
    sizes : (M,) float64
    included : (M,) boolean

    Returns
    -------
    coords_out : (3, K) float64
    sizes_out : (K,) float64
    indices_out : (K,) int64
        Node index (0..N-1) each entry of coords_out/sizes_out corresponds to.
    """
    N = nodes.shape[1]
    M = tets.shape[1]

    big = 1e300
    node_sizes = np.full(N, big)
    constrained = np.zeros(N, dtype=np.uint8)

    for e in range(M):
        if included[e]:
            s = sizes[e]
            n0 = tets[0, e]
            n1 = tets[1, e]
            n2 = tets[2, e]
            n3 = tets[3, e]

            if constrained[n0] == 0:
                node_sizes[n0] = s
                constrained[n0] = 1
            elif s < node_sizes[n0]:
                node_sizes[n0] = s

            if constrained[n1] == 0:
                node_sizes[n1] = s
                constrained[n1] = 1
            elif s < node_sizes[n1]:
                node_sizes[n1] = s

            if constrained[n2] == 0:
                node_sizes[n2] = s
                constrained[n2] = 1
            elif s < node_sizes[n2]:
                node_sizes[n2] = s

            if constrained[n3] == 0:
                node_sizes[n3] = s
                constrained[n3] = 1
            elif s < node_sizes[n3]:
                node_sizes[n3] = s

    K = 0
    for i in range(N):
        if constrained[i] == 1:
            K += 1

    coords_out = np.empty((3, K), dtype=nodes.dtype)
    sizes_out = np.empty(K, dtype=sizes.dtype)
    indices_out = np.empty(K, dtype=np.int64)

    j = 0
    for i in range(N):
        if constrained[i] == 1:
            coords_out[0, j] = nodes[0, i]
            coords_out[1, j] = nodes[1, i]
            coords_out[2, j] = nodes[2, i]
            sizes_out[j] = node_sizes[i]
            indices_out[j] = i
            j += 1

    return coords_out, sizes_out, indices_out


@njit(cache=True, nogil=True)
def diam_circum_circle(
    v1: np.ndarray, v2: np.ndarray, v3: np.ndarray, eps: float = 1e-14
) -> float:
    """
    Diameter of the circumcircle of triangle (v1,v2,v3) in 3D.
    Uses D = (|AB|*|AC|*|BC|) / ||AB x AC||.
    Returns np.inf if points are (near) collinear.
    """
    AB = v2 - v1
    AC = v3 - v1
    BC = v3 - v2

    a = np.linalg.norm(BC)
    b = np.linalg.norm(AC)
    c = np.linalg.norm(AB)

    cross = np.cross(AB, AC)
    denom = np.linalg.norm(cross)  # 2 * area of triangle

    if denom < eps:
        return np.inf  # degenerate/collinear

    return (a * b * c) / denom  # diameter = 2R


@njit(cache=True, nogil=True)
def circum_sphere_diam(
    v1: np.ndarray, v2: np.ndarray, v3: np.ndarray, v4: np.ndarray, eps: float = 1e-14
) -> float:
    """
    Diameter of the circumsphere of tetrahedron (v1,v2,v3,v4) in 3D.
    Solves for center c from 2*(pi - p4) · c = |pi|^2 - |p4|^2, i=1..3.
    Returns np.inf if points are (near) coplanar/degenerate.
    """
    p1, p2, p3, p4 = v1, v2, v3, v4

    M = np.empty((3, 3), dtype=np.float64)
    M[0, :] = 2.0 * (p1 - p4)
    M[1, :] = 2.0 * (p2 - p4)
    M[2, :] = 2.0 * (p3 - p4)

    # manual 3x3 determinant (Numba-friendly)
    det = (
        M[0, 0] * (M[1, 1] * M[2, 2] - M[1, 2] * M[2, 1])
        - M[0, 1] * (M[1, 0] * M[2, 2] - M[1, 2] * M[2, 0])
        + M[0, 2] * (M[1, 0] * M[2, 1] - M[1, 1] * M[2, 0])
    )
    if np.abs(det) < eps:
        return np.inf  # coplanar/degenerate

    rhs = np.empty(3, dtype=np.float64)
    rhs[0] = np.sum(p1**2) - np.sum(p4**2)
    rhs[1] = np.sum(p2**2) - np.sum(p4**2)
    rhs[2] = np.sum(p3**2) - np.sum(p4**2)

    # Solve for circumcenter
    c = np.linalg.solve(M, rhs)

    # Radius = distance to any vertex
    R = np.linalg.norm(c - p1)
    if R > 1e10:
        return 1e10
    return 2.0 * R  # diameter


def print_sparam_matrix(pre: str, S: np.ndarray):
    """
    Print an N x N complex S-parameter matrix in dB∠deg format.
    Magnitude in dB rounded to 2 decimals, phase in degrees with 1 decimal.
    """
    S = np.asarray(S)
    if S.ndim != 2 or S.shape[0] != S.shape[1]:
        raise ValueError("S must be a square (N x N) complex matrix")

    N = S.shape[0]
    logger.debug(pre + "S-parameter matrix (dB ∠ deg):")

    for i in range(N):
        row_str = []
        for j in range(N):
            mag_db = 20 * np.log10(np.abs(S[i, j]) + np.finfo(float).eps)
            phase_deg = np.degrees(np.angle(S[i, j]))
            row_str.append(f"{mag_db:6.2f} dB ∠ {phase_deg:6.1f}°")
        logger.debug(" | ".join(row_str))


def compute_convergence(Sold: np.ndarray, Snew: np.ndarray) -> float:
    """
    Return a single scalar: max |Snew - Sold|.
    Works for shapes (N,N) or (..., N, N); reduces over all axes.
    """

    Sold = np.asarray(Sold)
    Snew = np.asarray(Snew)
    print_sparam_matrix("Old:", Sold)
    print_sparam_matrix("New", Snew)
    if Sold.shape != Snew.shape:
        raise ValueError("Sold and Snew must have identical shapes")
    # amp_conv = float(np.abs(np.abs(Snew) - np.abs(Sold)).max())
    mag_conv = float(np.abs(np.abs(Snew) - np.abs(Sold)).max())
    amp_conv = float(np.abs(Snew - Sold).max())
    phase_conv = (
        float(np.abs(np.angle(np.diag(Sold) / np.diag(Snew))).max()) * 180 / np.pi
    )
    return amp_conv, mag_conv, phase_conv


def select_refinement_indices(errors: np.ndarray, refine: float) -> np.ndarray:
    """
    Dörfler marking:
    Choose the minimal number of elements whose squared error sum
    reaches at least 'refine' (theta in [0,1]) of the global squared error.

    Args:
        errors: 1D or ND array of local error indicators (nonnegative).
        refine: theta in [0,1]; fraction of total error energy to target.

    Returns:
        np.ndarray of indices (ints) sorted by decreasing error.
    """
    # Flatten and sanitize
    errs = np.abs(errors)

    ind = errs * errs
    total = np.sum(ind)

    # Sort by decreasing indicator
    order = np.argsort(ind)[::-1]
    sum_error = 0
    indices = []
    for index in order:
        sum_error += ind[index]
        indices.append(index)
        if sum_error >= refine * total:
            break

    return np.array(indices)


@njit(f8[:](i8, f8[:, :], f8, f8, f8[:]), cache=True, nogil=True, parallel=False)
def compute_size(
    index: int, coords: np.ndarray, gr: float, scaler: float, dss: np.ndarray
) -> float:
    """Optimized function to compute the size impressed by size constraint points on each other size constraint point.

    Args:
        id (int): _description_
        coords (np.ndarray): _description_
        gr (float): growth rate
        scaler (float): _description_
        dss (np.ndarray): _description_

    Returns:
        float: _description_
    """
    N = dss.shape[0]
    sizes = np.zeros((N,), dtype=np.float64) - 1.0
    x, y, z = coords[:, index]
    for n in prange(N):
        if n == index:
            sizes[n] = dss[n] * scaler
            continue
        distance = (
            (x - coords[0, n]) ** 2 + (y - coords[1, n]) ** 2 + (z - coords[2, n]) ** 2
        ) ** 0.5
        nsize = scaler * dss[n] / gr - (1 - gr) / gr * max(0, (distance - dss[n] * 0))
        sizes[n] = nsize
    return sizes


@njit(f8[:](f8[:, :], i8, i8[:]), cache=True, nogil=True, parallel=True)
def nbmin(matrix, axis, include):

    if axis == 0:
        N = matrix.shape[1]
        out = np.empty((N,), dtype=np.float64)
        for n in prange(N):
            out[n] = np.min(matrix[include == 1, n])
        return out
    if axis == 1:
        N = matrix.shape[0]
        out = np.empty((N,), dtype=np.float64)
        for n in prange(N):
            out[n] = np.min(matrix[n, include == 1])
        return out
    else:
        out = np.empty((N,), dtype=np.float64)
        return out


@njit(i8(i8, f8[:, :], f8, i8[:]), cache=True, nogil=True, parallel=False)
def can_remove(index: int, M: np.ndarray, scaling: float, include: np.ndarray) -> int:

    ratio = np.min(M[index, :] / nbmin(M, 1, include))

    if ratio > scaling:
        return 1
    return 0


@njit(i8[:](f8[:, :], f8, f8[:], f8, f8), cache=True, nogil=True, parallel=False)
def reduce_point_set(
    coords: np.ndarray,
    gr: float,
    dss: np.ndarray,
    scaler: float,
    keep_percentage: float,
) -> list[int]:
    N = dss.shape[0]
    impressed_size = np.zeros((N, N), np.float64)

    include = np.ones((N,), dtype=np.int64)

    for n in range(N):
        impressed_size[n, :] = compute_size(n, coords, gr, scaler, dss)

    current_min = nbmin(impressed_size, 1, include)
    counter = 0

    for i in range(N):
        if (N - counter) / N < keep_percentage:
            break

        if (N - counter) <= 20:
            break

        if current_min[i] <= impressed_size[i, i] * 0.8:
            include[i] = 0
            counter = counter + 1
            current_min = nbmin(impressed_size, 1, include)

    ids = np.arange(N)
    output = ids[include == 1]
    return output


@njit(i8[:, :](i8[:], i8[:, :]), cache=True, nogil=True)
def local_mapping(vertex_ids, triangle_ids):
    """
    Parameters
    ----------
    vertex_ids   : 1-D int64 array (length 4)
        Global vertex numbers of one tetrahedron, in *its* order
        (v0, v1, v2, v3).

    triangle_ids : 2-D int64 array (nTri × 3)
        Each row is a global-ID triple of one face that belongs to this tet.

    Returns
    -------
    local_tris   : 2-D int64 array (nTri × 3)
        Same triangles, but every entry replaced by the local index
        0,1,2,3 that the vertex has inside this tetrahedron.
        (Guaranteed to be ∈{0,1,2,3}; no -1 ever appears if the input
        really belongs to the tet.)
    """
    ndim = triangle_ids.shape[0]
    ntri = triangle_ids.shape[1]
    out = np.zeros(triangle_ids.shape, dtype=np.int64)

    for t in range(ntri):  # each triangle
        for j in range(ndim):  # each vertex in that triangle
            gid = triangle_ids[j, t]  # global ID to look up

            # linear search over the four tet vertices
            for k in range(4):
                if vertex_ids[k] == gid:
                    out[j, t] = k  # store local index 0-3
                    break  # stop the k-loop

    return out


@njit(f8(f8[:], f8[:], f8[:]), cache=True, nogil=True)
def compute_volume(xs, ys, zs):
    x1, x2, x3, x4 = xs
    y1, y2, y3, y4 = ys
    z1, z2, z3, z4 = zs

    return np.abs(
        -x1 * y2 * z3 / 6
        + x1 * y2 * z4 / 6
        + x1 * y3 * z2 / 6
        - x1 * y3 * z4 / 6
        - x1 * y4 * z2 / 6
        + x1 * y4 * z3 / 6
        + x2 * y1 * z3 / 6
        - x2 * y1 * z4 / 6
        - x2 * y3 * z1 / 6
        + x2 * y3 * z4 / 6
        + x2 * y4 * z1 / 6
        - x2 * y4 * z3 / 6
        - x3 * y1 * z2 / 6
        + x3 * y1 * z4 / 6
        + x3 * y2 * z1 / 6
        - x3 * y2 * z4 / 6
        - x3 * y4 * z1 / 6
        + x3 * y4 * z2 / 6
        + x4 * y1 * z2 / 6
        - x4 * y1 * z3 / 6
        - x4 * y2 * z1 / 6
        + x4 * y2 * z3 / 6
        + x4 * y3 * z1 / 6
        - x4 * y3 * z2 / 6
    )


@njit(
    types.Tuple((f8[:], f8[:], f8[:], f8[:], f8))(f8[:], f8[:], f8[:]),
    cache=True,
    nogil=True,
)
def tet_coefficients(xs, ys, zs):
    ## THIS FUNCTION WORKS
    x1, x2, x3, x4 = xs
    y1, y2, y3, y4 = ys
    z1, z2, z3, z4 = zs

    aas = np.empty((4,), dtype=np.float64)
    bbs = np.empty((4,), dtype=np.float64)
    ccs = np.empty((4,), dtype=np.float64)
    dds = np.empty((4,), dtype=np.float64)

    V = np.abs(
        -x1 * y2 * z3 / 6
        + x1 * y2 * z4 / 6
        + x1 * y3 * z2 / 6
        - x1 * y3 * z4 / 6
        - x1 * y4 * z2 / 6
        + x1 * y4 * z3 / 6
        + x2 * y1 * z3 / 6
        - x2 * y1 * z4 / 6
        - x2 * y3 * z1 / 6
        + x2 * y3 * z4 / 6
        + x2 * y4 * z1 / 6
        - x2 * y4 * z3 / 6
        - x3 * y1 * z2 / 6
        + x3 * y1 * z4 / 6
        + x3 * y2 * z1 / 6
        - x3 * y2 * z4 / 6
        - x3 * y4 * z1 / 6
        + x3 * y4 * z2 / 6
        + x4 * y1 * z2 / 6
        - x4 * y1 * z3 / 6
        - x4 * y2 * z1 / 6
        + x4 * y2 * z3 / 6
        + x4 * y3 * z1 / 6
        - x4 * y3 * z2 / 6
    )

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


DPTS_2D = np.array([
    [0.10995174365532200, 0.10995174365532200, 0.10995174365532200, 0.22338158967801100, 0.22338158967801100, 0.22338158967801100],  # weights
    [0.81684757298045896, 0.09157621350977101, 0.09157621350977101, 0.10810301816807000, 0.44594849091596500, 0.44594849091596500],  # L1
    [0.09157621350977101, 0.81684757298045896, 0.09157621350977101, 0.44594849091596500, 0.10810301816807000, 0.44594849091596500],  # L2
    [0.09157621350977004, 0.09157621350977008, 0.81684757298045807, 0.44594849091596500, 0.44594849091596500, 0.10810301816807000],  # L3
], dtype=np.float64)

DPTS_3D = np.array([
    [-0.07893333333333329982, 0.04573333333333329948, 0.04573333333333329948, 0.04573333333333329948, 0.04573333333333329948, 0.14933333333333329018, 0.14933333333333329018, 0.14933333333333329018, 0.14933333333333329018, 0.14933333333333329018, 0.14933333333333329018],  # weights
    [0.25000000000000000000, 0.78571428571428569843, 0.07142857142857139685, 0.07142857142857139685, 0.07142857142857139685, 0.10059642383320080428, 0.39940357616679922348, 0.39940357616679922348, 0.39940357616679922348, 0.10059642383320080428, 0.10059642383320080428],  # L1
    [0.25000000000000000000, 0.07142857142857139685, 0.07142857142857139685, 0.07142857142857139685, 0.78571428571428569843, 0.39940357616679922348, 0.10059642383320080428, 0.39940357616679922348, 0.10059642383320080428, 0.39940357616679922348, 0.10059642383320080428],  # L2
    [0.25000000000000000000, 0.07142857142857139685, 0.07142857142857139685, 0.78571428571428569843, 0.07142857142857139685, 0.39940357616679922348, 0.39940357616679922348, 0.10059642383320080428, 0.10059642383320080428, 0.10059642383320080428, 0.39940357616679922348],  # L3
    [0.25000000000000000000, 0.07142857142857150787, 0.78571428571428580945, 0.07142857142857150787, 0.07142857142857150787, 0.10059642383320077652, 0.10059642383320077652, 0.10059642383320074877, 0.39940357616679922348, 0.39940357616679922348, 0.39940357616679922348],  # L4
], dtype=np.float64)


############################################################
#   GENERIC, DOFCODE-DRIVEN FIELD / CURL / DIV / CURLCURL   #
############################################################
# Replaces the old fixed 6/4/6/4 (Em1s/Ef1s/Em2s/Ef2s) split. Each of these
# dispatches per-DOF via parse_dofcode + _eval_*_f_3d, exactly like
# ned2_tet_interp / ned2_tet_interp_curl in nedelec2.py -- so a single tet's
# DOF layout is read the same way everywhere in the codebase now, instead of
# this file assuming a fixed order-1 Nedelec-2 layout independently.
@njit(c16[:, :](f8[:, :], f8[:, :], c16[:], i8[:, :], i8[:, :], i8[:]),
      cache=True, nogil=True)
def compute_field(
    coords: np.ndarray,
    vertices: np.ndarray,
    Etet: np.ndarray,
    l_edge_ids: np.ndarray,
    l_tri_ids: np.ndarray,
    dofcodes: np.ndarray,
):
    N = coords.shape[1]
    xvs = vertices[0, :]
    yvs = vertices[1, :]
    zvs = vertices[2, :]
 
    a_s, b_s, c_s, d_s, V = tet_coefficients(xvs, yvs, zvs)
    coeff = np.empty((4, 4), dtype=np.float64)
    coeff[0, :] = a_s
    coeff[1, :] = b_s
    coeff[2, :] = c_s
    coeff[3, :] = d_s
 
    Exl = np.zeros(N, dtype=np.complex128)
    Eyl = np.zeros(N, dtype=np.complex128)
    Ezl = np.zeros(N, dtype=np.complex128)
 
    typearray, indexarray = parse_dofcode(dofcodes)
    nidof = dofcodes.shape[0]
 
    # Allocated ONCE, reused across the DOF loop -- _eval_f_3d writes into
    # it in place now (reverted void/in-place convention), matching how
    # ned2_tet_interp already does this. No per-DOF reallocation.
    F = np.zeros((3, N), dtype=np.complex128)
    for idof in range(nidof):
        i_type = typearray[idof]
        i_index = indexarray[idof]
 
        if i_type == 0:
            i1 = l_edge_ids[0, i_index]
            j1 = l_edge_ids[1, i_index]
            k1 = 0
        else:
            i1 = l_tri_ids[0, i_index]
            j1 = l_tri_ids[1, i_index]
            k1 = l_tri_ids[2, i_index]
 
        _eval_f_3d(coeff, coords, i1, j1, k1, dofcodes[idof], F)
        Exl += F[0, :] * Etet[idof]
        Eyl += F[1, :] * Etet[idof]
        Ezl += F[2, :] * Etet[idof]
 
    out = np.zeros((3, N), dtype=np.complex128)
    out[0, :] = Exl
    out[1, :] = Eyl
    out[2, :] = Ezl
    return out
 
 
@njit(c16[:, :](f8[:, :], f8[:, :], c16[:], i8[:, :], i8[:, :], i8[:]),
      cache=True, nogil=True)
def compute_curl(
    coords: np.ndarray,
    vertices: np.ndarray,
    Etet: np.ndarray,
    l_edge_ids: np.ndarray,
    l_tri_ids: np.ndarray,
    dofcodes: np.ndarray,
):
    N = coords.shape[1]
    xvs = vertices[0, :]
    yvs = vertices[1, :]
    zvs = vertices[2, :]
 
    a_s, b_s, c_s, d_s, V = tet_coefficients(xvs, yvs, zvs)
    coeff = np.empty((4, 4), dtype=np.float64)
    coeff[0, :] = a_s
    coeff[1, :] = b_s
    coeff[2, :] = c_s
    coeff[3, :] = d_s
 
    Exl = np.zeros(N, dtype=np.complex128)
    Eyl = np.zeros(N, dtype=np.complex128)
    Ezl = np.zeros(N, dtype=np.complex128)
 
    typearray, indexarray = parse_dofcode(dofcodes)
    nidof = dofcodes.shape[0]
 
    F = np.zeros((3, N), dtype=np.complex128)
    for idof in range(nidof):
        i_type = typearray[idof]
        i_index = indexarray[idof]
 
        if i_type == 0:
            i1 = l_edge_ids[0, i_index]
            j1 = l_edge_ids[1, i_index]
            k1 = 0
        else:
            i1 = l_tri_ids[0, i_index]
            j1 = l_tri_ids[1, i_index]
            k1 = l_tri_ids[2, i_index]
 
        _eval_curl_f_3d(coeff, coords, i1, j1, k1, dofcodes[idof], F)
        Exl += F[0, :] * Etet[idof]
        Eyl += F[1, :] * Etet[idof]
        Ezl += F[2, :] * Etet[idof]
 
    out = np.zeros((3, N), dtype=np.complex128)
    out[0, :] = Exl
    out[1, :] = Eyl
    out[2, :] = Ezl
    return out
 
 
@njit(c16[:](f8[:, :], f8[:, :], c16[:], i8[:, :], i8[:, :], c16[:, :], i8[:]),
      cache=True, nogil=True)
def compute_div(
    coords: np.ndarray,
    vertices: np.ndarray,
    Etet: np.ndarray,
    l_edge_ids: np.ndarray,
    l_tri_ids: np.ndarray,
    Um: np.ndarray,
    dofcodes: np.ndarray,
):
    """Isotropic-Um assumption unchanged from before -- see prior notes.
    Only the internal _eval_div_f_3d calling convention changes here.
    """
    N = coords.shape[1]
    xvs = vertices[0, :]
    yvs = vertices[1, :]
    zvs = vertices[2, :]
 
    a_s, b_s, c_s, d_s, V = tet_coefficients(xvs, yvs, zvs)
    coeff = np.empty((4, 4), dtype=np.float64)
    coeff[0, :] = a_s / (6*V)
    coeff[1, :] = b_s / (6*V)
    coeff[2, :] = c_s / (6*V)
    coeff[3, :] = d_s / (6*V)
 
    u_scalar = Um[0, 0]
 
    typearray, indexarray = parse_dofcode(dofcodes)
    nidof = dofcodes.shape[0]
 
    difE = np.zeros(N, dtype=np.complex128)
    Fdiv = np.zeros(N, dtype=np.complex128)
    for idof in range(nidof):
        i_type = typearray[idof]
        i_index = indexarray[idof]
 
        if i_type == 0:
            i1 = l_edge_ids[0, i_index]
            j1 = l_edge_ids[1, i_index]
            k1 = 0
        else:
            i1 = l_tri_ids[0, i_index]
            j1 = l_tri_ids[1, i_index]
            k1 = l_tri_ids[2, i_index]
 
        _eval_div_f_3d(coeff, coords, i1, j1, k1, dofcodes[idof], Fdiv)
        difE += Etet[idof] * Fdiv
 
    return u_scalar * difE
 
 
@njit(c16[:](f8[:, :], c16[:], i8[:, :], i8[:, :], c16[:, :], i8[:]),
      cache=True, nogil=True)
def compute_curl_curl(
    vertices: np.ndarray,
    Etet: np.ndarray,
    l_edge_ids: np.ndarray,
    l_tri_ids: np.ndarray,
    Um: np.ndarray,
    dofcodes: np.ndarray,
):
    """Exact for general (even non-diagonal) Um -- see prior notes. Only
    the internal _eval_curlcurl_f_3d calling convention changes here.
    """
    xvs = vertices[0, :]
    yvs = vertices[1, :]
    zvs = vertices[2, :]
 
    a_s, b_s, c_s, d_s, V = tet_coefficients(xvs, yvs, zvs)
    coeff = np.empty((4, 4), dtype=np.float64)
    coeff[0, :] = a_s
    coeff[1, :] = b_s
    coeff[2, :] = c_s
    coeff[3, :] = d_s
 
    uxx, uxy, uxz = Um[0, 0], Um[0, 1], Um[0, 2]
    uyx, uyy, uyz = Um[1, 0], Um[1, 1], Um[1, 2]
    uzx, uzy, uzz = Um[2, 0], Um[2, 1], Um[2, 2]
 
    dummy_coords = np.zeros((3, 1), dtype=np.float64)
 
    typearray, indexarray = parse_dofcode(dofcodes)
    nidof = dofcodes.shape[0]
 
    Exl = 0.0 + 0.0j
    Eyl = 0.0 + 0.0j
    Ezl = 0.0 + 0.0j
 
    F = np.zeros((3, 1), dtype=np.complex128)
    for idof in range(nidof):
        i_type = typearray[idof]
        i_index = indexarray[idof]
 
        if i_type == 0:
            i1 = l_edge_ids[0, i_index]
            j1 = l_edge_ids[1, i_index]
            k1 = 0
        else:
            i1 = l_tri_ids[0, i_index]
            j1 = l_tri_ids[1, i_index]
            k1 = l_tri_ids[2, i_index]
 
        _eval_curlcurl_f_3d(coeff, dummy_coords, i1, j1, k1, dofcodes[idof], F)
        fx, fy, fz = F[0, 0], F[1, 0], F[2, 0]
 
        Exl += Etet[idof] * (uxx * fx + uxy * fy + uxz * fz)
        Eyl += Etet[idof] * (uyx * fx + uyy * fy + uyz * fz)
        Ezl += Etet[idof] * (uzx * fx + uzy * fy + uzz * fz)
 
    out = np.empty(3, dtype=np.complex128)
    out[0] = Exl
    out[1] = Eyl
    out[2] = Ezl
    return out

@njit(c16[:, :](c16[:], c16[:, :]), cache=True, fastmath=True, nogil=True)
def cross_c_arry(a: np.ndarray, b: np.ndarray):
    """Optimized complex single vector cross product

    Args:
        a (np.ndarray): (3,) vector a
        b (np.ndarray): (3,) vector b

    Returns:
        np.ndarray: a ⨉ b
    """
    crossv = np.empty((3, b.shape[1]), dtype=np.complex128)
    crossv[0, :] = a[1] * b[2, :] - a[2] * b[1, :]
    crossv[1, :] = a[2] * b[0, :] - a[0] * b[2, :]
    crossv[2, :] = a[0] * b[1, :] - a[1] * b[0, :]
    return crossv


@njit(c16[:](c16[:], c16[:, :]), cache=True, fastmath=True, nogil=True)
def dot_c_arry(a: np.ndarray, b: np.ndarray):
    """Optimized complex single vector cross product

    Args:
        a (np.ndarray): (3,) vector a
        b (np.ndarray): (3,) vector b

    Returns:
        np.ndarray: a ⨉ b
    """
    dotv = a[0] * b[0, :] + a[1] * b[1, :] + a[2] * b[2, :]
    return dotv


@njit(
    types.Tuple((f8[:], f8[:]))(
        f8[:, :],
        i8[:, :],
        i8[:, :],
        i8[:, :],
        f8[:, :],
        c16[:],
        f8[:],
        f8[:],
        i8[:, :],
        i8[:, :],
        f8[:, :],
        i8[:, :],
        i8[:, :],
        c16[:],
        c16[:],
        i8[:],
        f8,
        i8[:],
    ),
    cache=True,
    nogil=True,
)
def compute_error_single(
    nodes,
    tets,
    tris,
    edges,
    centers,
    Efield,
    edge_lengths,
    areas,
    tet_to_edge,
    tet_to_tri,
    tri_centers,
    tri_to_tet,
    tet_to_field,
    er,
    ur,
    pec_tris,
    k0,
    dofcodes3d,
) -> np.ndarray:

    tet_is_pec = np.zeros((tris.shape[1],), dtype=np.bool_)
    tet_is_pec[pec_tris] = True

    # CONSTANTS
    N_TETS = tets.shape[1]
    N2D = DPTS_2D.shape[1]
    WEIGHTS_VOL = DPTS_3D[0, :]

    W0 = k0 * C0
    Y0 = np.sqrt(1 / MU0)

    # INIT POSTERIORI ERROR ESTIMATE QUANTITIES
    alpha_t = np.zeros((N_TETS,), dtype=np.complex128)
    max_elem_size = np.zeros((N_TETS,), dtype=np.float64)

    Qf_face1 = np.zeros((4, N2D, N_TETS), dtype=np.complex128)
    Qf_face2 = np.zeros((4, N2D, N_TETS), dtype=np.complex128)
    Jf_face1 = np.zeros((4, 3, N2D, N_TETS), dtype=np.complex128)
    Jf_face2 = np.zeros((4, 3, N2D, N_TETS), dtype=np.complex128)

    areas_face_residual = np.zeros((4, N2D, N_TETS), dtype=np.float64)
    Rf_face_residual = np.zeros((4, N_TETS), dtype=np.float64)
    adj_tets_mat = -np.ones((4, N_TETS), dtype=np.int32)

    # Compute Error estimate
    for itet in range(N_TETS):
        uinv = (1 / ur[itet]) * np.eye(3)
        ermat = er[itet] * np.eye(3)
        erc = er[itet]
        urc = ur[itet]

        # GEOMETRIC QUANTITIES
        vertices = nodes[:, tets[:, itet]]
        v1 = vertices[:, 0]
        v2 = vertices[:, 1]
        v3 = vertices[:, 2]
        v4 = vertices[:, 3]

        # VOLUME INTEGRATION POINTS
        vxs = (
            DPTS_3D[1, :] * v1[0]
            + DPTS_3D[2, :] * v2[0]
            + DPTS_3D[3, :] * v3[0]
            + DPTS_3D[4, :] * v4[0]
        )
        vys = (
            DPTS_3D[1, :] * v1[1]
            + DPTS_3D[2, :] * v2[1]
            + DPTS_3D[3, :] * v3[1]
            + DPTS_3D[4, :] * v4[1]
        )
        vzs = (
            DPTS_3D[1, :] * v1[2]
            + DPTS_3D[2, :] * v2[2]
            + DPTS_3D[3, :] * v3[2]
            + DPTS_3D[4, :] * v4[2]
        )

        intpts = np.empty((3, DPTS_3D.shape[1]), dtype=np.float64)
        intpts[0, :] = vxs
        intpts[1, :] = vys
        intpts[2, :] = vzs

        # TET TRI/EDGE NODE COUPLINGS -- geometry only, independent of how
        # many DOF "orders" exist per edge/face (unlike the old code, which
        # derived these from tet_to_field[:6]/[6:10], silently assuming a
        # fixed 6-edge/4-face-DOF layout). tet_to_edge/tet_to_tri always
        # have exactly 6/4 entries per tet regardless of DOF order count.
        g_node_ids = tets[:, itet]
        g_edge_ids = edges[:, tet_to_edge[:, itet]]
        g_tri_ids = tris[:, tet_to_tri[:, itet]]
        l_edge_ids = local_mapping(g_node_ids, g_edge_ids)
        l_tri_ids = local_mapping(g_node_ids, g_tri_ids)
        triids = tet_to_tri[:, itet]

        #size_max = circum_sphere_diam(v1, v2, v3, v4)
        #size_max = min(size_max, np.max(edge_lengths[tet_to_edge[:, itet]]))
        size_max = np.max(edge_lengths[tet_to_edge[:, itet]])
        TET_VOLUME = compute_volume(vertices[0, :], vertices[1, :], vertices[2, :])
        Rt = ((3*TET_VOLUME)/(4*np.pi))**(1/3)

        # Efield -- full per-tet DOF vector, in the same order as dofcodes3d.
        # No more Em1s/Ef1s/Em2s/Ef2s fixed slicing -- compute_field/curl/div/
        # curl_curl dispatch per-DOF internally via dofcodes3d.
        Ef = Efield[tet_to_field[:, itet]]

        # Qt term
        Qt = (
            TET_VOLUME
            * EPS0
            * np.sum(
                WEIGHTS_VOL
                * compute_div(intpts, vertices, Ef, l_edge_ids, l_tri_ids, ermat, dofcodes3d),
                axis=0,
            )
        )

        # Jt term
        Rv1 = compute_curl_curl(vertices, Ef, l_edge_ids, l_tri_ids, uinv, dofcodes3d)
        Rv2 = -(k0**2) * matmul(
            ermat, compute_field(intpts, vertices, Ef, l_edge_ids, l_tri_ids, dofcodes3d)
        )
        Rv = 1 * Rv2
        Rv[0, :] += Rv1[0]  # X-component
        Rv[1, :] += Rv1[1]  # Y-component
        Rv[2, :] += Rv1[2]  # Z-component

        Rv[0, :] = Rv[0, :] * WEIGHTS_VOL
        Rv[1, :] = Rv[1, :] * WEIGHTS_VOL
        Rv[2, :] = Rv[2, :] * WEIGHTS_VOL

        Jt = -TET_VOLUME * np.sum(1 / (1j * W0 * MU0) * Rv, axis=1)

        Gt = 1j * W0 * np.exp(-1j * k0 * Rt) / (4 * np.pi * Rt + DEPS)

        alpha_t[itet] = -Gt / (erc * EPS0 + DEPS) * Qt * Qt - Gt * urc * MU0 * np.sum(
            Jt * Jt
        )
        if np.isnan(
            np.abs(-Gt / (erc * EPS0) * Qt * Qt - Gt * urc * MU0 * np.sum(Jt * Jt))
        ):
            print(f"tet {itet}:")
            print(f"  W0={W0}, k0={k0}, Rt={Rt}")
            print(f"  exp term={np.exp(-1j * k0 * Rt)}")
            print(f"  Gt={Gt}")
            print(f"  erc={erc}, urc={urc}")
            print(f"  Qt={Qt}")
            print(f"  Jt={Jt}, sum(Jt*Jt)={np.sum(Jt * Jt)}")
            print(f"  term1={-Gt / (erc * EPS0) * Qt * Qt}")
            print(f"  term2={-Gt * urc * MU0 * np.sum(Jt * Jt)}")
            print(
                f"  alpha_t={-Gt / (erc * EPS0) * Qt * Qt - Gt * urc * MU0 * np.sum(Jt * Jt)}"
            )
            raise ValueError(f"NaN detected at tet {itet}")

        # Face Residual computation

        all_face_coords = np.empty((3, 4 * N2D), dtype=np.float64)
        for itri in range(4):
            triid = triids[itri]
            tnodes = nodes[:, tris[:, triid]]
            n1 = tnodes[:, 0]
            n2 = tnodes[:, 1]
            n3 = tnodes[:, 2]
            all_face_coords[0, itri * N2D : (itri + 1) * N2D] = (
                DPTS_2D[1, :] * n1[0] + DPTS_2D[2, :] * n2[0] + DPTS_2D[3, :] * n3[0]
            )
            all_face_coords[1, itri * N2D : (itri + 1) * N2D] = (
                DPTS_2D[1, :] * n1[1] + DPTS_2D[2, :] * n2[1] + DPTS_2D[3, :] * n3[1]
            )
            all_face_coords[2, itri * N2D : (itri + 1) * N2D] = (
                DPTS_2D[1, :] * n1[2] + DPTS_2D[2, :] * n2[2] + DPTS_2D[3, :] * n3[2]
            )

        Qf_all = (
            erc
            * EPS0
            * compute_field(all_face_coords, vertices, Ef, l_edge_ids, l_tri_ids, dofcodes3d)
        )
        Jf_all = (
            -1
            / (1j * MU0 * W0)
            * matmul(
                uinv, compute_curl(all_face_coords, vertices, Ef, l_edge_ids, l_tri_ids, dofcodes3d)
            )
        )
        E_face_all = compute_field(all_face_coords, vertices, Ef, l_edge_ids, l_tri_ids, dofcodes3d)
        tetc = centers[:, itet].flatten()

        max_elem_size[itet] = size_max

        for iface in range(4):
            tri_index = triids[iface]

            pec_face = tet_is_pec[tri_index]

            i1, i2, i3 = tris[:, tri_index]

            slc1 = iface * N2D
            slc2 = slc1 + N2D

            normal = calc_inward_normal(
                nodes[:, i1], nodes[:, i2], nodes[:, i3], tetc
            ).astype(np.complex128)

            area = areas[triids[iface]]

            n1 = nodes[:, i1]
            n2 = nodes[:, i2]
            n3 = nodes[:, i3]
            #l1 = np.linalg.norm(n2 - n1)
            #l2 = np.linalg.norm(n3 - n1)
            #l3 = np.linalg.norm(n3 - n2)

            #Rf = np.max(np.array([l1, l2, l3]))
            #Rf = diam_circum_circle(n1, n2, n3)
            Rf = (area/np.pi)**0.5

            Rf_face_residual[iface, itet] = Rf
            areas_face_residual[iface, :, itet] = area

            adj_tets = [int(tri_to_tet[j, triids[iface]]) for j in range(2)]
            adj_tets = [num for num in adj_tets if num not in (itet, -1234)]

            if len(adj_tets) == 0:
                continue

            if pec_face is True:
                Jtan = (
                    Y0
                    * np.sqrt(1 / urc)
                    * cross_c_arry(
                        normal, -cross_c_arry(normal, E_face_all[:, slc1:slc2])
                    )
                )

                itet_adj = adj_tets[0]
                iface_adj = np.argwhere(tet_to_tri[:, itet_adj] == triids[iface])[0][0]

                Jf_face1[iface, :, :, itet] = Jtan

                adj_tets_mat[iface, itet] = itet_adj
                continue

            itet_adj = adj_tets[0]
            iface_adj = np.argwhere(tet_to_tri[:, itet_adj] == triids[iface])[0][0]

            Jf_face1[iface, :, :, itet] = cross_c_arry(normal, Jf_all[:, slc1:slc2])
            Jf_face2[iface_adj, :, :, itet_adj] = -cross_c_arry(
                normal, Jf_all[:, slc1:slc2]
            )

            Qf_face1[iface, :, itet] = dot_c_arry(normal, Qf_all[:, slc1:slc2])
            Qf_face2[iface_adj, :, itet_adj] = -dot_c_arry(normal, Qf_all[:, slc1:slc2])

            adj_tets_mat[iface, itet] = itet_adj

    # Compute 2D Gauss quadrature weight matrix
    fWs = np.empty_like(areas_face_residual, dtype=np.float64)
    for i in range(N2D):
        fWs[:, i, :] = DPTS_2D[0, i]

    # Compute the εE field difference (4, NDPTS, NTET)
    Qf_delta = Qf_face1 - Qf_face2
    Jf_delta = Jf_face1 - Jf_face2

    # Perform Gauss-Quadrature integration (4, NTET)
    Qf_int = np.sum(Qf_delta * areas_face_residual * fWs, axis=1)
    Jf_int_x = np.sum(Jf_delta[:, 0, :, :] * areas_face_residual * fWs, axis=1)
    Jf_int_y = np.sum(Jf_delta[:, 1, :, :] * areas_face_residual * fWs, axis=1)
    Jf_int_z = np.sum(Jf_delta[:, 2, :, :] * areas_face_residual * fWs, axis=1)

    Gf = (
        1j
        * W0
        * np.exp(-1j * k0 * Rf_face_residual)
        / (4 * np.pi * Rf_face_residual + DEPS)
    )
    alpha_Df = -Gf / (er * EPS0) * (Qf_int * Qf_int) - Gf * (ur * MU0) * (
        Jf_int_x * Jf_int_x + Jf_int_y * Jf_int_y + Jf_int_z * Jf_int_z
    )

    alpha_Nf = np.zeros((4, N_TETS), dtype=np.complex128)
    for it in range(N_TETS):
        for iface in range(4):
            it2 = adj_tets_mat[iface, it]
            if it2 == -1:
                continue
            alpha_Nf[iface, it] = alpha_t[it2]

    alpha_f = np.sum((alpha_t / (alpha_t + alpha_Nf + DEPS)) * alpha_Df, axis=0)

    error = (np.abs(alpha_t + alpha_f)) ** 0.5

    return error, max_elem_size


def compute_error_estimate(
    field: MWField, pec_tris: list[int]
) -> tuple[np.ndarray, np.ndarray]:
    """Top level function to compute the EM error of a field solution.

    Args:
        field (MWField): The MWField object to analyse
        pec_tris (list[int]): A list of triangles that ought to be considered PEC (non-neighbouring.)

    Returns:
        np.ndarray, np.ndarray: The error estimate in an (Ntet,) float array and the tetrahedral size value.
    """
    mesh = field.mesh

    nodes = mesh.nodes
    tris = mesh.tris
    tets = mesh.tets
    edges = mesh.edges
    centers = mesh.centers

    As = mesh.areas
    tet_to_edge = mesh.tet_to_edge
    tet_to_tri = mesh.tet_to_tri
    tri_centers = mesh.tri_centers
    tri_to_tet = mesh.tri_to_tet
    tet_to_field = field.basis.tet_to_field
    dofcodes3d = field.basis.dofcodes3d  # NEW: needed for generic DOF dispatch
    er = field._der
    ur = field._dur

    Ls = mesh.edge_lengths

    pec_tris = np.sort(np.unique(np.array(pec_tris))).astype(np.int64)

    errors = []
    for key in field._fields.keys():
        excitation = field._fields[key]
        error, sizes = compute_error_single(
            nodes,
            tets,
            tris,
            edges,
            centers,
            excitation,
            Ls,
            As,
            tet_to_edge,
            tet_to_tri,
            tri_centers,
            tri_to_tet,
            tet_to_field,
            er,
            ur,
            pec_tris,
            field.k0,
            dofcodes3d,
        )
        errors.append(error)

    error = np.max(np.array(errors), axis=0)
    return error, sizes