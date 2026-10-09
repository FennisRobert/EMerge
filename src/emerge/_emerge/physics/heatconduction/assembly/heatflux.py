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
from ....elements.leg2 import Legrange2
from numba import f8, i8, njit, prange
from .grad import tet_coefficients
from ....mth.optimized import calc_area
from typing import Callable

############################################################
#               TRIANGLE AND VOLUME GQ POINTS              #
############################################################
# fmt: off
TRI_DPTS = np.array([
    [-0.28125,     0.26041667,  0.26041667,  0.26041667],
    [ 1.0/3.0,     0.6,         0.2,         0.2        ],
    [ 1.0/3.0,     0.2,         0.6,         0.2        ],
    [ 1.0/3.0,     0.2,         0.2,         0.6        ],
], dtype=np.float64)

DPTS = np.array([
    [-0.078933,    0.04573333,  0.04573333,  0.04573333,  0.04573333,
      0.14933333,  0.14933333,  0.14933333,  0.14933333,  0.14933333,  0.14933333],
    [ 0.25,        0.78571429,  0.07142857,  0.07142857,  0.07142857,
      0.39940358,  0.39940358,  0.39940358,  0.10059642,  0.10059642,  0.10059642],
    [ 0.25,        0.07142857,  0.07142857,  0.07142857,  0.78571429,
      0.10059642,  0.10059642,  0.39940358,  0.39940358,  0.39940358,  0.10059642],
    [ 0.25,        0.07142857,  0.07142857,  0.78571429,  0.07142857,
      0.39940358,  0.10059642,  0.10059642,  0.39940358,  0.10059642,  0.39940358],
    [ 0.25,        0.07142857,  0.78571429,  0.07142857,  0.07142857,
      0.10059642,  0.39940358,  0.10059642,  0.10059642,  0.39940358,  0.39940358],
], dtype=np.float64)
# fmt: on


############################################################
#                   SURFACE HEAT FLUX BC                  #
############################################################


@njit(f8[:](), cache=True, nogil=True)
def _assemble_force_vector():
    weights = TRI_DPTS[0, :]
    nq = weights.shape[0]

    local_edge_map = np.array([[0, 1, 0], [1, 2, 2]], dtype=np.int64)
    f_local = np.zeros(6, dtype=np.float64)

    for iq in range(nq):
        L1 = TRI_DPTS[1, iq]
        L2 = TRI_DPTS[2, iq]
        L3 = TRI_DPTS[3, iq]
        w = weights[iq]

        Ls = np.empty(3, dtype=np.float64)
        Ls[0] = L1
        Ls[1] = L2
        Ls[2] = L3

        for iv in range(3):
            f_local[iv] += w * Ls[iv] * (2.0 * Ls[iv] - 1.0)

        for ie in range(3):
            li = Ls[local_edge_map[0, ie]]
            lj = Ls[local_edge_map[1, ie]]
            f_local[3 + ie] += w * 4.0 * li * lj
    return f_local


@njit(
    f8[:](f8[:, :], i8[:, :], i8[:, :], i8[:, :], f8, i8),
    cache=True,
    nogil=True,
    parallel=True,
)
def _surface_flux_builder(nodes, tris, tri_to_field, tri_ids_2d, q_flux, n_field):
    n_sel = tri_ids_2d.shape[1]

    f_global = np.zeros((n_field,), dtype=np.float64)

    # Serial loop: neighbouring triangles share DOFs, so a prange scatter-add races.
    for idx in range(n_sel):
        itri = tri_ids_2d[0, idx]
        iv1 = tris[0, itri]
        iv2 = tris[1, itri]
        iv3 = tris[2, itri]
        A = calc_area(nodes[:, iv1], nodes[:, iv2], nodes[:, iv3])
        field_ids = tri_to_field[:, itri]
        f_local = _assemble_force_vector()
        scale = q_flux * 2.0 * A

        for i in range(6):
            f_global[field_ids[i]] += f_local[i] * scale

    return f_global


############################################################
#                       SURFACE FLUX                      #
############################################################


def assemble_surface_flux(
    field: Legrange2, face_tags: list[int], q_flux: float
) -> np.ndarray:
    """Assemble surface Neumann heat flux into global load vector.

    Args:
        field (Legrange2): Field object
        face_tags (list[int]): list/array of GMSH face tags for this BC
        q_flux (float): float, constant heat flux [W/m^2], positive = into domain

    Returns:
        f_global: (n_field,) load vector contribution
    """
    mesh = field.mesh
    tri_ids = mesh.get_triangles(face_tags)

    tri_ids_2d = tri_ids.reshape(1, -1).astype(np.int64)

    return _surface_flux_builder(
        mesh.nodes,
        mesh.tris,
        field.tri_to_field,
        tri_ids_2d,
        float(q_flux),
        field.n_field,
    )


@njit(
    f8[:](f8[:, :], i8[:, :], f8[:, :], i8),
    cache=True,
    nogil=True,
    parallel=True,
)
def _volume_source_builder(nodes, tet_to_field, q_at_quad, n_field):
    """Parallel assembly of volumetric heat source load vector.

    Args:
        nodes: (3, n_nodes)
        tets: (4, n_tets_total)
        edges: (2, n_edges)
        tet_to_field: (10, n_tets_total)
        q_at_quad: (n_selected, n_quad) heat source values at quadrature points
        n_field: total number of DOFs

    Returns:
        f_global: (n_field,) load vector
    """
    n_sel = q_at_quad.shape[0]
    nq = DPTS.shape[1]
    weights = DPTS[0, :]

    f_global = np.zeros((n_field), dtype=np.float64)
    local_edge_map = np.array([[0, 0, 0, 1, 1, 2], [1, 2, 3, 2, 3, 3]])

    for idx in range(n_sel):
        # Tet vertices from node coordinates (vertex field IDs = node IDs for Lagrange)
        ids = tet_to_field[:4, idx]

        txs = nodes[0, ids]
        tys = nodes[1, ids]
        tzs = nodes[2, ids]

        aas, bbs, ccs, dds, V = tet_coefficients(txs, tys, tzs)
        inv6V = 1.0 / (6.0 * V)
        a = aas * inv6V
        b = bbs * inv6V
        c = ccs * inv6V
        d = dds * inv6V

        # Gauss Quadrature Barycentric Coordinates
        L1_pts = DPTS[1, :]
        L2_pts = DPTS[2, :]
        L3_pts = DPTS[3, :]
        L4_pts = DPTS[4, :]

        field_ids = tet_to_field[:, idx]

        f_local = np.zeros(10, dtype=np.float64)

        for iq in range(nq):
            x = (
                txs[0] * L1_pts[iq]
                + txs[1] * L2_pts[iq]
                + txs[2] * L3_pts[iq]
                + txs[3] * L4_pts[iq]
            )
            y = (
                tys[0] * L1_pts[iq]
                + tys[1] * L2_pts[iq]
                + tys[2] * L3_pts[iq]
                + tys[3] * L4_pts[iq]
            )
            z = (
                tzs[0] * L1_pts[iq]
                + tzs[1] * L2_pts[iq]
                + tzs[2] * L3_pts[iq]
                + tzs[3] * L4_pts[iq]
            )

            Ls = np.empty(4, dtype=np.float64)
            for iv in range(4):
                Ls[iv] = a[iv] + b[iv] * x + c[iv] * y + d[iv] * z

            wq = weights[iq] * q_at_quad[idx, iq]

            for iv in range(4):
                f_local[iv] += wq * Ls[iv] * (2.0 * Ls[iv] - 1.0)

            for ie in range(6):
                li = Ls[local_edge_map[0, ie]]
                lj = Ls[local_edge_map[1, ie]]
                f_local[4 + ie] += wq * 4.0 * li * lj

        for i in range(10):
            f_global[field_ids[i]] += f_local[i] * V

    return f_global


def assemble_volume_source(
    field: Legrange2, tet_ids: list[int], q_func: Callable
) -> np.ndarray:
    """Assemble volumetric heat source into global load vector.

    Args:
        field: Legrange2 field
        tet_ids: array of tet indices where source is active
        q_func: callable(xs, ys, zs) -> array of q_V values [W/m³]

    Returns:
        f_global: (n_field,) load vector contribution
    """
    mesh = field.mesh
    tet_ids = np.asarray(tet_ids, dtype=np.int64)
    n_sel = len(tet_ids)
    nq = DPTS.shape[1]

    L1 = DPTS[1, :]
    L2 = DPTS[2, :]
    L3 = DPTS[3, :]
    L4 = DPTS[4, :]

    verts = mesh.tets[:, tet_ids]  # (4, n_sel)

    x0, x1, x2, x3 = mesh.nodes[0, verts]
    y0, y1, y2, y3 = mesh.nodes[1, verts]
    z0, z1, z2, z3 = mesh.nodes[2, verts]

    # Construct gauss quadrature integration points
    all_xs = x0[:, None] * L1 + x1[:, None] * L2 + x2[:, None] * L3 + x3[:, None] * L4
    all_ys = y0[:, None] * L1 + y1[:, None] * L2 + y2[:, None] * L3 + y3[:, None] * L4
    all_zs = z0[:, None] * L1 + z1[:, None] * L2 + z2[:, None] * L3 + z3[:, None] * L4

    # Call the volumetric heat flux function q_func
    q_at_quad = q_func(all_xs.ravel(), all_ys.ravel(), all_zs.ravel()).reshape(
        n_sel, nq
    )

    tet_to_field_sel = field.tet_to_field[:, tet_ids].copy()

    return _volume_source_builder(
        mesh.nodes,
        tet_to_field_sel,
        q_at_quad,
        field.n_field,
    )
