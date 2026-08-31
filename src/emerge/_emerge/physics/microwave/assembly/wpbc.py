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
from typing import Callable
from ....elements import Nedelec2
from ....const import MU0, C0
from .robinbc import assemble_robin_bc_bvec


############################################################
#           DENSE WAVE PORT BOUNDARY CONDITION             #
############################################################


def assemble_Bmatrix_entries(
    G_global: np.ndarray,
    constant: complex,
    active_dof_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Builds the dense rank-1 boundary matrix update constant * G Gᵀ.

    Only the DoFs with a non-zero mode overlap (active_dof_ids) are expanded,
    so the result can be scattered directly into a sparse COO matrix.
    """
    G_active = G_global[active_dof_ids]

    Bdense = constant * np.outer(G_active, G_active)

    cols_grid, rows_grid = np.meshgrid(active_dof_ids, active_dof_ids)

    return Bdense.ravel(), rows_grid.ravel(), cols_grid.ravel()


def assemble_wpbc(
    field: Nedelec2,
    surf_triangle_indices: np.ndarray,
    mprof: Callable,
    mode_xy: Callable,
    kappa_m: complex,
    gamma_m: complex,
    k0: float,
    port_normal: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Assembles the dense (full modal) Wave Port Boundary Condition.

    The port mode overlap vector G = ∫ (γ_m e_tm - ∇e_zm) · N dS is computed by
    reusing the Robin BC force-vector assembly (identical Nedelec2 overlap
    integral), once for the effective mode profile `mprof` and once for the
    transverse mode field `mode_xy`. G is then used to build the rank-1 system
    matrix contribution 1/(j w0 mu0 kappa_m) * G Gᵀ and the excitation vector
    -2 gamma_m * G_xy.

    Args:
        field (Nedelec2): The Nedelec2 field object.
        surf_triangle_indices (np.ndarray): Indices of the port surface triangles.
        mprof (Callable): The effective mode profile function γ_m e_tm - ∇e_zm.
        mode_xy (Callable): The transverse mode field function e_tm.
        kappa_m (complex): The port mode kappa coefficient.
        gamma_m (complex): The port mode propagation constant (j*beta).
        k0 (float): The free space wavenumber.
        port_normal (np.ndarray): The port face normal (kept for interface
            compatibility; the overlap integral is orientation independent).

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: The dense B-matrix
        data, rows and cols (COO format) and the excitation vector.
    """
    G = assemble_robin_bc_bvec(field, surf_triangle_indices, mprof)
    G_xy = assemble_robin_bc_bvec(field, surf_triangle_indices, mode_xy)

    ids = np.argwhere(G != 0).ravel()
    w0 = C0 * k0
    Bvec, rows, cols = assemble_Bmatrix_entries(G, -1.0 / (1j * w0 * MU0 * kappa_m), ids)
    return Bvec, rows, cols, - 2 * gamma_m * G_xy
