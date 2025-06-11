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
from typing import Callable
from numba import njit, f8, i8, types, c16
from .optimized import gaus_quad_tri, generate_int_points_tri, calc_area


@njit(c16(f8[:,:], i8[:,:], c16[:], f8[:,:], c16[:,:]), cache=True, nogil=True)
def _fast_integral_c(nodes, triangles, constants, DPTs, field_values):
    tot = np.complex128(0.0)

    for it in range(triangles.shape[1]):
        vertex_ids = triangles[:, it]
        v1 = nodes[:,vertex_ids[0]]
        v2 = nodes[:,vertex_ids[1]]
        v3 = nodes[:,vertex_ids[2]]
        A = calc_area(v1, v2, v3)
        field = np.sum(DPTs[0,:]*field_values[:,it])
        tot = tot + constants[it] * field * A
    return tot

@njit(f8(f8[:,:], i8[:,:], f8[:], f8[:,:], f8[:,:]), cache=True, nogil=True)
def _fast_integral_f(nodes, triangles, constants, DPTs, field_values):
    tot = np.float64(0.0)
    for it in range(triangles.shape[1]):
        vertex_ids = triangles[:, it]
        v1 = nodes[:,vertex_ids[0]]
        v2 = nodes[:,vertex_ids[1]]
        v3 = nodes[:,vertex_ids[2]]
        A = calc_area(v1, v2, v3)
        field = np.sum(DPTs[0,:]*field_values[:,it])
        tot = tot + constants[it] * field * A
    return tot

def surface_integral(nodes: np.ndarray, 
                     triangles: np.ndarray, 
                     function: Callable, 
                     constants: np.ndarray = None,
                     ndpts: int = 4):

    if constants is None:
        constants = np.ones(triangles.shape[1])
        
    DPTs = gaus_quad_tri(ndpts)
    xall_flat, yall_flat, zall_flat, shape = generate_int_points_tri(nodes, triangles, DPTs)

    fvals = function(xall_flat, yall_flat, zall_flat)

    fA = fvals.reshape(shape)

    if np.iscomplexobj(fA) or np.iscomplexobj(constants):
        return _fast_integral_c(nodes, triangles, constants.astype(np.complex128), DPTs, fA.astype(np.complex128))
    else:
        return _fast_integral_f(nodes, triangles, constants, DPTs, fA)
    