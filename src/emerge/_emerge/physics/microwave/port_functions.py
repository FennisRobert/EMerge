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
from ...elements.nedleg2 import NedelecLegrange2
from typing import Callable


def compute_port_power_flux(
    nodes: np.ndarray,
    tris: np.ndarray,
    Efunc: Callable,
    Hfunc: Callable,
    inward_normal: np.ndarray,
):
    from ...mth.integrals import surface_integral

    nx, ny, nz = inward_normal

    def S(x, y, z):
        Ex, Ey, Ez = Efunc(x, y, z)
        Hx, Hy, Hz = Hfunc(x, y, z)
        Sx = 1 / 2 * np.real(Ey * np.conj(Hz) - Ez * np.conj(Hy))
        Sy = 1 / 2 * np.real(Ez * np.conj(Hx) - Ex * np.conj(Hz))
        Sz = 1 / 2 * np.real(Ex * np.conj(Hy) - Ey * np.conj(Hx))
        return nx * Sx + ny * Sy + nz * Sz

    Ptot = surface_integral(nodes, tris, S, None, 5)
    return Ptot


def compute_avg_power_flux(
    field: NedelecLegrange2,
    Efunc: Callable,
    Hfunc: Callable,
    inward_normal: np.ndarray,
):
    from ...mth.integrals import surface_integral

    nx, ny, nz = inward_normal

    def S(x, y, z):
        Ex, Ey, Ez = Efunc(x, y, z)
        Hx, Hy, Hz = Hfunc(x, y, z)
        Sx = 1 / 2 * np.real(Ey * np.conj(Hz) - Ez * np.conj(Hy))
        Sy = 1 / 2 * np.real(Ez * np.conj(Hx) - Ex * np.conj(Hz))
        Sz = 1 / 2 * np.real(Ex * np.conj(Hy) - Ey * np.conj(Hx))
        return nx * Sx + ny * Sy + nz * Sz

    Ptot = surface_integral(field.mesh.nodes, field.mesh.tris, S, None, 5)
    return Ptot
