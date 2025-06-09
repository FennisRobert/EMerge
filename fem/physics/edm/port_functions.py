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

from ...mth.tri import ned2_tri_interp, ned2_tri_interp_curl
import numpy as np
from ...elements.nedleg2 import NedelecLegrange2
from ...mth.integrals import surface_integral


# S = E x H*
# Sx = Ey*Hz - Ez*Hy
# Sy = Ez*Hx - Ex*Hz
# Sz = Ex*Hy - Ey*Hx
def compute_avg_power_flux(field: NedelecLegrange2, mode: np.ndarray, k0: float, ur: np.ndarray, beta: float):

    Efunc = field.interpolate_Ef(mode)
    Hfunc = field.interpolate_Hf(mode, k0, ur, beta)
    nx, ny, nz = field.mesh.normals[:,0]
    def S(x,y,z):
        Ex, Ey, Ez = Efunc(x,y,z)
        Hx, Hy, Hz = Hfunc(x,y,z)
        Sx = 1/2*np.real(Ey*np.conj(Hz) - Ez*np.conj(Hy))
        Sy = 1/2*np.real(Ez*np.conj(Hx) - Ex*np.conj(Hz))
        Sz = 1/2*np.real(Ex*np.conj(Hy) - Ey*np.conj(Hx))
        return nx*Sx + ny*Sy + nz*Sz

    Ptot = surface_integral(field.mesh.nodes, field.mesh.tris, S, None, 4)
    #print(f'Total integrated power = {Ptot}W')
    return Ptot

def compute_peak_power_flux(field: NedelecLegrange2, mode: np.ndarray, k0: float, ur: np.ndarray, beta: float):

    Efunc = field.interpolate_Ef(mode)
    Hfunc = field.interpolate_Hf(mode, k0, ur, beta)
    nx, ny, nz = field.mesh.normals[:,0]
    def S(x,y,z):
        Ex, Ey, Ez = Efunc(x,y,z)
        Hx, Hy, Hz = Hfunc(x,y,z)
        Sx = np.real(Ey*np.conj(Hz) - Ez*np.conj(Hy))
        Sy = np.real(Ez*np.conj(Hx) - Ex*np.conj(Hz))
        Sz = np.real(Ex*np.conj(Hy) - Ey*np.conj(Hx))
        return nx*Sx + ny*Sy + nz*Sz

    Ptot = surface_integral(field.mesh.nodes, field.mesh.tris, S, None, 4)
    #print(f'Total integrated power = {Ptot}W')
    return Ptot