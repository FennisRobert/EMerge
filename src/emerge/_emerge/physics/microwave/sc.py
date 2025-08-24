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

from ...mesh3d import SurfaceMesh
import numpy as np
from loguru import logger
from numba import c16, njit, prange, f8
from numba.types import Tuple as TupleType  # ty: ignore
from ...const import Z0

LR = 0.001

@njit(
    TupleType((c16[:, :], c16[:, :]))(
        c16[:, :],
        c16[:, :],
        f8[:, :],
        f8[:, :],
        f8[:, :],
        f8,
    ),
    parallel=True,
    fastmath=True,
    cache=True,
    nogil=True,
)
def stratton_chu_ff(Ein, Hin, vis, wns, tpout, k0):
    
    Ex = Ein[0, :].flatten()
    Ey = Ein[1, :].flatten()
    Ez = Ein[2, :].flatten()
    Hx = Hin[0, :].flatten()
    Hy = Hin[1, :].flatten()
    Hz = Hin[2, :].flatten()
    vx = vis[0, :].flatten()
    vy = vis[1, :].flatten()
    vz = vis[2, :].flatten()
    nx = wns[0, :].flatten()
    ny = wns[1, :].flatten()
    nz = wns[2, :].flatten()
    
    Emag = np.sqrt(np.abs(Ex)**2 + np.abs(Ey)**2 + np.abs(Ez)**2)
    
    Elevel = np.max(Emag) * LR
    ids = np.argwhere(Emag > Elevel)
    Nids = ids.shape[0]
    #iadd = NT // Nids
    Ex = Ex[Emag > Elevel]
    Ey = Ey[Emag > Elevel]
    Ez = Ez[Emag > Elevel]
    Hx = Hx[Emag > Elevel]
    Hy = Hy[Emag > Elevel]
    Hz = Hz[Emag > Elevel]
    vx = vx[Emag > Elevel]
    vy = vy[Emag > Elevel]
    vz = vz[Emag > Elevel]
    nx = nx[Emag > Elevel]
    ny = ny[Emag > Elevel]
    nz = nz[Emag > Elevel]

    thout = tpout[0, :]
    phout = tpout[1, :]

    rx = np.sin(thout) * np.cos(phout)
    ry = np.sin(thout) * np.sin(phout)
    rz = np.cos(thout)

    kx = k0 * rx
    ky = k0 * ry
    kz = k0 * rz

    N = tpout.shape[1]

    Eout = np.zeros((3, N)).astype(np.complex128)
    Hout = np.zeros((3, N)).astype(np.complex128)

    Eoutx = np.zeros((N,)).astype(np.complex128)
    Eouty = np.zeros((N,)).astype(np.complex128)
    Eoutz = np.zeros((N,)).astype(np.complex128)

    Q = np.complex128(-1j * k0 / (4 * np.pi))
    ii = np.complex128(1j)

    NxHx = ny * Hz - nz * Hy
    NxHy = nz * Hx - nx * Hz
    NxHz = nx * Hy - ny * Hx

    NxEx = ny * Ez - nz * Ey
    NxEy = nz * Ex - nx * Ez
    NxEz = nx * Ey - ny * Ex

    for j in prange(Nids): # ty: ignore
        xi = vx[j]
        yi = vy[j]
        zi = vz[j]
        G = np.exp(ii * (kx * xi + ky * yi + kz * zi))

        RxNxHx = ry * NxHz[j] - rz * NxHy[j]
        RxNxHy = rz * NxHx[j] - rx * NxHz[j]
        RxNxHz = rx * NxHy[j] - ry * NxHx[j]

        ie1x = (NxEx[j] - Z0 * RxNxHx) * G
        ie1y = (NxEy[j] - Z0 * RxNxHy) * G
        ie1z = (NxEz[j] - Z0 * RxNxHz) * G

        Eoutx += Q * (ry * ie1z - rz * ie1y)
        Eouty += Q * (rz * ie1x - rx * ie1z)
        Eoutz += Q * (rx * ie1y - ry * ie1x)

        # ii += iadd
    Eout[0, :] = Eoutx
    Eout[1, :] = Eouty
    Eout[2, :] = Eoutz

    Hout[0, :] = (ry * Eoutz - rz * Eouty) / Z0
    Hout[1, :] = (rz * Eoutx - rx * Eoutz) / Z0
    Hout[2, :] = (rx * Eouty - ry * Eoutx) / Z0

    return Eout, Hout

def stratton_chu(Ein, Hin, mesh: SurfaceMesh, theta: np.ndarray, phi: np.ndarray, k0: float):

    Ein = np.array(Ein)
    Hin = np.array(Hin)

    Emag = np.sqrt(np.abs(Ein[0,:])**2 + np.abs(Ein[1,:])**2 + np.abs(Ein[2,:])**2)
    Ntot = np.argwhere(Emag>0.000001*np.max(Emag)).shape[0]
    logger.debug(f'Percentage Included: {Ntot/Emag.shape[0]*100:.0f}%')
    areas = mesh.areas
    vis = mesh.edge_centers

    wns = np.zeros_like(vis).astype(np.float64)

    tri_normals = mesh.normals
    tri_ids = mesh.tri_to_edge

    for i in range(mesh.n_tris):
        n = tri_normals[:,i]
        i1, i2, i3 = tri_ids[:,i]
        wns[:,i1] += n*areas[i]/3
        wns[:,i2] += n*areas[i]/3
        wns[:,i3] += n*areas[i]/3
    
    Eout = None
    Hout = None
    tpout = np.array([theta, phi])

    Eout, Hout = stratton_chu_ff(
        Ein.astype(np.complex128),
        Hin.astype(np.complex128),
        vis.astype(np.float64),
        wns.astype(np.float64),
        tpout.astype(np.float64),
        np.float64(k0),
    )
    return Eout.astype(np.complex128), Hout.astype(np.complex128)
