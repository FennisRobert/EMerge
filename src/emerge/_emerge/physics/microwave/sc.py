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
from ...mesh3d import SurfaceMesh
import numpy as np
from loguru import logger
from emr.sc import stratton_chu_ff


def stratton_chu(
    Ein, Hin, mesh: SurfaceMesh, theta: np.ndarray, phi: np.ndarray, k0: float
):

    Ein = np.array(Ein)
    Hin = np.array(Hin)

    Emag = np.sqrt(
        np.abs(Ein[0, :]) ** 2 + np.abs(Ein[1, :]) ** 2 + np.abs(Ein[2, :]) ** 2
    )
    Ntot = np.argwhere(Emag > 0.000001 * np.max(Emag)).shape[0]
    logger.debug(f"Percentage Included: {Ntot / Emag.shape[0] * 100:.0f}%")
    areas = mesh.areas
    edge_nodes = mesh.edge_centers

    weighted_normals = np.zeros_like(edge_nodes).astype(np.float64)

    tri_normals = mesh.normals
    tri_to_edge = mesh.tri_to_edge

    for i in range(mesh.n_tris):
        n = tri_normals[:, i]
        i1, i2, i3 = tri_to_edge[:, i]
        weighted_normals[:, i1] += n * areas[i] / 3
        weighted_normals[:, i2] += n * areas[i] / 3
        weighted_normals[:, i3] += n * areas[i] / 3

    Eout = None
    Hout = None
    tpout = np.array([theta, phi])

    Eout, Hout = stratton_chu_ff(
        Ein.astype(np.complex128),
        Hin.astype(np.complex128),
        edge_nodes.astype(np.float64),
        weighted_normals.astype(np.float64),
        tpout.astype(np.float64),
        np.float64(k0),
    )
    return Eout.astype(np.complex128), Hout.astype(np.complex128), weighted_normals
