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
from ..mesh3d import Mesh3D
import numpy as np
from typing import Callable
from emsutil import Saveable
from scipy.sparse import csc_matrix  # type: ignore

class FEMBasis(Saveable):
    def __init__(self, mesh: Mesh3D):
        self.mesh: Mesh3D = mesh
        self.n_edges: int = self.mesh.n_edges
        self.n_tris: int = self.mesh.n_tris
        self.n_tets: int = self.mesh.n_tets

        self.n_tet_dofs: int = -1
        self.n_tri_dofs: int = -1

        self.n_field: int = 2 * self.n_edges + 2 * self.n_tris

        self.tet_to_field: np.ndarray = np.array([])

        self.edge_to_field: np.ndarray = np.array([])

        self.tri_to_field: np.ndarray = np.array([])

        self._rows: np.ndarray = np.array([])
        self._cols: np.ndarray = np.array([])
        self._cached_csc_ids = None

    def empty_tet_matrix(self) -> np.ndarray:
        nnz = self.n_tets * self.n_tet_dofs**2
        matrix = np.empty((nnz,), dtype=np.complex128)
        return matrix

    def empty_tri_matrix(self) -> np.ndarray:
        nnz = self.n_tris * self.n_tri_dofs**2
        matrix = np.zeros((nnz,), dtype=np.complex128)
        return matrix

    def empty_tet_rowcol(self) -> tuple[np.ndarray, np.ndarray]:
        N = self.n_tet_dofs
        N2 = N**2
        nnz = self.n_tets * N2
        rows = np.empty(nnz, dtype=np.int64)
        cols = np.empty(nnz, dtype=np.int64)

        for itet in range(self.n_tets):
            p = itet * N2

            indices = self.tet_to_field[:, itet]
            for ii in range(self.n_tet_dofs):
                rows[p + N * ii : p + N * (ii + 1)] = indices[ii]
                cols[p + ii : p + N2 : N] = indices[ii]
        self._rows = rows
        self._cols = cols
        return rows, cols

    def empty_tri_rowcol(
        self, other_side: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        N = self.n_tri_dofs
        N2 = N**2
        nnz = self.n_tris * N2
        rows = np.empty(nnz, dtype=np.int64)
        cols = np.empty(nnz, dtype=np.int64)

        t2f = self.tri_to_field
        if other_side:
            t2f = self.tri_to_field_os

        for itri in range(self.n_tris):
            p = itri * N2
            indices = t2f[:, itri]
            for ii in range(N):
                rows[p + N * ii : p + N * (ii + 1)] = indices[ii]
                cols[p + ii : p + N2 : N] = indices[ii]
        return rows, cols

    def tetslice(self, itet: int) -> slice:
        N = self.n_tet_dofs**2
        return slice(itet * N, (itet + 1) * N)

    def trislice(self, itri: int) -> slice:
        N = self.n_tri_dofs**2
        return slice(itri * N, (itri + 1) * N)

    def generate_csc(
        self, data: np.ndarray, rowcol: tuple[np.ndarray, np.ndarray] | None = None
    ):
        if rowcol is None:
            rows, cols = self._rows, self._cols
        else:
            rows, cols = rowcol

        if self._cached_csc_ids is None:
            self._cached_csc_ids = np.argwhere(data != 0)[:, 0]

        return csc_matrix(
            (data[self._cached_csc_ids], (rows[self._cached_csc_ids], cols[self._cached_csc_ids])),
            shape=(self.n_field, self.n_field),
        )

    ############################################################
    #                         INTERPOLATORS                    #
    ############################################################

    def interpolate_Ef(
        self,
        field: np.ndarray,
        basis: np.ndarray | None = None,
        origin: np.ndarray | None = None,
        tetids: np.ndarray | None = None,
    ) -> Callable:
        """Generates the Interpolation function as a function object for a given coordiante basis and origin."""
        from ..mth.optimized import matmul

        if basis is None:
            basis = np.eye(3)

        if origin is None:
            origin = np.zeros(3)

        ibasis = np.linalg.pinv(basis)

        def func(xs: np.ndarray, ys: np.ndarray, zs: np.ndarray) -> np.ndarray:
            xyz = np.array([xs, ys, zs]) + origin[:, np.newaxis]
            xyzg = matmul(basis, xyz)
            tet_mapping = self.interpolate_index(
                xyzg[0, :], xyzg[1, :], xyzg[2, :], tetids, usenan=False
            )
            return matmul(
                ibasis,
                np.array(
                    self.interpolate(
                        field,
                        xyzg[0, :],
                        xyzg[1, :],
                        xyzg[2, :],
                        tet_mapping,
                        tetids,
                        usenan=False,
                    )
                ),
            )

        return func

    def interpolate(
        self,
        field: np.ndarray,
        xs: np.ndarray,
        ys: np.ndarray,
        zs: np.ndarray,
        tet_mapping: np.ndarray,
        tetids: np.ndarray | None = None,
        usenan: bool = True,
        dofcodes: np.ndarray = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        raise NotImplementedError()

    def interpolate_grad(
        self,
        field: np.ndarray,
        xs: np.ndarray,
        ys: np.ndarray,
        zs: np.ndarray,
        tet_mapping: np.ndarray,
        tetids: np.ndarray | None = None,
        usenan: bool = True,
        dofcodes: np.ndarray = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        raise NotImplementedError()

    def interpolate_curl(
        self,
        field: np.ndarray,
        xs: np.ndarray,
        ys: np.ndarray,
        zs: np.ndarray,
        constants: np.ndarray,
        tet_mapping: np.ndarray,
        tetids: np.ndarray | None = None,
        usenan: bool = True,
        dofcodes: np.ndarray = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Interpolates the curl of the field at the given points.
        """
        raise NotImplementedError()

    def interpolate_error(
        self,
        field: np.ndarray,
        xs: np.ndarray,
        ys: np.ndarray,
        zs: np.ndarray,
        tet_mapping: np.ndarray,
        tetids: np.ndarray | None = None,
        cs: tuple[float, float] = (1.0, 1.0),
        dofcodes: np.ndarray = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        raise NotImplementedError()

    ####### INDEX MAPPINGS
    def interpolate_index(
        self,
        xs: np.ndarray,
        ys: np.ndarray,
        zs: np.ndarray,
        tetids: np.ndarray | None = None,
        usenan: bool = True,
    ) -> np.ndarray:
        raise NotImplementedError()

    # ##### CUTS

    @property
    def bounds(
        self,
    ) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
        xmin = np.min(self.mesh.nodes[0, :])
        xmax = np.max(self.mesh.nodes[0, :])
        ymin = np.min(self.mesh.nodes[1, :])
        ymax = np.max(self.mesh.nodes[1, :])
        zmin = np.min(self.mesh.nodes[2, :])
        zmax = np.max(self.mesh.nodes[2, :])
        return (xmin, xmax), (ymin, ymax), (zmin, zmax)
