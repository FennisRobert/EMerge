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
from ..mesh3d import SurfaceMesh
from .femdata import FEMBasis
from ..cs import CoordinateSystem
from ..const import MU0, C0
from emsutil import Saveable
from typing import Literal
from .dofsets import DoFSet
## TODO: TEMPORARY SOLUTION FIX THIS


class FieldFunctionClass:
    """This Class serves as a picklable class so that ModalPort boundary conditions
    can actually be stored with the Simulation class. Functions aren't picklable in
    Python.

    I am not happy with the existence of this class, it feels too ad-hoc but for now it
    is the simplest way. It stores all actually required information needed to do a
    surface field interpolation without needing to store the Mesh3D and SurfaceMesh class
    objects plus the NedelecLegrange2 classes with the Simulation.

    As it stands currently, only the GMSH mesh is stored plus the geometry objects. The
    mesh is reconstructed as it is deterministic.
    """

    def __init__(
        self,
        field: np.ndarray,
        cs: CoordinateSystem,
        nodes: np.ndarray,
        tris: np.ndarray,
        tri_to_field: np.ndarray,
        EH: Literal["E", "H"] = "E",
        diadic: np.ndarray | None = None,
        beta: float | None = None,
        constant: float | int | complex = 1.0,
        dofcodes: np.ndarray = None,
    ):
        self.field: np.ndarray = field
        self.cs: CoordinateSystem = cs
        self.nodes: np.ndarray = nodes
        self.tris: np.ndarray = tris
        self.tri_to_field: np.ndarray = tri_to_field
        self.eh: str = EH
        self.diadic: np.ndarray | None = diadic
        self.beta: float | None = beta
        self.constant: float = constant
        self.dofcodes: np.ndarray = dofcodes

        if EH == "H":
            if diadic is None:
                self.diadic = np.eye(3)[:, :, np.newaxis()] * np.ones(
                    (self.tris.shape[1])
                )  # type: ignore

    def flip_polarity(self) -> None:
        """Flips the polarity of the mode"""
        self.field *= -1.0

    def __call__(self, xs: np.ndarray, ys: np.ndarray, zs: np.ndarray) -> np.ndarray:
        """Computes the global Electric vector field

        Args:
            xs (np.ndarray): x-coordinates
            ys (np.ndarray): y-coordinates
            zs (np.ndarray): z-coordinates

        Returns:
            np.ndarray: Efield vector field
        """
        xl, yl, zl = self.cs.in_local_cs(xs, ys, zs)
        if self.eh == "E":
            Fxl, Fyl, Fzl = self.calcE_loc(xl, yl)
        else:
            Fxl, Fyl, Fzl = self.calcH_loc(xl, yl)
        Fx, Fy, Fz = self.cs.in_global_basis(Fxl, Fyl, Fzl)
        return np.array([Fx, Fy, Fz]) * self.constant

    def calcE_loc(
        self, xs: np.ndarray, ys: np.ndarray, usenan: bool = False
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        from ..compiled import MATHLIB

        coordinates = np.array([xs, ys])
        vals = MATHLIB.ned2_tri_interp_full(
            coordinates, self.field, self.tris, self.nodes, self.tri_to_field, self.dofcodes
        )
        if not usenan:
            vals = np.nan_to_num(vals)
        return vals

    def calcExy(self, xs: np.ndarray, ys: np.ndarray, zs: np.ndarray) -> np.ndarray:
        """Computes the global Ex, Ey, Ez vector field

        Args:
            xs (np.ndarray): x-coordinates
            ys (np.ndarray): y-coordinates
            zs (np.ndarray): z-coordinates

        Returns:
            np.ndarray: Efield vector field
        """
        xl, yl, zl = self.cs.in_local_cs(xs, ys, zs)
        Fxl, Fyl, Fzl = self.calcE_loc(xl, yl)
        Fxl = np.nan_to_num(Fxl)
        Fyl = np.nan_to_num(Fyl)
        Fzl = np.nan_to_num(Fzl)
        Fx, Fy, Fz = self.cs.in_global_basis(Fxl, Fyl, 0 * Fzl)
        return np.array([Fx, Fy, Fz]) * self.constant

    def calcHxy(self, xs: np.ndarray, ys: np.ndarray, zs: np.ndarray) -> np.ndarray:
        """Computes the global Hx, Hy, Hz vector field

        Args:
            xs (np.ndarray): x-coordinates
            ys (np.ndarray): y-coordinates
            zs (np.ndarray): z-coordinates

        Returns:
            np.ndarray: Efield vector field
        """
        xl, yl, zl = self.cs.in_local_cs(xs, ys, zs)
        Fxl, Fyl, Fzl = self.calcH_loc(xl, yl)
        Fxl = np.nan_to_num(Fxl)
        Fyl = np.nan_to_num(Fyl)
        Fzl = np.nan_to_num(Fzl)
        Fx, Fy, Fz = self.cs.in_global_basis(Fxl, Fyl, 0 * Fzl)
        return np.array([Fx, Fy, Fz]) * self.constant

    def calcHz(self, xs: np.ndarray, ys: np.ndarray, zs: np.ndarray) -> np.ndarray:
        """Computes the global Hx, Hy, Hz vector field of the out of plane component

        Args:
            xs (np.ndarray): x-coordinates
            ys (np.ndarray): y-coordinates
            zs (np.ndarray): z-coordinates

        Returns:
            np.ndarray: Efield vector field
        """
        xl, yl, zl = self.cs.in_local_cs(xs, ys, zs)
        Fxl, Fyl, Fzl = self.calcH_loc(xl, yl)
        Fxl = np.nan_to_num(Fxl)
        Fyl = np.nan_to_num(Fyl)
        Fzl = np.nan_to_num(Fzl)
        Fx, Fy, Fz = self.cs.in_global_basis(0 * Fxl, 0 * Fyl, Fzl)
        return np.array([Fx, Fy, Fz]) * self.constant

    def calcEz(self, xs: np.ndarray, ys: np.ndarray, zs: np.ndarray) -> np.ndarray:
        """Computes the global Ez vector field

        Args:
            xs (np.ndarray): x-coordinates
            ys (np.ndarray): y-coordinates
            zs (np.ndarray): z-coordinates

        Returns:
            np.ndarray: Efield vector field
        """
        xl, yl, zl = self.cs.in_local_cs(xs, ys, zs)
        Fxl, Fyl, Fzl = self.calcE_loc(xl, yl)
        Fxl = np.nan_to_num(Fxl)
        Fyl = np.nan_to_num(Fyl)
        Fzl = np.nan_to_num(Fzl)
        Fx, Fy, Fz = self.cs.in_global_basis(0 * Fxl, 0 * Fxl, Fzl)
        return np.array([Fx, Fy, Fz]) * self.constant

    def calcEzGrad(self, xs: np.ndarray, ys: np.ndarray, zs: np.ndarray) -> np.ndarray:
        """COmpute the global ∇Ezm vector field

        Args:
            xs (np.ndarray): _description_
            ys (np.ndarray): _description_
            zs (np.ndarray): _description_

        Returns:
            np.ndarray: _description_
        """
        from ..compiled import MATHLIB

        xl, yl, zl = self.cs.in_local_cs(xs, ys, zs)
        coordinates = np.array([xl, yl])
        gEzx, gEzy = MATHLIB.ned2_tri_interp_ezgrad(
            coordinates, self.field, self.tris, self.nodes, self.tri_to_field, self.dofcodes
        )
        gEzx = np.nan_to_num(gEzx)
        gEzy = np.nan_to_num(gEzy)
        Fx, Fy, Fz = self.cs.in_global_basis(gEzx, gEzy, 0 * gEzy)
        return np.array([Fx, Fy, Fz]) * self.constant

    def calc_eff_modeprofile(
        self, xs: np.ndarray, ys: np.ndarray, zs: np.ndarray
    ) -> np.ndarray:
        """Computes the global Electric mode profile

        Args:
            xs (np.ndarray): x-coordinates
            ys (np.ndarray): y-coordinates
            zs (np.ndarray): z-coordinates

        Returns:
            np.ndarray: Efield vector field
        """

        Exy = self.calcExy(xs, ys, zs)
        gEzxy = self.calcEzGrad(xs, ys, zs)
        gamma_m = 1j * self.beta
        field = gamma_m * Exy - gEzxy
        return field

    def calcH_loc(
        self, xs: np.ndarray, ys: np.ndarray, usenan: bool = False
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        from ..compiled import MATHLIB

        coordinates = np.array([xs, ys])

        vals = MATHLIB.ned2_tri_interp_curl(
            coordinates,
            self.field,
            self.tris,
            self.nodes,
            self.tri_to_field,
            self.diadic,
            self.beta,
            self.dofcodes,
        )
        if not usenan:
            vals = np.nan_to_num(vals)
        return vals


############### Nedelec2 Class


class NedelecLegrange2(FEMBasis, Saveable):
    def __init__(self, mesh: SurfaceMesh, cs: CoordinateSystem, dofset: DoFSet):

        self.mesh: SurfaceMesh = mesh

        self.cs: CoordinateSystem = cs

        self.dofset2d: DoFSet = dofset.set2d
        self.dofcodes: np.ndarray = np.array(dofset.set2d.codes, dtype=np.int64)

        ##
        nodes = self.mesh.nodes
        self.local_nodes: np.ndarray = np.array(
            self.cs.in_local_cs(nodes[0, :], nodes[1, :], nodes[2, :])
        )

        ## Counters
        self.n_nodes: int = self.mesh.n_nodes
        self.n_edges: int = self.mesh.n_edges
        self.n_tris: int = self.mesh.n_tris
        self.n_tri_dofs: int = None

        self.n_field: int = (
            self.dofset2d.n_edge_dofs * self.n_edges + self.dofset2d.n_face_dofs * self.n_tris + self.n_nodes + self.n_edges
        )

        self.n_xy: int = self.dofset2d.n_edge_dofs * self.n_edges + self.dofset2d.n_face_dofs * self.n_tris
        
        ######## MESH Derived
        Nn = self.mesh.n_nodes
        Ne = self.mesh.n_edges
        Nt = self.mesh.n_tris


        self.tri_to_field: np.ndarray = np.zeros((self.dofset2d.n_edge_dofs*3 + self.dofset2d.n_face_dofs + 6, self.n_tris), dtype=int)

        for i in range(self.dofset2d.n_edge_dofs):
            self.tri_to_field[i*3:(i+1)*3, :] = self.mesh.tri_to_edge + i*Ne
        
        N = self.dofset2d.n_edge_dofs * 3
        for i in range(self.dofset2d.n_face_dofs):
            self.tri_to_field[i + N, :] = np.arange(Nt) + Ne*self.dofset2d.n_edge_dofs + i * Nt
        
        N = self.dofset2d.n_edge_dofs * 3 + self.dofset2d.n_face_dofs

        self.tri_to_field[N:N+3, :] = self.mesh.tris + self.n_xy
        self.tri_to_field[N+3:, :] = self.mesh.tri_to_edge + (self.n_xy + Nn)

        
        self.edge_to_field: np.ndarray = np.zeros(
            (self.dofset2d.n_edge_dofs + 3, Ne), dtype=int
        ) 

        for i in range(self.dofset2d.n_edge_dofs):
            self.edge_to_field[i, :] = np.arange(Ne) + i*Ne
        
        # Edge to field indices
        # 0..n1 - Edge modes
        # n1..n1+2 - Vector modes (used for PEC 0 setting)
        # n1+2 - Last edge mode (used for PEC 0 setting)
        ne0 = self.dofset2d.n_edge_dofs
        self.edge_to_field[ne0, :] = self.mesh.edges[0,:] + self.n_xy
        self.edge_to_field[ne0+1, :] = self.mesh.edges[1,:] + self.n_xy
        self.edge_to_field[ne0+2, :] = np.arange(Ne) + self.n_xy + self.n_nodes
        
        ##
        self._field: np.ndarray = None
        self._rows: np.ndarray = None
        self._cols: np.ndarray = None

    def __call__(self, **kwargs) -> NedelecLegrange2:
        self._field = self.fielddata(**kwargs)
        return self

    def interpolate_Ef(self, field: np.ndarray) -> FieldFunctionClass:
        """Generates the Interpolation function as a function object for a given coordiante basis and origin."""
        return FieldFunctionClass(
            field, self.cs, self.local_nodes, self.mesh.tris, self.tri_to_field, "E", dofcodes=self.dofcodes
        )

    def interpolate_Hf(
        self, field: np.ndarray, k0: float, ur: np.ndarray, beta: float
    ) -> FieldFunctionClass:
        """Generates the Interpolation function as a function object for a given coordiante basis and origin."""
        from ..mth.optimized import matinv

        constant = 1j / ((k0 * C0) * MU0)
        urinv = np.zeros_like(ur)

        for i in range(ur.shape[2]):
            urinv[:, :, i] = matinv(ur[:, :, i])

        return FieldFunctionClass(
            field,
            self.cs,
            self.local_nodes,
            self.mesh.tris,
            self.tri_to_field,
            "H",
            urinv,
            beta,
            constant,
            dofcodes=self.dofcodes
        )
