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

from __future__ import annotations
from ...simulation_data import BaseDataset, DataContainer
from ...elements.femdata import FEMBasis
from dataclasses import dataclass, field
import numpy as np
from typing import Literal
from loguru import logger
from ...selection import FaceSelection
from ...mesh3d import Mesh3D
from ...file import Saveable
from emsutil import DataStructure
from emsutil.emdata import FieldPlotData


class HCData(Saveable):
    scalar: BaseDataset[HCScalar, HCScalarNdim]
    field: BaseDataset[HCField, None]

    def __init__(self):
        """Container for all heat conduction simulation results (scalar data, field data and solver reports)."""
        self.scalar = BaseDataset[HCScalar, HCScalarNdim](HCScalar, HCScalarNdim, True)
        self.field = BaseDataset[HCField, None](HCField, None, None)
        self.sim: DataContainer = DataContainer()

    def merge_with(self, *others: HCData) -> HCData:
        """Merges this dataset with other datasets

        Returns:
            HCData: the merged dataset
        """
        self.sim.merge_with(*[other.sim for other in others])
        self.scalar.merge_with(*[other.scalar for other in others])
        self.field.merge_with(*[other.field for other in others])
        return self

    def setreport(self, report, **vars):
        """Store the solver report for a set of simulation parameters.

        Args:
            report (SolveReport): The solver report
            **vars: The simulation parameter values the report belongs to
        """
        self.sim.new(**vars)["report"] = report


class HCScalar(Saveable):
    _fields: list[str] = []
    _copy: list[str] = []

    def __init__(self):
        """The scalar results of a single heat conduction solution."""
        pass


class HCScalarNdim(Saveable):
    _fields: list[str] = []
    _copy: list[str] = []

    def __init__(self):
        """The scalar heat conduction results of a sweep as N-dimensional arrays."""
        pass


@dataclass
class TField:
    """The temperature field and heat flux at a set of points.

    Args:
        T (np.ndarray): The temperature in Kelvin
        qx (np.ndarray): The x-component of the heat flux density
        qy (np.ndarray): The y-component of the heat flux density
        qz (np.ndarray): The z-component of the heat flux density
        x (np.ndarray): The x-coordinates
        y (np.ndarray): The y-coordinates
        z (np.ndarray): The z-coordinates
        conductivity (np.ndarray): The thermal conductivity
        structure (DataStructure, optional): The point structure (grid, surface, ...). Defaults to DataStructure.NONE.
        aux (dict[str, np.ndarray], optional): Auxiliary data. Defaults to an empty dict.
    """
    T: np.ndarray
    qx: np.ndarray
    qy: np.ndarray
    qz: np.ndarray
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    conductivity: np.ndarray
    structure: DataStructure = field(default=DataStructure.NONE)
    aux: dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def TdegC(self) -> np.ndarray:
        """The temperature in degrees Celsius."""
        return self.T - 273.15

    @property
    def F(self) -> np.ndarray:
        """The plotted scalar field (the temperature in Kelvin)."""
        return self.T

    @property
    def q(self) -> np.ndarray:
        """The (3, N) heat flux density vector."""
        return np.array([self.qx, self.qy, self.qz])

    @property
    def normq(self) -> np.ndarray:
        """The magnitude of the heat flux density."""
        return (self.qx**2 + self.qy**2 + self.qz**2) ** (0.5)

    def vector(self, field: Literal["q"]) -> FieldPlotData:

        """The data for a vector plot of the heat flux.

        Args:
            field (Literal["q"]): The vector field to plot

        Returns:
            FieldPlotData: The plot data
        """
        Fx, Fy, Fz = self.qx, self.qy, self.qz

        return FieldPlotData(
            x=self.x,
            y=self.y,
            z=self.z,
            vx=Fx,
            vy=Fy,
            vz=Fz,
            structure=self.structure,
            name="field",
        )

    def scalar(
        self, field: Literal["T", "TdegC"], metric: Literal["abs", "real"] = "real"
    ) -> FieldPlotData:
        """Returns the data X, Y, Z, Field based on the interpolation

        For animations, make sure to select the complex metric.

        Args:
            field (Literal["T", "TdegC"]): The field to plot
            metric (Literal["abs", "real"], optional): The metric to impose on the plot. Defaults to 'real'.

        Returns:
            FieldPlotData: The plot data object
        """
        fieldname = field
        if field in self.aux:
            field_arry = self.aux[field]
        else:
            field_arry = getattr(self, field)

        if metric == "abs":
            field = np.abs(field_arry)
        elif metric == "real":
            field = field_arry.real

        if "boundary" not in self.aux:
            return FieldPlotData(
                x=self.x,
                y=self.y,
                z=self.z,
                F=field_arry,
                structure=self.structure,
                name=f"{fieldname} {metric}",
            )
        else:
            return FieldPlotData(
                x=self.x,
                y=self.y,
                z=self.z,
                F=field_arry,
                tris=self.aux["tris"],
                structure=self.structure,
                boundary=True,
                name=f"{fieldname} {metric}",
            )


class HCField(Saveable):
    def __init__(self):
        """The temperature solution of a single (time or parameter) point."""
        self._ddensity: np.ndarray = None
        self._dcond_thermal: np.ndarray = []
        self._dspecific_heat: np.ndarray = []
        self.T: np.ndarray = None
        self.basis: FEMBasis = None
        self.time: float = None

    @property
    def mesh(self) -> Mesh3D:
        """The mesh of this solution."""
        return self.basis.mesh

    def interpolate(
        self, xs: np.ndarray, ys: np.ndarray, zs: np.ndarray, usenan: bool = True
    ) -> TField:
        """Interpolate the dataset in the provided coordinates

        Args:
            xs (np.ndarray): The X-coordinates
            ys (np.ndarray): The Y-coordinates
            zs (np.ndarray): The Z-coordinates
            usenan (bool, optional): Return NaN for points outside the mesh (otherwise zero). Defaults to True.

        Returns:
            TField: The temperature field at the provided coordinates
        """

        if isinstance(xs, (float, int, complex)):
            xs = np.array([xs])
            ys = np.array([ys])
            zs = np.array([zs])

        shp = xs.shape
        xf = xs.flatten()
        yf = ys.flatten()
        zf = zs.flatten()

        logger.info(f"Interpolating {xf.shape[0]} field points.")
        tet_mapping = self.basis.interpolate_index(xf, yf, zf, None)
        T = self.basis.interpolate(self.T, xf, yf, zf, tet_mapping, usenan=usenan)
        gTx, gTy, gTz = self.basis.interpolate_grad(
            self.T, xf, yf, zf, None, tet_mapping, usenan=usenan
        )
        qx = -gTx.reshape(shp) * self._dcond_thermal[0, 0, tet_mapping].reshape(shp)
        qy = -gTy.reshape(shp) * self._dcond_thermal[1, 1, tet_mapping].reshape(shp)
        qz = -gTz.reshape(shp) * self._dcond_thermal[2, 2, tet_mapping].reshape(shp)

        self.conductivity = self._dcond_thermal[0, 0, tet_mapping].reshape(shp)

        return TField(T.reshape(shp), qx, qy, qz, xs, ys, zs, self.conductivity)

    def grid(
        self,
        ds: float | None = None,
        N: int = 10_000,
        usenan: bool = True,
        x_range: tuple[float, float] | None = None,
        y_range: tuple[float, float] | None = None,
        z_range: tuple[float, float] | None = None,
    ) -> TField:
        """Interpolate a uniform grid sampled at ds

        Args:
            ds (float, optional): the sampling grid size. Defaults to None (uses N)
            N (int, optional): The approximate total number of sample points. Defaults to 10,000
            usenan (bool, optional): Return NaN for points outside the mesh (otherwise zero). Defaults to True.
            x_range (tuple[float, float] | None, optional): The x-range of the grid. Defaults to None (the mesh bounds).
            y_range (tuple[float, float] | None, optional): The y-range of the grid. Defaults to None (the mesh bounds).
            z_range (tuple[float, float] | None, optional): The z-range of the grid. Defaults to None (the mesh bounds).

        Returns:
            TField: Storage container for data
        """
        xb, yb, zb = self.basis.bounds
        if x_range is not None:
            xb = x_range
        if y_range is not None:
            yb = y_range
        if z_range is not None:
            zb = z_range
        DX = xb[1] - xb[0]
        DY = yb[1] - yb[0]
        DZ = zb[1] - zb[0]
        if ds is None:
            ds = ((DX * DY * DZ) / N) ** (1 / 3)

        xs = np.linspace(xb[0], xb[1], int(DX / ds) + 1)
        ys = np.linspace(yb[0], yb[1], int(DY / ds) + 1)
        zs = np.linspace(zb[0], zb[1], int(DZ / ds) + 1)
        X, Y, Z = np.meshgrid(xs, ys, zs)
        field = self.interpolate(X, Y, Z, usenan=usenan)
        field.structure = DataStructure.GRID3D
        return field

    def cutplane(
        self,
        ds: float,
        x: float | None = None,
        y: float | None = None,
        z: float | None = None,
        usenan: bool = True,
    ) -> TField:
        """Create a cartesian cut plane (XY, YZ or XZ) and compute the E and H-fields there

        Only one coordiante and thus cutplane may be defined. If multiple are defined only the last (x->y->z) is used.

        Args:
            ds (float): The discretization step size
            x (float | None, optional): The X-coordinate in case of a YZ-plane. Defaults to None.
            y (float | None, optional): The Y-coordinate in case of an XZ-plane. Defaults to None.
            z (float | None, optional): The Z-coordinate in case of an XY-plane. Defaults to None.
            usenan (bool, optional): Return NaN for points outside the mesh (otherwise zero). Defaults to True.

        Returns:
            TField: The resultant TField object
        """
        xb, yb, zb = self.basis.bounds
        xs = np.linspace(xb[0], xb[1], int((xb[1] - xb[0]) / ds))
        ys = np.linspace(yb[0], yb[1], int((yb[1] - yb[0]) / ds))
        zs = np.linspace(zb[0], zb[1], int((zb[1] - zb[0]) / ds))

        if x is not None:
            Y, Z = np.meshgrid(ys, zs)
            X = x * np.ones_like(Y)
        if y is not None:
            X, Z = np.meshgrid(xs, zs)
            Y = y * np.ones_like(X)
        if z is not None:
            X, Y = np.meshgrid(xs, ys)
            Z = z * np.ones_like(Y)
        field = self.interpolate(X, Y, Z, usenan=usenan)
        field.structure = DataStructure.GRID2D
        return field

    def cutplane_normal(
        self, point=(0, 0, 0), normal=(0, 0, 1), npoints: int = 300, usenan: bool = True
    ) -> TField:
        """
        Take a 2D slice of the field along an arbitrary plane.
        Args:
            point: (x0,y0,z0), a point on the plane
            normal: (nx,ny,nz), plane normal vector
            npoints: number of grid points per axis
            usenan (bool, optional): Return NaN for points outside the mesh (otherwise zero). Defaults to True.

        Returns:
            TField: The temperature field on the cut plane
        """

        n = np.array(normal, dtype=float)
        n /= np.linalg.norm(n)
        point = np.array(point)

        tmp = np.array([1, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1, 0])
        u = np.cross(n, tmp)
        u /= np.linalg.norm(u)
        v = np.cross(n, u)

        xb, yb, zb = self.basis.bounds
        nx, ny, nz = 5, 5, 5
        Xg = np.linspace(xb[0], xb[1], nx)
        Yg = np.linspace(yb[0], yb[1], ny)
        Zg = np.linspace(zb[0], zb[1], nz)
        Xg, Yg, Zg = np.meshgrid(Xg, Yg, Zg, indexing="ij")
        geometry = np.vstack([Xg.ravel(), Yg.ravel(), Zg.ravel()]).T  # Nx3

        rel_pts = geometry - point
        S = rel_pts @ u
        T = rel_pts @ v

        margin = 0.01
        s_min, s_max = S.min(), S.max()
        t_min, t_max = T.min(), T.max()
        s_bounds = (s_min - margin * (s_max - s_min), s_max + margin * (s_max - s_min))
        t_bounds = (t_min - margin * (t_max - t_min), t_max + margin * (t_max - t_min))

        S_grid = np.linspace(s_bounds[0], s_bounds[1], npoints)
        T_grid = np.linspace(t_bounds[0], t_bounds[1], npoints)
        S_mesh, T_mesh = np.meshgrid(S_grid, T_grid)

        X = point[0] + S_mesh * u[0] + T_mesh * v[0]
        Y = point[1] + S_mesh * u[1] + T_mesh * v[1]
        Z = point[2] + S_mesh * u[2] + T_mesh * v[2]

        field = self.interpolate(X, Y, Z, usenan=usenan)
        field.structure = DataStructure.GRID2D
        return field

    def boundary(self, selection: FaceSelection) -> TField:
        """Interpolate the field on the node coordinates of the surface.

        Args:
            selection (FaceSelection): The surface

        Returns:
            TField: The temperature field on the surface nodes
        """
        tris = self.mesh.tris[:, self.mesh.get_triangles(selection.tags)]

        # Get unique node indices and remap triangles to dense 0..N-1
        unique_nodes, inverse = np.unique(tris.ravel(), return_inverse=True)
        tris_dense = inverse.reshape(tris.shape)

        xs = self.mesh.nodes[0, unique_nodes]
        ys = self.mesh.nodes[1, unique_nodes]
        zs = self.mesh.nodes[2, unique_nodes]
        T_surf = self.T[unique_nodes]

        tfield = TField(
            T_surf,
            0 * T_surf,
            0 * T_surf,
            0 * T_surf,
            xs,
            ys,
            zs,
            0 * xs,
            structure=DataStructure.TRISURF,
        )
        tfield.aux["tris"] = tris_dense
        tfield.aux["boundary"] = True
        return tfield
