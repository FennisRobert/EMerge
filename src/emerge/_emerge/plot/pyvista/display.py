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
from ...mesh3d import Mesh3D
from ...simstate import SimState
from ...geometry import GeoObject
from ...selection import (
    FaceSelection,
    DomainSelection,
    EdgeSelection,
    Selection,
    encode_data,
)
from .... import __version__
from ...physics.microwave.bcs import PortBC, ModalPort
from ...physics.microwave.bcs.background_field import BackgroundField
from ...cs import Anchor
from ...coord import Line
from pathlib import Path
from importlib.resources import files

from emsutil.pyvista import EMergeDisplay, setdefault, cmap_names, _AnimObject
from emsutil import themes
from emsutil.emdata import EHFieldFF

import numpy as np
from typing import Iterable, Literal
from loguru import logger
import pyvista as pv


def _min_distance(xs, ys, zs):
    """
    Compute the minimum Euclidean distance between any two points
    defined by the 1D arrays xs, ys, zs.

    Parameters:
        xs (np.ndarray): x-coordinates of the points
        ys (np.ndarray): y-coordinates of the points
        zs (np.ndarray): z-coordinates of the points

    Returns:
        float: The minimum Euclidean distance between any two points
    """
    # Stack the coordinates into a (N, 3) array
    points = np.stack((xs, ys, zs), axis=-1)

    # Compute pairwise squared distances using broadcasting
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists_squared = np.sum(diff**2, axis=-1)

    # Set diagonal to infinity to ignore zero distances to self
    np.fill_diagonal(dists_squared, np.inf)

    # Get the minimum distance
    min_dist = np.sqrt(np.min(dists_squared))
    return min_dist


def _select(obj: GeoObject | Selection) -> Selection:
    if isinstance(obj, GeoObject):
        return obj.selection
    return obj


def _merge(lst: Iterable[GeoObject | Selection]) -> Selection:
    selections = [_select(item) for item in lst]
    dim = selections[0].dim
    all_tags = []
    for item in lst:
        all_tags.extend(_select(item).tags)

    if dim == 1:
        return EdgeSelection(all_tags)
    elif dim == 2:
        return FaceSelection(all_tags)
    elif dim == 3:
        return DomainSelection(all_tags)
    else:
        return Selection(all_tags)


def _print_coords(x1, y1, x2, y2, z):
    # ALLOWED PRINT
    print(f"lp = pcb.lumped_port_pts(({x1:.6f},{y1:.6f}),({x2:.6f},{y2:.6f}),{z:.6f})")


class PVDisplay(EMergeDisplay):
    def __post_init__(self, state: SimState):
        self._state: SimState = state
        self._selector._set_encoder_function(encode_data)
        self._plot.add_key_event("l", self.activate_line_selector)
        self._plot.add_key_event("n", self.activate_point_selector)
        self._selectable_edges = []
        self._selectable_faces = []
        self._selectable_nodes = []
        self._selectable_volumes = []

        # New Selector
        self._selected_points = [(0, 0, 0), (0, 0, 0)]
        self._selectable_points = []

        self.set_theme(themes.EMV3)

    def _add_selectable_edges(self) -> None:
        self._clear_selectable_objects()
        mesh = self._state.mesh
        self._plot._render = False
        for edge_id, edge_tags in self._state.mesh.etag_to_edge.items():
            # No tags in the edge, abort
            if len(edge_tags) < 1:
                continue
            vertex_ids = mesh.edges[:, edge_tags].ravel()
            edge_nodes = mesh.nodes[:, vertex_ids]
            center = np.mean(edge_nodes, axis=1)
            # Points are not on a straight line, not printable like this
            if (
                not np.linalg.svd(edge_nodes - edge_nodes.mean(axis=1, keepdims=True))[
                    1
                ][1]
                < 1e-6
            ):
                continue
            vs = np.where(np.bincount(vertex_ids) == 1)[0]
            # No unique start and end point (loop line)
            if len(vs) != 2:
                logger.warning(f"Skipping edge {edge_id} with no start and end point.")
                continue

            n1 = mesh.nodes[:, vs[0]]
            n2 = mesh.nodes[:, vs[1]]
            line = pv.Line(n1, n2)
            line_mesh = self._plot.add_mesh(
                line, color="black", line_width=6, opacity=1.0, render=False
            )
            line_mesh._seldata = (
                tuple([float(c) for c in n1]),
                tuple([float(c) for c in n2]),
                edge_id,
                center,
            )
            self._selectable_edges.append(line_mesh)
        self._plot._render = True
        self._plot.render()

    def _get_emerge_path(self, filename: str) -> str:
        """Generates a filename for the EMerge package directory in the PyVista folder

        Args:
            filename (str): _description_

        Returns:
            str: _description_
        """
        return str(Path(files("emerge")) / "_emerge" / "plot" / "pyvista" / filename)

    def show(self, screenshot: str | None = None, off_screen: bool = False):
        """Shows the Pyvista display."""
        logo_path = self._get_emerge_path("EMS_small.png")
        self._plot.add_logo_widget(
            logo_path, position=(0.87, 0.87), size=(0.10, 0.10), opacity=1.0
        )
        text_actor = self._plot.add_text(
            f"EMerge Version: {__version__}",
            color="white",
            font_size=12,
            position="lower_right",
        )
        super().show(screenshot, off_screen)

    def _add_selectable_points(self) -> None:
        self._clear_selectable_objects()
        mesh = self._state.mesh
        self._plot._render = False

        pointarray = mesh.nodes[:, mesh.geonodes]
        pointcloud = pv.PolyData(pointarray.T)

        pointmesh = self._plot.add_mesh(
            pointcloud,
            render_points_as_spheres=True,
            point_size=10,
            color="red",
            pickable=True,
        )
        self._selectable_nodes.append(pointmesh)

    def _clear_selectable_objects(self) -> None:
        self._clear_highlight()
        self._ruler.turn_off()
        for obj in (
            self._selectable_edges
            + self._selectable_volumes
            + self._selectable_nodes
            + self._selectable_faces
        ):
            self._plot.remove_actor(obj)

    def activate_point_selector(self):
        self._add_selectable_points()
        self._ruler.turn_on()

    def activate_line_selector(self):

        def callback(obj):
            if "_seldata" in obj.__dict__:
                n1, n2, edge_id, center = obj._seldata
                ds = n2[0] - n1[0], n2[1] - n1[1], n2[2] - n1[2]
                print(f"Edge tag [{edge_id}] data:")
                print(f"    vertex[1] = {n1}")
                print(f"    vertex[2] = {n2}")
                print(f"    direction = {ds}")
                print(f"    center = {center}")
                print(f"    plane = em.geo.Plate(origin={n1},u={ds},v=(x,y,z))")

        self._add_selectable_edges()
        self._plot.disable_picking()
        self._plot.enable_mesh_picking(
            callback,
            style="wireframe",
            show=True,
            left_clicking=True,
            use_actor=True,
            picker="hardware",
        )
        # self._plot.enable_element_picking(callback, mode='edge', show=True, tolerance=0.02, left_clicking=True, picker='cell')

    def clean(self) -> None:
        del self._state
        self._state = None

    def _get_edge_length(self):
        return max(1e-3, min(self._mesh.edge_lengths))

    ############################################################
    #                       SPECIFIC METHODS                  #
    ############################################################

    def _register_printer(self):
        self._ruler._call_coords = _print_coords

    def _volume_edges(self, obj: GeoObject | Selection) -> pv.UnstructuredGrid:
        """Adds the edges of objects

        Args:
            obj (DomainSelection | None, optional): _description_. Defaults to None.

        Returns:
            pv.UnstructuredGrid: The unstrutured grid object
        """
        edge_ids = self._mesh.domain_edges(obj.dimtags)
        if len(edge_ids) == 0:
            raise ValueError(f"Cannot plot {obj}")
            return None
        nedges = edge_ids.shape[0]
        cells = np.zeros((nedges, 3), dtype=np.int64)
        cells[:, 1:] = self._mesh.edges[:, edge_ids].T
        cells[:, 0] = 2
        celltypes = np.full(nedges, fill_value=pv.CellType.CUBIC_LINE, dtype=np.uint8)
        points = self._mesh.nodes.copy().T
        return pv.UnstructuredGrid(cells, celltypes, points)

    def mesh_surface(self, surface: FaceSelection) -> pv.UnstructuredGrid:
        tris = self._mesh.get_triangles(surface.tags)
        if tris.shape[0] == 0:
            return None
        ntris = tris.shape[0]
        cells = np.zeros((ntris, 4), dtype=np.int64)
        cells[:, 1:] = self._mesh.tris[:, tris].T
        cells[:, 0] = 3
        celltypes = np.full(ntris, fill_value=pv.CellType.TRIANGLE, dtype=np.uint8)
        points = self._mesh.nodes.copy().T
        points[:, 2] += self.set.z_boost
        return pv.UnstructuredGrid(cells, celltypes, points)

    def mesh(self, obj: GeoObject | Selection | Iterable) -> pv.UnstructuredGrid | None:
        if isinstance(obj, Iterable):
            obj = _merge(obj)
        else:
            obj = _select(obj)

        if isinstance(obj, DomainSelection):
            return self.mesh_volume(obj)
        elif isinstance(obj, FaceSelection):
            return self.mesh_surface(obj)
        else:
            return None

    ############################################################
    #                        EMERGE METHODS                    #
    ############################################################

    def add_anchors(self, anchors: list[Anchor], size: float = 1.0) -> None:
        """Adds a list of anchors to display in the current view"""
        xaxs = [f._x for f in anchors]
        yaxs = [f._y for f in anchors]
        zaxs = [f._z for f in anchors]
        c0s = [f.c0 for f in anchors]

        x, y, z = zip(*c0s)
        xx, xy, xz = zip(*xaxs)
        yx, yy, yz = zip(*yaxs)
        zx, zy, zz = zip(*zaxs)
        x, y, z, xx, xy, xz, yx, yy, yz, zx, zy, zz = [
            np.array(_) for _ in [x, y, z, xx, xy, xz, yx, yy, yz, zx, zy, zz]
        ]
        self.add_quiver(
            x, y, z, xx, xy, xz, scale=size, color=self.set.theme.axis_x_color
        )
        self.add_quiver(
            x, y, z, yx, yy, yz, scale=size, color=self.set.theme.axis_y_color
        )
        self.add_quiver(
            x, y, z, zx, zy, zz, scale=size, color=self.set.theme.axis_z_color
        )

    def mesh_volume(self, volume: DomainSelection) -> pv.UnstructuredGrid:
        tets = self._mesh.get_tetrahedra(volume.tags)
        ntets = tets.shape[0]
        cells = np.zeros((ntets, 5), dtype=np.int64)
        cells[:, 1:] = self._mesh.tets[:, tets].T
        cells[:, 0] = 4
        celltypes = np.full(ntets, fill_value=pv.CellType.TETRA, dtype=np.uint8)
        points = self._mesh.nodes.copy().T
        return pv.UnstructuredGrid(cells, celltypes, points)

    @property
    def _mesh(self) -> Mesh3D:
        return self._state.mesh

    def add_object(
        self,
        obj: GeoObject | Selection,
        mesh: bool = False,
        volume_mesh: bool = True,
        label: bool = False,
        label_text: str | None = None,
        texture: str | None = None,
        opacity: float | None = None,
        draw_line: bool = True,
        pbr: bool = True,
        smooth_shading: bool = False,
        selectable_as: str | None = None,
        minimize_opacity: bool = False,
        *args,
        **kwargs,
    ):
        """Adds an EMerge object or selection to the plot

        Args:
            obj (GeoObject | Selection): The object or object selection
            mesh (bool, optional): If the mesh should be plotted. Defaults to False.
            volume_mesh (bool, optional): If the mesh (in case mesh=True) should be plotted as a volume mesh (As opposed to surface). Defaults to True.
            label (bool, optional): If object name labels should be added. Defaults to False.
            label_text (str | None, optional): Specify a custom label text. Defaults to None.
            texture (str | None, optional): Selects a surface texture name for rendering. Defaults to None.
            opacity (float | None, optional): The objects plot opacity. Defaults to None.
            draw_line (bool, optional): Draw a perimiter line of the geometry edges. Defaults to True.
            pbr (bool, optional): Turns on physics based rendering in PyVista. Defaults to True.
            smooth_shading (bool, optional): Turns on Smooth shading of geometries (smoothes out surfaces). Defaults to False.
            selectable_as (str | None, optional): Make the geometry selectable under a given name. Defaults to None.
            minimize_opacity (bool, optional): Minimize the provided opacity with the material opacity. Defaults to False.
        """
        if isinstance(obj, GeoObject):
            if obj._hidden or obj.dim in (0,1):
                return

        mesh_obj = self.mesh(obj)
        if mesh_obj is None:
            return
        actor = self._add_obj(
            mesh_obj,
            obj.dim,
            plot_mesh=mesh,
            volume_mesh=volume_mesh,
            metal=obj._metal,
            opacity=self.parse_opacity(opacity, obj.opacity, minimize=minimize_opacity),
            color=obj.color_str,
            texture=texture,
            allow_pbr=pbr,
            smooth_shading=smooth_shading,
        )

        if selectable_as:
            self._add_selectable(mesh=mesh_obj, actor=actor, name=selectable_as)

        if draw_line:
            mesh_obj = self._volume_edges(_select(obj))

            if mesh_obj is not None:
                self._plot.add_mesh(
                    mesh_obj,
                    line_width=self.set.theme.geo_edge_width,
                    color=self.set.theme.geo_edge_color,
                    pickable=False,
                    show_edges=True,
                )
            else:
                return

        if label:
            points = []
            labels = []
            label_text = obj.name if label_text is None else label_text
            for dim, tag in obj.dimtags:
                if dim == 2:
                    points.append(self._mesh.ftag_to_point[tag])
                else:
                    points.append(self._mesh.dimtag_to_center[(dim, tag)])
                labels.append(label_text)
            self._plot.add_point_labels(
                points,
                labels,
                text_color=self.set.theme.text_color,
                shape_color=self.set.theme.label_color,
            )

    def add_objects(self, *objects, opacity: float | None = None, minimize_opacity: bool = True, **kwargs) -> None:
        """Add a series of objects provided as a list of arguments"""
        for obj in objects:
            self.add_object(obj, opacity=opacity, minimize_opacity=minimize_opacity, **kwargs)

    def populate(self, opacity: float | None = None, minimize_opacity: bool = True, **kwargs) -> None:
        """Populate the view with all objects in your simulation model

        Args:
            opacity (float, optional): The max opacity to use for all geometries. Defaults to 1.0.
            minimize_opacity (bool, optional): If the minimum of the objects natural opacity and true opacity should be chosen. Defaults to True.
        """
        for obj in self._state.current_geo_state:
            self.add_object(obj, opacity=opacity, minimize_opacity=minimize_opacity, **kwargs)

    def add_scatter(self, xs: np.ndarray, ys: np.ndarray, zs: np.ndarray):
        """Adds a scatter point cloud

        Args:
            xs (np.ndarray): The X-coordinate
            ys (np.ndarray): The Y-coordinate
            zs (np.ndarray): The Z-coordinate
        """
        cloud = pv.PolyData(np.array([xs, ys, zs]).T)
        self._data_sets.append(cloud)
        self._plot.add_points(cloud)

    def add_line(self, line: Line, width: float = 3.0, color: str = "EMERGE-RED"):
        """Adds a Line object to the plot

        Args:
            line (Line): The line object
            width (float, optional): The line width. Defaults to 3.0.
            color (str, optional): The line color. Defaults to "EMERGE-RED".
        """
        xs, ys, zs = line.cpoint
        p_line = pv.Line(
            pointa=(xs[0], ys[0], zs[0]),
            pointb=(xs[-1], ys[-1], zs[-1]),
        )
        self._plot.add_mesh(
            p_line,
            color=self.set.theme.parse_color(color),
            pickable=False,
            line_width=width,
        )

    def add_portmode(
        self,
        port: PortBC,
        dv: tuple[float, float, float] = (0, 0, 0),
        XYZ: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
        field: Literal["E", "H", "gradE", "Ez"] = "E",
        k0: float | None = None,
        cmap: str | None = None,
        mode_number: int | None = None,
        color: str | None = None,
        Npoints: int = 10,
    ) -> None:
        """Add a port boundary condition's mode to the plot window

        Args:
            port (PortBC): The port boundary condition
            dv (tuple[float, float, float], optional): Slight offset to give to the port mode plot. Defaults to (0, 0, 0).
            XYZ (tuple[np.ndarray, np.ndarray, np.ndarray] | None, optional): A set of grid points where to evaluate the modal field vectors. Defaults to None.
            field ("E","H","gradE","Ez", optional): The field component to plot. Defaults to "E".
            k0 (float | None, optional): The k0 valow for which to evaluate the mode. Defaults to None.
            cmap (str | None, optional): The colormap to use. Defaults to None.
            mode_number (int | None, optional): The mode number to plot. Defaults to None.
            color (str | None, optional): The color to use for the vector arrow. Defaults to None.
            Npoints (int, optional): Depricated property. Defaults to 10.
        """
        if XYZ:
            X, Y, Z = XYZ
        else:
            tris = self._mesh.get_triangles(port.selection.tags)
            ids = np.sort(np.unique(self._mesh.tris[:, tris].flatten()))
            X = self._mesh.tri_centers[0, tris]
            Y = self._mesh.tri_centers[1, tris]
            Z = self._mesh.tri_centers[2, tris]
            X2 = self._mesh.nodes[0, ids]
            Y2 = self._mesh.nodes[1, ids]
            Z2 = self._mesh.nodes[2, ids]
            X = np.concatenate((X, X2))
            Y = np.concatenate((Y, Y2))
            Z = np.concatenate((Z, Z2))

        X = X + dv[0]
        Y = Y + dv[1]
        Z = Z + dv[2]
        xf = X.flatten()
        yf = Y.flatten()
        zf = Z.flatten()

        d = _min_distance(xf, yf, zf)

        if port.voltage_integration_line is not None:
            for line in port.voltage_integration_line:
                xs, ys, zs = line.cpoint
                p_line = pv.Line(
                    pointa=(xs[0], ys[0], zs[0]),
                    pointb=(xs[-1], ys[-1], zs[-1]),
                )
                self._plot.add_mesh(
                    p_line,
                    color="red",
                    pickable=False,
                    line_width=3.0,
                )
        if port.current_integration_line is not None:
            for line in port.current_integration_line:
                xs, ys, zs = line.cpoint
                for x1, x2, y1, y2, z1, z2 in zip(
                    xs[:-1], xs[1:], ys[:-1], ys[1:], zs[:-1], zs[1:]
                ):
                    p_line = pv.Line(
                        pointa=(x1, y1, z1),
                        pointb=(x2, y2, z2),
                    )
                    self._plot.add_mesh(
                        p_line,
                        color="blue",
                        pickable=False,
                        line_width=3.0,
                    )
        if k0 is None:
            if isinstance(port, ModalPort):
                k0 = port.get_modes(0)[0].k0
            else:
                k0 = 1

        if mode_number is None:
            mode_number = 1

        F = port.port_mode_3d_global(xf, yf, zf, k0, which=field, mode_nr=mode_number)

        Fx = F[0, :].reshape(X.shape).T
        Fy = F[1, :].reshape(X.shape).T
        Fz = F[2, :].reshape(X.shape).T

        if cmap is None:
            cmap = self.set.theme.default_amplitude_cmap
        else:
            cmap = self.set.theme.parse_cmap_name(cmap)

        if field in ["H", "modprof"]:
            F = np.real(F.T)
            Fnorm = np.sqrt(Fx.real**2 + Fy.real**2 + Fz.real**2).T
        elif field in ("gradE", "Ez", "modprof"):
            F = np.imag(F.T)
            Fnorm = np.sqrt(Fx.imag**2 + Fy.imag**2 + Fz.imag**2).T
        else:
            F = np.real(F.T)
            Fnorm = np.sqrt(Fx.real**2 + Fy.real**2 + Fz.real**2).T

        if XYZ is not None:
            grid = pv.StructuredGrid(X, Y, Z)
            self.add_surf(X, Y, Z, Fnorm, _fieldname="portfield")
            self._wrap_plot(grid, scalars=Fnorm.T, opacity=0.8, pickable=False)

        Emag = F / np.max(Fnorm.flatten()) * d * 3
        actor = self._plot.add_arrows(
            np.array([xf, yf, zf]).T,
            Emag,
            cmap=cmap,
            color=self.set.theme.parse_color(color),
            show_scalar_bar=False,
        )
        self._data_sets.append(actor.mapper.dataset)

    def add_backgroundfields(self, fields: list[BackgroundField], radius: float) -> None:
        coords = []
        ks = []
        Es = []
        Hs = []
        for field in fields:
            E, H, K = field.EHK(0,0,0)
            R = K/np.linalg.norm(K)
            p0 = -R*radius
            coords.append(p0)
            ks.append(R*radius/10)
            Es.append(E/np.linalg.norm(E)*radius*0.1)
            Hs.append(H/np.linalg.norm(H)*radius*0.1)

        xs,ys,zs = zip(*coords)
        kx,ky,kz = zip(*ks)
        Ex,Ey,Ez = zip(*Es)
        Hx,Hy,Hz = zip(*Hs)

        xs = np.array(xs).real
        ys = np.array(ys).real
        zs = np.array(zs).real
        kx = np.array(kx).real
        ky = np.array(ky).real
        kz = np.array(kz).real
        Ex = np.array(Ex).real
        Ey = np.array(Ey).real
        Ez = np.array(Ez).real
        Hx = np.array(Hx).real
        Hy = np.array(Hy).real
        Hz = np.array(Hz).real
        
        self.add_quiver(xs,ys,zs,kx,ky,kz, color='green')#, color='EMERGE-RED')
        self.add_quiver(xs,ys,zs,Ex,Ey,Ez, color='red')#, color='EMERGE-BLUE')
        self.add_quiver(xs,ys,zs,Hx,Hy,Hz, color='blue')#, color='EMERGE-GREEN')
        

    def add_farfield3d(
        self,
        farfield_obj: EHFieldFF,
        component: Literal[
            "Ex", "Ey", "Ez", "Etheta", "Ephi", "normE", "Erhcp", "Elhcp", "AR"
        ] = "normE",
        quantity: Literal["abs", "real", "imag", "angle"] = "abs",
        dB: bool = False,
        dBfloor: float = -30,
        rmax: float | None = None,
        offset: tuple[float, float, float] = (0, 0, 0),
        opacity: float | None = None,
    ):
        surfobj = farfield_obj.surfplot(
            component,
            quantity=quantity,
            dB=dB,
            dBfloor=dBfloor,
            rmax=rmax,
            offset=offset,
        )

        self.add_surf(
            *surfobj.xyzf,
            clim=surfobj.clim,
            opacity=self.parse_opacity(opacity, "EMERGE-FFSURF"),
            _fieldname=surfobj.name,
            **self.set.theme.farfield_3d_kwarg,
        )

    def add_particle_lines(
        self, 
        lines: list[np.ndarray] | np.ndarray, 
        tube_radius: float = 0.0001, 
        cmap: str = 'coolwarm',
        add_arrows: bool = True,
        num_arrows: int = 8,
        arrow_scale: float = 0.003,
        show_scalar_bar: bool = False
    ) -> None:
        """Adds particle trajectories as smooth tubes with gradient coloring and directional arrows.
        
        Supports (3, N) arrays for arc-length coloring or (4, N) arrays where the 4th row 
        defines custom scalar values (e.g., velocity, Poynting magnitude, time).
        """
        if isinstance(lines, np.ndarray) and lines.ndim == 2:
            lines = [lines]

        for traj in lines:
            # Standardize orientation to N rows
            data = traj.T if traj.shape[0] in (3, 4) else traj
            n_pts, n_cols = data.shape
            
            if n_pts < 2:
                continue

            # Extract 3D spatial points
            pts = data[:, :3]

            # Determine scalar values: 4th axis if present, otherwise default to normalized arc length [0, 1]
            if n_cols >= 4:
                point_scalars = data[:, 3]
                point_scalars = np.log10(point_scalars/376)
                maxval = np.max(np.abs(point_scalars))
                clim = (-maxval, +maxval)
            else:
                point_scalars = np.linspace(0, 1, n_pts)
                clim = (0,1)
            # 1. Create base PolyData with attached scalars
            poly = pv.PolyData(pts)
            poly.point_data['line_scalars'] = point_scalars

            # Construct smooth spline; PyVista automatically interpolates point_data onto the spline
            spline_mesh = pv.Spline(pts)
            spline_mesh.point_data['line_scalars'] = point_scalars

            # Tube filter interpolates point scalars smoothly onto the output surface mesh
            tube = spline_mesh.tube(radius=tube_radius)

            self._plot.add_mesh(
                tube, 
                scalars='line_scalars', 
                cmap=cmap, 
                clim=clim,
                show_scalar_bar=show_scalar_bar,
                smooth_shading=True
            )

            # 2. Add directional glyph arrows along the path
            if add_arrows and n_pts >= 4:
                # Subsample points for arrow placement
                indices = np.linspace(0, n_pts - 2, min(num_arrows, n_pts - 1), dtype=int)
                arrow_positions = pts[indices]

                # Calculate forward tangent vectors
                tangents = pts[indices + 1] - arrow_positions
                norms = np.linalg.norm(tangents, axis=1, keepdims=True)
                tangents = np.divide(tangents, norms, out=np.zeros_like(tangents), where=norms != 0)

                # Map corresponding custom scalar values to arrows
                arrow_scalars = point_scalars[indices]

                # Build PolyData & apply vector glyph filter
                arrow_pd = pv.PolyData(arrow_positions)
                arrow_pd['tangents'] = tangents
                arrow_pd['colors'] = arrow_scalars

                glyphs = arrow_pd.glyph(
                    orient='tangents',
                    scale=False,
                    factor=arrow_scale,
                    geom=pv.Cone(radius=0.4, height=1.0)
                )

                self._plot.add_mesh(
                    glyphs, 
                    scalars='colors', 
                    cmap=cmap, 
                    show_scalar_bar=False,
                    ambient=0.3
                )

    def add_solution_error(
        self,
        error_value: np.ndarray,
        tet_ids: np.ndarray | None = None,
        mode: Literal["volume", "points", "surface"] = "volume",
        cmap: str | None = None,
        log_scale: bool = True,
        opacity: float | str = "linear",
        show_scalar_bar: bool = True,
        clim: tuple[float, float] | None = None,
        scalar_name: str = "Error",
        point_size_range: tuple[float, float] | None = None,
    ) -> None:
        """Displays a per-tetrahedron error indicator.
 
        Intended for visualizing the output of the adaptive mesh refinement
        error estimator (compute_error_estimate) -- either the full-domain
        error field, or just the tets that got flagged for refinement.
 
        Args:
            error_value (np.ndarray): Error values, one per tetrahedron. If
                tet_ids is None, this must have length self._mesh.n_tets and
                is assumed to be given in the same order as self._mesh.tets.
                If tet_ids is provided, error_value must have the same
                length as tet_ids, and only those tetrahedra are used.
            tet_ids (np.ndarray | None, optional): The tetrahedron indices
                to plot (e.g. the output of select_refinement_indices).
                Defaults to None (use all tetrahedra).
            mode ("volume", "points", "surface", optional): How to render.
                "volume" does true volumetric rendering with transparency,
                so interior tets are actually visible through the exterior
                -- this is almost always what you want for an error plot,
                since a plain opaque solid ("surface") only ever shows you
                the outer boundary tets. "points" instead draws one sphere
                per tet at its centroid, sized AND colored by error, which
                avoids any transparency/occlusion ambiguity at the cost of
                losing the continuous solid shape. Defaults to "volume".
            cmap (str | None, optional): Colormap name. Defaults to the
                theme's default amplitude colormap.
            log_scale (bool, optional): If True, colors on a log scale --
                error estimates commonly span many orders of magnitude and
                a linear scale usually washes out everything except the
                single worst tet. Defaults to True.
            opacity (float | str, optional): For mode="volume", this is
                pyvista's opacity TRANSFER FUNCTION (e.g. "linear",
                "sigmoid", or an array), not a single alpha value -- it
                controls how strongly each error magnitude is allowed to
                occlude what's behind it. For "points"/"surface" this is a
                plain 0-1 opacity. Defaults to "linear".
            show_scalar_bar (bool, optional): Whether to show the color
                legend. Defaults to True.
            clim (tuple[float, float] | None, optional): Manual color
                limits. Defaults to the data's own min/max.
            scalar_name (str, optional): Label used for the scalar bar.
                Defaults to "Error".
            point_size_range (tuple[float, float] | None, optional): Only
                used for mode="points". Min/max sphere RADIUS in your
                model's own length units (not fixed on-screen pixels) --
                the lowest-error tet gets the smaller sphere, the highest
                gets the larger. Defaults to roughly 2%/8% of the mesh's
                overall bounding-box diagonal, so it scales sensibly with
                model size instead of needing manual tuning per model.
        """
        error_value = np.asarray(error_value)
 
        if tet_ids is None:
            tet_ids = np.arange(self._mesh.n_tets)
        else:
            tet_ids = np.asarray(tet_ids)
 
        if error_value.shape[0] != tet_ids.shape[0]:
            raise ValueError(
                f"error_value length ({error_value.shape[0]}) must match the "
                f"number of tetrahedra being plotted ({tet_ids.shape[0]}). "
                "Pass tet_ids explicitly if error_value only covers a subset "
                "(e.g. the tets flagged by select_refinement_indices)."
            )
 
        if cmap is None:
            cmap = self.set.theme.default_amplitude_cmap
        else:
            cmap = self.set.theme.parse_cmap_name(cmap)
 
        if mode == "points":
            centers = self._mesh.centers[:, tet_ids].T  # (ntets, 3)
            cloud = pv.PolyData(centers)
            cloud.point_data[scalar_name] = error_value
 
            if point_size_range is None:
                bbox = self._mesh.nodes.max(axis=1) - self._mesh.nodes.min(axis=1)
                diag = float(np.linalg.norm(bbox))
                point_size_range = (0.02 * diag, 0.08 * diag)
 
            plot_values = (
                np.log10(error_value + np.finfo(float).eps) if log_scale else error_value
            )
            norm = (plot_values - plot_values.min()) / (np.ptp(plot_values) + 1e-30)
            cloud.point_data["_size"] = (
                point_size_range[0] + norm * (point_size_range[1] - point_size_range[0])
            )
 
            glyphs = cloud.glyph(scale="_size", geom=pv.Sphere(radius=1.0), orient=False)
            actor = self._plot.add_mesh(
                glyphs,
                scalars=scalar_name,
                cmap=cmap,
                opacity=opacity if isinstance(opacity, (int, float)) else 1.0,
                clim=clim,
                show_scalar_bar=show_scalar_bar,
            )
            self._data_sets.append(glyphs)
            return
 
        # volume / surface modes both need the tet UnstructuredGrid
        ntets = tet_ids.shape[0]
        cells = np.zeros((ntets, 5), dtype=np.int64)
        cells[:, 1:] = self._mesh.tets[:, tet_ids].T
        cells[:, 0] = 4
        celltypes = np.full(ntets, fill_value=pv.CellType.TETRA, dtype=np.uint8)
        points = self._mesh.nodes.copy().T
 
        grid = pv.UnstructuredGrid(cells, celltypes, points)
        grid.cell_data[scalar_name] = error_value
 
        if mode == "volume":
            actor = self._plot.add_volume(
                grid,
                scalars=scalar_name,
                cmap=cmap,
                opacity=opacity,
                clim=clim,
                show_scalar_bar=show_scalar_bar,
                log_scale=log_scale,  # pyvista's own log-scale handling
            )
        else:  # "surface"
            plot_values = (
                np.log10(error_value + np.finfo(float).eps) if log_scale else error_value
            )
            grid.cell_data[scalar_name] = plot_values
            actor = self._plot.add_mesh(
                grid,
                scalars=scalar_name,
                cmap=cmap,
                opacity=opacity if isinstance(opacity, (int, float)) else 1.0,
                clim=clim,
                show_scalar_bar=show_scalar_bar,
            )
 
        self._data_sets.append(grid)