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

"""GDSII layout import.

GDS files only contain 2D shapes per (layer, datatype). The vertical stack-up
(height, thickness and material of every layer) is supplied with GDSLayer:

>>> gds = GDSFile('inductor.gds', layers={
...     1:  GDSLayer(z=5e-6, thickness=2e-6, material=em.lib.COPPER, name='M_top'),
...     3:  GDSLayer(z=3e-6, thickness=2e-6, name='Via'),
...     2:  GDSLayer(z=2e-6, thickness=1e-6, name='M_under'),
...     10: GDSLayer(z=0.0, name='Shield'),          # no thickness -> GeoSurface
... })
>>> gds['M_top']                   # GeoVolume
>>> gds.label('P1')                # GDSLabel with x, y in meters

Pipeline per layer:
    1. Every polygon (boundaries, boxes, paths, flattened cell references) becomes a
       gmsh plane surface.
    2. The surfaces of the layer are fused, so overlapping shapes become one face.
    3. The result is extruded by the layer thickness (GeoVolume) or kept as a
       sheet (GeoSurface).
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import gmsh
import numpy as np
from loguru import logger
from emsutil import Material

from ..._emerge.cs import CoordinateSystem, GCS
from ..._emerge.geometry import GeoSurface, GeoVolume
from ..._emerge.geo.operations import extrude
from ..._emerge.geo.shapes import Plate
from ..._emerge.attributes import LumpedPortAttribute
from .gdsii import GDSLibrary, GDSLabel, GDSPolygon, GDSParseError, drop_short_edges
from .ports import GDSPortEdge, find_label_edge

LayerKey = int | tuple[int, int]


@dataclass
class GDSLayer:
    """Stack-up settings of one GDS layer.

    Args:
        z (float): Height of the bottom of the layer in meters.
        thickness (float | None): Thickness in meters. None creates a zero-thickness sheet (GeoSurface).
        material (Material | None): Material assigned to the object. None keeps the default.
        name (str | None): Name used for the object and for lookups (gds['name']). Defaults to 'GDS_L<layer>'.
        via_merge (float | None): Treat this layer as a via layer and merge vias whose centers are closer
            than this distance (meters) into thickened walls/blocks, using emcad's via-wall routine (the
            same one as the ODB++ importer's autojoin_limit). Dense via arrays then become a few solids
            instead of hundreds. Each via is drawn as a circle with the same area as its GDS shape.
            Requires emcad (pip install emerge[cad]). None (default) keeps every via as drawn.
        via_segments (int): Segments per via circle when merging vias.
        via_min_hole_area (float | None): Enclosed gaps in a merged via network smaller than this (m^2) are
            filled instead of kept as holes. Defaults to 1.1 * via_merge**2, which fills the gaps inside
            regular via arrays.
        max_meshsize (float | None): Maximum mesh element size in this layer, in physical meters (scaled with
            GDSFile.scale_factor internally). Useful for thin metal layers: without it the wavelength-based
            default can leave long sliver triangles on their side faces, which gmsh may fail to mesh around.
    """
    z: float
    thickness: float | None = None
    material: Material | None = None
    name: str | None = None
    via_merge: float | None = None
    via_segments: int = 8
    via_min_hole_area: float | None = None
    max_meshsize: float | None = None

    # Set by GDSFile once the layer is built, so settings can still be changed afterwards.
    _object: GeoSurface | GeoVolume | None = field(default=None, init=False, repr=False)
    _scale_factor: float = field(default=1.0, init=False, repr=False)

    @property
    def object(self) -> GeoSurface | GeoVolume | None:
        """The geometry built for this layer (None until built, or if the layer has no polygons)."""
        return self._object

    def set_max_meshsize(self, size: float | None) -> GDSLayer:
        """Sets the maximum mesh element size of this layer, in physical meters. Can be called before or
        after the geometry is built; the modelling scale factor is applied internally."""
        self.max_meshsize = size
        if self._object is not None:
            self._object.max_meshsize = 1e9 if size is None else size * self._scale_factor
        return self


class GDSFile:
    """Imports a GDSII file as EMerge geometry.

    Args:
        filename (str | Path): Path to the .gds file.
        layers (dict | None): Maps a GDS layer number (any datatype) or a (layer, datatype) tuple to
            a GDSLayer. Layers that are not mapped are skipped. With None, the file is only parsed so
            you can inspect it (see .found_layers and .summary()).
        cell (str | None): Cell to import. Defaults to the single top cell.
        cs (CoordinateSystem): Coordinate system the layout is placed in. Layers extrude along its z-axis.
        scale (float): Extra scale factor applied to all coordinates (GDS units are already converted to meters).
        scale_factor (float): Modelling scale factor. Must match Simulation.set_scale_factor(...). All geometry
            is built this many times larger, so that micron-sized layouts stay well above the OpenCASCADE
            tolerance; EMerge scales the mesh back down after meshing. GDSLayer values stay in real meters,
            and lumped port widths/heights stay physical. bounds() and layer_rings() return modelling units,
            so they can be used directly to size other geometry.
        min_edge (float): Polygon edges shorter than this (physical meters) are removed before building
            geometry. GDS grid rounding leaves one-database-unit steps at corners (e.g. 5 nm) that gmsh cannot
            represent without scaling and that force an extremely fine mesh with it.
    """

    def __init__(self,
                 filename: str | Path,
                 layers: dict[LayerKey, GDSLayer] | None = None,
                 cell: str | None = None,
                 cs: CoordinateSystem = GCS,
                 scale: float = 1.0,
                 scale_factor: float = 1.0,
                 min_edge: float = 20e-9):
        self.filename: Path = Path(filename)
        self.cs: CoordinateSystem = cs
        self.min_edge: float = min_edge
        self.scale_factor: float = scale_factor
        self.library: GDSLibrary = GDSLibrary(self.filename)
        self.cell: str = cell if cell is not None else self._top_cell()

        polygons, labels = self.library.flatten(self.cell)
        for p in polygons:
            p.xy = p.xy * scale
        for lb in labels:
            lb.x, lb.y = lb.x * scale, lb.y * scale

        self.polygons: list[GDSPolygon] = polygons
        self.labels: list[GDSLabel] = labels
        self.layers: dict[LayerKey, GDSLayer] = dict(layers or {})
        self.objects: dict[LayerKey, GeoSurface | GeoVolume] = {}

        if layers is None:
            logger.info(f"GDS '{self.filename.name}' parsed without a layer mapping. "
                        f"Found layers (layer, datatype): {self.found_layers}")
            return
        self._build()

    # ----------------------------------------------------------------- setup

    def _top_cell(self) -> str:
        tops = self.library.top_cells
        if len(tops) != 1:
            raise ValueError(f"'{self.filename.name}' has {len(tops)} top cells {tops}, choose one with cell=...")
        return tops[0]

    def layer_rings(self, layer: LayerKey) -> list[np.ndarray]:
        """The polygons (N, 2) of one layer in modelling units, cleaned the same way as for building geometry.
        Useful to build extra geometry from a layer, e.g. a conformal dielectric around it."""
        rings = []
        for p in self.polygons:
            if (p.layer, p.datatype) == layer or p.layer == layer:
                xy = drop_short_edges(p.xy, self.min_edge)
                if xy is not None:
                    rings.append(xy * self.scale_factor)
        return rings

    @property
    def found_layers(self) -> list[tuple[int, int]]:
        """All (layer, datatype) combinations that contain polygons."""
        return sorted({(p.layer, p.datatype) for p in self.polygons})

    def _key_for(self, layer: int, datatype: int) -> LayerKey | None:
        if (layer, datatype) in self.layers:
            return (layer, datatype)
        if layer in self.layers:
            return layer
        return None

    # ----------------------------------------------------------------- build

    def _build(self) -> None:
        grouped: dict[LayerKey, list[np.ndarray]] = defaultdict(list)
        skipped: set[tuple[int, int]] = set()
        for p in self.polygons:
            key = self._key_for(p.layer, p.datatype)
            if key is None:
                skipped.add((p.layer, p.datatype))
                continue
            xy = drop_short_edges(p.xy, self.min_edge)
            if xy is not None:
                grouped[key].append(xy)

        if skipped:
            logger.info(f"GDS: skipping unmapped layers (layer, datatype): {sorted(skipped)}")

        for key, settings in self.layers.items():
            rings = grouped.get(key, [])
            if not rings:
                logger.warning(f"GDS: layer {key} contains no polygons, nothing created.")
                continue
            self.objects[key] = self._build_layer(key, settings, rings)
            logger.debug(f"GDS: layer {key} -> {self.objects[key]} from {len(rings)} polygons.")

    def _build_layer(self, key: LayerKey, settings: GDSLayer, rings: list[np.ndarray]) -> GeoSurface | GeoVolume:
        occ = gmsh.model.occ
        if settings.via_merge:
            faces = self._via_wall_faces(rings, settings)
        else:
            faces = [(2, self._plane_surface(xy, settings.z)) for xy in rings]
        if len(faces) > 1:
            faces, _ = occ.fuse(faces[:1], faces[1:])
        return self._finish_layer(key, settings, faces)

    def _finish_layer(self, key: LayerKey, settings: GDSLayer, faces: list[tuple[int, int]]) -> GeoSurface | GeoVolume:
        """Turns the layer's faces into the final object: extrusion, material and mesh size."""
        name = settings.name or (f"GDS_L{key}" if isinstance(key, int) else f"GDS_L{key[0]}D{key[1]}")
        obj: GeoSurface | GeoVolume = GeoSurface([t for d, t in faces if d == 2], name=name)
        if settings.thickness:
            dx, dy, dz = np.asarray(self.cs.zax.np, dtype=float) * settings.thickness * self.scale_factor
            obj = extrude(obj, dx, dy, dz)
            obj.name = name
        if settings.material is not None:
            obj.set_material(settings.material)
        settings._object, settings._scale_factor = obj, self.scale_factor
        if settings.max_meshsize is not None:
            settings.set_max_meshsize(settings.max_meshsize)
        return obj

    def _curve_loop(self, xy: np.ndarray, z: float) -> int:
        """Curve loop from physical coordinates (m), built in modelling units."""
        occ = gmsh.model.occ
        k = self.scale_factor
        xg, yg, zg = self.cs.in_global_cs(xy[:, 0] * k, xy[:, 1] * k, np.full(len(xy), z * k))
        pts = [occ.addPoint(x, y, zz) for x, y, zz in zip(xg, yg, zg)]
        n = len(pts)
        lines = [occ.addLine(pts[i], pts[(i + 1) % n]) for i in range(n)]
        return occ.addCurveLoop(lines)

    def _plane_surface(self, xy: np.ndarray, z: float) -> int:
        return gmsh.model.occ.addPlaneSurface([self._curve_loop(xy, z)])

    def _polygon_surfaces(self, poly, z: float) -> list[int]:
        """Plane surface(s) of an emcad Polygon with holes. Islands inside holes become separate surfaces."""
        ring = lambda p: np.column_stack([p.xs, p.ys])  # noqa: E731
        loops = [self._curve_loop(ring(poly), z)] + [self._curve_loop(ring(h), z) for h in poly.holes]
        tags = [gmsh.model.occ.addPlaneSurface(loops)]
        for hole in poly.holes:
            for island in hole.holes:
                tags += self._polygon_surfaces(island, z)
        return tags

    def _via_wall_faces(self, rings: list[np.ndarray], settings: GDSLayer) -> list[tuple[int, int]]:
        """Merges the vias of a layer into wall polygons with emcad's via_wall_polygons."""
        try:
            from emcad import via_wall_polygons
        except ImportError as e:
            raise ImportError("GDSLayer(via_merge=...) needs emcad. Install it with: pip install emerge[cad]") from e

        # Group vias by size, since via_wall_polygons draws every via of a call at one diameter.
        groups: dict[float, list[np.ndarray]] = defaultdict(list)
        for xy in rings:
            x, y = xy[:, 0], xy[:, 1]
            area = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
            diameter = round(2.0 * np.sqrt(area / np.pi), 12)  # circle with the same area as the via
            groups[diameter].append((xy.min(axis=0) + xy.max(axis=0)) / 2.0)

        min_hole_area = settings.via_min_hole_area
        if min_hole_area is None:
            min_hole_area = 1.1 * settings.via_merge ** 2

        tags: list[int] = []
        n_walls = 0
        for diameter, centers in groups.items():
            walls = via_wall_polygons(
                np.array(centers), max_dist=settings.via_merge, thickness=diameter,
                circle_segments=settings.via_segments, include_isolated_vias=True,
                min_hole_area=min_hole_area,
            )
            n_walls += len(walls)
            for poly in walls:
                tags += self._polygon_surfaces(poly, settings.z)
        logger.info(f"GDS: merged {len(rings)} vias into {n_walls} via wall(s) (via_merge={settings.via_merge:g} m).")
        return [(2, t) for t in tags]

    # --------------------------------------------------------------- access

    def __getitem__(self, key: LayerKey | str) -> GeoSurface | GeoVolume:
        """Look up an object by layer number, (layer, datatype) or GDSLayer name."""
        if isinstance(key, str):
            for k, settings in self.layers.items():
                if settings.name == key and k in self.objects:
                    return self.objects[k]
            raise KeyError(f"No GDS layer named '{key}'. Names: {[s.name for s in self.layers.values() if s.name]}")
        return self.objects[key]

    def __iter__(self) -> Iterator[GeoSurface | GeoVolume]:
        return iter(self.objects.values())

    def __len__(self) -> int:
        return len(self.objects)

    @property
    def volumes(self) -> list[GeoVolume]:
        return [o for o in self.objects.values() if isinstance(o, GeoVolume)]

    @property
    def surfaces(self) -> list[GeoSurface]:
        return [o for o in self.objects.values() if isinstance(o, GeoSurface)]

    def label(self, text: str) -> GDSLabel:
        """Returns the text label with this string (e.g. a port name)."""
        matches = [lb for lb in self.labels if lb.text == text]
        if not matches:
            raise KeyError(f"No label '{text}'. Labels: {[lb.text for lb in self.labels]}")
        if len(matches) > 1:
            logger.warning(f"GDS: {len(matches)} labels named '{text}', returning the first.")
        return matches[0]

    # ---------------------------------------------------------------- ports

    def _layer_settings(self, layer: int) -> GDSLayer:
        for key, settings in self.layers.items():
            if key == layer or (isinstance(key, tuple) and key[0] == layer):
                return settings
        raise ValueError(f"Layer {layer} is not in the layer mapping, so its height and thickness are unknown.")

    def _global(self, xy: np.ndarray, z: float) -> np.ndarray:
        """Physical layout point (m) to a global point in modelling units."""
        k = self.scale_factor
        x, y, zz = self.cs.in_global_cs(np.array([xy[0] * k]), np.array([xy[1] * k]), np.array([z * k]))
        return np.array([x[0], y[0], zz[0]])

    def port_edge(self, label: str | GDSLabel, layer: int | None = None,
                  max_distance: float | None = None) -> GDSPortEdge:
        """Returns the metal edge a port label sits on, with its width, outward normal and z range.

        Args:
            label (str | GDSLabel): The label text (e.g. 'P1') or a GDSLabel.
            layer (int | None): Metal layer to search. Defaults to the label's own layer. Use this when
                the labels are on a separate text/pin layer.
            max_distance (float | None): How far (m) the label may be from the edge. Defaults to 10 database units.
        """
        if isinstance(label, str):
            label = self.label(label)
        if max_distance is None:
            max_distance = 10 * self.library.db_unit
        layer = label.layer if layer is None else layer
        p0, p1, normal = find_label_edge(label, self.polygons, max_distance, layer)
        settings = self._layer_settings(layer)
        z_top = settings.z + (settings.thickness or 0.0)
        return GDSPortEdge(label, p0, p1, normal, settings.z, z_top)

    def _port_plate(self, origin, u, v, port_number: int, width: float, height: float,
                    direction: np.ndarray, impedance: float, name: str | None) -> Plate:
        plate = Plate(tuple(origin), tuple(u), tuple(v), name=name or f"GDS_LumpedPort{port_number}")
        plate.properties += LumpedPortAttribute(
            port_number=port_number, direction=tuple(direction), width=width, height=height, impedance=impedance
        )
        return plate

    def lumped_port_between(self, label_a: str | GDSLabel, label_b: str | GDSLabel, port_number: int,
                            impedance: float = 50.0, name: str | None = None,
                            layer: int | None = None, max_distance: float | None = None) -> Plate:
        """Creates one (differential) lumped port in the gap between two labelled metal edges.

        Both labels must sit on edges that lie on one line with a gap between them, like the two feed
        ends of an inductor. The port plate fills that gap over the metal thickness, so the excitation
        runs from the end of label_a to the end of label_b. No ground is needed.

        The plate carries a LumpedPortAttribute, so it becomes a LumpedPort automatically.

        Args:
            label_a, label_b (str | GDSLabel): The two port labels (e.g. 'P1', 'P2').
            port_number (int): The port number.
            impedance (float): Port reference impedance in Ohm.
            name (str | None): Object name.
            layer, max_distance: See port_edge.
        """
        ea = self.port_edge(label_a, layer, max_distance)
        eb = self.port_edge(label_b, layer, max_distance)
        if not (np.isclose(ea.z_bottom, eb.z_bottom) and np.isclose(ea.z_top, eb.z_top)):
            raise ValueError("Both port labels must be on metal with the same height and thickness.")
        if ea.z_top <= ea.z_bottom:
            raise ValueError(f"Layer {ea.label.layer} has no thickness, a gap port needs metal with a thickness.")

        tol = max(10 * self.library.db_unit, 1e-3 * min(ea.width, eb.width))
        for p in (eb.p0, eb.p1):  # both edges must lie on one line
            d = p - ea.p0
            if abs(ea.tangent[0] * d[1] - ea.tangent[1] * d[0]) > tol:
                raise ValueError(
                    f"The edges of '{ea.label.text}' and '{eb.label.text}' do not lie on one line, "
                    "so there is no straight gap between them. Use lumped_port_to_ground instead."
                )

        # The facing ends of the two edges bound the gap.
        ia, ib = min(((a, b) for a in (ea.p0, ea.p1) for b in (eb.p0, eb.p1)),
                     key=lambda ab: np.linalg.norm(ab[1] - ab[0]))
        gap = float(np.linalg.norm(ib - ia))
        if gap <= tol:
            raise ValueError(f"The edges of '{ea.label.text}' and '{eb.label.text}' touch, there is no gap for a port.")

        origin = self._global(ia, ea.z_bottom)
        u = self._global(ib, ea.z_bottom) - origin       # across the gap
        v = self._global(ia, ea.z_top) - origin          # over the metal thickness
        return self._port_plate(origin, u, v, port_number, width=ea.z_top - ea.z_bottom, height=gap,
                                direction=u / np.linalg.norm(u), impedance=impedance, name=name)

    def lumped_port_to_ground(self, label: str | GDSLabel, port_number: int,
                              ground_layer: LayerKey | None = None, z_ground: float | None = None,
                              impedance: float = 50.0, name: str | None = None,
                              layer: int | None = None, max_distance: float | None = None) -> Plate:
        """Creates a vertical lumped port from a ground conductor to the labelled metal edge.

        The plate runs along the full labelled edge, from the ground up to the bottom of the metal
        (or from the top of the metal up to the ground, if the ground is above it).

        Give the ground either as a mapped GDS layer (ground_layer), in which case its height comes from
        the stack-up and it is checked that the ground actually lies under the port, or as a height (z_ground).

        The plate carries a LumpedPortAttribute, so it becomes a LumpedPort automatically.

        Example:
            >>> p1 = gds.lumped_port_to_ground('LA', 1, ground_layer=250, layer=126, max_distance=3e-6)

        Args:
            label (str | GDSLabel): The port label (e.g. 'P1').
            port_number (int): The port number.
            ground_layer (int | tuple | None): GDS layer of the ground conductor (must be in the layer mapping).
            z_ground (float | None): Height of the ground conductor in meters, instead of ground_layer.
            impedance (float): Port reference impedance in Ohm.
            name (str | None): Object name.
            layer, max_distance: See port_edge.
        """
        if (ground_layer is None) == (z_ground is None):
            raise ValueError("Give exactly one of ground_layer or z_ground.")
        e = self.port_edge(label, layer, max_distance)

        if ground_layer is not None:
            g = self.layers.get(ground_layer)
            if g is None:
                raise ValueError(f"Ground layer {ground_layer} is not in the layer mapping.")
            g_top = g.z + (g.thickness or 0.0)
            z_ground = g_top if g_top <= e.z_bottom else g.z
            self._check_ground_under(ground_layer, e)

        if e.z_bottom <= z_ground <= e.z_top:
            raise ValueError(f"z_ground={z_ground:g} m lies inside the metal of '{e.label.text}'.")
        z_metal = e.z_bottom if z_ground < e.z_bottom else e.z_top

        origin = self._global(e.p0, z_ground)
        u = self._global(e.p1, z_ground) - origin        # along the edge
        v = self._global(e.p0, z_metal) - origin         # from ground to metal
        return self._port_plate(origin, u, v, port_number, width=e.width, height=abs(z_metal - z_ground),
                                direction=v / np.linalg.norm(v), impedance=impedance, name=name)

    def _check_ground_under(self, ground_layer: LayerKey, e: GDSPortEdge) -> None:
        """Warns if the ground layer's shapes do not cover both ends of the port edge."""
        from matplotlib.path import Path as MplPath
        shapes = [MplPath(p.xy) for p in self.polygons if self._key_for(p.layer, p.datatype) == ground_layer]
        tol = 10 * self.library.db_unit
        for pt in (e.p0, e.p1):
            if not any(s.contains_point(pt, radius=tol) or s.contains_point(pt, radius=-tol) for s in shapes):
                logger.warning(f"GDS: the ground layer {ground_layer} does not lie under the end "
                               f"({pt[0]:.4g}, {pt[1]:.4g}) m of port '{e.label.text}'.")

    def bounds(self) -> tuple[float, float, float, float]:
        """(xmin, ymin, xmax, ymax) of all mapped polygons in the layout plane, in modelling units
        (meters times scale_factor), for sizing other geometry around the layout."""
        mapped = [p.xy for p in self.polygons if self._key_for(p.layer, p.datatype) is not None] or \
                 [p.xy for p in self.polygons]
        allxy = np.vstack(mapped) * self.scale_factor
        return tuple(float(v) for v in (*allxy.min(axis=0), *allxy.max(axis=0)))

    def summary(self) -> str:
        """Human readable overview of the file contents."""
        counts: dict[tuple[int, int], int] = defaultdict(int)
        for p in self.polygons:
            counts[(p.layer, p.datatype)] += 1
        lines = [(f"GDS file '{self.filename.name}', cell '{self.cell}' "
                  f"(database unit {self.library.db_unit:g} m)")]
        for (layer, datatype), n in sorted(counts.items()):
            key = self._key_for(layer, datatype)
            target = f"-> {self.objects[key].name}" if key in self.objects else "(not mapped)"
            lines.append(f"  layer {layer:>3} / datatype {datatype:<3} {n:>5} polygons  {target}")
        for lb in self.labels:
            lines.append(f"  label '{lb.text}' on layer {lb.layer} at ({lb.x:.4g}, {lb.y:.4g}) m")
        return "\n".join(lines)


__all__ = ["GDSFile", "GDSLabel", "GDSLayer", "GDSParseError", "GDSPortEdge"]
