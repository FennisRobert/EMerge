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

"""GDSII import driven by a gds2palace stack-up XML file.

>>> stack = GDSFromXML(xml="SG13G2_200um.xml", gds="layout.gds", via_merge=2e-6, size_scale=1000)
>>> stack.layers[126].set_max_meshsize(8e-6)
>>> dielectrics = stack.build_dielectrics(margin=100e-6)
>>> p2 = stack.via_port(202, 2, from_layer="SUBGND", to_layer="TopMetal1")
>>> model.mw.bc.AbsorbingBoundary(dielectrics.boundary())

The XML provides materials, the dielectric stack, z-positions of all drawn layers and derived
(boolean) layers. All 2D polygon work (merging, derived layers, via merging) uses emcad.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path

import numpy as np
from loguru import logger
from emsutil import Material, lib

from ..._emerge.cs import CoordinateSystem, GCS
from ..._emerge.geometry import GeoSurface, GeoVolume
from ..._emerge.geo.operations import extrude
from ..._emerge.geo.shapes import Box, Plate
from .gds import GDSFile, GDSLayer
from .gdsii import drop_short_edges
from .stackup_xml import XMLStackup, XMLLayer, read_stackup_xml

# Material assignment priorities: metals win over drawn dielectrics (e.g. conformal oxide), which win
# over the planar dielectric stack.
PRIO_METAL = 20
PRIO_DRAWN_DIELECTRIC = 15
PRIO_STACK_DIELECTRIC = 10


class DielectricStack:
    """The planar dielectric stack built by GDSFromXML.build_dielectrics().

    Attributes:
        volumes (dict[str, GeoVolume]): One object per <Dielectric>, by name.
        region (GeoVolume): Air box enclosing the whole stack, lowest priority. Its faces are the
            outer boundary of the model.
    """

    def __init__(self, volumes: dict[str, GeoVolume], region: GeoVolume):
        self.volumes = volumes
        self.region = region

    def boundary(self):
        """The outer faces of the model, e.g. for an AbsorbingBoundary."""
        return self.region.boundary()

    def __getitem__(self, name: str) -> GeoVolume:
        return self.volumes[name]

    def __iter__(self) -> Iterator[GeoVolume]:
        return iter(self.volumes.values())


class GDSFromXML(GDSFile):
    """Imports a GDSII layout using a gds2palace stack-up XML file for all layer and material data.

    Every <Layer> of the XML that has polygons in the GDS file (directly or through a <DerivedLayer>)
    is built; the others are skipped. All label-based port functions of GDSFile are available too.

    Args:
        xml (str | Path): The stack-up XML file.
        gds (str | Path): The GDSII file.
        via_merge (float | None): Merge vias of every via-type layer whose centers are closer than this
            distance (physical meters) into solid blocks/walls. None keeps every via.
        variables (dict | None): Overrides for the XML's <Variables>.
        size_scale (float): Modelling scale factor. Must match Simulation.set_scale_factor(...).
        cell (str | None): Cell to import. Defaults to the single top cell.
        cs (CoordinateSystem): Coordinate system the layout is placed in.
        datatypes (tuple[int, ...] | None): GDS datatypes ("purposes") to read. Defaults to (0,), the drawing
            datatype, as in gds2palace. None reads all datatypes.
        min_edge (float): Edges shorter than this (physical meters) are removed (GDS grid steps).
        merge_tol (float): Vertex merge tolerance (physical meters) for the emcad polygon operations.
    """

    def __init__(self,
                 xml: str | Path,
                 gds: str | Path,
                 via_merge: float | None = None,
                 variables: dict[str, float | str] | None = None,
                 size_scale: float = 1.0,
                 cell: str | None = None,
                 cs: CoordinateSystem = GCS,
                 datatypes: tuple[int, ...] | None = (0,),
                 min_edge: float = 20e-9,
                 merge_tol: float = 1e-9):
        try:
            import emcad  # noqa: F401
        except ImportError as e:
            raise ImportError("GDSFromXML needs emcad. Install it with: pip install emerge[cad]") from e

        self.stackup: XMLStackup = read_stackup_xml(xml, variables)
        self.datatypes = None if datatypes is None else set(datatypes)
        self.merge_tol = merge_tol
        self.dielectrics: DielectricStack | None = None
        self._xml_layers: dict[int, XMLLayer] = {}
        self._shapes: dict[int, list] = {}

        layers: dict[int, GDSLayer] = {}
        for lay in self.stackup.layers:
            if lay.gds_layer in layers:
                raise ValueError(f"GDS layer {lay.gds_layer} is used by two stack-up layers "
                                 f"('{layers[lay.gds_layer].name}' and '{lay.name}').")
            xml_mat = self.stackup.materials.get(lay.material)
            if xml_mat is not None and xml_mat.type.lower() == "resistor":
                logger.warning(f"GDS: sheet resistor layer '{lay.name}' (Rs={xml_mat.rs} Ohm/sq) is not supported "
                               "yet and is skipped.")
                continue
            layers[lay.gds_layer] = GDSLayer(
                z=lay.zmin, thickness=lay.thickness if lay.thickness > 0 else None,
                material=self._material(lay.material), name=lay.name,
                via_merge=via_merge if lay.type == "via" else None,
            )
            self._xml_layers[lay.gds_layer] = lay

        super().__init__(gds, layers=layers, cell=cell, cs=cs, scale_factor=size_scale, min_edge=min_edge)

    # ------------------------------------------------------------- materials

    def _material(self, name: str) -> Material:
        if name == "PEC":
            return lib.PEC
        m = self.stackup.materials[name]
        color = f"#{m.color}" if m.color else "#BEBEBE"
        if m.type.lower() == "conductor":
            return Material(cond=m.conductivity, color=color, name=name)
        return Material(er=m.permittivity, tand=m.loss_tangent, cond=m.conductivity, color=color, name=name)

    # ------------------------------------------------------------ 2D shapes

    def _tol(self) -> dict:
        # emcad's defaults are tuned for PCB sizes in meters (area_tol = 100 um^2!), far too coarse on chip.
        return {"merge_tol": self.merge_tol, "area_tol": self.merge_tol ** 2}

    def _union(self, polys: list) -> list:
        from emcad import add_polygons
        return add_polygons(*polys, **self._tol()) if polys else []

    def _subtract(self, a: list, b: list) -> list:
        from emcad import subtract_polygons
        if not a or not b:
            return list(a)
        return subtract_polygons(tuple(a), tuple(b), **self._tol())

    def _offset(self, polys: list, distance: float) -> list:
        from emcad import offset_polygon
        grown = [q for p in polys for q in offset_polygon(p, distance, **self._tol())]
        return self._union(grown)

    def _native_polygons(self, layer: int) -> list:
        """The drawn polygons of a GDS layer as emcad polygons (physical meters)."""
        import emcad as cad
        from emcad.kernel.api import sanitize_polygon
        polys = []
        for p in self.polygons:
            if p.layer != layer:
                continue
            xy = drop_short_edges(p.xy, self.min_edge)
            if xy is None:
                continue
            try:
                polys.append(cad.Polygon(*sanitize_polygon(xy[:, 0], xy[:, 1], self.merge_tol, True)))
            except ValueError:
                pass  # degenerate polygon
        return polys

    def shapes(self, layer: int, _visiting: tuple[int, ...] = ()) -> list:
        """The merged emcad polygons (with holes, physical meters) of a native or derived GDS layer."""
        if layer in self._shapes:
            return self._shapes[layer]
        derived = {d.gds_layer: d for d in self.stackup.derived_layers}
        if layer in derived:
            if layer in _visiting:
                raise ValueError(f"Circular derived layer definition involving layer {layer}.")
            d = derived[layer]
            ops = [self.shapes(o, _visiting + (layer,)) for o in d.operands]
            if d.operation == "SIZE":
                result = self._offset(ops[0], d.oversize)
            else:
                result = ops[0]
                for other in ops[1:]:
                    if d.operation == "OR":
                        result = self._union(result + other)
                    elif d.operation == "AND":
                        result = self._subtract(result, self._subtract(result, other))
                    elif d.operation == "NOT":
                        result = self._subtract(result, other)
                    elif d.operation == "XOR":
                        result = self._union(self._subtract(result, other) + self._subtract(other, result))
                if d.oversize:
                    result = self._offset(result, d.oversize)
        else:
            result = self._union(self._native_polygons(layer))
        for poly in result:
            poly.simplify(self.min_edge)  # drop GDS grid steps along straight runs
        self._shapes[layer] = result
        return result

    # ----------------------------------------------------------------- build

    def _build(self) -> None:
        if self.datatypes is not None:
            self.polygons = [p for p in self.polygons if p.datatype in self.datatypes]
        drawn = {p.layer for p in self.polygons}

        built, empty = [], []
        for key, settings in self.layers.items():
            lay = self._xml_layers[key]
            if settings.via_merge and key in drawn:
                rings = [xy for p in self.polygons if p.layer == key
                         if (xy := drop_short_edges(p.xy, self.min_edge)) is not None]
                faces = self._via_wall_faces(rings, settings)
            else:
                polys = self.shapes(key)
                if not polys:
                    empty.append(lay.name)
                    continue
                faces = [(2, t) for poly in polys for t in self._polygon_surfaces(poly, settings.z)]
            obj = self._finish_layer(key, settings, faces)
            obj.prio_set(PRIO_DRAWN_DIELECTRIC if lay.type == "dielectric" else PRIO_METAL)
            self.objects[key] = obj
            built.append(lay.name)

        known = set(self.layers) | {d.gds_layer for d in self.stackup.derived_layers} | \
            {o for d in self.stackup.derived_layers for o in d.operands}
        logger.info(f"GDS: built {built} from the stack-up.")
        if empty:
            logger.info(f"GDS: stack-up layers without polygons in this layout (skipped): {empty}")
        unused = sorted(drawn - known)
        if unused:
            logger.info(f"GDS: layout layers not in the stack-up (skipped, e.g. port markers): {unused}")

    # ----------------------------------------------------------- dielectrics

    def build_dielectrics(self, margin: float) -> DielectricStack:
        """Builds the planar dielectric stack (<Dielectrics>, including the air layer on top) as boxes
        around the layout, plus an enclosing air region whose faces are the model's outer boundary.

        Args:
            margin (float): Lateral distance (physical meters) between the layout and the model edge.
        """
        k = self.scale_factor
        x0, y0, x1, y1 = (v / k for v in self.bounds())
        x0, y0, x1, y1 = x0 - margin, y0 - margin, x1 + margin, y1 + margin

        volumes: dict[str, GeoVolume] = {}
        for d in self.stackup.dielectrics:
            material = self._material(d.material)
            if d.boundary_layer is not None:
                polys = self.shapes(d.boundary_layer)
                if not polys:
                    logger.warning(f"GDS: boundary layer {d.boundary_layer} of dielectric '{d.name}' is empty, skipped.")
                    continue
                faces = [t for poly in polys for t in self._polygon_surfaces(poly, d.zmin)]
                dx, dy, dz = np.asarray(self.cs.zax.np, dtype=float) * (d.zmax - d.zmin) * k
                obj = extrude(GeoSurface(faces, name=d.name), dx, dy, dz)
                obj.name = d.name
            else:
                obj = Box((x1 - x0) * k, (y1 - y0) * k, (d.zmax - d.zmin) * k, (x0 * k, y0 * k, d.zmin * k),
                          cs=self.cs, name=d.name)
            volumes[d.name] = obj.set_material(material).prio_set(PRIO_STACK_DIELECTRIC)

        z0 = min(d.zmin for d in self.stackup.dielectrics)
        z1 = max(d.zmax for d in self.stackup.dielectrics)
        region = Box((x1 - x0) * k, (y1 - y0) * k, (z1 - z0) * k, (x0 * k, y0 * k, z0 * k), cs=self.cs,
                     name="OpenRegion").set_material(lib.AIR).background()
        self.dielectrics = DielectricStack(volumes, region)
        return self.dielectrics

    # ----------------------------------------------------------------- ports

    def _layer_by_name(self, name: str) -> GDSLayer:
        for settings in self.layers.values():
            if settings.name == name:
                return settings
        raise KeyError(f"No built stack-up layer named '{name}'. Layers: {[s.name for s in self.layers.values()]}")

    def via_port(self, marker_layer: int, port_number: int, from_layer: str, to_layer: str,
                 impedance: float = 50.0, name: str | None = None) -> Plate:
        """Creates a vertical lumped port from a line-like port marker polygon, as in gds2palace.

        The marker (a thin strip on its own GDS layer) defines where the port runs in the layout plane.
        The port spans the gap between the two named stack-up layers, e.g. from a ground layer up to a
        metal. If the marker lies within a few database units of a parallel edge of either layer (as
        is common), the port is snapped onto that edge.

        The plate carries a LumpedPortAttribute, so it becomes a LumpedPort automatically.

        Args:
            marker_layer (int): GDS layer holding exactly one port marker polygon.
            port_number (int): The port number.
            from_layer, to_layer (str): Names of the stack-up layers the port connects.
            impedance (float): Port reference impedance in Ohm.
            name (str | None): Object name.
        """
        markers = [p for p in self.polygons if p.layer == marker_layer]
        if len(markers) != 1:
            raise ValueError(f"Expected exactly one port marker polygon on layer {marker_layer}, found {len(markers)}.")
        xy = markers[0].xy
        center = xy.mean(axis=0)
        _, _, vt = np.linalg.svd(xy - center)
        along, across = vt[0], vt[1]
        t = (xy - center) @ along
        length, width = float(np.ptp(t)), float(np.ptp((xy - center) @ across))
        if width > 0.05 * length:
            raise ValueError(f"The marker on layer {marker_layer} is not line-like ({length:.3g} x {width:.3g} m); "
                             "via_port expects a thin strip.")
        p0, p1 = center + along * t.min(), center + along * t.max()
        p0, p1 = self._snap_to_edge(p0, p1, [from_layer, to_layer], max(width, 10 * self.library.db_unit))

        f, g = self._layer_by_name(from_layer), self._layer_by_name(to_layer)
        f_top, g_top = f.z + (f.thickness or 0.0), g.z + (g.thickness or 0.0)
        if f_top <= g.z:
            z_from, z_to = f_top, g.z
        elif g_top <= f.z:
            z_from, z_to = f.z, g_top
        else:
            raise ValueError(f"Layers '{from_layer}' and '{to_layer}' overlap in z, there is no gap for a port.")

        origin = self._global(p0, z_from)
        u = self._global(p1, z_from) - origin
        v = self._global(p0, z_to) - origin
        return self._port_plate(origin, u, v, port_number, width=float(np.linalg.norm(p1 - p0)),
                                height=abs(z_to - z_from), direction=v / np.linalg.norm(v),
                                impedance=impedance, name=name)

    def _snap_to_edge(self, p0: np.ndarray, p1: np.ndarray, layer_names: list[str], tol: float):
        """Moves the segment p0-p1 sideways onto a parallel polygon edge of the given layers within tol."""
        d = (p1 - p0) / np.linalg.norm(p1 - p0)
        n = np.array([-d[1], d[0]])
        mid = (p0 + p1) / 2
        best = None
        for key, settings in self.layers.items():
            if settings.name not in layer_names:
                continue
            for poly in self.shapes(key):
                rings = [np.column_stack([poly.xs, poly.ys])] + [np.column_stack([h.xs, h.ys]) for h in poly.holes]
                for ring in rings:
                    for a, b in zip(ring, np.roll(ring, -1, axis=0)):
                        e = b - a
                        le = np.linalg.norm(e)
                        if le == 0 or abs(e[0] * d[1] - e[1] * d[0]) / le > 1e-6:
                            continue  # not parallel
                        offset = float(np.dot(a - mid, n))
                        s = np.dot(mid - a, e) / le ** 2
                        if abs(offset) <= tol and -1e-9 <= s <= 1 + 1e-9 and (best is None or abs(offset) < abs(best)):
                            best = offset
        if best is None:
            return p0, p1
        return p0 + n * best, p1 + n * best


__all__ = ["DielectricStack", "GDSFromXML"]
