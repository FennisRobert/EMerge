
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

from collections import defaultdict

from loguru import logger

from ..._emerge.attributes import FiniteThickness
from ..._emerge.geo import PCBNew, extrude
from ..._emerge.geometry import GeoSurface, GeoVolume
from .gerber import GerberLayer
from .gerber_occ import CurvedGerberLayer
from .excellon import parse_excellon_file


class FileBasedPCB(PCBNew):
    """Adds PCB construction from Gerber and Excellon files."""

    def layer_from_file(self, layer: int, filename: str,
                        res_mm: float = 0.01,
                        n_circ_segments: int = 8,
                        segment_size_mm: float | None = None,
                        min_incl_angle: float = 2.0,
                        min_remove_angle: float = 10.0,
                        simplify: bool = True,
                        curved: bool = False) -> GeoSurface | GeoVolume:
        """Create a layer from a Gerber file.

        Args:
            layer (int): The layer number.
            filename (str): The path to the Gerber file.
            res_mm (float, optional): The geometric resolution in mm (arc chord tolerance and
                minimum edge length during simplification). Defaults to 0.01.
            n_circ_segments (int, optional): The number of segments for round apertures. Defaults to 8.
            segment_size_mm (float | None, optional): Target segment size for round apertures in mm. Defaults to None.
            min_incl_angle (float, optional): The polygon section angle below which points are removed in degrees. Defaults to 2.0
            min_remove_angle (float, optional): The polygon section angle below which close points are considered for removal in degrees. Defaults to 10.0
            simplify (bool | optional): If the geometries should be simplified. Defaults to True.
            curved (bool, optional): Build the layer from exact lines and circular arcs instead of
                polygons. Round pads, track ends and arcs are then modelled exactly and no
                simplification is needed. n_circ_segments, min_incl_angle, min_remove_angle and
                simplify are ignored (except for the few shapes that fall back to polygons). Defaults to False.

        Returns:
            GeoSurface | GeoVolume: The generated surface (or volume when thick_traces=True).
        """
        z = self.z(layer)*self.unit
        if curved:
            gerber = CurvedGerberLayer(filename, res_mm, cs=self.cs)
            surf = gerber.flatten(z)
        else:
            gerber = GerberLayer(filename, res_mm, n_circ_segments, segment_size_mm, cs=self.cs,
                                 min_incl_angle=min_incl_angle, min_remove_angle=min_remove_angle,
                                 simplify=simplify)
            logger.debug(f'Gerber {filename}: {len(gerber.polygons)} polygons, {gerber.n_points} vertices')
            surf = gerber.flatten(z)

        self.xs.extend([x/self.unit for x in gerber.xs])
        self.ys.extend([y/self.unit for y in gerber.ys])

        if self._thick_traces:
            if self.trace_thickness is None:
                raise ValueError('Trace thickness not defined. Make sure to define a trace thickness in the PCB() constructor.')
            dx, dy, dz = self.cs.zax.np*self.trace_thickness
            vol = extrude(surf, dx, dy, dz).prio_set(self.conductor_priority)
            vol.properties += self.trace_material
            return vol

        surf.properties += FiniteThickness(self.trace_thickness) + self.trace_material
        return surf

    def vias_from_file(self, filename: str,
                       layer1: int = 0,
                       layer2: int = -1,
                       Nsections: int = 6,
                       z1: float | None = None,
                       z2: float | None = None) -> None:
        """Add vias from an Excellon file.

        Args:
            filename (str): The path to the Excellon file.
            layer1 (int, optional): The bottom layer index. Defaults to 0.
            layer2 (int, optional): The top layer index. Defaults to -1 (top layer).
            Nsections (int, optional): The number of sections for the vias. Defaults to 6.
            z1, z2: Deprecated and ignored, use layer1 and layer2.
        """
        if z1 is not None or z2 is not None:
            logger.warning('vias_from_file: z1/z2 are ignored, use layer1/layer2 instead.')
        via_buffer = defaultdict(list)
        for hit in parse_excellon_file(filename):
            if hit['radius'] is None:
                logger.warning(f'Drill hit at ({hit["x"]}, {hit["y"]}) mm uses undefined tool {hit["tool"]}; skipped.')
                continue
            rad = hit['radius']*0.001/self.unit
            via_buffer[rad].append((hit['x']*0.001/self.unit, hit['y']*0.001/self.unit))

        for radius, coords in via_buffer.items():
            self.add_vias(*coords, radius=radius, layer1=layer1, layer2=layer2, segments=Nsections)
        logger.debug(f'Excellon {filename}: {sum(len(v) for v in via_buffer.values())} vias in {len(via_buffer)} sizes')
