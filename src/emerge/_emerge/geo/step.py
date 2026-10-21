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
import gmsh
from ..geometry import GeoPoint, GeoEdge, GeoVolume, GeoSurface, GeoObject
from ..selection import FaceSelection
from .shapes import Box
from .operations import unite

from pathlib import Path
import numpy as np
from typing import Callable
from ._step_name_parser import get_volume_names
from loguru import logger
import re


def _select_num(num1: float | None, num2: float | None) -> float:
    if num1 is None and num2 is None:
        return 0.0
    if num2 is None:
        return num1
    if num1 is None:
        return num2
    return max(num1, num2)


class _FaceSliceSelector:
    def __init__(self, face_numbers: list[int], selector: Callable):
        self.numbers: list[str] = face_numbers
        self.selector: Callable = selector

    def __getitem__(self, slice) -> FaceSelection:
        nums = self.numbers.__getitem__(slice)
        if isinstance(nums, int):
            nums = [
                nums,
            ]
        return self.selector(*nums)


class StepVolume(GeoVolume):
    """The StepVoume class extens the EMerge GeoVolume class to add easier
    face selection functionalities based on numbers as face names are not
    imported currently

    Args:
        GeoVolume (_type_): _description_

    Returns:
        _type_: _description_
    """

    @property
    def _face_numbers(self) -> list[int]:
        return sorted([int(name[4:]) for name in self._face_pointers.keys()])

    @property
    def face_slice(self) -> _FaceSliceSelector:
        """A selector that selects faces by slicing the face numbers, e.g. vol.face_slice[2:5]."""
        return _FaceSliceSelector(self._face_numbers, self.faces)

    def faces(self, *numbers: int) -> FaceSelection:
        """Select a set of faces by number

        Args:
            *numbers (int): The face numbers

        Returns:
            FaceSelection: The selected faces
        """
        names = [f"Face{num}" for num in numbers]
        return super().faces(*names)


class STEPNameGenerator:
    def __init__(self, filename: str, string_parser: Callable | None):
        """Generates names for the objects imported from a STEP file.

        Args:
            filename (str): The STEP file
            string_parser (Callable | None): A function applied to every name, None to keep names as they are
        """
        self.filename: str = filename
        self.parser: Callable | None = string_parser
        self.names: list[str] = []
        self.pulled: bool = None
        self.counter: int = 0
        if self.parser is None:
            self.parser = lambda x: x

    def gen_new(self) -> str:
        """Generate a new unique name.

        Returns:
            str: The name "Obj<counter>"
        """
        name = f"Obj{self.counter}"
        self.counter += 1
        return name

    def sanitize(self, name: str) -> str:
        """Replace the Open CASCADE translator name by "OCC".

        Args:
            name (str): The name

        Returns:
            str: The sanitized name
        """
        pattern = r"Open CASCADE STEP translator \d+\.\d+(?:\.\d+)?"
        name = re.sub(pattern, "OCC", name)
        return name

    def pull_from_file(self) -> None:
        """Read the volume names from the STEP file (only done once)."""
        if self.pulled is None:
            self.names = get_volume_names(self.filename)
            if any(self.names):
                self.pulled = True

    def get_from_file(self) -> str:
        """Take the next volume name from the STEP file.

        Returns:
            str: The name, empty if no names are left
        """
        self.pull_from_file()
        if not self.pulled or not self.names:
            return ""
        name = self.names.pop(0)
        return name

    def get_name(self, dim: int, tag: int) -> str:
        """The name of an imported entity: its GMSH name, else the next STEP file name, else a generated name.

        Args:
            dim (int): The entity dimension
            tag (int): The entity tag

        Returns:
            str: The parsed and sanitized name
        """
        name = gmsh.model.getEntityName(dim, tag)
        if not name:
            name = self.get_from_file()
        if not name:
            name = self.gen_new()
        return self.parser(self.sanitize(name))


class STEPItems:
    """STEPItems imports geometries form a STEP file and exposes them to the user."""

    def __init__(
        self,
        name: str,
        filename: str,
        unit: float = 1.0,
        string_parser: Callable = None,
    ):
        """Imports the provided STEP file.
        Specify the unit in case of scaling issues where mm units are not taken into consideration.

        Args:
            name (str): The name of the imported object
            filename (str): The filename
            unit (float, optional): The STEP file size unit. Defaults to 1.0.
            string_parser (Callable, optional): A function applied to every volume name read from the STEP file. Defaults to None.

        Raises:
            FileNotFoundError: If a file does not exist
        """
        self.name: str = name

        stl_path = Path(filename)

        gmsh.option.setNumber("Geometry.OCCScaling", unit)
        gmsh.option.setNumber("Geometry.OCCImportLabels", 1)

        if not stl_path.exists():
            raise FileNotFoundError(f"File with name {stl_path} does not exist.")

        dimtags = gmsh.model.occ.import_shapes(filename, format='step')

        self.points: list[GeoPoint] = []
        self.edges: list[GeoEdge] = []
        self.surfaces: list[GeoSurface] = []
        self.volumes: list[GeoVolume] = []
    
        namegen = STEPNameGenerator(filename, string_parser)

        for dim, tag in dimtags:
            name = namegen.get_name(dim, tag)
            if dim == 0:
                self.points.append(GeoPoint(tag, name=f"{self.name}_{name}"))
            elif dim == 1:
                self.edges.append(GeoEdge(tag, name=f"{self.name}_{name}"))
            elif dim == 2:
                self.surfaces.append(GeoSurface(tag, name=f"{self.name}_{name}"))
            elif dim == 3:
                vol = StepVolume(tag, name=f"{self.name}_{name}")
                vol._set_bb_anchors()
                self.volumes.append(vol)

        gmsh.model.occ.synchronize()

    @property
    def names(self) -> list[str]:
        """A list of all object names

        Returns:
            list[str]: The names of all imported objects
        """
        return [obj.name for obj in self.objects]

    @property
    def dictionary(self) -> dict[str, GeoObject]:
        """A dictionary of all imported objects by name."""
        return {obj.name: obj for obj in self.objects}

    @property
    def objects(self) -> tuple[GeoObject, ...]:
        """Returns a list of all objects in the STEP file

        Returns:
            tuple[GeoObject,...]: All imported points, edges, surfaces and volumes
        """
        return tuple(self.points + self.edges + self.surfaces + self.volumes)

    def __getitem__(self, name: str) -> GeoObject | None:
        return self.dictionary.get(name, None)

    def as_volume(self) -> StepVolume:
        """Returns the 3D volumetric part of the STEP file as a single geometry

        Returns:
            StepVolume: The resultant StepVolume(GeoVolume) object.
        """
        if len(self.volumes) == 1:
            return self.volumes[0]
        vol = unite(*self.volumes)._auto_face_tag()
        vol._set_bb_anchors()
        return vol

    def as_surface(self) -> GeoSurface:
        """Returns the 2D surface part of the STEP file as a single geometry


        Returns:
            GeoSurface: The resultant GeoSurface object
        """
        if len(self.surfaces) == 1:
            return self.surfaces[0]
        return unite(*self.surfaces)

    def as_edge(self) -> GeoEdge:
        """Returns the 1D Edge part of the STEP file as a single geometry

        Returns:
            GeoEdge: The resultant GeoEdge object
        """
        if len(self.edges) == 1:
            return self.edges[1]
        return unite(*self.edges)

    def as_point(self) -> GeoPoint:
        """Returns the 0D Point part of the STEP file as a single geometry

        Returns:
            GeoPoint: The resultant GeoPoint object.
        """
        if len(self.points) == 1:
            return self.points[0]
        return unite(*self.points)

    def enclose_params(
        self,
        margin: float = None,
        x_margins: tuple[float, float] = (None, None),
        y_margins: tuple[float, float] = (None, None),
        z_margins: tuple[float, float] = (None, None),
    ) -> tuple[float, float, float, tuple[float, float, float]]:
        """Compute the size and position of a box enclosing the step model.

        Args:
            margin (float, optional): The margin added on all sides. Defaults to None (0).
            x_margins (tuple[float, float], optional): The (-x, +x) margins, the larger of this and margin is used. Defaults to (None, None).
            y_margins (tuple[float, float], optional): The (-y, +y) margins, the larger of this and margin is used. Defaults to (None, None).
            z_margins (tuple[float, float], optional): The (-z, +z) margins, the larger of this and margin is used. Defaults to (None, None).

        Returns:
            tuple[float, float, float, tuple[float, float, float]]: The width, depth, height and minimum corner of the box
        """
        xminm = _select_num(margin, x_margins[0])
        xmaxm = _select_num(margin, x_margins[1])
        yminm = _select_num(margin, y_margins[0])
        ymaxm = _select_num(margin, y_margins[1])
        zminm = _select_num(margin, z_margins[0])
        zmaxm = _select_num(margin, z_margins[1])

        xmin = 1000000
        xmax = -1000000
        ymin = 1000000
        ymax = -1000000
        zmin = 1000000
        zmax = -1000000

        for obj in self.objects:
            for dim, tag in obj.dimtags:
                x1, y1, z1, x2, y2, z2 = gmsh.model.occ.getBoundingBox(dim, tag)
                xmin = min(xmin, x1)
                xmax = max(xmax, x2)
                ymin = min(ymin, y1)
                ymax = max(ymax, y2)
                zmin = min(zmin, z1)
                zmax = max(zmax, z2)

        width = xmax - xmin + xminm + xmaxm
        depth = ymax - ymin + yminm + ymaxm
        height = zmax - zmin + zminm + zmaxm
        return width, depth, height, (xmin - xminm, ymin - yminm, zmin - zminm)

    def enclose(
        self,
        margin: float = None,
        x_margins: tuple[float, float] = (None, None),
        y_margins: tuple[float, float] = (None, None),
        z_margins: tuple[float, float] = (None, None),
    ) -> Box:
        """Create an enclosing bounding box for the step model.

        Args:
            margin (float, optional): The margin added on all sides. Defaults to None (0).
            x_margins (tuple[float, float], optional): The (-x, +x) margins, the larger of this and margin is used. Defaults to (None, None).
            y_margins (tuple[float, float], optional): The (-y, +y) margins, the larger of this and margin is used. Defaults to (None, None).
            z_margins (tuple[float, float], optional): The (-z, +z) margins, the larger of this and margin is used. Defaults to (None, None).

        Returns:
            Box: The enclosing box (as a background object)
        """

        return Box(
            *self.enclose_params(margin, x_margins, y_margins, z_margins)
        ).background()
