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

from ..cs import CoordinateSystem, GCS, Anchor, argparse_xyz, _parse_vector
from ..geometry import GeoPolygon, GeoVolume, GeoSurface
from emsutil import Material, AIR, PEC
from .shapes import Box, Plate, Cylinder, Alignment, Point
from .polybased import XYPolygon
from .operations import change_coordinate_system, unite, remove
from .pcb_tools.macro import parse_macro
from .pcb_tools.calculator import PCBCalculator
from ..logsettings import DEBUG_COLLECTOR
import numpy as np
from loguru import logger
from typing import Literal, Callable, overload, Generator
from dataclasses import dataclass
from emsutil import Saveable

import math
import gmsh

############################################################
#                        EXCEPTIONS                        #
############################################################


class RouteException(Exception):
    pass


############################################################
#                         CONSTANTS                        #
############################################################

SIZE_NAMES = Literal[
    "0402", "0603", "1005", "1608", "2012", "3216", "3225", "4532", "5025", "6332"
]

def _str_to_size(code: str) -> tuple[float, float]:
    return (float(code[:2]) * 0.05, float(code[2:]) * 0.1)

_SMD_SIZE_DICT = {
    x: (float(x[:2]) * 0.05, float(x[2:]) * 0.1)
    for x in [
        "0402",
        "0603",
        "0805",
        "1005",
        "1608",
        "1210",
        "2012",
        "3216",
        "3225",
        "4532",
        "5025",
        "6332",
    ]
}


class PCB_MANAGER:
    """This manager class ensures that unique names are provided to instances of PCB objects."""

    def __init__(self):
        self.names: set[str] = set()
        self.unit: float = 1.0
        self.cs: CoordinateSystem = GCS

    def clear(self) -> None:
        self.names = set()

    def __call__(self, name: str | None, classname: str | None = None) -> str:
        if name is None:
            return self(classname)

        if name not in self.names:
            self.names.add(name)
            return name
        for i in range(1_000_000):
            newname = f"{name}_{i}"
            if newname not in self.names:
                self.names.add(newname)
                return newname


_PCB_MANAGER = PCB_MANAGER()

############################################################
#                         FUNCTIONS                        #
############################################################


def approx(a: float, b: float) -> bool:
    """Tests if a and b are approximately equal/close (10e-8 margin)

    Args:
        a (float): num 1
        b (float): num 2

    Returns:
        bool: if they are equal
    """
    return abs(a - b) < 1e-12


def normalize(vector: np.ndarray) -> np.ndarray:
    """Convenience function for normalizing vectors by L2 norm

    Args:
        vector (np.ndarray): _description_

    Returns:
        np.ndarray: _description_
    """
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm


def _rot_mat(angle: float) -> np.ndarray:
    """Returns a 2D rotation matrix given an angle in degrees

    Args:
        angle (float): The angle in degrees

    Returns:
        np.ndarray: The rotation matrix
    """
    ang = -angle * np.pi / 180
    return np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])


############################################################
#                          CLASSES                         #
############################################################

@dataclass
class Via:
    x: float
    y: float
    z1: float
    z2: float
    radius: float
    segments: int

    @property
    def pnt(self) -> Anchor:
        unit = _PCB_MANAGER.unit
        cs = _PCB_MANAGER.cs
        x, y, z = _PCB_MANAGER.cs.in_global_cs(
            self.x * unit, self.y * unit, self.z1 * unit
        )
        return Anchor((x, y, z), cs.gx, cs.gy, cs.gz)


class RouteElement:
    _DEFNAME: str = "RouteElement"
    _UNIT: float = 1.0

    def __init__(self):
        self.width: float = None
        self.x: float = None
        self.y: float = None
        self.z: float = None
        self.direction: np.ndarray = None
        self.dirright: np.ndarray = None
        self.rcutnext: bool = False
        self.rcutprev: bool = False
        self.lcutnext: bool = False
        self.lcutprev: bool = False

    def _back(self, dist: float) -> None:
        """Moves the start of this element back in its primary direction"""
        self.x = self.x - self.direction[0] * dist
        self.y = self.y - self.direction[1] * dist

    @property
    def pnt(self) -> Anchor:
        unit = _PCB_MANAGER.unit
        cs = _PCB_MANAGER.cs
        ux = np.array(
            _PCB_MANAGER.cs.in_local_basis(self.direction[0], self.direction[1], 0.0)
        )
        uy = np.array(
            _PCB_MANAGER.cs.in_local_basis(-self.dirright[0], -self.dirright[1], 0.0)
        )
        uz = np.array(_PCB_MANAGER.cs.in_local_basis(0.0, 0.0, 1.0))
        gx, gy, gz = cs.in_global_cs(
            self.x * _PCB_MANAGER.unit, self.y * _PCB_MANAGER.unit, self.z
        )
        return Anchor((gx, gy, gz), ux, uy, uz)

    @property
    def xy(self) -> tuple[float, float]:
        return self.x, self.y

    @property
    def nr(self) -> tuple[float, float]:
        return (
            self.x + self.dirright[0] * self.width / 2,
            self.y + self.dirright[1] * self.width / 2,
        )

    @property
    def nl(self) -> tuple[float, float]:
        return (
            self.x - self.dirright[0] * self.width / 2,
            self.y - self.dirright[1] * self.width / 2,
        )

    @property
    def right(self) -> list[tuple[float, float]]:
        raise NotImplementedError()

    @property
    def left(self) -> list[tuple[float, float]]:
        raise NotImplementedError()

    def __eq__(self, other: RouteElement) -> bool:
        return (
            approx(self.x, other.x)
            and approx(self.y, other.y)
            and (1 - abs(np.sum(self.direction * other.direction))) < 1e-12
        )


class StripLine(RouteElement):
    _DEFNAME: str = "StripLine"

    def __init__(
        self,
        x: float | Anchor,
        y: float,
        z: float,
        width: float,
        direction: tuple[float, float],
    ):
        x, y, z = argparse_xyz(x, y, z)
        self.x = x
        self.y = y
        self.z = z
        self.width = width
        self.direction = normalize(np.array(direction))
        self.dirright = np.array([self.direction[1], -self.direction[0]])
        self.rcutnext: bool = False
        self.rcutprev: bool = False
        self.lcutnext: bool = False
        self.lcutprev: bool = False

    def __str__(self) -> str:
        return f"StripLine[{self.x},{self.y},w={self.width},d=({self.direction})]"

    @property
    def right(self) -> list[tuple[float, float]]:
        return [
            (
                self.x + self.width / 2 * self.dirright[0],
                self.y + self.width / 2 * self.dirright[1],
            )
        ]

    @property
    def left(self) -> list[tuple[float, float]]:
        return [
            (
                self.x - self.width / 2 * self.dirright[0],
                self.y - self.width / 2 * self.dirright[1],
            )
        ]


class StripTurn(RouteElement):
    _DEFNAME: str = "StripTurn"

    def __init__(
        self,
        x: float | Anchor,
        y: float,
        z: float,
        width: float,
        direction: tuple[float, float],
        angle: float,
        corner_type: str = "square",
        champher_distance: float | None = None,
        dsratio: float = 1.0,
    ):
        x, y, z = argparse_xyz(x, y, z)
        self.xold: float = x
        self.yold: float = y
        self.width: float = width
        self.old_direction: np.ndarray = normalize(np.array(direction))
        self.direction: np.ndarray = _rot_mat(angle) @ self.old_direction
        self.angle: float = angle
        self.corner_type: str = corner_type
        self.dirright: np.ndarray = np.array(
            [self.old_direction[1], -self.old_direction[0]]
        )
        self.rcutnext: bool = False
        self.rcutprev: bool = False
        self.lcutnext: bool = False
        self.lcutprev: bool = False

        hang = np.abs(angle * np.pi / 180 * 0.5)
        if champher_distance is None:
            self.champher_distance: float = (
                dsratio * self.width / (np.cos(hang) * np.cos(np.pi / 2 - hang))
            )
            if self.champher_distance > self.width * np.tan(hang):
                if self.angle > 0:
                    self.lcutprev = True
                else:
                    self.rcutprev = True
        else:
            self.champher_distance: float = champher_distance

        turnvec = _rot_mat(angle) @ self.dirright * self.width / 2

        if angle > 0:
            self.x = x + width / 2 * self.dirright[0] - turnvec[0]
            self.y = y + width / 2 * self.dirright[1] - turnvec[1]
        else:
            self.x = x - width / 2 * self.dirright[0] + turnvec[0]
            self.y = y - width / 2 * self.dirright[1] + turnvec[1]

    def __str__(self) -> str:
        return f"StripTurn[{self.x},{self.y},w={self.width},d=({self.direction})]"

    @property
    def right(self) -> list[tuple[float, float]]:
        if self.angle > 0:
            return []

        # turning left
        xl = self.xold - self.width / 2 * self.dirright[0]
        yl = self.yold - self.width / 2 * self.dirright[1]
        xr = self.xold + self.width / 2 * self.dirright[0]
        yr = self.yold + self.width / 2 * self.dirright[1]

        dist = self.width * np.tan(np.abs(self.angle) / 2 * np.pi / 180)

        dcorner = self.width * (_rot_mat(self.angle) @ self.dirright)

        xend = xl + dcorner[0]
        yend = yl + dcorner[1]

        out = [(xend, yend)]

        if self.corner_type in ("champher","miter"):
            dist = dist - self.champher_distance

        if dist == 0:
            return out

        x1 = xr + dist * self.old_direction[0]
        y1 = yr + dist * self.old_direction[1]

        if self.corner_type == "square":
            return [(x1, y1), (xend, yend)]
        if self.corner_type in ("champher","miter"):
            x2 = xend - dist * self.direction[0]
            y2 = yend - dist * self.direction[1]
            if self.rcutprev:
                return [(x1, y1), (x2, y2)]
            return [(x1, y1), (x2, y2), (xend, yend)]
        else:
            raise RouteException(
                f"Trying to route a StripTurn with an unknown corner type: {self.corner_type}"
            )

    @property
    def left(self) -> list[tuple[float, float]]:
        if self.angle < 0:
            return []

        # turning right
        xl = self.xold - self.width / 2 * self.dirright[0]
        yl = self.yold - self.width / 2 * self.dirright[1]
        xr = self.xold + self.width / 2 * self.dirright[0]
        yr = self.yold + self.width / 2 * self.dirright[1]

        dist = self.width * np.tan(np.abs(self.angle) / 2 * np.pi / 180)

        dcorner = self.width * (_rot_mat(self.angle) @ -self.dirright)

        xend = xr + dcorner[0]
        yend = yr + dcorner[1]

        out = [(xend, yend)]

        if self.corner_type in ("champher","miter"):
            dist = dist - self.champher_distance

        if dist == 0:
            return out

        x1 = xl + dist * self.old_direction[0]
        y1 = yl + dist * self.old_direction[1]

        if self.corner_type == "square":
            return [(x1, y1), (xend, yend)]
        if self.corner_type in ("champher","miter"):
            x2 = xend - dist * self.direction[0]
            y2 = yend - dist * self.direction[1]
            if self.lcutprev:
                return [(x1, y1), (x2, y2)]
            return [(x1, y1), (x2, y2), (xend, yend)]
        else:
            raise RouteException(
                f"Trying to route a StripTurn with an unknown corner type: {self.corner_type}"
            )


class StripCurve(StripTurn):
    def __init__(
        self,
        x: float | Anchor,
        y: float,
        z: float,
        width: float,
        direction: tuple[float, float],
        angle: float,
        radius: float,
        dang: float = 10.0,
    ):
        x, y, z = argparse_xyz(x, y, z)
        self.xold: float = x
        self.yold: float = y
        self.width: float = width
        self.old_direction: np.ndarray = normalize(np.array(direction))
        self.direction: np.ndarray = _rot_mat(angle) @ self.old_direction
        self.angle: float = angle
        self.radius: float = radius
        self.dirright: np.ndarray = np.array(
            [self.old_direction[1], -self.old_direction[0]]
        )
        self.dang: float = dang
        self.rcutnext: bool = False
        self.rcutprev: bool = False
        self.lcutnext: bool = False
        self.lcutprev: bool = False

        angd = abs(angle * np.pi / 180)
        self.start = np.array([x, y])
        self.circ_origin = self.start + radius * np.sign(angle) * self.dirright

        self._xhat = -self.dirright * np.sign(angle)
        self._yhat = self.old_direction

        self.end = self.circ_origin + radius * (
            self._xhat * np.cos(angd) + self._yhat * np.sin(angd)
        )
        self.x, self.y = self.end

    def __str__(self) -> str:
        return f"StripCurve[{self.x},{self.y},w={self.width},d=({self.direction})]"

    @property
    def right(self) -> list[tuple[float, float]]:
        points: list[tuple[float, float]] = []
        Npts = int(np.ceil(abs(self.angle / self.dang)))
        R = self.radius - np.sign(self.angle) * self.width / 2
        for i in range(Npts):
            ang = abs((i + 1) / Npts * self.angle * np.pi / 180)
            pnew = self.circ_origin + R * (
                self._xhat * np.cos(ang) + self._yhat * np.sin(ang)
            )
            points.append((pnew[0], pnew[1]))

        return points

    @property
    def left(self) -> list[tuple[float, float]]:
        points: list[tuple[float, float]] = []

        Npts = int(np.ceil(abs(self.angle / self.dang)))
        R = self.radius + np.sign(self.angle) * self.width / 2
        for i in range(Npts):
            ang = abs((i + 1) / Npts * self.angle * np.pi / 180)
            pnew = self.circ_origin + R * (
                self._xhat * np.cos(ang) + self._yhat * np.sin(ang)
            )
            points.append((pnew[0], pnew[1]))

        return points


class PCBPoly:
    _DEFNAME: str = "Poly"

    def __init__(
        self,
        xs: list[float],
        ys: list[float],
        z: float = 0,
        material: Material = PEC,
        name: str | None = None,
    ):
        self.xs: list[float] = xs
        self.ys: list[float] = ys
        self.z: float = z
        self.material: Material = material
        self.name: str = _PCB_MANAGER(name, self._DEFNAME)

    @property
    def xys(self) -> list[tuple[float, float]]:
        return list([(x, y) for x, y in zip(self.xs, self.ys)])

    def segment(self, index: int) -> StripLine:
        N = len(self.xs)
        x1 = self.xs[index % N]
        x2 = self.xs[(index + 1) % N]
        y1 = self.ys[index % N]
        y2 = self.ys[(index + 1) % N]

        W = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** (0.5)
        wdir = ((y2 - y1) / W, -(x2 - x1) / W)

        return StripLine((x2 + x1) / 2, (y1 + y2) / 2, self.z, W, wdir)

    @staticmethod
    def circle(
        x: float, y: float, radius: float, z: float, NSegments: int = 12
    ) -> PCBPoly:
        angles = np.linspace(0, 2 * np.pi, NSegments, endpoint=False)
        xs = list(x + radius * np.cos(angles))
        ys = list(y + radius * np.sin(angles))
        return PCBPoly(xs, ys, z)


############################################################
#                    THE STRIP PATH CLASS                  #
############################################################


class StripPath:
    _DEFNAME: str = "Path"

    def __init__(self, pcb: PCB, name: str | None = None):
        self.pcb: PCB = pcb
        self.path: list[RouteElement] = []
        self.z: float = 0
        self.name: str = _PCB_MANAGER(name, self._DEFNAME)
        self._consume: float = 0

    def iter_right(
        self,
    ) -> Generator[tuple[RouteElement, RouteElement, RouteElement], None, None]:
        N = len(self.path)
        for i in range(N):
            yield self.path[(i - 1) % N], self.path[i], self.path[(i + 1) % N]

    def iter_left(
        self,
    ) -> Generator[tuple[RouteElement, RouteElement, RouteElement], None, None]:
        N = len(self.path)
        for i in range(N):
            yield self.path[(i - 1) % N], self.path[i], self.path[(i + 1) % N]

    def _has(self, element: RouteElement) -> bool:
        if element in self.path:
            return True
        return False

    @property
    def xs(self) -> list[float]:
        return [elem.x for elem in self.path]

    @property
    def ys(self) -> list[float]:
        return [elem.y for elem in self.path]

    @property
    def start(self) -> RouteElement:
        """The start of the stripline."""
        return self.path[0]

    @property
    def end(self) -> RouteElement:
        """The end of the stripline"""
        return self.path[-1]

    @property
    def bck(self) -> StripPath:
        """Proceed one half stripline distance back."""
        w = self.end.width
        self.end._back(w / 2)
        return self

    @property
    def skip(self) -> StripPath:
        """Remove half the stripline distance from the next element

        Returns:
            StripPath: _description_
        """
        w = self.end.width / 2
        self._consume = w
        return self

    def _cons(self, dist: float) -> float:
        dist = dist - self._consume
        self._consume = 0
        return dist

    def _check_loops(self) -> None:
        if self.path[0] == self.path[-1]:
            raise RouteException(
                "Loops are currently not supported. To fix this problem, implement a single .cut() call before a .straight() call to break the loop."
            )
        return None

    def init(
        self,
        x: float | Anchor,
        y: float,
        width: float,
        direction: tuple[float, float],
        z: float = 0,
    ) -> StripPath:
        """Initializes the StripPath object for routing."""
        x, y, z = argparse_xyz(x, y, z)
        self.path.append(StripLine(x, y, z, width, direction))
        self.z = z
        return self

    def _add_element(self, element: RouteElement) -> StripPath:
        """Adds the provided RouteElement to the path."""
        if len(self.path) == 1 and self.path[0] == element:
            self.path.pop(0)
            self.path.append(element)
            return self
        self.path.append(element)
        self._check_loops()
        return self

    def straight(
        self, distance: float, width: float | None = None, dx: float = 0, dy: float = 0
    ) -> StripPath:
        """Add A straight section to the stripline.

        Adds a straight section with a length determined by "distance". Optionally, a
        different "width" can be provided. The start of the straight section will be
        at the end of the last section. The optional dx, dy arguments can be used to offset
        the starting coordinate of the straight segment.

        Args:
            distance (float): The length of the stripline
            width (float, optional): The width of the stripline. Defaults to None.
            dx (float, optional): An x-direction offset. Defaults to 0.
            dy (float, optional): A y-direction offset. Defaults to 0.

        Returns:
            StripPath: The current StripPath object.
        """

        x = self.end.x + dx
        y = self.end.y + dy

        distance = self._cons(distance)
        dx_2, dy_2 = self.end.direction
        x1 = x + distance * dx_2
        y1 = y + distance * dy_2

        if width is not None:
            if width != self.end.width:
                self._add_element(StripLine(x, y, self.z, width, (dx_2, dy_2)))

        self._add_element(StripLine(x1, y1, self.z, self.end.width, (dx_2, dy_2)))
        return self

    def taper(self, distance: float, width: float) -> StripPath:
        """Add A taper section to the stripline.

        Adds a taper section with a length determined by "distance". Optionally, a
        different "width" can be provided. The start of the straight section will be
        at the end of the last section. The optional dx, dy arguments can be used to offset
        the starting coordinate of the straight segment.

        Args:
            distance (float): The length of the stripline
            width (float, optional): The width of the stripline. Defaults to None.
            dx (float, optional): An x-direction offset. Defaults to 0.
            dy (float, optional): A y-direction offset. Defaults to 0.

        Returns:
            StripPath: The current StripPath object.
        """

        x = self.end.x
        y = self.end.y

        distance = self._cons(distance)
        dx_2, dy_2 = self.end.direction
        x1 = x + distance * dx_2
        y1 = y + distance * dy_2

        self._add_element(StripLine(x1, y1, self.z, width, (dx_2, dy_2)))

        return self

    def turn(
        self,
        angle: float,
        width: float | None = None,
        corner_type: Literal["champher", "square","miter"] = "square",
        dsratio: float = 0.7,
    ) -> StripPath:
        """Adds a turn to the strip path.

        The angle is specified in degrees. The width of the turn will be the same as the last segment.
        optionally, a different width may be provided.
        By default, all corners will be cut using the "square" type. Other options are not yet provided.

        Args:
            angle (float): The turning angle
            width (float, optional): The stripline width. Defaults to None.
            corner_type (str, optional): The corner type. Defaults to 'champher'.
            dsratio: (float, optional): The chamfer distance expressed as the proportional of the corner diagonal.

        Returns:
            StripPath: The current StripPath object
        """
        x, y = self.end.x, self.end.y
        dx, dy = self.end.direction
        if abs(angle) <= 20:
            corner_type = "square"
            logger.warning(
                "Small turn detected, defaulting to rectangular corners because chamfers would add to much detail."
            )
        if width is not None:
            if width != self.end.width:
                self._add_element(StripLine(x, y, self.z, width, (dx, dy)))
        else:
            width = self.end.width
        self._add_element(
            StripTurn(
                x, y, self.z, width, (dx, dy), angle, corner_type, dsratio=dsratio
            )
        )
        return self

    def pturn(
        self,
        angle: float,
        corner_type: Literal["champher", "square"] = "square",
        dsratio: float = 0.7,
    ) -> StripPath:
        """Adds a turn to the strip path.

        The angle is specified in degrees. The width of the turn will be the same as the last segment.
        optionally, a different width may be provided.
        By default, all corners will be cut using the "square" type. Other options are not yet provided.

        Args:
            angle (float): The turning angle
            width (float, optional): The stripline width. Defaults to None.
            corner_type (str, optional): The corner type. Defaults to 'champher'.
            dsratio: (float, optional): The chamfer distance expressed as the proportional of the corner diagonal.

        Returns:
            StripPath: The current StripPath object
        """
        w = self.end.width
        dist = abs(w * np.tan(angle * np.pi / 360))
        self.end._back(dist)
        obj = self.turn(angle, corner_type=corner_type, dsratio=dsratio)
        obj._consume = dist
        return obj

    def curve(
        self, angle: float, radius: float, width: float | None = None, dang: float = 10
    ) -> StripPath:
        """Adds a bend to the strip path.

        The angle is specified in degrees. The width of the turn will be the same as the last segment.
        optionally, a different width may be provided.
        By default, all corners will be cut using the "champher" type. Other options are not yet provided.

        Args:
            angle (float): The turning angle
            width (float, optional): The stripline width. Defaults to None.

        Returns:
            StripPath: The current StripPath object
        """
        if angle == 0:
            logger.trace("Zero angle turn, passing action")
            return self
        x, y = self.end.x, self.end.y
        dx, dy = self.end.direction
        if width is not None:
            if width != self.end.width:
                self._add_element(StripLine(x, y, self.z, width, (dx, dy)))
        else:
            width = self.end.width
        self._add_element(
            StripCurve(x, y, self.z, width, (dx, dy), angle, radius, dang=dang)
        )
        return self

    def store(self, name: str) -> StripPath:
        """Store the current x,y coordinate labeled in the PCB object.

        The stored coordinate can be accessed by calling the .load() method on the PCBRouter class.

        Args:
            name (str): The coordinate label

        Returns:
            StripPath: The current StripPath object.
        """
        self.pcb.stored_striplines[str(name)] = self.end
        return self

    def __getitem__(self, name: str) -> StripPath:
        return self.store(name)

    def split(
        self,
        direction: tuple[float, float] | None = None,
        angle: float | None = None,
        width: float | None = None,
    ) -> StripPath:
        """Split the current path in N new paths given by a new departure direction

        Args:
            directions (list[tuple[float, float]]): a list of directions example: [(1,0),(-1,0)]
            widths (list[float], optional): The width for each new path. Defaults to None.

        Returns:
            list[StripPath]: A list of new StripPath objects
        """
        if width is None:
            width = self.end.width
        if direction is None:
            direction = self.end.direction

        x = self.end.x
        y = self.end.y
        z = self.z
        if angle is not None:
            direction = _rot_mat(angle) @ np.array(direction)
        paths = self.pcb.new(x, y, width, direction, z)
        self.pcb._checkpoint.append(self)
        return paths

    def lumped_element(
        self, impedance_function: Callable, size: SIZE_NAMES | tuple[float, float]
    ) -> StripPath:
        """Adds a lumped element to the PCB.

        The first argument should be the impedance function as function of frequency. For a capacitor this would be:
        Z(f) = 1/(j2πfC).
        The second argument specifies the size of the element (length x width) as a tuple or it can be a string for a
        package. For example "0402". The size of the lumped component does not inlcude the footprint.

        For example a 0602 pacakge has timensions: length=0.6mm, width=0.3mm. The actual length of the component
        not overlapping with the solder pad is 0.3mm (always half) so the component size added is 0.3mm x 0.3mm.

        After creation, the trace continues after the lumped component.

        You can add the components to your model as following:

        >>> lumped_elements = pcb.lumped_elments
        for le in lumped_elements:
            model.mw.bc.LumpedElement(le)

        The impedance function and geometry is automatically passed on with the lumped element added.

        Args:
            impedance_function (Callable): A function that computes the component impedance as a function of frequency.
            size (SizeNames | tuple): The dimensions of the lumped element on PCB.

        Returns:
            StripPath: The same strip path object
        """
        if size in _SMD_SIZE_DICT:
            length, width = _SMD_SIZE_DICT[size]
            length = length * 0.001 / self.pcb.unit * 2.54
            width = width * 0.001 / self.pcb.unit * 2.54
        elif isinstance(size, str):
            length, width = _str_to_size(size)
            length = length * 0.001 / self.pcb.unit * 2.54
            width = width * 0.001 / self.pcb.unit * 2.54
        else:
            length, width = size
        
        dx, dy = self.end.direction
        x, y = self.end.x, self.end.y
        rx, ry = self.end.dirright
        wh = width / 2
        xs = (
            np.array(
                [
                    x + rx * wh,
                    x + rx * wh + length * dx,
                    x - rx * wh + length * dx,
                    x - rx * wh,
                ]
            )
            * self.pcb.unit
        )
        ys = (
            np.array(
                [
                    y + ry * wh,
                    y + ry * wh + length * dy,
                    y - ry * wh + length * dy,
                    y - ry * wh,
                ]
            )
            * self.pcb.unit
        )
        poly = XYPolygon(xs, ys)

        self.pcb._lumped_element(poly, impedance_function, width, length)
        return self.pcb.new(
            x + dx * length, y + dy * length, self.end.width, self.end.direction, self.z
        )

    def lumped_element_from_sp(self, filename: str, size: SIZE_NAMES | tuple[float, float], z_gnd: float | None = None) -> GeoVolume:
        """Adds a lumped element to the PCB.

        The first argument should be the impedance function as function of frequency. For a capacitor this would be:
        Z(f) = 1/(j2πfC).
        The second argument specifies the size of the element (length x width) as a tuple or it can be a string for a
        package. For example "0402". The size of the lumped component does not inlcude the footprint.

        For example a 0602 pacakge has timensions: length=0.6mm, width=0.3mm. The actual length of the component
        not overlapping with the solder pad is 0.3mm (always half) so the component size added is 0.3mm x 0.3mm.

        After creation, the trace continues after the lumped component.

        You can add the components to your model as following:

        >>> lumped_elements = pcb.lumped_elments
        for le in lumped_elements:
            model.mw.bc.LumpedElement(le)

        The impedance function and geometry is automatically passed on with the lumped element added.

        Args:
            impedance_function (Callable): A function that computes the component impedance as a function of frequency.
            size (SizeNames | tuple): The dimensions of the lumped element on PCB.

        Returns:
            StripPath: The same strip path object
        """
        from ...auxilliary.touchstone import TouchstoneData

        if size in _SMD_SIZE_DICT:
            length, width = _SMD_SIZE_DICT[size]
            length = length * 0.001 / self.pcb.unit * 2.54
            width = width * 0.001 / self.pcb.unit * 2.54
        elif isinstance(size, str):
            length, width = _str_to_size(size)
            length = length * 0.001 / self.pcb.unit * 2.54
            width = width * 0.001 / self.pcb.unit * 2.54
        else:
            length, width = size
        
        if z_gnd is None:
            z_gnd = self.pcb.z(0)

        td = TouchstoneData(filename, )
        dx, dy = self.end.direction
        x, y = self.end.x, self.end.y
        rx, ry = self.end.dirright
        wh = width / 2
        xs = (
            np.array(
                [
                    x + rx * wh,
                    x + rx * wh + length * dx,
                    x - rx * wh + length * dx,
                    x - rx * wh,
                ]
            )
            * self.pcb.unit
        )
        ys = (
            np.array(
                [
                    y + ry * wh,
                    y + ry * wh + length * dy,
                    y - ry * wh + length * dy,
                    y - ry * wh,
                ]
            )
            * self.pcb.unit
        )
        poly = XYPolygon(xs, ys)

        if td.n_ports == 1:
            
            self.pcb._lumped_element(poly, td.z_series(), width, length)
            return self.pcb.new(
                x + dx * length, y + dy * length, self.end.width, self.end.direction, self.z
            )
        elif td.n_ports == 2 and td.is_singular():
            self.pcb._lumped_element(poly, td.z_series(), width, length)
            return self.pcb.new(
                x + dx * length, y + dy * length, self.end.width, self.end.direction, self.z
            )
        else:
            Rx = rx*wh*self.pcb.unit
            Ry = ry*wh*self.pcb.unit
            x0l = x*self.pcb.unit - Rx
            y0l = y*self.pcb.unit - Ry
            dlx = length*dx*self.pcb.unit
            dly = length*dy*self.pcb.unit
            dz = np.abs(self.z-z_gnd)*self.pcb.unit
            poly_shunt1 = Plate((x0l, y0l, self.z), (Rx*2,Ry*2,0),(0,0,-(self.z-z_gnd)*self.pcb.unit))
            poly_shunt2 = Plate((x0l+dlx, y0l+dly, self.z), (Rx*2,Ry*2,0),(0,0,-(self.z-z_gnd)*self.pcb.unit))

            self.pcb._lumped_element(poly, td.z_series_pi(), width, length)
            self.pcb._lumped_element(poly_shunt1, td.z_shunt_pi_1(), width, dz)
            self.pcb._lumped_element(poly_shunt2, td.z_shunt_pi_2(), width, dz)
            return self.pcb.new(
                x + dx * length, y + dy * length, self.end.width, self.end.direction, self.z
            )
    
    def cut(self) -> StripPath:
        """Split the current path in N new paths given by a new departure direction

        Args:
            directions (list[tuple[float, float]]): a list of directions example: [(1,0),(-1,0)]
            widths (list[float], optional): The width for each new path. Defaults to None.

        Returns:
            list[StripPath]: A list of new StripPath objects
        """
        width = self.end.width
        direction = self.end.direction
        x = self.end.x
        y = self.end.y
        z = self.z
        paths = self.pcb.new(x, y, width, direction, z)
        return paths

    def stub(
        self,
        direction: tuple[float, float],
        width: float,
        length: float,
        mirror: bool = False,
    ) -> StripPath:
        """Add a single rectangular strip line section at the current coordinate"""
        self.pcb.new(self.end.x, self.end.y, width, direction, self.z).straight(length)
        if mirror:
            self.pcb.new(
                self.end.x, self.end.y, width, (-direction[0], -direction[1]), self.z
            ).straight(length)
        return self

    def merge(self) -> StripPath:
        """Continue at the last point where .split() is called"""
        if len(self.pcb._checkpoint) == 0:
            raise RouteException(
                "No checkpoint known. Make sure to call .check() first"
            )
        return self.pcb._checkpoint.pop(-1)

    def via(
        self,
        znew: float,
        radius: float,
        proceed: bool = True,
        direction: tuple[float, float] | None = None,
        width: float | None = None,
        extra: float | None = None,
        segments: int = 6,
        hole_radius: float | None = None,
        hole_skip_layers: list[int] | None = None,
        reverse: float = 0.0,
    ) -> StripPath:
        """Adds a via to the circuit

        If proceed is set to True, a new StripPath will be started. The width and direction properties
        will be inherited from the current one if not specified.
        The extra parameter specifies how much extra stripline is added beyond the current point and before
        the new segment to include the via. If not specifies it defaults to width/2.

        Args:
            znew (float): The new Z-height for the stripline
            radius (float): The via radius
            proceed (bool, optional): Wether to continue with a new trace. Defaults to True.
            direction (tuple[float, float], optional): The new direction. Defaults to None.
            width (float, optional): The new width. Defaults to None.
            extra (float, optional): How much extra stripline to add around the via. Defaults to None.
            segments (int, optional): The number of via polygon sections. Defaults to 6.
            hole_radius (float | None, optional): The via hole radius for the Ground layers and polies. Defaults to None.
            hole_skip_layers (list[int] | None, optional): A list of layer numbers where the via hole should not be created. Defaults to None.
            reverse (float, optional): A distance to place the via back in the direction the stripline is coming from. Defaults to 0.0.

        Returns:
            StripPath: The new StripPath object
        """

        if extra is None:
            extra = self.end.width / 2

        dx, dy = self.end.direction

        x, y = self.end.x, self.end.y
        x = x - dx * reverse
        y = y - dy * reverse
        z1 = self.z
        z2 = znew
        if extra > 0:
            self.straight(extra)

        self.pcb.vias.append(Via(x, y, z1, z2, radius, segments))

        # Create via hole objects
        if hole_radius is not None:
            self.pcb._via_hole(x, y, hole_radius, z1, z2, hole_skip_layers)

        if proceed:
            if width is None:
                width = self.end.width
            if direction is None:
                direction = self.end.direction
            dx = direction[0] * extra
            dy = direction[1] * extra
            return self.pcb.new(x - dx, y - dy, width, direction, z2)
        return self

    def short(
        self, ground_layer: int = 0, radius: float | None = None, reverse: float = 0
    ) -> StripPath:
        """Create a short circuit via at the current location.

        Args:
            ground_layer (int, optional): The layer number to short to. Defaults to 0.
            radius (float | None, optional): The via radius. Defaults to None.
            reverse (float, optional): A displacement distance in the reverse direction. Defaults to 0.

        Returns:
            StripPath: Returns the same strip path object.
        """
        if radius is None:
            radius = self.end.width / 3
        self.via(self.pcb.z(ground_layer), radius, False, reverse=reverse)
        return self

    def jump(
        self,
        dx: float | None = None,
        dy: float | None = None,
        width: float | None = None,
        direction: tuple[float, float] | None = None,
        gap: float | None = None,
        side: Literal["left", "right"] | None = None,
        reverse: float | None = None,
    ) -> StripPath:
        """Add an unconnected jump to the currenet stripline.

        The last stripline path will be terminated and a new one will be started based on the
        displacement provided by dx and dy. The new path will proceed in the same direction or
        another one based ont the "direction" argument.
        An alternative one can define a "gap", "side" and "reverse" argument. The stripline
        will make a lateral jump ensuring a gap between the current and new line. The direction
        of the jump is either "left" or "right" as seen from the direction of the stripline.
        The reverse argument is a distance by which the stripline moves back.

        Args:
            dx (float, optional): The jumps dx distance. Defaults to None.
            dy (float, optional): The jumps dy distance. Defaults to None.
            width (float, optional): The new stripline width. Defaults to None.
            direction (tuple[float, float], optional): The new stripline direction. Defaults to None.
            gap (float, optional): The gap between the current and next stripline. Defaults to None.
            side (Literal[left, right], optional): The lateral jump direction. Defaults to None.
            reverse (float, optional): How much to move back if a lateral jump is made. Defaults to None.

        Example:
        The current example would yield a coupled line filter parallel jump.
        >>> StripPath.jump(gap=1, side="left", reverse=quarter_wavelength).straight(...)

        Returns:
            StripPath: The new StripPath object
        """
        if width is None:
            width = self.end.width
        if direction is None:
            direction = self.end.direction
        else:
            direction = np.array(direction)

        ending = self.end

        if gap is not None and side is not None and reverse is not None:
            Q = 1
            if side == "left":
                Q = -1
            x = (
                ending.x
                - reverse * ending.direction[0]
                + Q * ending.dirright[0] * (width / 2 + ending.width / 2 + gap)
            )
            y = (
                ending.y
                - reverse * ending.direction[1]
                + Q * ending.dirright[1] * (width / 2 + ending.width / 2 + gap)
            )
        else:
            x = ending.x + dx
            y = ending.y + dy
        return self.pcb.new(x, y, width, direction, z=self.z)

    def move(self,
                 dx: float | None = None,
                 dy: float | None = None,
                 x: float | None = None,
                 y: float | None = None,
                 direction: tuple[float, float] | None = None,
                 width: float | None = None) -> StripPath:
            """Move the routing cursur to any new desired location
    
            Args:
                dx (float | None, optional): A horizontal displacement. Defaults to None.
                dy (float | None, optional): A vertical displacement. Defaults to None.
                x (float | None, optional): The new x-coordinate. Defaults to None.
                y (float | None, optional): The new y-coordinate. Defaults to None.
                direction (tuple[float, float] | None, optional): The new direction. Defaults to None.
                width (float | None, optional): The new width. Defaults to None.
    
            Returns:
                StripPath: _description_
            """
            if width is None:
                width = self.end.width
    
            if direction is None:
                direction = self.end.direction
            else:
                direction = np.array(direction)
    
            if x is not None and y is not None:
                xnew = x
                ynew = x
            elif dx is not None and dy is not None:
                xnew = self.end.x + dx
                ynew = self.end.y + dy
            else:
                raise RouteException("Provide either an (x,y) pair or (dx,dy) pair for the move rout instruction")
    
            self._add_element(StripLine(xnew, ynew, self.end.z, width, direction))
            return self
    
    def to(
        self,
        dest: tuple[float, float] | Anchor,
        arrival_dir: tuple[float, float] | None = None,
        arrival_margin: float | None = None,
        angle_step: float = 90,
        corner_type: Literal['champher','square','miter'] = 'square',
        dsratio: float = 0.7,
    ) -> StripPath:
        """Extend the path from current end point to dest (x, y).
        Optionally ensure arrival in arrival_dir after a straight segment of arrival_margin.
        Turns are quantized to multiples of angle_step (divisor of 360, <=90).

        Args:
            dest (tuple[float, float] | Anchor): The destination xy coordinate
            arrival_dir (tuple[float, float] | None, optional): The arrival direction as dx/dy tuple. Defaults to None.
            arrival_margin (float | None, optional): The minimal travel distance in the arrival direction. Defaults to None.
            angle_step (float, optional): The angle step size to solve for. Defaults to 90.
            corner_type (["miter","square","champher"], optional): The corner type. Defaults to 'square'.
            dsratio (float, optional): The cut ratio for champher/miter corners. Defaults to 0.7.
        Returns:
            StripPath: _description_
        """
        dest = _parse_vector(dest)[:2]
        # Validate angle_step
        if 360 % angle_step != 0 or angle_step > 90 or angle_step <= 0:
            raise ValueError(
                f"angle_step must be a positive divisor of 360 <= 90, got {angle_step}"
            )

        # Current state
        x0, y0 = self.end.x, self.end.y
        vx, vy = self.end.direction  # unit heading
        tx, ty = dest

        # Compute unit arrival direction
        if arrival_dir is not None:
            adx, ady = arrival_dir
            mag = math.hypot(adx, ady)
            if mag == 0:
                raise ValueError("arrival_dir must be non-zero")
            ux, uy = adx / mag, ady / mag
        else:
            # if no arrival_dir, just head to point
            ux, uy = 0.0, 0.0
        arrival_margin += self.end.width
        # Compute base point: destination minus arrival margin
        bx = tx - ux * arrival_margin
        by = ty - uy * arrival_margin

        # Parametric search along negative arrival direction
        # we seek t >= 0 such that angle from (vx,vy) to (bx - x0 - ux*t, by - y0 - uy*t)
        # quantizes exactly to a multiple of angle_step
        atol = angle_step / 10
        dtol = 0.01
        t = 0.0
        max_t = math.hypot(bx - x0, by - y0) + arrival_margin + 1e-3
        dt = max_t / 1000.0  # resolution of search
        found = False
        desired_q = None
        # cand_dx = cand_dy = 0.0

        while t <= max_t:
            # candidate intercept point
            cx = bx - ux * t
            cy = by - uy * t
            dx = cx - x0
            dy = cy - y0
            if abs(dx) < dtol and abs(dy) < dtol:
                # reached start; skip
                t += dt
                continue
            # compute angle
            cross = vx * dy - vy * dx
            dot = vx * dx + vy * dy
            ang = math.degrees(math.atan2(cross, dot))
            # quantize
            q_ang = math.ceil(ang / angle_step) * angle_step
            if abs(ang - q_ang) <= atol:
                found = True
                desired_q = q_ang
                break
            t += dt

        if not found:
            raise RuntimeError(
                "Could not find an intercept angle matching quantization"
            )

        # 1) Perform initial quantized turn
        if abs(desired_q) > atol:  # type: ignore
            self.turn(-desired_q, corner_type=corner_type, dsratio=dsratio)  # type: ignore
        x0 = self.end.x
        y0 = self.end.y
        # compute new heading vector after turn
        theta = math.radians(desired_q)
        nvx = math.cos(theta) * vx - math.sin(theta) * vy
        nvy = math.sin(theta) * vx + math.cos(theta) * vy

        # 2) Compute exact intercept distance via line intersection:
        # Solve: (x0,y0) + s*(nvx,nvy) = (bx,by) - t*(ux,uy)
        # Unknowns s,t (we reuse t from above as initial guess, but solve fresh):
        tol_dist = 1e-6
        A11, A12 = nvx, ux
        A21, A22 = nvy, uy
        B1 = bx - x0
        B2 = by - y0
        det = A11 * A22 - A12 * A21
        if abs(det) < tol_dist:
            raise RuntimeError(
                "Initial heading parallel to arrival line, no unique intercept"
            )
        s = (B1 * A22 - B2 * A12) / det
        t_exact = (A11 * B2 - A21 * B1) / det
        if s < -tol_dist or (arrival_dir is not None and t_exact < -tol_dist):
            raise RuntimeError(
                "Computed intercept lies behind start or before arrival point"
            )

        # 3) Turn into arrival direction (if provided)

        # we need to rotate from current heading (vx,vy) by desired_q to (nvx,nvy)
        theta = math.radians(desired_q)
        nvx = math.cos(theta) * vx - math.sin(theta) * vy
        nvy = math.sin(theta) * vx + math.cos(theta) * vy
        # target heading is (ux,uy)
        cross2 = nvx * uy - nvy * ux
        dot2 = nvx * ux + nvy * uy
        back_ang = math.degrees(math.atan2(cross2, dot2))

        backoff = math.tan(abs(back_ang) * np.pi / 360) * self.end.width / 2

        self.straight(s - backoff)
        self.turn(-back_ang, corner_type=corner_type, dsratio=dsratio)

        x0 = self.end.x
        y0 = self.end.y
        D = math.hypot(tx - x0, ty - y0)
        # 4) Final straight into destination by arrival_margin + t
        self.straight(D)

        return self

    def macro(
        self,
        path: str,
        width: float | None = None,
        start_dir: tuple[float, float] | None = None,
    ) -> StripPath:
        r"""Parse an EMerge macro command string

        The start direction by default is the abslute current heading. If a specified heading is provided
        the macro language will assume that as the current heading and generate commands accordingly.

        The language is specified by a symbol plus a number.
        Symbols:
        - = X: Move X units forward
        - \> X: Turn to right and move X forward
        - < X: Turn to left and move X forward
        - v X: Turn to down and move X forward
        - ^ X: Turn to up and move X forward
        - T X,Y: Taper X forward to width Y
        - \\ X: Turn relative right 90 degrees and X forward
        - / X: Turn relative left 90 degrees and X forward

        (*) All commands X can also be provided as X,Y to change the width

        Args:
            path (str): The path command string
            width (float, optional): The width to start width. Defaults to None.
            start_dir (tuple[float, float], optional): The start direction to assume. Defaults to None.

        Example:
        >>> my_pcb.macro("= 5 v 4,1.2 > 5 ^ 2 > 3 T 4, 2.1")

        Returns:
            StripPath: The strippath object
        """
        if start_dir is None:
            start_dir = self.end.direction
        if width is None:
            width = self.end.width
        for instr in parse_macro(path, width, start_dir):
            getattr(self, instr.instr)(*instr.args, **instr.kwargs)
        return self

    def __call__(self, element_nr: int) -> RouteElement:
        if element_nr >= len(self.path):
            self.path.append(RouteElement())
        return self.path[element_nr]


class PCBLayer:
    _DEFNAME: str = "PCBLayer"

    def __init__(self, thickness: float, material: Material, name: str | None = None):
        self.th: float = thickness
        self.mat: Material = material
        self.name: str = _PCB_MANAGER(name, self._DEFNAME)


############################################################
#                     PCB DESIGN CLASS                     #
############################################################


class PCBNew:
    _DEFNAME: str = "PCB"
    """ The PCB Class can be used to efficiently generate PCB
    models using method chaining functions to describe traces
    as a sequence of instructions.
    """

    def __init__(
        self,
        thickness: float,
        unit: float = 0.001,
        cs: CoordinateSystem | None = None,
        material: Material = AIR,
        trace_material: Material = PEC,
        layers: int = 2,
        stack: list[PCBLayer] = None,
        name: str | None = None,
        trace_thickness: float | None = None,
        zs: np.ndarray | None = None,
        thick_traces: bool = False,
    ):
        """Creates a new PCB layout class instance

        Args:
            thickness (float): The total PCB thickness
            unit (float, optional): The units used for all dimensions. Defaults to 0.001 (mm).
            cs (CoordinateSystem | None, optional): The coordinate system to place the PCB in (XY). Defaults to None.
            material (Material, optional): The dielectric material. Defaults to AIR.
            trace_material (Material, optional): The trace material. Defaults to PEC.
            layers (int, optional): The number of copper layers. Defaults to 2.
            stack (list[PCBLayer], optional): Optional list of PCBLayer classes for multilayer PCB with different dielectrics. Defaults to None.
            name (str | None, optional): The PCB object name. Defaults to None.
            trace_thickness (float | None, optional): The conductor trace thickness if important. Defaults to None.
            thick_traces: (bool, optional): If traces should be given a thickness and modeled in 3D. Defaults to False
        """

        self.thickness: float = thickness
        self._thick_traces: bool = thick_traces
        self._stack: list[PCBLayer] = []

        if zs is not None:
            self._zs = zs
            self.thickness = np.max(zs) - np.min(zs)
            self._stack = [PCBLayer(th, material) for th in np.diff(self._zs)]
        elif stack is not None:
            self._stack = stack
            ths = [ly.th for ly in stack]
            zbot = -sum(ths)
            self._zs = np.concatenate(
                [
                    np.array(
                        [
                            zbot,
                        ]
                    ),
                    zbot + np.cumsum(np.array(ths)),
                ]
            )
            self.thickness = sum(ths)
        else:
            self._zs: np.ndarray = np.linspace(-self.thickness, 0, layers)
            ths = np.diff(self._zs)
            self._stack = [PCBLayer(th, material) for th in ths]

        self.material: Material = material
        self.trace_material: Material = trace_material
        self.width: float | None = None
        self.length: float | None = None
        self.origin: np.ndarray = np.array([0.0, 0.0, 0.0])

        self.paths: list[StripPath] = []
        self.polies: list[PCBPoly] = []

        self.hole_polies: list[PCBPoly] = []
        self.via_holes: list[PCBPoly] = []

        self.lumped_ports: list[StripLine] = []
        self.lumped_elements: list[GeoPolygon] = []
        self.trace_thickness: float | None = trace_thickness

        self.unit: float = unit

        self.cs: CoordinateSystem = cs
        if self.cs is None:
            self.cs = GCS

        self.dielectric_priority: int = 11
        self.via_priority: int = 12
        self.conductor_priority: int = 13

        self.traces: list[GeoPolygon | GeoVolume] = []
        self.ports: list[GeoPolygon | GeoVolume] = []
        self.vias: list[Via] = []

        self.xs: list[float] = []
        self.ys: list[float] = []
        self.zs: list[float] = []
        self._poly_out: list[PCBPoly] = []
        self._embedded_points: list[tuple[float, float, float]] = []

        self.stored_coords: dict[str, tuple[float, float]] = dict()
        self.stored_striplines: dict[str, RouteElement] = dict()
        self._checkpoint: list[StripPath] = []

        self.calc: PCBCalculator = PCBCalculator(
            self._zs, [layer.mat for layer in self._stack], self.unit
        )

        self.name: str = _PCB_MANAGER(name, self._DEFNAME)

    ############################################################
    #                          PROPERTIES                     #
    ############################################################

    @property
    def trace(self) -> GeoPolygon:
        ""
        tags = []
        for trace in self.traces:
            tags.extend(trace.tags)
        return GeoPolygon(tags)

    @property
    def all_objects(self) -> list[GeoPolygon]:
        """Returns all objects gnerated by the PCB layer."""
        return self.traces + self.ports

    @property
    def top(self) -> float:
        """The top conductor later height (z-value in meters)."""
        return self._zs[-1]

    @property
    def bottom(self) -> float:
        """The bottom conductor layer height (z-value in meters)"""
        return self._zs[0]

    ############################################################
    #                       PRIVATE FUNCTIONS                  #
    ############################################################

    def _lumped_element(
        self,
        poly: XYPolygon | GeoSurface,
        function: Callable,
        width: float,
        length: float,
        name: str | None = "LumpedElement",
    ) -> None:
        if isinstance(poly, XYPolygon):
            geopoly = poly._finalize(self.cs, name=name)
        else:
            geopoly = change_coordinate_system(poly, self.cs)
        area = width*length
        Npts = 50
        ds = (area/Npts)**0.5
        geopoly._mdi.add('lumpedelement', func=function, width=width, height=length)
        geopoly.max_meshsize = ds
        self.lumped_elements.append(geopoly)

    def _get_z(self, element: RouteElement) -> float:
        """Return the z-height of a given Route Element

        Args:
            element (RouteElement): The requested route element

        Returns:
            float: The z-height.
        """
        for path in self.paths:
            if path._has(element):
                return path.z
        raise RouteException(
            "Requesting z-height of route element that is not contained in a path."
        )

    def __call__(self, name: str) -> StripLine:
        """A quick way to call self.load(x)

        Args:
            name (str): The name of the StriLine segment

        Returns:
            StripLine: The stripline element
        """
        return self.load(name)

    def __getitem__(self, name: str) -> StripLine:
        """A quick way to call self.load(x)

        Args:
            name (str): The name of the StriLine segment

        Returns:
            StripLine: The stripline element
        """
        return self.load(name)

    def _gen_poly(
        self, xys: list[tuple[float, float]], z: float, name: str | None = None
    ) -> GeoPolygon | GeoVolume:
        """Generates a GeoPoly out of a list of (x,y) coordinate tuples.


        Args:
            xys (list[tuple[float, float]]): A list of (x,y) coordinate tuples.
            z (float, optional): The z-height of the polygon. Defaults to the top layer.
            name (str, optional): The name of the polygon.
        """

        self._poly_out.append(
            PCBPoly([xy[0] for xy in xys], [xy[1] for xy in xys], z=z)
        )

        ptags = []
        for x, y in xys:
            px, py, pz = self.cs.in_global_cs(
                x * self.unit, y * self.unit, z * self.unit
            )
            ptags.append(gmsh.model.occ.addPoint(px, py, pz))
        
            self._embedded_points.append((px,py,pz))
        ltags = []
        for t1, t2 in zip(ptags[:-1], ptags[1:]):
            ltags.append(gmsh.model.occ.addLine(t1, t2))
        ltags.append(gmsh.model.occ.addLine(ptags[-1], ptags[0]))

        tag_wire = gmsh.model.occ.addWire(ltags)
        planetag = gmsh.model.occ.addPlaneSurface([tag_wire,])

        if self._thick_traces:
            if self.trace_thickness is None:
                raise ValueError(
                    "Trace thickness not defined, cannot generate polygons. Make sure to define a trace thickness in the PCB() constructor."
                )
            dx, dy, dz = self.cs.zax.np * self.trace_thickness
            dimtags = gmsh.model.occ.extrude([(2, planetag),],dx,dy,dz)
            voltags = [dt[1] for dt in dimtags if dt[0] == 3]
            poly = GeoVolume(voltags, name=name).prio_set(self.conductor_priority)
        else:
            poly = GeoPolygon([planetag,],name=name,).set_material(self.trace_material)
            poly._store("thickness", self.trace_thickness)
        return poly

    def _via_hole(
        self,
        x: float,
        y: float,
        radius: float,
        z1: float,
        z2: float,
        skip_layers: list[int] | None = None,
    ) -> None:
        """Generates via holes in the ground planes and polies.

        Args:
            x (float): The x-coordinate of the via
            y (float): The y-coordinate of the via
            radius (float): The via hole radius
            z1 (float): The bottom z-coordinate of the via
            z2 (float): The top z-coordinate of the via
            skip_layers (list[int] | None, optional): A list of layer numbers where the via hole should not be created. Defaults to None.
        """
        if skip_layers is None:
            skip_layers = []
        for layer_nr, z in enumerate(self._zs):
            if layer_nr in skip_layers:
                continue
            if min(z1, z2) < z < max(z1, z2):
                hole_poly = PCBPoly.circle(x, y, radius, z=z)
                self.via_holes.append(hole_poly)

    ############################################################
    #                        USER FUNCTIONS                   #
    ############################################################

    def z(self, layer: int) -> float:
        """Returns the z-height of the given layer number counter from 0 (bottom) to N-1 (top)

        Args:
            layer (int): The layer number (0 to N-1)

        Returns:
            float: the z-height
        """
        if layer >= len(self._zs):
            raise ValueError(
                f"Layer {layer} does not exist in PCB with {len(self._zs)} layers. Since the new version, indexing starts from 0. Perhaps you meant layer {layer - 1}?"
            )
        return self._zs[layer]

    def add_vias(
        self,
        *coordinates: tuple[float, float] | Anchor,
        radius: float,
        z1: float | None = None,
        z2: float | None = None,
        segments: int = 6,
    ) -> None:
        """Add a series of vias provided by a list of coordinates.

        Vias will not be created yet. To generate the actual geometries use the function .generate_vias().

        Make sure to define the radius explicitly, otherwise the radius gets interpreted as a coordinate:

        >>> pcb.add_vias((x1,y1), (x1,y2), radius=1)

        Args:
            *coordinates (tuple(float, float)): A series of coordinates
            radius (float): The radius
            z1 (float | None, optional): The bottom z-coordinate. Defaults to None.
            z2 (float | None, optional): The top z-coordinate. Defaults to None.
            segments (int, optional): The number of segmets for the via. Defaults to 6.
        """
        coordinates = [_parse_vector(coord)[:2] for coord in coordinates]
        if z1 is None:
            z1 = self.z(0)
        if z2 is None:
            z2 = self.z(-1)

        for x, y in coordinates:
            self.vias.append(Via(x, y, z1, z2, radius, segments))

    def load(self, name: str) -> RouteElement:
        """Acquire the x,y, coordinate associated with the label name.

        Args:
            name (str): The name of the x,y coordinate

        """
        _PCB_MANAGER.unit = self.unit
        _PCB_MANAGER.cs = self.cs

        name = str(name)
        if name in self.stored_striplines:
            return self.stored_striplines[name]
        else:
            for poly in self.polies:
                if poly.name == name:
                    return poly
            raise ValueError(
                f"There is no stripline or coordinate under the name of {name}"
            )

    def determine_bounds(
        self,
        leftmargin: float = 0,
        topmargin: float = 0,
        rightmargin: float = 0,
        bottommargin: float = 0,
    ):
        """Defines the rectangular boundary of the PCB.

        Args:
            leftmargin (float, optional): The left margin. Defaults to 0.
            topmargin (float, optional): The top margin. Defaults to 0.
            rightmargin (float, optional): The right margin. Defaults to 0.
            bottommargin (float, optional): The bottom margin. Defaults to 0.
        """
        if len(self.xs) == 0:
            raise ValueError(
                "PCB path is not compiled. Compile before defining boundaries."
            )
        minx = min(self.xs)
        maxx = max(self.xs)
        miny = min(self.ys)
        maxy = max(self.ys)
        ml = leftmargin
        mt = topmargin
        mr = rightmargin
        mb = bottommargin
        self.width = maxx - minx + mr + ml
        self.length = maxy - miny + mt + mb
        self.origin = np.array([-ml + minx, -mb + miny, 0])

    def set_bounds(self, xmin: float, ymin: float, xmax: float, ymax: float) -> None:
        """Define the bounds of the PCB

        Args:
            xmin (float): The minimum x-coordinate
            ymin (float): The minimum y-coordinate
            xmax (float): The maximum x-coordinate
            ymax (float): The maximum y-coordinate
        """
        self.origin = np.array([xmin, ymin, 0])
        self.width = xmax - xmin
        self.length = ymax - ymin

    def plane(
        self,
        z: float,
        width: float | None = None,
        height: float | None = None,
        origin: tuple[float, float] | Anchor | None = None,
        alignment: Alignment = Alignment.CORNER,
        material: Material | None = None,
        name: str | None = None,
    ) -> GeoSurface | GeoVolume:
        """Generates a generic rectangular plate in the XY grid.
        If no size is provided, it defaults to the entire PCB size assuming that the bounds are determined.

        Args:
            z (float): The Z-height for the plate.
            width (float, optional): The width of the plate. Defaults to None.
            height (float, optional): The height of the plate. Defaults to None.
            origin (tuple[float, float], optional): The origin of the plate. Defaults to None.
            alignment (['corner','center], optional): The alignment of the plate. Defaults to 'corner'.
            material (Material, optional): The optional plane material. Defaults to the trace material specified.
        Returns:
            GeoSurface: The resultant GeoSurface of the plane
        """
        if width is not None and height is not None:
            self.xs.append(origin[0] / self.unit)
            self.xs.append(origin[0] / self.unit + width)
            self.ys.append(origin[1] / self.unit)
            self.ys.append(origin[1] / self.unit + height)

        if width is None or height is None or origin is None:
            if self.width is None or self.length is None or self.origin is None:
                raise RouteException(
                    "Cannot define a plane with no possible definition of its size."
                )
            width = self.width
            height = self.length
            origin = (self.origin[0], self.origin[1])

        if material is None:
            material = self.trace_material

        origin = tuple(_parse_vector(origin))
        origin: tuple[float, ...] = origin + (z,)  # type: ignore
        origin = tuple([param*self.unit for param in origin])
        if alignment is Alignment.CENTER:
            origin = (
                origin[0] - width * self.unit / 2,
                origin[1] - height * self.unit / 2,
                origin[2],
            )

        if self._thick_traces:
            plane = Box(
                width * self.unit,
                height * self.unit,
                self.trace_thickness,
                position=origin,
                name=name,
            ).set_material(material)
        else:
            plane = Plate(
                origin, (width * self.unit, 0, 0), (0, height * self.unit, 0), name=name
            )  # type: ignore
            plane._store("thickness", self.thickness)
            plane = change_coordinate_system(plane, self.cs)  # type: ignore
            plane.set_material(material)

        # subtract via holes:
        holes = []
        for via_hole in self.via_holes:
            holes.append(self._gen_poly(via_hole.xys, via_hole.z, name=via_hole.name))
        if len(holes) > 0:
            plane = remove(plane, unite(*holes))
        return plane  # type: ignore

    def radial_stub(
        self,
        pos: tuple[float, float] | Anchor,
        length: float,
        angle: float,
        direction: tuple[float, float],
        Nsections: int = 8,
        w0: float = 0,
        z: float = 0,
        material: Material = None,
        name: str = None,
    ) -> None:
        """Generates a radial stub polygon

        Args:
            pos (tuple[float, float] | Point): The position of the stub origin
            length (float): The length of the stub
            angle (float): The angle of the stub in degrees
            direction (tuple[float, float]): The direction vector
            Nsections (int, optional): Number of angle sections. Defaults to 8.
            w0 (float, optional): the start width. Defaults to 0.
            z (float, optional): the Z-height. Defaults to 0.
            material (Material, optional): the stub material. Defaults to None.
            name (str, optional): The geometry name. Defaults to None.
        """
        x0, y0 = _parse_vector(pos)[:2]
        dx, dy = direction

        rx, ry = dy, -dx

        points = []
        if w0 == 0:
            points.append(pos)
        else:
            points.append((x0 - rx * w0 / 2, y0 - ry * w0 / 2))
            points.append((x0 + rx * w0 / 2, y0 + ry * w0 / 2))

        angs = np.linspace(-angle / 2, angle / 2, Nsections) * np.pi / 180
        c0 = x0 + 1j * y0
        vec = length * dx + 1j * length * dy
        for a in angs:
            p = c0 + vec * np.exp(1j * a)
            points.append((p.real, p.imag))

        xs, ys = zip(*points)
        self.add_poly(xs, ys, z, material, name)

    def generate_pcb(
        self, split_z: bool = True, merge: bool = True
    ) -> list[GeoVolume] | GeoVolume:
        """Generate the PCB Block object

        Args:
            split_z (bool, optional): If a PCB consisting of a thickness, material and n_layers should be split in sub domains. Defaults to True
            merge: (bool, optional): If an output list of multiple volumes should be merged into a single object.

        Returns:
            GeoVolume | List[GeoVolume]: The PCB Block or blocks
        """
        x0, y0, z0 = self.origin * self.unit

        n_materials = len(set([id(layer.mat) for layer in self._stack]))

        if split_z and self._zs.shape[0] > 2 or n_materials > 1:
            boxes: list[GeoVolume] = []
            for i, (z1, z2, layer) in enumerate(
                zip(self._zs[:-1], self._zs[1:], self._stack)
            ):
                h = z2 - z1
                box = Box(
                    self.width * self.unit,
                    self.length * self.unit,
                    h * self.unit,
                    position=(x0, y0, z0 + z1 * self.unit),
                    name=layer.name,
                )
                box.material = layer.mat
                box = change_coordinate_system(box, self.cs)
                box.prio_set(self.dielectric_priority)
                boxes.append(box)
            if merge and n_materials == 1:
                return GeoVolume.merged(boxes).prio_set(self.dielectric_priority)  # type: ignore
            return boxes  # type: ignore

        box = Box(
            self.width * self.unit,
            self.length * self.unit,
            self.thickness * self.unit,
            position=(x0, y0, z0 - self.thickness * self.unit),
            name=f"{self.name}_diel",
        )
        box.material = self._stack[0].mat
        box.prio_set(self.dielectric_priority)
        box = change_coordinate_system(box, self.cs)
        return box  # type: ignore

    def generate_air(
        self, height: float, name: str = "PCBAirbox", bottom: bool = False
    ) -> GeoVolume:
        """Generate the Air Block object

        This requires that the width, depth and origin are deterimed. This
        can either be done manually or via the .determine_bounds() method.

        Returns:
            GeoVolume: The PCB Block
        """
        dz = 0

        x0, y0, z0 = self.origin * self.unit
        if bottom:
            dz = z0 - self.thickness * self.unit - height * self.unit

        box = Box(
            self.width * self.unit,
            self.length * self.unit,
            height * self.unit,
            position=(x0, y0, z0 + dz),
            name=name,
        )
        box = change_coordinate_system(box, self.cs)
        return box  # type: ignore

    def new(
        self,
        x: float | Anchor,
        y: float,
        width: float,
        direction: tuple[float, float],
        z: float = 0,
        name: str | None = None,
    ) -> StripPath:
        """Start a new trace

        The trace is started at the provided x,y, coordinates with a width "width".
        The direction must be provided as an (dx,dy) vector provided as tuple.

        Args:
            x (float): The starting X-coordinate (local)
            y (float): The starting Y-coordinate (local)
            width (float): The (micro)-stripline width
            direction (tuple[float, float]): The direction.

        Returns:
            StripPath: A StripPath object that can be extended with method chaining.

        Example:
        >>> PCB.new(...).straight(...).turn(...).straight(...) etc.

        """
        path = StripPath(self, name=name)
        path.init(x, y, width, direction, z=z)
        self.paths.append(path)
        return path

    def lumped_port(
        self,
        stripline: StripLine | str,
        z_ground: float | None = None,
        name: str | None = "LumpedPort",
    ) -> GeoPolygon:
        """Generate a lumped-port object to be created.

        You can flag any point during routing using list comprehensions and then pass those keys here.

        Example:
         >>> mypcb.new(...).straight(...)['mykey']
         >>> myport = mypcb.lumped_port('mykey')

        Lumped ports created this way automatically have auxilliary data properties set for the LumpedPort boundary condition.
        Example:
         >>> mymodel.mw.bc.LumpedPort(myport, port_nr)

        No width, height or axis data is needed.

        Args:
            stripline (StripLine | str): The stripline object or a reference to the stripline object
            z_ground (float | None, optional): The ground height to extrude the port to. Defaults to None.
            name (str | None, optional): The name for the resultant object. Defaults to 'LumpedPort'.

        Returns:
            GeoPolygon: The rectangular shaped lumped port object
        """

        if not isinstance(stripline, StripLine):
            stripline = self.load(stripline)

        xy1 = stripline.right[0]
        xy2 = stripline.left[0]
        z = self._get_z(stripline)
        if z_ground is None:
            z_ground = -self.thickness
        height = z - z_ground
        x1, y1, z1 = self.cs.in_global_cs(
            xy1[0] * self.unit, xy1[1] * self.unit, z * self.unit - height * self.unit
        )
        x2, y2, z2 = self.cs.in_global_cs(
            xy1[0] * self.unit, xy1[1] * self.unit, z * self.unit
        )
        x3, y3, z3 = self.cs.in_global_cs(
            xy2[0] * self.unit, xy2[1] * self.unit, z * self.unit
        )
        x4, y4, z4 = self.cs.in_global_cs(
            xy2[0] * self.unit, xy2[1] * self.unit, z * self.unit - height * self.unit
        )

        ptag1 = gmsh.model.occ.addPoint(x1, y1, z1)
        ptag2 = gmsh.model.occ.addPoint(x2, y2, z2)
        ptag3 = gmsh.model.occ.addPoint(x3, y3, z3)
        ptag4 = gmsh.model.occ.addPoint(x4, y4, z4)

        ltag1 = gmsh.model.occ.addLine(ptag1, ptag2)
        ltag2 = gmsh.model.occ.addLine(ptag2, ptag3)
        ltag3 = gmsh.model.occ.addLine(ptag3, ptag4)
        ltag4 = gmsh.model.occ.addLine(ptag4, ptag1)

        ltags = [ltag1, ltag2, ltag3, ltag4]

        tag_wire = gmsh.model.occ.addWire(ltags)
        planetag = gmsh.model.occ.addPlaneSurface(
            [
                tag_wire,
            ]
        )
        poly = GeoPolygon(
            [
                planetag,
            ],
            name="name",
        )
        poly._mdi.add('lumpedport', width=abs(stripline.width * self.unit), height=abs(height * self.unit), vdir=self.cs.zax)

        return poly

    def lumped_port_pts(
        self,
        p1: tuple[float, float],
        p2: tuple[float, float],
        z: float,
        z_ground: float | None = None,
        name: str | None = "LumpedPort",
    ) -> GeoPolygon:
        """Generate a lumped-port object to be created.

        The lumped port will be drawn and extruded form the line specified by two points.
        Args:
            p1: (tuple[float, float]) - Point one of the line to be extruded
            p2: (tuple[float, float]) - POint two of the line to be extruded
            z: (float) - The starting z-height for the extrusion.
            z_ground (float | None, optional): The ground height to extrude the port to. Defaults to None.
            name (str | None, optional): The name for the resultant object. Defaults to 'LumpedPort'.

        Returns:
            GeoPolygon: The rectangular shaped lumped port object
        """

        xy1 = p1
        xy2 = p2
        z = z / self.unit
        if z_ground is None:
            z_ground = -self.thickness
        height = z - z_ground

        width = abs(np.linalg.norm(np.array(xy1) - np.array(xy2)))
        x1, y1, z1 = self.cs.in_global_cs(
            xy1[0], xy1[1], z * self.unit - height * self.unit
        )
        x2, y2, z2 = self.cs.in_global_cs(xy1[0], xy1[1], z * self.unit)
        x3, y3, z3 = self.cs.in_global_cs(xy2[0], xy2[1], z * self.unit)
        x4, y4, z4 = self.cs.in_global_cs(
            xy2[0], xy2[1], z * self.unit - height * self.unit
        )

        ptag1 = gmsh.model.occ.addPoint(x1, y1, z1)
        ptag2 = gmsh.model.occ.addPoint(x2, y2, z2)
        ptag3 = gmsh.model.occ.addPoint(x3, y3, z3)
        ptag4 = gmsh.model.occ.addPoint(x4, y4, z4)

        ltag1 = gmsh.model.occ.addLine(ptag1, ptag2)
        ltag2 = gmsh.model.occ.addLine(ptag2, ptag3)
        ltag3 = gmsh.model.occ.addLine(ptag3, ptag4)
        ltag4 = gmsh.model.occ.addLine(ptag4, ptag1)

        ltags = [ltag1, ltag2, ltag3, ltag4]

        tag_wire = gmsh.model.occ.addWire(ltags)
        planetag = gmsh.model.occ.addPlaneSurface(
            [
                tag_wire,
            ]
        )
        poly = GeoPolygon(
            [
                planetag,
            ],
            name="name",
        )
        poly._mdi.add('lumpedport', width=abs(width), height=abs(height * self.unit), vdir = self.cs.zax)

        return poly

    def modal_port(
        self,
        point: StripLine | str,
        height: float | tuple[float, float],
        width_multiplier: float = 5.0,
        width: float | None = None,
        name: str | None = "ModalPort",
    ) -> GeoSurface:
        """Generate a wave-port as a GeoSurface.

        The port is placed at the coordinate of the provided stripline. The width
        is determined as a multiple of the stripline width. The height will be
        extended to the air height from the bottom of the PCB unless a different height is specified.

        Args:
            point (StripLine): The location of the port.
            width_multiplier (float, optional): The width of the port in stripline widths. Defaults to 5.0.
            height (float, optional): The height of the port. Defaults to None.

        Returns:
            GeoSurface: The GeoSurface object that can be used for the waveguide.
        """

        if not isinstance(point, StripLine):
            point = self.load(point)

        if isinstance(height, tuple):
            dz, height = height
        else:
            dz = 0

        height = self.thickness + height + dz

        if width is not None:
            W = width
        else:
            W = point.width * width_multiplier

        ds = point.dirright
        x0 = point.x - ds[0] * W / 2
        y0 = point.y - ds[1] * W / 2
        z0 = -self.thickness - dz
        ax1 = np.array([ds[0], ds[1], 0]) * self.unit * W
        ax2 = np.array([0, 0, 1]) * height * self.unit

        plate = Plate(np.array([x0, y0, z0]) * self.unit, ax1, ax2, name=name)
        plate = change_coordinate_system(plate, self.cs)
        return plate  # type: ignore

    @overload
    def generate_vias(self, merge=Literal[True], material: Material | None = None) -> GeoVolume: ...

    @overload
    def generate_vias(self, merge=Literal[False], material: Material | None = None) -> list[Cylinder]: ...

    def generate_vias(self, merge=False, material: Material | None = None) -> list[Cylinder] | GeoVolume:
        """Generates the via objects.

        Args:
            merge (bool, optional): Whether to merge the result into a final object. Defaults to False.
            material (Material, optional): The optional material for the vias.
        Returns:
            list[Cylinder] | Cylinder: Either al ist of cylllinders or a single one (merge=True)
        """
        vias = []
        if material is None:
            material = self.trace_material
        for via in self.vias:
            x0 = via.x * self.unit
            y0 = via.y * self.unit
            z0 = via.z1 * self.unit
            xg, yg, zg = self.cs.in_global_cs(x0, y0, z0)
            cs = CoordinateSystem(
                self.cs.xax, self.cs.yax, self.cs.zax, np.array([xg, yg, zg])
            )
            cyl = Cylinder(
                via.radius * self.unit, (via.z2 - via.z1) * self.unit, cs, via.segments
            )
            cyl.material = material
            cyl.prio_set(self.via_priority)
            vias.append(cyl)
        if merge:
            return GeoVolume.merged(vias)  # type: ignore
        return vias

    def add_poly(
        self,
        xs: list[float],
        ys: list[float],
        z: float = 0,
        material: Material = None,
        name: str | None = None,
    ) -> None:
        """Add a custom polygon to the PCB

        Args:
            xs (list[float]): A list of x-coordinates
            ys (list[float]): A list of y-coordinates
            z (float, optional): The z-height. Defaults to 0.
            material (Material, optional): The material. Defaults to COPPER.
        """
        if material is None:
            material = self.trace_material
        poly = PCBPoly(xs, ys, z, material, name=name)

        self.polies.append(poly)

    def add_hole(
        self,
        xs: list[float],
        ys: list[float],
        z: float = 0,
        name: str | None = None,
    ) -> None:
        """Add a custom polygon hole to the PCB

        Args:
            xs (list[float]): A list of x-coordinates
            ys (list[float]): A list of y-coordinates
            z (float, optional): The z-height. Defaults to 0.
            
        """
        material = self.trace_material
        poly = PCBPoly(xs, ys, z, material, name=name)

        self.hole_polies.append(poly)

    @overload
    def compile_paths(self, merge: Literal[True]) -> GeoSurface | GeoVolume: ...

    @overload
    def compile_paths(
        self, merge: Literal[False] = ...
    ) -> list[GeoSurface] | list[GeoVolume]: ...

    def compile_paths(
        self, 
        merge: bool = False, 
        fragment: bool = True
    ) -> list[GeoPolygon] | GeoSurface | GeoVolume:
        """Compiles the striplines and returns a list of polygons or asingle one.

        The Z=0 argument determines the height of the striplines. Z=0 corresponds to the top of
        the PCB.

        Args:
            merge (bool, optional): Whether to merge the Polygons into a single. Defaults to False.
            fragment (bool, optional): Adds line segment points to improve mesh quality in combination with set_trace_refinement().
        Returns:
            list[Polygon] | GeoSurface: The output stripline polygons possibly merged if merge = True.
        """
        polys: list[GeoSurface] = []
        allx = []
        ally = []

        from .pcb_tools.poly_parcelator import PolygonSet

        polyset = PolygonSet()

        for path in self.paths:
            z = path.z
            self.zs.append(z)
            xysL = []
            xysR = []

            for eprev, elemn, enext in path.iter_right():
                coords = elemn.right
                if enext.rcutprev and coords:
                    coords.pop(-1)
                if eprev.rcutnext and coords:
                    coords.pop(0)

                xysR.extend(coords)

            for eprev, elemn, enext in path.iter_left():
                coords = elemn.left
                if enext.lcutprev and coords:
                    coords.pop(-1)
                if eprev.lcutnext and coords:
                    coords.pop(0)

                xysL.extend(coords)

            xys = xysR + xysL[::-1]
            xm, ym = xys[0]
            xys2 = [
                (xm, ym),
            ]

            for x, y in xys[1:]:
                if (((x - xm) ** 2 + (y - ym) ** 2) ** 0.5 * self.unit) > (1e-9):
                    xys2.append((x, y))
                    xm, ym = x, y
                    allx.append(x)
                    ally.append(y)

            polyset.add_poly(xys2, z, self.trace_material)
            #poly = self._gen_poly(xys2, z)
            #poly.material = self.trace_material
            #polys.append(poly)

        for pcbpoly in self.polies:
            self.zs.append(pcbpoly.z)
            polyset.add_poly(pcbpoly.xys, pcbpoly.z, pcbpoly.material)
            #poly = self._gen_poly(pcbpoly.xys, pcbpoly.z, name=pcbpoly.name)
            #poly.material = pcbpoly.material
            #polys.append(poly)
            xs, ys = zip(*pcbpoly.xys)
            allx.extend(xs)
            ally.extend(ys)

        logger.info('Fragmenting polygons.')
        polyset.generate_cutlines()
        
        if fragment:
            polyset.fragment()
        for poly in polyset.polies:
            new_poly = self._gen_poly(poly.xys, poly.z)
            new_poly.material = poly.material
            polys.append(new_poly)
        
        holes = []
        for holepoly in self.hole_polies + self.via_holes:
            self.zs.append(holepoly.z)
            poly = self._gen_poly(holepoly.xys, holepoly.z, name=holepoly.name)
            holes.append(poly)


        self.xs = allx
        self.ys = ally

        self.traces = polys

        # For some reason this breaks GMSH. Something to ask Christophe
        # if fragment:
        #     for x,y,z in self._embedded_points:
        #         ptobj = Point(x,y,z)

        if merge:
            polys = unite(*polys)
            if holes:
                holes_union = unite(*holes)
                polys = remove(polys, holes_union)
            
        else:
            if holes:
                holes_union = unite(*holes)
                polys = [
                    remove(p, holes_union, remove_tool=i == (len(polys) - 1))
                    for i, p in enumerate(polys)
                ]
        return polys


class PCB(PCBNew):
    """DEPRICATED CLASS. Use PCBNew()"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        logger.warning(
            "PCB() is now depricated due to a change in the behavior of the .z() function.\n"
            + "Layers will be counted from 0 instead of 1. Use PCBNew(). Later these will be merged."
        )
        DEBUG_COLLECTOR.add_report(
            "PCB() is now depricated due to a change in the behavior of the .z() function.\n"
            + "Layers will be counted from 0 instead of 1. Use PCBNew(). Later these will be merged."
        )

    def z(self, layer: int) -> float:
        """
        Args:
            layer (int): The layer number (1 to N)

        Returns:
            float: the z-height
        """
        logger.warning(
            "The PCB class will be depricated. Move to PCBNew and index layers counting from 0 instead of 1."
        )
        if layer <= 0:
            return self._zs[layer]

        return self._zs[layer - 1]
