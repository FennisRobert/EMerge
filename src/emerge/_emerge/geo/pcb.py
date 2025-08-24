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

from ..cs import CoordinateSystem, GCS, Axis
from ..geometry import GeoPolygon, GeoVolume, GeoSurface
from ..material import Material, AIR, COPPER
from .shapes import Box, Plate, Cylinder
from .polybased import XYPolygon
from .operations import change_coordinate_system, unite
from .pcb_tools.macro import parse_macro
from .pcb_tools.calculator import PCBCalculator

import numpy as np
from loguru import logger
from typing import Literal, Callable, overload
from dataclasses import dataclass

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

SIZE_NAMES = Literal['0402','0603','1005','1608','2012','3216','3225','4532','5025','6332']
_SMD_SIZE_DICT = {x: (float(x[:2])*0.05, float(x[2:])*0.1) for x in ['0402','0603','1005','1608','2012','3216','3225','4532','5025','6332']}


############################################################
#                         FUNCTIONS                        #
############################################################


def approx(a,b):
    return abs(a-b) < 1e-8

def normalize(vector: np.ndarray) -> np.ndarray:
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
    ang = -angle * np.pi/180
    return np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])

############################################################
#                          CLASSES                         #
############################################################


class PCBPoly:

    def __init__(self, 
                 xs: list[float],
                 ys: list[float],
                 z: float = 0,
                 material: Material = COPPER):
        self.xs: list[float] = xs
        self.ys: list[float] = ys
        self.z: float = z
        self.material: Material = material

    @property
    def xys(self) -> list[tuple[float, float]]:
        return list([(x,y) for x,y in zip(self.xs, self.ys)])

@dataclass
class Via:
    x: float
    y: float
    z1: float
    z2: float
    radius: float
    segments: int

class RouteElement:

    def __init__(self):
        self.width: float = None
        self.x: float = None
        self.y: float = None
        self.direction: np.ndarray = None
        self.dirright: np.ndarray = None
    
    @property
    def xy(self) -> tuple[float, float]:
        return self.x, self.y

    @property
    def nr(self) -> tuple[float,float]:
        return (self.x + self.dirright[0]*self.width/2, self.y + self.dirright[1]*self.width/2)
    
    @property
    def nl(self) -> tuple[float,float]:
        return (self.x - self.dirright[0]*self.width/2, self.y - self.dirright[1]*self.width/2)
    
    @property
    def right(self) -> list[tuple[float, float]]:
        raise NotImplementedError()

    @property
    def left(self) -> list[tuple[float, float]]:
        raise NotImplementedError()
    
    def __eq__(self, other: RouteElement) -> bool:
        return approx(self.x, other.x) and approx(self.y, other.y) and (1-abs(np.sum(self.direction*other.direction)))<1e-8
    
class StripLine(RouteElement):

    def __init__(self,
                 x: float,
                 y: float,
                 width: float,
                 direction: tuple[float, float]):
        self.x = x
        self.y = y
        self.width = width
        self.direction = normalize(np.array(direction))
        self.dirright = np.array([self.direction[1], -self.direction[0]])

    def __str__(self) -> str:
        return f'StripLine[{self.x},{self.y},w={self.width},d=({self.direction})]'
    @property
    def right(self) -> list[tuple[float, float]]:
        return [(self.x + self.width/2 * self.dirright[0], self.y + self.width/2 * self.dirright[1])]

    @property
    def left(self) -> list[tuple[float, float]]:
        return [(self.x - self.width/2 * self.dirright[0], self.y - self.width/2 * self.dirright[1])]
      
class StripTurn(RouteElement):

    def __init__(self,
                 x: float,
                 y: float,
                 width: float,
                 direction: tuple[float, float],
                 angle: float,
                 corner_type: str = 'round',
                 champher_distance: float | None = None):
        self.xold: float = x
        self.yold: float = y
        self.width: float = width
        self.old_direction: np.ndarray = normalize(np.array(direction))
        self.direction: np.ndarray = _rot_mat(angle) @ self.old_direction
        self.angle: float = angle
        self.corner_type: str = corner_type
        self.dirright: np.ndarray = np.array([self.old_direction[1], -self.old_direction[0]])
        
        if champher_distance is None:
            self.champher_distance: float = 0.75 * self.width*np.tan(np.abs(angle)/2*np.pi/180)
        else:
            self.champher_distance: float = champher_distance

        turnvec = _rot_mat(angle) @ self.dirright * self.width/2

        if angle > 0:
            self.x = x + width/2 * self.dirright[0] - turnvec[0]
            self.y = y + width/2 * self.dirright[1] - turnvec[1]
        else:
            self.x = x - width/2 * self.dirright[0] + turnvec[0]
            self.y = y - width/2 * self.dirright[1] + turnvec[1]

    def __str__(self) -> str:
        return f'StripTurn[{self.x},{self.y},w={self.width},d=({self.direction})]'
    
    @property
    def right(self) -> list[tuple[float, float]]:
        if self.angle > 0:
            return []
        
        #turning left
        xl = self.xold - self.width/2 * self.dirright[0]
        yl = self.yold - self.width/2 * self.dirright[1]
        xr = self.xold + self.width/2 * self.dirright[0]
        yr = self.yold + self.width/2 * self.dirright[1]

        dist = min(self.width*np.sqrt(2), self.width * np.tan(np.abs(self.angle)/2*np.pi/180))        

        dcorner = self.width*(_rot_mat(self.angle) @ self.dirright)

        xend = xl + dcorner[0]
        yend = yl + dcorner[1]

        out = [(xend, yend)]

        if self.corner_type == 'champher':
            dist = max(0.0, dist - self.champher_distance)
        
        if dist==0:
            return  out
        
        x1 = xr + dist * self.old_direction[0]
        y1 = yr + dist * self.old_direction[1]

        if self.corner_type == 'square':
            return [(x1, y1), (xend, yend)]
        if self.corner_type == 'champher':
            x2 = xend - dist * self.direction[0]
            y2 = yend - dist * self.direction[1]

            return [(x1, y1), (x2, y2), (xend, yend)]
        else:
            raise RouteException(f'Trying to route a StripTurn with an unknown corner type: {self.corner_type}')
    @property
    def left(self) -> list[tuple[float, float]]:
        if self.angle < 0:
            return []
        
        #turning right
        xl = self.xold - self.width/2 * self.dirright[0]
        yl = self.yold - self.width/2 * self.dirright[1]
        xr = self.xold + self.width/2 * self.dirright[0]
        yr = self.yold + self.width/2 * self.dirright[1]
        
        dist = min(self.width*np.sqrt(2), self.width * np.tan(np.abs(self.angle)/2*np.pi/180))        

        dcorner = self.width*(_rot_mat(self.angle) @ -self.dirright)

        xend = xr + dcorner[0]
        yend = yr + dcorner[1]

        out = [(xend, yend)]

        if self.corner_type == 'champher':
            dist =max(0.0, dist - self.champher_distance)
        
        if dist==0:
            return  out
        
        x1 = xl + dist * self.old_direction[0]
        y1 = yl + dist * self.old_direction[1]

        if self.corner_type == 'square':
            return [(xend, yend), (x1, y1)]
        if self.corner_type == 'champher':
            x2 = xend - dist * self.direction[0]
            y2 = yend - dist * self.direction[1]

            return [(xend, yend), (x2, y2), (x1, y1)]
        else:
            raise RouteException(f'Trying to route a StripTurn with an unknown corner type: {self.corner_type}')
 
class StripCurve(StripTurn):
    def __init__(self,
                 x: float,
                 y: float,
                 width: float,
                 direction: tuple[float, float],
                 angle: float,
                 radius: float,
                 dang: float = 10.0):
        self.xold: float = x
        self.yold: float = y
        self.width: float = width
        self.old_direction: np.ndarray = normalize(np.array(direction))
        self.direction: np.ndarray = _rot_mat(angle) @ self.old_direction
        self.angle: float = angle
        self.radius: float = radius
        self.dirright: np.ndarray = np.array([self.old_direction[1], -self.old_direction[0]])
        self.dang: float = dang

        angd = abs(angle*np.pi/180)
        self.start = np.array([x,y])
        self.circ_origin = self.start + radius * np.sign(angle) * self.dirright

        self._xhat = -self.dirright * np.sign(angle)
        self._yhat = self.old_direction
        
        self.end = self.circ_origin + radius*(self._xhat*np.cos(angd)+self._yhat*np.sin(angd))
        self.x, self.y = self.end

    def __str__(self) -> str:
        return f'StripCurve[{self.x},{self.y},w={self.width},d=({self.direction})]'

    @property
    def right(self) -> list[tuple[float, float]]:
        points: list[tuple[float, float]] = []
        Npts = int(np.ceil(abs(self.angle/self.dang)))
        R = self.radius-np.sign(self.angle)*self.width/2
        for i in range(Npts):
            ang = abs((i+1)/Npts * self.angle * np.pi/180)
            pnew = self.circ_origin + R*(self._xhat*np.cos(ang)+self._yhat*np.sin(ang))
            points.append((pnew[0], pnew[1]))
        
        return points

    @property
    def left(self) -> list[tuple[float, float]]:
        points: list[tuple[float, float]] = []

        Npts = int(np.ceil(abs(self.angle/self.dang)))
        R = self.radius+np.sign(self.angle)*self.width/2
        for i in range(Npts):
            ang = abs((i+1)/Npts * self.angle * np.pi/180)
            pnew = self.circ_origin + R*(self._xhat*np.cos(ang)+self._yhat*np.sin(ang))
            points.append((pnew[0], pnew[1]))
        
        return points[::-1]

    
############################################################
#                    THE STRIP PATH CLASS                  #
############################################################

class StripPath:

    def __init__(self, pcb: PCB):
        self.pcb: PCB = pcb
        self.path: list[RouteElement] = []
        self.z: float = 0

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
        """ The start of the stripline. """
        return self.path[0]
    
    @property
    def end(self) -> RouteElement:
        """ The end of the stripline """
        return self.path[-1]
    
    def _check_loops(self) -> None:
        if self.path[0]==self.path[-1]:
            raise RouteException('Loops are currently not supported. To fix this problem, implement a single .cut() call before a .straight() call to break the loop.')
        return None
    
    def init(self, 
             x: float, 
             y: float, 
             width: float, 
             direction: tuple[float, float],
             z: float = 0) -> StripPath:
        """ Initializes the StripPath object for routing. """
        self.path.append(StripLine(x, y, width, direction))
        self.z = z
        return self
    
    def _add_element(self, element: RouteElement) -> StripPath:
        """ Adds the provided RouteElement to the path. """
        self.path.append(element)
        self._check_loops()
        return self
    
    def straight(self, distance: 
                 float, width: float | None = None, 
                 dx: float = 0, 
                 dy: float = 0) -> StripPath:
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

        dx_2, dy_2 = self.end.direction
        x1 = x + distance * dx_2
        y1 = y + distance * dy_2

        if width is not None:
            if width != self.end.width:
                self._add_element(StripLine(x, y, width, (dx_2, dy_2)))

        self._add_element(StripLine(x1, y1, self.end.width, (dx_2, dy_2)))
        return self
    
    def taper(self, distance: float, 
                 width: float) -> StripPath:
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

        dx_2, dy_2 = self.end.direction
        x1 = x + distance * dx_2
        y1 = y + distance * dy_2

        self._add_element(StripLine(x1, y1, width, (dx_2, dy_2)))
        
        return self
    
    def turn(self, angle: float, 
             width: float | None = None, 
             corner_type: Literal['champher','square'] = 'champher') -> StripPath:
        """Adds a turn to the strip path.

        The angle is specified in degrees. The width of the turn will be the same as the last segment.
        optionally, a different width may be provided. 
        By default, all corners will be cut using the "champher" type. Other options are not yet provided.

        Args:
            angle (float): The turning angle
            width (float, optional): The stripline width. Defaults to None.
            corner_type (str, optional): The corner type. Defaults to 'champher'.

        Returns:
            StripPath: The current StripPath object
        """
        x, y = self.end.x, self.end.y
        dx, dy = self.end.direction
        if abs(angle) <= 20:
            corner_type = 'square'
            logger.warning('Small turn detected, defaulting to rectangular corners because chamfers would add to much detail.')
        if width is not None:
            if width != self.end.width:
                self._add_element(StripLine(x, y, width, (dx, dy)))
        else:
            width=self.end.width
        self._add_element(StripTurn(x, y, width, (dx, dy), angle, corner_type))
        return self

    def curve(self, angle: float, radius: float,
             width: float | None = None, 
             corner_type: Literal['champher','square'] = 'champher',
             dang: float = 10) -> StripPath:
        """Adds a bend to the strip path.

        The angle is specified in degrees. The width of the turn will be the same as the last segment.
        optionally, a different width may be provided. 
        By default, all corners will be cut using the "champher" type. Other options are not yet provided.

        Args:
            angle (float): The turning angle
            width (float, optional): The stripline width. Defaults to None.
            corner_type (str, optional): The corner type. Defaults to 'champher'.

        Returns:
            StripPath: The current StripPath object
        """
        if angle == 0:
            logger.trace('Zero angle turn, passing action')
            return self
        x, y = self.end.x, self.end.y
        dx, dy = self.end.direction
        if width is not None:
            if width != self.end.width:
                self._add_element(StripLine(x, y, width, (dx, dy)))
        else:
            width=self.end.width
        self._add_element(StripCurve(x, y, width, (dx, dy), angle, radius, dang=dang))
        return self
    
    def store(self, name: str) -> StripPath:
        """ Store the current x,y coordinate labeled in the PCB object.

        The stored coordinate can be accessed by calling the .load() method on the PCBRouter class.

        Args:
            name (str): The coordinate label

        Returns:
            StripPath: The current StripPath object.
        """
        self.pcb.stored_striplines[name] = self.end
        return self
    
    def split(self, 
              direction: tuple[float, float]  | None= None,
              width: float  | None= None) -> StripPath:
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
        paths = self.pcb.new(x,y,width, direction, z)
        self.pcb._checkpoint = self
        return paths

    def lumped_element(self, impedance_function: Callable, size: SIZE_NAMES | tuple) -> StripPath:
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
        else:
            length, width = size
        
        dx, dy = self.end.direction
        x, y = self.end.x, self.end.y
        rx, ry = self.end.dirright
        wh = width/2
        xs = np.array([x+rx*wh, x+rx*wh+length*dx, x-rx*wh+length*dx, x-rx*wh])*self.pcb.unit
        ys = np.array([y+ry*wh, y+ry*wh+length*dy, y-ry*wh+length*dy, y-ry*wh])*self.pcb.unit
        poly = XYPolygon(xs, ys)
        
        self.pcb._lumped_element(poly, impedance_function, width, length)
        return self.pcb.new(x+dx*length, y+dy*length, self.end.width, self.end.direction, self.z)

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
        paths = self.pcb.new(x,y,width, direction, z)
        return paths
    
    def stub(self, direction: tuple[float, float],
             width: float,
             length: float, 
             mirror: bool = False) -> StripPath:
        """ Add a single rectangular strip line section at the current coordinate"""
        self.pcb.new(self.end.x, self.end.y, width, direction, self.z).straight(length)
        if mirror:
            self.pcb.new(self.end.x, self.end.y, width, (-direction[0], -direction[1]), self.z).straight(length)
        return self
    
    def merge(self) -> StripPath:
        """Continue at the last point where .split() is called"""
        if self.pcb._checkpoint is None:
            raise RouteException('No checkpoint known. Make sure to call .check() first')
        return self.pcb._checkpoint
    
    def via(self,
            znew: float,
            radius: float,
            proceed: bool = True,
            direction: tuple[float, float] | None = None,
            width: float | None = None,
            extra: float | None = None,
            segments: int = 6) -> StripPath:
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

        Returns:
            StripPath: The new StripPath object
        """
        
        if extra is None:
            extra = self.end.width/2
        x, y = self.end.x, self.end.y
        z1 = self.z
        z2 = znew
        if extra > 0:
            self.straight(extra)
        self.pcb.vias.append(Via(x,y,z1,z2,radius, segments))
        if proceed:
            if width is None:
                width = self.end.width
            if direction is None:
                direction = self.end.direction
            dx = direction[0]*extra
            dy = direction[1]*extra
            return self.pcb.new(x-dx, y-dy, width, direction, z2)
        return self
    
    def short(self) -> StripPath:
        self.via(self.pcb.z(1), self.end.width/3, False)
        return self

    def jump(self, 
             dx: float | None = None,
             dy: float | None = None,
             width: float | None = None,
             direction: tuple[float, float] | None = None,
             gap: float | None = None,
             side: Literal['left','right'] | None = None,
             reverse: float | None = None) -> StripPath:
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
            if side=='left':
                Q = -1
            x = ending.x - reverse*ending.direction[0] + Q*ending.dirright[0]*(width/2 + ending.width/2 + gap)
            y = ending.y - reverse*ending.direction[1] + Q*ending.dirright[1]*(width/2 + ending.width/2 + gap)
        else:
            x = ending.x + dx
            y = ending.y + dy
        return self.pcb.new(x, y, width, direction)
    
    def to(self, dest: tuple[float, float], 
           arrival_dir: tuple[float, float] | None = None,
           arrival_margin: float  | None= None,
           angle_step: float = 90) -> StripPath:
        """
        Extend the path from current end point to dest (x, y).
        Optionally ensure arrival in arrival_dir after a straight segment of arrival_margin.
        Turns are quantized to multiples of angle_step (divisor of 360, <=90).
        """
        # Validate angle_step
        if 360 % angle_step != 0 or angle_step > 90 or angle_step <= 0:
            raise ValueError(f"angle_step must be a positive divisor of 360 <= 90, got {angle_step}")

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
            ux, uy = adx/mag, ady/mag
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
        atol = angle_step/10
        dtol = 0.01
        t = 0.0
        max_t = math.hypot(bx - x0, by - y0) + arrival_margin + 1e-3
        dt = max_t / 1000.0  # resolution of search
        found = False
        desired_q = None
        #cand_dx = cand_dy = 0.0
        
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
            cross = vx*dy - vy*dx
            dot = vx*dx + vy*dy
            ang = math.degrees(math.atan2(cross, dot))
            # quantize
            q_ang = math.ceil(ang / angle_step) * angle_step
            if abs(ang - q_ang) <= atol:
                found = True
                desired_q = q_ang
                break
            t += dt

        if not found:
            raise RuntimeError("Could not find an intercept angle matching quantization")

        # 1) Perform initial quantized turn
        if abs(desired_q) > atol: # type: ignore
            self.turn(-desired_q) # type: ignore
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
            raise RuntimeError("Initial heading parallel to arrival line, no unique intercept")
        s = (B1 * A22 - B2 * A12) / det
        t_exact = (A11 * B2 - A21 * B1) / det
        if s < -tol_dist or (arrival_dir is not None and t_exact < -tol_dist):
            raise RuntimeError("Computed intercept lies behind start or before arrival point")

        
        # 3) Turn into arrival direction (if provided)
    
        # we need to rotate from current heading (vx,vy) by desired_q to (nvx,nvy)
        theta = math.radians(desired_q)
        nvx = math.cos(theta)*vx - math.sin(theta)*vy
        nvy = math.sin(theta)*vx + math.cos(theta)*vy
        # target heading is (ux,uy)
        cross2 = nvx*uy - nvy*ux
        dot2 = nvx*ux + nvy*uy
        back_ang = math.degrees(math.atan2(cross2, dot2))
        
        backoff = math.tan(abs(back_ang)*np.pi/360)*self.end.width/2

        self.straight(s - backoff)
        self.turn(-back_ang)

        x0 = self.end.x
        y0 = self.end.y
        D = math.hypot(tx-x0, ty-y0)
        # 4) Final straight into destination by arrival_margin + t
        self.straight(D)

        return self

    def macro(self, path: str, width: float | None = None, start_dir: tuple[float, float] | None = None) -> StripPath:
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

############################################################
#                     PCB DESIGN CLASS                     #
############################################################

class PCB:
    def __init__(self,
                 thickness: float,
                 unit: float = 0.001,
                 cs: CoordinateSystem | None = None,
                 material: Material = AIR,
                 layers: int = 2,
                 ):

        self.thickness: float = thickness
        self._zs: np.ndarray = np.linspace(-self.thickness, 0, layers)
        self.material: Material = material
        self.width: float | None = None
        self.length: float | None = None
        self.origin: np.ndarray = np.array([0.,0.,0.])
        self.paths: list[StripPath] = []
        self.polies: list[PCBPoly] = []

        self.lumped_ports: list[StripLine] = []
        self.lumped_elements: list[GeoPolygon] = []

        self.unit: float = unit

        self.cs: CoordinateSystem = cs
        if self.cs is None:
            self.cs = GCS

        self.traces: list[GeoPolygon] = []
        self.ports: list[GeoPolygon] = []
        self.vias: list[Via] = []

        self.xs: list[float] = []
        self.ys: list[float] = []
        self.zs: list[float] = []

        self.stored_coords: dict[str,tuple[float, float]] = dict()
        self.stored_striplines: dict[str, StripLine] = dict()
        self._checkpoint: StripPath = None

        self.calc: PCBCalculator = PCBCalculator(self.thickness, self._zs, self.material, self.unit)

    @property
    def trace(self) -> GeoPolygon:
        tags = []
        for trace in self.traces:
            tags.extend(trace.tags)
        return GeoPolygon(tags)
    
    
    @property
    def all_objects(self) -> list[GeoPolygon]:
        return self.traces + self.ports
    

    def z(self, layer: int) -> float:
        """Returns the z-height of the given layer number counter from 1 (bottom) to N (top)

        Args:
            layer (int): The layer number (1 to N)

        Returns:
            float: the z-height
        """
        if layer <= 0:
            return self._zs[layer]
        return self._zs[layer-1]
    
    @property
    def top(self) -> float:
        return self._zs[-1]
    
    @property
    def bottom(self) -> float:
        return self._zs[0]
    
    def _get_z(self, element: RouteElement) -> float :
        """Return the z-height of a given Route Element

        Args:
            element (RouteElement): The requested route element

        Returns:
            float: The z-height.
        """
        for path in self.paths:
            if path._has(element):
                return path.z
        raise RouteException('Requesting z-height of route element that is not contained in a path.')

    def add_vias(self, *coordinates: tuple[float, float], radius: float,
                 z1: float | None = None,
                 z2: float | None = None,
                 segments: int = 6) -> None:
        """Add a series of vias provided by a list of coordinates.
        
        Make sure to define the radius explicitly, otherwise the radius gets interpreted as a coordinate:
        
        >>> pcb.add_vias((x1,y1), (x1,y2), radius=1)

        Args:
            *coordinates (tuple(float, float)): A series of coordinates
            radius (float): The radius
            z1 (float | None, optional): The bottom z-coordinate. Defaults to None.
            z2 (float | None, optional): The top z-coordinate. Defaults to None.
            segments (int, optional): The number of segmets for the via. Defaults to 6.
        """
        if z1 is None:
            z1 = self.z(0)
        if z2 is None:
            z2 = self.z(-1)
        
        for x,y in coordinates:
            self.vias.append(Via(x,y,z1,z2,radius,segments))
        
    def load(self, name: str) -> StripLine:
        """Acquire the x,y, coordinate associated with the label name.
        
        Args:
            name (str): The name of the x,y coordinate
            
        """
        if name in self.stored_striplines:
            return self.stored_striplines[name]
        else:
            raise ValueError(f'There is no stripline or coordinate under the name of {name}')
    
    def __call__(self, path_nr: int) -> StripPath:
        if path_nr >= len(self.paths):
            self.paths.append(StripPath(self))
        return self.paths[path_nr]
    
    def determine_bounds(self, 
                         leftmargin: float = 0,
                         topmargin: float = 0,
                         rightmargin: float = 0,
                         bottommargin: float = 0):
        """Defines the rectangular boundary of the PCB.

        Args:
            leftmargin (float, optional): The left margin. Defaults to 0.
            topmargin (float, optional): The top margin. Defaults to 0.
            rightmargin (float, optional): The right margin. Defaults to 0.
            bottommargin (float, optional): The bottom margin. Defaults to 0.
        """
        if len(self.xs) == 0:
            raise ValueError('PCB path is not compiled. Compile before defining boundaries.')
        minx = min(self.xs)
        maxx = max(self.xs)
        miny = min(self.ys)
        maxy = max(self.ys)
        ml = leftmargin
        mt = topmargin
        mr = rightmargin
        mb = bottommargin
        self.width = (maxx - minx + mr + ml)
        self.length = (maxy - miny + mt + mb)
        self.origin = np.array([-ml+minx, -mb+miny, 0])
    
    def set_bounds(self,
                   xmin: float,
                   ymin: float,
                   xmax: float,
                   ymax: float) -> None:
        """Define the bounds of the PCB

        Args:
            xmin (float): The minimum x-coordinate
            ymin (float): The minimum y-coordinate
            xmax (float): The maximum x-coordinate
            ymax (float): The maximum y-coordinate
        """
        self.origin = np.array([xmin, ymin, 0])
        self.width = xmax-xmin
        self.length = ymax-ymin

    def plane(self,
              z: float,
              width: float | None = None,
              height: float | None = None,
              origin: tuple[float, float] | None = None,
              alignment: Literal['corner','center'] = 'corner') -> GeoSurface:
        """Generates a generic rectangular plate in the XY grid.
        If no size is provided, it defaults to the entire PCB size assuming that the bounds are determined.

        Args:
            z (float): The Z-height for the plate.
            width (float, optional): The width of the plate. Defaults to None.
            height (float, optional): The height of the plate. Defaults to None.
            origin (tuple[float, float], optional): The origin of the plate. Defaults to None.
            alignment (['corner','center], optional): The alignment of the plate. Defaults to 'corner'.

        Returns:
            GeoSurface: _description_
        """
        if width is None or height is None or origin is None:
            if self.width is None or self.length is None or self.origin is None:
                raise RouteException('Cannot define a plane with no possible definition of its size.')
            width = self.width
            height = self.length
            origin = (self.origin[0]*self.unit, self.origin[1]*self.unit)
        
        origin: tuple[float, ...] = origin + (z*self.unit, ) # type: ignore

        if alignment == 'center':
            origin = (origin[0] - width*self.unit/2, 
                                                        origin[1] - height*self.unit/2, 
                                                        origin[2])

        plane = Plate(origin, (width*self.unit, 0, 0), (0, height*self.unit, 0)) # type: ignore
        plane = change_coordinate_system(plane, self.cs) # type: ignore
        plane.set_material(COPPER)
        return plane # type: ignore
    
    def generate_pcb(self, 
                split_z: bool = True,
                layer_tolerance: float = 1e-6,
                merge: bool = True) -> GeoVolume:
        """Generate the PCB Block object

        Returns:
            GeoVolume: The PCB Block
        """
        x0, y0, z0 = self.origin*self.unit

        if split_z:
            zvalues = sorted(list(set(self.zs + [-self.thickness, 0.0])))
            zvalues_isolated = [zvalues[0],]
            for z in zvalues[1:]:
                if (z-zvalues_isolated[-1]) <= layer_tolerance:
                    continue
                zvalues_isolated.append(z)
            boxes: list[GeoVolume] = []
            for z1, z2 in zip(zvalues_isolated[:-1],zvalues_isolated[1:]):
                h = z2-z1
                box = Box(self.width*self.unit,
                          self.length*self.unit,
                          h*self.unit,
                          position=(x0, y0, z0+z1*self.unit))
                box.material = self.material
                box = change_coordinate_system(box, self.cs)
                boxes.append(box)
            if merge:
                return GeoVolume.merged(boxes) # type: ignore
            return boxes # type: ignore
        
        box = Box(self.width*self.unit, 
                  self.length*self.unit, 
                  self.thickness*self.unit, 
                  position=(x0,y0,z0-self.thickness*self.unit))
        box.material = self.material
        box = change_coordinate_system(box, self.cs)
        return box # type: ignore

    def generate_air(self, height: float) -> GeoVolume:
        """Generate the Air Block object

        This requires that the width, depth and origin are deterimed. This 
        can either be done manually or via the .determine_bounds() method.

        Returns:
            GeoVolume: The PCB Block
        """
        x0, y0, z0 = self.origin*self.unit
        box = Box(self.width*self.unit, 
                  self.length*self.unit, 
                  height*self.unit, 
                  position=(x0,y0,z0))
        box = change_coordinate_system(box, self.cs)
        return box # type: ignore

    def new(self, 
            x: float, 
            y: float, 
            width: float, 
            direction: tuple[float, float],
            z: float = 0) -> StripPath:
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
        path = StripPath(self)
        path.init(x, y, width, direction, z=z)
        self.paths.append(path)  
        return path
    
    def lumped_port(self, stripline: StripLine, z_ground: float | None = None) -> GeoPolygon:
        """Generate a lumped-port object to be created.

        Args:
            stripline (StripLine): _description_
        """
        
        xy1 = stripline.right[0]
        xy2 = stripline.left[0]
        z = self._get_z(stripline)
        if z_ground is None:
            z_ground = -self.thickness
        height = z-z_ground
        x1, y1, z1 = self.cs.in_global_cs(xy1[0]*self.unit, xy1[1]*self.unit, z*self.unit - height*self.unit)
        x2, y2, z2 = self.cs.in_global_cs(xy1[0]*self.unit, xy1[1]*self.unit, z*self.unit )
        x3, y3, z3 = self.cs.in_global_cs(xy2[0]*self.unit, xy2[1]*self.unit, z*self.unit )
        x4, y4, z4 = self.cs.in_global_cs(xy2[0]*self.unit, xy2[1]*self.unit, z*self.unit - height*self.unit)
        
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
        planetag = gmsh.model.occ.addPlaneSurface([tag_wire,])
        poly = GeoPolygon([planetag,])
        poly._aux_data['width'] = stripline.width*self.unit
        poly._aux_data['height'] = height*self.unit
        poly._aux_data['vdir'] = self.cs.zax
        poly._aux_data['idir'] = Axis(self.cs.xax.np*stripline.dirright[0] + self.cs.yax.np*stripline.dirright[1])
        
        return poly

    def _lumped_element(self, poly: XYPolygon, function: Callable, width: float, length: float) -> None:

        geopoly = poly._finalize(self.cs)
        geopoly._aux_data['func'] = function
        geopoly._aux_data['width'] = width
        geopoly._aux_data['height'] = length
        self.lumped_elements.append(geopoly)


    def modal_port(self,
                  point: StripLine,
                  height: float,
                  width_multiplier: float = 5.0,
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
        
        height = (self.thickness + height)
        
        ds = point.dirright
        x0 = point.x - ds[0]*point.width*width_multiplier/2
        y0 = point.y - ds[1]*point.width*width_multiplier/2
        z0 =  - self.thickness
        ax1 = np.array([ds[0], ds[1], 0])*self.unit*point.width*width_multiplier
        ax2 = np.array([0,0,1])*height*self.unit

        plate = Plate(np.array([x0,y0,z0])*self.unit, ax1, ax2)
        plate = change_coordinate_system(plate, self.cs)
        return plate # type: ignore

    @overload
    def generate_vias(self, merge=Literal[True]) -> GeoVolume: ...
    
    @overload
    def generate_vias(self, merge=Literal[False]) -> list[Cylinder]: ...
        
    def generate_vias(self, merge=False) -> list[Cylinder] | GeoVolume:
        """Generates the via objects.

        Args:
            merge (bool, optional): Whether to merge the result into a final object. Defaults to False.

        Returns:
            list[Cylinder] | Cylinder: Either al ist of cylllinders or a single one (merge=True)
        """
        vias = []
        for via in self.vias:
            x0 = via.x*self.unit
            y0 = via.y*self.unit
            z0 = via.z1*self.unit
            xg, yg, zg = self.cs.in_global_cs(x0, y0, z0)
            cs = CoordinateSystem(self.cs.xax, self.cs.yax, self.cs.zax, np.array([xg, yg, zg]))
            cyl = Cylinder(via.radius*self.unit, (via.z2-via.z1)*self.unit, cs, via.segments)
            cyl.material = COPPER
            vias.append(cyl)
        if merge:
            
            return GeoVolume.merged(vias) # type: ignore
        return vias   

    def add_poly(self, 
                 xs: list[float],
                 ys: list[float],
                 z: float = 0,
                 material: Material = COPPER) -> None:
        """Add a custom polygon to the PCB

        Args:
            xs (list[float]): A list of x-coordinates
            ys (list[float]): A list of y-coordinates
            z (float, optional): The z-height. Defaults to 0.
            material (Material, optional): The material. Defaults to COPPER.
        """
        self.polies.append(PCBPoly(xs, ys, z, material))

    def _gen_poly(self, xys: list[tuple[float, float]], z: float) -> GeoPolygon:
        """ Generates a GeoPoly out of a list of (x,y) coordinate tuples"""
        ptags = []
        for x,y in xys:
            px, py, pz = self.cs.in_global_cs(x*self.unit, y*self.unit, z*self.unit)
            ptags.append(gmsh.model.occ.addPoint(px, py, pz))
        
        ltags = []
        for t1, t2 in zip(ptags[:-1], ptags[1:]):
            ltags.append(gmsh.model.occ.addLine(t1, t2))
        ltags.append(gmsh.model.occ.addLine(ptags[-1], ptags[0]))
        
        tag_wire = gmsh.model.occ.addWire(ltags)
        planetag = gmsh.model.occ.addPlaneSurface([tag_wire,])
        poly = GeoPolygon([planetag,])
        return poly
    
    @overload
    def compile_paths(self, merge: Literal[True]) -> GeoSurface: ...
    
    @overload
    def compile_paths(self, merge: Literal[False] = ...) -> list[GeoSurface]: ...

    def compile_paths(self, merge: bool = False) -> list[GeoPolygon] | GeoSurface:
        """Compiles the striplines and returns a list of polygons or asingle one.

        The Z=0 argument determines the height of the striplines. Z=0 corresponds to the top of
        the PCB.

        Args:
            merge (bool, optional): Whether to merge the Polygons into a single. Defaults to False.

        Returns:
            list[Polygon] | GeoSurface: The output stripline polygons possibly merged if merge = True.
        """
        polys: list[GeoSurface] = []
        allx = []
        ally = []

        for path in self.paths:
            z = path.z
            self.zs.append(z)
            xys = []
            for elemn in path.path:
                xys.extend(elemn.right)
            for element in path.path[::-1]:
                xys.extend(element.left)

            xm, ym = xys[0]
            xys2 = [(xm,ym),]
            
            for x,y in xys[1:]:
                if ((x-xm)**2 + (y-ym)**2)>1e-6:
                    xys2.append((x,y))
                    xm, ym = x, y
                    allx.append(x)
                    ally.append(y)
            
            poly = self._gen_poly(xys2, z)
            poly.material = COPPER
            polys.append(poly)

        for pcbpoly in self.polies:
            self.zs.append(pcbpoly.z)
            poly = self._gen_poly(pcbpoly.xys, pcbpoly.z)
            poly.material = pcbpoly.material
            polys.append(poly)
            xs, ys = zip(*pcbpoly.xys)
            allx.extend(xs)
            ally.extend(ys)
            

        self.xs = allx
        self.ys = ally

        self.traces = polys
        
        if merge:
            polys = unite(*polys)
            
        return polys
                
############################################################
#                        DEPRICATED                       #
############################################################

class PCBLayouter(PCB):

    def __init__(self, *args, **kwargs):
        logger.warning('PCBLayouter will be depricated. Use PCB instead.')
        super().__init__(*args, **kwargs)
        