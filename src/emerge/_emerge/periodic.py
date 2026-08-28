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
from ._global import _GlobalHandler
from .cs import Axis, _parse_axis, GCS, _parse_vector
from .selection import Selection, FaceSelection
from .geo import GeoPrism, XYPolygon, Alignment, XYPlate
from .bc import BoundaryCondition
from typing import Generator
from .bc import Periodic
import numpy as np


############################################################
#                         FUNCTIONS                        #
############################################################

def _rotnorm(v: np.ndarray) -> np.ndarray:
    """Rotate 3D vector field v 90° counterclockwise around z axis.

    v shape = (3, Ny, Nx)
    """
    ax = np.array([-v[1], v[0], v[2]])
    ax = ax/np.linalg.norm(ax)
    return ax

def _pair_selection(f1: Selection, 
                    f2: Selection, 
                    translation: tuple[float, float, float]):
    if len(f1.tags) == 1:
        return [f1,], [f2,]
    c1s = [np.array(c) for c in f1.centers]
    c2s = [np.array(c) for c in f2.centers]
    ds = np.array(translation)
    f1s = []
    f2s = []
    for t1, c1 in zip(f1.tags, c1s):
        for t2, c2 in zip(f2.tags, c2s):
            if np.linalg.norm((c1 + ds)-c2) < 1e-8:
                f1s.append(FaceSelection([t1,]))
                f2s.append(FaceSelection([t2,]))
    return f1s, f2s



############################################################
#                 BASE PERIODIC CELL CLASS                #
############################################################

# TODO: This must be moved to mw physics if possible
class PeriodicCell:

    def __init__(self, 
                 origins: list[tuple[float, float, float] | list[float] | np.ndarray | Axis],
                 vectors: list[tuple[float, float, float] | list[float] | np.ndarray | Axis]):

        self.origins: list[tuple[float, float, float]] = [_parse_vector(origin) for origin in origins] # type: ignore
        self.vectors: list[Axis] = [_parse_axis(vec) for vec in vectors]
        self.included_faces: Selection | None = None
        self._bcs: list[Periodic] = []
        self._ports: list[BoundaryCondition] = []

        self.__post_init__()

    def __post_init__(self):
        pass

    def volume(self, z1: float, z2: float) -> GeoPrism:
        """Genereates a volume with the cell geometry ranging from z1 tot z2

        Args:
            z1 (float): The start height
            z2 (float): The end height

        Returns:
            GeoPrism: The resultant prism
        """
        raise NotImplementedError('This method is not implemented for this subclass.')
    
    def cell_data(self) -> Generator[tuple[Selection, Selection, np.ndarray], None, None]:
        """An iterator that yields the two faces of the hex cell plus a cell periodicity vector

        Yields:
            Generator[np.ndarray, np.ndarray, np.ndarray]: The face and periodicity data
        """
        raise NotImplementedError('This method is not implemented for this subclass.')

    def generate_bcs(self) -> list[Periodic]:
        """Generate the priodic boundary conditions
        """
        bcs = []
        for f1, f2, a in self.cell_data():
            f1_new = f1
            f2_new = f2
            if self.included_faces is not None:
                f1_new = f1 & self.included_faces # type: ignore
                f2_new = f2 & self.included_faces # type: ignore
            if len(f1_new.tags)==0:
                continue
            bcs.append(Periodic(f1_new, f2_new, tuple(a)))
        self._bcs = bcs
        return bcs
    
    @property
    def bcs(self) -> list[BoundaryCondition]:
        """Returns a list of Periodic boundary conditions for the given PeriodicCell

        Args:
            exclude_faces (list[FaceSelection], optional): A possible list of faces to exclude from the bcs. Defaults to None.

        Returns:
            list[Periodic]: The list of Periodic boundary conditions
        """
        if not self._bcs:
            raise ValueError('Periodic Boundary conditions not generated')
        return self._bcs + self._ports # type: ignore
    
    def set_scanangle(self, theta: float, phi: float, degree: bool = True) -> None:
        """Sets the scanangle for the periodic condition. (0,0) is defined along the Z-axis

        Args:
            theta (float): The theta angle
            phi (float): The phi angle
            degree (bool): If the angle is in degrees. Defaults to True
        """
        if degree:
            theta = theta*np.pi/180
            phi = phi*np.pi/180

        
        ux = np.sin(theta)*np.cos(phi)
        uy = np.sin(theta)*np.sin(phi)
        uz = np.cos(theta)
        for bc in self._bcs:
            bc.ux = ux
            bc.uy = uy
            bc.uz = uz
        for port in self._ports:
            port.scan_theta = theta # type: ignore
            port.scan_phi = phi # type: ignore

    def port_face(self, z: float):
        """Generate a floquet port face object at the given
        z-height. This will automatically also add a Floquet Port boundary condition.

        Args:
            z (float): The z-height for the port

        Raises:
            NotImplementedError: _description_
        """
        raise NotImplementedError('')
    


############################################################
#                    RECTANGULAR TILING                   #
############################################################

class RectCell(PeriodicCell):
    """This class represents the unit cell environment of a regular rectangular tiling.

    Args:
        PeriodicCell (_type_): _description_
    """
    def __init__(self, 
                 width: float,
                 height: float,):
        """The RectCell class represents a regular rectangular tiling in the XY plane where
        the width is along the X-axis (centered at x=0) and the height along the Y-axis (centered at y=0)

        Args:
            width (float): The Cell width
            height (float): The Cell height
        """
        v1 = (width, 0, 0)
        o1 = (-width/2, 0, 0)
        v2 = (0, height, 0)
        o2 = (0, -height/2, 0)
        super().__init__([o1, o2], [v1, v2])
        self.width: float = width
        self.height: float = height
        self.fleft = (o1, v1)
        self.fbot = (o2, v2)
        self.ftop = ((0, height/2, 0), v2)
        self.fright = ((width/2, 0, 0), v1)

    def port_face(self, z: float):
        return XYPlate(self.width, self.height, position=(0,0,z), alignment=Alignment.CENTER)
        
    def cell_data(self):
        f1s = _GlobalHandler.active().selector.inplane(*self.fleft[0], self.fleft[1])
        f2s = _GlobalHandler.active().selector.inplane(*self.fright[0], self.fright[1])
        vec = (self.fright[0][0]-self.fleft[0][0], 
               self.fright[0][1]-self.fleft[0][1], 
               self.fright[0][2]-self.fleft[0][2])
        for f1, f2 in zip(*_pair_selection(f1s, f2s, vec)):
            yield f1, f2, vec

        f1s = _GlobalHandler.active().selector.inplane(*self.fbot[0], self.fbot[1])
        f2s = _GlobalHandler.active().selector.inplane(*self.ftop[0], self.ftop[1])
        vec = (self.ftop[0][0]-self.fbot[0][0], 
               self.ftop[0][1]-self.fbot[0][1], 
               self.ftop[0][2]-self.fbot[0][2])
        for f1, f2 in zip(*_pair_selection(f1s, f2s, vec)):
            yield f1, f2, vec

    def volume(self, 
               z1: float,
               z2: float) -> GeoPrism:
        xs = np.array([-self.width/2, self.width/2, self.width/2, -self.width/2])
        ys = np.array([-self.height/2, -self.height/2, self.height/2, self.height/2])
        poly = XYPolygon(xs, ys)
        length = z2-z1
        return poly.extrude(length, cs=GCS.displace(0,0,z1))
    


############################################################
#                     HEXAGONAL TILING                    #
############################################################

class HexCell(PeriodicCell):

    def __init__(self,
                 point1: tuple[float, float, float],
                 point2: tuple[float, float, float],
                 point3: tuple[float, float, float]):
        """
        Generates a Hexagonal periodic tiling by providing three coordinates.
        The layout of the tiling assumes a hexagon with a single vertex at the top and bottom,
        and one vertex on the bottom and right faces (⬢).

        Args:
            point1 (tuple[float, float, float]): left face top vertex
            point2 (tuple[float, float, float]): left face bottom vertex
            point3 (tuple[float, float, float]): bottom vertex
        """
        p1, p2, p3 = np.array(point1), np.array(point2), np.array(point3)
        p4 = -p1
        self.p1: np.ndarray = p1
        self.p2: np.ndarray = p2
        self.p3: np.ndarray = p3
        o1 = (p1+p2)/2
        o2 = (p2+p3)/2
        o3 = (p3+p4)/2
        self.o1 = o1
        self.o2 = o2
        self.o3 = o3
        n1 = _rotnorm(p2-p1)
        n2 = _rotnorm(p3-p2)
        n3 = _rotnorm(p4-p3)
        
        super().__init__([o1, o2, o3], [n1,n2,n3])

        self.f11 = (o1, n1)
        self.f21 = (o2, n2)
        self.f31 = (o3, n3)
        self.f12 = (-o1, n1)
        self.f22 = (-o2, n2)
        self.f32 = (-o3, n3)

    @property
    def area(self) -> float:
        """Area of the centrally symmetric hexagon defined by p1, p2, p3."""
        p1, p2, p3 = self.p1, self.p2, self.p3
        area = float(np.linalg.norm(abs(
            np.cross(p1, p2) +
            np.cross(p2, p3) -
            np.cross(p3, p1)
        )))
        return area
        
    def port_face(self, z: float):
        xs, ys, zs = zip(self.p1, self.p2, self.p3, -self.p1, -self.p2, -self.p3)
        poly = XYPolygon(xs, ys).geo(GCS.displace(0,0,z))
        return poly
    
    def cell_data(self) -> Generator[tuple[Selection, Selection, np.ndarray], None, None]:
        nrm = np.linalg.norm

        o = self.o1[:-1]
        n = self.f11[1][:-1]
        w = nrm(self.p2-self.p1)/2
        f1s = _GlobalHandler.active().selector.inplane(*self.f11[0], self.f11[1])\
            .exclude(lambda x, y, z: (nrm(np.array([x,y])-o)>w) or (abs((np.array([x,y])-o) @ n ) > 1e-6))
        f2s = _GlobalHandler.active().selector.inplane(*self.f12[0], self.f12[1])\
            .exclude(lambda x, y, z: (nrm(np.array([x,y])+o)>w) or (abs((np.array([x,y])+o) @ n ) > 1e-6))
        vec = - (self.p1 + self.p2)

        for f1, f2 in zip(*_pair_selection(f1s, f2s, vec)): # type: ignore
            yield f1, f2, vec

        o = self.o2[:-1]
        n = self.f21[1][:-1]
        w = nrm(self.p3-self.p2)/2
        f1s = _GlobalHandler.active().selector.inplane(*self.f21[0], self.f21[1])\
            .exclude(lambda x, y, z: (nrm(np.array([x,y])-o)>w) or (abs((np.array([x,y])-o) @ n ) > 1e-6))
        f2s = _GlobalHandler.active().selector.inplane(*self.f22[0], self.f22[1])\
            .exclude(lambda x, y, z: (nrm(np.array([x,y])+o)>w) or (abs((np.array([x,y])+o) @ n ) > 1e-6))
        vec = - (self.p2 + self.p3)
        for f1, f2 in zip(*_pair_selection(f1s, f2s, vec)): # type: ignore
            yield f1, f2, vec
        
        o = self.o3[:-1]
        n = self.f31[1][:-1]
        w = nrm(-self.p1-self.p3)/2
        f1s = _GlobalHandler.active().selector.inplane(*self.f31[0], self.f31[1])\
            .exclude(lambda x, y, z: (nrm(np.array([x,y])-o)>w) or (abs((np.array([x,y])-o) @ n ) > 1e-6))
        f2s = _GlobalHandler.active().selector.inplane(*self.f32[0], self.f32[1])\
            .exclude(lambda x, y, z: (nrm(np.array([x,y])+o)>w) or (abs((np.array([x,y])+o) @ n ) > 1e-6))
        vec = - (self.p3 - self.p1)
        for f1, f2 in zip(*_pair_selection(f1s, f2s, vec)): # type: ignore
            yield f1, f2, vec

    def volume(self, 
               z1: float,
               z2: float) -> GeoPrism:
        """Generate a volume object for the given cell geometry
        # From z1 to z2. 

        Args:
            z1 (float): The starting z-height
            z2 (float): The ending z-height

        Returns:
            GeoPrism: _description_
        """
        z1, z2 = min(z1, z2), max(z1, z2)
        xs, ys, zs = zip(self.p1, self.p2, self.p3)
        xs2 = np.array(xs) # type: ignore
        ys2 = np.array(ys) # type: ignore
        xs3 = np.concatenate([xs2, -xs2]) # type: ignore
        ys3 = np.concatenate([ys2, -ys2]) # type: ignore
        poly = XYPolygon(xs3, ys3)
        length = z2-z1
        return poly.extrude(length, cs=GCS.displace(0,0,z1))
    