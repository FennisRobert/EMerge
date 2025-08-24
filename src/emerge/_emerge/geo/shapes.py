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

import gmsh
from ..geometry import GeoObject, GeoSurface, GeoVolume
from ..cs import CoordinateSystem, GCS
import numpy as np
from enum import Enum
from .operations import subtract
from ..selection import FaceSelection, Selector, SELECTOR_OBJ

from typing import Literal
from functools import reduce

class Alignment(Enum):
    CENTER = 1
    CORNER = 2


class Box(GeoVolume):
    """Creates a box volume object.
        Specify the alignment of the box with the provided position. The options are CORNER (default)
        for the front-left-bottom node of the box or CENTER for the center of the box.

        Args:
            width (float): The x-size
            depth (float): The y-size
            height (float): The z-size
            position (tuple, optional): The position of the box. Defaults to (0,0,0).
            alignment (Alignment, optional): Which point of the box is placed at the position. 
                Defaults to Alignment.CORNER.
        """

    def __init__(self, 
                 width: float, 
                 depth: float, 
                 height: float, 
                 position: tuple = (0,0,0),
                 alignment: Alignment = Alignment.CORNER,
                 cs: CoordinateSystem = GCS):
        """Creates a box volume object.
        Specify the alignment of the box with the provided position. The options are CORNER (default)
        for the front-left-bottom node of the box or CENTER for the center of the box.

        Args:
            width (float): The x-size
            depth (float): The y-size
            height (float): The z-size
            position (tuple, optional): The position of the box. Defaults to (0,0,0).
            alignment (Alignment, optional): Which point of the box is placed at the position. 
                Defaults to Alignment.CORNER.
        """
        if alignment is Alignment.CENTER:
            position = (position[0]-width/2, position[1]-depth/2, position[2]-height/2)
        
        x,y,z = position

        tag = gmsh.model.occ.addBox(x,y,z,width,depth,height)
        super().__init__(tag)

        self.center = (x+width/2, y+depth/2, z+height/2)
        self.width = width
        self.height = height
        self.depth = depth

        wax = cs.xax.np
        dax = cs.yax.np
        hax = cs.zax.np

        p0 = self.center
        pc = p0
        self._add_face_pointer('front', pc - depth/2*dax, -dax)
        self._add_face_pointer('back', pc + depth/2*dax, dax)
        self._add_face_pointer('left', pc - width/2*wax, -wax)
        self._add_face_pointer('right', pc + width/2*wax, wax)
        self._add_face_pointer('top', pc + height/2*hax, hax)
        self._add_face_pointer('bottom', pc - height/2*hax, -hax)
        
    
    def outside(self, *exclude: Literal['bottom','top','right','left','front','back']) -> FaceSelection:
        """Select all outside faces except for the once specified by outside

        Returns:
            FaceSelection: The resultant face selection
        """
        tagslist = [self._face_tags(name) for name in  ['bottom','top','right','left','front','back'] if name not in exclude]
        
        tags = list(reduce(lambda a,b: a+b, tagslist))
        return FaceSelection(tags)
    
    
class Sphere(GeoVolume):
    """Generates a sphere objected centered ont he position with the given radius

    Args:
        radius (float): The sphere radius
        position (tuple, optional): The center position. Defaults to (0,0,0).
    """
    def __init__(self, 
                 radius: float,
                 position: tuple = (0,0,0)):
        """Generates a sphere objected centered ont he position with the given radius

        Args:
            radius (float): The sphere radius
            position (tuple, optional): The center position. Defaults to (0,0,0).
        """
        super().__init__([])
        x,y,z = position
        self.tags: list[int] = [gmsh.model.occ.addSphere(x,y,z,radius),]


class XYPlate(GeoSurface):
    """Generates and XY-plane oriented plate
        
        Specify the alignment of the plate with the provided position. The options are CORNER (default)
        for the front-left node of the plate or CENTER for the center of the plate.

        Args:
            width (float): The x-size of the plate
            depth (float): The y-size of the plate
            position (tuple, optional): The position of the alignment node. Defaults to (0,0,0).
            alignment (Alignment, optional): Which node to align to. Defaults to Alignment.CORNER.
        """
    def __init__(self, 
                 width: float, 
                 depth: float, 
                 position: tuple = (0,0,0),
                 alignment: Alignment = Alignment.CORNER):
        """Generates and XY-plane oriented plate
        
        Specify the alignment of the plate with the provided position. The options are CORNER (default)
        for the front-left node of the plate or CENTER for the center of the plate.

        Args:
            width (float): The x-size of the plate
            depth (float): The y-size of the plate
            position (tuple, optional): The position of the alignment node. Defaults to (0,0,0).
            alignment (Alignment, optional): Which node to align to. Defaults to Alignment.CORNER.
        """
        super().__init__([])
        if alignment is Alignment.CENTER:
            position = (position[0]-width/2, position[1]-depth/2, position[2])
        
        x,y,z = position
        self.tags: list[int] = [gmsh.model.occ.addRectangle(x,y,z,width,depth),]


class Plate(GeoSurface):
    """A generalized 2D rectangular plate in XYZ-space.

        The plate is specified by an origin (o) in meters coordinate plus two vectors (u,v) in meters
        that span two of the sides such that all points of the plate are defined by:
            p1 = o
            p2 = o+u
            p3 = o+v
            p4 = o+u+v
        Args:
            origin (tuple[float, float, float]): The origin of the plate in meters
            u (tuple[float, float, float]): The u-axis of the plate
            v (tuple[float, float, float]): The v-axis of the plate
        """
    def __init__(self,
                origin: tuple[float, float, float],
                u: tuple[float, float, float],
                v: tuple[float, float, float]):
        """A generalized 2D rectangular plate in XYZ-space.

        The plate is specified by an origin (o) in meters coordinate plus two vectors (u,v) in meters
        that span two of the sides such that all points of the plate are defined by:
            p1 = o
            p2 = o+u
            p3 = o+v
            p4 = o+u+v
        Args:
            origin (tuple[float, float, float]): The origin of the plate in meters
            u (tuple[float, float, float]): The u-axis of the plate
            v (tuple[float, float, float]): The v-axis of the plate
        """
        
        origin = np.array(origin)
        u = np.array(u)
        v = np.array(v)
        
        tagp1 = gmsh.model.occ.addPoint(*origin)
        tagp2 = gmsh.model.occ.addPoint(*(origin+u))
        tagp3 = gmsh.model.occ.addPoint(*(origin+v))
        tagp4 = gmsh.model.occ.addPoint(*(origin+u+v))

        tagl1 = gmsh.model.occ.addLine(tagp1, tagp2)
        tagl2 = gmsh.model.occ.addLine(tagp2, tagp4)
        tagl3 = gmsh.model.occ.addLine(tagp4, tagp3)
        tagl4 = gmsh.model.occ.addLine(tagp3, tagp1)

        tag_wire = gmsh.model.occ.addWire([tagl1,tagl2, tagl3, tagl4])

        tags: list[int] = [gmsh.model.occ.addPlaneSurface([tag_wire,]),]
        super().__init__(tags)


class Cylinder(GeoVolume):
    """Generates a Cylinder object in 3D space.
        The cylinder will always be placed in the origin of the provided CoordinateSystem.
        The bottom cylinder plane is always placed in the XY-plane. The length of the cylinder is
        oriented along the Z-axis.

        By default the cylinder uses the Open Cascade modeling for a cylinder. In this representation
        the surface of the cylinder is approximated with a tolerance thay may be irregular.
        As an alternative, the argument Nsections may be provided in which case the Cylinder is replaced
        by an extrusion of a regular N-sided polygon.

        Args:
            radius (float): The radius of the Cylinder
            height (float): The height of the Cylinder
            cs (CoordinateSystem, optional): The coordinate system. Defaults to GCS.
            Nsections (int, optional): The number of sections. Defaults to None.
        """
    def __init__(self, 
                 radius: float,
                 height: float,
                 cs: CoordinateSystem = GCS,
                 Nsections: int | None = None):
        """Generates a Cylinder object in 3D space.
        The cylinder will always be placed in the origin of the provided CoordinateSystem.
        The bottom cylinder plane is always placed in the XY-plane. The length of the cylinder is
        oriented along the Z-axis.

        By default the cylinder uses the Open Cascade modeling for a cylinder. In this representation
        the surface of the cylinder is approximated with a tolerance thay may be irregular.
        As an alternative, the argument Nsections may be provided in which case the Cylinder is replaced
        by an extrusion of a regular N-sided polygon.

        Args:
            radius (float): The radius of the Cylinder
            height (float): The height of the Cylinder
            cs (CoordinateSystem, optional): The coordinate system. Defaults to GCS.
            Nsections (int, optional): The number of sections. Defaults to None.
        """
        ax = cs.zax.np

        if Nsections:
            from .polybased import XYPolygon
            cyl = XYPolygon.circle(radius, Nsections=Nsections).extrude(height, cs)
            cyl._exists = False
            self._face_pointers = cyl._face_pointers
            super().__init__(cyl.tags)
        else:
            cyl = gmsh.model.occ.addCylinder(cs.origin[0], cs.origin[1], cs.origin[2],
                                         height*ax[0], height*ax[1], height*ax[2],
                                         radius)
            super().__init__(cyl)
        self._add_face_pointer('front', cs.origin, -cs.zax.np)
        self._add_face_pointer('back', cs.origin+height*cs.zax.np, cs.zax.np)
        self._add_face_pointer('bottom', cs.origin, -cs.zax.np)
        self._add_face_pointer('top', cs.origin+height*cs.zax.np, cs.zax.np)
        self._add_face_pointer('left', cs.origin, -cs.zax.np)
        self._add_face_pointer('right', cs.origin+height*cs.zax.np, cs.zax.np)
            
        self.cs: CoordinateSystem = cs
        self.radius = radius
        self.height = height

    def face_points(self, nRadius: int = 10, Angle: int = 10, face_number: int = 1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns the points of the cylinder."""
        rs = np.linspace(0, self.radius, nRadius)
        angles = np.linspace(0, 2 * np.pi, int(360 / Angle), endpoint=False)
        R, A = np.meshgrid(rs, angles)
        x = R * np.cos(A)
        y = R * np.sin(A)
        z = np.zeros_like(x)
        if face_number == 2:
            z = z + self.height  

        xo, yo, zo = self.cs.in_global_cs(x.flatten(), y.flatten(), z.flatten())
        return xo, yo, zo


class CoaxCylinder(GeoVolume):
    """Generates a Coaxial cylinder object in 3D space.
        The coaxial cylinder will always be placed in the origin of the provided CoordinateSystem.
        The bottom coax plane is always placed in the XY-plane. The lenth of the coax is
        oriented along the Z-axis.

        By default the coax uses the Open Cascade modeling for a cylinder. In this representation
        the surface of the cylinder is approximated with a tolerance thay may be irregular.
        As an alternative, the argument Nsections may be provided in which case the Cylinder is replaced
        by an extrusion of a regular N-sided polygon.

        Args:
            radius (float): The radius of the Cylinder
            height (float): The height of the Cylinder
            cs (CoordinateSystem, optional): The coordinate system. Defaults to GCS.
            Nsections (int, optional): The number of sections. Defaults to None.
        """
    def __init__(self, 
                 rout: float,
                 rin: float,
                 height: float,
                 cs: CoordinateSystem = GCS,
                 Nsections: int | None = None):
        """Generates a Coaxial cylinder object in 3D space.
        The coaxial cylinder will always be placed in the origin of the provided CoordinateSystem.
        The bottom coax plane is always placed in the XY-plane. The lenth of the coax is
        oriented along the Z-axis.

        By default the coax uses the Open Cascade modeling for a cylinder. In this representation
        the surface of the cylinder is approximated with a tolerance thay may be irregular.
        As an alternative, the argument Nsections may be provided in which case the Cylinder is replaced
        by an extrusion of a regular N-sided polygon.

        Args:
            radius (float): The radius of the Cylinder
            height (float): The height of the Cylinder
            cs (CoordinateSystem, optional): The coordinate system. Defaults to GCS.
            Nsections (int, optional): The number of sections. Defaults to None.
        """
        if rout <= rin:
            raise ValueError("Outer radius must be greater than inner radius.")
        
        self.rout = rout
        self.rin = rin
        self.height = height

        self.cyl_out = Cylinder(rout, height, cs, Nsections=Nsections)
        self.cyl_in = Cylinder(rin, height, cs, Nsections=Nsections)
        self.cyl_in._exists = False
        self.cyl_out._exists = False
        cyltags, _ = gmsh.model.occ.cut(self.cyl_out.dimtags, self.cyl_in.dimtags)
        
        super().__init__([dt[1] for dt in cyltags])

        self._add_face_pointer('front', cs.origin, -cs.zax.np)
        self._add_face_pointer('back', cs.origin+height*cs.zax.np, cs.zax.np)
        self._add_face_pointer('bottom', cs.origin, -cs.zax.np)
        self._add_face_pointer('top', cs.origin+height*cs.zax.np, cs.zax.np)
        self._add_face_pointer('left', cs.origin, -cs.zax.np)
        self._add_face_pointer('right', cs.origin+height*cs.zax.np, cs.zax.np)

        self.cs = cs

    def face_points(self, nRadius: int = 10, Angle: int = 10, face_number: int = 1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns the points of the coaxial cylinder."""
        rs = np.linspace(self.rin, self.rout, nRadius)
        angles = np.linspace(0, 2 * np.pi, int(360 / Angle), endpoint=False)
        R, A = np.meshgrid(rs, angles)
        x = R * np.cos(A)
        y = R * np.sin(A)
        z = np.zeros_like(x)
        if face_number == 2:
            z = z + self.height  

        xo, yo, zo = self.cs.in_global_cs(x.flatten(), y.flatten(), z.flatten())
        return xo, yo, zo
    
class HalfSphere(GeoVolume):
    """A half sphere volume."""
    def __init__(self,
                 radius: float,
                 position: tuple = (0,0,0),
                 direction: tuple = (1,0,0)):
        
        sphere = Sphere(radius, position=position)
        cx, cy, cz = position

        dx, dy, dz = direction
        fx = 0.5**dx
        fy = 0.5**dy
        fz = 0.5**dz
        box = Box(2.2*radius*fx, 2.2*radius*fy, 2.2*radius*fz, position=(cx-radius*1.1*dx*0.5,cy-radius*1.1*dy*0.5, cz-radius*1.1*dz*0.5), alignment=Alignment.CENTER)
        
        dimtags, _ = gmsh.model.occ.cut(sphere.dimtags, box.dimtags)
        
        sphere._exists = False
        box._exists = False

        super().__init__([dt[1] for dt in dimtags])
        
        self._add_face_pointer('front',np.array(position), np.array(direction))
        self._add_face_pointer('back',np.array(position), np.array(direction))
        self._add_face_pointer('bottom',np.array(position), np.array(direction))
        self._add_face_pointer('face',np.array(position), np.array(direction))


class OldBox(GeoVolume):
    '''The sided box class creates a box just like the Box class but with selectable face tags.
    This class is more convenient in use when defining radiation boundaries.'''
    def __init__(self, 
                 width: float, 
                 depth: float, 
                 height: float, 
                 position: tuple = (0,0,0),
                 cs: CoordinateSystem = GCS,
                 alignment: Alignment = Alignment.CORNER):
        """Creates a box volume object with selectable sides.

        Specify the alignment of the box with the provided position. The options are CORNER (default)
        for the front-left-bottom node of the box or CENTER for the center of the box.

        Args:
            width (float): The x-size
            depth (float): The y-size
            height (float): The z-size
            position (tuple, optional): The position of the box. Defaults to (0,0,0).
            alignment (Alignment, optional): Which point of the box is placed at the position. 
                Defaults to Alignment.CORNER.
        """
        if alignment is Alignment.CORNER:
            position = (position[0]+width/2, position[1]+depth/2, position[2])
        elif alignment is Alignment.CENTER:
            position = (position[0], position[1], position[2] - height/2)
        p0 = np.array(position)
        p1 = p0 + cs.zax.np * height

        wax = cs.xax.np
        dax = cs.yax.np
        hax = cs.zax.np

        w = width
        d = depth

        p11 = p0 + wax * w/2 + dax * d/2 # right back
        p12 = p0 + wax * w/2 - dax * d/2 # right front
        p13 = p0 - wax * w/2 - dax * d/2 # Left front
        p14 = p0 - wax * w/2 + dax * d/2 # left back

        p21 = p1 + wax * w/2 + dax * d/2
        p22 = p1 + wax * w/2 - dax * d/2
        p23 = p1 - wax * w/2 - dax * d/2
        p24 = p1 - wax * w/2 + dax * d/2

        pt11 = gmsh.model.occ.addPoint(*p11) # right back
        pt12 = gmsh.model.occ.addPoint(*p12) # Right front
        pt13 = gmsh.model.occ.addPoint(*p13) # left front
        pt14 = gmsh.model.occ.addPoint(*p14) # Left back
        pt21 = gmsh.model.occ.addPoint(*p21)
        pt22 = gmsh.model.occ.addPoint(*p22)
        pt23 = gmsh.model.occ.addPoint(*p23)
        pt24 = gmsh.model.occ.addPoint(*p24)

        l1r = gmsh.model.occ.addLine(pt11, pt12) #Right
        l1f = gmsh.model.occ.addLine(pt12, pt13) #Front
        l1l = gmsh.model.occ.addLine(pt13, pt14) #Left
        l1b = gmsh.model.occ.addLine(pt14, pt11) #Back
        
        l2r = gmsh.model.occ.addLine(pt21, pt22)
        l2f = gmsh.model.occ.addLine(pt22, pt23)
        l2l = gmsh.model.occ.addLine(pt23, pt24)
        l2b = gmsh.model.occ.addLine(pt24, pt21)

        dbr = gmsh.model.occ.addLine(pt11, pt21)
        dfr = gmsh.model.occ.addLine(pt12, pt22)
        dfl = gmsh.model.occ.addLine(pt13, pt23)
        dbl = gmsh.model.occ.addLine(pt14, pt24)

        wbot = gmsh.model.occ.addWire([l1r, l1b, l1l, l1f])
        wtop = gmsh.model.occ.addWire([l2r, l2b, l2l, l2f])
        wright = gmsh.model.occ.addWire([l1r, dbr, l2r, dfr])
        wleft = gmsh.model.occ.addWire([l1l, dbl, l2l, dfl])
        wback = gmsh.model.occ.addWire([l1b, dbl, l2b, dbr])
        wfront = gmsh.model.occ.addWire([l1f, dfr, l2f, dfl])

        bottom_tag = gmsh.model.occ.addSurfaceFilling(wbot)
        top_tag = gmsh.model.occ.addSurfaceFilling(wtop)
        front_tag = gmsh.model.occ.addSurfaceFilling(wfront)
        back_tag = gmsh.model.occ.addSurfaceFilling(wback)
        left_tag = gmsh.model.occ.addSurfaceFilling(wleft)
        right_tag = gmsh.model.occ.addSurfaceFilling(wright)

        sv = gmsh.model.occ.addSurfaceLoop([bottom_tag,
                                            top_tag,
                                            right_tag,
                                            left_tag,
                                            front_tag,
                                            back_tag])

        volume_tag: int = gmsh.model.occ.addVolume([sv,])

        super().__init__(volume_tag)
        #self.tags: list[int] = [volume_tag,]

        pc = p0 + height/2*hax
        self._add_face_pointer('front', pc - depth/2*dax, -dax)
        self._add_face_pointer('back', pc + depth/2*dax, dax)
        self._add_face_pointer('left', pc - width/2*wax, -wax)
        self._add_face_pointer('right', pc + width/2*wax, wax)
        self._add_face_pointer('top', pc + height/2*hax, hax)
        self._add_face_pointer('bottom', pc - height/2*hax, -hax)
    
    def outside(self, *exclude: Literal['bottom','top','right','left','front','back']) -> FaceSelection:
        """Select all outside faces except for the once specified by outside

        Returns:
            FaceSelection: The resultant face selection
        """
        tagslist = [self._face_tags(name) for name in  ['bottom','top','right','left','front','back'] if name not in exclude]
        
        tags = list(reduce(lambda a,b: a+b, tagslist))
        return FaceSelection(tags)


class Cone(GeoVolume):
    """Constructis a cone that starts at position p0 and is aimed in the given direction.
        r1 is the start radius and r2 the end radius. The magnitude of direction determines its length.

        Args:
            p0 (tuple[float, float, float]): _description_
            direction (tuple[float, float, float]): _description_
            r1 (float): _description_
            r2 (float): _description_
        """
    def __init__(self, p0: tuple[float, float, float],
                 direction: tuple[float, float, float],
                 r1: float,
                 r2: float):
        """Constructis a cone that starts at position p0 and is aimed in the given direction.
        r1 is the start radius and r2 the end radius. The magnitude of direction determines its length.

        Args:
            p0 (tuple[float, float, float]): _description_
            direction (tuple[float, float, float]): _description_
            r1 (float): _description_
            r2 (float): _description_
        """
        tag = gmsh.model.occ.add_cone(*p0, *direction, r1, r2)
        super().__init__(tag)
        
        p0 = np.array(p0)
        ds = np.array(direction)
        
        self._add_face_pointer('front', p0, ds)
        if r2>0:
            self._add_face_pointer('back', p0+ds, ds)