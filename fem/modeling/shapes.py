import gmsh
from ..geo3d import GMSHObject, GMSHSurface, GMSHVolume
from ..cs import CoordinateSystem
import numpy as np
from enum import Enum
from .operations import subtract
from ..selection import FaceSelection, Selector, SELECTOR_OBJ

class Alignment(Enum):
    CENTER = 1
    CORNER = 2

class Box(GMSHVolume):

    def __init__(self, 
                 width: float, 
                 depth: float, 
                 height: float, 
                 position: tuple = (0,0,0),
                 alignment: Alignment = Alignment.CORNER):
        super().__init__([])
        if alignment is Alignment.CENTER:
            position = (position[0]-width/2, position[1]-depth/2, position[2]-height/2)
        
        x,y,z = position
        self.tags: list[int] = [gmsh.model.occ.addBox(x,y,z,width,depth,height),]
        self.centre = (x+width/2, y+depth/2, z+height/2)
        self.width = width
        self.height = height
        self.depth = depth

    @property
    def front(self) -> FaceSelection:
        return SELECTOR_OBJ.face.near(self.centre[0], self.centre[1]-self.depth/2, self.centre[2])
    
    @property
    def back(self) -> FaceSelection:
        return SELECTOR_OBJ.face.near(self.centre[0], self.centre[1]+self.depth/2, self.centre[2])
    
    @property
    def left(self) -> FaceSelection:
        return SELECTOR_OBJ.face.near(self.centre[0]-self.width/2, self.centre[1], self.centre[2])
    
    @property
    def right(self) -> FaceSelection:
        return SELECTOR_OBJ.face.near(self.centre[0]+self.width/2, self.centre[1], self.centre[2])
    
    @property
    def bottom(self) -> FaceSelection:
        return SELECTOR_OBJ.face.near(self.centre[0], self.centre[1], self.centre[2]-self.height/2)
    
    @property
    def top(self) -> FaceSelection:
        return SELECTOR_OBJ.face.near(self.centre[0], self.centre[1], self.centre[2]+self.height/2)
    
class Sphere(GMSHVolume):

    def __init__(self, 
                 radius: float,
                 position: tuple = (0,0,0)):
        super().__init__([])
        x,y,z = position
        self.tags: list[int] = [gmsh.model.occ.addSphere(x,y,z,radius),]

class XYPlate(GMSHSurface):
    def __init__(self, 
                 width: float, 
                 depth: float, 
                 position: tuple = (0,0,0),
                 alignment: Alignment = Alignment.CORNER):
        super().__init__([])
        if alignment is Alignment.CENTER:
            position = (position[0]-width/2, position[1]-depth/2, position[2])
        
        x,y,z = position
        self.tags: list[int] = [gmsh.model.occ.addRectangle(x,y,z,width,depth),]


class Plate(GMSHSurface):
        
    def __init__(self,
                origin: tuple[float, float, float],
                ax1: tuple[float, float, float],
                ax2: tuple[float, float, float]):
        super().__init__([])
        origin = np.array(origin)
        ax1 = np.array(ax1)
        ax2 = np.array(ax2)
        
        tagp1 = gmsh.model.occ.addPoint(*origin)
        tagp2 = gmsh.model.occ.addPoint(*(origin+ax1))
        tagp3 = gmsh.model.occ.addPoint(*(origin+ax2))
        tagp4 = gmsh.model.occ.addPoint(*(origin+ax1+ax2))

        tagl1 = gmsh.model.occ.addLine(tagp1, tagp2)
        tagl2 = gmsh.model.occ.addLine(tagp2, tagp4)
        tagl3 = gmsh.model.occ.addLine(tagp4, tagp3)
        tagl4 = gmsh.model.occ.addLine(tagp3, tagp1)

        tag_wire = gmsh.model.occ.addWire([tagl1,tagl2, tagl3, tagl4])

        self.tags: list[int] = [gmsh.model.occ.addPlaneSurface([tag_wire,]),]


class Cyllinder(GMSHVolume):

    def __init__(self, 
                 radius: float,
                 height: float,
                 cs: CoordinateSystem = None,):
        ax = cs.zax.np
        cyl = gmsh.model.occ.addCylinder(cs.origin[0], cs.origin[1], cs.origin[2],
                                         height*ax[0], height*ax[1], height*ax[2],
                                         radius)
        self.cs: CoordinateSystem = cs
        self.radius = radius
        self.height = height
        super().__init__(cyl)

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

class CoaxCyllinder(GMSHVolume):
    """A coaxial cylinder with an inner and outer radius."""
    
    def __init__(self, 
                 rout: float,
                 rin: float,
                 height: float,
                 cs: CoordinateSystem = None,):
        if rout <= rin:
            raise ValueError("Outer radius must be greater than inner radius.")
        
        self.rout = rout
        self.rin = rin
        self.height = height
        self.cyl_out = Cyllinder(rout, height, cs)
        self.cyl_in = Cyllinder(rin, height, cs)
        cyltags, _ = gmsh.model.occ.cut(self.cyl_out.dimtags, self.cyl_in.dimtags)
        super().__init__([dt[1] for dt in cyltags])

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
        return super().boundary()
    
class HalfSphere(GMSHVolume):

    def __init__(self,
                 radius: float,
                 position: tuple = (0,0,0),
                 direction: tuple = (1,0,0)):
        super().__init__([])
        sphere = Sphere(radius, position=position)
        cx, cy, cz = position

        box = Box(1.1*radius, 2.2*radius, 2.2*radius, position=(cx-radius*1.1,cy-radius*1.1, cz-radius*1.1))

        self.tag = subtract(sphere, box)[0].tag

