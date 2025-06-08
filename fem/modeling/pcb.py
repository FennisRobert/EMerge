from __future__ import annotations

import numpy as np
import gmsh

from ..cs import CoordinateSystem, GCS
from ..geo3d import Polygon, GMSHVolume, GMSHSurface
from ..material import Material, AIR
from .shapes import Box, Plate
from .operations import change_coordinate_system


from typing import Literal

def normalize(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm

class RouteElement:

    def __init__(self):
        self.width: float = None
        self.x: float = None
        self.y: float = None
        self.direction: float = np.ndarray
    
    @property
    def right(self) -> list[tuple[float, float]]:
        pass

    @property
    def left(self) -> list[tuple[float, float]]:
        pass

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

    @property
    def right(self) -> list[tuple[float, float]]:
        return [(self.x + self.width/2 * self.dirright[0], self.y + self.width/2 * self.dirright[1])]

    @property
    def left(self) -> list[tuple[float, float]]:
        return [(self.x - self.width/2 * self.dirright[0], self.y - self.width/2 * self.dirright[1])]
    
def _rot_mat(angle):
    ang = -angle * np.pi/180
    return np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])

class StripTurn(RouteElement):

    def __init__(self,
                 x: float,
                 y: float,
                 width: float,
                 direction: tuple[float, float],
                 angle: float,
                 corner_type: str = 'round',
                 champher_distance: float = None):
        self.xold = x
        self.yold = y
        self.width = width
        self.old_direction = normalize(np.array(direction))
        self.direction = _rot_mat(angle) @ self.old_direction
        self.angle = angle
        self.corner_type: str = corner_type
        self.dirright = np.array([self.old_direction[1], -self.old_direction[0]])
        if champher_distance is None:
            self.champher_distance = 0.75 * self.width*np.tan(np.abs(angle)/2*np.pi/180)
        else:
            self.champher_distance = champher_distance

        turnvec = _rot_mat(angle) @ self.dirright * self.width/2

        if angle > 0:
            self.x = x + width/2 * self.dirright[0] - turnvec[0]
            self.y = y + width/2 * self.dirright[1] - turnvec[1]
        else:
            self.x = x - width/2 * self.dirright[0] + turnvec[0]
            self.y = y - width/2 * self.dirright[1] + turnvec[1]

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

class StripPath:

    def __init__(self, pcb: PCBLayouter):
        self.pcb: PCBLayouter = pcb
        self.path: list[RouteElement] = []
        self.last: RouteElement = None
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
        self.last = element
        return self
    
    def straight(self, distance: 
                 float, width: float = None, 
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
        
        x = self.path[-1].x + dx
        y = self.path[-1].y + dy

        dx_2, dy_2 = self.path[-1].direction
        x1 = x + distance * dx_2
        y1 = y + distance * dy_2

        if width is not None:
            if width != self.end.width:
                self._add_element(StripLine(x, y, width, (dx_2, dy_2)))

        self._add_element(StripLine(x1, y1, self.end.width, (dx_2, dy_2)))
        
        return self
    
    def turn(self, angle: float, 
             width: float = None, 
             corner_type: Literal['champher'] = 'champher') -> StripPath:
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
        x, y = self.path[-1].x, self.path[-1].y
        dx, dy = self.path[-1].direction
        
        if width is not None:
            if width != self.end.width:
                self._add_element(StripLine(x, y, width, (dx, dy)))
        else:
            width=self.last.width
        self._add_element(StripTurn(x, y, width, (dx, dy), angle, corner_type))
        return self
    
    def store(self, name: str) -> StripPath:
        """ Store the current x,y coordinate labeled in the PCB object.

        The stored coordinate can be accessed by calling the .load() method on the PCBRouter class.

        Args:
            name (str): The coordinate label

        Returns:
            StripPath: The current StripPath object.
        """
        self.pcb.store(name, self.last.x, self.last.y)
        return self

    def jump(self, 
             dx: float = None,
             dy: float = None,
             width: float = None,
             direction: tuple[float, float] = None,
             gap: float = None,
             side: Literal['left','right'] = None,
             reverse: float = None) -> StripPath:
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
    
    def __call__(self, element_nr: int) -> RouteElement:
        if element_nr >= len(self.path):
            self.path.append(RouteElement())
        return self.path[element_nr]
    
class PCBLayouter:
    def __init__(self,
                 thickness: float, 
                 hair: float,
                 unit: float = 0.001,
                 cs: CoordinateSystem = None,
                 material: Material = AIR,
                 ):

        self.material: Material = material
        self.width: float = None
        self.length: float = None
        self.thickness: float = thickness
        self.hair: float = hair
        self.origin: np.ndarray = None
        self.paths: list[StripPath] = []

        self.lumped_ports: list[StripLine] = []

        self.last: StripPath = None
        self.unit = unit

        self.cs: CoordinateSystem = cs
        if self.cs is None:
            self.cs = GCS

        self.traces: list[Polygon] = []
        self.ports: list[Polygon] = []

        self.xs: list[float] = []
        self.ys: list[float] = []

        self.stored_coords: dict[str,tuple[float, float]] = dict()
    
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
        return None
    
    @property
    def trace(self) -> Polygon:
        tags = []
        for trace in self.traces:
            tags.extend(trace.tags)
        return Polygon(tags)

    def store(self, name: str, x: float, y:float):
        """Store the x,y coordinate pair one label provided by name

        Args:
            name (str): The corodinate label name
            x (float): The x-coordinate
            y (float): The y-coordinate
        """
        self.stored_coords[name] = (x,y)

    def load(self, name: str) -> tuple[float, float] | None:
        """Acquire the x,y, coordinate associated with the label name.
        
        Args:
            name (str): The name of the x,y coordinate
            
        """
        return self.stored_coords.get(name, None)
    
    def __call__(self, path_nr: int) -> StripPath:
        if path_nr >= len(self.paths):
            self.paths.append(StripPath())
        return self.paths[path_nr]

    @property
    def all_objects(self) -> list[Polygon]:
        return self.traces + self.ports
    
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
        print(self.width, self.length, self.origin)
        
    def gen_pcb(self) -> GMSHVolume:
        """Generate the PCB Block object

        Returns:
            GMSHVolume: The PCB Block
        """
        x0, y0, z0 = self.origin*self.unit
        box = Box(self.width*self.unit, 
                  self.length*self.unit, 
                  self.thickness*self.unit, 
                  position=(x0,y0,z0-self.thickness*self.unit))
        box.material = self.material
        box = change_coordinate_system(box, self.cs)
        return box

    def gen_air(self) -> GMSHVolume:
        """Generate the Air Block object

        This requires that the width, depth and origin are deterimed. This 
        can either be done manually or via the .determine_boudns() method.

        Returns:
            GMSHVolume: The PCB Block
        """
        x0, y0, z0 = self.origin*self.unit
        box = Box(self.width*self.unit, 
                  self.length*self.unit, 
                  self.hair*self.unit, 
                  position=(x0,y0,z0))
        box = change_coordinate_system(box, self.cs)
        return box

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
        self.last = path
        return path
    
    def lumped_port(self, stripline: StripLine) -> None:
        """Generate a lumped-port object to be created.

        Args:
            stripline (StripLine): _description_
        """
        
        xy1 = stripline.right[0]
        xy2 = stripline.left[0]
        z = self._get_z(stripline)

        x1, y1, z1 = self.cs.in_global_cs(xy1[0]*self.unit, xy1[1]*self.unit, z*self.unit - self.thickness*self.unit)
        x2, y2, z2 = self.cs.in_global_cs(xy1[0]*self.unit, xy1[1]*self.unit, z*self.unit )
        x3, y3, z3 = self.cs.in_global_cs(xy2[0]*self.unit, xy2[1]*self.unit, z*self.unit )
        x4, y4, z4 = self.cs.in_global_cs(xy2[0]*self.unit, xy2[1]*self.unit, z*self.unit - self.thickness*self.unit)
        
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
        poly = Polygon([planetag,])
        return poly

    def wave_port(self,
                  point: StripLine,
                  width_multiplier: float = 5.0,
                  height: float = None) -> GMSHSurface:
        """Generate a wave-port as a GMSHSurface.

        The port is placed at the coordinate of the provided stripline. The width
        is determined as a multiple of the stripline width. The height will be 
        extended to the air height from the bottom of the PCB unless a different height is specified.

        Args:
            point (StripLine): The location of the port.
            width_multiplier (float, optional): The width of the port in stripline widths. Defaults to 5.0.
            height (float, optional): The height of the port. Defaults to None.

        Returns:
            GMSHSurface: The GMSHSurface object that can be used for the waveguide.
        """
        if height is None:
            height = (self.thickness + self.hair)
        
        ds = point.dirright
        x0 = point.x - ds[0]*point.width*width_multiplier/2
        y0 = point.y - ds[1]*point.width*width_multiplier/2
        z0 =  - self.thickness
        ax1 = np.array([ds[0], ds[1], 0])*self.unit*point.width*width_multiplier
        ax2 = np.array([0,0,1])*height*self.unit

        plate = Plate(np.array([x0,y0,z0])*self.unit, ax1, ax2)
        plate = change_coordinate_system(plate, self.cs)
        return plate

    def compile_paths(self, merge: bool = False) -> list[Polygon] | GMSHSurface:
        """Compiles the striplines and returns a list of polygons or asingle one.

        The Z=0 argument determines the height of the striplines. Z=0 corresponds to the top of
        the PCB.

        Args:
            merge (bool, optional): Whether to merge the Polygons into a single. Defaults to False.

        Returns:
            list[Polygon] | GMSHSUrface: The output stripline polygons possibly merged if merge = True.
        """
        polys = []
        allx = []
        ally = []

        for path in self.paths:
            z = path.z
            xys = []
            for elemn in path.path:
                xys.extend(elemn.right)
            for element in path.path[::-1]:
                xys.extend(element.left)
            
            ptags = []

            xm, ym = xys[0]
            xys2 = [(xm,ym),]
            
            for x,y in xys[1:]:
                if ((x-xm)**2 + (y-ym)**2)>1e-6:
                    xys2.append((x,y))
                    xm, ym = x, y
                    allx.append(x)
                    ally.append(y)
            
            for x,y in xys2:
                px, py, pz = self.cs.in_global_cs(x*self.unit, y*self.unit, z*self.unit)
                ptags.append(gmsh.model.occ.addPoint(px, py, pz))
            
            ltags = []
            for t1, t2 in zip(ptags[:-1], ptags[1:]):
                ltags.append(gmsh.model.occ.addLine(t1, t2))
            ltags.append(gmsh.model.occ.addLine(ptags[-1], ptags[0]))
            
            tag_wire = gmsh.model.occ.addWire(ltags)
            planetag = gmsh.model.occ.addPlaneSurface([tag_wire,])
            polys.append(Polygon([planetag,]))

        self.xs = allx
        self.ys = ally

        self.traces = polys
        if merge:
            tags = []
            for p in polys:
                tags.extend(p.tags)
            polys = GMSHSurface(tags)
        return polys
                
