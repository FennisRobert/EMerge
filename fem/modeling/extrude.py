import numpy as np
from ..cs import CoordinateSystem
from ..geo3d import GMSHVolume
import gmsh
from typing import Generator
from ..selection import FaceSelection

class XYPolygon:

    def __init__(self, 
                 xs: np.ndarray,
                 ys: np.ndarray):
        """Constructs an XY-plane placed polygon.

        Args:
            xs (np.ndarray): The X-points
            ys (np.ndarray): The Y-points
        """

        self.x: np.ndarray = xs
        self.y: np.ndarray = ys

        if self.x[-1] == self.x[0] and self.y[-1] == self.y[0]:
            self.x = self.x[:-1]
            self.y = self.y[:-1]

        self.N: int = self.x.shape[0]
    
    def iterate(self) -> Generator[tuple[float, float],None, None]:
        """ Iterates over the x,y coordinates as a tuple."""
        for i in range(self.N):
            yield (self.x[i], self.y[i])

    @staticmethod
    def circle(radius: float, 
               dsmax: float = None,
               tolerance: float = None,
               Nsections: int = None):
        """This method generates a segmented circle.

        The number of points along the circumpherence can be specified in 3 ways. By a maximum
        circumpherential length (dsmax), by a radial tolerance (tolerance) or by a number of 
        sections (Nsections).

        Args:
            radius (float): The circle radius
            dsmax (float, optional): The maximum circumpherential angle. Defaults to None.
            tolerance (float, optional): The maximum radial error. Defaults to None.
            Nsections (int, optional): The number of sections. Defaults to None.

        Returns:
            XYPolygon: The XYPolygon object.
        """
        if Nsections is not None:
            N = Nsections+1
        elif dsmax is not None:
            N = int(np.ceil((2*np.pi*radius)/dsmax))
        elif tolerance is not None:
            N = int(np.ceil(2*np.pi/np.arccos(1-tolerance)))

        angs = np.linspace(0,2*np.pi,N)

        xs = radius*np.cos(angs[:-1])
        ys = radius*np.sin(angs[:-1])
        return XYPolygon(xs, ys)

class Prism(GMSHVolume):
    """The prism class generalizes the GMSHVolume for extruded convex polygons.
    Besides having a volumetric definitions, the class offers a .front_face 
    and .back_face property that selects the respective faces.

    Args:
        GMSHVolume (_type_): _description_
    """
    def __init__(self,
                 volume_tag: int,
                 front_tag: int,
                 back_tag: int):
        super().__init__(volume_tag)
        self.front_tag: int = front_tag
        self.back_tag: int = back_tag

    @property
    def front_face(self) -> FaceSelection:
        if self.front_tag is None:
            raise ValueError('Front tag is not defined. Make sure to extrude first.')
        return FaceSelection([self.front_tag,])

    @property
    def back_face(self) -> FaceSelection:
        if self.front_tag is None:
            raise ValueError('Back tag is not defined. Make sure to extrude first.')
        return FaceSelection([self.back_tag,])


class Extrusion:

    def __init__(self, poly: XYPolygon, cs: CoordinateSystem):
        """Generates an Extrusion class object. 
        This requires an XYPolygon object representing the surface to extrude and a 
        CoordinateSystem object that defines the orientation of the extrusion.
        Extusions always happen via an XY-polygon along the Z-axis.

        Args:
            poly (XYPolygon): The surface to extrude
            cs (CoordinateSystem): The CoordinateSystem in which to extrude.
        """
        self.poly: XYPolygon = poly
        self.cs: CoordinateSystem = cs
        self.front_tag: int = None
        self.back_tag: int = None

    

    def extrude_z(self, 
                  z1: float, 
                  z2: float,
                  dz: float = None,
                  N: int = None) -> Prism:
        """Extrues the polygon along the Z-axis.
        The z-coordinates go from z1 to z2 (in meters). Then the extrusion
        is either provided by a maximum dz distance (in meters) or a number
        of sections N.

        Args:
            z1 (float): The start z-coordinate (meters)
            z2 (float): The end z-coordinate (meters)
            dz (float, optional): The z-step size (meters). Defaults to None.
            N (int, optional): The number of steps. Defaults to None.

        Returns:
            GMSHVolume: The resultant Volumetric object.
        """
        
        if dz is not None:
            N = int(np.ceil((z1-z1)/dz))
        N = max(2, N)
        zs = np.linspace(z1, z2, N)
        
        point_lists = []

        Nl = self.poly.N
        # Generate point lists
        for z in zs:
            points = []
            for (x, y) in self.poly.iterate():
                xl1, yl1, zl1 = self.cs.in_global_cs(x, y, z)
                pointtag = gmsh.model.occ.add_point(xl1, yl1, zl1)
                points.append(pointtag)
            point_lists.append(points)
        loop_edges = []
        connecting_edges = []
        for point_tags in point_lists:
            loop_tags = []
            for i in range(Nl):
                t1 = point_tags[i]
                t2 = point_tags[(i+1)%Nl]
                line_tag = gmsh.model.occ.add_line(t1, t2)
                loop_tags.append(line_tag)
            loop_edges.append(loop_tags)
        for points1, points2 in zip(point_lists[:-1], point_lists[1:]):
            loop_tags = []
            for i in range(Nl):
                t1 = points1[i]
                t2 = points2[i]
                line_tag = gmsh.model.occ.add_line(t1, t2)
                loop_tags.append(line_tag)
            connecting_edges.append(loop_tags)
        surfs = []

        for botloop, toploop, conloop in zip(loop_edges[:-1], 
                                             loop_edges[1:], 
                                             connecting_edges):
            for i in range(Nl):
                et1 = botloop[i]
                et2 = conloop[(i+1) % Nl]
                et3 = toploop[i]
                et4 = conloop[i]
                wt = gmsh.model.occ.add_wire([et1, et2, -et3, -et4])
                st = gmsh.model.occ.add_plane_surface([wt,])
                surfs.append(st)
        botwire = gmsh.model.occ.add_wire(loop_edges[0])
        topwire = gmsh.model.occ.add_wire(loop_edges[-1])
        front_tag = gmsh.model.occ.add_plane_surface([botwire,])
        back_tag = gmsh.model.occ.add_plane_surface([topwire,])
        surfs.append(front_tag)
        surfs.append(back_tag)
        volloop = gmsh.model.occ.add_surface_loop(surfs)
        voltag = gmsh.model.occ.add_volume([volloop,])
        
        return Prism(voltag, front_tag, back_tag)