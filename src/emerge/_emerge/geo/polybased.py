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
import numpy as np
from ..cs import CoordinateSystem, GCS, Axis, _parse_axis, _parse_vector, Anchor
from ..geometry import GeoVolume, GeoPolygon, GeoEdge, GeoSurface
from .shapes import Alignment
import gmsh
from typing import Generator, Callable
from ..selection import FaceSelection
from typing import Literal
from functools import reduce
from loguru import logger


def _discretize_curve(xfunc: Callable, 
                      yfunc: Callable, 
                      t0: float, 
                      t1: float, 
                      xmin: float, 
                      tol: float=1e-4) -> tuple[np.ndarray, np.ndarray]:
    """Computes a discreteized curve in X/Y coordinates based on the input parametric coordinates.

    Args:
        xfunc (Callable): The X-coordinate function fx(t)
        yfunc (Callable): The Y-coordinate function fy(t)
        t0 (float): The minimum value for the t-prameter
        t1 (float): The maximum value for the t-parameter
        xmin (float): The minimum distance for subsequent points
        tol (float, optional): The curve matching tolerance. Defaults to 1e-4.

    Returns:
        tuple[np.ndarray, np.ndarray]: _description_
    """
    from ..mth.optimized import _subsample_coordinates
    
    td = np.linspace(t0, t1, 10_001)
    xs = xfunc(td)
    ys = yfunc(td)
    XS, YS = _subsample_coordinates(xs, ys, tol, xmin)
    return XS, YS

def rotate_point(point: tuple[float, float, float],
                 axis: tuple[float, float, float],
                 ang: float,
                 origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
                 degrees: bool = False) -> tuple[float, float, float]:
    """
    Rotate a 3‑D point around an arbitrary axis that passes through `origin`.

    Args:
        point (tuple[float, float, float]): (x, y, z) coordinate of the point to rotate.
        axis (tuple[float, float, float]): (ux, uy, uz) direction vector of the rotation axis (need not be unit length).
        ang (float): rotation angle.  Positive values follow the right‑hand rule.
        origin (tuple[float, float, float]): (ox, oy, oz) point through which the axis passes.  Defaults to global origin.
        degrees (bool): If True, `ang` is interpreted in degrees; otherwise in radians.

    Returns:
        (x,y,z) : tuple with the rotated coordinates.
    """

    p = np.asarray(point, dtype=float)
    o = np.asarray(origin, dtype=float)
    u = np.asarray(axis, dtype=float)

    p_shifted = p - o

    norm = np.linalg.norm(u)
    if norm == 0:
        raise ValueError("Axis direction vector must be non‑zero.")
    u = u / norm

    if degrees:
        ang = np.radians(ang)

    cos_a = np.cos(ang)
    sin_a = np.sin(ang)
    cross = np.cross(u, p_shifted)
    dot = np.dot(u, p_shifted)

    rotated = (p_shifted * cos_a
               + cross * sin_a
               + u * dot * (1 - cos_a))
    rotated += o
    return tuple(rotated)
            
def orthonormalize(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generates a set of orthonormal vectors given an input vector X

    Args:
        axis (np.ndarray): The X-axis

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: The X, Y and Z axis (orthonormal)
    """
    Xaxis = axis/np.linalg.norm(axis)
    V = np.array([0,1,0])
    if 1-np.abs(np.dot(Xaxis, V)) < 1e-12:
        V = np.array([0,0,1])
    Yaxis = np.cross(Xaxis, V)
    Yaxis = np.abs(Yaxis/np.linalg.norm(Yaxis))
    Zaxis = np.cross(Xaxis, Yaxis)
    Zaxis = np.abs(Zaxis/np.linalg.norm(Zaxis))
    return Xaxis, Yaxis, Zaxis


class GeoPrism(GeoVolume):
    """The GepPrism class generalizes the GeoVolume for extruded convex polygons.
    Besides having a volumetric definitions, the class offers a .front_face 
    and .back_face property that selects the respective faces.

    Args:
        GeoVolume (_type_): _description_
    """
    def __init__(self,
                 volume_tag: int,
                 front_tag: int | None = None,
                 side_tags: list[int] | None = None,
                 _axis: Axis | None = None,
                 name: str | None = None):
        super().__init__(volume_tag, name=name)
        
        
        
        if front_tag is not None and side_tags is not None:
            self.front_tag: int = front_tag
            self.back_tag: int = None

            gmsh.model.occ.synchronize()
            self._add_face_pointer('back', tag=self.front_tag)

            tags = gmsh.model.get_boundary(self.dimtags, oriented=False)
            
            for dim, tag in tags:
                if (dim,tag) in side_tags:
                    continue
                self._add_face_pointer('front',tag=tag)
                self.back_tag = tag
                break

            self.side_tags: list[int] = [dt[1] for dt in tags if dt[1]!=self.front_tag and dt[1]!=self.back_tag]

            for tag in self.side_tags:
    
                self._add_face_pointer(f'side{tag}', tag=tag)
                self.back_tag = tag
                
        elif _axis is not None:
            _axis = _parse_axis(_axis)
            gmsh.model.occ.synchronize()
            tags = gmsh.model.get_boundary(self.dimtags, oriented=False)
            faces = []
            for dim, tag in tags:
                o1 = np.array(gmsh.model.occ.get_center_of_mass(2, tag))
                n1 = np.array(gmsh.model.get_normal(tag, (0,0)))
                if abs(np.sum(n1*_axis.np)) > 0.99:
                    dax = sum(o1 * _axis.np)
                    faces.append((o1, n1, dax, tag))
            
            faces = sorted(faces, key=lambda x: x[2])
            ftags = []
            if len(faces) >= 2:
                ftags.append(faces[0][3])
                ftags.append(faces[-1][3])
                self._add_face_pointer('front',faces[0][0], faces[0][1])
                self._add_face_pointer('back', faces[-1][0], faces[-1][1])
            elif len(faces)==1:
                ftags.append(faces[0][3])
                self._add_face_pointer('cap',faces[0][0], faces[0][1])
            
            ictr = 1
            for dim, tag in tags:
                if tag in ftags:
                    continue
                self._add_face_pointer(f'side{ictr}', tag=tag)
                ictr += 1

    def outside(self, *exclude: Literal['front','back']) -> FaceSelection:
        """Select all outside faces except for the once specified by outside

        Returns:
            FaceSelection: The resultant face selection
        """
        tagslist = [self._face_tags(name) for name in  self._face_pointers.keys() if name not in exclude]
        
        tags = list(reduce(lambda a,b: a+b, tagslist))
        return FaceSelection(tags)
    
    @property
    def front(self) -> FaceSelection:
        """The first local -Z face of the prism."""
        return self.face('front')
    
    @property
    def back(self) -> FaceSelection:
        """The back local +Z face of the prism."""
        return self.face('back')
    
    @property
    def sides(self) -> FaceSelection:
        """The outside faces excluding the top and bottom."""
        return self.boundary(exclude=('front','back'))

class XYPolygon:
    """This class generalizes a polygon in an un-embedded local 2D XY-space that can be embedded in 3D space.
    This class is not necessarily restricted to the 3D XY plane. Any function like .geo() can accept coordinate systems
    that may be rotate or otherwise oriented so as to put the XYPolygon in any arbitrary orientation.

    """
    def __init__(self, 
                 xs: np.ndarray | list[float] | tuple[float,...] | None = None,
                 ys: np.ndarray | list[float] | tuple[float,...] | None = None,
                 cs: CoordinateSystem | None = None,
                 resolution: float = 1e-6):
        """Constructs an XY-plane placed polygon.

        The resolution parameter will define if points are removed if they are too close. Default limit is 1 um.
        Args:
            xs (np.ndarray, optional): The starting stet of X-points
            ys (np.ndarray, optional): The starting stet of Y-points
            cs (CoordinateSystem, optional): The coordinate system to put the XYPolygon in. Optional can be provided later.
            resolution (float, optional): The numerical resolution to use for polygon simplification. 1e-6 is the default.
        """
        if xs is None:
            xs = []
        if ys is None:
            ys = []

        self.x: np.ndarray = np.asarray(xs)
        self.y: np.ndarray = np.asarray(ys)

        self.fillets: list[tuple[float, int]] = []
        
        self._cs: CoordinateSystem = cs
        self.resolution: float = resolution

    @property
    def center(self) -> tuple[float, float]:
        """ Center point of the vertices. """
        return np.mean(self.x), np.mean(self.y)
    
    @property
    def length(self):
        """ Total length of the polygon path"""
        return sum([((self.x[i2]-self.x[i1])**2 + (self.y[i2]-self.y[i1])**2)**0.5 for i1, i2 in zip(range(self.N-1),range(1, self.N))])
        
    @property
    def N(self) -> int:
        """The number of polygon points

        Returns:
            int: The number of points
        """
        return len(self.x)
    
    def _check(self) -> None:
        """Checks if the last point is the same as the first point.
        The XYPolygon does not store redundant points p[0]==p[N] so if these are
        the same, this function will remove the last point.
        """
        if np.sqrt((self.x[-1]-self.x[0])**2 + (self.y[-1]-self.y[0])**2) < 1e-6:
            self.x = self.x[:-1]
            self.y = self.y[:-1]
        
    @property
    def area(self) -> float:
        """The Area of the polygon

        Returns:
            float: The area in square meters
        """
        return 0.5*np.abs(np.dot(self.x,np.roll(self.y,1))-np.dot(self.y,np.roll(self.x,1)))

    def incs(self, cs: CoordinateSystem) -> XYPolygon:
        """ Sets a coordinate system and returns the same XYPolygon object. """
        self._cs = cs
        return self

    def extend(self, xpts: list[float], ypts: list[float]) -> XYPolygon:
        """Adds a series for x and y coordinates to the existing polygon.

        Args:
            xpts (list[float]): The list of x-coordinates
            ypts (list[float]): The list of y-coordinates

        Returns:
            XYPolygon: The same XYpolygon object
        """
        self.x = np.hstack([self.x, np.array(xpts)])
        self.y = np.hstack([self.y, np.array(ypts)])
        return self
    
    def iterate(self) -> Generator[tuple[float, float],None, None]:
        """ Iterates over the x,y coordinates as a tuple."""
        for i in range(self.N):
            yield (self.x[i], self.y[i])

    def fillet(self, radius: float, *indices: int) -> XYPolygon:
        """Add a fillet rounding with a given radius to the provided nodes.

        Example:
         >>> my_polygon.fillet(0.05, 2, 3, 4, 6)

        Args:
            radius (float): The radius
            *indices (int): The indices for which to apply the fillet.
        """
        for i in indices:
            self.fillets.append((radius, i))
        return self

    def _cleanup(self, resolution: float | None = None) -> None:
        """ Cleanup routine to simplify the polygon. Removes points that are closer than the polygon resolution. """
        if resolution is None:
            resolution = self.resolution
        dx = np.diff(self.x)
        dy = np.diff(self.y)
        dist = np.sqrt(dx**2 + dy**2)
        keep = np.insert(dist >= resolution, 1, True)
        self.x = self.x[keep]
        self.y = self.y[keep]
        
    def _make_wire(self, cs: CoordinateSystem) -> tuple[list[int], list[int], int]:
        """Turns the XYPolygon object into a GeoPolygon that is embedded in 3D space.

        The polygon will be placed in the XY-plane of the provided coordinate center.

        Args:
            cs (CoordinateSystem, optional): The coordinate system in which to put the polygon. Defaults to None.

        Returns:
            GeoPolygon: The resultant 3D GeoPolygon object.
        """
        self._check()

        ptags = []
        
        # Original global transformation preserved
        xg, yg, zg = cs.in_global_cs(self.x, self.y, 0*self.x)
        
        points = dict()
        for x, y, z in zip(xg, yg, zg):
            reuse = False
            for key, (px, py, pz) in points.items():
                if ((x-px)**2 + (y-py)**2 + (z-pz)**2)**0.5 < 1e-12:
                    ptags.append(key)
                    reuse = True
                    break
            if reuse:
                logger.warning(f'Reusing {ptags[-1]}')
                continue
            ptag = gmsh.model.occ.add_point(x, y, z)
            points[ptag] = (x, y, z)
            ptags.append(ptag)
        
        lines = []
        num_pts = len(ptags)
        for i in range(num_pts):
            p1 = ptags[i]
            p2 = ptags[(i + 1) % num_pts]
            lines.append(gmsh.model.occ.add_line(p1, p2))
        
        # Map indices to radii for O(1) lookup
        fillet_map = {index: radius for radius, index in self.fillets}
        final_edges = []

        for i in range(num_pts):
            # Vertex i always connects the end of line(i-1) to the start of line(i)
            t1 = lines[(i - 1) % num_pts]
            t2 = lines[i]

            if i in fillet_map:
                radius = fillet_map[i]
                # OCC trims t1 and t2 in place, and returns the newly created arc
                arc_tag = gmsh.model.occ.fillet2_d(t1, t2, radius)
                final_edges.append(arc_tag)
            
            # The outgoing line t2 is always appended after its preceding vertex
            final_edges.append(t2)

        wiretag = gmsh.model.occ.add_wire(final_edges)
        return ptags, final_edges, wiretag
        
    def _finalize(self, cs: CoordinateSystem, name: str | None = 'GeoPolygon') -> GeoPolygon:
        """Turns the XYPolygon object into a GeoPolygon that is embedded in 3D space.

        The polygon will be placed in the XY-plane of the provided coordinate center.

        Args:
            cs (CoordinateSystem, optional): The coordinate system in which to put the polygon. Defaults to None.

        Returns:
            GeoPolygon: The resultant 3D GeoPolygon object.
        """
        self._cleanup()
        ptags, lines, wiretag = self._make_wire(cs)
        surftag = gmsh.model.occ.add_plane_surface([wiretag,])
        gmsh.model.occ.remove([(1,wiretag),]+[(1,t) for t in lines], recursive=True)
        poly = GeoPolygon([surftag,], name=name)
        poly.points = ptags
        poly.lines = lines
        return poly
    
    def extrude(self, length: float, cs: CoordinateSystem | None = None, name: str = 'Extrusion') -> GeoPrism:
        """Extrues the polygon along the Z-axis.
        The z-coordinates go from z1 to z2 (in meters). Then the extrusion
        is either provided by a maximum dz distance (in meters) or a number
        of sections N.

        Args:
            length (length): The length of the extrusion.

        Returns:
            GeoVolume: The resultant Volumetric object.
        """
        if cs is None:
            cs = GCS
        poly_fin = self._finalize(cs)
        zax = length*cs.zax.np
        poly_fin._exists = False
        volume = gmsh.model.occ.extrude(poly_fin.dimtags, zax[0], zax[1], zax[2])
        tags = [t for d,t in volume if d==3]
        surftags = [t for d,t in volume if d==2]
        return GeoPrism(tags, surftags[0], surftags, name=name)
    
    def geo(self, cs: CoordinateSystem | None = None, name: str = 'GeoPolygon') -> GeoPolygon:
        """Returns a GeoPolygon object for the current polygon.

        Args:
            cs (CoordinateSystem, optional): The Coordinate system of which the XY plane will be used. Defaults to None.

        Returns:
            GeoPolygon: The resultant object.
        """
        if self._cs is not None:
            return self._finalize(self._cs)
        if cs is None:
            cs = GCS
        return self._finalize(cs) 
    
    def revolve(self, cs: CoordinateSystem, 
                origin: tuple[float, float, float] | Anchor, 
                axis: tuple[float, float,float] | Axis, 
                angle: float = 360.0, 
                name: str = 'Revolution') -> GeoPrism:
        """Applies a revolution to the XYPolygon along the provided rotation ais

        Args:
            cs (CoordinateSystem, optional): _description_. Defaults to None.
            origin: tuple[float, float, float]: The origin of the revolution axis
            axis: tuple[float, float, float] | Axis: The revolution axis.
            angle: float - The revolution angle. Defaults to 360 degrees
            angle (float, optional): _description_. Defaults to 360.0.

        Returns:
            Prism: The resultant 
        """
        
        if cs is None:
            cs = GCS
        poly_fin = self._finalize(cs)
        
        x,y,z = _parse_vector(origin)
        ax, ay, az = _parse_axis(axis).tuple
        
        volume = gmsh.model.occ.revolve(poly_fin.dimtags, x,y,z, ax, ay, az, angle*np.pi/180)
        
        tags = [t for d,t in volume if d==3]
        poly_fin.remove()
        return GeoPrism(tags, _axis=axis, name=name)

    @staticmethod
    def circle(radius: float, 
               dsmax: float | None= None,
               tolerance: float | None = None,
               Nsections: int | None = None):
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

    @staticmethod
    def rect(width: float,
             height: float,
             origin: tuple[float, float] | Anchor,
             alignment: Alignment = Alignment.CORNER) -> XYPolygon:
        """Create a rectangle in the XY-plane as polygon

        Args:
            width (float): The width (X)
            height (float): The height (Y)
            origin (tuple[float, float]): The origin (x,y)
            alignment (Alignment, optional): What point the origin describes. Defaults to Alignment.CORNER.

        Returns:
            XYPolygon: A new XYpolygon object
        """
        origin = _parse_vector(origin)[:-1]
        
        if alignment is Alignment.CORNER:
            x0, y0 = origin
        else:
            x0 = origin[0]-width/2
            y0 = origin[1]-height/2
        xs = np.array([x0, x0, x0 + width, x0+width])
        ys = np.array([y0, y0+height, y0+height, y0])
        return XYPolygon(xs, ys)
    
    def parametric(self, 
                   xfunc: Callable,
                   yfunc: Callable,
                   xmin: float = 1e-3,
                   tolerance: float = 1e-5,
                   tmin: float = 0,
                   tmax: float = 1,
                   reverse: bool = False) -> XYPolygon:
        """Adds the points of a parametric curve to the polygon.
        The parametric curve is defined by two parametric functions of a parameter t that (by default) lives in the interval from [0,1].
        thus the curve x(t) = xfunc(t), and y(t) = yfunc(t).

        The tolerance indicates a maximum deviation from the true path.

        Args:
            xfunc (Callable): The x-coordinate function.
            yfunc (Callable): The y-coordinate function
            tolerance (float): A maximum distance tolerance. Defaults to 10um.
            tmin (float, optional): The start value of the t-parameter. Defaults to 0.
            tmax (float, optional): The end value of the t-parameter. Defaults to 1.
            reverse (bool, optional): Reverses the curve.

        Returns:
            XYPolygon: _description_
        """
        xs, ys = _discretize_curve(xfunc, yfunc, tmin, tmax, xmin, tolerance)

        if reverse:
            xs = xs[::-1]
            ys = ys[::-1]
        self.extend(xs, ys)
        return self
    
    def corrugated_line(self, 
                        direction: tuple[float, float, float] | Axis,
                        depth: float,
                        total_length: float,
                        period: float,
                        corr_axis: tuple[float, float] | Axis | None = None,
                        side: Literal['left','right'] | None = None,
                        duty_cycle: float = 0.5,
                        offset: float = 0.0,) -> XYPolygon:
        """Add a corrugating line to the polygon path

        Args:
            direction (tuple[float, float, float] | Axis): The direction of the line
            depth (float): The depth of the corrugations
            total_length (float): The total length of the line segment [meters]
            period (float): The corrugation period in meters.
            corr_axis (tuple[float, float] | Axis | None, optional): The corrugation axis direction of displacement (unit length). Defaults to None.
            side (Literal[left, right] | None, optional): Alternative definition of the corrugation direction (left or right). Defaults to None.
            duty_cycle (float, optional): The duty cycle percentage (defaults to 50% or 0.5). Defaults to 0.5.
            offset (float, optional): The offset of the first corrugation in percentage of a single period. Defaults to 0.0.

        Returns:
            XYPolygon: _description_
        """
        ds = _parse_vector(direction)
        if corr_axis is not None:
            dcorr = _parse_vector(corr_axis)
        else:
            if side=='left':
                dcorr = np.array([-ds[1], ds[0]])
            else:
                dcorr = np.array([ds[1], -ds[0]])
        ds = ds/np.linalg.norm(ds)
        dcorr = dcorr/np.linalg.norm(dcorr)
        p0x = self.x[-1]
        p0y = self.y[-1]
        p0cx = depth*dcorr[0]
        p0cy = depth*dcorr[1]
        xs = []
        ys = []
        N = int(np.floor(total_length/period))
        offx = 0.0
        offy = 0.0
        dsx = ds[0]*period
        dsy = ds[1]*period
        
        dx0 = offset*dsx
        dy0 = offset*dsy
        tmplx = [dx0, dx0 + p0cx, dx0 + p0cx + duty_cycle*dsx, dx0 + duty_cycle*dsx]
        tmply = [dy0, dy0 + p0cy, dy0 + p0cy + duty_cycle*dsy, dy0 + duty_cycle*dsy]
        for n in range(N):
            for x0, y0 in zip(tmplx, tmply):
                xs.append(x0+p0x+offx)
                ys.append(y0+p0y+offy)
            offx += dsx
            offy += dsy
        xs.append(p0x + ds[0]*total_length)
        ys.append(p0y + ds[1]*total_length)
        self.extend(xs, ys)
        return self
        
    def connect(self, other: XYPolygon, name: str = 'Connection') -> GeoVolume:
        """Connect two XYPolygons with a defined coordinate system

        The coordinate system must be defined before this function can be used. To add a coordinate systme without
        rendering the Polygon to a GeoVolume, use:
        >>> polygon.incs(my_cs_obj)
        
        Args:
            other (XYPolygon): _descrThe otheiption_

        Returns:
            GeoVolume: The resultant volume object
        """
        if self._cs is None:
            raise RuntimeError('Cannot connect XYPolygons without a defined coordinate system. Set this first using .incs()')
        if other._cs is None:
            raise RuntimeError('Cannot connect XYPolygons without a defined coordinate system. Set this first using .incs()')
        p1, l1, w1 = self._make_wire(self._cs)
        p2, l2, w2 = other._make_wire(other._cs)
        o1 = np.array(self._cs.in_global_cs(*self.center, 0)).flatten()
        o2 = np.array(other._cs.in_global_cs(*other.center, 0)).flatten()
        dts = gmsh.model.occ.addThruSections([w1, w2], True, True, parametrization="IsoParametric")
        vol = GeoVolume([t for d,t in dts if d==3], name=name)
        
        vol._add_face_pointer('front',o1, self._cs.zax.np)
        vol._add_face_pointer('back', o2, other._cs.zax.np)
        return vol
    
    def shrink(self, distance: float) -> XYPolygon:
        """Shrinks (inward offsets) the polygon by translating each edge inward
        by the given distance, then recomputing vertices from adjacent edge intersections.

        The polygon must have a consistent winding order. The inward direction is
        determined from the signed area (positive area = CCW winding = inward normals
        point left of each edge direction).

        Args:
            distance (float): The offset distance to shrink inward.

        Returns:
            XYPolygon: A new shrunk XYPolygon.
        """
        N = self.N
        if N < 3:
            raise ValueError("Need at least 3 points to shrink a polygon.")

        # Signed area to determine winding direction
        signed_area = 0.5 * (np.dot(self.x, np.roll(self.y, -1)) - np.dot(self.y, np.roll(self.x, -1)))
        # If signed_area > 0, winding is CCW and inward normal is to the left of edge direction.
        # If signed_area < 0, winding is CW and inward normal is to the right.
        sign = 1.0 if signed_area > 0 else -1.0

        # For each edge, compute the inward-offset line (stored as a point + direction,
        # or equivalently two offset points).
        # Edge i goes from vertex i to vertex (i+1) % N.
        offset_edges = []  # Each entry: (x1, y1, x2, y2) of the offset edge
        for i in range(N):
            j = (i + 1) % N
            dx = self.x[j] - self.x[i]
            dy = self.y[j] - self.y[i]
            length = np.sqrt(dx**2 + dy**2)
            # Inward normal: for CCW, left normal is (-dy, dx)
            nx = -dy / length * sign * distance
            ny =  dx / length * sign * distance
            offset_edges.append((
                self.x[i] + nx, self.y[i] + ny,
                self.x[j] + nx, self.y[j] + ny
            ))

        # Intersect consecutive offset edges to find new vertices
        new_xs = np.empty(N)
        new_ys = np.empty(N)
        for i in range(N):
            j = (i + 1) % N
            # Line 1: offset_edges[i], Line 2: offset_edges[j]
            x1, y1, x2, y2 = offset_edges[i]
            x3, y3, x4, y4 = offset_edges[j]
            # Intersect using parametric form:
            #   P = (x1,y1) + t*(x2-x1, y2-y1)
            #   Q = (x3,y3) + s*(x4-x3, y4-y3)
            d1x, d1y = x2 - x1, y2 - y1
            d2x, d2y = x4 - x3, y4 - y3
            denom = d1x * d2y - d1y * d2x
            if abs(denom) < 1e-15:
                # Parallel edges — just use the shared endpoint of the offset edges
                new_xs[j] = x2
                new_ys[j] = y2
            else:
                t = ((x3 - x1) * d2y - (y3 - y1) * d2x) / denom
                new_xs[j] = x1 + t * d1x
                new_ys[j] = y1 + t * d1y

        return XYPolygon(new_xs, new_ys, cs=self._cs, resolution=self.resolution)
            
class Disc(GeoSurface):
    _default_name: str = 'Disc'
    
    def __init__(self, 
                 origin: tuple[float, float, float] | Anchor,
                 radius: float,
                 axis: tuple[float, float, float] | Axis = (0.,0.,1.),
                 radius_opt: float | None = None,
                 axis_opt: tuple[float, float, float] | None = None,
                 name: str | None = None):
        """Creates a circular Disc surface.

        Args:
            origin (tuple[float, float, float]): The center of the disc
            radius (float): The radius of the disc
            axis (tuple[float, float, float], optional): The disc normal axis. Defaults to (0,0,1.0).
            radius_opt (float, None): Secondary radius in case where one wants to make an ellipse.
        """
        origin = _parse_vector(origin)
        axis = _parse_vector(axis)
        
        if radius_opt is None:
            radius_opt = radius
            axis_opt = []
        else:
            if axis_opt is None:
                raise ValueError('A secondary axis is required when making an ellipse')
        
        disc = gmsh.model.occ.addDisk(*origin, radius, radius_opt, zAxis=axis, xAxis=axis_opt)
        super().__init__(disc, name=name)
    
    
class Curve(GeoEdge):
    _default_name: str = 'Curve'
    
    def __init__(self, 
                 xpts: np.ndarray, 
                 ypts: np.ndarray, 
                 zpts: np.ndarray, 
                 degree: int = 3,
                 weights: list[float] | None = None,
                 knots: list[float] | None = None,
                 multiplicities: list[float] | None = None,
                 ctype: Literal['Spline','BSpline','Bezier'] = 'Spline',
                 name: str | None = None):
        """Generate a Spline/Bspline or Bezier curve based on a series of points

        This calls the different curve features in OpenCASCADE.
        
        The dstart parameter defines the departure direction of the curve. If not provided this is inferred as the
        discrete derivative from the first to second coordinate. 
        
        Args:
            xpts (np.ndarray): The X-coordinates
            ypts (np.ndarray): The Y-coordinates
            zpts (np.ndarray): The Z-coordinates
            degree (int, optional): The BSpline degree parameter. Defaults to 3.
            weights (list[float] | None, optional): An optional point weights list. Defaults to None.
            knots (list[float] | None, optional): A nkots list. Defaults to None.
            ctype (Literal['Spline','BSpline','Bezier'], optional): The type of curve. Defaults to 'Spline'.
            dstart (tuple[float, float, float] | None, optional): The departure direction. Defaults to None.
        """
        self.xpts: np.ndarray = xpts
        self.ypts: np.ndarray = ypts
        self.zpts: np.ndarray = zpts

        points = [gmsh.model.occ.add_point(x,y,z) for x,y,z in zip(xpts, ypts, zpts)]
        
        if ctype.lower()=='spline':
            tags = gmsh.model.occ.addSpline(points)
            
        elif ctype.lower()=='bspline':
            if weights is None:
                weights = []
            if knots is None:
                knots = []
            tags = gmsh.model.occ.addBSpline(points, degree=degree, weights=weights, knots=knots, multiplicities=multiplicities)
        else:
            tags = gmsh.model.occ.addBezier(points)
        
        tags = gmsh.model.occ.addWire([tags,])
        gmsh.model.occ.remove([(0,tag) for tag in points])
        super().__init__(tags, name=name)
    
        gmsh.model.occ.synchronize()
        # 2. Compute analytical initial tangent from P1 - P0 (Always exact for Bezier/Splines)
        dx = float(xpts[1] - xpts[0])
        dy = float(ypts[1] - ypts[0])
        dz = float(zpts[1] - zpts[0])
        self.dstart: tuple[float, float, float] = (dx, dy, dz)  
    
        
    @property
    def p0(self) -> tuple[float, float, float]:
        """The start coordinate
        """
        return (self.xpts[0], self.ypts[0], self.zpts[0])

    @staticmethod
    def helix_rh(
              pstart: tuple[float, float, float] | Anchor,
              pend: tuple[float, float, float] | Anchor,
              r_start: float,
              pitch: float,
              r_end: float | None = None,
              _narc: int = 8,
              startfeed: float = 0.0) -> Curve:
        """Generates a Helical curve

        Args:
            pstart (tuple[float, float, float]): The start of the center of rotation (not the start of the curve)
            pend (tuple[float, float, float]): The end of the center of rotation
            r_start (float): The (start) radius of the helix
            pitch (float): The pitch angle of the helix
            r_end (float | None, optional): The ending radius. If default, the same is used as the start. Defaults to None.
            _narc (int, optional): The number of Spline arc sections used. Defaults to 8.

        Returns:
            Curve: The Curve geometry object
        """
        pstart = _parse_vector(pstart)
        pend = _parse_vector(pend)
        
        if r_end is None:
            r_end = r_start
        
        pitch = pitch*np.pi/180
        
        R1, R2, DR = r_start, r_end, r_end-r_start
        p0 = np.array(pstart)
        p1 = np.array(pend)
        dp = (p1-p0)
        L = (dp[0]**2 + dp[1]**2 + dp[2]**2)**(0.5)
        dp = dp/L
        
        Z, X, Y = orthonormalize(dp)
        
        Q = L/np.tan(pitch)
        C = 0

        wtot = C/R2 + Q/R2
        nt = int(np.ceil(wtot/(2*np.pi)*_narc))
        
        t = np.linspace(0, 1, nt)
        Rt = R1 + DR*t
        wt = C/Rt + (Q*t)/Rt
        
        xs = (R1 + DR*t)*np.cos(wt)
        ys = (R1 + DR*t)*np.sin(wt)
        zs = L*t

        xp = xs*X[0] + ys*Y[0] + zs*Z[0] + p0[0]
        yp = xs*X[1] + ys*Y[1] + zs*Z[1] + p0[1]
        zp = xs*X[2] + ys*Y[2] + zs*Z[2] + p0[2]
        
        dp = tuple(Y)
        if startfeed > 0:
            dpx, dpy, dpz = Y
            dx = Z[0]
            dy = Z[1]
            dz = Z[2]
            d = startfeed

            fx = np.array([xp[0] - dx*d/2 - d*dpx, xp[0] - dx*d*0.8/2 - d*dpx])
            fy = np.array([yp[0] - dy*d/2 - d*dpy, yp[0] - dy*d*0.8/2 - d*dpy])
            fz = np.array([zp[0] - dz*d/2 - d*dpz, zp[0] - dz*d*0.8/2 - d*dpz])
            
            xp = np.concat([fx ,xp])
            yp = np.concat([fy, yp])
            zp = np.concat([fz, zp])
            xp[2] += d/2*dx
            yp[2] += d/2*dy
            zp[2] += d/2*dz
            dp = tuple(Z)
        
        return Curve(xp, yp, zp, ctype='Spline')
    
    @staticmethod
    def helix_lh(
              pstart: tuple[float, float, float] | Anchor,
              pend: tuple[float, float, float] | Anchor,
              r_start: float,
              pitch: float,
              r_end: float | None = None,
              _narc: int = 8,
              startfeed: float = 0.0) -> Curve:
        """Generates a Helical curve

        Args:
            pstart (tuple[float, float, float]): The start of the center of rotation (not the start of the curve)
            pend (tuple[float, float, float]): The end of the center of rotation
            r_start (float): The (start) radius of the helix
            pitch (float): The pitch angle of the helix
            r_end (float | None, optional): The ending radius. If default, the same is used as the start. Defaults to None.
            _narc (int, optional): The number of Spline arc sections used. Defaults to 8.

        Returns:
            Curve: The Curve geometry object
        """
        pstart = _parse_vector(pstart)
        pend = _parse_vector(pend)
        
        if r_end is None:
            r_end = r_start
        
        pitch = pitch*np.pi/180
        
        R1, R2, DR = r_start, r_end, r_end-r_start
        p0 = np.array(pstart)
        p1 = np.array(pend)
        dp = (p1-p0)
        L = (dp[0]**2 + dp[1]**2 + dp[2]**2)**(0.5)
        dp = dp/L
        
        Z, X, Y = orthonormalize(dp)
        
        Q = L/np.tan(pitch)
        C = 0

        wtot = C/R2 + Q/R2
        nt = int(np.ceil(wtot/(2*np.pi)*_narc))
        
        t = np.linspace(0, 1, nt)
        Rt = R1 + DR*t
        wt = C/Rt + (Q*t)/Rt
        
        xs = (R1 + DR*t)*np.cos(-wt)
        ys = (R1 + DR*t)*np.sin(-wt)
        zs = L*t

        xp = xs*X[0] + ys*Y[0] + zs*Z[0] + p0[0]
        yp = xs*X[1] + ys*Y[1] + zs*Z[1] + p0[1]
        zp = xs*X[2] + ys*Y[2] + zs*Z[2] + p0[2]
        
        dp = tuple(Y)
        if startfeed > 0:
            dpx, dpy, dpz = Y
            dx = Z[0]
            dy = Z[1]
            dz = Z[2]
            d = startfeed

            fx = np.array([xp[0] - dx*d/2 + d*dpx, xp[0] - dx*d*0.8/2 + d*dpx])
            fy = np.array([yp[0] - dy*d/2 + d*dpy, yp[0] - dy*d*0.8/2 + d*dpy])
            fz = np.array([zp[0] - dz*d/2 + d*dpz, zp[0] - dz*d*0.8/2 + d*dpz])
            
            xp = np.concat([fx ,xp])
            yp = np.concat([fy, yp])
            zp = np.concat([fz, zp])
            xp[2] += d/2*dx
            yp[2] += d/2*dy
            zp[2] += d/2*dz
            dp = tuple(Z)
        
        return Curve(xp, yp, zp, ctype='Spline')
        
    def pipe(self, 
             crossection: GeoSurface | XYPolygon, 
             max_mesh_size: float | None = None,
             start_tangent: Axis | tuple[float, float, float] | np.ndarray | None = None,
             x_axis: Axis | tuple[float, float, float] | np.ndarray | None = None,
             name: str = 'PipedVolume') -> GeoVolume:
        """Extrudes a surface or XYPolygon object along the given curve

        If a GeoSurface object is used, make sure it starts at the center of the curve. This property
        can be accessed with curve_obj.p0.Alignment
        If an XYPolygon is used, it will be automatically centered with XY=0 at the start of the curve with
        the Z-axis align along the initial departure direction curve_obj.dstart.Alignment
        
        Args:
            crossection (GeoSurface | XYPolygon): The cross section definition to be used
            max_mesh_size (float, optional): The maximum mesh size. Defaults to None
            start_tangent (Axis, tuple, ndarray, optional): The input polygon plane normal direction. Defaults to None
            x_axis (Axis, tuple, ndarray optional): The reference X-axis to align the input polygon. Defaults to None
        Returns:
            GeoVolume: The resultant volume object
        """
        if isinstance(crossection, XYPolygon):
            if start_tangent is not None:
                cs = _parse_axis(start_tangent).construct_cs(self.p0)
                surf = crossection.geo(cs)
            elif x_axis is not None:
                xax = _parse_axis(x_axis)
                zax = _parse_axis(self.dstart)
                yax = zax.cross(xax)
                cs = CoordinateSystem(xax, yax, zax, self.p0)
            else:
                zax = self.dstart
                cs = Axis(np.array(zax)).construct_cs(self.p0)
            surf = crossection.geo(cs)
        else:
            surf = crossection
        x1, y1, z1, x2, y2, z2 = gmsh.model.occ.getBoundingBox(*surf.dimtags[0])
        diag = ((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)**(0.5)
        gmsh.model.occ.synchronize()
        gmsh.fltk.run()
        pipetag = gmsh.model.occ.addPipe(surf.dimtags, self.tags[0], trihedron='GuidePlan')
        self.remove()
        surf.remove()
        volume = GeoVolume(pipetag[0][1], name=name)
        volume.max_meshsize = diag/2
        return volume
        