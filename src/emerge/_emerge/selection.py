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
import gmsh # type: ignore
import numpy as np
from scipy.spatial import ConvexHull # type: ignore
from .cs import Axis, CoordinateSystem, _parse_vector, Plane
from typing import Callable, TypeVar, Iterable, Any

def align_rectangle_frame(pts3d: np.ndarray, normal: np.ndarray) -> dict[str, Any]:
    """Tries to find a rectangle as convex-hull of a set of points with a given normal vector.

    Args:
        pts3d (np.ndarray): The points (N,3)
        normal (np.ndarray): The normal vector.

    Returns:
        dict[str, np.ndarray]: The output data
    """
    
    # 1. centroid
    Omat = np.squeeze(np.mean(pts3d, axis=0))

    # 2. build e_x, e_y
    n = np.squeeze(normal/np.linalg.norm(normal))
    seed = np.array([1.,0.,0.])
    if abs(seed.dot(n)) > 0.9:
        seed = np.array([0.,1.,0.])
    e_x = seed - n*(seed.dot(n))
    e_x /= np.linalg.norm(e_x)
    e_y = np.cross(n, e_x)

    # 3. project into 2D
    pts2d = np.vstack([[(p-Omat).dot(e_x), (p-Omat).dot(e_y)] for p in pts3d])

    # 4. convex hull
    hull = ConvexHull(pts2d)
    hull_pts = pts2d[hull.vertices]

    # 5. rotating calipers: find min-area rectangle
    best = (None, np.inf, None)  # (angle, area, (xmin,xmax,ymin,ymax))
    for i in range(len(hull_pts)):
        p0 = hull_pts[i]
        p1 = hull_pts[(i+1)%len(hull_pts)]
        edge = p1 - p0
        theta = -np.arctan2(edge[1], edge[0])  # rotate so edge aligns with +X
        R = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta),  np.cos(theta)]])
        rot = hull_pts.dot(R.T)
        xmin, ymin = rot.min(axis=0)
        xmax, ymax = rot.max(axis=0)
        area = (xmax-xmin)*(ymax-ymin)
        if area < best[1]:
            best = (theta, area, (xmin,xmax,ymin,ymax), R) # type: ignore

    theta, _, (xmin,xmax,ymin,ymax), R = best # type: ignore

    # 6. rectangle axes in 3D
    u =  np.cos(-theta)*e_x + np.sin(-theta)*e_y
    v = -np.sin(-theta)*e_x + np.cos(-theta)*e_y

    # corner points in 3D:
    corners = []
    for a in (xmin, xmax):
        for b in (ymin, ymax):
            # back-project to the original 2D frame:
            p2 = np.array([a, b]).dot(R)  # rotate back
            P3 = Omat + p2[0]*e_x + p2[1]*e_y
            corners.append(P3)

    return {
      "origin": Omat,
      "axes": (u, v, n),
      "corners": np.array(corners).reshape(4,3)
    }

TSelection = TypeVar("TSelection", bound="Selection")

# Your custom alphabet
ALPHABET = (
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "012345%#"
)
# Map char → int and int → char
CHAR_TO_VAL = {ch: i for i, ch in enumerate(ALPHABET)}
VAL_TO_CHAR = {i: ch for i, ch in enumerate(ALPHABET)}

def encode_data(values: tuple[float,...]) -> str:
    """
    Convert a tuple of floats into a custom base64-like encoded string
    using 16-bit float representation and a 64-character alphabet.
    """
    # Convert floats to 16-bit half-floats
    arr = np.array(values, dtype=np.float16)
    # Get raw bytes
    byte_data = arr.tobytes()
    # Convert bytes to a bitstring
    bitstring = ""
    for byte in byte_data:
        bitstring += f"{byte:08b}"
    
    # Pad the bitstring to a multiple of 6 bits
    pad_len = (6 - len(bitstring) % 6) % 6
    bitstring += "0" * pad_len
    
    # Encode 6 bits at a time
    encoded = ""
    for i in range(0, len(bitstring), 6):
        chunk = bitstring[i:i+6]
        val = int(chunk, 2)
        encoded += VAL_TO_CHAR[val]
    
    # Optionally store how many pad bits we added
    # so we can remove them during decoding.
    # We'll prepend it as one character encoding pad_len (0-5).
    encoded = VAL_TO_CHAR[pad_len] + encoded
    return encoded

def decode_data(encoded: str) -> tuple[float,...]:
    """
    Decode a string produced by floats_to_custom_string
    back into a tuple of float64 values.
    """
    # The first character encodes how many zero bits were padded
    pad_char = encoded[0]
    pad_len = CHAR_TO_VAL[pad_char]
    data = encoded[1:]
    
    # Convert each char back to 6 bits
    bitstring = ""
    for ch in data:
        val = CHAR_TO_VAL[ch]
        bitstring += f"{val:06b}"
    
    # Remove any padding bits at the end
    if pad_len > 0:
        bitstring = bitstring[:-pad_len]
    
    # Split into bytes
    byte_data = []
    for i in range(0, len(bitstring), 8):
        byte_chunk = bitstring[i:i+8]
        if len(byte_chunk) < 8:
            break
        byte_val = int(byte_chunk, 2)
        byte_data.append(byte_val)
    
    byte_array = bytes(byte_data)
    # Recover as float16
    arr = np.frombuffer(byte_array, dtype=np.float16)
    # Convert back to float64 for higher precision
    return tuple(np.array(arr, dtype=np.float64))

class Selection:
    """A generalized class representing a slection of tags.

    """
    dim: int = -1
    def __init__(self, tags: list[int] | set[int] | tuple[int] | None = None):

        self._tags: set[int] = set()

        if tags is not None:
            if not isinstance(tags, (list,set,tuple)):
                raise TypeError(f'Argument tags must be of type list, tuple or set, instead its {type(tags)}')
            self._tags = set(tags)
    
    @staticmethod
    def from_dim_tags(dim: int, tags: list[int] | set[int]) -> Selection:
        if dim==0:
            return PointSelection(tags)
        elif dim==1:
            return EdgeSelection(tags)
        elif dim==2:
            return FaceSelection(tags)
        elif dim==3:
            return DomainSelection(tags)
        raise ValueError(f'Dimension must be 0,1,2 or 3. Not {dim}')
    
    @property
    def tags(self) -> list[int]:
        return list(self._tags)
    
    @property
    def color_rgb(self) -> tuple[float, float, float]:
        return (0.5,0.5,1.0)
    
    @property
    def centers(self) -> list[tuple[float, float, float],]:
        return [gmsh.model.occ.get_center_of_mass(self.dim, tag) for tag in self.tags]
    
    @property
    def opacity(self) -> float:
        return 0.6
    ####### DUNDER METHODS
    def __repr__(self) -> str:
        return f'{type(self).__name__}({self.tags})'
    
    ####### PROPERTIES
    @property
    def dimtags(self) -> list[tuple[int,int]]:
        return [(self.dim, tag) for tag in self.tags]
    
    @property
    def center(self) -> np.ndarray | list[np.ndarray]:
        if len(self.tags)==1:
            return gmsh.model.occ.getCenterOfMass(self.dim, self.tags[0])
        else:
            return [gmsh.model.occ.getCenterOfMass(self.dim, tag) for tag in self.tags]
    
    @property
    def points(self) -> list[np.ndarray]:
        '''A list of 3D coordinates of all nodes comprising the selection.'''
        points = gmsh.model.get_boundary(self.dimtags, recursive=True)
        coordinates = [gmsh.model.getValue(*p, []) for p in points]
        return coordinates
    
    @property
    def bounding_box(self) -> tuple[Iterable, Iterable]:
        if len(self.tags)==1:
            return gmsh.model.occ.getBoundingBox(self.dim, self.tags[0])
        else:
            minx = miny = minz = 1e10
            maxx = maxy = maxz = -1e10
            for tag in self.tags:
                x0, y0, z0, x1, y1, z1 = gmsh.model.occ.getBoundingBox(self.dim, tag)
                minx = min(minx, x0)
                miny = min(miny, y0)
                minz = min(minz, z0)
                maxx = max(maxx, x1)
                maxy = max(maxy, y1)
                maxz = max(maxz, z1)
            return (minx, miny, minz), (maxx, maxy, maxz)
        
    def exclude(self, xyz_excl_function: Callable = lambda x,y,z: True, plane: Plane | None = None, axis: Axis | None = None) -> Selection:
        """Exclude points by evaluating a function(x,y,z)-> bool

        This modifies the selection such that the selection does not contain elements
        of this selection of which the center of mass is excluded by the exclusion function.

        Args:
            xyz_excl_function (Callable): A callable for (x,y,z) that returns True if the point should be excluded.

        Returns:
            Selection: This Selection modified without the excluded points.
        """
        include = [~xyz_excl_function(*gmsh.model.occ.getCenterOfMass(*tag)) for tag in self.dimtags]
        
        if axis is not None:
            norm = axis.np
            include2 = [abs(gmsh.model.getNormal(tag, np.array([0,0]))@norm)<0.9 for tag in self.tags]
            include = [i1 for i1, i2 in zip(include, include2) if i1 and i2]
        self._tags = set([t for incl, t in zip(include, self._tags) if incl])
        return self
    
    def isolate(self, xyz_excl_function: Callable = lambda x,y,z: True, plane: Plane | None = None, axis: Axis | None = None) -> Selection:
        """Include points by evaluating a function(x,y,z)-> bool

        This modifies the selection such that the selection does not contain elements
        of this selection of which the center of mass is excluded by the exclusion function.

        Args:
            xyz_excl_function (Callable): A callable for (x,y,z) that returns True if the point should be excluded.

        Returns:
            Selection: This Selection modified without the excluded points.
        """
        include1 = [xyz_excl_function(*gmsh.model.occ.getCenterOfMass(*tag)) for tag in self.dimtags]
        
        if axis is not None:
            norm = axis.np
            include2 = [(gmsh.model.getNormal(tag, np.array([0,0]))@norm)>0.99 for tag in self.tags]
            include1 = [i1 for i1, i2 in zip(include1, include2) if i1 and i2]
        self._tags = set([t for incl, t in zip(include1, self._tags) if incl])
        return self

    def __operable__(self, other: Selection) -> None:
        if not self.dim == other.dim:
            raise ValueError(f'Selection dimensions must be equal. Trying to operate on dim {self.dim} and {other.dim}')
        pass

    def __add__(self, other: Selection) -> Selection:
        self.__operable__(other)
        return Selection.from_dim_tags(self.dim, self._tags.union(other._tags))
    
    def __and__(self, other: Selection) -> Selection:
        self.__operable__(other)
        return Selection.from_dim_tags(self.dim, self._tags.intersection(other._tags))
    
    def __or__(self, other: Selection) -> Selection:
        self.__operable__(other)
        return Selection.from_dim_tags(self.dim, self._tags.union(other._tags))
    
    def __sub__(self, other: Selection) -> Selection:
        self.__operable__(other)
        return Selection.from_dim_tags(self.dim, self._tags.difference(other.tags))

class PointSelection(Selection):
    """A Class representing a selection of points.

    """
    dim: int = 0
    def __init__(self, tags: list[int] | set [int] | None = None):
        super().__init__(tags)

class EdgeSelection(Selection):
    """A Class representing a selection of edges.

    """
    dim: int = 1
    def __init__(self, tags: list[int] | set [int] | None = None):
        super().__init__(tags)

class FaceSelection(Selection):
    """A Class representing a selection of Faces.

    """
    dim: int = 2
    def __init__(self, tags: list[int] | set [int] | None = None):
        super().__init__(tags)

    @property
    def normal(self) -> np.ndarray:
        ''' Returns a 3x3 coordinate matrix of the XY + out of plane basis matrix defining the face assuming it can be projected on a flat plane.'''
        ns = [gmsh.model.getNormal(tag, np.array([0,0])) for tag in self.tags]
        return ns[0]
    
    def rect_basis(self) -> tuple[CoordinateSystem, tuple[float, float]]:
        ''' Returns a dictionary with keys: origin, axes, corners. The axes are the 3D basis vectors of the rectangle. The corners are the 4 corners of the rectangle.
        
        Returns:
            cs: CoordinateSystem: The coordinate system of the rectangle.
            size: tuple[float, float]: The size of the rectangle (width [m], height[m])
        '''
        if len(self.tags) != 1:
            raise ValueError('rect_basis only works for single face selections')
        
        pts3d = self.points
        normal = self.normal
        data = align_rectangle_frame(np.array(pts3d), normal)
        plane = data['axes'][:2]
        origin = data['origin']

        cs = CoordinateSystem(Axis(plane[0]), Axis(plane[1]), Axis(data['axes'][2]), origin)

        size1 = float(np.linalg.norm(data['corners'][1] - data['corners'][0]))
        size2 = float(np.linalg.norm(data['corners'][2] - data['corners'][0]))

        if size1>size2:
            cs.swapxy()
            return cs, (size1, size2)
        else:
            return cs, (size2, size1)
            
    def sample(self, Npts: int) -> tuple[np.ndarray, np.ndarray, np.ndarray] | list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        ''' Sample the surface at the compiler defined parametric coordinate range.
        This function usually returns a square region that contains the surface.
        
        Returns:
        --------
            X: np.ndarray
                a NxN numpy array of X coordinates.
            Y: np.ndarray
                a NxN numpy array of Y coordinates.
            Z: np.ndarray
                a NxN numpy array of Z coordinates'''
        coordset: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        for tag in self.tags:
            tags, coord, param = gmsh.model.mesh.getNodes(2, tag, includeBoundary=True)
            
            uss = param[0::2]
            vss = param[1::2]

            umin = min(uss)
            umax = max(uss)
            vmin = min(vss)
            vmax = max(vss)

            us = np.linspace(umin, umax, Npts)
            vs = np.linspace(vmin, vmax, Npts)

            U, V = np.meshgrid(us, vs, indexing='ij')

            shp = U.shape

            uax = U.flatten()
            vax = V.flatten()
            
            pcoords = np.zeros((2*uax.shape[0],))

            pcoords[0::2] = uax
            pcoords[1::2] = vax

            coords = gmsh.model.getValue(2, tag, pcoords).reshape(-1,3).T
           
            coordset.append((coords[0,:].reshape(shp), 
                             coords[1,:].reshape(shp), 
                             coords[2,:].reshape(shp)))
            
        X = [c[0] for c in coordset]
        Y = [c[1] for c in coordset]
        Z = [c[2] for c in coordset]
        if len(X) == 1:
            return X[0], Y[0], Z[0]
        return np.array(X), np.array(Y), np.array(Z)
    
class DomainSelection(Selection):
    """A Class representing a selection of domains.

    """
    dim: int = 3
    def __init__(self, tags: list[int] | set[int] | None = None):
        super().__init__(tags)

SELECT_CLASS: dict[int, type[Selection]] = {
    0: PointSelection,
    1: EdgeSelection,
    2: FaceSelection,
    3: DomainSelection
}

######## SELECTOR

class Selector:
    """A class instance with convenient methods to generate selections using method chaining.

    Use the specific properties and functions in a "language" like way to make selections.

    To specify what to select, use the .node, .edge, .face or .domain property.
    These properties return the Selector after which you can say how to execute a selection.


    """
    def __init__(self):
        self._current_dim: int = -1
    
    ## DIMENSION CHAIN
    @property
    def node(self) -> Selector:
        self._current_dim = 0
        return self

    @property
    def edge(self) -> Selector:
        self._current_dim = 1
        return self
    
    @property
    def face(self) -> Selector:
        self._current_dim = 2
        return self

    @property
    def domain(self) -> Selector:
        self._current_dim = 3
        return self
    
    def near(self,
             x: float,
             y: float,
             z: float = 0) -> Selection | PointSelection | EdgeSelection | FaceSelection | DomainSelection:
        """Returns a selection of the releative dimeions by which of the instances is most proximate to a coordinate.

        Args:
            x (float): The X-coordinate
            y (float): The Y-coordinate
            z (float, optional): The Z-coordinate. Defaults to 0.

        Returns:
            Selection | PointSelection | EdgeSelection | FaceSelection | DomainSelection: The resultant selection.
        """
        dimtags = gmsh.model.getEntities(self._current_dim)
        
        
        dists = [np.linalg.norm(np.array([x,y,z]) - gmsh.model.occ.getCenterOfMass(*tag)) for tag in dimtags]
        index_of_closest = np.argmin(dists)

        return SELECT_CLASS[self._current_dim]([dimtags[index_of_closest][1],])
    
    def inlayer(self, 
                x: float,
                y: float,
                z: float,
                vector: tuple[float, float, float] | np.ndarray | Axis) -> Selection:
        '''Returns a list of selections that are in the layer defined by the plane normal vector and the point (x,y,z)
        
        The layer is specified by two infinite planes normal to the provided vector. The first plane is originated
        at the provided origin. The second plane is placed such that it contains the point origin+vector.

        Args:
            x (float): The X-coordinate from which to select
            y (float): The Y-coordinate from which to select
            z (float): The Z-coordinate from which to select
            vector (np.ndarray, tuple, Axis): A vector with length in (meters) originating at the origin.

        Returns:
            Selection | PointSelection | EdgeSelection | FaceSelection | DomainSelection: The resultant selection.

        '''
        vector = _parse_vector(vector)
        
        dimtags = gmsh.model.getEntities(self._current_dim)

        coords = [gmsh.model.occ.getCenterOfMass(*tag) for tag in dimtags]

        L = np.linalg.norm(vector)
        vector = vector / L

        output = []
        for i, c in enumerate(coords):
            c_local = c - np.array([x,y,z])
            if 0 < np.dot(vector, c_local) < L: # type: ignore
                output.append(dimtags[i][1])
        return SELECT_CLASS[self._current_dim](output)
    
    def inplane(self,
                x: float,
                y: float,
                z: float,
                nx: float,
                ny: float,
                nz: float,
                tolerance: float = 1e-8) -> FaceSelection:
        """Returns a FaceSelection for all faces that lie in a provided infinite plane
        specified by an origin plus a plane normal vector.

        Args:
            x (float): The plane origin X-coordinate
            y (float): The plane origin Y-coordinate
            z (float): The plane origin Z-coordinate
            nx (float): The plane normal X-component
            ny (float): The plane normal Y-component
            nz (float): The plane normal Z-component
            tolerance (float, optional): An in plane tolerance (displacement and normal dot product). Defaults to 1e-6.

        Returns:
            FaceSelection: All faces that lie in the specified plane
        """
        orig = np.array([x,y,z])
        norm = np.array([nx,ny,nz])
        norm = norm/np.linalg.norm(norm)
        
        dimtags = gmsh.model.getEntities(2)
        coords = [gmsh.model.occ.getCenterOfMass(*tag) for tag in dimtags]
        normals = [gmsh.model.get_normal(t, (0,0)) for d,t, in dimtags]
        tags = []
        for (d,t), o, n in zip(dimtags, coords, normals):
            normdist = np.abs((o-orig) @ norm)
            dotnorm = np.abs(n @ norm)
            if normdist < tolerance and dotnorm > 1-tolerance:
                tags.append(t)
        return FaceSelection(tags)
    
    def code(self, code: str):
        nums1 = decode_data(code)
        
        dimtags = gmsh.model.getEntities(2)
        
        scoring = dict()
        for dim, tag in dimtags:
            x1, y1, z1 = gmsh.model.occ.getCenterOfMass(2, tag)
            xmin2, ymin2, zmin2, xmax2, ymax2, zmax2 = gmsh.model.occ.getBoundingBox(dim, tag)
            nums2 = [x1, y1, z1, xmin2, ymin2, zmin2, xmax2, ymax2, zmax2]
            score = np.sqrt(sum([(a-b)**2 for a,b in zip(nums1, nums2)]))
            scoring[tag] = score
        
        min_val = min(scoring.values())

        # Find all keys whose value == min_val
        candidates = [k for k, v in scoring.items() if v == min_val]

        # Pick the lowest key
        lowest_key = min(candidates)
        return FaceSelection([lowest_key,])

SELECTOR_OBJ = Selector()

