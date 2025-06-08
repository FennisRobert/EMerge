from __future__ import annotations
import gmsh
import numpy as np
from scipy.spatial import ConvexHull
from .cs import Axis, Plane, CoordinateSystem
from typing import Callable

def align_rectangle_frame(pts3d, normal):
    # 1. centroid
    O = np.squeeze(np.mean(pts3d, axis=0))

    # 2. build e_x, e_y
    n = np.squeeze(normal/np.linalg.norm(normal))
    seed = np.array([1.,0.,0.])
    if abs(seed.dot(n)) > 0.9:
        seed = np.array([0.,1.,0.])
    e_x = seed - n*(seed.dot(n))
    e_x /= np.linalg.norm(e_x)
    e_y = np.cross(n, e_x)

    # 3. project into 2D
    pts2d = np.vstack([[(p-O).dot(e_x), (p-O).dot(e_y)] for p in pts3d])

    # 4. convex hull
    hull = ConvexHull(pts2d)
    hull_pts = pts2d[hull.vertices]

    # 5. rotating calipers: find min-area rectangle
    best = (None, np.inf, None)  # (angle, area, (xmin,xmax,ymin,ymax))
    for i in range(len(hull_pts)):
        p0 = hull_pts[i]
        p1 = hull_pts[(i+1)%len(hull_pts)]
        edge = p1 - p0
        θ = -np.arctan2(edge[1], edge[0])  # rotate so edge aligns with +X
        R = np.array([[np.cos(θ), -np.sin(θ)],
                      [np.sin(θ),  np.cos(θ)]])
        rot = hull_pts.dot(R.T)
        xmin, ymin = rot.min(axis=0)
        xmax, ymax = rot.max(axis=0)
        area = (xmax-xmin)*(ymax-ymin)
        if area < best[1]:
            best = (θ, area, (xmin,xmax,ymin,ymax), R)

    θ, _, (xmin,xmax,ymin,ymax), R = best

    # 6. rectangle axes in 3D
    u =  np.cos(-θ)*e_x + np.sin(-θ)*e_y
    v = -np.sin(-θ)*e_x + np.cos(-θ)*e_y

    # corner points in 3D:
    corners = []
    for a in (xmin, xmax):
        for b in (ymin, ymax):
            # back-project to the original 2D frame:
            p2 = np.array([a, b]).dot(R)  # rotate back
            P3 = O + p2[0]*e_x + p2[1]*e_y
            corners.append(P3)

    return {
      "origin": O,
      "axes": (u, v, n),
      "corners": np.array(corners).reshape(4,3)
    }

class Selection:
    dim: int = -1
    def __init__(self, tags: list[int] = None):

        self.tags: list[int] = []
        if tags is not None:
            if not isinstance(tags, list):
                raise TypeError(f'Argument tags must be of type list, instead its {type(tags)}')
            self.tags = tags

    ####### DUNDER METHODS
    def __repr__(self) -> str:
        return f'{type(self).__name__}({self.tags})'
    

    def __add__(self, other: Selection) -> Selection:
        if self.dim != other.dim:
            raise ValueError(f'Cannot add selections of different dimensions {self.dim} and {other.dim}')
        return SELECT_CLASS[self.dim](self.tags + other.tags)

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
    def points(self) -> np.ndarray | list[np.ndarray]:
        '''A list of 3D coordinates of all nodes comprising the selection.'''
        points = gmsh.model.get_boundary(self.dimtags, recursive=True)
        coordinates = [gmsh.model.getValue(*p, []) for p in points]
        return coordinates
    
    @property
    def bounding_box(self) -> tuple[np.ndarray, np.ndarray]:
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
    
    # @staticmethod
    # def from_object(obj: GMSHSurface | GMSHVolume) -> Selection:
        
    #     if isinstance(obj, GMSHSurface):
    #         return FaceSelection(obj.tags)
    #     elif isinstance(obj, GMSHVolume):
    #         return DomainSelection(obj.tags)
    #     else:
    #         raise TypeError(f'Object {obj} is not a GMSHSurface or GMSHVolume')
    
    def exclude(self, xyz_excl_function: Callable) -> Selection:
        include = [xyz_excl_function(*gmsh.model.occ.getCenterOfMass(*tag)) for tag in self.dimtags]
        
        self.tags = [t for incl, t in zip(include, self.tags) if incl]
        return self


class PointSelection(Selection):
    dim: int = 0
    def __init__(self, tags: list[int] = None):
        super().__init__(tags)

class EdgeSelection(Selection):
    dim: int = 1
    def __init__(self, tags: list[int] = None):
        super().__init__(tags)

class FaceSelection(Selection):
    dim: int = 2
    def __init__(self, tags: list[int] = None):
        super().__init__(tags)

    # @property
    # def obj(self) -> GMSHSurface:
    #     ''' Returns a GMSHSurface object representing the face selection.'''
    #     return GMSHSurface(self.tags)
    
    @property
    def normal(self) -> np.ndarray:
        ''' Returns a 3x3 coordinate matrix of the XY + out of plane basis matrix defining the face assuming it can be projected on a flat plane.'''
        ns = [gmsh.model.getNormal(tag, np.array([0,0])) for tag in self.tags]
        return ns[0]
    
    def rect_basis(self) -> tuple[CoordinateSystem, tuple[float, float]]:
        ''' Returns a dictionary with keys: origin, axes, corners. The axes are the 3D basis vectors of the rectangle. The corners are the 4 corners of the rectangle.
        
        Returns:
            cs: CoordinateSystem: The coordinate system of the rectangle.
            size: tuple[float, float]: The size of the rectangle (width [m], height[m])'''
        if len(self.tags) != 1:
            raise ValueError('rect_basis only works for single face selections')
        
        pts3d = self.points
        normal = self.normal
        data = align_rectangle_frame(pts3d, normal)
        plane = data['axes'][:2]
        origin = data['origin']

        cs = CoordinateSystem(Axis(plane[0]), Axis(plane[1]), Axis(data['axes'][2]), origin)

        size1 = np.linalg.norm(data['corners'][1] - data['corners'][0])
        size2 = np.linalg.norm(data['corners'][2] - data['corners'][0])

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
        coordset = []
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
        return X, Y, Z
    
class DomainSelection(Selection):
    dim: int = 3
    def __init__(self, tags: list[int] = None):
        super().__init__(tags)

SELECT_CLASS: dict[int, type[Selection]] = {
    0: PointSelection,
    1: EdgeSelection,
    2: FaceSelection,
    3: DomainSelection
}

######## SELECTOR

class Selector:

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

        dimtags = gmsh.model.getEntities(self._current_dim)
        
        
        dists = [np.linalg.norm(np.array([x,y,z]) - gmsh.model.occ.getCenterOfMass(*tag)) for tag in dimtags]
        index_of_closest = np.argmin(dists)

        return SELECT_CLASS[self._current_dim]([dimtags[index_of_closest][1],])
    
    def inlayer(self, 
                x: float,
                y: float,
                z: float,
                vector: np.ndarray,) -> FaceSelection | EdgeSelection | DomainSelection:
        '''Returns a list of selections that are in the layer defined by the plane normal vector and the point (x,y,z)'''
        dimtags = gmsh.model.getEntities(self._current_dim)

        coords = [gmsh.model.occ.getCenterOfMass(*tag) for tag in dimtags]

        L = np.linalg.norm(vector)
        vector = vector / L

        output = []
        for i, c in enumerate(coords):
            c_local = c - np.array([x,y,z])
            if 0 < np.dot(vector, c_local) < L:
                output.append(dimtags[i][1])
        return SELECT_CLASS[self._current_dim](output)
    
SELECTOR_OBJ = Selector()