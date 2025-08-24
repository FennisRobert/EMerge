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
from .material import Material, AIR
from .selection import FaceSelection, DomainSelection, EdgeSelection, PointSelection, Selection
from loguru import logger
from typing import Literal, Any, Iterable, TypeVar
import numpy as np



def _map_tags(tags: list[int], mapping: dict[int, list[int]]):
    new_tags = []
    for tag in tags:
        new_tags.extend(mapping.get(tag, [tag,]))
    return new_tags

def _bbcenter(x1, y1, z1, x2, y2, z2):
    return np.array([(x1+x2)/2, (y1+y2)/2, (z1+z2)/2])
FaceNames = Literal['back','front','left','right','top','bottom']

class _KEY_GENERATOR:

    def __init__(self):
        self.start = -1
    
    def new(self) -> int:
        self.start += 1
        return self.start

class _GeometryManager:

    def __init__(self):
        self.geometry_list: dict[str, list[GeoObject]] = dict()
        self.active: str = ''

    def all_geometries(self, model: str | None = None) -> list[GeoObject]:
        if model is None:
            model = self.active
        return [geo for geo in self.geometry_list[model] if geo._exists]

    def submit_geometry(self, geo: GeoObject, model: str | None = None) -> None:
        if model is None:
            model = self.active
        self.geometry_list[model].append(geo)

    def sign_in(self, modelname: str) -> None:
        if modelname not in self.geometry_list:
            self.geometry_list[modelname] = []
        self.active = modelname

    def reset(self, modelname: str) -> None:
        self.geometry_list[modelname] = []
    
    def lowest_priority(self) -> int:
        return min([geo._priority for geo in self.all_geometries()])
    
    def highest_priority(self) -> int:
        return min([geo._priority for geo in self.all_geometries()])
    
class _FacePointer:
    """The FacePointer class defines a face to be selectable as a
    face normal vector plus an origin. All faces of an object
    can be selected based on the projected distance to the defined
    selection plane of the center of mass of a face iff the normals
    also align with some tolerance.

    """
    def __init__(self, 
                 origin: np.ndarray, 
                 normal: np.ndarray):
        self.o = np.array(origin)
        self.n = np.array(normal)

    def find(self, dimtags: list[tuple[int,int]],
             origins: list[np.ndarray],
             normals: list[np.ndarray]) -> list[int]:
        tags = []
        for (d,t), o, n in zip(dimtags, origins, normals):
            normdist = np.abs((o-self.o)@self.n)
            dotnorm = np.abs(n@self.n)
            if normdist < 1e-3 and dotnorm > 0.99:
                tags.append(t)
        return tags
    
    def rotate(self, c0, ax, angle):
        """
        Rotate self.o and self.n about axis `ax`, centered at `c0`, by `angle` radians.

        Parameters
        ----------
        c0 : np.ndarray
            The center of rotation, shape (3,).
        ax : np.ndarray
            The axis to rotate around, shape (3,). Need not be unit length.
        angle : float
            Rotation angle in radians.
        """
        angle = -angle
        # Ensure axis is a unit vector
        k = ax / np.linalg.norm(ax)

        # Precompute trig values
        cos_theta = np.cos(angle)
        sin_theta = np.sin(angle)

        def rodrigues(v: np.ndarray) -> np.ndarray:
            """
            Rotate vector v around axis k by angle using Rodrigues' formula.
            """
            # term1 = v * cosθ
            term1 = v * cos_theta
            # term2 = (k × v) * sinθ
            term2 = np.cross(k, v) * sin_theta
            # term3 = k * (k ⋅ v) * (1 - cosθ)
            term3 = k * (np.dot(k, v)) * (1 - cos_theta)
            return term1 + term2 + term3

        # Rotate the origin point about c0:
        rel_o = self.o - c0            # move to rotation-centre coordinates
        rot_o = rodrigues(rel_o)       # rotate
        self.o = rot_o + c0            # move back

        # Rotate the normal vector (pure direction, no translation)
        self.n = rodrigues(self.n)

    def translate(self, dx, dy, dz):
        self.o = self.o + np.array([dx, dy, dz])
    
    def mirror(self, c0: np.ndarray, pln: np.ndarray) -> None:
        """
        Reflect self.o and self.n across the plane passing through c0
        with normal pln.

        Parameters
        ----------
        c0 : np.ndarray
            A point on the mirror plane, shape (3,).
        pln : np.ndarray
            The normal of the mirror plane, shape (3,). Need not be unit length.
        """
        # Normalize the plane normal
        k = pln / np.linalg.norm(pln)
        

        # Reflect the origin point:
        # compute vector from plane point to self.o
        v_o = self.o - c0
        # signed distance along normal
        dist_o = np.dot(v_o, k)
        # reflection
        self.o = self.o - 2 * dist_o * k

        # Reflect the normal/direction vector:
        dist_n = np.dot(self.n, k)
        self.n = self.n - 2 * dist_n * k

    def affine_transform(self, M: np.ndarray):
        """
        Apply a 4×4 affine transformation matrix to both self.o and self.n.

        Parameters
        ----------
        M : np.ndarray
            The 4×4 affine transformation matrix.
            - When applied to a point, use homogeneous w=1.
            - When applied to a direction/vector, use homogeneous w=0.
        """
        # Validate shape
        if M.shape != (4, 4):
            raise ValueError(f"Expected M to be 4×4, got shape {M.shape}")

        # Transform origin point (homogeneous w=1)
        homo_o = np.empty(4)
        homo_o[:3] = self.o
        homo_o[3] = 1.0
        transformed_o = M @ homo_o
        self.o = transformed_o[:3]

        # Transform normal/direction vector (homogeneous w=0)
        homo_n = np.empty(4)
        homo_n[:3] = self.n
        homo_n[3] = 0.0
        transformed_n = M @ homo_n
        self.n = transformed_n[:3]
        # Optionally normalize self.n if you need to keep it unit-length:
        # self.n = self.n / np.linalg.norm(self.n)

    def copy(self) -> _FacePointer:
        return _FacePointer(self.o, self.n)


_GENERATOR = _KEY_GENERATOR()
_GEOMANAGER = _GeometryManager()

class GeoObject:
    """A generalization of any OpenCASCADE entity described by a dimension and a set of tags.
    """
    dim: int = -1
    def __init__(self, tags: list[int] | None = None):
        if tags is None:
            tags = []
        self.old_tags: list[int] = []
        self.tags: list[int] = tags
        self.material: Material = AIR
        self.mesh_multiplier: float = 1.0
        self.max_meshsize: float = 1e9
        
        self._unset_constraints: bool = False
        self._embeddings: list[GeoObject] = []
        self._face_pointers: dict[str, _FacePointer] = dict()
        self._tools: dict[int, dict[str, _FacePointer]] = dict()
        
        self._key = _GENERATOR.new()
        self._aux_data: dict[str, Any] = dict()
        self._priority: int = 10

        self._exists: bool = True
        _GEOMANAGER.submit_geometry(self)

    @property
    def color_rgb(self) -> tuple[float, float, float]:
        return self.material.color_rgb
    
    @property
    def opacity(self) -> float:
        return self.material.opacity
    
    @property
    def select(self) -> Selection:
        '''Returns a corresponding Face/Domain or Edge Selection object'''
        if self.dim==1:
            return EdgeSelection(self.tags)
        elif self.dim==2:
            return FaceSelection(self.tags)
        elif self.dim==3:
            return DomainSelection(self.tags)
        else:
            return Selection(self.tags)
    
    @staticmethod
    def merged(objects: list[GeoPoint | GeoEdge | GeoSurface | GeoVolume | GeoObject]) -> list[GeoPoint | GeoEdge | GeoSurface | GeoVolume | GeoObject] | GeoPoint | GeoEdge | GeoSurface | GeoVolume | GeoObject:
        dim = objects[0].dim
        tags = []
        out: GeoObject | None = None
        for obj in objects:
            tags.extend(obj.tags)
        if dim==2:
            out = GeoSurface(tags)
        elif dim==3:
            out = GeoVolume(tags)
        else:
            out = GeoObject(tags)
        out.material = objects[0].material
        return out
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.dim},{self.tags})'

    def _data(self, *labels) -> tuple[Any | None, ...]:
        return tuple([self._aux_data[lab] for lab in labels])
    
    def _add_face_pointer(self, 
                          name: str,
                          origin: np.ndarray,
                          normal: np.ndarray):
        self._face_pointers[name] = _FacePointer(origin, normal)
    
    def make_copy(self) -> GeoObject:
        new_dimtags = gmsh.model.occ.copy(self.dimtags)
        new_obj = GeoObject.from_dimtags(new_dimtags)
        new_obj.material = self.material
        new_obj.mesh_multiplier = self.mesh_multiplier
        new_obj.max_meshsize = self.max_meshsize
        
        new_obj._unset_constraints = self._unset_constraints
        new_obj._embeddings = [emb.make_copy() for emb in self._embeddings]
        new_obj._face_pointers = {key: value.copy() for key,value in self._face_pointers.items()}
        new_obj._tools = {key: {key2: value2.copy() for key2, value2 in value.items()} for key,value in self._tools.items()}
        
        new_obj._aux_data = self._aux_data.copy()
        new_obj._priority = self._priority
        new_obj._exists = self._exists
        return new_obj

    def replace_tags(self, tagmap: dict[int, list[int]]):
        self.old_tags = self.tags
        newtags = []
        for tag in self.tags:
            newtags.extend(tagmap.get(tag, [tag,]))
        self.tags = newtags
        logger.debug(f'{self} Replaced {self.old_tags} -> {self.tags}')
    
    def update_tags(self, tag_mapping: dict[int,dict]) -> GeoObject:
        ''' Update the tag definition of a GeoObject after fragementation.'''
        self.replace_tags(tag_mapping[self.dim])
        return self
    
    def _take_pointers(self, *others: GeoObject) -> GeoObject:
        for other in others:
            self._face_pointers.update(other._face_pointers)
            self._tools.update(other._tools)
        return self
    
    @property
    def _all_pointers(self) -> list[_FacePointer]:
        pointers = list(self._face_pointers.values())
        for dct in self._tools.values():
            pointers.extend(list(dct.values()))
        return pointers
    
    @property
    def _all_pointer_names(self) -> set[str]:
        keys = set(self._face_pointers.keys())
        for dct in self._tools.values():
            keys = keys.union(set(dct.keys()))
        return keys
    
    def _take_tools(self, *objects: GeoObject) -> GeoObject:
        for obj in objects:
            self._tools[obj._key] = obj._face_pointers
            self._tools.update(obj._tools)
        return self
    
    def _face_tags(self, name: FaceNames, tool: GeoObject | None = None) -> list[int]:
        names = self._all_pointer_names
        if name not in names:
            raise ValueError(f'The face {name} does not exist in {self}')
        
        gmsh.model.occ.synchronize()
        dimtags = gmsh.model.get_boundary(self.dimtags, True, False)
        
        normals = [gmsh.model.get_normal(t, [0,0]) for d,t, in dimtags]
        origins = [gmsh.model.occ.get_center_of_mass(d, t) for d,t in dimtags]
        
        if tool is not None:
            tags = self._tools[tool._key][name].find(dimtags, origins, normals)
        else:
            tags = self._face_pointers[name].find(dimtags, origins, normals)
        logger.info(f'Selected face {tags}.')
        return tags

    def set_material(self, material: Material) -> GeoObject:
        self.material = material
        return self
    
    def prio_set(self, level: int) -> GeoObject:
        """Defines the material assignment priority level of this geometry.
        By default all objects have priority level 10. If you assign a lower number,
        in cases where multiple geometries occupy the same volume, the highest priority
        will be chosen.

        Args:
            level (int): The priority level

        Returns:
            GeoObject: The same object
        """
        self._priority = level
        return self
    
    def prio_up(self) -> GeoObject:
        """Increase priority by 1

        Returns:
            GeoObject: _description_
        """
        self._priority += 1
        return self
    
    def prio_down(self) -> GeoObject:
        """Decrase priority by 1

        Returns:
            GeoObject: _description_
        """
        self._priority -= 1
        return self

    def background(self) -> GeoObject:
        """Set the priority to be on the background.

        Returns:
            GeoObject: _description_
        """
        self._priority = _GEOMANAGER.lowest_priority()-10
        return self

    def foreground(self) -> GeoObject:
        """Set the priority to be on top.

        Returns:
            GeoObject: _description_
        """
        self._priority = _GEOMANAGER.highest_priority()+10
        return self
    
    def boundary(self, exclude: tuple[FaceNames,...] | None = None, tags: list[int] | None = None) -> FaceSelection:
        """Returns the complete set of boundary faces.

        If implemented, it is possible to exclude a set of faces based on their name
        or a list of tags (integers)

        Returns:
            FaceSelection: The selected faces
        """
        if exclude is None:
            exclude = tuple()
            
        if tags is None:
            tags = []
        
        
        for name in exclude:
            tags.extend(self.face(name).tags)
        dimtags = gmsh.model.get_boundary(self.dimtags, True, False)
        return FaceSelection([t for d,t in dimtags if t not in tags])
    
    def face(self, name: FaceNames, tool: GeoObject | None = None) -> FaceSelection:
        """Returns the FaceSelection for a given face name.
        
        The face name must be defined for the type of geometry.

        Args:
            name (FaceNames): The name of the face to select.

        Returns:
            FaceSelection: The selected face
        """
        
        return FaceSelection(self._face_tags(name, tool))

    def faces(self, *names: FaceNames, tool: GeoObject | None = None) -> FaceSelection:
        """Returns the FaceSelection for a given face names.
        
        The face name must be defined for the type of geometry.

        Args:
            name (FaceNames): The name of the face to select.

        Returns:
            FaceSelection: The selected face
        """
        tags = []
        for name in names:
            tags.extend(self._face_tags(name, tool))
        return FaceSelection(tags)
        
    @property
    def dimtags(self) -> list[tuple[int, int]]:
        return [(self.dim, tag) for tag in self.tags]
    
    @property
    def embeddings(self) -> list[tuple[int,int]]:
        return []

    @staticmethod
    def from_dimtags(dimtags: list[tuple[int,int]]) -> GeoVolume | GeoSurface | GeoObject:
        dim = dimtags[0][0]
        tags = [t for d,t in dimtags]
        if dim==0:
            return GeoPoint(tags)
        elif dim==1:
            return GeoEdge(tags)
        if dim==2:
            return GeoSurface(tags)
        if dim==3:
            return GeoVolume(tags)
        return GeoObject(tags)
    
    def remove(self) -> None:
        self._exists = False
        gmsh.model.occ.remove(self.dimtags, True)
        
class GeoVolume(GeoObject):
    '''GeoVolume is an interface to the GMSH CAD kernel. It does not represent EMerge
    specific geometry data.'''
    dim = 3
    def __init__(self, tag: int | Iterable[int]):
        super().__init__()
        self.tags: list[int] = []
        if isinstance(tag, Iterable):
            self.tags = list(tag)
        else:
            self.tags = [tag,]

    @property
    def select(self) -> DomainSelection:
        return DomainSelection(self.tags)
    
class GeoPoint(GeoObject):
    dim = 0

    @property
    def select(self) -> PointSelection:
        return PointSelection(self.tags)
    
    def __init__(self, tag: int | list[int]):
        super().__init__()

        self.tags: list[int] = []
        if isinstance(tag, Iterable):
            self.tags = list(tag)
        else:
            self.tags = [tag,]

class GeoEdge(GeoObject):
    dim = 1

    @property
    def select(self) -> EdgeSelection:
        return EdgeSelection(self.tags)
    
    def __init__(self, tag: int | list[int]):
        super().__init__()
        self.tags: list[int] = []
        if isinstance(tag, Iterable):
            self.tags = list(tag)
        else:
            self.tags = [tag,]
        

class GeoSurface(GeoObject):
    '''GeoVolume is an interface to the GMSH CAD kernel. It does not reprsent Emerge
    specific geometry data.'''
    dim = 2

    @property
    def select(self) -> FaceSelection:
        return FaceSelection(self.tags)
    
    def __init__(self, tag: int | list[int]):
        super().__init__()
        self.tags: list[int] = []
        if isinstance(tag, Iterable):
            self.tags = list(tag)
        else:
            self.tags = [tag,]

class GeoPolygon(GeoSurface):
    
    def __init__(self,
                 tags: list[int]):
        super().__init__(tags)
        self.points: list[int] = []
        self.lines: list[int] = []


############################################################
#                      SHORT FUNCTIONS                     #
############################################################

def select(*items: Selection | GeoObject) -> Selection:
    """Generate a selection from a series of selections and/or objects that share the same dimension.

    Raises:
        ValueError: Raised if the dimensions provided are not consistent

    Returns:
        Selection: An output selection object.
    """
    dim = items[0].dim
    tags = []
    for item in items:
        if item.dim!=dim:
            raise ValueError(f'Cannot group of objects with a dissimilar dimensions. Trying to include {item} in a list of dimension {dim}.')
        tags.extend(item.tags)
    return Selection.from_dim_tags(dim, tags)
            