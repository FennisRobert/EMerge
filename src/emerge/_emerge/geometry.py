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

# Last Cleanup: 2026-01-04

from __future__ import annotations
import gmsh  # type: ignore
from emsutil import Material
from emsutil.lib import AIR
from .attributes import PhysicalAttributeSet, PhysicalAttribute, PhysicalAttributeDescriptor
from .selection import (
    FaceSelection,
    DomainSelection,
    EdgeSelection,
    PointSelection,
    Selection,
)
from loguru import logger
from typing import Literal, Any, Iterable, Callable
import numpy as np
from .cs import Anchor
from .file import Saveable
from typing import TypeVar
from .mdi import MetaDataInstructions
from ._global import _GlobalHandler, _BaseManager

FaceNames = Literal["back", "front", "left", "right", "top", "bottom"]

############################################################
#                          CLASSES                         #
############################################################


class _KeyGenerator(_BaseManager):
    def __init__(self):
        self.start = -1

    def reset(self) -> None:
        self.start = -1

    def new(self) -> int:
        self.start += 1
        return self.start


class _GeometryManager(_BaseManager):
    """The Geometry manager is a singleton class that keeps track of all geometries

    The GeometryManager exists to make sure that EMerge geometry state is in sync with
    GMSH's internal management of geometries.

    """

    def __init__(self):
        self.geometry_list: dict[str, dict[str, GeoObject]] = dict()
        self.active: str = ""

    def get_surfaces(self) -> list[GeoSurface]:
        """Return a list of all GeoSurface objects

        Returns:
            list[GeoSurface]: _description_
        """
        return [geo for geo in self.all_geometries() if geo.dim == 2]

    def find_geo(self, name: str, model: str | None = None) -> GeoObject:
        if model is None:
            model = self.active
        if name in self.geometry_list[model]:
            return self.geometry_list[model][name]
        for key, geo in self.geometry_list[model].items():
            if name.lower() in key:
                return geo
        raise ValueError(
            f"Couldn't find a geometry by name {name}. Only know: {self.all_names(model)}"
        )

    def self_destruct(self):
        for geo in self.geometry_list[self.active].values():
            geo._do_self_destruct()
        gmsh.model.occ.synchronize()

    def all_geometries(self, model: str | None = None) -> list[GeoObject]:
        if model is None:
            model = self.active
        return [geo for geo in self.geometry_list[model].values() if geo._exists]

    def all_plottable_geometries(self, model: str | None = None) -> list[GeoObject]:
        if model is None:
            model = self.active
        return [geo for geo in self.geometry_list[model].values() if geo._exists and geo.dim in (2,3)]
    
    def set_geometries(self, geos: list[GeoObject], model: str | None = None):
        if model is None:
            model = self.active
        self.geometry_list[model] = {geo.name: geo for geo in geos}

    def all_names(self, model: str | None = None) -> set[str]:
        if model is None:
            model = self.active
        return set(self.geometry_list[model].keys())
        # return set([geo.name for geo in self.all_geometries(model)])

    def get_name(self, suggestion: str, model: str | None = None) -> str:
        names = self.all_names(model)
        if suggestion not in names:
            return suggestion
        for i in range(1_000_000):
            if f"{suggestion}_{i}" not in names:
                return f"{suggestion}_{i}"
        raise RuntimeError("Cannot generate a unique name.")

    def set_name(self, geo_obj: GeoObject) -> None:
        self.geometry_list[self.active][geo_obj.name] = geo_obj
        to_remove = []
        for key, obj in self.geometry_list[self.active].items():
            if obj is geo_obj and key != geo_obj.name:
                to_remove.append(key)
        for key in to_remove:
            del self.geometry_list[self.active][key]

    def submit_geometry(self, geo: GeoObject, model: str | None = None) -> None:
        if model is None:
            model = self.active
        self.geometry_list[model][geo.name] = geo

    def sign_in(self, modelname: str) -> None:
        self.geometry_list[modelname] = dict()
        self.active = modelname

    def reset(self) -> None:
        self.geometry_list: dict[str, dict[str, GeoObject]] = dict()
        self.active: str = ""

    def clear(self) -> None:
        self.geometry_list = dict()
        self.active = ""

    def lowest_priority(self) -> int:
        return min([geo._base_priority for geo in self.all_geometries()])

    def highest_priority(self) -> int:
        return max([geo._base_priority for geo in self.all_geometries()])

    def __getitem__(self, key: str) -> GeoObject:
        return self.geometry_list[self.active][key]


class _FacePointer(Saveable):
    """The FacePointer class defines a face to be selectable as a
    face normal vector plus an origin. All faces of an object
    can be selected based on the projected distance to the defined
    selection plane of the center of mass of a face iff the normals
    also align with some tolerance.

    """

    def __init__(self, origin: np.ndarray, normal: np.ndarray):
        self.o = np.array(origin)
        self.n = np.array(normal)
        self.n = self.n / np.linalg.norm(self.n)

    def _apply_size_scaling(self, factor: float) -> None:
        self.o = self.o/factor

    def find(
        self,
        dimtags: list[tuple[int, int]],
        origins: list[np.ndarray],
        normals: list[np.ndarray],
    ) -> list[int]:
        tags = []
        for (d, t), o, n in zip(dimtags, origins, normals):
            normdist = np.abs((o - self.o) @ self.n)
            dotnorm = np.abs(n @ self.n)
            if normdist < 1e-5 and dotnorm > 0.999:
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
        rel_o = self.o - c0  # move to rotation-centre coordinates
        rot_o = rodrigues(rel_o)  # rotate
        self.o = rot_o + c0  # move back

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

    def __eq__(self, other: _FacePointer) -> bool:
        return (
            np.linalg.norm(self.o - other.o) + np.linalg.norm(self.n - other.n)
        ) < 1e-6




class AnchorSet(Saveable):
    """The AnchorSet class is a cartesian bounding box
    class with a set of Anchors as points plus orientations on the
    faces, edges and vertices of the bounding box including the center.

    Points and axes are indicated by the letters: x, y, z
    where lower-case is used for negative axes and upper-case for positive.
        x = -x = left
        X = +x = right
        y = -y = front
        Y = +y = back
        z = -z = bottom
        Z = +z = top

    alternative naming convention (Rubix cube):
        L/l = left
        R/r = right
        F/f = front
        B/b = back
        U/u = top
        D/d = bottom

    Examples:
        x: -x side -> Left face
        xy: -x,-y edge -> front left edge
        XyZ: +x,-y,+z corner -> front right top corner

    Attributes:
        x:    The right side
    """

    name_map: dict[str, str] = {
        "l": "x",
        "L": "x",
        "r": "X",
        "R": "X",
        "f": "y",
        "F": "y",
        "b": "Y",
        "B": "Y",
        "d": "z",
        "D": "z",
        "u": "Z",
        "U": "Z",
    }

    def __init__(self) -> None:

        self.c: Anchor = None
        """ The Center point"""
        self.x: Anchor = None
        """ The Left side"""
        self.X: Anchor = None
        """ The Right side"""
        self.y: Anchor = None
        """ The front side"""
        self.Y: Anchor = None
        """The Back side"""
        self.z: Anchor = None
        """The bottom side"""
        self.Z: Anchor = None

        """The top side"""
        self.xy: Anchor = None
        """The front left edge"""
        self.xY: Anchor = None
        """The back left edge"""
        self.Xy: Anchor = None
        """The front right edge"""
        self.XY: Anchor = None
        """The back right edge"""
        self.xz: Anchor = None
        """The bottom left edge"""
        self.xZ: Anchor = None
        """The top left edge"""
        self.Xz: Anchor = None
        """The bottom right edge"""
        self.XZ: Anchor = None
        """The top right edge"""
        self.yz: Anchor = None
        """The front bottom edge"""
        self.yZ: Anchor = None
        """The front top edge"""
        self.Yz: Anchor = None
        """The back bottom edge"""
        self.YZ: Anchor = None
        """The back top edge"""
        self.xyz: Anchor = None
        """The front bottom left corner"""
        self.xyZ: Anchor = None
        """The front top left corner"""
        self.xYz: Anchor = None
        """The bottom back left corner"""
        self.Xyz: Anchor = None
        """The bottom back right corner"""
        self.xYZ: Anchor = None
        """The back left top corner"""
        self.XyZ: Anchor = None
        """The front right top corner"""
        self.XYz: Anchor = None
        """The back right bottom corner"""
        self.XYZ: Anchor = None
        """The back top right corner"""
        self.initialized: bool = False
        self._args: tuple[np.ndarray, ...] = None
        self._anchors: list[Anchor] = []
        self.tool: dict[int, AnchorSet] = dict()

    def __call__(self, other: int | GeoObject) -> AnchorSet:
        if isinstance(other, GeoObject):
            key = other._key
        return self.tool[key]

    def __getattr__(self, name: str) -> Anchor:
        for k, v in self.name_map.items():
            name = name.replace(k, v)
        name = "".join(sorted(name, key=str.lower))
        if name in self.__dict__:
            return self.__dict__[name]
        raise AttributeError(f"AnchorSet has no attribute {name}")

    def _apply_size_scaling(self, factor: float) -> None:
        for anch in self._anchors:
            anch.c0 = anch.c0/factor
        for anchset in self.tool.values():
            anchset._apply_size_scaling(factor)

    def take_as_tools(self, key: int, other: AnchorSet) -> None:
        self.tool[key] = other
        self.tool.update(other.tool)

    def update(self, other: AnchorSet) -> None:
        self.init(*other._args)
        self.tool.update(other.tool)

    def copy(self) -> AnchorSet:
        """Copies the anchor set

        Returns:
            AnchorSet: _description_
        """
        anch = AnchorSet()
        if self._args is not None:
            anch.init(*self._args)
            anch.tool = {key: value.copy() for key, value in self.tool.items()}
            anch._args = self._args
        return anch

    def all_anchors(self) -> list[Anchor]:
        """Returns a list of all anchors"""
        anchors = self._anchors
        for key, value in self.tool.items():
            anchors.extend(value._anchors)
        return anchors

    def init_corners(
        self,
        xyz: np.ndarray,
        xYz: np.ndarray,
        Xyz: np.ndarray,
        XYz: np.ndarray,
        xyZ: np.ndarray,
        xYZ: np.ndarray,
        XyZ: np.ndarray,
        XYZ: np.ndarray,
        dx: np.ndarray,
        dy: np.ndarray,
        dz: np.ndarray,
    ) -> AnchorSet:

        xy = (xyz + xyZ) / 2
        xY = (xYz + xYZ) / 2
        Xy = (Xyz + XyZ) / 2
        XY = (XYz + XYZ) / 2
        xz = (xyz + xYz) / 2
        xZ = (xyZ + xYZ) / 2
        Xz = (Xyz + XYz) / 2
        XZ = (XyZ + XYZ) / 2
        yz = (xyz + Xyz) / 2
        yZ = (xyZ + XyZ) / 2
        Yz = (xYz + XYz) / 2
        YZ = (xYZ + XYZ) / 2
        x = (xy + xY) / 2
        X = (Xy + XY) / 2
        y = (xy + Xy) / 2
        Y = (xY + XY) / 2
        z = (xz + Xz) / 2
        Z = (xZ + XZ) / 2
        c = (x + X) / 2

        self.x = Anchor(x, dx, dy, dz)
        self.X = Anchor(X, dx, dy, dz)
        self.y = Anchor(y, dx, dy, dz)
        self.Y = Anchor(Y, dx, dy, dz)
        self.z = Anchor(z, dx, dy, dz)
        self.Z = Anchor(Z, dx, dy, dz)
        self.c = Anchor(c, dx, dy, dz)
        self.xy = Anchor(xy, dx, dy, dz)
        self.xY = Anchor(xY, dx, dy, dz)
        self.Xy = Anchor(Xy, dx, dy, dz)
        self.XY = Anchor(XY, dx, dy, dz)
        self.xz = Anchor(xz, dx, dy, dz)
        self.xZ = Anchor(xZ, dx, dy, dz)
        self.Xz = Anchor(Xz, dx, dy, dz)
        self.XZ = Anchor(XZ, dx, dy, dz)
        self.yz = Anchor(yz, dx, dy, dz)
        self.yZ = Anchor(yZ, dx, dy, dz)
        self.Yz = Anchor(Yz, dx, dy, dz)
        self.YZ = Anchor(YZ, dx, dy, dz)
        self.xyz = Anchor(xyz, dx, dy, dz)
        self.xyZ = Anchor(xyZ, dx, dy, dz)
        self.xYz = Anchor(xYz, dx, dy, dz)
        self.Xyz = Anchor(Xyz, dx, dy, dz)
        self.xYZ = Anchor(xYZ, dx, dy, dz)
        self.XyZ = Anchor(XyZ, dx, dy, dz)
        self.XYz = Anchor(XYz, dx, dy, dz)
        self.XYZ = Anchor(XYZ, dx, dy, dz)

        self._anchors.append(self.c)
        self.initialized = True
        for cx in ("x", "X", ""):
            for cy in ("y", "Y", ""):
                for cz in ("z", "Z", ""):
                    if cx == "" and cy == "" and cz == "":
                        continue
                    self._anchors.append(getattr(self, f"{cx}{cy}{cz}"))
        return self

    def init(
        self, p0: np.ndarray, dx: np.ndarray, dy: np.ndarray, dz: np.ndarray
    ) -> AnchorSet:
        self._args = (p0, dx, dy, dz)
        vecs = {
            "x": -dx,
            "X": dx,
            "y": -dy,
            "Y": dy,
            "z": -dz,
            "Z": dz,
            "": 0.0,
        }
        nx = dx / np.linalg.norm(dx)
        ny = dy / np.linalg.norm(dy)
        if np.linalg.norm(dz) == 0:
            nz = np.cross(nx, ny)
            nz = dz / np.linalg.norm(nz)
        else:
            nz = dz / np.linalg.norm(dz)

        anchor = Anchor(p0, nx, ny, nz)
        self.c = anchor
        self._anchors.append(anchor)

        for cx in ("x", "X", ""):
            for cy in ("y", "Y", ""):
                for cz in ("z", "Z", ""):
                    if cx == "" and cy == "" and cz == "":
                        continue
                    anchor = Anchor((p0 + vecs[cx] + vecs[cy] + vecs[cz]), nx, ny, nz)
                    setattr(self, f"{cx}{cy}{cz}", anchor)
                    self._anchors.append(anchor)

        self.initialized: bool = False
        return self


T = TypeVar("T", bound="GeoObject")


class GeoObject(Saveable):
    """A generalization of any OpenCASCADE entity described by a dimension and a set of tags."""

    dim: int = -1
    _default_name: str = "GeoObject"
    properties: PhysicalAttributeSet = PhysicalAttributeDescriptor()

    def __init__(
        self,
        tags: list[int] | None = None,
        name: str | None = None,
        _submit_geometry: bool = True,
    ):
        if tags is None:
            tags = []
        self.old_tags: list[int] = []
        self.tags: list[int] = tags
        self.mesh_multiplier: float = 1.0
        self.max_meshsize: float = 1e9

        self._unset_constraints: bool = False
        self._embeddings: list[GeoObject] = []
        self._face_pointers: dict[str, _FacePointer] = dict()
        self.anch: AnchorSet = AnchorSet()
        self._tools: dict[int, dict[str, _FacePointer]] = dict()
        self._hidden: bool = False
        self._key = _GlobalHandler.active().generator.new()
        
        self._base_priority: int = 10.0
        self._sub_priority: int = 0
        self._self_destruct: bool = False

        self._get_boundary_cache: list[tuple[int, int]] = None
        self._cached: bool = False

        self._exists: bool = True

        self.give_name(name)
        if _submit_geometry:
            _GlobalHandler.active().geomanager.submit_geometry(self)

        if self.dim==3:
            self.properties += AIR
        

    @property
    def material(self) -> Material:
        return self.properties.get(Material)
    
    @property
    def _priority(self) -> float:
        """The Priority of the geometry material

        Returns:
            float: The Priority of the geometry material
        """
        return self._base_priority + self._sub_priority / 2

    def _become(self: T, other: GeoObject) -> T:
        """In-place mutating transformation of self into another GeoObject instance.
        
        Transfers all internal attributes, tags, anchors, and geometric properties
        from `other` to `self`, deactivates `other`, and updates the geometry manager.
        """
        # 1. Clean up old GMSH entity tags if self is completely replaced
        if self._exists and self.dimtags != other.dimtags:
            self._do_self_destruct()

        # 2. Preserve self's name and registration key identity
        target_name = self.name

        # 3. Swap instance state completely
        self.__dict__.clear()
        self.__dict__.update(other.__dict__)

        # 4. Restore original name to avoid global manager lookup conflicts
        self.name = target_name

        # 5. Deactivate and mark the transient object as removed
        other._exists = False
        other.deactivate() if hasattr(other, "deactivate") else None

        # 6. Re-register self with the active GeometryManager
        _GlobalHandler.active().geomanager.submit_geometry(self)

        return self
    
    def _apply_size_scaling(self, factor: float) -> None:
        """Apply a size scaling to all geometric quantities

        Args:
            factor (float): The global scaling factor
        """
        self.anch._apply_size_scaling(factor)

        for face_pointer in self._face_pointers.values():
            face_pointer._apply_size_scaling(factor)

        for toolfps in self._tools.values():
            for face_pointer in toolfps.values():
                face_pointer._apply_size_scaling(factor)
        
    def _do_self_destruct(self) -> None:
        if self._self_destruct:
            self.remove()

    def _fill_face_pointers(self) -> None:
        """Fills the list of all face pointers of this object"""
        current = list(self._face_pointers.values())
        ctr = 0
        gmsh.model.occ.synchronize()
        for dim, tag in gmsh.model.get_boundary(self.dimtags, True, False):
            if dim != 2:
                continue
            o = gmsh.model.occ.get_center_of_mass(2, tag)
            n = gmsh.model.get_normal(tag, (0, 0))
            fp = _FacePointer(o, n)
            if fp not in current:
                self._face_pointers[f"Face{ctr}"] = fp
                ctr += 1

    def _get_boundary(self) -> list[tuple[float, float]]:
        if not self._cached:
            return gmsh.model.get_boundary(self.dimtags, True, False)
        else:
            return self._get_boundary_cache

    def _cache_gmsh(self) -> None:
        """
        Caches GMSH calls for after loading.
        """
        self._get_boundary_cache = gmsh.model.get_boundary(self.dimtags, True, False)
        self._cached = True

    def give_name(self, name: str | None = None) -> GeoObject:
        """Assign a name to this object

        Args:
            name (str | None, optional): The name for the object. Defaults to None.
        """
        if name is None:
            name = self._default_name
        self.name: str = _GlobalHandler.active().geomanager.get_name(name)
        _GlobalHandler.active().geomanager.set_name(self)
        return self

    @property
    def color_str(self) -> str:
        """The string getter for the object color"""
        if mat := self.material:
            return mat.color
        return "#999999"

    @property
    def color_rgb(self) -> tuple[float, float, float]:
        """The color of the object in RGB float tuple

        Returns:
            tuple[float, float, float]: The color
        """
        if mat := self.material:
            return mat.color_rgb
        return (1.0, 1.0, 1.0)

    @property
    def opacity(self) -> float:
        """The opacity of the object

        Returns:
            float: The opacity
        """
        if mat := self.material:
            return mat.opacity
        return 0.0

    @property
    def _metal(self) -> bool:
        """If the material should be rendered as metal"""
        if mat := self.material:
            return mat._metal
        return False

    @property
    def selection(self) -> Selection:
        """Returns a corresponding Face/Domain or Edge Selection object"""
        if self.dim == 1:
            return EdgeSelection(self.tags)
        elif self.dim == 2:
            return FaceSelection(self.tags)
        elif self.dim == 3:
            return DomainSelection(self.tags)
        else:
            return Selection(self.tags)

    @staticmethod
    def merged(
        objects: list[GeoPoint | GeoEdge | GeoSurface | GeoVolume | GeoObject],
    ) -> (
        list[GeoPoint | GeoEdge | GeoSurface | GeoVolume | GeoObject]
        | GeoPoint
        | GeoEdge
        | GeoSurface
        | GeoVolume
        | GeoObject
    ):
        """Create a GeoObject by merging an iterable of GeoObjects

        Args:
            objects (list[GeoPoint  |  GeoEdge  |  GeoSurface  |  GeoVolume  |  GeoObject]): A list of geo objects

        Returns:
            GeoPoint | GeoEdge | GeoSurface | GeoVolume | GeoObject: The resultant object
        """
        dim = objects[0].dim
        tags = []
        out: GeoObject | None = None
        for obj in objects:
            tags.extend(obj.tags)
            obj.deactivate()

        if dim == 2:
            out = GeoSurface(tags)
        elif dim == 3:
            out = GeoVolume(tags)
        else:
            out = GeoObject(tags)

        out.properties += objects[0].properties.copy(deep=True)
        out.prio_set(objects[0]._priority)
        out.give_name(f"MergedGeometries{[obj.name for obj in objects]}")
        return out

    def remove_tags(self, tags: list[int]) -> None:
        """Removes tags from this assignment

        Args:
            tags (list[int]): _description_
        """
        self.tags = [tag for tag in self.tags if tag not in tags]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.name},{self.dim},{self.tags})"

    def _replace_pointer(self, name: str, face_pointer: _FacePointer) -> None:
        """Will be used to replace face pointers so only one unique one exists.

        Args:
            name (str): The name of the pointer
            face_pointer (_FacePointer): THe new face pointer objet
        """
        for key, fp in self._face_pointers.items():
            if fp == face_pointer:
                self._face_pointers.pop(key)
                break
        self._face_pointers[name] = face_pointer

    def _add_face_pointer(
        self,
        name: str,
        origin: np.ndarray | None = None,
        normal: np.ndarray | None = None,
        tag: int | None = None,
    ):
        """Adds a face identifier (face pointer) to this object

        Args:
            name (str): The name for the face
            origin (np.ndarray | None, optional): A point on the object. Defaults to None.
            normal (np.ndarray | None, optional): The normal of the face. Defaults to None.
            tag (int | None, optional): The tace tag used to extract the origin and normal. Defaults to None.

        """
        if tag is not None:
            origin = gmsh.model.occ.get_center_of_mass(2, tag)
            normal = gmsh.model.get_normal(tag, (0, 0))

        if origin is None or normal is None:
            raise ValueError("Eitehr a tag or an origin + normal must be provided!")
        fp = _FacePointer(origin, normal)
        self._face_pointers[name] = fp

    def make_copy(self: T) -> T:
        """Copies this object and returns a new object (also in GMSH)"""
        new_dimtags = gmsh.model.occ.copy(self.dimtags)
        new_obj = GeoObject.from_dimtags(new_dimtags)
        new_obj.mesh_multiplier = self.mesh_multiplier
        new_obj.max_meshsize = self.max_meshsize
        new_obj.properties = self.properties.copy(deep=True)
        new_obj._unset_constraints = self._unset_constraints
        new_obj._embeddings = [emb.make_copy() for emb in self._embeddings]
        new_obj._face_pointers = {
            key: value.copy() for key, value in self._face_pointers.items()
        }
        new_obj._tools = {
            key: {key2: value2.copy() for key2, value2 in value.items()}
            for key, value in self._tools.items()
        }
        new_obj.anch = self.anch.copy()

        new_obj._base_priority = self._base_priority
        new_obj._exists = self._exists
        new_obj._self_destruct = self._self_destruct
        return new_obj

    def replace_tags(self, tagmap: dict[int, list[int]]):
        """Replaces the GMSH tags assigned to this objects

        Args:
            tagmap (dict[int, list[int]]): A map that shows which tag is mapped to which set of new tags.
        """
        self.old_tags = self.tags
        newtags = []
        for tag in self.tags:
            newtags.extend(
                tagmap.get(
                    tag,
                    [
                        tag,
                    ],
                )
            )
        self.tags = newtags
        logger.debug(f"{self} Replaced {self.old_tags} -> {self.tags}")

    def update_tags(self: T, tag_mapping: dict[int, dict]) -> T:
        """Update the tag definition of a GeoObject after fragementation."""
        self.replace_tags(tag_mapping[self.dim])
        return self

    def _take_pointers(self: T, *others: GeoObject) -> T:
        for other in others:
            self._face_pointers.update(other._face_pointers)
            self._tools.update(other._tools)
            self.anch.update(other.anch)
        return self

    @property
    def _all_anchors(self) -> list[Anchor]:
        return self.anch.all_anchors()

    @property
    def _all_pointers(self) -> list[_FacePointer]:
        pointers = list(self._face_pointers.values())
        for dct in self._tools.values():
            pointers.extend(list(dct.values()))
        return pointers

    @property
    def _all_transformable(self) -> list[Anchor | _FacePointer]:
        items: list[Anchor | _FacePointer] = []
        items.extend(self._all_anchors)
        items.extend(self._all_pointers)
        return items

    @property
    def _all_pointer_names(self) -> set[str]:
        keys = set(self._face_pointers.keys())
        for dct in self._tools.values():
            keys = keys.union(set(dct.keys()))
        return keys

    def _take_tools(self: T, *objects: GeoObject) -> T:
        for obj in objects:
            self._tools[obj._key] = obj._face_pointers
            self._tools.update(obj._tools)
            self.anch.take_as_tools(obj._key, obj.anch)
        return self

    def set_material(self: T, material: Material) -> T:
        """Assign a material to this geometry object.

        Args:
            material (Material): The material to assign

        Returns:
            GeoObject: This same object
        """
        if isinstance(material, Material):
            self.properties += material
            self._prio_half_up()
        else:
            logger.warning(f'Trying to assign material {material} to {self} which is not of type Material. Skipping the assignment.')
        return self

    def prio_set(self: T, level: int) -> T:
        """Defines the material assignment priority level of this geometry.
        By default all objects have priority level 10. If you assign a lower number,
        in cases where multiple geometries occupy the same volume, the highest priority
        will be chosen.

        Args:
            level (int): The priority level

        Returns:
            GeoObject: The same object
        """
        self._base_priority = int(level)
        return self

    def _prio_half_up(self: T) -> T:
        """Adds one half to the priority"""

        self._sub_priority = 1

    def above(self: T, other: GeoObject) -> T:
        """Puts the priority of this object one higher than the other, then returns this object

        Args:
            other (GeoObject): The other object to put below this object

        Returns:
            GeoObject: This object
        """
        self._base_priority = other._base_priority + 1.0
        return self

    def below(self: T, other: GeoObject) -> T:
        """Puts the priority of this object one lower than the other, then returns this object

        Args:
            other (GeoObject): The other object to put above this object

        Returns:
            GeoObject: This object
        """
        self._base_priority = other._base_priority - 1
        return self

    def prio_up(self: T) -> T:
        """Increases the material selection priority by 1"""
        self._base_priority += 1
        return self

    def prio_down(self: T) -> T:
        """Decreases the material selection priority by 1"""
        self._base_priority -= 1
        return self

    def background(self: T) -> T:
        """Set the material selection priority to be on the background."""
        self._base_priority = _GlobalHandler.active().geomanager.lowest_priority() - 10
        return self

    def foreground(self: T) -> T:
        """Set the material selection priority to be on top."""
        self._base_priority = _GlobalHandler.active().geomanager.highest_priority() + 10
        return self

    
    def hide(self: T) -> T:
        """Hides the object from views"""
        self._hidden = True
        return self

    def unhide(self: T) -> T:
        """Unhides the object from views"""
        self._hidden = False
        return self

    @property
    def dimtags(self) -> list[tuple[int, int]]:
        return [(self.dim, tag) for tag in self.tags]

    @property
    def embeddings(self) -> list[tuple[int, int]]:
        return []

    @staticmethod
    def from_dimtags(
        dimtags: list[tuple[int, int]], _submit_geometry: bool = True
    ) -> GeoVolume | GeoSurface | GeoObject:
        dim = dimtags[0][0]
        tags = [t for d, t in dimtags]
        if dim == 0:
            return GeoPoint(tags, _submit_geometry=_submit_geometry)
        elif dim == 1:
            return GeoEdge(tags, _submit_geometry=_submit_geometry)
        if dim == 2:
            return GeoSurface(tags, _submit_geometry=_submit_geometry)
        if dim == 3:
            return GeoVolume(tags, _submit_geometry=_submit_geometry)
        return GeoObject(tags, _submit_geometry=_submit_geometry)

    def remove(self) -> None:
        """Delete the geometry from the simulation model"""
        self._exists = False
        gmsh.model.occ.remove(self.dimtags, True)

    def deactivate(self) -> None:
        """Make the geometry not considered as a valid EMerge geometry"""
        self._exists = False

    def extract(self: T, tags: int | list[int]) -> T:
        """Returns a new GeoObject of the same dimensional type that isolates a set of given tags

        Args:
            tags (list[int]): A list of GMSH tags

        Returns:
            GeoObject: The resultant geometry object.
        """
        if isinstance(tags, int):
            tags = [
                tags,
            ]
        self.tags = [t for t in self.tags if t not in tags]
        dts = [(self.dim, t) for t in tags]
        return GeoObject.from_dimtags(dts)


class GeoVolume(GeoObject):
    """GeoVolume is an interface to the GMSH CAD kernel. It does not represent EMerge
    specific geometry data."""

    dim = 3
    _default_name: str = "GeoVolume"

    def __init__(
        self, tag: int | Iterable[int], name: str | None = None, _submit_geometry=True
    ):
        super().__init__(name=name, _submit_geometry=_submit_geometry)


        self.tags: list[int] = []

        if isinstance(tag, Iterable):
            self.tags = list(tag)
        else:
            self.tags = [
                tag,
            ]
        self.tags = self._rebuild_positive_volume(self.tags)

        self._normals_cache: list[tuple[float, float, float]] = None
        self._origins_cache: list[tuple[float, float, float]] = None
        self._face_tag_cache: dict[str, list[int]] = dict()

        self._fill_face_pointers()
        self._autoname()

    @staticmethod
    def _check_negative_volume(tags: list[int]) -> bool:
        for t in tags:
            if gmsh.model.occ.get_mass(3,t) < 0:
                return True
        return False

    @staticmethod
    def _rebuild_positive_volume(tags: list[int]) -> list[int]:
        """
        Extract surface loops from 3D entities and rebuild them as positive-mass volumes.
        """
        
        if not GeoVolume._check_negative_volume(tags):
            return tags

        logger.warning(f'Negative volumes detected for dimtags {tags}. Reconstructing topology.')

        gmsh.model.occ.synchronize()
        new_solids = []
        
        for tag in tags:
            surfloop, _ = gmsh.model.occ.getSurfaceLoops(tag)
            gmsh.model.occ.remove([(3, tag)])
            new_volume = gmsh.model.occ.addVolume(surfloop)
            new_solids.append((3, new_volume))

        gmsh.model.occ.synchronize()
        return new_solids

    def _apply_size_scaling(self, factor: float) -> None:
        """Apply a size scaling to all geometric quantities

        Args:
            factor (float): The global scaling factor
        """
        super()._apply_size_scaling(factor)
        self._origins_cache = [(x/factor, y/factor, z/factor) for x,y,z in self._origins_cache]
                
    def _cache_gmsh(self):
        self._fill_face_pointers()
        self._get_boundary_cache = gmsh.model.get_boundary(self.dimtags, True, False)
        self._normals_cache = [
            gmsh.model.get_normal(t, [0, 0]) for d, t in self._get_boundary_cache
        ]
        self._origins_cache = [
            gmsh.model.occ.get_center_of_mass(d, t) for d, t in self._get_boundary_cache
        ]
        self._cached = True
        for name in self._face_pointers.keys():
            self._face_tag_cache[name] = self._face_tags(name)
        
        for tool, fpdict in self._tools.items():
            for name in fpdict.keys():
                self._face_tag_cache[f'{tool}.{name}'] = self._face_tags(name, tool_key=tool)

    def _get_normal(self):
        if not self._cached:
            return [
                gmsh.model.get_normal(t, [0, 0]) for d, t in self._get_boundary_cache
            ]
        else:
            return self._normals_cache

    def _get_center_of_mass(self) -> list[tuple[float, float, float]]:
        if not self._cached:
            return [
                gmsh.model.occ.get_center_of_mass(d, t)
                for d, t in self._get_boundary_cache
            ]
        else:
            return self._origins_cache

    @property
    def selection(self) -> DomainSelection:
        return DomainSelection(self.tags)

    def _auto_face_tag(self) -> GeoVolume:
        self._face_pointers = dict()
        self._tools = dict()
        logger.trace("Automatically assigning face pointers.")
        ctr = 0
        for tag in self.tags:
            loops = gmsh.model.occ.getSurfaceLoops(tag)
            for loopcluster in loops[1]:
                for st in loopcluster:
                    self._add_face_pointer(f"Face{ctr}", tag=st)
                    ctr += 1
        return self

    def _set_bb_anchors(self) -> None:
        """Sets the bounding box anchor points for the volume"""
        xmins = []
        xmaxs = []
        ymins = []
        ymaxs = []
        zmins = []
        zmaxs = []
        for tag in self.tags:
            x_min, y_min, z_min, x_max, y_max, z_max = gmsh.model.occ.getBoundingBox(
                3, tag
            )
            xmins.append(x_min)
            xmaxs.append(x_max)
            ymins.append(y_min)
            ymaxs.append(y_max)
            zmins.append(z_min)
            zmaxs.append(z_max)

        x_min = min(xmins)
        x_max = max(xmaxs)
        y_min = min(ymins)
        y_max = max(ymaxs)
        z_min = min(zmins)
        z_max = max(zmaxs)

        dx = np.array([1.0, 0.0, 0.0]) * (x_max - x_min) / 2
        dy = np.array([0.0, 1.0, 0.0]) * (y_max - y_min) / 2
        dz = np.array([0.0, 0.0, 1.0]) * (z_max - z_min) / 2
        p0 = np.array([(x_max + x_min) / 2, (y_max + y_min) / 2, (z_max + z_min) / 2])
        self.anch.init(p0, dx, dy, dz)

    def exterior_faces(self, base_object: GeoObject) -> FaceSelection:
        """Select the exterior faces of an object based on the face
        pointers of an original base object. For example

        Example:
        >>> cheese = em.geo.Box(...)
        >>> hole = em.geo.Box(...)
        >>> holed_cheese = em.geo.subtract(cheese, hole)
        >>> holed_cheese.exterior_faces(cheese)

        Args:
            base_object (GeoObject): The object to base the selection on.

        Returns:
            FaceSelection: The resultant face selection
        """
        dimtags = self._get_boundary()

        normals = self._get_normal()
        origins = self._get_center_of_mass()

        tool_tags = []
        for key, tool in self._tools.items():
            if key == base_object._key:
                continue
            for name in tool.keys():
                tags = tool[name].find(dimtags, origins, normals)
                tool_tags.extend(tags)
        tags = [dt[1] for dt in dimtags if dt[1] not in tool_tags]
        return FaceSelection(tags)

    def _find_fp(self, condition: Callable):
        for key, fp in self._face_pointers.items():
            if condition(fp):
                return key, fp
        return None, None

    def _rename_fp(self, new_name: str, condition: Callable):
        name, fp = self._find_fp(condition)
        if fp is not None:
            self._face_pointers.pop(name)
            self._face_pointers[new_name] = fp

    def _autoname(self) -> None:
        if len(self._face_pointers) == 0:
            return

        xs = []
        ys = []
        zs = []
        for fp in self._face_pointers.values():
            xs.append(fp.o[0])
            ys.append(fp.o[1])
            zs.append(fp.o[2])
        minx = min(xs)
        maxx = max(xs)
        miny = min(ys)
        maxy = max(ys)
        minz = min(zs)
        maxz = max(zs)

        self._rename_fp(
            "-x",
            lambda fp: (
                (fp.o[0] == minx) and np.abs(np.dot(fp.n, np.array([1, 0, 0]))) > 0.999
            ),
        )
        self._rename_fp(
            "+x",
            lambda fp: (
                (fp.o[0] == maxx) and np.abs(np.dot(fp.n, np.array([1, 0, 0]))) > 0.999
            ),
        )
        self._rename_fp(
            "-y",
            lambda fp: (
                (fp.o[1] == miny) and np.abs(np.dot(fp.n, np.array([0, 1, 0]))) > 0.999
            ),
        )
        self._rename_fp(
            "+y",
            lambda fp: (
                (fp.o[1] == maxy) and np.abs(np.dot(fp.n, np.array([0, 1, 0]))) > 0.999
            ),
        )
        self._rename_fp(
            "-z",
            lambda fp: (
                (fp.o[2] == minz) and np.abs(np.dot(fp.n, np.array([0, 0, 1]))) > 0.999
            ),
        )
        self._rename_fp(
            "+z",
            lambda fp: (
                (fp.o[2] == maxz) and np.abs(np.dot(fp.n, np.array([0, 0, 1]))) > 0.999
            ),
        )

    def boundary(
        self,
        exclude: Iterable[FaceNames, ...] | str | None = None,
        tags: list[int] | None = None,
        tool: GeoObject | None = None,
    ) -> FaceSelection:
        """Returns the complete set of boundary faces.

        If implemented, it is possible to exclude a set of faces based on their name
        or a list of tags.

        Args:
            exclude: (Iterable[str], str, None): A single string or list/tuple of strings.
            tags: A list of face integers (if known)
            tool: The tool object to base the selection face names one.

        Returns:
            FaceSelection: The selected faces
        """
        # if not self._exists:
        #     raise SelectionError('Cannot select faces from an object that no longer exists.')

        if isinstance(exclude, str):
            exclude = (exclude,)

        if exclude is None:
            exclude = tuple()

        if tags is None:
            tags = []

        for name in exclude:
            tags.extend(self.face(name, tool=tool).tags)
        dimtags = self._get_boundary()
        selname = "Boundary"
        if exclude:
            selname = selname + "Except[" + ",".join(exclude) + "]"
        return FaceSelection([t for d, t in dimtags if t not in tags])._named(selname)

    def face(
        self,
        name: FaceNames = None,
        tool: GeoObject | None = None,
        no: FaceNames = None,
    ) -> FaceSelection:
        """Returns the FaceSelection for a given face name.

        The face name must be defined for the type of geometry.

        FaceNames include: front, back, left, right, top, bottom, disc

        Args:
            name (FaceNames): The name of the face to select.
            tool (GeoObject, None): Which object should be used as a source for the face selection.
            no (FaceNames): If everything BUT a face name should be selected, Equivalent to .boundary(exclude=name).

        Returns:
            FaceSelection: The selected face
        """
        if no is not None:
            return self.boundary(exclude=no)

        return FaceSelection(self._face_tags(name, tool))._named(name)

    def all_faces(self, tool: GeoObject | None = None) -> list[FaceSelection]:
        """Returns a list of all face selections of this object

        Returns:
            list[FaceSelection]: A list of all face selections
        """
        if tool is not None:
            return [
                self.face(name, tool=tool) for name in self._tools[tool._key].keys()
            ]
        return [self.face(name) for name in self._face_pointers]

    def faces(self, *names: FaceNames, tool: GeoObject | None = None) -> FaceSelection:
        """Returns the FaceSelection for a given face names.

        The face name must be defined for the type of geometry.

        Args:
            name (FaceNames): The name of the face to select.
            tool (GeoObject, None): The tool object to use as source of the selection.

        Returns:
            FaceSelection: The selected face
        """
        tags = []
        for name in names:
            tags.extend(self._face_tags(name, tool))
        return FaceSelection(tags)._named("Faces[" + ",".join(names) + "]")

    def _face_tags(self, name: FaceNames, tool: GeoObject | None = None, tool_key: int | None = None) -> list[int]:
        names = self._all_pointer_names

        if name not in names:
            raise ValueError(
                f"The face {name} does not exist in {self}. Only {list(self._face_pointers.keys())}"
            )

        cachename = name

        if tool is not None:
            cachename = f'{tool._key}.{name}'

        if tool_key is not None:
            cachename = f'{tool_key}.{name}'
        
        if cachename in self._face_tag_cache:
            return self._face_tag_cache[cachename]
        
        gmsh.model.occ.synchronize()
        dimtags = gmsh.model.get_boundary(self.dimtags, True, False)

        normals = [gmsh.model.get_normal(t, [0, 0]) for d, t in dimtags]
        origins = [gmsh.model.occ.get_center_of_mass(d, t) for d, t in dimtags]

        if tool is not None:
            tags = self._tools[tool._key][name].find(dimtags, origins, normals)
        if tool_key is not None:
            tags = self._tools[tool_key][name].find(dimtags, origins, normals)
        else:
            tags = self._face_pointers[name].find(dimtags, origins, normals)
        logger.trace(f" - Face {name} with tool {tool} on {self} resolved in tags: {tags}.")
        return tags


class GeoPoint(GeoObject):
    dim = 0
    _default_name: str = "GeoPoint"

    @property
    def selection(self) -> PointSelection:
        return PointSelection(self.tags)

    def __init__(
        self,
        tag: int | list[int],
        name: str | None = None,
        _submit_geometry: bool = True,
    ):
        super().__init__(name=name, _submit_geometry=_submit_geometry)

        self.tags: list[int] = []
        if isinstance(tag, Iterable):
            self.tags = list(tag)
        else:
            self.tags = [
                tag,
            ]


class GeoEdge(GeoObject):
    dim = 1
    _default_name: str = "GeoEdge"

    @property
    def selection(self) -> EdgeSelection:
        return EdgeSelection(self.tags)

    def __init__(
        self,
        tag: int | list[int],
        name: str | None = None,
        _submit_geometry: bool = True,
    ):
        super().__init__(name=name, _submit_geometry=_submit_geometry)
        self.tags: list[int] = []
        if isinstance(tag, Iterable):
            self.tags = list(tag)
        else:
            self.tags = [
                tag,
            ]


class GeoSurface(GeoObject):
    """GeoVolume is an interface to the GMSH CAD kernel. It does not reprsent Emerge
    specific geometry data."""

    dim = 2
    _default_name: str = "GeoSurface"

    @property
    def selection(self) -> FaceSelection:
        return FaceSelection(self.tags)

    def __init__(
        self,
        tag: int | list[int],
        name: str | None = None,
        _submit_geometry: bool = True,
    ):
        super().__init__(name=name, _submit_geometry=_submit_geometry)
        self.tags: list[int] = []
        if isinstance(tag, Iterable):
            self.tags = list(tag)
        else:
            self.tags = [
                tag,
            ]

    def assign_thickness(self, thickness: float) -> GeoSurface:
        """Assign a virtual thickness to this surface object. The thickness will be used
        by the EM assembler to assign the ThinCondutor boundary condition.

        Args:
            thickness (float): The thickness in meters

        Returns:
            GeoSurface: The same object
        """
        self._store("thickness", thickness)
        return self

class GeoPolygon(GeoSurface):
    _default_name: str = "GeoPolygon"

    def __init__(
        self, tags: list[int], name: str | None = None, _submit_geometry: bool = True
    ):
        super().__init__(tags, name=name, _submit_geometry=_submit_geometry)
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
        if item.dim != dim:
            raise ValueError(
                f"Cannot group of objects with a dissimilar dimensions. Trying to include {item} in a list of dimension {dim}."
            )
        tags.extend(item.tags)
    return Selection.from_dim_tags(dim, tags)
