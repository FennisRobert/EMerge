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
import numpy as np
from typing import Callable, Iterable
from .shapes import Box, Cylinder, Alignment
from ..geometry import GeoVolume, GeoObject
from .operations import rotate, mirror, translate, add, subtract, embed
from numbers import Number
from functools import reduce
from operator import mul
from ..cs import CoordinateSystem, GCS
from .polybased import XYPolygon

def get_flat_index(indices, shape):
    flat_index = 0
    for i, idx in enumerate(indices):
        stride = reduce(mul, shape[i+1:], 1)
        flat_index += idx * stride
    return flat_index


class Series:

    def __init__(self, values: Iterable[Number]):
        self.values: list[Number] = list(values)
        self.N: int = len(self.values)

    def __len__(self) -> int:
        return self.N
    
    def __getitem__(self, index: int):
        return self.values[index]

def get_count(args) -> int:
    N = 1
    for arg in args:
        if isinstance(arg,Series):
            N = max(N, arg.N)
            return N
        elif isinstance(arg, Iterable):
            return get_count(arg)
    return 1

def get_item(value: Number | Series | list, n: int) -> Number | Series | Iterable | None:
    if isinstance(value, Number):
        return value
    elif isinstance(value, Series):
        return value[n]
    elif isinstance(value, Iterable):
        return type(value)([get_item(v, n) for v in value])
    return None

def unpack(*args):
    N = get_count(args)
    return tuple(zip(*[[get_item(a,n) for a in args] for n in range(N)]))

ObjectTransformer = Callable[[GeoObject,], GeoObject]

class PartialFunction:

    def __init__(self, 
                 function: Callable, 
                 identity: bool = False,
                 permute: bool = False,
                 **kwargs):
        self.f: Callable = function
        self.kwargs: dict[str, float | Series] = kwargs
        self.identity: bool = identity
        self.permute: bool = permute
        self.N: int = 1
        self.kwargset: list[dict] | None = None

        for key, value in self.kwargs.items():
            if not isinstance(value, Series):
                continue
            self.N = max(self.N, value.N)
        
        # Include the identity operation
        if self.identity:
            self.N = self.N + 1
    
    def _compile(self) -> None:
        kwargset: list = []
        N = self.N

        if self.identity:
            kwargset.append(None)
            N = N - 1
        
        for n in range(N):
            kwargs = dict()
            for key,value in self.kwargs.items():
                if not isinstance(value, Series):
                    kwargs[key] = value
                else:
                    kwargs[key] = value[n]
            kwargset.append(kwargs)
        self.kwargset = kwargset

    def __call__(self, objects: list[GeoObject], index: int) -> list[GeoObject]:
        if self.kwargset is None:
            raise ValueError('self.kwargset is not yet defined.')
        kwargs = self.kwargset[index]
        if kwargs is None:
            return objects
        
        objects_out = []
        for obj in objects:
            objects_out.append(self.f(obj, **kwargs))
        return objects_out

class NDimContainer:

    def __init__(self):
        self.items: list= []
        self.dimensions: list[tuple[int,int]] = []
        self.expanded_dims: list[tuple[int,int,int]] = []
        self.map: np.ndarray = []
        self.Ncopies: int = 1

    @property
    def Ntot(self) -> int:
        """The Total number of different objects this NDimContainer acts on.
        """
        n = self.Ncopies
        for i,N in self.dimensions:
            n = n*N
        
        return n

    @property
    def Nop(self) -> int:
        """The number of operations corresponding to this NDimContainer

        Returns:
            int: The number of operations this container can support.
        """
        n = 1
        for i,N in self.dimensions:
            n = n*N
        
        return n
    
    @property
    def dimtup(self) -> tuple[int, ...]:
        """A dimension tuple containing the dimensions of the NDimContainer

        Returns:
            tuple[int]: A tuple of dimension sizes to be used in np.zeros(shape)
        """
        dimtup =  tuple([dim[1] for dim in self.dimensions])
        if self.Ncopies:
            dimtup = dimtup + (self.Ncopies,)
        return dimtup
    
    def _init(self) -> None:
        """Initialization function to set the index map and expanded dimension list.
        """
        i = 0
        for i,(ncopies,N) in enumerate(self.dimensions):
            for n in range(ncopies):
                self.expanded_dims.append((i,n,N))
        
        if self.Ncopies > 1:
            self.expanded_dims.append((i+1, 1, self.Ncopies))

        self.map = np.arange(self.Ntot).reshape(self.dimtup)

    def add_dim(self, N: int, same_axis: bool = False) -> None:
        """Aads a new dimension to iterate over.

        Args:
            N (int): The size of the dimension (number of differnt items contained)
            same_axis (bool, optional): Defines that the next iteration dimension is actually parallel
            to the last existing dimension. In this case N must be equal to the size of the last defined dimension. Defaults to False.

        Raises:
            ValueError: An error if the provided dimension is not correct.
        """
        if same_axis:
            if N != self.dimensions[-1][1]:
                raise ValueError('Trying to add a dimension with the same size as previous but the sizes are different.' + 
                                 f'The provided size is {N} but the last dimensions is {self.dimensions[-1][1]}')
            self.dimensions[-1] = (self.dimensions[-1][0]+1, N)
        else:
            self.dimensions.append((1,N))

    def set_copies(self, N: int) -> None:
        """Define how many original copies of the new object will be created.

        Args:
            N (int): The number of copies.
        """
        self.Ncopies = N

    def get(self, 
            dim_ex_dim: int,
            number: int) -> list:
        dimindex, niter, Nvariations = self.expanded_dims[dim_ex_dim]
        slclist = [slice(None) for _ in self.dimensions]
        slclist[dimindex] = number # type: ignore
        return list(self.map[tuple(slclist)].flatten())
        
        
class Modeler:
    
    def __init__(self):
        ## Function importing
        self.rotate = rotate
        self.add = add
        self.translate = translate
        self.remove = subtract
        self.subtract = subtract
        self.embed = embed
        self.mirror = mirror

        self.pre_transforms: list[PartialFunction] = []
        self.post_transforms: list[PartialFunction] = []
        self.last_object: GeoObject = None
        self.ndimcont: NDimContainer = NDimContainer()
        self._and: bool = False

    @property
    def AND(self) -> Modeler:
        """This method chain property ensures that the following transformation will be merged
        with the previous one. For example, a rotation plus a translation with AND in between
        both containing three operations will merge their operations so that each single value
        is paired up with the next. 

        For example, if you wish to create a three boxes that are rotated in increments of 90
        degrees and traslated by 1, 2 and 3 meters respectively (90deg, 1m) (180deg, 2m) and (270deg, 3m)
        you would do:

        >>> m.rotated(...,..., m.Series(90,180,270)).AND.translated(m.Series(1,2,3),...)

        If you would not use AND, the operations are permuted resulting in 9 boxes.
        """
        self._and = True
        return self

    def series(self, *values: Number) -> Series:
        """Returns a Series object to represent a series of values instead of a single value.
        
        A series of values can be used at any argument that accepts Series objects. If such a
        variable is provided as a series, the modeler will execute multiple iterations of the 
        object or transformation for each value in the series.

        Returns:
            Series: The generated Series object.

        Example:
        >>> modeler.box(modeler.Series(1,2,3), 2, 3)

        """
        return Series(values)
    
    def nvars(self) -> int:
        return self.ndimcont.Nop
    
    def _merge_object(self, objs: list[GeoObject]) -> GeoObject:
        """Merges a list of GeoObjects into a single GeoObject.

        Args:
            objs (list[GeoObject]): A list of GMSH objects

        Returns:
            GeoObject: A single GeoObject
        """
        tags = []
        for obj in objs:
            tags.extend(obj.tags)
        gmshobj = GeoObject.from_dimtags([(objs[0].dim, tag) for tag in tags])
        for obj in objs:
            gmshobj._take_tools(obj)
        return gmshobj
    
    def _clean(self) -> None:
        self.pre_transforms = []
        self._and = False
        self._combine = False

    def _add_dim(self, N: int) -> None:
        self.ndimcont.add_dim(N, self._and)
        self._and = False

    def _add_function(self, function: PartialFunction):
        self._add_dim(function.N)
        self.pre_transforms.append(function)

    def _apply_presteps(self, objects: list[GeoObject]) -> list[GeoObject]:
        self.ndimcont._init()
        for func in self.pre_transforms:
            func._compile()
        if not self.pre_transforms:
            return objects
        
        for i, function in enumerate(self.pre_transforms):
            for n in range(function.N):
                ids = self.ndimcont.get(i,n)
                sel = [objects[i] for i in ids]
                objects_out = function(sel, n)
                for j, obj in zip(ids, objects_out):
                    objects[j] = obj

        self._clean()
        return objects
    
    def rotated(self,
                c0: tuple[float, float, float],
                ax: tuple[float, float, float],
                angle: float | Series,
                ) -> Modeler:
        """Adds a rotation task to the geometry builder.

        Args:
            c0 (tuple[float, float, float]): The origin at which the rotation axis is placed.
            ax (tuple[float, float, float]): The Rotation axis to be used
            angle (float | Series): The angles or series of angles provided in degrees.

        Returns:
            Modeler: The same modeler instance
        """
        function = PartialFunction(rotate, c0=c0, ax=ax, angle=angle)
        self._add_function(function)
        return self
    
    def translated(self,
                   dx: float | Series,
                   dy: float | Series,
                   dz: float | Series) -> Modeler:
        """Adds a translation task to the geometry builder pipeline.
        The dx, dy, dz may either be a single displacement coordinate or a series of them in the 
        shape of a series object.

        Args:
            dx (float | Series): The x-displacement
            dy (float | Series): The y-displacement
            dz (float | Series): The z-displacement

        Returns:
            Modeler: The Modeler object
        """
        function = PartialFunction(translate, dx=dx, dy=dy, dz=dz)
        self._add_function(function)
        return self
    
    def mirrored(self,
               origin: tuple[float, float, float],
               direction: tuple[float, float, float],
               keep_original: bool = True) -> Modeler:
        """Adds a mirror transformation to the geometry builder pipeline.

        Args:
            origin (tuple[float, float, float]): The origin of the mirror axis plane
            direction (tuple[float, float, float]): The mirror direction which is normal to the plane of reflection.
            keep_original (bool, optional): Whether to keep the original object. Defaults to True.

        Returns:
            Modeler: The Modeler object
        """
        function = PartialFunction(mirror, 
                                                   identity=keep_original, 
                                                   origin=origin, 
                                                   direction=direction)
        self._add_function(function)
        return self
    
    def cylinder(self,
                  radius: float | Series,
                  height: float | Series,
                  position: tuple[float | Series, float | Series, float | Series],
                  cs: CoordinateSystem = GCS,
                  merge: bool = False,
                  NPoly: int = False) -> GeoVolume | list[GeoVolume]:
        
        N_objects = self.nvars()
        Rs, Hs, Ps = unpack(radius, height, position)
        N = len(Rs)
        cyls: list[GeoObject] = []
        
        for _ in range(N_objects):
            for r,h,p in zip(Rs, Hs, Ps):
                cs2 = cs.displace(p[0], p[1], p[2])
                if NPoly:
                    cyl = XYPolygon.circle(r, Nsections=NPoly).extrude(h, cs2)
                else:
                    cyl  = Cylinder(r,h, cs2)
                cyls.append(cyl)

        self.ndimcont.set_copies(N)
        self._apply_presteps(cyls)

        if merge:
            return self._merge_object(cyls)
        
        return cyls
    
    def box(self, width: float | Series,
            depth: float | Series,
            height: float | Series,
            position: tuple[float | Series, float | Series, float | Series],
            alignment: Alignment = Alignment.CORNER,
            merge: bool = False) -> list[GeoVolume] | GeoVolume:
        """Create a box object which will be transformed by the transformation pipeline.

        Args:
            width (float): The box's width (X direction)
            depth (float): The box's depth (Y direction)
            height (float): The box's height (Z direction)
            position (tuple[float, float, float]): The position of the box object.
            alignment (Alignment, optional): Where to alight the box. Defaults to Alignment.CORNER.
            merge (bool): Whether to merge the final result into a single GeoVolume object.

        Returns:
            list[GeoVolume]: A list of GeoVolume objects for each box.
        """
        N_objects = self.nvars()
        Ws, Ds, Hs, Ps = unpack(width, depth, height, position)
        N = len(Ws)
        boxes: list[GeoVolume] = []
        
        for _ in range(N_objects):
            for w, d, h, p in zip(Ws, Ds, Hs, Ps):
                box  = Box(w,d,h, position=p, alignment=alignment)
                boxes.append(box)
        self.ndimcont.set_copies(N)
        self._apply_presteps(boxes)

        if merge:
            return self._merge_object(boxes)
        
        return boxes