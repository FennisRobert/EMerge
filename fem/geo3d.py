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
import gmsh
from .material import Material, AIR
from .selection import FaceSelection, DomainSelection, EdgeSelection
from loguru import logger

class GMSHObject:
    """A generalization of any OpenCASCADE entity described by a dimension and a set of tags.
    """
    dim: int = -1
    def __init__(self):
        self.old_tags: list[int] = []
        self.tags: list[int] = []
        self.material: Material = AIR
        self.mesh_multiplier: float = 1.0
        self.max_meshsize: float = 1e9
        self._unset_constraints: bool = False
        self._embeddings: list[GMSHObject] = []

        self._aux_tags: dict[int, list[int]] = {0: [],
                                                1: [],
                                                2: [],
                                                3: []}

    @property
    def color(self) -> tuple[int,int,int]:
        return self.material.color
    
    @property
    def opacity(self) -> float:
        return self.material.opacity
    
    @property
    def select(self) -> FaceSelection | DomainSelection | EdgeSelection | None:
        '''Returns a corresponding Face/Domain or Edge Selection object'''
        if self.dim==1:
            return EdgeSelection(self.tags)
        elif self.dim==2:
            return FaceSelection(self.tags)
        elif self.dim==3:
            return DomainSelection(self.tags)
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.dim},{self.tags})'

    def replace_tags(self, tagmap: dict[int, list[int]]):
        self.old_tags = self.tags
        newtags = []
        for tag in self.tags:
            newtags.extend(tagmap.get(tag, [tag,]))
        self.tags = newtags
        logger.debug(f'Replaced {self.old_tags} with {self.tags}')
    
    def update_tags(self, tag_mapping: dict[int,dict]) -> None:
        ''' Update the tag definition of a GMSHObject after fragementation.'''
        self.replace_tags(tag_mapping[self.dim])

        for dim in range(4):
            new_tags = []
            for tag in self._aux_tags[dim]:
                new_tags.extend(tag_mapping[dim].get(tag, [tag,]))
            self._aux_tags[dim] = new_tags
        
    @property
    def dimtags(self) -> list[tuple[int, int]]:
        return [(self.dim, tag) for tag in self.tags]
    
    @property
    def embeddings(self) -> list[tuple[int,int]]:
        return []
    
    def boundary(self) -> FaceSelection:
        if self.dim == 3:
            tags = gmsh.model.get_boundary(self.dimtags, oriented=False)
            return FaceSelection([t[1] for t in tags])
        if self.dim == 2:
            return FaceSelection(self.tags)
        if self.dim < 2:
            raise ValueError('Can only generate faces for objects of dimension 2 or higher.')

    @staticmethod
    def from_dimtags(dim: int, tags: list[int]) -> GMSHVolume | GMSHSurface | GMSHObject:
        if dim==2:
            return GMSHSurface(tags)
        if dim==3:
            return GMSHVolume(tags)
        return GMSHObject(tags)
    
class GMSHVolume(GMSHObject):
    '''GMSHVolume is an interface to the GMSH CAD kernel. It does not represent EMerge
    specific geometry data.'''
    dim = 3
    def __init__(self, tag: int | list[int]):
        super().__init__()
        if isinstance(tag, list):
            self.tags: list[int] = tag
        else:
            self.tags: list[int] = [tag,]

    @property
    def select(self) -> DomainSelection:
        return DomainSelection(self.tags)
    
class GMSHSurface(GMSHObject):
    '''GMSHVolume is an interface to the GMSH CAD kernel. It does not reprsent Emerge
    specific geometry data.'''
    dim = 2

    @property
    def select(self) -> FaceSelection:
        return FaceSelection(self.tags)
    
    def __init__(self, tag: int | list[int]):
        super().__init__()
        if isinstance(tag, list):
            self.tags: list[int] = tag
        else:
            self.tags: list[int] = [tag,]

class Polygon(GMSHSurface):
    
    def __init__(self,
                 tags: list[int]):
        super().__init__([])
        self.tags: list[int] = tags
