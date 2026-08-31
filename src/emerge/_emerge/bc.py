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
from enum import Enum
from loguru import logger
from .selection import Selection, FaceSelection
import numpy as np
from .geometry import GeoObject
from typing import TypeVar, Type
from emsutil import Saveable

T = TypeVar('T')

class BoundaryConditionError(Exception):
    pass

def _unique(input: list[int]) -> list:
    """ Returns a sorted list of all unique integers/floats in a list."""
    return sorted(list(set(input)))

############################################################
#               BASE BOUNDARY CONDITION CLASS              #
############################################################

class BoundaryCondition(Saveable):
    """A generalized class for all boundary condition objects.
    """
    _color: str = "#a54141"
    _name: str = "UnnamedBC"
    _texture: str = "None"
    dim: int = -1
    _is_exclusive: bool = True
    
    def __init__(self, assignment: GeoObject | Selection):
        
        if assignment.dim != self.dim and self.dim != -1:
            raise ValueError(f'Boundary condition of type {type(self).__name__} requires a selection of dimension {self.dim}, but got dimension {assignment.dim} instead.')
        self.indices: list[int] = []
        self.face_indices: list[int] = []
        self.edge_indices: list[int] = []
        
        if isinstance(assignment, GeoObject):
            assignment = assignment.selection
        
        self.selection: Selection = assignment
        self.tags: list[int] = self.selection.tags
        
        
    @property
    def _size_constraint(self) -> float | None:
        return None
    
    def valid_after_fragment(self) -> bool:
        return self.selection._valid_after_fragment
    
    def __repr__(self) -> str:
        return f'{type(self).__name__}{self.tags}'

    def __str__(self) -> str:
        return self.__repr__()
    
    def add_tags(self, dimtags: list[tuple[int,int]]) -> None:
        """Adds the given taggs to this boundary condition.

        Args:
            tags (list[tuple[int,int]]): The tags to include
        """
        tags = [x[1] for x in dimtags]
        self.tags = _unique(self.tags + tags)
    
    def remove_tags(self, tags: list[int]) -> list[int]:
        """Removes the tags provided by tags from this boundary condition.

        Return sonly the tags that are actually excluded from this face.

        Args:
            tags (list[int]): The tags to exclude.

        Returns:
            list[int]: A list of actually excluded tags.
        """
        excluded_edges = [x for x in self.tags if x in tags]
        self.tags = [x for x in self.tags if x not in tags]
        self.selection.remove_tags(tags)
        return excluded_edges
    
    def exclude_bc(self, other: BoundaryCondition) -> list[int]:
        """Excludes all faces for a provided boundary condition object from this boundary condition assignment.

        Args:
            other (BoundaryCondition): The boundary condition of which the faces should be excluded

        Returns:
            list[int]: A list of excluded face tags.
        """
        return self.remove_tags(other.tags)

class BoundaryConditionSet(Saveable):

    def __init__(self):

        self.boundary_conditions: list[BoundaryCondition] = []
        self._initialized_with_defaults: bool = False
        self._overwrite: bool = True
        
    def cleanup(self) -> None:
        """ Removes non assigned boundary conditions"""
        logger.trace("Cleaning up boundary conditions.")
        toremove = [bc for bc in self.boundary_conditions if len(bc.tags)==0]
        logger.trace(f"Removing: {toremove}")
        self.boundary_conditions = [bc for bc in self.boundary_conditions if len(bc.tags)>0]

    # Checks
    def _selections_post_boolean_fragment(self) -> None:
        """ Perform BC Validation""" 

        for bc in self.boundary_conditions:
            
            if not bc.valid_after_fragment():
                raise BoundaryConditionError(f'Boundary condition {bc} is based on a selection that is made before commit_geometry().')
            logger.debug(f'   BC {bc} Ok!')

    def _construct_bc(self, constructor: type) -> type:
        """ A helper function to construct boundary condition objects and assign them to this set.

        Args:
            constructor (type): The boundary condition constructor.

        Returns:
            type: The constructed boundary condition.
        """
        def constr(*args, **kwargs):
            obj = constructor(*args, **kwargs)
            self.assign(obj)
            return obj
        return constr # type: ignore
    
    def assigned(self, dim: int = 2) -> list[int]:
        """Returns all boundary tags that have a boundary condition assigned to them

        Args:
            dim (int, optional): The dimension. Defaults to 2.

        Returns:
            list[int]: The list of tags
        """
        tags = []
        for bc in self.boundary_conditions:
            if bc.dim != dim:
                continue
            tags.extend(bc.tags)
        return tags

    def count(self, bctype: type) -> int:
        """Returns the number of a certain boundary condition type

        Args:
            bctype (type): The boundary condtion type

        Returns:
            int: The number of occurances
        """
        return len(self.oftype(bctype))
    
    def oftype(self, bctype: Type[T]) -> list[T]:
        """Returns a list of all boundary conditions of a certain type.

        Args:
            bctype (type): The boundar condition type

        Returns:
            list[BoundaryCondition]: The list of boundary conditions
        """
        return [item for item in self.boundary_conditions if isinstance(item, bctype)]

    def reset(self) -> None:
        """Resets the boundary conditions that are defined
        """
        self.boundary_conditions = []

    def no_overwrite(self: T) -> T:
        """Turns overwrite of for the next boundary condition assignment

        Returns:
            BoundaryConditionSet: _description_

        """

        self._overwrite = False

        return self

    def assign(self, bc: BoundaryCondition) -> None:
        """Assign a boundary-condition object to a domain or list of domains.
        This method must be called to submit any boundary condition object you made to the physics.

        Args:
            bcs *(BoundaryCondition): A list of boundary condition objects.
        """
        if bc in self.boundary_conditions:
            return
        
        self._initialized_with_defaults = True

        bc.add_tags(bc.selection.dimtags)

        if bc._is_exclusive:
            for existing_bc in self.boundary_conditions:
                if existing_bc.dim != bc.dim:
                    continue
                
                if self._overwrite:
                    excluded = existing_bc.exclude_bc(bc)
                    
                    if excluded:
                        logger.debug(f'Removed the {excluded} tags from object with dimension {bc.dim} BC {existing_bc}')
                else:
                    bc.exclude_bc(existing_bc)
        dname = {
            0: 'point',
            1: 'edge',
            2: 'surface',
            3: 'domain',
        }
        logger.info(f'Adding {type(bc).__name__} to {dname[bc.dim]} tags {bc.tags}(dim = {bc.dim})')
        self._overwrite = True
        self.boundary_conditions.append(bc)

class Periodic(BoundaryCondition, Saveable):

    _color: str = "#5d4fda"
    _name: str = "PeriodicBC"
    _texture: str = "tex6.png"
    dim: int = 2
    def __init__(self, 
                 selection1: Selection,
                 selection2: Selection,
                 dv: tuple[float,float,float],
                 ):
        self.face1: BoundaryCondition = BoundaryCondition(selection1)
        self.face2: BoundaryCondition = BoundaryCondition(selection2)
        super().__init__(FaceSelection(selection1.tags + selection2.tags))
        self.dv: tuple[float,float,float] = dv
        self.ux: float = 0
        self.uy: float = 0
        self.uz: float = 0

    def phi(self, k0) -> complex:
        dx, dy, dz = self.dv
        return np.exp(-1j*k0*(self.ux*dx+self.uy*dy+self.uz*dz))
    