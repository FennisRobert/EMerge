
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

from functools import reduce
from enum import Enum
from emsutil import Saveable

def merge_lists(lists):
    if not lists:
        return []
    return reduce(lambda a,b: a+b, lists)

class _DoFSet(Saveable):

    def __init__(self, edge_dof_ids: list[int], face_dof_ids: list[int], dim: int):
        if dim == 2:
            ne = 3
            nf = 1
        elif dim == 3:
            ne = 6
            nf = 4
        self.codes = merge_lists([[64+idof,]*ne for idof in edge_dof_ids]) + merge_lists([[128+idof,]*nf for idof in face_dof_ids])
        self.n_edge_dofs: int = len(edge_dof_ids)
        self.n_face_dofs: int = len(face_dof_ids)
        self.n_edge_dofs_tot: int = len(edge_dof_ids)*ne
        self.n_face_dofs_tot: int = len(face_dof_ids)*nf
        self.n_dof_tot: int = self.n_edge_dofs_tot + self.n_face_dofs_tot
        self.n_node_dofs = 0
        self.n_vol_dofs = 0

    def __str__(self):
        line = ''
        for key,value in self.__dict__.items():
            line = line + f',{key} = {value}'
        return line
    
class DoFSet(Saveable):

    def __init__(self, edge_dof_ids: list[int], face_dof_ids: list[int], name: str = "UnnamedSet"):
        """The DoFSet class manages the definition of the vector finite element bases.
        It currently only supports Nedelec elements.

        Degree-of-freedom functions used in EMerge can currently only be specified as edge
        or face degrees of freedom (up to second order complete).

        The degrees of freedom for edges and faces should be provided as a list of integers
        corresponding to the desired basis functions to use.
        
        The following DoF Defintions are available:
        All Edge basis functions
         - 0 (curl) = Li ∇ Lj - Lj ∇ Li
         - 1 (grad) = ∇ (LiLj)
         - 2 (curl) = (Li - Lj) (Li∇Lj - Lj∇Li)
         - 3 (grad) = ∇(LiLj(Li-Lj))
         - 4 (curl) = Li (Li ∇ Lj - Lj ∇ Li)
         - 5 (curl) = Lj (Li ∇ Lj - Lj ∇ Li)

        All Face basis functions
         - 0 (curl) = Li(Lj ∇Lk - Lk ∇Lj)
         - 1 (curl) = Lj(Li ∇Lk - Lk ∇Li)
         - 2 (curl) = Lk(Lj ∇Li - Li ∇Lj)
         - 3 (curl) = LjLk∇Li + LiLk∇Lj - 2LiLj∇Lk
         - 4 (curl) = LkLi∇Lj + LjLi∇Lk - 2LjLk∇Li
         - 5 (curl) = LiLj∇Lk + LkLj∇Li - 2LkLi∇Lj
         - 6 (grad) = ∇(LiLjLk)

        Args:
            edge_dof_ids (list[int]): A list of the desired edge DoF
            face_dof_ids (list[int]): A list of the desired face DoF
            name (str, optional): The name of the basis function set. Defaults to "UnnamedSet".
        """
        
        self.set2d: _DoFSet = _DoFSet(edge_dof_ids, face_dof_ids, 2)
        self.set3d: _DoFSet = _DoFSet(edge_dof_ids, face_dof_ids, 3)
        self.name: str = name

    def __str__(self):
        return self.name

    def unmatched_gradients(self) -> tuple[list[int], list[int]]:
        """The positions of the cubic gradient DoFs, edge DoF 3 ∇(LiLj(Li-Lj)) and face DoF 6 ∇(LiLjLk), among the
        edge and face DoFs of this set.

        The boundary mode analysis has no cubic Lagrange partner for them, so on port faces they are pinned to zero,
        both in the mode analysis and in the 3D system. Without that, the 3D field carries a gradient part on the port
        face that is not in the mode, which the port absorbs but the S-parameters do not count (power is lost).

        Returns:
            tuple[list[int], list[int]]: The edge DoF positions and the face DoF positions.
        """
        s = self.set3d
        edge = [i for i, code in enumerate(s.codes[:s.n_edge_dofs_tot:6]) if code == 64 + 3]
        face = [i for i, code in enumerate(s.codes[s.n_edge_dofs_tot::4]) if code == 128 + 6]
        return edge, face

class ElementSpace(Enum):
    FIRST_ORDER_MIXED = 0
    FIRST_ORDER_COMPLETE = 1
    SECOND_MIXED_SAVAGE = 2
    SECOND_MIXED_VOLAKIS = 3
    SECOND_MIXED_WEBB = 4
    SECOND_COMPLETE_WEBB = 5
    SECOND_COMPLETE_VOLAKIS = 6

    def get_set(self) -> DoFSet:
        match self:
            case ElementSpace.FIRST_ORDER_MIXED:
                return DoFSet([0], [], '1st Order Mixed')
            case ElementSpace.FIRST_ORDER_COMPLETE:
                return DoFSet([0, 1], [], '1st Order Complete')
            case ElementSpace.SECOND_MIXED_SAVAGE:
                return DoFSet([0, 1], [0, 1], '2nd Order Mixed (Savage)')
            case ElementSpace.SECOND_MIXED_VOLAKIS:
                return DoFSet([0, 2], [0, 1], '2nd Order Mixed (Volakis)')
            case ElementSpace.SECOND_MIXED_WEBB:
                return DoFSet([0, 1], [3, 4], '2nd Order Mixed (Webb)')
            case ElementSpace.SECOND_COMPLETE_WEBB:
                return DoFSet([0, 1, 3], [3, 4, 6], '2nd Order Complete (Webb)')
            case ElementSpace.SECOND_COMPLETE_VOLAKIS:
                return DoFSet([0, 2, 3], [0, 1, 6], '2nd Order Complete (Volakis)')
            case _:
                raise ValueError(f'No DoFSet defined for {self!r}')