# [Written by Claude] All code in this module was written by Claude (AI).
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

"""Mesh information that preconditioned iterative solvers (CORVID) need next to A and b.

The matrix and right hand side are already handed to every solver. A mesh-aware preconditioner
additionally needs to know, per degree of freedom, which mesh entity it lives on and what kind of
basis function it is. That is all this module provides: no coordinates, tets or materials.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from loguru import logger

# Degree of freedom kinds, as used by corvid.Topology
EDGE_WHITNEY = 0  # the lowest order (Whitney) edge DoF
EDGE_OTHER = 1  # any other edge DoF
FACE = 2  # a face DoF


@dataclass
class PreconData:
    """The mesh topology of a Nedelec field, per global degree of freedom.

    Attributes:
        edges (np.ndarray): The (2, E) node indices of the mesh edges.
        tris (np.ndarray): The (3, F) node indices of the mesh triangles.
        dof_entity (np.ndarray): The (n_field,) edge index (edge DoFs) or triangle index (face DoFs) of
            every global DoF.
        dof_kind (np.ndarray): The (n_field,) kind of every global DoF (EDGE_WHITNEY, EDGE_OTHER or FACE).
        nodes (np.ndarray): The (3, V) node coordinates, used to locate coarsely meshed metal walls.
    """

    edges: np.ndarray
    tris: np.ndarray
    dof_entity: np.ndarray
    dof_kind: np.ndarray
    nodes: np.ndarray

    def wall_vertices(self, solve_ids: np.ndarray) -> np.ndarray:
        """The mesh vertices of the eliminated (not solved) DoFs, which are the metal (PEC) walls.

        Args:
            solve_ids (np.ndarray): The solved global DoFs

        Returns:
            np.ndarray: A boolean mask over the vertices used by the edges and triangles
        """
        eliminated = np.ones(self.dof_entity.shape[0], dtype=bool)
        eliminated[solve_ids] = False
        entity, kind = self.dof_entity[eliminated], self.dof_kind[eliminated]
        n_vertices = int(max(self.edges.max(initial=-1), self.tris.max(initial=-1))) + 1
        wall = np.zeros(n_vertices, dtype=bool)
        wall[self.edges[:, entity[kind != FACE]].ravel()] = True
        wall[self.tris[:, entity[kind == FACE]].ravel()] = True
        return wall

    @classmethod
    def from_field(cls, field) -> PreconData | None:
        """Builds the topology data of a Nedelec field.

        Args:
            field (Nedelec2): The field.

        Returns:
            PreconData | None: The data, or None if the DoFs of the field cannot be described this way
                (thin conductor DoF splits add DoFs that belong to no single mesh entity).
        """
        if getattr(field, "_partitioned", False):
            logger.debug("No preconditioner data: the field has split (thin conductor) DoFs.")
            return None

        nedges, ntris = field.nedges, field.ntris
        dof_entity = np.full(field.n_field, -1, dtype=np.int64)
        dof_kind = np.full(field.n_field, -1, dtype=np.int8)

        # The first edge DoF of every element space is the Whitney function (dofcode 64)
        for i in range(field.nedof):
            ids = field.edge_to_field[i, :]
            dof_entity[ids] = np.arange(nedges)
            dof_kind[ids] = EDGE_WHITNEY if i == 0 else EDGE_OTHER

        for i in range(field.nfdof):
            ids = field.tri_to_field[3 * field.nedof + i, :]
            dof_entity[ids] = np.arange(ntris)
            dof_kind[ids] = FACE

        if np.any(dof_kind < 0):
            logger.debug("No preconditioner data: some DoFs are not tied to an edge or triangle.")
            return None

        return cls(field.mesh.edges, field.mesh.tris, dof_entity, dof_kind, field.mesh.nodes)
