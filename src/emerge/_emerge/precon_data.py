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
basis function it is, plus the node coordinates. No tets or materials.

Thin conductors split the DoFs on their sheet: the tets on one side of the sheet use a copied set
of DoFs on the sheet's interior edges and triangles. That is described as a cut through the mesh:
every interior sheet vertex gets a copy (same coordinates), the copied DoFs live on copies of the
sheet edges and triangles that run between copied vertices, and the edges and faces that touch the
sheet on the copied side are attached to the copied vertices. Every DoF then belongs to one edge or
triangle of the cut mesh again.
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
            PreconData | None: The data, or None if the DoFs of the field cannot be described this way.
        """
        mesh = field.mesh
        nedges, ntris = field.nedges, field.ntris
        edges = np.asarray(mesh.edges)
        tris = np.asarray(mesh.tris)
        nodes = np.asarray(mesh.nodes)
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

        if getattr(field, "_partitioned", False):
            cut = _cut_along_sheets(field, edges, tris, nodes, dof_entity, dof_kind)
            if cut is None:
                return None
            edges, tris, nodes = cut

        if np.any(dof_kind < 0):
            logger.debug("No preconditioner data: some DoFs are not tied to an edge or triangle.")
            return None

        return cls(edges, tris, dof_entity, dof_kind, nodes)


def _cut_along_sheets(field, edges, tris, nodes, dof_entity, dof_kind):
    """The cut mesh of a field with thin conductor DoF splits (see the module docstring).

    Fills dof_entity / dof_kind of the copied DoFs in place.

    Returns:
        tuple | None: The (2, E') edges, (3, F') triangles and (3, V') nodes of the cut mesh, or None
            if the split cannot be described as a cut.
    """
    mesh = field.mesh
    nedof, nfdof = field.nedof, field.nfdof
    nv, ne, nf = nodes.shape[1], edges.shape[1], tris.shape[1]
    e2f, e2f_os = field.edge_to_field, field.edge_to_field_os
    t2f, t2f_os = field.tri_to_field, field.tri_to_field_os
    if e2f_os is None or t2f_os is None:
        logger.debug("No preconditioner data: split field without copied DoF maps.")
        return None

    split_e = np.flatnonzero(e2f_os[0, :] != e2f[0, :])            # interior sheet edges
    split_t = np.flatnonzero(t2f_os[3 * nedof, :] != t2f[3 * nedof, :])  # sheet triangles
    if split_t.size == 0:
        return edges, tris, nodes

    # Interior sheet vertices: on a split edge and not on the sheet's rim (sheet edges that are not split)
    sheet_e = np.unique(np.asarray(mesh.tri_to_edge)[:, split_t])
    rim_e = np.setdiff1d(sheet_e, split_e)
    dup_v = np.setdiff1d(np.unique(edges[:, split_e]), np.unique(edges[:, rim_e]))
    vmap = np.arange(nv)
    vmap[dup_v] = nv + np.arange(dup_v.size)

    # Copied sheet edges and triangles run between copied vertices
    new_e = vmap[edges[:, split_e]]
    new_t = vmap[tris[:, split_t]]
    for i in range(nedof):
        ids = e2f_os[i, split_e]
        dof_entity[ids] = ne + np.arange(split_e.size)
        dof_kind[ids] = EDGE_WHITNEY if i == 0 else EDGE_OTHER
    for i in range(nfdof):
        ids = t2f_os[3 * nedof + i, split_t]
        dof_entity[ids] = nf + np.arange(split_t.size)
        dof_kind[ids] = FACE

    # Edges and faces touching an interior sheet vertex on the copied side move to the copied vertex.
    # Around every such vertex the sheet cuts its tets into two face-connected halves; the half that
    # holds a tet with copied DoFs is the copied side.
    n_orig = field.n_field - nedof * split_e.size - nfdof * split_t.size
    tet_to_field = np.asarray(field.tet_to_field)
    copied_tet = np.any(tet_to_field >= n_orig, axis=0)
    tets = np.asarray(mesh.tets)
    t2e, t2t = np.asarray(mesh.tet_to_edge), np.asarray(mesh.tet_to_tri)
    sheet = np.zeros(nf, dtype=bool)
    sheet[split_t] = True
    is_split_e = np.zeros(ne, dtype=bool)
    is_split_e[split_e] = True
    is_dup = np.zeros(nv, dtype=bool)
    is_dup[dup_v] = True

    # vertex -> tets, only for the vertices that get a copy
    rows, cols = np.nonzero(is_dup[tets])
    vt = tets[rows, cols]
    order = np.argsort(vt, kind="stable")
    vt, vtet = vt[order], cols[order]
    bounds = np.searchsorted(vt, dup_v, side="left"), np.searchsorted(vt, dup_v, side="right")

    move_e: list[tuple[int, int]] = []
    move_t: list[tuple[int, int]] = []
    ambiguous = 0
    for k, v in enumerate(dup_v.tolist()):
        ring = vtet[bounds[0][k]:bounds[1][k]]
        if ring.size == 0:
            continue
        # union-find over the ring through shared non-sheet faces that contain v
        parent = {int(t): int(t) for t in ring}

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        face_owner: dict[int, int] = {}
        for t in ring.tolist():
            for itri in t2t[:, t].tolist():
                if sheet[itri] or v not in tris[:, itri]:
                    continue
                if itri in face_owner:
                    ra, rb = find(face_owner[itri]), find(t)
                    if ra != rb:
                        parent[ra] = rb
                else:
                    face_owner[itri] = t
        copied_roots = {find(int(t)) for t in ring.tolist() if copied_tet[t]}
        if not copied_roots:
            continue
        if len({find(int(t)) for t in ring.tolist()} - copied_roots) == 0:
            ambiguous += 1       # the sheet does not separate the ring: leave this vertex shared
            continue
        for t in ring.tolist():
            if find(t) not in copied_roots:
                continue
            for ie in t2e[:, t].tolist():
                if not is_split_e[ie] and v in edges[:, ie]:
                    move_e.append((ie, v))
            for itri in t2t[:, t].tolist():
                if not sheet[itri] and v in tris[:, itri]:
                    move_t.append((itri, v))

    edges = edges.copy()
    tris = tris.copy()
    for ie, v in set(move_e):
        edges[edges[:, ie] == v, ie] = vmap[v]
    for itri, v in set(move_t):
        tris[tris[:, itri] == v, itri] = vmap[v]
    if ambiguous:
        logger.debug(f"Thin conductor cut: {ambiguous} sheet vertices are not separated by the sheet.")
    logger.debug(
        f"Thin conductor cut: {dup_v.size} copied vertices, {split_e.size} copied edges, "
        f"{split_t.size} copied triangles"
    )
    return (np.concatenate([edges, new_e], axis=1), np.concatenate([tris, new_t], axis=1),
            np.concatenate([nodes, nodes[:, dup_v]], axis=1))
