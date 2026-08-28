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
from __future__ import annotations
import numpy as np
from ..mesh3d import Mesh3D
from .femdata import FEMBasis
from emsutil import Saveable
from loguru import logger
from ..compiled import MATHLIB
from .dofsets import DoFSet

############### Nedelec2 Class
USE_NUMBA = False


class DoFSplitException(Exception):
    pass


class Nedelec2(FEMBasis, Saveable):
    def __init__(self, mesh: Mesh3D, dofset: DoFSet):
        super().__init__(mesh)

        self.nedges: int = self.mesh.n_edges
        self.ntris: int = self.mesh.n_tris
        self.ntets: int = self.mesh.n_tets

        self.n_field: int = dofset.set3d.n_edge_dofs * self.nedges + dofset.set3d.n_face_dofs * self.ntris

        ######## MESH Derived
        self.dofset: DoFSet = dofset
        self.dofcodes2d: np.ndaray = np.array(dofset.set2d.codes, dtype=np.int64)
        self.dofcodes3d: np.ndaray = np.array(dofset.set3d.codes, dtype=np.int64)
        
        self.nedof: int = dofset.set3d.n_edge_dofs
        self.nfdof: int = dofset.set3d.n_face_dofs
        self.nvdof: int = 0

        self.nedof_tot: int = self.nedof*self.n_edges
        self.nfdof_tot: int = self.nfdof*self.n_tris

        self.n_tri_dofs: int = 3*self.nedof + self.nfdof
        self.n_tet_dofs: int = 6*self.nedof + 4*self.nfdof
        
        ndof = dofset.set3d.n_dof_tot
        nedges = self.mesh.n_edges
        ntris = self.mesh.n_tris

        self.tet_to_field: np.ndarray = np.zeros(
            (ndof, mesh.n_tets), dtype=int
        )
        self.edge_to_field: np.ndarray = np.zeros((self.nedof, nedges), dtype=int)
        self.tri_to_field: np.ndarray = np.zeros((self.nedof*3 + self.nfdof, ntris), dtype=int)
        
        for i in range(self.nedof):
            self.tet_to_field[6*i:6*(i+1), :] = self.mesh.tet_to_edge + i*nedges
            self.edge_to_field[i,:] = np.arange(nedges) + i*nedges
            self.tri_to_field[3*i:3*(i+1), :] = self.mesh.tri_to_edge + i*nedges 
            
        ned = self.nedof*6
        for i in range(dofset.set3d.n_face_dofs):
            self.tet_to_field[ned+4*i:ned+4*(i+1), :] = self.mesh.tet_to_tri + i*ntris + self.nedof_tot
            self.tri_to_field[self.nedof*3 + i,:] = np.arange(ntris) + i*ntris + self.nedof_tot


        self.tri_to_field_os: np.ndarray | None = None
        self.edge_to_field_os: np.ndarray | None = None

        ##
        self._field: np.ndarray | None = None
        self._all_tet_ids = np.arange(self.ntets)

        ##
        self.edge_to_split: list[int] = []
        self.tri_to_split: list[int] = []

        self._dof_mapping: dict[int, int] = dict()
        self._tet_to_field_orig: np.ndaray = None
        self._partitioned: bool = False
        self.diagnose()

        rows, cols = self.empty_tri_rowcol()
        self._rows = rows
        self._cols = cols
        
    def compute_global_dofcodes(self) -> np.ndarray:
        """Expand `field.dofcodes3d` (fixed-length, one entry per local basis-
        function slot of a single tet, identical for every tet of this element
        order) into a length-`field.n_field` array aligned with the row/column
        order of the assembled E/B matrices.

        A given global DOF is referenced by every tet that shares the mesh
        edge/face it lives on; those references must all agree on its dofcode
        (edge-vs-face + which basis-function definition), since that's a
        property of the abstract basis function, not of any one tet. This is
        checked rather than assumed -- a mismatch means something about the
        element setup violates that assumption and should be looked at, not
        silently averaged over.

        Args:
            field: a Nedelec2 field object (anything exposing `get_tet_to_field()`,
                `dofcodes3d`, and `n_field`).

        Returns:
            np.ndarray, shape (field.n_field,), int64 dofcode per global DOF.
        """
        field = self
        tet_to_field = field.get_tet_to_field()  # (ndof_local, ntets)
        dofcodes3d = np.asarray(field.dofcodes3d)  # (ndof_local,)

        global_codes = np.full(field.n_field, -1, dtype=np.int64)

        for idof in range(dofcodes3d.shape[0]):
            gids = tet_to_field[idof, :]
            code = int(dofcodes3d[idof])

            existing = global_codes[gids]
            conflict = (existing != -1) & (existing != code)
            if np.any(conflict):
                bad = int(gids[conflict][0])
                raise ValueError(
                    f"Inconsistent dofcode for global DOF {bad}: "
                    f"already tagged {global_codes[bad]}, local slot {idof} says {code}. "
                    f"This means a DOF is being reached by basis-function definitions "
                    f"that disagree -- check element-order consistency across the mesh."
                )
            global_codes[gids] = code

        unset = np.flatnonzero(global_codes == -1)
        if unset.size:
            raise ValueError(
                f"{unset.size} of {field.n_field} global DOFs were never referenced "
                f"by any tet (e.g. dof {int(unset[0])}) -- dofcode undefined for them."
            )

        return global_codes

    @staticmethod
    def _parse_dofcode_np(dofcodes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Pure-numpy reimplementation of compiled/ccbf.py:parse_dofcode -- kept
        inline (rather than imported) so this file has no relative-import
        dependency on wherever it ends up living inside EMerge.

        Returns (typearray, indexarray): typearray 0=edge/1=face, indexarray =
        the *local* edge index (0-5) or local face index (0-3) within a tet that
        this local dof-template slot corresponds to.
        """
        dofcodes = np.asarray(dofcodes)
        idofcode = dofcodes & 0b00111111
        is_edge = (dofcodes & 0b11000000) == 64
        typearray = np.where(is_edge, 0, 1)

        indexarray = np.empty_like(dofcodes, dtype=np.int64)
        counters: dict[tuple[int, bool], int] = {}
        for i in range(dofcodes.shape[0]):
            key = (int(idofcode[i]), bool(is_edge[i]))
            indexarray[i] = counters.get(key, 0)
            counters[key] = counters.get(key, 0) + 1
        return typearray, indexarray


    def compute_global_dof_coords(self) -> np.ndarray:
        """Representative world-space coordinate per global DOF: the midpoint of
        its mesh edge (edge-type DOFs) or centroid of its mesh face (face-type
        DOFs). Deliberately doesn't distinguish basis-function definition on the
        same edge/face (e.g. curl-type vs. grad-type both live at the same
        midpoint) -- that distinction is `dof_types`' job, not this one's.

        Returns:
            np.ndarray, shape (3, field.n_field), same axis convention as
            `field.mesh.nodes`.
        """
        field = self
        mesh = field.mesh
        nodes = mesh.nodes            # (3, n_nodes)
        edges = mesh.edges            # (2, n_edges) -- global node ids per edge
        tris = mesh.tris              # (3, n_tris)  -- global node ids per face
        tet_to_edge = mesh.tet_to_edge  # (6, ntets)
        tet_to_tri = mesh.tet_to_tri    # (4, ntets)
        tet_to_field = field.get_tet_to_field()  # (ndof_local, ntets)
        dofcodes3d = np.asarray(field.dofcodes3d)

        typearray, indexarray = self._parse_dofcode_np(dofcodes3d)

        edge_coords = (nodes[:, edges[0, :]] + nodes[:, edges[1, :]]) / 2.0
        tri_coords = (nodes[:, tris[0, :]] + nodes[:, tris[1, :]] + nodes[:, tris[2, :]]) / 3.0

        dof_coords = np.full((3, field.n_field), np.nan)

        for idof in range(dofcodes3d.shape[0]):
            gids = tet_to_field[idof, :]
            if typearray[idof] == 0:
                eids = tet_to_edge[indexarray[idof], :]
                dof_coords[:, gids] = edge_coords[:, eids]
            else:
                tids = tet_to_tri[indexarray[idof], :]
                dof_coords[:, gids] = tri_coords[:, tids]

        unset = np.flatnonzero(np.isnan(dof_coords[0, :]))
        if unset.size:
            raise ValueError(
                f"{unset.size} of {field.n_field} global DOFs got no coordinate "
                f"(e.g. dof {int(unset[0])})."
            )

        return dof_coords


    def diagnose(self):
        visited_field = np.zeros((self.n_field,), dtype=np.bool)
        for i in range(self.mesh.n_tets):
            visited_field[self.tet_to_field[:, i]] = True
        assert np.all(visited_field)

    def get_tet_to_field(self) -> np.ndarray:
        if self._tet_to_field_orig is not None:
            return self._tet_to_field_orig
        return self.tet_to_field

    def partition_dof(self, list_tags: list[list[int]]) -> None:
        """Partition_DOF splits the degrees of freedom for an input list of face tags."""

        # If no tri_to_field mapping for side 2 is managed, do it now
        if self._partitioned:
            return

        if self.tri_to_field_os is None:
            self.tri_to_field_os = self.tri_to_field.copy()
            self.edge_to_field_os = self.edge_to_field.copy()
            self._tet_to_field_orig = self.tet_to_field.copy()

        tags: list[int] = []
        for sublist in list_tags:
            tags.extend(sublist)

        split_tets = []
        split_edges = []
        split_tris = []

        for tag in tags:
            stets, stris, sedges = self._partition_dof_face(tag)
            split_tets.extend(stets)
            split_edges.extend(sedges)
            split_tris.extend(stris)

        split_edges = sorted(list(set(split_edges)))
        split_tris = sorted(list(set(split_tris)))

        NF = self.n_field
        NE_SPLIT = len(split_edges)
        NF_SPLIT = len(split_tris)

        dofset = self.dofset.set3d

        for iedof in range(self.nedof):
            self._dof_mapping.update({split_edges[i] + (iedof)*self.nedges: NF + i + iedof*NE_SPLIT for i in range(NE_SPLIT)})
        for ifdof in range(self.nfdof):
            self._dof_mapping.update({split_tris[i] + self.nedof_tot + ifdof*self.ntris: NF + i + self.nedof * NE_SPLIT + NF_SPLIT*ifdof for i in range(NF_SPLIT)})

        self.n_field = self.n_field + self.nedof * NE_SPLIT + dofset.n_face_dofs*NF_SPLIT

        for tet in split_tets:
            for i, idof in enumerate(self.tet_to_field[:, tet]):
                self.tet_to_field[i, tet] = self._dof_mapping.get(idof, idof)

        for itri in split_tris:
            for i, idof in enumerate(self.tri_to_field[:, itri]):
                self.tri_to_field_os[i, itri] = self._dof_mapping.get(idof, idof)

        for iedge in split_edges:
            
            for i in range(self.nedof):
                idof = self.edge_to_field[i, iedge]
                self.edge_to_field_os[i, iedge] = self._dof_mapping.get(idof,idof)


        self._partitioned = True

    def _partition_dof_face(self, tag: int) -> None:
        """Splits DOFs on a specific face tag between two volumes.

        Splitting means that one domain gets one set of degrees of freedom
        and the other a copied set of degrees of freedom.

        the mapping between them is stored in self._dof_mapping

        """
        split_tets = []
        split_edges = []
        split_tris = []

        tris = self.mesh.get_triangles([tag])
        linked_tets = self.mesh.tri_to_tet[:, tris]

        # Check exterior
        if np.any(linked_tets == self.mesh._MISSING_ID):
            raise DoFSplitException(
                f"Cannot split DoF for boundary {tag} because its on the exterior "
                f"of the simulation domain. Check the ThermalContact boundary assignment."
            )

        # Check same-domain
        # Build tet -> vol mapping as array (faster than dict)
        tet_to_volume = np.full(self.ntets, -1, dtype=np.int64)
        for vtag, tet_ids in self.mesh.vtag_to_tet.items():
            tet_to_volume[tet_ids] = vtag

        vol_a_of_tri = tet_to_volume[linked_tets[0, :]]
        vol_b_of_tri = tet_to_volume[linked_tets[1, :]]

        connected_volumes = set(vol_a_of_tri) | set(vol_b_of_tri)
        connected_volumes = sorted(list(connected_volumes))

        # Boundary DOFs
        boundary_edges = self.mesh.get_edges(tag)

        # figure out which edges are counted only ones
        edge_counter = {ie: 0 for ie in boundary_edges}

        # base normal:
        normal = self.mesh.ftag_to_normal[tag]

        for itri in tris:
            e1, e2, e3 = self.mesh.tri_to_edge[:, itri]
            edge_counter[e1] += 1
            edge_counter[e2] += 1
            edge_counter[e3] += 1

        for itri in tris:
            tet1 = self.mesh.tri_to_tet[0, itri]
            tet2 = self.mesh.tri_to_tet[1, itri]

            tri_c = self.mesh.tri_centers[:, itri]
            tet_c = self.mesh.centers[:, tet2]
            vec = tet_c - tri_c
            vec = vec / (np.linalg.norm(vec) + 1e-13)
            trinorm = self.mesh.get_normal(itri)
            normdot = np.sum(trinorm * normal)

            e1, e2, e3 = self.mesh.tri_to_edge[:, itri]
            for ie in (e1, e2, e3):
                if edge_counter[ie] == 2 and ie not in split_edges:
                    split_edges.append(ie)
            split_tris.append(itri)

            pick = tet1
            if normdot > 0:
                pick = tet2

            split_tets.append(pick)

        return split_tets, split_tris, split_edges

    def interpolate(
        self,
        field: np.ndarray,
        xs: np.ndarray,
        ys: np.ndarray,
        zs: np.ndarray,
        tet_mapping: np.ndarray,
        tetids: np.ndarray | None = None,
        usenan: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Interpolate the provided field data array at the given xs, ys and zs coordinates
        """

        tetids = self._all_tet_ids
        vals = MATHLIB.ned2_tet_interp(
            np.array([xs, ys, zs]),
            field,
            self.mesh.tets,
            self.mesh.tris,
            self.mesh.edges,
            self.mesh.nodes,
            self.tet_to_field,
            self.mesh.tet_to_edge,
            self.mesh.tet_to_tri,
            tetids,
            tet_mapping,
            self.dofcodes3d
        )
        n_zeros = np.isnan(vals).sum()
        if not usenan and n_zeros > 0:
            logger.debug(f"Converted {n_zeros} to zeros.")
            vals = np.nan_to_num(vals)
        return vals

    def interpolate_curl(
        self,
        field: np.ndarray,
        xs: np.ndarray,
        ys: np.ndarray,
        zs: np.ndarray,
        c: np.ndarray,
        tet_mapping: np.ndarray,
        tetids: np.ndarray | None = None,
        usenan: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Interpolates the curl of the field at the given points.
        """
        tetids = self._all_tet_ids
        vals = vals = MATHLIB.ned2_tet_interp_curl(
            np.array([xs, ys, zs]),
            field,
            self.mesh.tets,
            self.mesh.tris,
            self.mesh.edges,
            self.mesh.nodes,
            self.tet_to_field,
            self.mesh.tet_to_edge,
            self.mesh.tet_to_tri,
            c,
            tetids,
            tet_mapping,
            self.dofcodes3d
        )
        n_zeros = np.isnan(vals).sum()
        if not usenan and n_zeros > 0:
            logger.debug(f"Converted {n_zeros} to zeros.")
            vals = np.nan_to_num(vals)
        return vals

    def interpolate_index(
        self,
        xs: np.ndarray,
        ys: np.ndarray,
        zs: np.ndarray,
        tetids: np.ndarray | None = None,
        usenan: bool = True,
    ) -> np.ndarray:
        if tetids is None:
            tetids = self._all_tet_ids

        vals = MATHLIB.index_interp(
            np.array([xs, ys, zs]), self.mesh.tets, self.mesh.nodes, tetids
        )

        n_zeros = np.isnan(vals).sum()

        if not usenan and n_zeros > 0:
            logger.debug(f"Converted {n_zeros} to zeros.")
            vals[vals == -1] == 0

        return vals
