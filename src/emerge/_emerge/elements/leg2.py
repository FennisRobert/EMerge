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

# Last Cleanup: 2025-05-29
from __future__ import annotations
import numpy as np
from ..mesh3d import Mesh3D
from .femdata import FEMBasis
from emsutil import Saveable
from loguru import logger
from ..compiled import MATHLIB

############### Legrange 2 Class


class DoFSplitException(Exception):
    pass


class Legrange2(FEMBasis, Saveable):
    """Implementation of the Legrange order 2 basis functions.

    Basis functions are numbered with the node DoF first and then the edge DoF

    There are 4 node DoF per tetrahedron and 6 edge DoF.

    The total system is also assembled in order: all nodes first then al edges

    Args:
        FEMBasis (_type_): _description_
        Saveable (_type_): _description_
    """

    def __init__(self, mesh: Mesh3D):
        super().__init__(mesh)

        self.nnodes: int = self.mesh.n_nodes
        self.nedges: int = self.mesh.n_edges
        self.ntris: int = self.mesh.n_tris
        self.ntets: int = self.mesh.n_tets

        self.n_field: int = self.nedges + self.nnodes

        ######## MESH Derived
        ntris = self.mesh.n_tris
        nedges = self.mesh.n_edges
        nnodes = self.mesh.n_nodes

        self.tet_to_field: np.ndarray = np.zeros(
            (10, self.mesh.tets.shape[1]), dtype=np.int64
        )
        self.tet_to_field[:4, :] = self.mesh.tets
        self.tet_to_field[4:, :] = self.mesh.tet_to_edge + nnodes

        self.edge_to_field: np.ndarray = np.zeros((nedges,), dtype=np.int64)

        self.node_to_field = np.arange(nnodes)
        self.edge_to_field = np.arange(nedges) + nnodes

        self.tri_to_field: np.ndarray = np.zeros((6, ntris), dtype=np.int64)

        self.tri_to_field[:3, :] = self.mesh.tris
        self.tri_to_field[3:, :] = self.mesh.tri_to_edge + nnodes

        ##
        self._field: np.ndarray | None = None
        self.n_tet_dofs = 10
        self.n_tri_dofs = 6
        self._all_tet_ids = np.arange(self.ntets)

        self._dof_mapping: dict[int, int] = dict()

        self.diagnose()

        self.empty_tri_rowcol()

    def partition_dof(self, list_tags: list[list[int]]) -> None:
        """Partition_DOF splits the degrees of freedom for an input list of face tags."""
        tags: list[int] = []
        for sublist in list_tags:
            tags.extend(sublist)
        for tag in tags:
            self._partition_dof_face(tag)

    def _partition_dof_face(self, tag: int) -> None:
        """Splits DOFs on a specific face tag between two volumes.

        Splitting means that one domain gets one set of degrees of freedom
        and the other a copied set of degrees of freedom.

        the mapping between them is stored in self._dof_mapping

        """

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
        tet_vol = np.full(self.ntets, -1, dtype=np.int64)
        for vtag, tet_ids in self.mesh.vtag_to_tet.items():
            tet_vol[tet_ids] = vtag

        vol_a = tet_vol[linked_tets[0, :]]
        vol_b = tet_vol[linked_tets[1, :]]

        if np.any(vol_a == vol_b):
            raise DoFSplitException(
                f"Cannot split DoF on boundary {tag} because it is joining the same domain. "
                f"Only boundaries between two separate domains can be split. "
                f"Check the ThermalContact boundary assignment."
            )

        connected_volumes = set(vol_a) | set(vol_b)
        assert len(connected_volumes) == 2
        vol_tag_1, vol_tag_2 = sorted(connected_volumes)

        # Boundary DOFs
        boundary_nodes = self.mesh.get_nodes(tag)
        boundary_edges = self.mesh.get_edges(tag)

        boundary_dofs = np.unique(
            np.concatenate(
                [self.node_to_field[boundary_nodes], self.edge_to_field[boundary_edges]]
            )
        )

        # Build mapping: old DOF -> new DOF
        n_new = len(boundary_dofs)
        new_start = self.n_field
        new_dofs = np.arange(new_start, new_start + n_new, dtype=np.int64)
        self.n_field += n_new

        # Sparse remap array: remap[old_dof] = new_dof, default = old_dof
        remap = np.arange(new_start + n_new, dtype=np.int64)
        remap[boundary_dofs] = new_dofs

        # Find tets in vol_tag_2 that touch the boundary
        surf_nodes = np.zeros(self.mesh.n_nodes, dtype=np.bool_)
        surf_nodes[boundary_nodes] = True

        # Vectorized: check which tets have any vertex on the boundary
        tet_touches = np.any(surf_nodes[self.mesh.tets], axis=0)  # (n_tets,)
        tet_is_vol2 = tet_vol == vol_tag_2

        mask = tet_touches & tet_is_vol2
        affected_tets = np.where(mask)[0]

        # Remap tet_to_field for affected tets
        block = self.tet_to_field[:, affected_tets]
        self.tet_to_field[:, affected_tets] = remap[block]

        # Store mapping as dict for downstream use
        dof_mapping = {int(old): int(new) for old, new in zip(boundary_dofs, new_dofs)}
        self._dof_mapping.update(dof_mapping)

    def diagnose(self):
        visited_field = np.zeros((self.n_field,), dtype=np.bool_)
        visited_field[self.tet_to_field] = True
        assert np.all(visited_field)

    def interpolate(
        self,
        field: np.ndarray,
        xs: np.ndarray,
        ys: np.ndarray,
        zs: np.ndarray,
        tet_mapping: np.ndarray,
        tetids: np.ndarray | None = None,
        usenan: bool = True,
    ) -> np.ndarray:
        """
        Interpolate the provided field data array at the given xs, ys and zs coordinates
        """

        tetids = self._all_tet_ids
        vals = MATHLIB.leg2_tet_interp(
            np.array([xs, ys, zs]),
            field,
            self.mesh.tets,
            self.mesh.edges,
            self.mesh.nodes,
            self.tet_to_field,
            tetids,
            tet_mapping,
        )
        n_zeros = np.isnan(vals).sum()
        if not usenan and n_zeros > 0:
            logger.debug(f"Converted {n_zeros} to zeros.")
            vals = np.nan_to_num(vals)
        return vals

    def interpolate_grad(
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
        vx, vy, vz = MATHLIB.leg2_tet_interp_grad(
            np.array([xs, ys, zs]),
            field,
            self.mesh.tets,
            self.mesh.edges,
            self.mesh.nodes,
            self.tet_to_field,
            tetids,
            tet_mapping,
        )
        n_zeros = np.isnan(vx).sum()
        if not usenan and n_zeros > 0:
            logger.debug(f"Converted {n_zeros} to zeros.")
            vx = np.nan_to_num(vx)
            vy = np.nan_to_num(vy)
            vz = np.nan_to_num(vz)
        return vx, vy, vz

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
            vals[vals == -1] = 0
        return vals
