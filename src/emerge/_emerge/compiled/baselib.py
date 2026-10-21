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

# Last Cleanup: 2025-03-12
import numpy as np
from emr import interp, index_interp


class CompiledLib:
    ############################################################
    #                      NEDELEC 3D ELEMENTS                 #
    ############################################################

    @staticmethod
    def ned2_tet_interp(
        coords: np.ndarray,
        solutions: np.ndarray,
        tets: np.ndarray,
        tris: np.ndarray,
        edges: np.ndarray,
        nodes: np.ndarray,
        tet_to_field: np.ndarray,
        tet_to_edge: np.ndarray,
        tet_to_tri: np.ndarray,
        tetids: np.ndarray,
        tet_mapping: np.ndarray,
        dofcodes: np.ndarray,
    ):
        return interp.ned2_tet_interp(
            coords,
            solutions,
            tets,
            tris,
            edges,
            nodes,
            tet_to_field,
            tet_to_edge,
            tet_to_tri,
            tetids,
            tet_mapping,
            dofcodes,
        )

    @staticmethod
    def ned2_tet_interp_curl(
        coords: np.ndarray,
        solutions: np.ndarray,
        tets: np.ndarray,
        tris: np.ndarray,
        edges: np.ndarray,
        nodes: np.ndarray,
        tet_to_field: np.ndarray,
        tet_to_edge: np.ndarray,
        tet_to_tri: np.ndarray,
        c: np.ndarray,
        tetids: np.ndarray,
        tet_mapping: np.ndarray,
        dofcodes: np.ndarray,
    ):
        return interp.ned2_tet_interp_curl(
            coords,
            solutions,
            tets,
            tris,
            edges,
            nodes,
            tet_to_field,
            tet_to_edge,
            tet_to_tri,
            c,
            tetids,
            tet_mapping,
            dofcodes,
        )

    ############################################################
    #                     LEGRANGE 3D ELEMENTS                 #
    ############################################################

    @staticmethod
    def leg2_tet_interp(
        coords: np.ndarray,
        solutions: np.ndarray,
        tets: np.ndarray,
        edges: np.ndarray,
        nodes: np.ndarray,
        tet_to_field: np.ndarray,
        tetids: np.ndarray,
        tet_mapping: np.ndarray,
    ) -> np.ndarray:
        return interp.leg2_tet_interp(
            coords,
            solutions,
            tets,
            edges,
            nodes,
            tet_to_field,
            tetids,
            tet_mapping,
        )

    @staticmethod
    def leg2_tet_interp_grad(
        coords: np.ndarray,
        solutions: np.ndarray,
        tets: np.ndarray,
        edges: np.ndarray,
        nodes: np.ndarray,
        tet_to_field: np.ndarray,
        tetids: np.ndarray,
        tet_mapping: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return interp.leg2_tet_interp_grad(
            coords, solutions, tets, edges, nodes, tet_to_field, tetids, tet_mapping
        )

    ############################################################
    #                      NEDELEC 2D ELEMENTS                 #
    ############################################################

    @staticmethod
    def ned2_tri_interp_full(
        coords: np.ndarray,
        solutions: np.ndarray,
        tris: np.ndarray,
        nodes: np.ndarray,
        tri_to_field: np.ndarray,
        dofcodes: np.ndarray,
    ):
        return interp.ned2_tri_interp_full(coords, solutions, tris, nodes, tri_to_field, dofcodes)

    @staticmethod
    def ned2_tri_interp_ezgrad(
        coords: np.ndarray,
        solutions: np.ndarray,
        tris: np.ndarray,
        nodes: np.ndarray,
        tri_to_field: np.ndarray,
        dofcodes: np.ndarray,
    ):
        return interp.ned2_tri_interp_grad(coords, solutions, tris, nodes, tri_to_field, dofcodes)

    @staticmethod
    def ned2_tri_interp_curl(
        coords: np.ndarray,
        solutions: np.ndarray,
        tris: np.ndarray,
        nodes: np.ndarray,
        tri_to_field: np.ndarray,
        diadic: np.ndarray,
        beta: float,
        dofcodes: np.ndarray,
    ):
        return interp.ned2_tri_interp_curl(
            coords, solutions, tris, nodes, tri_to_field, diadic, beta, dofcodes
        )

    ############################################################
    #                         INDEX INTERP                    #
    ############################################################

    @staticmethod
    def index_interp(
        coords: np.ndarray, tets: np.ndarray, nodes: np.ndarray, tetids: np.ndarray
    ):
        return index_interp.index_interp(coords, tets, nodes, tetids)
