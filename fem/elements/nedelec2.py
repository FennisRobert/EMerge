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
from ..mesh3d import Mesh3D
from .femdata import FEMBasis
from ..mth.tet import ned2_tet_interp, ned2_tet_interp_curl
from ..mth.tri import ned2_tri_stiff_force, ned2_tri_stiff_mass, ned2_tri_surface_integral, ned2_tri_stiff
from ..mth.optimized import local_mapping


############### Nedelec2 Class

class Nedelec2(FEMBasis):


    def __init__(self, mesh: Mesh3D):
        super().__init__(mesh)
        
        self.nedges: int = self.mesh.n_edges
        self.ntris: int = self.mesh.n_tris
        self.ntets: int = self.mesh.n_tets

        self.nfield: int = 2*self.nedges + 2*self.ntris
        
        ######## MESH Derived

        nedges = self.mesh.n_edges
        ntris = self.mesh.n_tris

        self.tet_to_field: np.ndarray = np.zeros((20, self.mesh.tets.shape[1]), dtype=int)
        self.tet_to_field[:6,:] = self.mesh.tet_to_edge
        self.tet_to_field[6:10,:] = self.mesh.tet_to_tri + nedges
        self.tet_to_field[10:16,:] = self.mesh.tet_to_edge + (ntris+nedges)
        self.tet_to_field[16:20,:] = self.mesh.tet_to_tri + (ntris+2*nedges)
    
        self.edge_to_field: np.ndarray = np.zeros((2,nedges), dtype=int)

        self.edge_to_field[0,:] = np.arange(nedges)
        self.edge_to_field[1,:] = np.arange(nedges) + ntris + nedges

        self.tri_to_field: np.ndarray = np.zeros((8,ntris), dtype=int)

        self.tri_to_field[:3,:] = self.mesh.tri_to_edge
        self.tri_to_field[3,:] = np.arange(ntris) + nedges
        self.tri_to_field[4:7,:] = self.mesh.tri_to_edge + nedges + ntris
        self.tri_to_field[7,:] = np.arange(ntris) + 2*nedges + ntris

        ##
        self._field: np.ndarray = None
        self.n_tet_dofs = 20
        self.n_tri_dofs = 8
        self._all_tet_ids = np.arange(self.ntets)

        self.empty_tri_rowcol()
    
    def interpolate(self, field: np.ndarray, xs: np.ndarray, ys: np.ndarray, zs:np.ndarray, tetids: np.ndarray = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        ''' 
        Interpolate the provided field data array at the given xs, ys and zs coordinates
        '''
        if tetids is None:
            tetids = self._all_tet_ids
        return ned2_tet_interp(np.array([xs, ys, zs]), field, self.mesh.tets, self.mesh.tris, self.mesh.edges, self.mesh.nodes, self.tet_to_field, tetids)
    
    def interpolate_curl(self, field: np.ndarray, xs: np.ndarray, ys: np.ndarray, zs:np.ndarray, c: np.ndarray, tetids: np.ndarray = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Interpolates the curl of the field at the given points.
        """
        if tetids is None:
            tetids = self._all_tet_ids
        return ned2_tet_interp_curl(np.array([xs, ys,zs]), field, self.mesh.tets, self.mesh.tris, self.mesh.edges, self.mesh.nodes, self.tet_to_field, c, tetids)
    
    ###### INDEX MAPPINGS

    def local_tet_to_triid(self, itet: int) -> np.ndarray:
        tri_ids = self.tet_to_field[6:10, itet] - self.n_edges
        global_tri_map = self.mesh.tris[:, tri_ids]
        return local_mapping(self.mesh.tets[:, itet], global_tri_map)

    def local_tet_to_edgeid(self, itet: int) -> np.ndarray:
        global_edge_map = self.mesh.edges[:, self.tet_to_field[:6,itet]]
        return local_mapping(self.mesh.tets[:, itet], global_edge_map)

    def local_tri_to_edgeid(self, itri: int) -> np.ndarray:
        global_edge_map = self.mesh.edges[:, self.tri_to_field[:3,itri]]
        return local_mapping(self.mesh.tris[:, itri], global_edge_map)
    
    def map_edge_to_field(self, edge_ids: np.ndarray) -> np.ndarray:
        """
        Returns the field ids for the edges.
        """
        # Concatinate the edges with the edges + ntris + nedges
        edge_ids = np.array(edge_ids)
        return np.concatenate((edge_ids, edge_ids + self.ntris + self.nedges))
    
    ########
    # @staticmethod
    # def tet_stiff_mass_submatrix(tet_vertices: np.ndarray, 
    #                              edge_lengths: np.ndarray, 
    #                              local_edge_map: np.ndarray, 
    #                              local_tri_map: np.ndarray, 
    #                              C_stiffness: float, 
    #                              C_mass: float) -> tuple[np.ndarray, np.ndarray]:
    #     return ned2_tet_stiff_mass(tet_vertices, edge_lengths, local_edge_map, local_tri_map, C_stiffness, C_mass)
    
    @staticmethod
    def tri_stiff_mass_submatrix(tri_vertices: np.ndarray,
                                 local_edge_map: np.ndarray,
                                 C_stiffness: float, 
                                 C_mass: float) -> tuple[np.ndarray, np.ndarray]:
        return ned2_tri_stiff_mass(tri_vertices, local_edge_map, C_stiffness, C_mass)
    
    @staticmethod
    def tri_stiff_vec_matrix(lcs_vertices: np.ndarray, 
                             gamma: complex, 
                             lcs_Uinc: np.ndarray, 
                             DPTs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return ned2_tri_stiff_force(lcs_vertices, gamma, lcs_Uinc, DPTs)
    
    @staticmethod
    def tri_stiff_matrix(lcs_vertices: np.ndarray, 
                             edge_lengths: np.ndarray, 
                             gamma: complex) -> tuple[np.ndarray, np.ndarray]:
        return ned2_tri_stiff(lcs_vertices, edge_lengths, gamma)
    

    @staticmethod
    def tri_surf_integral(lcs_vertices: np.ndarray, 
                          edge_lengths: np.ndarray, 
                          lcs_Uinc: np.ndarray, 
                          DPTs: np.ndarray) -> complex:
        return ned2_tri_surface_integral(lcs_vertices, edge_lengths, lcs_Uinc, DPTs)