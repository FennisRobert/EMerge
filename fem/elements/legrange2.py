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
from ..dataset_old import Dataset, Axis
from .femdata import FEMBasis
from ..mth.tet import leg2_tet_interp, leg2_tet_grad_interp, leg2_tet_stiff
from ..mth.tri import leg2_tri_stiff
from ..mth.optimized import local_mapping
from typing import Callable

############### Nedelec2 Class

class Legrange2(FEMBasis):


    def __init__(self, mesh: Mesh3D):
        super().__init__(mesh)

        self.nedges: int = self.mesh.n_edges
        self.ntris: int = self.mesh.n_tris
        self.ntets: int = self.mesh.n_tets

        self.nfield: int = 2*self.nedges + 2*self.ntris

        self.axes: list[Axis] = []
        self.fieldata: Dataset = None
        
        ######## MESH Derived

        nedges = self.mesh.n_edges
        ntris = self.mesh.n_tris

        self.tet_to_field: np.ndarray = np.zeros((10, self.mesh.tets.shape[1]), dtype=int)
        self.tet_to_field[:4,:] = self.mesh.tets
        self.tet_to_field[4:10,:] = self.mesh.tet_to_edge + self.mesh.n_nodes

        self.edge_to_field: np.ndarray = np.arange(nedges) + self.mesh.n_nodes

        self.tri_to_field: np.ndarray = np.zeros((6,ntris), dtype=int)

        self.tri_to_field[:3,:] = self.mesh.tris
        self.tri_to_field[3:6,:] = self.mesh.tri_to_edge + self.mesh.n_nodes

        ##
        self._field: np.ndarray = None

    def __call__(self, **kwargs) -> Legrange2:
        self._field = self.fielddata(**kwargs)
        return self
    
    def interpolate(self, field, xs: np.ndarray, ys: np.ndarray, zs:np.ndarray, tet_ids: np.ndarray = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if tet_ids is None:
            tet_ids = np.arange(self.mesh.n_tets)
        return leg2_tet_interp(np.array([xs, ys,zs]), field, self.mesh.tets, self.mesh.tris, self.mesh.edges, self.mesh.nodes, self.tet_to_field, self.mesh.tet_to_edge, self.mesh.tet_to_tri, tet_ids)
    
    def interpolate_grad(self, field, xs: np.ndarray, ys: np.ndarray, zs:np.ndarray, tet_ids: np.ndarray = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if tet_ids is None:
            tet_ids = np.arange(self.mesh.tet_ids)
        return leg2_tet_grad_interp(np.array([xs, ys,zs]), field, self.mesh.tets, self.mesh.tris, self.mesh.edges, self.mesh.nodes, self.tet_to_field, self.mesh.tet_to_edge, self.mesh.tet_to_tri, tet_ids)
    
    def fieldf(self, field: np.ndarray, basis: np.ndarray = None, origin: np.ndarray = None) -> Callable:
        if basis is None:
            basis = np.eye(3)

        if origin is None:
            origin = np.zeros(3)
        
        ibasis = np.linalg.pinv(basis)
        def func(xs: np.ndarray, ys: np.ndarray, zs: np.ndarray) -> np.ndarray:
            xyz = np.array([xs, ys, zs]) + origin[:, np.newaxis]
            xyzg = basis @ xyz
            return ibasis @ np.array(self.interpolate(field, xyzg[0,:], xyzg[1,:], xyzg[2,:]))
        return func
    

    ###### Vertex getter

    def field_to_vertices(self, ifield: np.ndarray):
        return_ids = ifield[ifield < self.mesh.n_nodes]
        return return_ids
    
    def field_to_edges(self, ifield: np.ndarray):
        return_ids = ifield[ifield >= self.mesh.n_nodes] - self.mesh.n_nodes
        return return_ids
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
    @staticmethod
    def tet_stiff_mass_submatrix(tet_vertices: np.ndarray, 
                                 edge_lengths: np.ndarray, 
                                 local_edge_map: np.ndarray, 
                                 local_tri_map: np.ndarray, 
                                 C_stiffness: float, 
                                 C_mass: float) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError("tet_stiff_mass_submatrix is not implemented for Legrange2")
    
    @staticmethod
    def tet_stiff_submatrix(tet_vertices: np.ndarray, 
                                 edge_lengths: np.ndarray, 
                                 local_edge_map: np.ndarray, 
                                 local_tri_map: np.ndarray, 
                                 C_stiffness: float) -> tuple[np.ndarray, np.ndarray]:
        raise leg2_tet_stiff(tet_vertices, edge_lengths, local_edge_map, local_tri_map, C_stiffness)
    

    @staticmethod
    def tri_stiff_mass_submatrix(tri_vertices: np.ndarray, 
                                 edge_lengths: np.ndarray,
                                 local_edge_map: np.ndarray,
                                 C_stiffness: float, 
                                 C_mass: float) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError("tri stiff mass is not implemented for Legrange2")
    
    @staticmethod
    def tri_stiff_submatrix(tri_vertices: np.ndarray, 
                                 local_edge_map: np.ndarray,
                                 C_stiffness: float) -> np.ndarray:
        return leg2_tri_stiff(tri_vertices, local_edge_map, C_stiffness)
    

    @staticmethod
    def tri_stiff_vec_matrix(lcs_vertices: np.ndarray, 
                             edge_lengths: np.ndarray, 
                             gamma: complex, 
                             lcs_Uinc: np.ndarray, 
                             DPTs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError("tri stiff mass is not implemented for Legrange2")
    
    @staticmethod
    def tri_surf_integral(lcs_vertices: np.ndarray, 
                          edge_lengths: np.ndarray, 
                          lcs_Uinc: np.ndarray, 
                          DPTs: np.ndarray) -> complex:
       raise NotImplementedError("tri stiff mass is not implemented for Legrange2")