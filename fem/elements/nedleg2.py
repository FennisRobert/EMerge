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
from ..mesh3d import SurfaceMesh
from ..dataset_old import Dataset, Axis
from .femdata import FEMBasis
from ..mth.tri import ned2_tri_interp, ned2_tri_interp_full, ned2_tri_interp_curl
from ..mth.optimized import local_mapping, matinv
from ..cs import CoordinateSystem
from typing import Callable

############### Nedelec2 Class

class NedelecLegrange2(FEMBasis):


    def __init__(self, mesh: SurfaceMesh, cs: CoordinateSystem):

        self.mesh: SurfaceMesh = mesh

        self.cs: CoordinateSystem = cs

        ## 
        nodes = self.mesh.nodes
        self.local_nodes: np.ndarray = np.array(self.cs.in_local_cs(nodes[0,:], nodes[1,:], nodes[2,:]))
                                                
        ## Counters
        self.n_nodes: int = self.mesh.n_nodes
        self.n_edges: int = self.mesh.n_edges
        self.n_tris: int = self.mesh.n_tris
        self.n_tri_dofs: int = None

        self.n_field: int = 2*self.n_edges + 2*self.n_tris + self.n_nodes + self.n_edges
        self.n_xy: int = 2*self.n_edges + 2*self.n_tris

        ######## MESH Derived
        Nn = self.mesh.n_nodes
        Ne = self.mesh.n_edges
        Nt = self.mesh.n_tris

        self.tri_to_field: np.ndarray = np.zeros((8 + 6, self.n_tris), dtype=int)

        self.tri_to_field[:3,:] = self.mesh.tri_to_edge
        self.tri_to_field[3,:] = np.arange(Nt) + Ne
        self.tri_to_field[4:7,:] = self.mesh.tri_to_edge + Ne + Nt
        self.tri_to_field[7,:] = np.arange(Nt) + 2*Ne + Nt
        self.tri_to_field[8:11,:] = self.mesh.tris + (2*Ne + 2*Nt) # + E + T + E + T
        self.tri_to_field[11:14,:] = self.mesh.tri_to_edge + (2*Ne + 2*Nt + Nn)
    
        self.edge_to_field: np.ndarray = np.zeros((5,Ne), dtype=int)

        self.edge_to_field[0,:] = np.arange(Ne)
        self.edge_to_field[1,:] = np.arange(Ne) + Nt + Ne
        self.edge_to_field[2,:] = np.arange(Ne) + Ne*2 + Nt*2 + Nn
        self.edge_to_field[3:,:] = self.mesh.edges + Ne*2 + Nt*2

        ##
        self._field: np.ndarray = None   
        self._rows: np.ndarray = None
        self._cols: np.ndarray = None 

    def __call__(self, **kwargs) -> NedelecLegrange2:
        self._field = self.fielddata(**kwargs)
        return self
    
    def interpolate_Ef(self, field: np.ndarray) -> Callable:
        '''Generates the Interpolation function as a function object for a given coordiante basis and origin.'''
        
        def func(xs: np.ndarray, ys: np.ndarray, zs: np.ndarray) -> np.ndarray:
            xl, yl, zl = self.cs.in_local_cs(xs, ys, zs)
            Exl, Eyl, Ezl = self.tri_interpolate(field, xl, yl)
            Ex, Ey, Ez = self.cs.in_global_basis(Exl, Eyl, Ezl)
            return np.array([Ex, Ey, Ez])
        return func

    def interpolate_Hf(self, field: np.ndarray, k0: float, ur: np.ndarray, beta: float) -> Callable:
        '''Generates the Interpolation function as a function object for a given coordiante basis and origin.'''
        constant = 1j/ ((k0*299792458)*(4*np.pi*1e-7))
        urinv = np.zeros_like(ur)
        
        for i in range(ur.shape[2]):
            urinv[:,:,i] = matinv(ur[:,:,i])

        def func(xs: np.ndarray, ys: np.ndarray, zs: np.ndarray) -> np.ndarray:
            xl, yl, _ = self.cs.in_local_cs(xs, ys, zs)
            Exl, Eyl, Ezl = self.tri_interpolate_curl(field, xl, yl, urinv, beta)
            Ex, Ey, Ez = self.cs.in_global_basis(Exl, Eyl, Ezl)
            return np.array([Ex, Ey, Ez])*constant
        return func
    
    def tri_interpolate(self, field, xs: np.ndarray, ys: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        coordinates = np.array([xs, ys])
        return ned2_tri_interp_full(coordinates, 
                               field, 
                               self.mesh.tris,  
                               self.local_nodes, 
                               self.tri_to_field)
    
    def tri_interpolate_curl(self, field, xs: np.ndarray, ys: np.ndarray,diadic: np.ndarray = None,beta: float = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        coordinates = np.array([xs, ys])
        if diadic is None:
            diadic = np.eye(3)[:,:,np.newaxis()] * np.ones((self.mesh.n_tris))
        return ned2_tri_interp_curl(coordinates, 
                               field, 
                               self.mesh.tris,  
                               self.local_nodes, 
                               self.tri_to_field,
                               diadic,
                               beta)
    
    
    # def interpolate_curl(self, field: np.ndarray, xs: np.ndarray, ys: np.ndarray, zs:np.ndarray, c: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    #     """
    #     Interpolates the curl of the field at the given points.
    #     """
    #     return ned2_tet_interp_curl(np.array([xs, ys,zs]), field, self.mesh.tets, self.mesh.tris, self.mesh.edges, self.mesh.nodes, self.tet_to_field, c)
    
    # def fieldf(self, field: np.ndarray, basis: np.ndarray = None, origin: np.ndarray = None) -> Callable:
    #     if basis is None:
    #         basis = np.eye(3)

    #     if origin is None:
    #         origin = np.zeros(3)
        
    #     ibasis = np.linalg.pinv(basis)
    #     def func(xs: np.ndarray, ys: np.ndarray, zs: np.ndarray) -> np.ndarray:
    #         xyz = np.array([xs, ys, zs]) + origin[:, np.newaxis]
    #         xyzg = basis @ xyz
    #         return ibasis @ np.array(self.interpolate(field, xyzg[0,:], xyzg[1,:], xyzg[2,:]))
    #     return func
    
    ###### INDEX MAPPINGS

    