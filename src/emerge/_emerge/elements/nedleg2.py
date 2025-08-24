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
from .femdata import FEMBasis
from .ned2_interp import ned2_tri_interp_full, ned2_tri_interp_curl
from ..mth.optimized import matinv
from ..cs import CoordinateSystem
from ..const import MU0, C0

## TODO: TEMPORARY SOLUTION FIX THIS

class FieldFunctionClass:
    """ This Class serves as a picklable class so that ModalPort boundary conditions
    can actually be stored with the Simulation class. Functions aren't picklable in 
    Python. 
    
    I am not happy with the existence of this class, it feels too ad-hoc but for now it
    is the simplest way. It stores all actually required information needed to do a
    surface field interpolation without needing to store the Mesh3D and SurfaceMesh class
    objects plus the NedelecLegrange2 classes with the Simulation. 

    As it stands currently, only the GMSH mesh is stored plus the geometry objects. The
    mesh is reconstructed as it is deterministic.
    """
    def __init__(self, 
                 field: np.ndarray,
                 cs: CoordinateSystem,
                 nodes: np.ndarray,
                 tris: np.ndarray,
                 tri_to_field: np.ndarray,
                 EH: str = 'E',
                 diadic: np.ndarray | None = None,
                 beta: float | None = None,
                 constant: float | int | complex = 1.0):
        self.field: np.ndarray = field
        self.cs: CoordinateSystem = cs
        self.nodes: np.ndarray = nodes
        self.tris: np.ndarray = tris
        self.tri_to_field: np.ndarray = tri_to_field
        self.eh: str = EH
        self.diadic: np.ndarray | None = diadic
        self.beta: float | None = beta
        self.constant: float = constant
        if EH == 'H':
            if diadic is None:
                self.diadic = np.eye(3)[:,:,np.newaxis()] * np.ones((self.tris.shape[1])) # type: ignore

    def __call__(self, xs: np.ndarray,
             ys: np.ndarray,
             zs: np.ndarray):
        xl, yl, zl = self.cs.in_local_cs(xs, ys, zs)
        if self.eh == 'E':
            Fxl, Fyl, Fzl = self.calcE(xl, yl)
        else:
            Fxl, Fyl, Fzl = self.calcH(xl, yl)
        Fx, Fy, Fz = self.cs.in_global_basis(Fxl, Fyl, Fzl)
        return np.array([Fx, Fy, Fz])*self.constant
    
    def calcE(self, xs: np.ndarray, ys: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        coordinates = np.array([xs, ys])
        return ned2_tri_interp_full(coordinates, 
                               self.field, 
                               self.tris,  
                               self.nodes, 
                               self.tri_to_field)
    
    def calcH(self, xs: np.ndarray, ys: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        coordinates = np.array([xs, ys])
        
        return ned2_tri_interp_curl(coordinates, 
                               self.field, 
                               self.tris,  
                               self.nodes, 
                               self.tri_to_field,
                               self.diadic,
                               self.beta)

############### Nedelec2 Class

class NedelecLegrange2(FEMBasis):


    def __init__(self, mesh: SurfaceMesh, cs: CoordinateSystem):

        self.mesh: SurfaceMesh = mesh

        self.cs: CoordinateSystem = cs

        LGORDER = 2
        
        ## 
        nodes = self.mesh.nodes
        self.local_nodes: np.ndarray = np.array(self.cs.in_local_cs(nodes[0,:], nodes[1,:], nodes[2,:]))
                                                
        ## Counters
        self.n_nodes: int = self.mesh.n_nodes
        self.n_edges: int = self.mesh.n_edges
        self.n_tris: int = self.mesh.n_tris
        self.n_tri_dofs: int = None

        if LGORDER == 2:
            self.n_field: int = 2*self.n_edges + 2*self.n_tris + self.n_nodes + self.n_edges
        else:
            self.n_field: int = 2*self.n_edges + 2*self.n_tris + self.n_nodes + self.n_edges*2 + self.n_tris
        
        self.n_xy: int = 2*self.n_edges + 2*self.n_tris
        ######## MESH Derived
        Nn = self.mesh.n_nodes
        Ne = self.mesh.n_edges
        Nt = self.mesh.n_tris

        
        
        if LGORDER==3:
            self.tri_to_field: np.ndarray = np.zeros((8 + 10, self.n_tris), dtype=int)
        else:
            self.tri_to_field: np.ndarray = np.zeros((8 + 6, self.n_tris), dtype=int)

        self.tri_to_field[:3,:] = self.mesh.tri_to_edge
        self.tri_to_field[3,:] = np.arange(Nt) + Ne
        self.tri_to_field[4:7,:] = self.mesh.tri_to_edge + Ne + Nt
        self.tri_to_field[7,:] = np.arange(Nt) + 2*Ne + Nt
        self.tri_to_field[8:11,:] = self.mesh.tris + (2*Ne + 2*Nt) # + E + T + E + T
        self.tri_to_field[11:14,:] = self.mesh.tri_to_edge + (2*Ne + 2*Nt + Nn)
        
        if LGORDER==3:
            # Legrange 3
            self.tri_to_field[14:17,:] = self.mesh.tri_to_edge + (2*Ne + 2*Nt + Nn + Ne)
            self.tri_to_field[17,:] = np.arange(Nt) + (2*Ne + 2*Nt + Nn + Ne + Ne)
        
        self.edge_to_field: np.ndarray = np.zeros((5,Ne), dtype=int) #edge mode 1, edge mode 2, edge legrande mode, edge vertex mode 1, edge vertex mode 2

        self.edge_to_field[0,:] = np.arange(Ne)
        self.edge_to_field[1,:] = np.arange(Ne) + Nt + Ne
        self.edge_to_field[2:4,:] = self.mesh.edges + Ne*2 + Nt*2
        self.edge_to_field[4,:] = np.arange(Ne) + Ne*2 + Nt*2 + Nn
        
        if LGORDER==3:
            self.edge_to_field[5:,:] = np.arange(Ne)  + Ne*2 + Nt*2 + Nn + Ne

        ##
        self._field: np.ndarray = None   
        self._rows: np.ndarray = None
        self._cols: np.ndarray = None 

    def __call__(self, **kwargs) -> NedelecLegrange2:
        self._field = self.fielddata(**kwargs)
        return self
    
    def interpolate_Ef(self, field: np.ndarray) -> FieldFunctionClass:
        '''Generates the Interpolation function as a function object for a given coordiante basis and origin.'''
        return FieldFunctionClass(field, self.cs, self.local_nodes, self.mesh.tris, self.tri_to_field, 'E')

    def interpolate_Hf(self, field: np.ndarray, k0: float, ur: np.ndarray, beta: float) -> FieldFunctionClass:
        '''Generates the Interpolation function as a function object for a given coordiante basis and origin.'''
        constant = 1j / ((k0*C0)*MU0)
        urinv = np.zeros_like(ur)
        
        for i in range(ur.shape[2]):
            urinv[:,:,i] = matinv(ur[:,:,i])

        return FieldFunctionClass(field, self.cs, self.local_nodes, self.mesh.tris, self.tri_to_field, 'H', urinv, beta, constant)
    
    def tri_interpolate(self, field, xs: np.ndarray, ys: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        coordinates = np.array([xs, ys])
        return ned2_tri_interp_full(coordinates, 
                               field, 
                               self.mesh.tris,  
                               self.local_nodes, 
                               self.tri_to_field)
    
    def tri_interpolate_curl(self, field, xs: np.ndarray, ys: np.ndarray, diadic: np.ndarray | None = None, beta: float = 0.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        coordinates = np.array([xs, ys])
        if diadic is None:
            diadic = np.eye(3)[:,:,np.newaxis()] * np.ones((self.mesh.n_tris)) # type: ignore
        return ned2_tri_interp_curl(coordinates, 
                               field, 
                               self.mesh.tris,  
                               self.local_nodes, 
                               self.tri_to_field,
                               diadic,
                               beta)