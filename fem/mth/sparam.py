from ..bc import RobinBC
import numpy as np
from loguru import logger
from typing import Callable
from .integrals import surface_integral

def sparam_waveport(nodes: np.ndarray,
                    tri_vertices: np.ndarray,
                    bc: RobinBC, 
                    freq: float,
                    fieldf: Callable,
                    ndpts: int = 4):
    ''' Compute the S-parameters assuming a wave port mode
    
    Arguments:
    ----------
    nodes: np.ndarray = (3,:) np.ndarray of all nodes in the mesh.
    tri_vertices: np.ndarray = (3,:) np.ndarray of triangle indices that need to be integrated,
    bc: RobinBC = The port boundary condition object
    freq: float = The frequency at which to do the calculation
    fielf: Callable = The interpolation fuction that computes the E-field from the simulation
    ndpts: int = 4 the number of Duvanant integration points to use (default = 4)
    '''
    
    def modef(x, y, z):
        return bc.port_mode_3d_global(x, y, z, freq)

    def modef_c(x, y, z):
        return np.conj(modef(x, y, z))
    
    Q = 0
    if bc.active:
        Q = 1

    def fieldf_p(x, y, z):
        return fieldf(x,y,z) - Q * modef(x,y,z)
    
    
    def inproduct1(x, y, z):
        Ex1, Ey1, Ez1 = fieldf_p(x,y,z)
        Ex2, Ey2, Ez2 = modef_c(x,y,z)
        return Ex1*Ex2 + Ey1*Ey2 + Ez1*Ez2
    
    def inproduct2(x, y, z):
        Ex1, Ey1, Ez1 = modef(x,y,z)
        Ex2, Ey2, Ez2 = modef_c(x,y,z)
        return Ex1*Ex2 + Ey1*Ey2 + Ez1*Ez2
    
    mode_dot_field = surface_integral(nodes, tri_vertices, inproduct1, ndpts=ndpts)
    norm = surface_integral(nodes, tri_vertices, inproduct2, ndpts=ndpts)
    
    svec = mode_dot_field/norm
    return svec

def sparam_mode_power(nodes: np.ndarray,
                    tri_vertices: np.ndarray,
                    bc: RobinBC, 
                    k0: float,
                    const: np.ndarray,
                    ndpts: int = 4):
    ''' Compute the S-parameters assuming a wave port mode
    
    Arguments:
    ----------
    nodes: np.ndarray = (3,:) np.ndarray of all nodes in the mesh.
    tri_vertices: np.ndarray = (3,:) np.ndarray of triangle indices that need to be integrated,
    bc: RobinBC = The port boundary condition object
    freq: float = The frequency at which to do the calculation
    fielf: Callable = The interpolation fuction that computes the E-field from the simulation
    ndpts: int = 4 the number of Duvanant integration points to use (default = 4)
    '''

    def modef(x, y, z):
        return bc.port_mode_3d_global(x, y, z, k0)
    
    def inproduct2(x, y, z):
        Ex1, Ey1, Ez1 = modef(x,y,z)
        Ex2, Ey2, Ez2 = np.conj(modef(x,y,z))
        return (Ex1*Ex2 + Ey1*Ey2 + Ez1*Ez2)/(2*bc.Zmode(k0))
    
    norm = surface_integral(nodes, tri_vertices, inproduct2, const, ndpts=ndpts)
    
    return norm

def sparam_field_power(nodes: np.ndarray,
                    tri_vertices: np.ndarray,
                    bc: RobinBC, 
                    k0: float,
                    fieldf: Callable,
                    const: np.ndarray,
                    ndpts: int = 4):
    ''' Compute the S-parameters assuming a wave port mode
    
    Arguments:
    ----------
    nodes: np.ndarray = (3,:) np.ndarray of all nodes in the mesh.
    tri_vertices: np.ndarray = (3,:) np.ndarray of triangle indices that need to be integrated,
    bc: RobinBC = The port boundary condition object
    freq: float = The frequency at which to do the calculation
    fielf: Callable = The interpolation fuction that computes the E-field from the simulation
    ndpts: int = 4 the number of Duvanant integration points to use (default = 4)
    '''
    
    def modef(x, y, z):
        return bc.port_mode_3d_global(x, y, z, k0)
    
    Q = 0
    if bc.active:
        Q = 1

    def fieldf_p(x, y, z):
        return fieldf(x,y,z) - Q * modef(x,y,z)
    
    def inproduct1(x, y, z):
        Ex1, Ey1, Ez1 = fieldf_p(x,y,z)
        Ex2, Ey2, Ez2 = np.conj(modef(x,y,z))
        return (Ex1*Ex2 + Ey1*Ey2 + Ez1*Ez2) / (2*bc.Zmode(k0))
    
    mode_dot_field = surface_integral(nodes, tri_vertices, inproduct1, const, ndpts=ndpts)
    
    svec = mode_dot_field
    return svec