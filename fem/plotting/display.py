from ..mesh3d import Mesh3D
from ..geo3d import GMSHObject, GMSHSurface, GMSHVolume
from ..selection import Selection
from ..bc import PortBC
from typing import Iterable, Literal
import numpy as np

class BaseDisplay:

    def __init__(self, mesh: Mesh3D):
        self._mesh: Mesh3D = mesh

    def show(self):
        raise NotImplementedError('This method is not implemented')
        
    def add_object(self, obj: GMSHObject | Selection | Iterable,*args, **kwargs):
        raise NotImplementedError('This method is not implemented')

    def add_scatter(self, xs: np.ndarray, ys: np.ndarray, zs: np.ndarray):
        raise NotImplementedError('This method is not implemented')

    def add_portmode(self, port: PortBC, k0: float, Npoints: int = 10, dv=(0,0,0), XYZ=None,
                      field: Literal['E','H'] = 'E'):
        raise NotImplementedError('This method is not implemented')

    def add_quiver(self, x: np.ndarray, y: np.ndarray, z: np.ndarray,
              dx: np.ndarray, dy: np.ndarray, dz: np.ndarray,
              scale: float = 1,
              scalemode: Literal['lin','log'] = 'lin'):
        
        raise NotImplementedError('This method is not implemented')