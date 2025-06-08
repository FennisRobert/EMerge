import numpy as np
from dataclasses import dataclass

@dataclass
class Material:
    er: float = 1
    ur: float = 1
    tand: float = 0
    sigma: float = 0
    _neff: float = None
    _fer: callable = None
    _fur: callable = None
    
    @property
    def ermat(self) -> np.ndarray:
        if isinstance(self.er, (float, complex, int, np.float64, np.complex128)):
            return self.er*np.eye(3)
        else:
            return self.er
    
    @property
    def urmat(self) -> np.ndarray:
        if isinstance(self.ur, (float, complex, int, np.float64, np.complex128)):
            return self.ur*np.eye(3)
        else:
            return self.ur
    
    @property
    def neff(self) -> complex:
        if self._neff is not None:
            return self._neff
        er = self.ermat[0,0]
        ur = self.urmat[0,0]

        return np.abs(np.sqrt(er*(1-1j*self.tand)*ur))
    
    @property
    def fer2d(self) -> callable:
        if self._fer is None:
            return lambda x,y: self.er*np.ones_like(x)
        else:
            return self._fer
        
    @property
    def fur2d(self) -> callable:
        if self._fur is None:

            return lambda x,y: self.ur*np.ones_like(x)
        else:
            return self._fur
    @property
    def fer3d(self) -> callable:
        if self._fer is None:
            return lambda x,y,z: self.er*np.ones_like(x)
        else:
            return self._fer
    
    @property
    def fur3d(self) -> callable:
        if self._fur is None:
            return lambda x,y,z: self.ur*np.ones_like(x)
        else:
            return self._fur
    @property
    def fer3d_mat(self) -> callable:
        if self._fer is None:
            
            return lambda x,y,z: np.repeat(self.ermat[:, :, np.newaxis], x.shape[0], axis=2)
        else:
            return self._fer
    
    @property
    def fur3d_mat(self) -> callable:
        if self._fur is None:
            return lambda x,y,z: np.repeat(self.urmat[:, :, np.newaxis], x.shape[0], axis=2)
        else:
            return self._fur
    
AIR = Material()
VACUUM = Material()
FR4 = Material(er=4.4, tand=0.001)

