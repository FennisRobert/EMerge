from __future__ import annotations
from ...dataset import SimData, DataSet, DataAxis
from ...elements.femdata import FEMBasis
from dataclasses import dataclass
import numpy as np
from typing import Sequence, Type, Literal

EMField = Literal[
    "er", "ur", "freq", "k0",
    "_Spdata", "_Spmapping", "_field", "_basis",
    "Nports", "Ex", "Ey", "Ez",
    "Hx", "Hy", "Hz",
    "mode", "beta",
]

@dataclass
class Sparam:
    """
    S-parameter matrix indexed by arbitrary port/mode labels (ints or floats).
    Internally stores a square numpy array; externally uses your mapping
    to translate (port1, port2) → (i, j).
    """
    def __init__(self, port_nrs: list[int | float]) -> None:
        # build label → index map
        self.map: dict[int | float, int] = {label: idx 
                                            for idx, label in enumerate(port_nrs)}
        n = len(port_nrs)
        # zero‐initialize the S‐parameter matrix
        self.arry: np.ndarray = np.zeros((n, n), dtype=np.complex128)

    def get(self, port1: int | float, port2: int | float) -> complex:
        """
        Return the S-parameter S(port1, port2).
        Raises KeyError if either port1 or port2 is not in the mapping.
        """
        try:
            i = self.map[port1]
            j = self.map[port2]
        except KeyError as e:
            raise KeyError(f"Port/mode {e.args[0]!r} not found in mapping") from None
        return self.arry[i, j]

    def set(self, port1: int | float, port2: int | float, value: complex) -> None:
        """
        Set the S-parameter S(port1, port2) = value.
        Raises KeyError if either port1 or port2 is not in the mapping.
        """
        try:
            i = self.map[port1]
            j = self.map[port2]
        except KeyError as e:
            raise KeyError(f"Port/mode {e.args[0]!r} not found in mapping") from None
        self.arry[i, j] = value

    # allow S(param1, param2) → complex, as before
    def __call__(self, port1: int | float, port2: int | float) -> complex:
        return self.get(port1, port2)

    # allow array‐style access: S[1, 1] → complex
    def __getitem__(self, key: tuple[int | float, int | float]) -> complex:
        port1, port2 = key
        return self.get(port1, port2)

    # allow array‐style setting: S[1, 2] = 0.3 + 0.1j
    def __setitem__(
        self,
        key: tuple[int | float, int | float],
        value: complex
    ) -> None:
        port1, port2 = key
        self.set(port1, port2, value)

@dataclass
class PortProperties:
    port_number: int | None = None
    k0: float | None= None
    beta: float | None = None
    Z0: float | None = None
    Pout: float | None = None
    mode_number: int = 1
    
class EMDataSet(DataSet):
    
    def __init__(self, **vars):
        self.er: np.ndarray = None
        self.ur: np.ndarray = None
        self.freq: float = None
        self.k0: float = None
        self.Sp: Sparam = None
        self._field: np.ndarray = None
        self._basis: FEMBasis = None
        self.Nports: int = None
        self.Ex: np.ndarray = None
        self.Ey: np.ndarray = None
        self.Ez: np.ndarray = None
        self.Hx: np.ndarray = None
        self.Hy: np.ndarray = None
        self.Hz: np.ndarray = None
        self.port_modes: list[PortProperties] = []
        self.mode: int = None
        self.beta: int = None

        super().__init__(**vars)

    @property
    def EH(self) -> tuple[np.ndarray, np.ndarray]:
        ''' Return the electric and magnetic field as a tuple of numpy arrays '''
        return np.array([self.Ex, self.Ey, self.Ez]), np.array([self.Hx, self.Hy, self.Hz])
    
    @property
    def E(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        ''' Return the electric field as a tuple of numpy arrays '''
        return self.Ex, self.Ey, self.Ez
    
    @property
    def Emat(self) -> np.ndarray:
        return np.array([self.Ex, self.Ey, self.Ez])
    
    @property
    def Hmat(self) -> np.ndarray:
        return np.array([self.Hx, self.Hy, self.Hz])

    @property
    def H(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        ''' Return the magnetic field as a tuple of numpy arrays '''
        return self.Hx, self.Hy, self.Hz
    
    def init_sp(self, portnumbers: list[int | float]) -> None:
        self.Sp = Sparam(portnumbers)

    def add_port_properties(self, 
                            port_number: int,
                            mode_number: int,
                            k0: float,
                            beta: float,
                            Z0: float,
                            Pout: float) -> None:
        self.port_modes.append(PortProperties(port_number=port_number,
                                              mode_number=mode_number,
                                              k0 = k0,
                                              beta=beta,
                                              Z0=Z0,
                                              Pout=Pout))
        
    def write_S(self, i1: int | float, i2: int | float, value: complex) -> None:
        self.Sp[i1,i2] = value

    def interpolate(self, xs: np.ndarray, ys: np.ndarray, zs: np.ndarray, freq=None) -> EMDataSet:
        ''' Interpolate the dataset in the provided xs, ys, zs values'''
        shp = xs.shape
        xf = xs.flatten()
        yf = ys.flatten()
        zf = zs.flatten()
        Ex, Ey, Ez = self._basis.interpolate(self._field, xf, yf, zf)
        self.Ex = Ex.reshape(shp)
        self.Ey = Ey.reshape(shp)
        self.Ez = Ez.reshape(shp)

        
        constants = 1/ (-1j*2*np.pi*self.freq*(self.ur*4*np.pi*1e-7) )
        Hx, Hy, Hz = self._basis.interpolate_curl(self._field, xf, yf, zf, constants)
        self.Hx = Hx.reshape(shp)
        self.Hy = Hy.reshape(shp)
        self.Hz = Hz.reshape(shp)

        return self

    def S(self, i1: int, i2: int) -> complex:
        ''' Returns the S-parameter S(i1,i2)'''
        return self.Sp(i1, i2)


class _DataSetProxy:
    """
    A “ghost” wrapper around a real DataSet.
    Any attr/method access is intercepted here first.
    """
    def __init__(self, field: str, dss: DataSet):
        # stash both the SimData (in case you need context)
        # and the real DataSet
        self._field = field
        self._dss = dss

    def __getattribute__(self, name: str):
       

        if name in ('_field','_dss'):
            return object.__getattribute__(self, name)
        xax = []
        yax = []
        field = object.__getattribute__(self, '_field')
        if callable(getattr(self._dss[0], name)):
            
            def wrapped(*args, **kwargs):
                
                for ds in self._dss:
                    # 1) grab the real attribute
                    xval = getattr(ds, field)
                    func = getattr(ds, name)
                    
                    yval = func(*args, **kwargs)
                    xax.append(xval)
                    yax.append(yval)
                return np.array(xax), np.array(yax)
            return wrapped
        else:
            for ds in self._dss:
                xax.append(getattr(ds, field))
                yax.append(getattr(ds, name))
            return np.array(xax), np.array(yax)


class EMSimData(SimData[EMDataSet]):
    datatype: type = EMDataSet
    def __init__(self, basis: FEMBasis):
        super().__init__()
        self._basis: FEMBasis = basis
        self._injections = dict(_basis=basis)
        self._axis = 'freq'

    def __getitem__(self, field: EMField) -> np.ndarray:
        return getattr(self, field)

    def ax(self, field: EMField) -> EMDataSet:
        # find the real DataSet
        return _DataSetProxy(field, self.datasets)