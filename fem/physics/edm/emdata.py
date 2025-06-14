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
from ...dataset import SimData, DataSet
from ...elements.femdata import FEMBasis
from dataclasses import dataclass
import numpy as np
from typing import Sequence, Type, Literal
from loguru import logger
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
    """The EMDataSet class stores solution data of FEM Time Harmonic simulations.

    
    """
    def __init__(self, **vars):
        self.er: np.ndarray = None
        self.ur: np.ndarray = None
        self.freq: float = None
        self.k0: float = None
        self.Sp: Sparam = None
        self._fields: dict[np.ndarray] = dict()
        self._mode_field: np.ndarray = None
        self.excitation: dict[int, complex] = dict()
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
    def _field(self) -> np.ndarray:
        if self._mode_field is not None:
            return self._mode_field
        return sum([self.excitation[mode.port_number]*self._fields[mode.port_number] for mode in self.port_modes]) 
    
    def set_field_vector(self) -> None:
        self.excitation = {key: 0.0 for key in self._fields.keys()}
        self.excitation[self.port_modes[0].port_number] = 1.0 + 0j

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
    """The EMSimData class contains all EM simulation data from a Time Harmonic simulation
    along all sweep axes.
    """
    datatype: type = EMDataSet
    def __init__(self, basis: FEMBasis):
        super().__init__()
        self._basis: FEMBasis = basis
        self._injections = dict(_basis=basis)
        self._axis = 'freq'

    def __getitem__(self, field: EMField) -> np.ndarray:
        return getattr(self, field)

    def howto(self) -> None:
        """To access data in the EMSimData class use the .ax method to extract properties selected
        along an access of global variables. The axes are all global properties that the EMDatasets manage.
        
        For example the following would return all S(2,1) parameters along the frequency axis.
        
        >>> freq, S21 = dataset.ax('freq').S(2,1)

        Alternatively, one can manually select any solution indexed in order of generation using.

        >>> S21 = dataset.item(3).S(2,1)

        To find the E or H fields at any coordinate, one can use the Dataset's .interpolate method. 
        This method returns the same Dataset object after which the computed fields can be accessed.

        >>> Ex = dataset.item(3).interpolate(xs,ys,zs).Ex

        Lastly, to find the solutions for a given frequency or other value, you can also just call the dataset
        class:
        
        >>> Ex, Ey, Ez = dataset(freq=1e9).interpolate(xs,ys,zs).E

        """

    def ax(self, field: EMField) -> EMDataSet:
        """Return a EMDataSet proxy object that you can request properties for along a provided axis.

        The EMSimData class contains a list of EMDataSet objects. Any global variable like .freq of the 
        EMDataSet object can be used as inner-axes after which the outer axis can be selected as if
        you are extract a single one.

        Args:
            field (EMField): The global field variable to select the data along

        Returns:
            EMDataSet: An EMDataSet object (actually a proxy for)

        Example:
        The following will select all S11 parameters along the frequency axis:

        >>> freq, S11 = dataset.ax('freq').S(1,1)

        """
        # find the real DataSet
        return _DataSetProxy(field, self.datasets)

    def export_touchstone(self, 
                          filename: str,
                          format: Literal['RI','MA','DB']):
        """Export the S-parameter data to a touchstone file

        This function assumes that all ports are numbered in sequence 1,2,3,4... etc with
        no missing ports. Otherwise it crashes. Will be update/improved soon with more features.

        Args:
            filename (str): The File name
            format (Literal[DB, RI, MA]): The dataformat used in the touchstone file.
        """
        from .touchstone import generate_touchstone
        logger.info(f'Exporting S-data to {filename}')
        # We will assume for now all ports are also numbered 1 to Nports+1
        Nports = len(self.datasets[0].excitation)
        freqs, _ = self.ax('freq').k0

        Smat = np.zeros((len(freqs),Nports,Nports), dtype=np.complex128)
        
        for i in range(1,Nports+1):
            for j in range(1,Nports+1):
                _, S = self.ax('freq').S(i,j)
                Smat[:,i-1,j-1] = S
        
        generate_touchstone(filename, freqs, Smat, data_format=format)
        
        logger.info('Export complete!')