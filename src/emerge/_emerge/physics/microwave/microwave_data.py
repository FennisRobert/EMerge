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
from ...simulation_data import BaseDataset, DataContainer
from ...elements.femdata import FEMBasis
from dataclasses import dataclass
import numpy as np
from typing import Literal
from loguru import logger
from .adaptive_freq import SparamModel
from ...cs import Axis, _parse_axis
from ...selection import FaceSelection
from ...geometry import GeoSurface
from ...mesh3d import Mesh3D
from ...const import Z0, MU0, EPS0

EMField = Literal[
    "er", "ur", "freq", "k0",
    "_Spdata", "_Spmapping", "_field", "_basis",
    "Nports", "Ex", "Ey", "Ez",
    "Hx", "Hy", "Hz",
    "mode", "beta",
]

def arc_on_plane(ref_dir, normal, angle_range_deg, num_points=100):
    """
    Generate theta/phi coordinates of an arc on a plane.

    Parameters
    ----------
    ref_dir : tuple (dx, dy, dz)
        Reference direction (angle zero) lying in the plane.
    normal : tuple (nx, ny, nz)
        Plane normal vector.
    angle_range_deg : tuple (deg_start, deg_end)
        Start and end angle of the arc in degrees.
    num_points : int
        Number of points along the arc.

    Returns
    -------
    theta : ndarray
        Array of theta angles (radians).
    phi : ndarray
        Array of phi angles (radians).
    """
    d = np.array(ref_dir, dtype=float)
    n = np.array(normal, dtype=float)

    # Normalize normal
    n = n / np.linalg.norm(n)

    # Project d into the plane
    d_proj = d - np.dot(d, n) * n
    if np.linalg.norm(d_proj) < 1e-12:
        raise ValueError("Reference direction is parallel to the normal vector.")

    e1 = d_proj / np.linalg.norm(d_proj)
    e2 = np.cross(n, e1)

    # Generate angles along the arc
    angles_deg = np.linspace(angle_range_deg[0], angle_range_deg[1], num_points)
    angles_rad = np.deg2rad(angles_deg)

    # Create unit vectors along the arc
    vectors = np.outer(np.cos(angles_rad), e1) + np.outer(np.sin(angles_rad), e2)

    # Convert to spherical angles
    ux, uy, uz = vectors[:,0], vectors[:,1], vectors[:,2]

    theta = np.arccos(uz)         # theta = arcsin(z)
    phi = np.arctan2(uy, ux)      # phi = atan2(y, x)

    return theta, phi

def renormalise_s(S: np.ndarray,
                  Zn: np.ndarray,
                  Z0: complex | float = 50) -> np.ndarray:
    S   = np.asarray(S,  dtype=complex)
    Zn  = np.asarray(Zn, dtype=complex)
    N   = S.shape[1]
    if S.shape[1:3] != (N, N):
        raise ValueError("S must have shape (M, N, N) with same N on both axes")
    if Zn.shape[1] != N:
        raise ValueError("Zn must be a length-N vector")

    # Constant matrices that do not depend on frequency
    
    W0_inv_sc = 1 / np.sqrt(Z0)               # scalar because Z0 is common
    I_N       = np.eye(N, dtype=complex)

    M = S.shape[0]
    S0 = np.empty_like(S)

    for k in range(M):
        Wref = np.diag(np.sqrt(Zn[k,:]))          # √Zn on the diagonal
        Sk = S[k, :, :]

        # Z  = Wref (I + S) (I – S)⁻¹ Wref
        Zk = Wref @ (I_N + Sk) @ np.linalg.inv(I_N - Sk) @ Wref

        # A  = W0⁻¹ Z W0⁻¹  → because W0 = √Z0·I → A = Z / Z0
        Ak = Zk * (W0_inv_sc ** 2)            # same as Zk / Z0

        # S0 = (A – I)(A + I)⁻¹
        S0[k, :, :] = (Ak - I_N) @ np.linalg.inv(Ak + I_N)

    return S0

def generate_ndim(
    outer_data: dict[str, list[float]],
    inner_data: list[float],
    outer_labels: tuple[str, ...]
) -> tuple[np.ndarray,...]:
    """
    Generates an N-dimensional grid of values from flattened data, and returns each axis array plus the grid.

    Parameters
    ----------
    outer_data : dict of {label: flat list of coordinates}
        Each key corresponds to one axis label, and the list contains coordinate values for each point.
    inner_data : list of float
        Flattened list of data values corresponding to each set of coordinates.
    outer_labels : tuple of str
        Order of axes (keys of outer_data) which defines the dimension order in the output array.

    Returns
    -------
    *axes : np.ndarray
        One 1D array for each axis, containing the sorted unique coordinates for that dimension, 
        in the order specified by outer_labels.
    grid : np.ndarray
        N-dimensional array of shape (n1, n2, ..., nN), where ni is the number of unique
        values along the i-th axis. Missing points are filled with np.nan.
    """
    # Convert inner data to numpy array
    values = np.asarray(inner_data)

    # Determine unique sorted coordinates for each axis
    axes = [np.unique(np.asarray(outer_data[label])) for label in outer_labels]
    grid_shape = tuple(axis.size for axis in axes)

    # Initialize grid with NaNs
    grid = np.full(grid_shape, np.nan, dtype=values.dtype)

    # Build coordinate arrays for each axis
    coords = [np.asarray(outer_data[label]) for label in outer_labels]

    # Map coordinates to indices in the grid for each axis
    idxs = [np.searchsorted(axes[i], coords[i]) for i in range(len(axes))]

    # Assign values into the grid
    grid[tuple(idxs)] = values

    # Return each axis array followed by the grid
    return (*axes, grid)

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
    port_number: int = -1
    k0: float | None= None
    beta: float | None = None
    Z0: float | complex | None = None
    Pout: float | None = None
    mode_number: int = 1

class MWData:
    scalar: BaseDataset[MWScalar, MWScalarNdim]
    field:   BaseDataset[MWField, None]

    def __init__(self):
        self.scalar = BaseDataset[MWScalar, MWScalarNdim](MWScalar, MWScalarNdim, True)
        self.field = BaseDataset[MWField, None](MWField, None, False)
        self.sim: DataContainer = DataContainer()

    def setreport(self, report, **vars):
        self.sim.new(**vars)['report'] = report
    
@dataclass
class FarFieldData:
    E: np.ndarray
    H: np.ndarray
    theta: np.ndarray
    phi: np.ndarray
    ang: np.ndarray | None = None

    @property
    def Ex(self) -> np.ndarray:
        return self.E[0,:]
    
    @property
    def Ey(self) -> np.ndarray:
        return self.E[1,:]
    
    @property
    def Ez(self) -> np.ndarray:
        return self.E[2,:]
    
    @property
    def Hx(self) -> np.ndarray:
        return self.H[0,:]
    
    @property
    def Hy(self) -> np.ndarray:
        return self.H[1,:]
    
    @property
    def Hz(self) -> np.ndarray:
        return self.H[2,:]
    
    @property
    def Etheta(self) -> np.ndarray:
        thx = -np.cos(self.theta)*np.cos(self.phi)
        thy = -np.cos(self.theta)*np.sin(self.phi)
        thz = np.sin(self.theta)
        return thx*self.E[0,:] + thy*self.E[1,:] + thz*self.E[2,:]
    
    @property
    def Ephi(self) -> np.ndarray:
        phx = -np.sin(self.phi)
        phy = np.cos(self.phi)
        phz = np.zeros_like(self.theta)
        return phx*self.E[0,:] + phy*self.E[1,:] + phz*self.E[2,:]
    
    @property
    def Erhcp(self) -> np.ndarray:
        return (self.Etheta - 1j*self.Ephi)/np.sqrt(2)
    
    @property
    def Elhcp(self) -> np.ndarray:
        return (self.Etheta + 1j*self.Ephi)/np.sqrt(2)
    
    @property
    def AR(self) -> np.ndarray:
        R = np.abs(self.Erhcp)
        L = np.abs(self.Elhcp)
        return (R+L)/(R-L)
    
    @property
    def normE(self) -> np.ndarray:
        return np.sqrt(np.abs(self.E[0,:])**2 + np.abs(self.E[1,:])**2 + np.abs(self.E[2,:])**2)
    
    @property
    def normH(self) -> np.ndarray:
        return np.sqrt(np.abs(self.H[0,:])**2 + np.abs(self.H[1,:])**2 + np.abs(self.H[2,:])**2)
    
    
    def surfplot(self, 
             polarization: Literal['Ex','Ey','Ez','Etheta','Ephi','normE','Erhcp','Elhcp','AR'],
             quantity: Literal['abs','real','imag','angle'] = 'abs',
             isotropic: bool = True, dB: bool = False, dBfloor: float = -30, rmax: float | None = None,
             offset: tuple[float, float, float] = (0,0,0)) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Returns the parameters to be used as positional arguments for the display.add_surf() function.

        Example:
        >>> model.display.add_surf(*dataset.field[n].farfield_3d(...).surfplot())

        Args:
            polarization ('Ex','Ey','Ez','Etheta','Ephi','normE'): What quantity to plot
            isotropic (bool, optional): Whether to look at the ratio with isotropic antennas. Defaults to True.
            dB (bool, optional): Whether to plot in dB's. Defaults to False.
            dBfloor (float, optional): The dB value to take as R=0. Defaults to -10.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: The X, Y, Z, F values
        """
        fmap = {
            'abs': np.abs,
            'real': np.real,
            'imag': np.imag,
            'angle': np.angle,
        }
        mapping = fmap.get(quantity.lower(),np.abs)
        
        F = mapping(self.__dict__.get(polarization, self.normE))
        
        if isotropic:
            F = F/np.sqrt(Z0/(2*np.pi))
        if dB:
            F = 20*np.log10(np.clip(np.abs(F), a_min=10**(dBfloor/20), a_max = 1e9))-dBfloor
        if rmax is not None:
            F = rmax * F/np.max(F)
        xs = F*np.sin(self.theta)*np.cos(self.phi) + offset[0]
        ys = F*np.sin(self.theta)*np.sin(self.phi) + offset[1]
        zs = F*np.cos(self.theta) + offset[2]

        return xs, ys, zs, F

@dataclass
class EHField:
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    Ex: np.ndarray
    Ey: np.ndarray
    Ez: np.ndarray
    Hx: np.ndarray
    Hy: np.ndarray
    Hz: np.ndarray
    freq: float
    er: np.ndarray
    ur: np.ndarray

    @property
    def k0(self) -> float:
        return self.freq*2*np.pi/299792458
    
    @property
    def Px(self) -> np.ndarray:
        return EPS0*(self.er-1)*self.Ex
    
    @property
    def Py(self) -> np.ndarray:
        return EPS0*(self.er-1)*self.Ey
    
    @property
    def Pz(self) -> np.ndarray:
        return EPS0*(self.er-1)*self.Ez
    
    @property
    def Dx(self) -> np.ndarray:
        return self.Ex*self.er
    
    @property
    def Dy(self) -> np.ndarray:
        return self.Ey*self.er
    
    @property
    def Dz(self) -> np.ndarray:
        return self.Ez*self.er
    
    @property
    def Bx(self) -> np.ndarray:
        return self.Hx/self.ur
    
    @property
    def By(self) -> np.ndarray:
        return self.Hy/self.ur
    
    @property
    def Bz(self) -> np.ndarray:
        return self.Hz/self.ur
    
    @property
    def Emat(self) -> np.ndarray:
        return np.array([self.Ex, self.Ey, self.Ez])
    
    @property
    def Hmat(self) -> np.ndarray:
        return np.array([self.Hx, self.Hy, self.Hz])
    
    @property
    def Pmat(self) -> np.ndarray:
        return np.array([self.Px, self.Py, self.Pz])
    
    @property
    def Bmat(self) -> np.ndarray:
        return np.array([self.Bx, self.By, self.Bz])
    
    @property
    def Dmat(self) -> np.ndarray:
        return np.array([self.Dx, self.Dy, self.Dz])
    
    @property
    def EH(self) -> tuple[np.ndarray, np.ndarray]:
        ''' Return the electric and magnetic field as a tuple of numpy arrays '''
        return np.array([self.Ex, self.Ey, self.Ez]), np.array([self.Hx, self.Hy, self.Hz])
    
    @property
    def E(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        ''' Return the electric field as a tuple of numpy arrays '''
        return self.Ex, self.Ey, self.Ez
    
    @property
    def Sx(self) -> np.ndarray:
        return self.Ey*self.Hz - self.Ez*self.Hy
    
    @property
    def Sy(self) -> np.ndarray:
        return self.Ez*self.Hx - self.Ex*self.Hz
    
    @property
    def Sz(self) -> np.ndarray:
        return self.Ex*self.Hy - self.Ey*self.Hx
    
    @property
    def B(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        ''' Return the magnetic field as a tuple of numpy arrays '''
        return self.Bx, self.By, self.Bz
    
    @property
    def P(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        ''' Return the polarization field as a tuple of numpy arrays '''
        return self.Px, self.Py, self.Pz

    @property
    def D(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        ''' Return the electric displacement field as a tuple of numpy arrays '''
        return self.Bx, self.By, self.Bz
    
    @property
    def H(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        ''' Return the magnetic field as a tuple of numpy arrays '''
        return self.Hx, self.Hy, self.Hz

    @property
    def S(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        ''' Return the poynting vector field as a tuple of numpy arrays '''
        return self.Sx, self.Sy, self.Sz
    
    @property
    def normE(self) -> np.ndarray:
        """The complex norm of the E-field
        """
        return np.sqrt(np.abs(self.Ex)**2 + np.abs(self.Ey)**2 + np.abs(self.Ez)**2)
    
    @property
    def normH(self) -> np.ndarray:
        """The complex norm of the H-field"""
        return np.sqrt(np.abs(self.Hx)**2 + np.abs(self.Hy)**2 + np.abs(self.Hz)**2)
    
    @property
    def normP(self) -> np.ndarray:
        """The complex norm of the P-field
        """
        return np.sqrt(np.abs(self.Px)**2 + np.abs(self.Py)**2 + np.abs(self.Pz)**2)
    
    @property
    def normB(self) -> np.ndarray:
        """The complex norm of the B-field
        """
        return np.sqrt(np.abs(self.Bx)**2 + np.abs(self.By)**2 + np.abs(self.Bz)**2)
    
    @property
    def normD(self) -> np.ndarray:
        """The complex norm of the D-field
        """
        return np.sqrt(np.abs(self.Dx)**2 + np.abs(self.Dy)**2 + np.abs(self.Dz)**2)
    
    @property
    def normS(self) -> np.ndarray:
        """The complex norm of the S-field
        """
        return np.sqrt(np.abs(self.Sx)**2 + np.abs(self.Sy)**2 + np.abs(self.Sz)**2)
    
    def vector(self, field: Literal['E','H'], metric: Literal['real','imag','complex'] = 'real') -> tuple[np.ndarray, np.ndarray,np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
        """Returns the X,Y,Z,Fx,Fy,Fz data to be directly cast into plot functions.

        The field can be selected by a string literal. The metric of the complex vector field by the metric.
        For animations, make sure to always use the complex metric.

        Args:
            field ('E','H'): The field to return
            metric ([]'real','imag','complex'], optional): the metric to impose on the field. Defaults to 'real'.

        Returns:
            tuple[np.ndarray,...]: The X,Y,Z,Fx,Fy,Fz arrays
        """
        Fx, Fy, Fz = getattr(self, field)
        
        if metric=='real':
            Fx, Fy, Fz = Fx.real, Fy.real, Fz.real
        elif metric=='imag':
            Fx, Fy, Fz = Fx.imag, Fy.imag, Fz.imag
        
        return self.x, self.y, self.z, Fx, Fy, Fz
    
    def scalar(self, field: Literal['Ex','Ey','Ez','Hx','Hy','Hz','normE','normH'], metric: Literal['abs','real','imag','complex'] = 'real') -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Returns the data X, Y, Z, Field based on the interpolation

        For animations, make sure to select the complex metric.

        Args:
            field (str): The field to plot
            metric (str, optional): The metric to impose on the plot. Defaults to 'real'.

        Returns:
            (X,Y,Z,Field): The coordinates plus field scalar
        """
        field_arry = getattr(self, field)
        if metric=='abs':
            field = np.abs(field_arry)
        elif metric=='real':
            field = field_arry.real
        elif metric=='imag':
            field = field_arry.imag
        elif metric=='complex':
            field = field_arry
        return self.x, self.y, self.z, field_arry

class _EHSign:
    """A small class to manage the sign of field components when computing the far-field with Stratton-Chu
    """
    def __init__(self):
        self.Ex = 1
        self.Ey = 1
        self.Ez = 1
        self.Hx = 1
        self.Hy = 1
        self.Hz = 1

    def fE(self):
        self.Ex = -1*self.Ex
        self.Ey = -1*self.Ey
        self.Ez = -1*self.Ez

    def fH(self):
        self.Hx = -1*self.Hx
        self.Hy = -1*self.Hy
        self.Hz = -1*self.Hz

    def fX(self):
        self.Ex = -1*self.Ex
        self.Hx = -1*self.Hx

    def fY(self):
        self.Ey = -1*self.Ey
        self.Hy = -1*self.Hy

    def fZ(self):
        self.Ez = -1*self.Ez
        self.Hz = -1*self.Hz

    def apply(self, symmetry: str):
        f, c = symmetry
        if f=='E':
            self.fE()
        elif f=='H':
            self.fH()

        if c=='x':
            self.fX()
        elif c=='y':
            self.fY()
        elif c=='z':
            self.fZ()
        
    def flip_field(self, E: tuple, H: tuple):
        Ex, Ey, Ez = E
        Hx, Hy, Hz = H
        return (Ex*self.Ex, Ey*self.Ey, Ez*self.Ez), (Hx*self.Hx, Hy*self.Hy, Hz*self.Hz)
    
class MWField:
    
    def __init__(self):
        self._der: np.ndarray = None
        self._dur: np.ndarray = None
        self.freq: float = None
        self.basis: FEMBasis = None
        self._fields: dict[int, np.ndarray] = dict()
        self._mode_field: np.ndarray = None
        self.excitation: dict[int, complex] = dict()
        self.Nports: int = None
        self.port_modes: list[PortProperties] = []
        self.Ex: np.ndarray = None
        self.Ey: np.ndarray = None
        self.Ez: np.ndarray = None
        self.Hx: np.ndarray = None
        self.Hy: np.ndarray = None
        self.Hz: np.ndarray = None
        self.er: np.ndarray = None
        self.ur: np.ndarray = None

    def add_port_properties(self, 
                            port_number: int,
                            mode_number: int,
                            k0: float,
                            beta: float,
                            Z0: float | complex | None,
                            Pout: float) -> None:
        self.port_modes.append(PortProperties(port_number=port_number,
                                              mode_number=mode_number,
                                              k0 = k0,
                                              beta=beta,
                                              Z0=Z0,
                                              Pout=Pout))
    
    @property
    def mesh(self) -> Mesh3D:
        return self.basis.mesh
    
    @property
    def k0(self) -> float:
        return self.freq*2*np.pi/299792458
    
    @property
    def _field(self) -> np.ndarray:
        if self._mode_field is not None:
            return self._mode_field
        return sum([self.excitation[mode.port_number]*self._fields[mode.port_number] for mode in self.port_modes]) # type: ignore
    
    def set_field_vector(self) -> None:
        """Defines the default excitation coefficients for the current dataset"""
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
    def normE(self) -> np.ndarray:
        """The complex norm of the E-field
        """
        return np.sqrt(np.abs(self.Ex)**2 + np.abs(self.Ey)**2 + np.abs(self.Ez)**2)
    
    @property
    def normH(self) -> np.ndarray:
        """The complex norm of the H-field"""
        return np.sqrt(np.abs(self.Hx)**2 + np.abs(self.Hy)**2 + np.abs(self.Hz)**2)
    
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
    
    def interpolate(self, xs: np.ndarray, ys: np.ndarray, zs: np.ndarray) -> EHField:
        ''' Interpolate the dataset in the provided xs, ys, zs values'''
        shp = xs.shape
        xf = xs.flatten()
        yf = ys.flatten()
        zf = zs.flatten()
        Ex, Ey, Ez = self.basis.interpolate(self._field, xf, yf, zf)
        self.Ex = Ex.reshape(shp)
        self.Ey = Ey.reshape(shp)
        self.Ez = Ez.reshape(shp)

        
        constants = 1/ (-1j*2*np.pi*self.freq*(self._dur*MU0) )
        Hx, Hy, Hz = self.basis.interpolate_curl(self._field, xf, yf, zf, constants)
        ids = self.basis.interpolate_index(xf, yf, zf)
        self.er = self._der[ids].reshape(shp)
        self.ur = self._dur[ids].reshape(shp)
        self.Hx = Hx.reshape(shp)
        self.Hy = Hy.reshape(shp)
        self.Hz = Hz.reshape(shp)

        self._x = xs
        self._y = ys
        self._z = zs
        return EHField(xs, ys, zs, self.Ex, self.Ey, self.Ez, self.Hx, self.Hy, self.Hz, self.freq, self.er, self.ur)

    def cutplane(self, 
                     ds: float,
                     x: float | None = None,
                     y: float | None = None,
                     z: float | None = None) -> EHField:
        """Create a cartesian cut plane (XY, YZ or XZ) and compute the E and H-fields there

        Only one coordiante and thus cutplane may be defined. If multiple are defined only the last (x->y->z) is used.
        
        Args:
            ds (float): The discretization step size
            x (float | None, optional): The X-coordinate in case of a YZ-plane. Defaults to None.
            y (float | None, optional): The Y-coordinate in case of an XZ-plane. Defaults to None.
            z (float | None, optional): The Z-coordinate in case of an XY-plane. Defaults to None.

        Returns:
            EHField: The resultant EHField object
        """
        xb, yb, zb = self.basis.bounds
        xs = np.linspace(xb[0], xb[1], int((xb[1]-xb[0])/ds))
        ys = np.linspace(yb[0], yb[1], int((yb[1]-yb[0])/ds))
        zs = np.linspace(zb[0], zb[1], int((zb[1]-zb[0])/ds))
        if x is not None:
            Y,Z = np.meshgrid(ys, zs)
            X = x*np.ones_like(Y)
        if y is not None:
            X,Z = np.meshgrid(xs, zs)
            Y = y*np.ones_like(X)
        if z is not None:
            X,Y = np.meshgrid(xs, ys)
            Z = z*np.ones_like(Y)
        return self.interpolate(X,Y,Z)
    def cutplane_normal(self,
             point=(0,0,0),
             normal=(0,0,1),
             npoints: int = 300) -> EHField:
        """
        Take a 2D slice of the field along an arbitrary plane.
        Args:
            point: (x0,y0,z0), a point on the plane
            normal: (nx,ny,nz), plane normal vector
            npoints: number of grid points per axis
        """

        n = np.array(normal, dtype=float)
        n /= np.linalg.norm(n)
        point = np.array(point) 

        tmp = np.array([1,0,0]) if abs(n[0]) < 0.9 else np.array([0,1,0])
        u = np.cross(n, tmp)
        u /= np.linalg.norm(u)
        v = np.cross(n, u)
        
        xb, yb, zb = self.basis.bounds
        nx, ny, nz = 5, 5, 5
        Xg = np.linspace(xb[0], xb[1], nx)
        Yg = np.linspace(yb[0], yb[1], ny)
        Zg = np.linspace(zb[0], zb[1], nz)
        Xg, Yg, Zg = np.meshgrid(Xg, Yg, Zg, indexing='ij')
        geometry = np.vstack([Xg.ravel(), Yg.ravel(), Zg.ravel()]).T  # Nx3
        
        rel_pts = geometry - point
        S = rel_pts @ u
        T = rel_pts @ v 
        
        margin = 0.01
        s_min, s_max = S.min(), S.max()
        t_min, t_max = T.min(), T.max()
        s_bounds = (s_min - margin*(s_max-s_min), s_max + margin*(s_max-s_min))
        t_bounds = (t_min - margin*(t_max-t_min), t_max + margin*(t_max-t_min))

        S_grid = np.linspace(s_bounds[0], s_bounds[1], npoints)
        T_grid = np.linspace(t_bounds[0], t_bounds[1], npoints)
        S_mesh, T_mesh = np.meshgrid(S_grid, T_grid)

        X = point[0] + S_mesh*u[0] + T_mesh*v[0]
        Y = point[1] + S_mesh*u[1] + T_mesh*v[1]
        Z = point[2] + S_mesh*u[2] + T_mesh*v[2]

        return self.interpolate(X, Y, Z)
    
    
    def grid(self, ds: float) -> EHField:
        """Interpolate a uniform grid sampled at ds

        Args:
            ds (float): the sampling distance

        Returns:
            This object
        """
        xb, yb, zb = self.basis.bounds
        xs = np.linspace(xb[0], xb[1], int((xb[1]-xb[0])/ds))
        ys = np.linspace(yb[0], yb[1], int((yb[1]-yb[0])/ds))
        zs = np.linspace(zb[0], zb[1], int((zb[1]-zb[0])/ds))
        X, Y, Z = np.meshgrid(xs, ys, zs)
        return self.interpolate(X,Y,Z)
    
    def vector(self, field: Literal['E','H'], metric: Literal['real','imag','complex'] = 'real') -> tuple[np.ndarray, np.ndarray,np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
        """Returns the X,Y,Z,Fx,Fy,Fz data to be directly cast into plot functions.

        The field can be selected by a string literal. The metric of the complex vector field by the metric.
        For animations, make sure to always use the complex metric.

        Args:
            field ('E','H'): The field to return
            metric ([]'real','imag','complex'], optional): the metric to impose on the field. Defaults to 'real'.

        Returns:
            tuple[np.ndarray,...]: The X,Y,Z,Fx,Fy,Fz arrays
        """
        if field=='E':
            Fx, Fy, Fz = self.Ex, self.Ey, self.Ez
        elif field=='H':
            Fx, Fy, Fz = self.Hx, self.Hy, self.Hz
        
        if metric=='real':
            Fx, Fy, Fz = Fx.real, Fy.real, Fz.real
        elif metric=='imag':
            Fx, Fy, Fz = Fx.imag, Fy.imag, Fz.imag
        
        return self._x, self._y, self._z, Fx, Fy, Fz
    
    def scalar(self, field: Literal['Ex','Ey','Ez','Hx','Hy','Hz','normE','normH'], metric: Literal['abs','real','imag','complex'] = 'real') -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Returns the data X, Y, Z, Field based on the interpolation

        For animations, make sure to select the complex metric.

        Args:
            field (str): The field to plot
            metric (str, optional): The metric to impose on the plot. Defaults to 'real'.

        Returns:
            (X,Y,Z,Field): The coordinates plus field scalar
        """
        field = getattr(self, field)
        if metric=='abs':
            field = np.abs(field)
        elif metric=='real':
            field = field.real
        elif metric=='imag':
            field = field.imag
        elif metric=='complex':
            field = field
        return self._x, self._y, self._z, field
    
    def farfield_2d(self,ref_direction: tuple[float,float,float] | Axis,
                         plane_normal: tuple[float,float,float] | Axis,
                         faces: FaceSelection | GeoSurface,
                         ang_range: tuple[float, float] = (-180, 180),
                         Npoints: int = 201,
                         origin: tuple[float, float, float] | None = None,
                         syms: list[Literal['Ex','Ey','Ez', 'Hx','Hy','Hz']] | None = None) -> FarFieldData:#tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute the farfield electric and magnetic field defined by a circle.

        Args:
            ref_direction (tuple[float,float,float] | Axis): The direction for angle=0
            plane_normal (tuple[float,float,float] | Axis): The rotation axis of the angular cutplane
            faces (FaceSelection | GeoSurface): The faces to integrate over
            ang_range (tuple[float, float], optional): The angular rage limits. Defaults to (-180, 180).
            Npoints (int, optional): The number of angular points. Defaults to 201.
            origin (tuple[float, float, float], optional): The farfield origin. Defaults to (0,0,0).
            syms (list[Literal['Ex','Ey','Ez','Hx','Hy','Hz']], optional): E and H-plane symmetry planes where Ex is E-symmetry in x=0. Defaults to []

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: Angles (N,), E(3,N), H(3,N)
        """
        refdir = _parse_axis(ref_direction).np
        plane_normal_parsed = _parse_axis(plane_normal).np
        theta, phi = arc_on_plane(refdir, plane_normal_parsed, ang_range, Npoints)
        E,H = self.farfield(theta, phi, faces, origin, syms = syms)
        angs = np.linspace(*ang_range, Npoints)*np.pi/180
        return FarFieldData(E, H, theta, phi, ang=angs)

    def farfield_3d(self, 
                    faces: FaceSelection | GeoSurface,
                    thetas: np.ndarray | None = None,
                    phis: np.ndarray | None = None,
                    origin: tuple[float, float, float] | None = None,
                    syms: list[Literal['Ex','Ey','Ez', 'Hx','Hy','Hz']] | None = None) -> FarFieldData:
        """Compute the farfield in a 3D angular grid

        If thetas and phis are not provided, they default to a sample space of 2 degrees.

        Args:
            faces (FaceSelection | GeoSurface): The integration faces
            thetas (np.ndarray, optional): The 1D array of theta values. Defaults to None.
            phis (np.ndarray, optional): A 1D array of phi values. Defaults to None.
            origin (tuple[float, float, float], optional): The boundary normal alignment origin. Defaults to (0,0,0).
            syms (list[Literal['Ex','Ey','Ez','Hx','Hy','Hz']], optional): E and H-plane symmetry planes where Ex is E-symmetry in x=0. Defaults to []
        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: The 2D theta, phi, E and H matrices.
        """
        if thetas is None:
            thetas = np.linspace(0,np.pi, 91)
        if phis is None:
            phis = np.linspace(-np.pi, np.pi, 181)

        T,P = np.meshgrid(thetas, phis)

        E, H = self.farfield(T.flatten(), P.flatten(), faces, origin, syms=syms)
        E = E.reshape((3, ) + T.shape)
        H = H.reshape((3, ) + T.shape)
        
        return FarFieldData(E, H, T, P)

    def farfield(self, theta: np.ndarray,
                 phi: np.ndarray,
                 faces: FaceSelection | GeoSurface,
                 origin: tuple[float, float, float] | None = None,
                 syms: list[Literal['Ex','Ey','Ez', 'Hx','Hy','Hz']] | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Compute the farfield at the provided theta/phi coordinates

        Args:
            theta (np.ndarray): The Theta coordinates as (N,) 1D Array
            phi (np.ndarray): The Phi coordinates as (N,) 1D Array
            faces (FaceSelection | GeoSurface): the faces to use as integration boundary
            origin (tuple[float, float, float], optional): The surface normal origin. Defaults to (0,0,0).
            syms (list[Literal['Ex','Ey','Ez','Hx','Hy','Hz']], optional): E and H-plane symmetry planes where Ex is E-symmetry in x=0. Defaults to []

        Returns:
            tuple[np.ndarray, np.ndarray]: The E and H field as (3,N) arrays
        """
        if syms is None:
            syms = []

        from .sc import stratton_chu
        surface = self.basis.mesh.boundary_surface(faces.tags, origin)
        field = self.interpolate(*surface.exyz)
        
        Eff,Hff = stratton_chu(field.E, field.H, surface, theta, phi, self.k0)
        
        if len(syms)==0:
            return Eff, Hff

        if len(syms)==1:
            perms = ((syms[0], '##', '##'),)
        elif len(syms)==2:
            s1, s2 = syms
            perms = ((s1, '##', '##'), (s2, '##', '##'), (s1, s2, '##'))
        elif len(syms)==3:
            s1, s2, s3 = syms
            perms = ((s1, '##', '##'), (s2, '##', '##'), (s3, '##', '##'), (s1, s2, '##'), (s1, s3, '##'), (s2, s3, '##'), (s1, s2, s3))
        
        for s1, s2, s3 in perms:
            surf = surface.copy()
            ehf = _EHSign()
            ehf.apply(s1)
            ehf.apply(s2)
            ehf.apply(s3)
            Ef, Hf = ehf.flip_field(field.E, field.H)
            surf.flip(s1[1])
            surf.flip(s2[1])
            surf.flip(s3[1])
            E2, H2 = stratton_chu(Ef, Hf, surf, theta, phi, self.k0)
            Eff = Eff + E2
            Hff = Hff + H2
        
        return Eff, Hff

    def optycal_surface(self, faces: FaceSelection | GeoSurface | None = None) -> tuple:
        """Export this models exterior to an Optical acceptable dataset

        Args:
            faces (FaceSelection | GeoSurface): The faces to export. Defaults to None

        Returns:
            tuple: _description_
        """
        if faces is None:
            tags = self.mesh.exterior_face_tags
        else:
            tags = faces.tags

        center = np.mean(self.mesh.nodes, axis=1).squeeze()
        surface = self.basis.mesh.boundary_surface(tags, center)
        field = self.interpolate(*surface.exyz)
        vertices = surface.nodes
        triangles = surface.tris
        origin = surface._origin
        E = field.E
        H = field.H
        k0 = self.k0
        return vertices, triangles, E, H, origin, k0
    
    def optycal_antenna(self, faces: FaceSelection | GeoSurface | None = None,
                        origin: tuple[float, float, float] | None = None,
                        syms: list[Literal['Ex','Ey','Ez', 'Hx','Hy','Hz']] | None = None) -> dict:
        """Export this models exterior to an Optical acceptable dataset

        Args:
            faces (FaceSelection | GeoSurface): The faces to export. Defaults to None

        Returns:
            tuple: _description_
        """
        freq = self.freq
        def function(theta: np.ndarray, phi: np.ndarray, k0: float):
            E, H = self.farfield(theta, phi, faces, origin, syms)
            return E[0,:], E[1,:], E[2,:], H[0,:], H[1,:], H[2,:]
    
        return dict(freq=freq, ff_function=function)

class MWScalar:
    """The MWDataSet class stores solution data of FEM Time Harmonic simulations.
    """
    _fields: list[str] = ['freq','k0','Sp','beta','Pout','Z0']
    _copy: list[str] = ['_portmap','_portnumbers','port_modes']

    def __init__(self):
        self.freq: float = None
        self.k0: float = None
        self.Sp: np.ndarray = None
        self.beta: np.ndarray = None
        self.Z0: np.ndarray = None
        self.Pout: np.ndarray = None
        self._portmap: dict[int, float|int] = dict()
        self._portnumbers: list[int | float] = []
        self.port_modes: list[PortProperties] = []

    def init_sp(self, portnumbers: list[int | float]) -> None:
        """Initialize the S-parameter dataset with the given number of ports."""
        self._portnumbers = portnumbers
        i = 0
        for n in portnumbers:
            self._portmap[n] = i
            i += 1

        self.Sp = np.zeros((i,i), dtype=np.complex128)
        self.Z0 = np.zeros((i,), dtype=np.complex128)
        self.Pout = np.zeros((i,), dtype=np.float64)
        self.beta = np.zeros((i,), dtype=np.complex128)

        
    def write_S(self, i1: int | float, i2: int | float, value: complex) -> None:
        self.Sp[self._portmap[i1], self._portmap[i2]] = value

    def S(self, i1: int, i2: int) -> complex:
        """Return the S-parameter corresponding to the given set of indices:

        S11 = obj.S(1,1)

        Args:
            i1 (int): The first port index
            i2 (int): The second port index

        Returns:
            complex: The S-parameter
        """
        return self.Sp[self._portmap[i1], self._portmap[i2]]
    
    def add_port_properties(self, 
                            port_number: int,
                            mode_number: int,
                            k0: float,
                            beta: float,
                            Z0: float | complex,
                            Pout: float) -> None:
        i = self._portmap[port_number]
        self.beta[i] = beta
        self.Z0[i] = Z0
        self.Pout[i] = Pout
    
class MWScalarNdim:
    _fields: list[str] = ['freq','k0','Sp','beta','Pout','Z0']
    _copy: list[str] = ['_portmap','_portnumbers']

    def __init__(self):
        self.freq: np.ndarray = None
        self.k0: np.ndarray = None
        self.Sp: np.ndarray = None
        self.beta: np.ndarray = None
        self.Z0: np.ndarray = None
        self.Pout: np.ndarray = None
        self._portmap: dict[int, float|int] = dict()
        self._portnumbers: list[int | float] = []

    def S(self, i1: int, i2: int) -> np.ndarray:
        return self.Sp[...,self._portmap[i1], self._portmap[i2]]
    
    @property
    def Smat(self) -> np.ndarray:
        """Returns the full S-matrix

        Returns:
            np.ndarray: The S-matrix with shape (nF, nP, nP)
        """
        Nports = len(self._portmap)
        nfreq = self.freq.shape[0]

        Smat = np.zeros((nfreq,Nports,Nports), dtype=np.complex128)
        
        for i in self._portnumbers:
            for j in self._portnumbers:
                Smat[:,i-1,j-1] = self.S(i,j)

        return Smat
    
    def emmodel(self, f_sample: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Returns the required date for a Heavi S-parameter component

        Returns:
            tuple[np.ndarray, np.ndarray]: Heavi data
        """
        
        if f_sample is not None:
            f = f_sample
            S = self.model_Smat(f_sample)
        else:
            f = self.freq
            S = self.Smat
        
        Z0s = self.Z0
        S = renormalise_s(S, Z0s, 50.0)
        return f, S

    def model_S(self, i: int, j: int, 
            freq: np.ndarray, 
            Npoles: int | Literal['auto'] = 'auto', 
            inc_real: bool = False,
            maxpoles: int = 30) -> np.ndarray:
        """Returns an S-parameter model object at a dense frequency range.
        This method uses vector fitting inside the datasets frequency points to determine a model for the linear system.

        Args:
            i (int): The first S-parameter index
            j (int): The second S-parameter index
            freq (np.ndarray): The frequency sample points
            Npoles (int | 'auto', optional): The number of poles to use (approx 2x divice order). Defaults to 10.
            inc_real (bool, optional): Wether to allow for a real-pole. Defaults to False.

        Returns:
            SparamModel: The SparamModel object
        """
        return SparamModel(self.freq, self.S(i,j), n_poles=Npoles, inc_real=inc_real, maxpoles=maxpoles)(freq)

    def model_Smat(self, frequencies: np.ndarray,
                        Npoles: int = 10,
                        inc_real: bool = False) -> np.ndarray:
        """Generates a full S-parameter matrix on the provided frequency points using the Vector Fitting algorithm.

        This function output can be used directly with the .save_matrix() method.

        Args:
            frequencies (np.ndarray): The sample frequencies
            Npoles (int, optional): The number of poles to fit. Defaults to 10.
            inc_real (bool, optional): Wether allow for a real pole. Defaults to False.

        Returns:
            np.ndarray: The (Nf,Np,Np) S-parameter matrix
        """
        Nports = len(self._portmap)
        nfreq = frequencies.shape[0]

        Smat = np.zeros((nfreq,Nports,Nports), dtype=np.complex128)
        
        for i in self._portnumbers:
            for j in self._portnumbers:
                S = self.model_S(i,j,frequencies, Npoles=Npoles, inc_real=inc_real)
                Smat[:,i-1,j-1] = S

        return Smat

    def export_touchstone(self, 
                            filename: str,
                            Z0ref: float | None = None,
                            format: Literal['RI','MA','DB'] = 'RI',
                            custom_comments: list[str] | None = None,
                            funit: Literal['HZ','KHZ','MHZ','GHZ'] = 'GHZ'):
        """Export the S-parameter data to a touchstone file

        This function assumes that all ports are numbered in sequence 1,2,3,4... etc with
        no missing ports. Otherwise it crashes. Will be update/improved soon with more features.

        Additionally, one may provide a reference impedance. If this argument is provided, a port impedance renormalization
        will be performed to that common impedance.

        Args:
            filename (str): The File name
            Z0ref (float): The reference impedance to normalize to. Defaults to None
            format (Literal[DB, RI, MA]): The dataformat used in the touchstone file.
            custom_comments : list[str], optional. List of custom comment strings to add to the touchstone file header.
                                                    Each string will be prefixed with "! " automatically.
        """
        
        logger.info(f'Exporting S-data to {filename}')
        Nports = len(self._portmap)
        freqs = self.freq

        Smat = np.zeros((len(freqs),Nports,Nports), dtype=np.complex128)
        
        for i in range(1,Nports+1):
            for j in range(1,Nports+1):
                S = self.S(i,j)
                Smat[:,i-1,j-1] = S
        
        self.save_smatrix(filename, Smat, freqs, format=format, Z0ref=Z0ref, custom_comments=custom_comments, funit=funit)

    def save_smatrix(self, 
                        filename: str,
                        Smatrix: np.ndarray,
                        frequencies: np.ndarray, 
                        Z0ref: float | None = None,
                        format: Literal['RI','MA','DB'] = 'RI',
                        custom_comments: list[str] | None = None,
                        funit: Literal['HZ','KHZ','MHZ','GHZ'] = 'GHZ') -> None:
        """Save an S-parameter matrix to a touchstone file.
        
        Additionally, a reference impedance may be supplied. In this case, a port renormalization will be performed on the S-matrix.

        Args:
            filename (str): The filename
            Smatrix (np.ndarray): The S-parameter matrix with shape (Nfreq, Nport, Nport)
            frequencies (np.ndarray): The frequencies with size (Nfreq,)
            Z0ref (float, optional): An optional reference impedance to normalize to. Defaults to None.
            format (Literal["RI","MA",'DB], optional): The S-parameter format. Defaults to 'RI'.
            custom_comments : list[str], optional. List of custom comment strings to add to the touchstone file header.
                                                    Each string will be prefixed with "! " automatically.
        """
        from .touchstone import generate_touchstone

        if Z0ref is not None:
            Z0s = self.Z0
            logger.debug(f'Renormalizing impedances {Z0s}Ω to {Z0ref}Ω')
            Smatrix = renormalise_s(Smatrix, Z0s, Z0ref)


        generate_touchstone(filename, frequencies, Smatrix, format, custom_comments, funit)
        
        logger.info('Export complete!')
        
# class MWSimData(SimData[MWConstants]):
#     """The MWSimData class contains all EM simulation data from a Time Harmonic simulation
#     along all sweep axes.
#     """
#     datatype: type = MWConstants
#     def __init__(self):
#         super().__init__()
#         self._injections = dict()
#         self._axis = 'freq'

#     def __getitem__(self, field: EMField) -> np.ndarray:
#         return getattr(self, field)

#     @property
#     def mesh(self) -> Mesh3D:
#         """Returns the relevant mesh object for this dataset assuming they are all the same.

#         Returns:
#             Mesh3D: The mesh object.
#         """
#         return self.datasets[0].basis.mesh
    
#     def howto(self) -> None:
#         """To access data in the MWSimData class use the .ax method to extract properties selected
#         along an access of global variables. The axes are all global properties that the MWDataSets manage.
        
#         For example the following would return all S(2,1) parameters along the frequency axis.
        
#         >>> freq, S21 = dataset.ax('freq').S(2,1)

#         Alternatively, one can manually select any solution indexed in order of generation using.

#         >>> S21 = dataset.item(3).S(2,1)

#         To find the E or H fields at any coordinate, one can use the Dataset's .interpolate method. 
#         This method returns the same Dataset object after which the computed fields can be accessed.

#         >>> Ex = dataset.item(3).interpolate(xs,ys,zs).Ex

#         Lastly, to find the solutions for a given frequency or other value, you can also just call the dataset
#         class:
        
#         >>> Ex, Ey, Ez = dataset(freq=1e9).interpolate(xs,ys,zs).E

#         """

#     def select(self, **axes: EMField) -> MWSimData:
#         """Takes the provided axis points and constructs a new dataset only for those axes values

#         Returns:
#             MWSimData: The new dataset
#         """
#         newdata = MWSimData()
#         for dataset in self.datasets:
#             if dataset.equals(**axes):
#                 newdata.datasets.append(dataset)
#         return newdata
    
#     def ax(self, *field: EMField) -> MWConstants:
#         """Return a MWDataSet proxy object that you can request properties for along a provided axis.

#         The MWSimData class contains a list of MWDataSet objects. Any global variable like .freq of the 
#         MWDataSet object can be used as inner-axes after which the outer axis can be selected as if
#         you are extract a single one.

#         Args:
#             field (EMField): The global field variable to select the data along

#         Returns:
#             MWDataSet: An MWDataSet object (actually a proxy for)

#         Example:
#         The following will select all S11 parameters along the frequency axis:

#         >>> freq, S11 = dataset.ax('freq').S(1,1)

#         """
#         # find the real DataSet
#         return _DataSetProxy(field, self.datasets)

