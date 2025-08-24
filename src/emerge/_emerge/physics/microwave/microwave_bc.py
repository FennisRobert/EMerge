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
from loguru import logger
from typing import Callable, Literal
from ...selection import Selection, FaceSelection
from ...cs import CoordinateSystem, Axis, GCS, _parse_axis
from ...coord import Line
from ...geometry import GeoSurface, GeoObject
from dataclasses import dataclass
from collections import defaultdict
from ...bc import BoundaryCondition, BoundaryConditionSet, Periodic
from ...periodic import PeriodicCell, HexCell, RectCell
from ...material import Material
from ...const import Z0, C0, EPS0, MU0



############################################################
#                     UTILITY FUNCTIONS                    #
############################################################

def _inner_product(function: Callable, x: np.ndarray, y: np.ndarray, z: np.ndarray, ax: Axis) -> float:
            Exyz = function(x,y,z)
            return np.sum(Exyz[0,:]*ax.x + Exyz[1,:]*ax.y + Exyz[2,:]*ax.z)
        
        

############################################################
#                   MAIN BC MANAGER CLASS                  #
############################################################

class MWBoundaryConditionSet(BoundaryConditionSet):

    def __init__(self, periodic_cell: PeriodicCell | None):
        super().__init__()

        self.PEC: type[PEC] = self._construct_bc(PEC)
        self.PMC: type[PMC] = self._construct_bc(PMC)
        self.AbsorbingBoundary: type[AbsorbingBoundary] = self._construct_bc(AbsorbingBoundary)
        self.ModalPort: type[ModalPort] = self._construct_bc(ModalPort)
        self.LumpedPort: type[LumpedPort] = self._construct_bc(LumpedPort)
        self.LumpedElement: type[LumpedElement] = self._construct_bc(LumpedElement)
        self.SurfaceImpedance: type[SurfaceImpedance] = self._construct_bc(SurfaceImpedance)
        self.RectangularWaveguide: type[RectangularWaveguide] = self._construct_bc(RectangularWaveguide)
        self.Periodic: type[Periodic] = self._construct_bc(Periodic)
        self.FloquetPort: type[FloquetPort] = self._construct_bc(FloquetPort)

        self._cell: PeriodicCell | None = None

    def get_conductors(self) -> list[BoundaryCondition]:
        """Returns a list of all boundary conditions that ought to be considered as a "conductor" 
        for the purpose of modal analyses.

        Returns:
            list[BoundaryCondition]: All conductor like boundary conditions
        """
        bcs = self.oftype(PEC)
        for bc in self.oftype(SurfaceImpedance):
            if bc.material.cond > 1e3:
                bcs.append(bc)

        return bcs

    def get_type(self, bctype: Literal['PEC','ModalPort','LumpedPort','PMC','LumpedElement','RectangularWaveguide','Periodic','FloquetPort','SurfaceImpedance']) -> FaceSelection:
        tags = []
        for bc in self.boundary_conditions:
            if bctype in str(bc.__class__):
                tags.extend(bc.selection.tags)
        return FaceSelection(tags)

    def floquet_port(self, poly: GeoSurface, port_number: int) -> FloquetPort:
        if self._cell is None:
            raise ValueError('Periodic cel must be defined for this simulation.')
        if isinstance(self._cell, RectCell):
            port = self.FloquetPort(poly, port_number)
            port.width = self._cell.width
            port.height = self._cell.height
        elif isinstance(self._cell, HexCell):
            port = self.FloquetPort(poly, port_number)
            port.area = 1.0
        self._cell._ports.append(port)
        return port


############################################################
#                    BOUNDARY CONDITIONS                   #
############################################################


class PEC(BoundaryCondition):
    
    def __init__(self,
                 face: FaceSelection | GeoSurface):
        """The general perfect electric conductor boundary condition.

        The physics compiler will by default always turn all exterior faces into a PEC.

        Args:
            face (FaceSelection | GeoSurface): The boundary surface
        """
        super().__init__(face)

class PMC(BoundaryCondition):
    pass

class RobinBC(BoundaryCondition):
    
    _include_stiff: bool = False
    _include_mass: bool = False
    _include_force: bool = False

    def __init__(self, selection: GeoSurface | Selection):
        """A Generalization of any boundary condition of the third kind (Robin).

        This should not be created directly. A robin boundary condition is the generalized type behind
        port boundaries, radiation boundaries etc. Since all boundary conditions of the thrid kind (Robin)
        are assembled the same, this class is used during assembly.

        Args:
            selection (GeoSurface | Selection): The boundary surface.
        """
        super().__init__(selection)
        self.v_integration: bool = False
        self.vintline: Line | None = None
    
    def get_basis(self) -> np.ndarray:
        raise NotImplementedError('This method is not implemented')
    
    def get_inv_basis(self) -> np.ndarray | None :
        raise NotImplementedError('This method is not implemented')
    
    def get_beta(self, k0: float) -> float:
        raise NotImplementedError('get_beta not implemented for Port class')
    
    def get_gamma(self, k0: float) -> complex:
        raise NotImplementedError('get_gamma not implemented for Port class')
    
    def get_Uinc(self, x_local: np.ndarray, y_local: np.ndarray, k0: float) -> np.ndarray:
        raise NotImplementedError('get_Uinc not implemented for Port class')

class PortBC(RobinBC):
    Zvac: float = Z0
    def __init__(self, face: FaceSelection | GeoSurface):
        """(DO NOT USE) A generalization of the Port boundary condition.
        
        DO NOT USE THIS TO DEFINE PORTS. This class is only indeded for 
        class inheritance and type checking. 

        Args:
            face (FaceSelection | GeoSurface): The port face
        """
        super().__init__(face)
        self.port_number: int = -1
        self.cs: CoordinateSystem = GCS
        self.selected_mode: int = 0
        self.Z0: complex | float | None = None
        self.active: bool | None = False
        self.power: float = 1.0

    @property
    def voltage(self) -> complex | None:
        return None
        
    def get_basis(self) -> np.ndarray:
        return self.cs._basis
    
    def get_inv_basis(self) -> np.ndarray:
        return self.cs._basis_inv
    
    def portZ0(self, k0: float) -> complex | float | None:
        """Returns the port characteristic impedance given a phase constant

        Args:
            k0 (float): The phase constant

        Returns:
            complex: The port impedance
        """
        return self.Z0

    def modetype(self, k0: float) -> Literal['TEM','TE','TM']:
        return 'TEM'
    
    def Zmode(self, k0: float) -> float:
        if self.modetype(k0)=='TEM':
            return self.Zvac
        elif self.modetype(k0)=='TE':
            return k0*299792458/self.get_beta(k0) * MU0
        elif self.modetype(k0)=='TM':
            return self.get_beta(k0)/(k0*299792458*EPS0)
        else:
            raise ValueError(f'Port mode type should be TEM, TE or TM but instead is {self.modetype(k0)}')
    
    def _qmode(self, k0: float) -> float:
        """Computes a mode amplitude correction factor.
        The total output power of a port as a function of the field amplitude is not constant.
        For TE and TM modes the output power depends on the mode impedance. This factor corrects
        the mode output power to 1W by scaling the E-field appropriately.

        Args:
            k0 (float): The phase constant of the simulation

        Returns:
            float: The mode amplitude correction factor.
        """
        return np.sqrt(self.Zmode(k0)/Z0)
    
    @property
    def mode_number(self) -> int:
        return self.selected_mode + 1

    def get_beta(self, k0) -> float:
        ''' Return the out of plane propagation constant. βz.'''
        return k0
    
    def get_gamma(self, k0) -> complex:
        """Computes the γ-constant for matrix assembly. This constant is required for the Robin boundary condition.

        Args:
            k0 (float): The free space propagation constant.

        Returns:
            complex: The γ-constant
        """
        return 1j*self.get_beta(k0)
    
    def port_mode_3d(self, 
                     xs: np.ndarray,
                     ys: np.ndarray,
                     k0: float,
                     which: Literal['E','H'] = 'E') -> np.ndarray:
        raise NotImplementedError('port_mode_3d not implemented for Port class')
    
    def port_mode_3d_global(self, 
                                x_global: np.ndarray,
                                y_global: np.ndarray,
                                z_global: np.ndarray,
                                k0: float,
                                which: Literal['E','H'] = 'E') -> np.ndarray:
            xl, yl, _ = self.cs.in_local_cs(x_global, y_global, z_global)
            Ex, Ey, Ez = self.port_mode_3d(xl, yl, k0)
            Exg, Eyg, Ezg = self.cs.in_global_basis(Ex, Ey, Ez)
            return np.array([Exg, Eyg, Ezg])

class AbsorbingBoundary(RobinBC):

    _include_stiff: bool = True
    _include_mass: bool = True
    _include_force: bool = False

    def __init__(self,
                 face: FaceSelection | GeoSurface,
                 order: int = 1,
                 origin: tuple | None = None):
        """Creates an AbsorbingBoundary condition.

        Currently only a first order boundary condition is possible. Second order will be supported later.
        The absorbing boundary is effectively a port boundary condition (Robin) with an assumption on
        the out-of-plane phase constant. For now it always assumes the free-space propagation (normal).

        Args:
            face (FaceSelection | GeoSurface): The absorbing boundary face(s)
            order (int, optional): The order (only 1 is supported). Defaults to 1.
            origin (tuple, optional): The radiation origin. Defaults to None.
        """
        super().__init__(face)
        if origin is None:
            origin = (0., 0., 0.)
        self.order: int = order
        self.origin: tuple = origin
        self.cs: CoordinateSystem = GCS

    def get_basis(self) -> np.ndarray:
        return np.eye(3)

    def get_inv_basis(self) -> np.ndarray | None:
        return None
    
    def get_beta(self, k0: float) -> float:
        ''' Return the out of plane propagation constant. βz.'''
        return k0

    def get_gamma(self, k0: float) -> complex:
        """Computes the γ-constant for matrix assembly. This constant is required for the Robin boundary condition.

        Args:
            k0 (float): The free space propagation constant.

        Returns:
            complex: The γ-constant
        """
        return 1j*self.get_beta(k0)
    
    def get_Uinc(self, x_local: np.ndarray, y_local: np.ndarray, k0: float) -> np.ndarray:
        return np.zeros((3, len(x_local)), dtype=np.complex128)

@dataclass
class PortMode:
    modefield: np.ndarray
    E_function: Callable
    H_function: Callable
    k0: float
    beta: float
    residual: float
    energy: float = 0
    norm_factor: float = 1
    freq: float = 0
    neff: float = 1
    TEM: bool = True
    Z0: float = 50.0
    polarity: float = 1.0
    modetype: Literal['TEM','TE','TM'] = 'TEM'

    def __post_init__(self):
        self.neff = self.beta/self.k0
        self.energy = np.mean(np.abs(self.modefield)**2)

    def __str__(self):
        return f'PortMode(k0={self.k0}, beta={self.beta}({self.neff:.3f}))'
    
    def set_power(self, power: complex) -> None:
        self.norm_factor = np.sqrt(1/np.abs(power))
        logger.info(f'Setting port mode amplitude to: {self.norm_factor:.2f} ')

class FloquetPort(PortBC):
    _include_stiff: bool = True
    _include_mass: bool = False
    _include_force: bool = True

    def __init__(self,
                 face: FaceSelection | GeoSurface,
                 port_number: int,
                 cs: CoordinateSystem |  None = None,
                 power: float = 1.0,
                 er: float = 1.0):
        super().__init__(face)
        if cs is None:
            cs = GCS
        self.port_number: int= port_number
        self.active: bool = True
        self.power: float = power
        self.type: str = 'TEM'
        self.mode: tuple[int,int] = (1,0)
        self.cs: CoordinateSystem = cs
        self.scan_theta: float = 0
        self.scan_phi: float = 0
        self.pol_s: complex = 1.0 + 0j
        self.pol_p: complex = 0j
        self.area: float = 1
        self.width: float | None = None
        self.height: float | None = None

        if self.cs is None:
            self.cs = GCS

    def portZ0(self, k0: float | None = None) -> complex | float | None:
        return Z0

    def get_amplitude(self, k0: float) -> float:
        return 1.0
    
    def get_beta(self, k0: float) -> float:
        ''' Return the out of plane propagation constant. βz.'''
        return k0*np.cos(self.scan_theta)
    
    def get_gamma(self, k0: float) -> complex:
        """Computes the γ-constant for matrix assembly. This constant is required for the Robin boundary condition.

        Args:
            k0 (float): The free space propagation constant.

        Returns:
            complex: The γ-constant
        """
        return 1j*self.get_beta(k0)
    
    def get_Uinc(self, x_global: np.ndarray, y_global: np.ndarray, z_global: np.ndarray, k0: float) -> np.ndarray:
        return -2*1j*self.get_beta(k0)*self.port_mode_3d_global(x_global, y_global, z_global, k0)
    
    def port_mode_3d(self, 
                     x_local: np.ndarray,
                     y_local: np.ndarray,
                     k0: float,
                     which: Literal['E','H'] = 'E') -> np.ndarray:
        ''' Compute the port mode E-field in local coordinates (XY) + Z out of plane.'''

        kx = k0*np.sin(self.scan_theta)*np.cos(self.scan_phi)
        ky = k0*np.sin(self.scan_theta)*np.sin(self.scan_phi)
        kz = k0*np.cos(self.scan_theta)
        phi = np.exp(-1j*(x_local*kx + y_local*ky))

        P = self.pol_p
        S = self.pol_s

        E0 = self.get_amplitude(k0)*np.sqrt(2*Z0/(self.area))
        Ex = E0*(-S*np.sin(self.scan_phi) - P*np.cos(self.scan_theta)*np.cos(self.scan_phi))*phi
        Ey = E0*(S*np.cos(self.scan_phi) - P*np.cos(self.scan_theta)*np.sin(self.scan_phi))*phi
        Ez = E0*(-P*E0*np.sin(self.scan_theta))*phi
        Exyz = np.array([Ex, Ey, Ez])
        return Exyz

    def port_mode_3d_global(self, 
                            x_global: np.ndarray,
                            y_global: np.ndarray,
                            z_global: np.ndarray,
                            k0: float,
                            which: Literal['E','H'] = 'E') -> np.ndarray:
        '''Compute the port mode field for global xyz coordinates.'''
        if self.cs is None:
            raise ValueError('No coordinate system is defined for this FloquetPort')
        xl, yl, _ = self.cs.in_local_cs(x_global, y_global, z_global)
        Ex, Ey, Ez = self.port_mode_3d(xl, yl, k0)
        Exg, Eyg, Ezg = self.cs.in_global_basis(Ex, Ey, Ez)
        return np.array([Exg, Eyg, Ezg])
        
class ModalPort(PortBC):
    
    _include_stiff: bool = True
    _include_mass: bool = False
    _include_force: bool = True

    def __init__(self,
                 face: FaceSelection | GeoSurface,
                 port_number: int, 
                 active: bool = False,
                 cs: CoordinateSystem | None = None,
                 power: float = 1,
                 TEM: bool = False,
                 mixed_materials: bool = False):
        """Generes a ModalPort boundary condition for a port that requires eigenmode solutions for the mode.

        The boundary condition requires a FaceSelection (or GeoSurface related) object for the face and a port
        number. 
        If the face coordinate system is not provided a local coordinate system will be derived automatically
        by finding the plane that spans the face nodes with minimial out-of-plane error. 

        All modal ports require the execution of a .modal_analysis() by the physics class to define
        the port mode. 

        Args:
            face (FaceSelection, GeoSurface): The port mode face
            port_number (int): The port number as an integer
            active (bool, optional): Whether the port is set active. Defaults to False.
            cs (CoordinateSystem, optional): The local coordinate system of the port face. Defaults to None.
            power (float, optional): The radiated power. Defaults to 1.
            TEM (bool, optional): Wether the mode should be considered as a TEM mode. Defaults to False
            mixed_materials (bool, optional): Wether the port consists of multiple different dielectrics. This requires
                A recalculation of the port mode at every frequency
        """
        super().__init__(face)

        self.port_number: int= port_number
        self.active: bool = active
        self.power: float = power
        self.alignment_vectors: list[Axis] = []

        self.selected_mode: int = 0
        self.modes: dict[float, list[PortMode]] = defaultdict(list)

        self.TEM: bool = TEM
        self.mixed_materials: bool = mixed_materials
        self.initialized: bool = False
        self._first_k0: float | None = None
        self._last_k0: float | None = None
        

        if cs is None:
            logger.info('Constructing coordinate system from normal port')
            self.cs = Axis(self.selection.normal).construct_cs() # type: ignore
        else:
            raise ValueError('No Coordinate System could be derived.')
        self._er: np.ndarray | None = None
        self._ur: np.ndarray | None = None
    
    def portZ0(self, k0: float) -> complex | float | None:
        return self.get_mode(k0).Z0
    
    def modetype(self, k0: float) -> Literal['TEM','TE','TM']:
        return self.get_mode(k0).modetype
    
    def align_modes(self, *axes: tuple | np.ndarray | Axis) -> None:
        """Set a reriees of Axis objects that define a sequence of mode field
        alignments.

        The modes will be sorted to maximize the inner product: |∬ E(x,y) · ax dS|
        
        Args:
            *axes (tuple, np.ndarray, Axis): The alignment vectors.
        """ 
        self.alignment_vectors = [_parse_axis(ax) for ax in axes]
        
    @property
    def nmodes(self) -> int:
        if self._last_k0 is None:
            raise ValueError('ModalPort is not properly configured. No modes are defined.')
        return len(self.modes[self._last_k0])
    
    @property
    def voltage(self) -> complex:
        mode = self.get_mode(0)
        return np.sqrt(mode.Z0)
        
    def sort_modes(self) -> None:
        """Sorts the port modes based on total energy
        """
        
        if len(self.alignment_vectors) > 0:
            logger.trace(f'Sorting modes based on alignment vectors: {self.alignment_vectors}')
            X, Y, Z = self.selection.sample(5)
            X = X.flatten()
            Y = Y.flatten()
            Z = Z.flatten()
            for k0, modes in self.modes.items():
                logger.trace(f'Aligning modes for k0={k0:.3f} rad/m')
                new_modes = []
                for ax in self.alignment_vectors:
                    logger.trace(f'.mode vector {ax}')
                    integrals = [_inner_product(m.E_function, X, Y, Z, ax) for m in modes]
                    integral, opt_mode = sorted([pair for pair in zip(integrals, modes)], key=lambda x: abs(x[0]), reverse=True)[0]
                    opt_mode.polarity = np.sign(integral.real)
                    logger.trace(f'Optimal mode = {opt_mode} ({integral}), polarization alignment = {opt_mode.polarity}')
                    new_modes.append(opt_mode)
                    
                self.modes[k0] = new_modes
            return
        for k0, modes in self.modes.items():
            self.modes[k0] = sorted(modes, key=lambda m: m.energy, reverse=True)

    def get_mode(self, k0: float, i=None) -> PortMode:
        """Returns a given mode solution in the form of a PortMode object.

        Args:
            i (_type_, optional): The mode solution number. Defaults to None.

        Returns:
            PortMode: The requested PortMode object
        """
        if i is None:
            i = self.selected_mode
        return self.modes[min(self.modes.keys(), key=lambda k: abs(k - k0))][i]
    
    def global_field_function(self, k0: float = 0, which: Literal['E','H'] = 'E') -> Callable:
        ''' The field function used to compute the E-field. 
        This field-function is defined in global coordinates (not local coordinates).'''
        mode = self.get_mode(k0)
        if which == 'E':
            return lambda x,y,z: mode.norm_factor * self._qmode(k0) * mode.E_function(x,y,z)*mode.polarity
        else:
            return lambda x,y,z: mode.norm_factor * self._qmode(k0) * mode.H_function(x,y,z)*mode.polarity
    
    def clear_modes(self) -> None:
        """Clear all port mode data"""
        self.modes = defaultdict(list)
        self.initialized = False

    def add_mode(self, 
                 field: np.ndarray,
                 E_function: Callable,
                 H_function: Callable,
                 beta: float,
                 k0: float,
                 residual: float,
                 TEM: bool,
                 freq: float) -> PortMode | None:
        """Add a mode function to the ModalPort

        Args:
            field (np.ndarray): The field value array
            E_function (Callable): The E-field callable
            H_function (Callable): The H-field callable
            beta (float): The out-of-plane propagation constant 
            k0 (float): The free space phase constant
            residual (float): The solution residual
            TEM (bool): Whether its a TEM mode
            freq (float): The frequency of the port mode

        Returns:
            PortMode: The port mode object.
        """
        mode = PortMode(field, E_function, H_function, k0, beta, residual, TEM=TEM, freq=freq)
        if mode.energy < 1e-4:
            logger.debug(f'Ignoring mode due to a low mode energy: {mode.energy}')
            return None
        self.modes[k0].append(mode)
        self.initialized = True

        self._last_k0 = k0
        if self._first_k0 is None:
            self._first_k0 = k0
        else:
            ref_field = self.get_mode(self._first_k0, -1).modefield
            polarity = np.sign(np.sum(field*ref_field).real)
            logger.debug(f'Mode polarity = {polarity}')
            mode.polarity = polarity

        return mode

    def get_basis(self) -> np.ndarray:
        return self.cs._basis

    def get_inv_basis(self) -> np.ndarray:
        return self.cs._basis_inv
    
    def get_beta(self, k0: float) -> float:
        mode = self.get_mode(k0)
        if mode.TEM:
            beta = mode.beta/mode.k0 * k0
        else:
            freq = k0*299792458/(2*np.pi)
            beta = np.sqrt(mode.beta**2 + k0**2 * (1-((mode.freq/freq)**2)))
        return beta

    def get_gamma(self, k0: float) -> complex:
        return 1j*self.get_beta(k0)
    
    def get_Uinc(self, x_global: np.ndarray, y_global: np.ndarray, z_global: np.ndarray, k0) -> np.ndarray:
        return -2*1j*self.get_beta(k0)*self.port_mode_3d_global(x_global, y_global, z_global, k0)
    
    def port_mode_3d(self, 
                     x_local: np.ndarray,
                     y_local: np.ndarray,
                     k0: float,
                     which: Literal['E','H'] = 'E') -> np.ndarray:
        x_global, y_global, z_global = self.cs.in_global_cs(x_local, y_local, 0*x_local)

        Egxyz = self.port_mode_3d_global(x_global,y_global,z_global,k0,which=which)
        
        Ex, Ey, Ez = self.cs.in_local_basis(Egxyz[0,:], Egxyz[1,:], Egxyz[2,:])

        Exyz = np.array([Ex, Ey, Ez])
        return Exyz

    def port_mode_3d_global(self,
                            x_global: np.ndarray,
                            y_global: np.ndarray,
                            z_global: np.ndarray,
                            k0: float,
                            which: Literal['E','H'] = 'E') -> np.ndarray:
        Ex, Ey, Ez = self.global_field_function(k0, which)(x_global,y_global,z_global)
        Exyz = np.array([Ex, Ey, Ez])
        return Exyz

class RectangularWaveguide(PortBC):
    
    _include_stiff: bool = True
    _include_mass: bool = False
    _include_force: bool = True

    def __init__(self, 
                 face: FaceSelection | GeoSurface,
                 port_number: int, 
                 active: bool = False,
                 cs: CoordinateSystem | None = None,
                 dims: tuple[float, float] | None = None,
                 power: float = 1):
        """Creates a rectangular waveguide as a port boundary condition.
        
        Currently the Rectangular waveguide only supports TE0n modes. The mode field
        is derived analytically. The local face coordinate system and dimensions can be provided
        manually. If not provided the class will attempt to derive the local coordinate system and
        face dimensions itself. It always orients the longest edge along the local X-direction.
        The information on the derived coordiante system will be shown in the DEBUG level logs.

        Args:
            face (FaceSelection, GeoSurface): The port boundary face selection
            port_number (int): The port number
            active (bool, optional): Ther the port is active. Defaults to False.
            cs (CoordinateSystem, optional): The local coordinate system. Defaults to None.
            dims (tuple[float, float], optional): The port face. Defaults to None.
            power (float): The port power. Default to 1.
        """
        super().__init__(face)

        self.port_number: int= port_number
        self.active: bool = active
        self.power: float = power
        self.type: str = 'TE'
        self.mode: tuple[int,int] = (1,0)

        if dims is None:
            logger.info("Determining port face based on selection")
            cs, (width, height) = self.selection.rect_basis() # type: ignore
            self.cs = cs # type: ignore
            self.dims = (width, height)
            logger.debug(f'Port CS: {self.cs}')
            logger.debug(f'Detected port {self.port_number} size = {width*1000:.1f} mm x {height*1000:.1f} mm')
        
        if self.cs is None:
            logger.info('Constructing coordinate system from normal port')
            self.cs = Axis(self.selection.normal).construct_cs()
        else:
            self.cs: CoordinateSystem = cs # type: ignore

    def get_basis(self) -> np.ndarray:
        return self.cs._basis
        
    def get_inv_basis(self) -> np.ndarray:
        return self.cs._basis_inv
    
    def portZ0(self, k0: float) -> complex:
        return k0*299792458 * MU0/self.get_beta(k0)

    def modetype(self, k0):
        return self.type
    
    def get_amplitude(self, k0: float) -> float:
        Zte = Z0
        amplitude= np.sqrt(self.power*4*Zte/(self.dims[0]*self.dims[1]))
        return amplitude

    def get_beta(self, k0: float) -> float:
        ''' Return the out of plane propagation constant. βz.'''
        width=self.dims[0]
        height=self.dims[1]
        beta = np.sqrt(k0**2 - (np.pi*self.mode[0]/width)**2 - (np.pi*self.mode[1]/height)**2)
        return beta
    
    def get_gamma(self, k0: float) -> complex:
        """Computes the γ-constant for matrix assembly. This constant is required for the Robin boundary condition.

        Args:
            k0 (float): The free space propagation constant.

        Returns:
            complex: The γ-constant
        """
        return 1j*self.get_beta(k0)
    
    def get_Uinc(self, x_global: np.ndarray, y_global: np.ndarray, z_global: np.ndarray, k0: float) -> np.ndarray:
        return -2*1j*self.get_beta(k0)*self.port_mode_3d_global(x_global, y_global, z_global, k0)
    
    def port_mode_3d(self, 
                     x_local: np.ndarray,
                     y_local: np.ndarray,
                     k0: float,
                     which: Literal['E','H'] = 'E') -> np.ndarray:
        ''' Compute the port mode E-field in local coordinates (XY) + Z out of plane.'''

        width = self.dims[0]
        height = self.dims[1]

        E = self.get_amplitude(k0)*np.cos(np.pi*self.mode[0]*(x_local)/width)*np.cos(np.pi*self.mode[1]*(y_local)/height)
        Ex = 0*E
        Ey = E
        Ez = 0*E
        Exyz =  self._qmode(k0) * np.array([Ex, Ey, Ez])
        return Exyz

    def port_mode_3d_global(self, 
                            x_global: np.ndarray,
                            y_global: np.ndarray,
                            z_global: np.ndarray,
                            k0: float,
                            which: Literal['E','H'] = 'E') -> np.ndarray:
        '''Compute the port mode field for global xyz coordinates.'''
        xl, yl, _ = self.cs.in_local_cs(x_global, y_global, z_global)
        Ex, Ey, Ez = self.port_mode_3d(xl, yl, k0)
        Exg, Eyg, Ezg = self.cs.in_global_basis(Ex, Ey, Ez)
        return np.array([Exg, Eyg, Ezg])

class LumpedPort(PortBC):
    
    _include_stiff: bool = True
    _include_mass: bool = False
    _include_force: bool = True

    def __init__(self, 
                 face: FaceSelection | GeoSurface,
                 port_number: int, 
                 width: float | None = None,
                 height: float | None = None,
                 direction: Axis | None = None,
                 active: bool = False,
                 power: float = 1,
                 Z0: float = 50):
        """Generates a lumped power boundary condition.
        
        The lumped port boundary condition assumes a uniform E-field along the "direction" axis.
        The port with and height must be provided manually in meters. The height is the size
        in the "direction" axis along which the potential is imposed. The width dimension
        is orthogonal to that. For a rectangular face its the width and for a cyllindrical face
        its the circumpherance.

        Args:
            face (FaceSelection, GeoSurface): The port surface
            port_number (int): The port number
            width (float): The port width (meters).
            height (float): The port height (meters).
            direction (Axis): The port direction as an Axis object (em.Axis(..) or em.ZAX)
            active (bool, optional): Whether the port is active. Defaults to False.
            power (float, optional): The port output power. Defaults to 1.
            Z0 (float, optional): The port impedance. Defaults to 50.
        """
        super().__init__(face)

        if width is None:
            if not isinstance(face, GeoObject):
                raise ValueError(f'The width, height and direction must be defined. Information cannot be extracted from {face}')
            width, height, direction = face._data('width','height','vdir')
            if width is None or height is None or direction is None:
                raise ValueError(f'The width, height and direction could not be extracted from {face}')
        
        logger.debug(f'Lumped port: width={1000*width:.1f}mm, height={1000*height:.1f}mm, direction={direction}') # type: ignore
        self.port_number: int= port_number
        self.active: bool = active

        self.power: float = power
        self.Z0: float = Z0
        
        self.width: float = width
        self.height: float = height # type: ignore
        self.Vdirection: Axis = direction # type: ignore
        self.type = 'TEM'
        
        logger.info('Constructing coordinate system from normal port')
        self.cs = Axis(self.selection.normal).construct_cs()  # type: ignore

        self.vintline: Line | None = None
        self.v_integration = True

    @property
    def surfZ(self) -> float:
        """The surface sheet impedance for the lumped port

        Returns:
            float: The surface sheet impedance
        """
        return self.Z0*self.width/self.height
    
    @property
    def voltage(self) -> float:
        """The Port voltage required for the provided output power (time average)

        Returns:
            float: The port voltage
        """
        return np.sqrt(2*self.power*self.Z0)
    
    def get_basis(self) -> np.ndarray:
        return self.cs._basis
        
    def get_inv_basis(self) -> np.ndarray:
        return self.cs._basis_inv
    
    def get_beta(self, k0: float) -> float:
        ''' Return the out of plane propagation constant. βz.'''

        return k0
    
    def get_gamma(self, k0: float) -> complex:
        """Computes the γ-constant for matrix assembly. This constant is required for the Robin boundary condition.

        Args:
            k0 (float): The free space propagation constant.

        Returns:
            complex: The γ-constant
        """
        return 1j*k0*Z0/self.surfZ
    
    def get_Uinc(self, x_global: np.ndarray, y_global: np.ndarray, z_global: np.ndarray, k0) -> np.ndarray:
        Emag = -1j*2*k0 * self.voltage/self.height * (Z0/self.surfZ)
        return Emag*self.port_mode_3d_global(x_global, y_global, z_global, k0)
    
    def port_mode_3d(self, 
                     x_local: np.ndarray,
                     y_local: np.ndarray,
                     k0: float,
                     which: Literal['E','H'] = 'E') -> np.ndarray:
        ''' Compute the port mode E-field in local coordinates (XY) + Z out of plane.'''

        px, py, pz = self.cs.in_local_basis(*self.Vdirection.np)
        
        Ex = px*np.ones_like(x_local)
        Ey = py*np.ones_like(x_local)
        Ez = pz*np.ones_like(x_local)
        Exyz = np.array([Ex, Ey, Ez])
        return Exyz

    def port_mode_3d_global(self, 
                            x_global: np.ndarray,
                            y_global: np.ndarray,
                            z_global: np.ndarray,
                            k0: float,
                            which: Literal['E','H'] = 'E') -> np.ndarray:
        """Computes the port-mode field in global coordinates.

        The mode field will be evaluated at x,y,z coordinates but projected onto the local 2D coordinate system.
        Additionally, the "which" parameter may be used to request the H-field. This parameter is not always supported.

        Args:
            x_global (np.ndarray): The X-coordinate
            y_global (np.ndarray): The Y-coordinate
            z_global (np.ndarray): The Z-coordinate
            k0 (float): The free space propagation constant
            which (Literal["E","H"], optional): Which field to return. Defaults to 'E'.

        Returns:
            np.ndarray: The E-field in (3,N) indexing.
        """
        xl, yl, _ = self.cs.in_local_cs(x_global, y_global, z_global)
        Ex, Ey, Ez = self.port_mode_3d(xl, yl, k0)
        Exg, Eyg, Ezg = self.cs.in_global_basis(Ex, Ey, Ez)
        return np.array([Exg, Eyg, Ezg])


class LumpedElement(RobinBC):
    
    _include_stiff: bool = True
    _include_mass: bool = False
    _include_force: bool = False

    def __init__(self, 
                 face: FaceSelection | GeoSurface,
                 impedance_function: Callable | None = None,
                 width: float | None = None,
                 height: float | None = None,
                 ):
        """Generates a lumped power boundary condition.
        
        The lumped port boundary condition assumes a uniform E-field along the "direction" axis.
        The port with and height must be provided manually in meters. The height is the size
        in the "direction" axis along which the potential is imposed. The width dimension
        is orthogonal to that. For a rectangular face its the width and for a cyllindrical face
        its the circumpherance.

        Args:
            face (FaceSelection, GeoSurface): The port surface
            port_number (int): The port number
            width (float): The port width (meters).
            height (float): The port height (meters).
            direction (Axis): The port direction as an Axis object (em.Axis(..) or em.ZAX)
            active (bool, optional): Whether the port is active. Defaults to False.
            power (float, optional): The port output power. Defaults to 1.
            Z0 (float, optional): The port impedance. Defaults to 50.
        """
        super().__init__(face)

        if width is None:
            if not isinstance(face, GeoObject):
                raise ValueError(f'The width, height and direction must be defined. Information cannot be extracted from {face}')
            width, height, impedance_function = face._data('width','height','func')
            if width is None or height is None or impedance_function is None:
                raise ValueError(f'The width, height and impedance function could not be extracted from {face}')
        
        logger.debug(f'Lumped port: width={1000*width:.1f}mm, height={1000*height:.1f}mm') # type: ignore

        self.Z0: Callable = impedance_function # type: ignore
        self.width: float = width # type: ignore
        self.height: float = height # type: ignore

    def surfZ(self, k0: float) -> float:
        """The surface sheet impedance for the lumped Element

        Returns:
            float: The surface sheet impedance
        """
        Z0 = self.Z0(k0*299792458/(2*np.pi))*self.width/self.height
        return Z0
    
    def get_basis(self) -> np.ndarray | None:
        return None

    def get_inv_basis(self) -> np.ndarray | None:
        return None
    
    def get_beta(self, k0: float) -> float:
        ''' Return the out of plane propagation constant. βz.'''

        return k0

    def get_gamma(self, k0: float) -> complex:
        """Computes the γ-constant for matrix assembly. This constant is required for the Robin boundary condition.

        Args:
            k0 (float): The free space propagation constant.

        Returns:
            complex: The γ-constant
        """
        return 1j*k0*Z0/self.surfZ(k0)
    
class SurfaceImpedance(RobinBC):
    
    _include_stiff: bool = True
    _include_mass: bool = False
    _include_force: bool = False

    def __init__(self, 
                 face: FaceSelection | GeoSurface,
                 material: Material | None = None,
                 surface_conductance: float | None = None,
                 surface_roughness: float = 0,
                 sr_model: Literal['Hammerstad-Jensen'] = 'Hammerstad-Jensen',
                 ):
        """Generates a SurfaceImpedance bounary condition.

        The surface impedance model treats a 2D surface selection as a finite conductor. It is not
        intended to be used for dielectric materials.

        The surface resistivity is computed based on the material properties: σ, ε and μ. 

        The user may also supply the surface condutivity directly. 

        Optionally, a surface roughness in meters RMS may be supplied. In the current implementation
        The Hammersstad-Jensen model is used increasing the resistivity by a factor (1 + 2/π tan⁻¹(1.4(Δ/δ)²).

        Args:
            face (FaceSelection | GeoSurface): The face to apply this condition to.
            material (Material | None, optional): The matrial to assign. Defaults to None.
            surface_conductance (float | None, optional): The specific bulk conductivity to use. Defaults to None.
            surface_roughness (float, optional): The surface roughness. Defaults to 0.
            sr_model (Literal[&#39;Hammerstad, optional): The surface roughness model. Defaults to 'Hammerstad-Jensen'.
        """
        super().__init__(face)

        self._material: Material | None = material
        self._mur: float | complex = 1.0
        self._epsr: float | complex = 1.0

        self.sigma: float = 0.0
        if material is not None:
            self.sigma = material.cond
            self._mur = material.ur
            self._epsr = material.er
        if surface_conductance is not None:
            self.sigma = surface_conductance
        
        self._sr: float = surface_roughness
        self._sr_model: str = sr_model
    
    def get_basis(self) -> np.ndarray | None:
        return None

    def get_inv_basis(self) -> np.ndarray | None:
        return None
    
    def get_beta(self, k0: float) -> float:
        ''' Return the out of plane propagation constant. βz.'''

        return k0

    def get_gamma(self, k0: float) -> complex:
        """Computes the γ-constant for matrix assembly. This constant is required for the Robin boundary condition.

        Args:
            k0 (float): The free space propagation constant.

        Returns:
            complex: The γ-constant
        """
        w0 = k0*C0
        sigma = self.sigma
        rho = 1/sigma
        d_skin = (2*rho/(w0*MU0*self._mur) * ((1+(w0*EPS0*self._epsr*rho)**2)**0.5 + rho*w0*EPS0*self._epsr))**0.5
        R = rho/d_skin
        if self._sr_model=='Hammerstad-Jensen' and self._sr > 0.0:
            R = R * (1 + 2/np.pi * np.arctan(1.4*(self._sr/d_skin)**2))
        return 1j*k0*Z0/R