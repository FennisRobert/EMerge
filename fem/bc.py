from __future__ import annotations
import numpy as np
from enum import Enum
from loguru import logger
from typing import Callable, Literal
from .selection import Selection, FaceSelection
from .cs import CoordinateSystem, Axis, GCS
from .coord import Line
from .geo3d import GMSHSurface, GMSHObject
from dataclasses import dataclass

class BCDimension(Enum):
    ANY = -1
    NODE = 0
    EDGE = 1
    FACE = 2
    DOMAIN = 3

def _unique(input: list[int]) -> list:
    """ Returns a sorted list of all unique integers/floats in a list."""
    output = sorted(list(set(input)))
    return output

class BoundaryCondition:
    """A generalized class for all boundary condition objects.
    """
    
    def __init__(self, assignment: GMSHObject | Selection):

        self.dimension: BCDimension = BCDimension.ANY
        self.indices: list[int] = []
        self.face_indices: list[int] = []
        self.edge_indices: list[int] = []
        
        
        if isinstance(assignment, GMSHObject):
            assignment = assignment.select
        
        self.selection: Selection = assignment
        self.tags: list[int] = self.selection.tags
    
    @property
    def dim(self) -> int:
        ''' The dimension of the boundary condition as integer (0,1,2,3).'''
        return self.dimension.value
    
    def __repr__(self) -> str:
        if self.dimension is BCDimension.ANY:
            return f'{type(self).__name__}{self.tags}'
        elif self.dimension is BCDimension.EDGE:
            return f'{type(self).__name__}{self.tags}'
        elif self.dimension is BCDimension.NODE:
            return f'{type(self).__name__}{self.tags}'
        elif self.dimension is BCDimension.FACE:
            return f'{type(self).__name__}{self.tags}'
        elif self.dimension is BCDimension.DOMAIN:
            return f'{type(self).__name__}{self.tags}'
    
    def __str__(self) -> str:
        return self.__repr__()
    
    def check_dimension(self, tags: list[tuple[int,int]]) -> None:
        # check if all tags have the same timension (dim, tag)
        if not isinstance(tags, list):
            raise TypeError(f'Argument tags must be of type list, instead its {type(tags)}')
        if len(tags) == 0:
            return
        if not all(isinstance(x, tuple) and len(x) == 2 for x in tags):
            raise TypeError(f'Argument tags must be of type list of tuples, instead its {type(tags)}')
        if not all(isinstance(x[0], int) and isinstance(x[1], int) for x in tags):
            raise TypeError(f'Argument tags must be of type list of tuples of ints, instead its {type(tags)}')
        if not all(x[0] == tags[0][0] for x in tags):
            raise ValueError(f'All tags must have the same dimension, instead its {tags}')
        dimension = tags[0][0]
        if self.dimension is BCDimension.ANY:
            logger.info(f'Assigning dimension {BCDimension(dimension)} to {self}')
            self.dimension = BCDimension(dimension)
        elif self.dimension != BCDimension(dimension):
            raise ValueError(f'Current boundary condition has dimension {self.dimension}, but tags have dimension {BCDimension(dimension)}')
        
    def add_tags(self, tags: list[tuple[int,int]]) -> None:
        """Adds the given taggs to this boundary condition.

        Args:
            tags (list[tuple[int,int]]): The tags to include
        """
        self.check_dimension(tags)
        tags = [x[1] for x in tags]
        self.tags = _unique(self.tags + tags)
    
    def remove_tags(self, tags: list[int]) -> list[int]:
        """Removes the tags provided by tags from this boundary condition.

        Return sonly the tags that are actually excluded from this face.

        Args:
            tags (list[int]): The tags to exclude.

        Returns:
            list[int]: A list of actually excluded tags.
        """
        excluded_edges = [x for x in self.tags if x in tags]
        self.tags = [x for x in self.tags if x not in tags]
        return excluded_edges
    
    def exclude_bc(self, other: BoundaryCondition) -> list[int]:
        """Excludes all faces for a provided boundary condition object from this boundary condition assignment.

        Args:
            other (BoundaryCondition): The boundary condition of which the faces should be excluded

        Returns:
            list[int]: A list of excluded face tags.
        """
        return self.remove_tags(other.tags)


class PEC(BoundaryCondition):
    
    def __init__(self,
                 face: FaceSelection | GMSHSurface):
        """The general perfect electric conductor boundary condition.

        The physics compiler will by default always turn all exterior faces into a PEC.

        Args:
            face (FaceSelection | GMSHSurface): The boundary surface
        """
        super().__init__(face)

class RobinBC(BoundaryCondition):
    
    _include_stiff: bool = False
    _include_mass: bool = False
    _include_force: bool = False

    def __init__(self, selection: GMSHSurface | Selection):
        """A Generalization of any boundary condition of the third kind (Robin).

        This should not be created directly. A robin boundary condition is the generalized type behind
        port boundaries, radiation boundaries etc. Since all boundary conditions of the thrid kind (Robin)
        are assembled the same, this class is used during assembly.

        Args:
            selection (GMSHSurface | Selection): The boundary surface.
        """
        super().__init__(selection)
        self.v_integration: bool = False
        self.vintline: Line = None
    
    def get_beta(self, k0) -> float:
        raise NotImplementedError('get_beta not implemented for Port class')
    
    def get_gamma(self, k0) -> float:
        raise NotImplementedError('get_gamma not implemented for Port class')
    
    def get_Uinc(self, k0) -> np.ndarray:
        raise NotImplementedError('get_Uinc not implemented for Port class')

class PortBC(RobinBC):
    Zvac: float = 376.730313412

    def __init__(self, face: FaceSelection | GMSHSurface):
        """(DO NOT USE) A generalization of the Port boundary condition.
        
        DO NOT USE THIS TO DEFINE PORTS. This class is only indeded for 
        class inheritance and type checking. 

        Args:
            face (FaceSelection | GMSHSurface): The port face
        """
        super().__init__(face)
        self.port_number: int = None
        self.cs: CoordinateSystem = None
        self.selected_mode: int = 0
        self.Z0 = None

    @property
    def modetype(self) -> Literal['TEM','TE','TM']:
        return 'TEM'
    
    def Zmode(self, k0: float) -> float:
        if self.modetype=='TEM':
            return self.Zvac
        elif self.modetype=='TE':
            return k0*299792458/self.get_beta(k0) * 4*np.pi*1e-7
        elif self.modetype=='TM':
            return self.get_beta(k0)/(k0*299792458*8.854187818814*1e-12)
        else:
            return ValueError(f'Port mode type should be TEM, TE or TM but instead is {self.type}')
    
    @property
    def mode_number(self) -> int:
        return self.selected_mode + 1

    def get_beta(self, k0) -> float:
        ''' Return the out of plane propagation constant. βz.'''
        return k0
    
    def get_gamma(self, k0):
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
                 face: FaceSelection | GMSHSurface,
                 order: int = 1,
                 origin: tuple = None):
        """Creates an AbsorbingBoundary condition.

        Currently only a first order boundary condition is possible. Second order will be supported later.
        The absorbing boundary is effectively a port boundary condition (Robin) with an assumption on
        the out-of-plane phase constant. For now it always assumes the free-space propagation (normal).

        Args:
            face (FaceSelection | GMSHSurface): The absorbing boundary face(s)
            order (int, optional): The order (only 1 is supported). Defaults to 1.
            origin (tuple, optional): The radiation origin. Defaults to None.
        """
        super().__init__(face)
    
        self.order: int = order
        self.origin: tuple = origin
        self.cs: CoordinateSystem = GCS

    def get_basis(self) -> np.ndarray:
        return np.eye(3)
    
    def get_beta(self, k0) -> float:
        ''' Return the out of plane propagation constant. βz.'''
        return k0

    def get_gamma(self, k0):
        """Computes the γ-constant for matrix assembly. This constant is required for the Robin boundary condition.

        Args:
            k0 (float): The free space propagation constant.

        Returns:
            complex: The γ-constant
        """
        return 1j*self.get_beta(k0)
    
    def get_Uinc(self, x_local, y_local, k0) -> np.ndarray:
        return np.zeros((3, len(x_local)), dtype=np.complex128)

@dataclass
class PortMode:
    modefield: np.ndarray
    E_function: Callable
    H_function: Callable
    k0: float
    beta: float
    residual: float
    energy: float = None
    norm_factor: float = 1
    freq: float = None
    neff: float = None
    TEM: bool = None
    Z0: float = None
    modetype: Literal['TEM','TE','TM'] = 'TEM'

    def __post_init__(self):
        self.neff = self.beta/self.k0
        self.energy = np.mean(np.abs(self.modefield)**2)

    def __str__(self):
        return f'PortMode(k0={self.k0}, beta={self.beta}, neff={self.neff}, energy={self.energy})'
    
    def set_power(self, power: complex) -> None:
        self.norm_factor = np.sqrt(1/np.abs(power))
        logger.info(f'Setting port mode amplitude to: {self.norm_factor} ')

class ModalPort(PortBC):
    
    _include_stiff: bool = True
    _include_mass: bool = False
    _include_force: bool = True

    def __init__(self,
                 face: FaceSelection | GMSHSurface,
                 port_number: int, 
                 active: bool = False,
                 cs: CoordinateSystem = None,
                 power: float = 1):
        """Generes a ModalPort boundary condition for a port that requires eigenmode solutions for the mode.

        The boundary condition requires a FaceSelection (or GMSHSurface related) object for the face and a port
        number. 
        If the face coordinate system is not provided a local coordinate system will be derived automatically
        by finding the plane that spans the face nodes with minimial out-of-plane error. 

        All modal ports require the execution of a .modal_analysis() by the physics class to define
        the port mode. 

        Args:
            face (FaceSelection, GMSHSurface): The port mode face
            port_number (int): The port number as an integer
            active (bool, optional): Whether the port is set active. Defaults to False.
            cs (CoordinateSystem, optional): The local coordinate system of the port face. Defaults to None.
            power (float, optional): The radiated power. Defaults to 1.
        """
        super().__init__(face)

        self.port_number: int= port_number
        self.active: bool = active

        self.power: float = power

        self.cs: CoordinateSystem = cs
        self.selected_mode: int = 0
        self.modes: list[PortMode] = []

        if self.cs is None:
            logger.info('Constructing coordinate system from normal port')
            self.cs = Axis(self.selection.normal).construct_cs()

        self._er: np.ndarray = None
        self._ur: np.ndarray = None
        
    @property
    def modetype(self) -> Literal['TEM','TE','TM']:
        return self.get_mode().modetype
    
    @property
    def nmodes(self) -> int:
        return len(self.modes)
    
    def sort_modes(self) -> None:
        """Sorts the port modes based on total energy
        """
        self.modes = sorted(self.modes, key=lambda m: m.energy, reverse=True)

    def get_mode(self, i=None) -> PortMode:
        """Returns a given mode solution in the form of a PortMode object.

        Args:
            i (_type_, optional): The mode solution number. Defaults to None.

        Returns:
            PortMode: The requested PortMode object
        """
        if i is None:
            i = self.selected_mode
        return self.modes[i]
    
    def global_field_function(self, which: Literal['E','H'] = 'E') -> Callable:
        ''' The field function used to compute the E-field. 
        This field-function is defined in global coordinates (not local coordinates).'''
        mode = self.get_mode()
        if which == 'E':
            return lambda x,y,z: mode.norm_factor*mode.E_function(x,y,z)
        else:
            return lambda x,y,z: mode.norm_factor*mode.H_function(x,y,z)
    
    def add_mode(self, field: np.ndarray,
                 E_function: Callable,
                 H_function: Callable,
                 beta: float,
                 k0: float,
                 residual: float,
                 TEM: bool,
                 freq: float) -> PortMode:
        """_summary_

        Args:
            field (np.ndarray): _description_
            E_function (Callable): _description_
            H_function (Callable): _description_
            beta (float): _description_
            k0 (float): _description_
            residual (float): _description_
            TEM (bool): _description_
            freq (float): _description_

        Returns:
            PortMode: _description_
        """
        mode = PortMode(field, E_function, H_function, k0, beta, residual, TEM=TEM, freq=freq)
        if mode.energy < 1e-4:
            logger.debug(f'Ignoring mode due to a low mode energy: {mode.energy}')
            return
        self.modes.append(mode)
        return mode

    def get_basis(self) -> np.ndarray:
        return self.cs._basis
    
    def get_beta(self, k0) -> float:
        ''' Return the out of plane propagation constant. βz.'''
        mode = self.get_mode()
        if mode.TEM:
            beta = mode.beta/mode.k0 * k0
        else:
            freq = k0*299792458/(2*np.pi)
            beta = np.sqrt(mode.beta**2 + k0**2 * (1-((mode.freq/freq)**2)))
        logger.debug(f'    Derived kz={beta.real:.2f} (k0 = {k0:.2f}), neff = {np.sqrt(beta/k0).real:.2f}')
        return beta

    def get_gamma(self, k0):
        """Computes the γ-constant for matrix assembly. This constant is required for the Robin boundary condition.

        Args:
            k0 (float): The free space propagation constant.

        Returns:
            complex: The γ-constant
        """
        return 1j*self.get_beta(k0)
    
    def get_Uinc(self, x_local, y_local, k0) -> np.ndarray:
        return -2*1j*self.get_beta(k0)*self.port_mode_3d(x_local, y_local, k0)
    
    def port_mode_3d(self, 
                     x_local: np.ndarray,
                     y_local: np.ndarray,
                     k0: float,
                     which: Literal['E','H'] = 'E') -> np.ndarray:
        
        ''' Compute the port mode E-field in local coordinates (XY) + Z out of plane.'''

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
        Ex, Ey, Ez = np.sqrt(self.power) * self.global_field_function(which)(x_global,y_global,z_global)
        Exyz = np.array([Ex, Ey, Ez])
        return Exyz

class RectangularWaveguide(PortBC):
    
    _include_stiff: bool = True
    _include_mass: bool = False
    _include_force: bool = True

    def __init__(self, 
                 face: FaceSelection | GMSHSurface,
                 port_number: int, 
                 active: bool = False,
                 cs: CoordinateSystem = None,
                 dims: tuple[float, float] = None,
                 power: float = 1):
        """Creates a rectangular waveguide as a port boundary condition.
        
        Currently the Rectangular waveguide only supports TE0n modes. The mode field
        is derived analytically. The local face coordinate system and dimensions can be provided
        manually. If not provided the class will attempt to derive the local coordinate system and
        face dimensions itself. It always orients the longest edge along the local X-direction.
        The information on the derived coordiante system will be shown in the DEBUG level logs.

        Args:
            face (FaceSelection, GMSHSurface): The port boundary face selection
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
        self._field_amplitude: np.ndarray = None
        self.mode: tuple[int,int] = (1,0)
        self.cs: CoordinateSystem = cs

        if dims is None:
            logger.info("Determining port face based on selection")
            cs, (width, height) = face.rect_basis()
            self.cs = cs
            self.dims = (width, height)
            logger.debug(f'Port CS: {self.cs}')
            logger.debug(f'Detected port {self.port_number} size = {width*1000:.1f} mm x {height*1000:.1f} mm')
        
        if self.cs is None:
            logger.info('Constructing coordinate system from normal port')
            self.cs = Axis(self.selection.normal).construct_cs()
        else:
            self.cs: CoordinateSystem = cs

    def get_amplitude(self, k0: float) -> float:
        w = k0*299792458
        lc = 2*self.dims[0]
        wc = 299792458 * 2 * np.pi / lc
        Q = np.sqrt(1-wc**2/w**2)
        Zte = 377
        amplitude= np.sqrt(self.power*4*Zte/(self.dims[0]*self.dims[1]))
        return amplitude
    
    def port_mode_2d(self, xs: np.ndarray, ys: np.ndarray, k0: float) -> tuple[np.ndarray, float]:
        x0 = xs[0]
        y0 = ys[0]
        x1 = xs[-1]
        y1 = ys[-1]
        xc = 0.5*(x0+x1)
        yc = 0.5*(y0+y1)
        a = np.sqrt((x1-x0)**2 + (y1-y0)**2)
        
        logger.debug(f'Detected port {self.port_number} width = {a*1000:.1f} mm')
        ds = np.sqrt((xs-xc)**2 + (ys-yc)**2)
        return self.amplitude*np.cos(ds*np.pi/a), np.sqrt(k0**2 - (np.pi/a)**2)
    
    def get_basis(self) -> np.ndarray:
        return self.cs._basis
    
    def get_beta(self, k0: float) -> float:
        ''' Return the out of plane propagation constant. βz.'''
        width=self.dims[0]
        height=self.dims[1]
        return np.sqrt(k0**2 - (np.pi*self.mode[0]/width)**2 - (np.pi*self.mode[1]/height)**2)
    
    def get_gamma(self, k0: float):
        """Computes the γ-constant for matrix assembly. This constant is required for the Robin boundary condition.

        Args:
            k0 (float): The free space propagation constant.

        Returns:
            complex: The γ-constant
        """
        return 1j*self.get_beta(k0)
    
    def get_Uinc(self, x_local: np.ndarray, y_local: np.ndarray, k0: float) -> np.ndarray:
        return -2*1j*self.get_beta(k0)*self.port_mode_3d(x_local, y_local, k0)
    
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
        Exyz = np.array([Ex, Ey, Ez])
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
                 face: FaceSelection | GMSHSurface,
                 port_number: int, 
                 width: float,
                 height: float,
                 direction: Axis,
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
            face (FaceSelection, GMSHSurface): The port surface
            port_number (int): The port number
            width (float): The port width (meters).
            height (float): The port height (meters).
            direction (Axis): The port direction as an Axis object (fem.Axis(..) or fem.ZAX)
            active (bool, optional): Whether the port is active. Defaults to False.
            power (float, optional): The port output power. Defaults to 1.
            Z0 (float, optional): The port impedance. Defaults to 50.
        """
        super().__init__(face)
        
        self.port_number: int= port_number
        self.active: bool = active

        self.power: float = power
        self.Z0: float = Z0
        
        self._field_amplitude: np.ndarray = None
        self.width: float = width
        self.height: float = height
        self.direction: Axis = direction
        
        self.type = 'TEM'
        
        logger.info('Constructing coordinate system from normal port')
        self.cs = Axis(self.selection.normal).construct_cs()

        self.vintline: Line = None
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
        return 1j*k0*376.7303/self.surfZ
    
    def get_Uinc(self, x_local, y_local, k0) -> np.ndarray:
        Emag = -1j*2*k0* self.voltage/self.height * (376.7303/self.surfZ)
        return Emag*self.port_mode_3d(x_local, y_local, k0)
    
    def port_mode_3d(self, 
                     x_local: np.ndarray,
                     y_local: np.ndarray,
                     k0: float,
                     which: Literal['E','H'] = 'E') -> np.ndarray:
        ''' Compute the port mode E-field in local coordinates (XY) + Z out of plane.'''

        px, py, pz = self.cs.in_local_basis(*self.direction.np)
        
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
    

#### LEGACY CODE

# class ABC(BoundaryCondition):
#     def __init__(self, 
#                  order: int = 2, 
#                  origin: tuple = None, 
#                  func: callable = None):
#         super().__init__()
#         self.order = order
#         self.origin = origin
#         self.func = func

#     @staticmethod
#     def spherical(order: int = 2, origin: tuple = (0,0)) -> ABC:
#         x0, y0 = origin
#         #func = lambda x,y,nx,ny,k0: np.exp(-1j*k0*np.sqrt((x-x0)**2+(y-y0)**2))/np.sqrt(np.sqrt((x-x0)**2+(y-y0)**2))#/((x-x0)**2 + (y-y0)**2)**(1/4)
#         def func(x,y,nx,ny,k0):
#             return np.exp(-1j*k0*np.sqrt((x-x0)**2+(y-y0)**2))/np.sqrt(np.sqrt((x-x0)**2+(y-y0)**2))
#         return ABC(order=order, origin = origin, func=func)

# class PointSource(BoundaryCondition):

#     def __init__(self, tags: list[int] = None):
#         super().__init__()
#         if tags is None:
#             tags = []
#         self.amplitude = 1