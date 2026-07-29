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

# Last Cleanup: 2025-01-01
from __future__ import annotations
import numpy as np
from loguru import logger
from typing import Callable, Literal, Generator
from ....selection import Selection, FaceSelection
from ....cs import CoordinateSystem, GCS
from ....geometry import GeoSurface, GeoObject
from ....bc import (
    BoundaryCondition,
)
from emsutil import Material, AIR, Saveable
from ....const import Z0, C0, EPS0, MU0
from . import background_field as bf
from ....logsettings import DEBUG_COLLECTOR

############################################################
#                    BOUNDARY CONDITIONS                   #
############################################################


class PEC(BoundaryCondition, Saveable):
    _color: str = "#f70a80"
    _name: str = "PEC"
    _texture: str = "tex1.png"
    dim: int = 2

    def __init__(self, face: FaceSelection | GeoSurface):
        """The general perfect electric conductor boundary condition.

        The physics compiler will by default always turn all exterior faces into a PEC.

        Args:
            face (FaceSelection | GeoSurface): The boundary surface
        """
        super().__init__(face)


class PMC(BoundaryCondition, Saveable):
    _color: str = "#0084ff"
    _name: str = "PMC"
    _texture: str = "tex4.png"
    dim: int = 2
    pass


class RobinBC(BoundaryCondition, Saveable):
    _color: str = "#e7c736"
    _name: str = "RobinBC"
    _texture: str = "tex5.png"
    _include_stiff: bool = False
    _include_mass: bool = False
    _include_force: bool = False
    _isabc: bool = False
    o2coeffs: tuple[float, float] = {
        "A": (1.0, -0.5),
        "B": (1.00023, -0.51555),
        "C": (1.03084, -0.73631),
        "D": (1.06103, -0.84883),
        "E": (1.12500, -1.00000),
    }
    dim: int = 2
    pml: bool = False

    def __init__(self, selection: GeoSurface | Selection):
        """A Generalization of any boundary condition of the third kind (Robin).

        This should not be created directly. A robin boundary condition is the generalized type behind
        port boundaries, radiation boundaries etc. Since all boundary conditions of the thrid kind (Robin)
        are assembled the same, this class is used during assembly.

        Args:
            selection (GeoSurface | Selection): The boundary surface.
        """
        super().__init__(selection)
        self._assemble_matrix: bool = True
        self.abctype: str = "A"
        self.material: Material = AIR

    def dont_assemble(self) -> None:
        """Prevent this port boundary condition from being assembled in the
        matrix. This means only the imposed forcing vector gets included.
        """
        self._assemble_matrix = False

    def get_basis(self) -> np.ndarray:
        raise NotImplementedError("This method is not implemented")

    def get_inv_basis(self) -> np.ndarray | None:
        raise NotImplementedError("This method is not implemented")

    def get_beta(self, k0: float) -> float:
        raise NotImplementedError("get_beta not implemented for Port class")

    def get_gamma(self, k0: float) -> complex:
        raise NotImplementedError("get_gamma not implemented for Port class")

    def get_Uinc(
        self, x_local: np.ndarray, y_local: np.ndarray, k0: float, mode_nr: int = 1
    ) -> np.ndarray:
        raise NotImplementedError("get_Uinc not implemented for Port class")

    def get_abccorr(self, k0: float) -> float:
        f = k0 * C0 / (2 * np.pi)
        return 1j * self.o2coeffs[self.abctype][1] / (self.material.neff(f) * k0)


class AbsorbingBoundary(RobinBC, Saveable):
    _include_stiff: bool = True
    _include_mass: bool = True
    _include_force: bool = False
    _isabc: bool = True
    _color: str = "#1ce13d"
    _name: str = "AbsorbingBC"
    _texture: str = "tex3.png"
    dim: int = 2

    def __init__(
        self,
        face: FaceSelection | GeoSurface,
        order: int = 2,
        origin: tuple | None = None,
        abctype: Literal["A", "B", "C", "D", "E"] = "B",
    ):
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
            origin = (0.0, 0.0, 0.0)
        self.order: int = order
        self.origin: tuple = origin
        self.cs: CoordinateSystem = GCS
        self.material: Material = AIR
        self.abctype: Literal["A", "B", "C", "D", "E"] = abctype

    def details(self) -> str:
        return f' ABC: order={self.order}, type={self.abctype}'
    def get_basis(self) -> np.ndarray:
        return np.eye(3)

    def get_inv_basis(self) -> np.ndarray | None:
        return None

    def get_beta(self, k0: float) -> float:
        """Return the out of plane propagation constant. βz."""
        return k0

    def get_gamma(self, k0: float) -> complex:
        """Computes the γ-constant for matrix assembly. This constant is required for the Robin boundary condition.

        Args:
            k0 (float): The free space propagation constant.

        Returns:
            complex: The γ-constant
        """
        f = k0 * C0 / (2 * np.pi)
        if self.order == 1:
            return 1j * k0 * self.material.neff(f)

        return 1j * k0 * self.o2coeffs[self.abctype][0] * self.material.neff(f)


class ScatteredField(RobinBC, Saveable):
    _include_stiff: bool = True
    _include_mass: bool = False
    _include_force: bool = True
    _isabc: bool = True
    _color: str = "#be9f11"
    _name: str = "UserDefined"
    _texture: str = "tex5.png"
    skip_fields = ("_fex", "_fey", "_fez", "_fkz")
    dim: int = 2

    def __init__(
        self,
        face: FaceSelection | GeoSurface,
        power_density: float = 1.0 / (2 * Z0),
        cs: CoordinateSystem | None = None,
    ):
        """Creates a user defined port field

        The UserDefinedPort is defined based on user defined field callables. All undefined callables will default to 0 field or k0.

        Define a set of excitation plane waves using: .set_excitation:
        The coordinate system used is different from spherical coordinates and is defined with three numbers:
         - θ (deg) - Elevation
         - ϕ (deg) - Azimuth
         - Ѱ (deg) - Polarization
        The following angles will yield the following propagation (k) and polarization (E) direcitons
         - (0,0,0):     k= +X, E= +Z
         - (0,90,0):    k= +Y, E= +Z
         - (90,0,0):    k= -Z, E= +X
         - (0,0,90):    k= +X, E= +Y

        Args:
            face (FaceSelection, GeoSurface): The port boundary face selection
            port_number (int): The port number
            Ex (Callable): The Ex(k0,x,y,z) field
            Ey (Callable): The Ey(k0,x,y,z) field
            Ez (Callable): The Ez(k0,x,y,z) field
            kz (Callable): The out of plane propagation constant kz(k0)
            power (float): The port output power
        """
        super().__init__(face)
        if cs is None:
            cs = GCS
        self.cs = cs
        self.order = 1
        self.pml: bool = False
        self.abctype = "A"
        self.radius: float = None
        self.thetas: list[float] = [
            0.0,
        ]
        self.phis: list[float] = [
            0.0,
        ]
        self.polarizations: list[float] = [
            0.0,
        ]
        self.E0: float = (power_density * 2 * Z0) ** 0.5
        self.defintion: bf.DEFINITIONS = "EA"
        self.bf: type[bf.BackgroundField] = bf.BackgroundField

    def get_basis(self) -> np.ndarray:
        return self.cs._basis

    def get_inv_basis(self) -> np.ndarray:
        return self.cs._basis_inv

    def modetype(self, k0: float):
        return self.type

    def get_amplitude(self, k0: float) -> float:
        return self.E0

    def get_beta(self, k0: float) -> float:
        """Return the out of plane propagation constant. βz."""
        return k0

    def _iter_fields(self, k0: float) -> Generator[bf.BackgroundField, None, None]:
        from itertools import product

        for theta, phi, psi in product(self.thetas, self.phis, self.polarizations):
            yield self.bf(k0, theta, phi, psi, self.cs.origin, self.E0, self.defintion)

    def set_backgroundfield_type(self, bftype: type[bf.BackgroundField]) -> None:
        """Provide a manual background field type

        Args:
            bftype (type[bf.BackgroundField]): A class or subclass of BackgroundField

        Returns:
            _type_: _description_

        Yields:
            _type_: _description_
        """
        if isinstance(bftype, type) and issubclass(bftype, bf.BackgroundField):
            self.bf = bftype
        else:
            raise TypeError(
                f"A custom backgroundfield must be provided as uninitialized class, not a class instance."
            )

    @property
    def curvature(self) -> float:
        if self.radius is None:
            return 0
        return 1 / (2 * self.radius)

    def get_gamma(self, k0: float) -> complex:
        """Computes the γ-constant for matrix assembly. This constant is required for the Robin boundary condition.

        Args:
            k0 (float): The free space propagation constant.

        Returns:
            complex: The γ-constant
        """
        return 1j * self.get_beta(k0) + self.curvature

    def set_excitations(
        self,
        thetas: np.ndarray | float = 0.0,
        phis: np.ndarray | float = 0.0,
        polarizations: np.ndarray | float = 0.0,
    ) -> None:
        """Set the excitations of different modes in degrees

        The coordinate system used is different from spherical coordinates and is defined with three numbers:
         - θ (deg) - Elevation
         - ϕ (deg) - Azimuth
         - Ѱ (deg) - Polarization
        The following angles will yield the following propagation (k) and polarization (E) direcitons
         - (0,0,0):     k= +X, E= +Z
         - (0,90,0):    k= +Y, E= +Z
         - (90,0,0):    k= -Z, E= +X
         - (0,0,90):    k= +X, E= +Y


        Args:
            thetas (np.ndarray | float, optional): The theta angles. Defaults to 0.0.
            phis (np.ndarray | float, optional): The phi angles. Defaults to 0.0.
            polarizations (np.ndarray | float, optional): The polarizaiton angles. Defaults to 0.0.
        """
        t, p, pol = np.broadcast_arrays(thetas, phis, polarizations)
        t = t * np.pi / 180
        p = p * np.pi / 180
        pol = pol * np.pi / 180
        # Convert to list if that is your preferred storage format
        self.thetas = t.tolist()
        self.phis = p.tolist()
        self.polarizations = pol.tolist()
        if not isinstance(self.thetas, list):
            self.thetas = [
                self.thetas,
            ]
        if not isinstance(self.phis, list):
            self.phis = [
                self.phis,
            ]
        if not isinstance(self.polarizations, list):
            self.polarizations = [
                self.polarizations,
            ]


class LumpedElement(RobinBC, Saveable):
    _include_stiff: bool = True
    _include_mass: bool = False
    _include_force: bool = False
    _color: str = "#e11c1c"
    _name: str = "LumpedElement"
    skip_fields = [
        "Z0",
    ]
    dim: int = 2

    def __init__(
        self,
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
                raise ValueError(
                    f"The width, height and direction must be defined. Information cannot be extracted from {face}"
                )
            for lpd in face._mdi.iter('lumpedelement'):
                width, height, impedance_function = lpd['width'], lpd['height'], lpd['func']

        if width is None or height is None:
            raise ValueError(
                f"The width, height and direction could not be extracted from {face}"
            )
            
        logger.debug(
            f"Lumped port: width={1000 * width:.1f}mm, height={1000 * height:.1f}mm"
        )  # type: ignore

        self.Z0: Callable = impedance_function  # type: ignore
        self.width: float = width  # type: ignore
        self.height: float = height  # type: ignore

    def surfZ(self, k0: float) -> float:
        """The surface sheet impedance for the lumped Element

        Returns:
            float: The surface sheet impedance
        """
        Z0 = self.Z0(k0 * 299792458 / (2 * np.pi)) * self.width / self.height
        return Z0

    def get_basis(self) -> np.ndarray | None:
        return None

    def get_inv_basis(self) -> np.ndarray | None:
        return None

    def get_beta(self, k0: float) -> float:
        """Return the out of plane propagation constant. βz."""

        return k0

    def get_gamma(self, k0: float) -> complex:
        """Computes the γ-constant for matrix assembly. This constant is required for the Robin boundary condition.

        Args:
            k0 (float): The free space propagation constant.

        Returns:
            complex: The γ-constant
        """
        return 1j * k0 * Z0 / self.surfZ(k0)


class SurfaceImpedance(RobinBC, Saveable):
    _include_stiff: bool = True
    _include_mass: bool = False
    _include_force: bool = False
    _color: str = "#49e8ed"
    _name: str = "SurfaceImpedance"
    skip_fields = ("_Zf",)
    dim: int = 2

    def __init__(
        self,
        face: FaceSelection | GeoSurface,
        material: Material | None = None,
        surface_conductivity: float | None = None,
        surface_roughness: float = 0,
        thickness: float | None = None,
        sr_model: Literal["Hammerstad-Jensen"] = "Hammerstad-Jensen",
        impedance_function: Callable | None = None,
        surface_conductance: float | None = None,
    ):
        """Generates a SurfaceImpedance bounary condition.

        The surface impedance model treats a 2D surface selection as a finite conductor. It is not
        intended to be used for dielectric materials.

        It is intended for modelling the walls of conductors. It is not intended for thin
        conducting surfaces like striplines. In that case use ThinConductor.

        The surface resistivity is computed based on the material properties: σ, ε and μ.

        The user may also supply the surface condutivity directly.

        Optionally, a surface roughness in meters RMS may be supplied. In the current implementation
        The Hammersstad-Jensen model is used increasing the resistivity by a factor (1 + 2/π tan⁻¹(1.4(Δ/δ)²).

        Args:
            face (FaceSelection | GeoSurface): The face to apply this condition to.
            material (Material | None, optional): The matrial to assign. Defaults to None.
            surface_conductivity (float | None, optional): The specific bulk conductivity to use. Defaults to None.
            surface_roughness (float, optional): The surface roughness. Defaults to 0.
            thickness (float | None, optional): The layer thickness. Defaults to None
            sr_model (Literal["Hammerstad-Jensen", optional): The surface roughness model. Defaults to 'Hammerstad-Jensen'.
            impedance_function (Callable, optional): A user defined surface impedance function as function of frequency.
        """
        super().__init__(face)
        
        # Backwards compatibility for this misnomed property :(
        if surface_conductance is not None:
            surface_conductivity = surface_conductance
        
        self._material: Material | None = material
        self._mur: float | complex = 1.0
        self._epsr: float | complex = 1.0
        self.sigma: float = 0.0
        self.thickness: float | None = thickness

        if isinstance(face, GeoObject) and thickness is None:
            self.thickness = face._load("thickness")

        if material is not None:
            self.sigma = material.cond.scalar(1e9)
            self._mur = material.ur
            self._epsr = material.er

        if surface_conductance is not None:
            self.sigma = surface_conductance

        self._sr: float = surface_roughness
        self._sr_model: str = sr_model
        self._Zf: Callable | None = impedance_function

    def get_basis(self) -> np.ndarray | None:
        return None

    def get_inv_basis(self) -> np.ndarray | None:
        return None

    def get_beta(self, k0: float) -> float:
        """Return the out of plane propagation constant. βz."""

        return k0

    def get_gamma(self, k0: float) -> complex:
        """Computes the γ-constant for matrix assembly. This constant is required for the Robin boundary condition.

        Args:
            k0 (float): The free space propagation constant.

        Returns:
            complex: The γ-constant
        """
        w0 = k0 * C0
        f0 = w0 / (2 * np.pi)
        if self._Zf is not None:
            return 1j * k0 * Z0 / self._Zf(f0)

        sigma = self.sigma
        mur = self._material.ur.scalar(f0)
        er = self._material.er.scalar(f0)
        eps = EPS0 * er
        mu = MU0 * mur
        rho = 1 / sigma
        d_skin = (
            2 * rho / (w0 * mu) * ((1 + (w0 * eps * rho) ** 2) ** 0.5 + rho * w0 * eps)
        ) ** 0.5
        logger.trace(f"Computed skin depth δ={d_skin * 1e6:.2}μm")
        R = (1 + 1j) * rho / d_skin
        # R = (w0*mu/(2*sigma))**0.5
        if self.thickness is not None:
            eps_c = eps - 1j * sigma / w0
            gamma_m = 1j * w0 * np.sqrt(mu * eps_c)
            R = R / np.tanh(gamma_m * self.thickness)
            logger.trace(
                f"Impedance scaler due to thickness: {1 / np.tanh(gamma_m * self.thickness):.4f}"
            )
        if self._sr_model == "Hammerstad-Jensen" and self._sr > 0.0:
            R = R * (1 + 2 / np.pi * np.arctan(1.4 * (self._sr / d_skin) ** 2))
            logger.debug(f' - Surface Roughness resistance scaler: {(1 + 2 / np.pi * np.arctan(1.4 * (self._sr / d_skin) ** 2)):.4f}')
        return 1j * k0 * Z0 / R


class ThinConductor(RobinBC, Saveable):
    _include_stiff: bool = True
    _include_mass: bool = False
    _include_force: bool = False
    _color: str = "#49e8ed"
    _name: str = "ThinConductor"
    skip_fields = ("_Zf",)
    dim: int = 2

    def __init__(
        self,
        face: FaceSelection | GeoSurface,
        material: Material | None = None,
        thickness: float | None = None,
        surface_conductance: float | None = None,
        surface_roughness: float = 0,
        sr_model: Literal["Hammerstad-Jensen"] = "Hammerstad-Jensen",
        impedance_function: Callable | None = None,
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
            thickness (float | None, optional): The layer thickness. Defaults to None
            sr_model (Literal["Hammerstad-Jensen", optional): The surface roughness model. Defaults to 'Hammerstad-Jensen'.
            impedance_function (Callable, optional): A user defined surface impedance function as function of frequency.
        """
        super().__init__(face)

        self._material: Material | None = material
        self._mur: float | complex = 1.0
        self._epsr: float | complex = 1.0
        self.sigma: float = 0.0
        self.thickness: float | None = thickness

        if isinstance(face, GeoObject) and thickness is None:
            self.thickness = face._load("thickness")

        if material is not None:
            self.sigma = material.cond.scalar(1e9)
            self._mur = material.ur
            self._epsr = material.er

        if surface_conductance is not None:
            self.sigma = surface_conductance

        self._sr: float = surface_roughness
        self._sr_model: str = sr_model
        self._Zf: Callable | None = impedance_function

        DEBUG_COLLECTOR.add_report(
            "The ThinConductor boundary condition uses a new DoF splitting algorithm which may not be perfect. If you run into any issues, please condtact info@emerge-software.com to resolve any issues! :)"
        )

    def get_basis(self) -> np.ndarray | None:
        return None

    def get_inv_basis(self) -> np.ndarray | None:
        return None

    def get_beta(self, k0: float) -> float:
        """Return the out of plane propagation constant. βz."""

        return k0

    def get_gamma(self, k0: float) -> complex:
        """Computes the γ-constant for matrix assembly. This constant is required for the Robin boundary condition.

        Args:
            k0 (float): The free space propagation constant.

        Returns:
            complex: The γ-constant
        """
        w0 = k0 * C0
        f0 = w0 / (2 * np.pi)
        if self._Zf is not None:
            return 1j * k0 * Z0 / self._Zf(f0)

        sigma = self.sigma
        mur = self._material.ur.scalar(f0)
        er = self._material.er.scalar(f0)
        eps = EPS0 * er
        mu = MU0 * mur
        rho = 1 / sigma
        d_skin = (
            2 * rho / (w0 * mu) * ((1 + (w0 * eps * rho) ** 2) ** 0.5 + rho * w0 * eps)
        ) ** (0.5)
        logger.trace(f"Computed skin depth δ={d_skin * 1e6:.2}μm")

        R = (1 + 1j) * rho / d_skin

        # R = (w0*mu/(2*sigma))**0.5
        if self.thickness is not None:
            eps_c = eps - 1j * sigma / w0
            gamma_m = 1j * w0 * (mu * eps_c) ** 0.5
            gamma_m = (1 + 1j) / d_skin
            R = R / np.tanh(gamma_m * self.thickness)
            logger.trace(
                f"Impedance scaler due to thickness: {1 / np.tanh(gamma_m * self.thickness):.4f}"
            )
        if self._sr_model == "Hammerstad-Jensen" and self._sr > 0.0:
            R = R * (1 + 2 / np.pi * np.arctan(1.4 * (self._sr / d_skin) ** 2))
        return 1j * k0 * Z0 / R
