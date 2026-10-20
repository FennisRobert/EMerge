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
from ...selection import Selection, FaceSelection, DomainSelection
from ...geometry import GeoSurface, GeoVolume
from ...bc import (
    BoundaryCondition,
    BoundaryConditionSet,
    BoundaryConditionError,
)
from ...periodic import PeriodicCell
from emsutil import Material, Saveable


class HCBoundaryConditionSet(BoundaryConditionSet):
    def __init__(self, periodic_cell: PeriodicCell | None):
        """The boundary conditions of the heat conduction physics. Use the attributes (Convection, ...) to create them.

        Args:
            periodic_cell (PeriodicCell | None): The periodic cell of the simulation, if any
        """
        super().__init__()

        self.FixedTemperatureBoundary: type[FixedTemperatureBoundary] = (
            self._construct_bc(FixedTemperatureBoundary)
        )
        self.FixedTemperatureVolume: type[FixedTemperatureVolume] = self._construct_bc(
            FixedTemperatureVolume
        )
        self.HeatFluxBoundary: type[HeatFluxBoundary] = self._construct_bc(
            HeatFluxBoundary
        )
        self.HeatFluxVolume: type[HeatFluxVolume] = self._construct_bc(HeatFluxVolume)
        self.ThermalContact: type[ThermalContact] = self._construct_bc(ThermalContact)
        self.ThinConductor: type[ThinConductor] = self._construct_bc(ThinConductor)
        self.Convection: type[Convection] = self._construct_bc(Convection)
        self.BlackBodyRadiation: type[BlackBodyRadiation] = self._construct_bc(
            BlackBodyRadiation
        )
        self.InitialTemperatureVolume: type[InitialTemperatureVolume] = (
            self._construct_bc(InitialTemperatureVolume)
        )
        self.InitialTemperatureBoundary: type[InitialTemperatureBoundary] = (
            self._construct_bc(InitialTemperatureBoundary)
        )

    def get_type(
        self,
        bctype: Literal[
            "FixedTemperatureBoundary", "FixedTemperatureVolume", "HeatFluxBoundary"
        ],
    ) -> Selection:
        """All faces with a boundary condition of a given type.

        Args:
            bctype (Literal["FixedTemperatureBoundary", "FixedTemperatureVolume", "HeatFluxBoundary"]): The boundary condition type

        Returns:
            Selection: The selection (always returned as a FaceSelection)
        """
        tags = []
        for bc in self.boundary_conditions:
            if bctype in str(bc.__class__):
                tags.extend(bc.selection.tags)
        return FaceSelection(tags)

    def CoupledEMHeating(
        self,
        selection: DomainSelection | GeoVolume,
        mwfield,
        excitation_W: list[float] | None = None,
    ):
        """Couples the volumetric heat dissipation into the thermal domain.

        To couple heating from RF losses, provide a domain selection.

        The excitation in W is a list of floats of port excitations in Watts.

        The mwfield object follows from the resultant dataset:

        Example:
        >>> data = sim.mw.run_sweep(...)
        >>> mwfield = data.field.find(freq=...)
        >>> sim.hc.bc.CoupledEMHeating(selection, mwfield, excitation)

        Args:
            selection (DomainSelection | GeoVolume): The domain to include in the coupling
            mwfield (MWField): The microwave field solution whose losses are used as heat source
            excitation_W (list[float] | None, optional): The power in Watts of each port excitation. Defaults to None
                (use the current excitation of the field).
        """
        from ..microwave.microwave_data import MWField

        mwfield: MWField = mwfield
        if excitation_W is not None:
            mwfield.set_excitations(*[x**0.5 for x in excitation_W])

        def qcallable(x, y, z):
            Q = mwfield.interpolate(x, y, z, False).scalar("Qv", "real").F
            return Q

        return self.HeatFluxVolume(selection, None, heatflux_func=qcallable)


class FixedTemperatureBoundary(BoundaryCondition, Saveable):
    _color: str = "#f70a80"
    _name: str = "FixedTemperature"
    _texture: str = "tex1.png"
    dim: int = 2

    def __init__(self, face: FaceSelection | GeoSurface, temperature_K: float):
        """Initializes a fixed temperature volumetric boundary condition.

        This boundary condition prescribes a temperature at a given volume

        Args:
            face (FaceSelection | GeoSurface): The Face assignment
            temperature_K (float): The fixed temperature.
        """
        super().__init__(face)
        self.T: float = temperature_K


class FixedTemperatureVolume(BoundaryCondition, Saveable):
    _color: str = "#f70a80"
    _name: str = "FixedTemperature"
    _texture: str = "tex1.png"
    dim: int = 3

    def __init__(self, domain: DomainSelection | GeoVolume, temperature_K: float):
        """Initializes a fixed temperature volumetric boundary condition.

        This boundary condition prescribes a temperature at a given volume

        Args:
            domain (DomainSelection | GeoVolume): The volume to apply the boundary condition to
            temperature_K (float): The temperature
        """
        super().__init__(domain)
        self.T: float = temperature_K


class InitialTemperatureBoundary(BoundaryCondition, Saveable):
    _color: str = "#f70a80"
    _name: str = "InitialTemperatureBoundary"
    _texture: str = "tex2.png"
    dim: int = 2

    def __init__(self, face: FaceSelection | GeoSurface, temperature_K: float):
        """Initializes an initial temperature volumetric boundary condition.

        This boundary condition prescribes a temperature at a given volume

        Args:
            face (FaceSelection | GeoSurface): The Face assignment
            temperature_K (float): The fixed temperature.
        """
        super().__init__(face)
        self.T: float = temperature_K


class InitialTemperatureVolume(BoundaryCondition, Saveable):
    _color: str = "#f70a80"
    _name: str = "InitialTemperatureVolume"
    _texture: str = "tex2.png"
    dim: int = 3

    def __init__(self, domain: DomainSelection | GeoVolume, temperature_K: float):
        """Initializes an initial temperature volumetric boundary condition.

        This boundary condition prescribes a temperature at a given volume

        Args:
            domain (DomainSelection | GeoVolume): The volume to apply the boundary condition to
            temperature_K (float): The temperature
        """
        super().__init__(domain)
        self.T: float = temperature_K


class HeatFluxBoundary(BoundaryCondition, Saveable):
    _color: str = "#0effa7"
    _name: str = "HeatFluxBoundary"
    _texture: str = "tex2.png"
    dim: int = 2
    _is_exclusive = False

    def __init__(self, face: FaceSelection | GeoSurface, heatflux: float):
        """Initializes a boundary heat flux boundary condition.

        This boundary condition uses a constant heat injected per square meter on the boundary.

        Args:
            face (FaceSelection | GeoSurface): The boundary to apply this heat flux to
            heatflux (float, optional): The constant heat flux
        """
        super().__init__(face)
        self.qm: float = heatflux


class HeatFluxVolume(BoundaryCondition, Saveable):
    _color: str = "#0effa7"
    _name: str = "HeatFluxVolume"
    _texture: str = "tex2.png"
    dim: int = 3
    _is_exclusive = False

    def __init__(
        self,
        face: DomainSelection | GeoVolume,
        heatflux: float | None = None,
        heatflux_func: Callable | None = None,
    ):
        """Initializes a volumetric heat flux boundary condition.

        This boundary condition either uses a constant heat injected per cubic meter or
        a heat flux defined by a function of x,y,z.


        Args:
            face (DomainSelection | GeoVolume): The domains to apply this heat flux to
            heatflux (float, optional): The constant heat flux
            heatflux_func (Callable | None, optional): The spatially dependent heat flux. Defaults to None.
        """
        super().__init__(face)

        self.qm: float = heatflux
        if heatflux_func is not None:
            self.fqm: callable = heatflux_func
        elif heatflux is not None:
            self.fqm: Callable = lambda x, y, z: np.ones_like(x) * self.qm
        else:
            raise BoundaryConditionError(
                "No heat flux provided for the HeatFluxVolume boundary condition."
            )


class ThermalContact(BoundaryCondition, Saveable):
    _color: str = "#0effa7"
    _name: str = "ThermalContact"
    _texture: str = "tex2.png"
    dim: int = 2

    def __init__(self, face: FaceSelection | GeoSurface, heat_transfer_coeff: float):
        """Initializes a Thermal contact boundary condition.

        This boundary condition must be paced between two separate domains.
        It models a finite heat-transfer coefficient between two objects.

        Args:
            face (FaceSelection | GeoSurface): The contact face between the two domains
            heat_transfer_coeff (float): The heat transfer coefficient in W/m2K
        """
        super().__init__(face)
        self.h: float = heat_transfer_coeff


class ThinConductor(BoundaryCondition, Saveable):
    _color: str = "#0effa7"
    _name: str = "ThinConductor"
    _texture: str = "tex2.png"
    dim: int = 2
    _is_exclusive = False

    def __init__(
        self,
        face: FaceSelection | GeoSurface,
        material: Material = None,
        thickness: float = None,
    ):
        """Initializes a Thin Conductor boundary condition.

        This boundary condition can be used to model thin thermal conductors like copper traces on PCBs.

        Args:
            face (FaceSelection | GeoSurface): The surface to assign the
            material (Material): The material of the thin conductor layer
            thickness (float): The thickness of the thin conductor layer
        """
        super().__init__(face)
        self.material: Material = material
        self.thickness: float = thickness

    def get_kappa(self) -> float:
        """Returns the heat transfer coefficient

        Returns:
            float: The thickness times the thermal conductivity
        """
        return self.thickness * self.material.cond_thermal.value


class Convection(BoundaryCondition, Saveable):
    _color: str = "#0effa7"
    _name: str = "ThermalContact"
    _texture: str = "tex2.png"
    dim: int = 2
    _is_exclusive = False

    def __init__(
        self,
        face: FaceSelection | GeoSurface,
        heat_transfer_coeff: float,
        Tamb_K: float,
    ):
        """Initializes a Convection boundary condition.

        The convection boundary condition models a thermal transfer coefficient h (W/m2K) to some ambient temperature Tamb_K in Kelvin

        Args:
            face (FaceSelection | GeoSurface): The boundary face
            heat_transfer_coeff (float): The heat-transfer coefficient
            Tamb_K (float): The ambient temperature.
        """
        super().__init__(face)
        self.h: float = heat_transfer_coeff
        self.Tamb: float = Tamb_K


class BlackBodyRadiation(BoundaryCondition, Saveable):
    _color: str = "#0effa7"
    _name: str = "BlackBodyRadiation"
    _texture: str = "tex2.png"
    dim: int = 2
    _is_exclusive = False

    def __init__(
        self,
        face: FaceSelection | GeoSurface,
        emissivity: float,
        Tamb_K: float,
    ):
        """Initializes a black-body radiation boundary condition to the simulation.

        It is defined by an ambient temperature Tamb_K in Kelvin and a surface emissivity.
        Notice that multiple black-body radiation boundary condition do not communicate energy with each other.

        Args:
            face (FaceSelection | GeoSurface): The face to apply the boundary condition to.
            emissivity (float): The emissivity coefficient
            Tamb_K (float): The ambient temperature that the energy is radiated into
        """
        super().__init__(face)
        self.emissivity: float = emissivity
        self.Tamb: float = Tamb_K
