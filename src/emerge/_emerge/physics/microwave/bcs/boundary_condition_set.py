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

from typing import Literal, Generator
from ....selection import FaceSelection
from ....geometry import GeoSurface
from ....bc import (
    BoundaryCondition,
    BoundaryConditionSet,
    Periodic,
    BoundaryConditionError,
)
from .port_bcs import (
    WavePortIH,
    ModalPort,
    PortBC,
    RectangularWaveguide,
    UserDefinedPort,
    LumpedPort,
    CoaxPort,
    FloquetPort,
)
from ....periodic import PeriodicCell, RectCell, HexCell
from .boundary_conditions import (
    PEC,
    PMC,
    AbsorbingBoundary,
    LumpedElement,
    SurfaceImpedance,
    ScatteredField,
    ThinConductor,
    RobinBC,
    Void,
)


class MWBoundaryConditionSet(BoundaryConditionSet):
    def __init__(self, periodic_cell: PeriodicCell | None):
        super().__init__()

        self.PEC: type[PEC] = self._construct_bc(PEC)
        self.PMC: type[PMC] = self._construct_bc(PMC)
        self.AbsorbingBoundary: type[AbsorbingBoundary] = self._construct_bc(
            AbsorbingBoundary
        )
        self.ModalPort: type[ModalPort] = self._construct_bc(ModalPort)
        self.LumpedPort: type[LumpedPort] = self._construct_bc(LumpedPort)
        self.LumpedElement: type[LumpedElement] = self._construct_bc(LumpedElement)
        self.SurfaceImpedance: type[SurfaceImpedance] = self._construct_bc(
            SurfaceImpedance
        )
        self.RectangularWaveguide: type[RectangularWaveguide] = self._construct_bc(
            RectangularWaveguide
        )
        self.Periodic: type[Periodic] = self._construct_bc(Periodic)
        self.FloquetPort: type[FloquetPort] = self._construct_bc(FloquetPort)
        self.UserDefinedPort: type[UserDefinedPort] = self._construct_bc(
            UserDefinedPort
        )
        self.CoaxPort: type[CoaxPort] = self._construct_bc(CoaxPort)
        self.WavePortIH: type[WavePortIH] = self._construct_bc(WavePortIH)
        self.ScatteredField: type[ScatteredField] = self._construct_bc(ScatteredField)
        self.ThinConductor: type[ThinConductor] = self._construct_bc(ThinConductor)

        self.Void: type[Void] = self._construct_bc(Void)
        self._cell: PeriodicCell | None = None

    def get_conductors(self) -> list[BoundaryCondition]:
        """Returns a list of all boundary conditions that ought to be considered as a "conductor"
        for the purpose of modal analyses.

        Returns:
            list[BoundaryCondition]: All conductor like boundary conditions
        """
        bcs = self.oftype(PEC)
        for bc in self.oftype(SurfaceImpedance):
            bcs.append(bc)
        for bc in self.oftype(ThinConductor):
            if bc.sigma > 10.0:
                bcs.append(bc)
        return bcs

    def get_type(
        self,
        bctype: Literal[
            "PEC",
            "ModalPort",
            "LumpedPort",
            "PMC",
            "LumpedElement",
            "RectangularWaveguide",
            "Periodic",
            "FloquetPort",
            "SurfaceImpedance",
            "ThinConductor"
        ],
    ) -> FaceSelection:
        tags = []
        for bc in self.boundary_conditions:
            if bctype in str(bc.__class__):
                tags.extend(bc.selection.tags)
        return FaceSelection(tags)

    def get_port(self, port_number: int) -> PortBC | None:
        """Return the port boundary condition object for the given port number.

        Args:
            port_number (int): The desired port number

        Returns:
            PortBC | None: _description_
        """
        for bc in self.oftype(PortBC):
            if bc.port_number == port_number:
                return bc
        return None
    
    def floquet_port(self, poly: GeoSurface, port_number: int) -> FloquetPort:
        if self._cell is None:
            raise ValueError("Periodic cel must be defined for this simulation.")
        if isinstance(self._cell, RectCell):
            port = self.FloquetPort(poly, port_number)
            port.width = self._cell.width
            port.height = self._cell.height
            port.area = port.width * port.height
        elif isinstance(self._cell, HexCell):
            port = self.FloquetPort(poly, port_number)
            port.area = self._cell.area
        self._cell._ports.append(port)
        return port

    def iter_port_modes(self) -> Generator[tuple[PortBC, int | float, int], None, None]:
        """Iterates through all port + smatrix + port mode idex

        Yields:
            Generator[tuple[PortBC, int | float, int], None, None]: _description_
        """
        ports = sorted(self.oftype(PortBC), key=lambda x: x.port_number)
        for port in ports:
            for mat_index, mode_nr in port._iter_port_numbers():
                yield port, mat_index, mode_nr

    # Checks
    def _is_excited(self) -> bool:
        for bc in self.boundary_conditions:
            if not isinstance(bc, RobinBC):
                continue
            if bc._include_force:
                return True
        raise BoundaryConditionError(
            "The simulation has no boundary conditions that insert energy. Make sure to include at least one Port into your simulation."
        )

    def _check_ports(self) -> None:
        numbers = []
        for bc in self.oftype(PortBC):
            numbers.append(bc.port_number)
        numbers = sorted(numbers)
        N = len(numbers)

        # check subsequent numbers
        if not all([pa == pb for pa, pb in zip(range(1, N + 1), numbers)]):
            raise BoundaryConditionError(
                f"Port numbers are not subsequent integers. Values are {numbers} instead of {list(range(1, N + 1))}"
            )
