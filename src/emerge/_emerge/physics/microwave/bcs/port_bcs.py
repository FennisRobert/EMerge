from ....selection import Selection, FaceSelection
from ....cs import CoordinateSystem, Axis, _parse_axis
from ....coord import Line
from ....geometry import GeoSurface, GeoObject
from ....const import Z0,EPS0, MU0
from ....cs import GCS
from ...._global import _GlobalHandler
from ....attributes import LumpedPortAttribute

from emsutil import Saveable
from typing import Literal
import numpy as np
from .boundary_conditions import RobinBC
from typing import Generator, Callable
from loguru import logger
from collections import defaultdict
from .portmode import PortMode


############################################################
#                     UTILITY FUNCTIONS                    #
############################################################


def _inner_product(
    function: Callable, x: np.ndarray, y: np.ndarray, z: np.ndarray, ax: Axis
) -> float:
    Exyz = function(x, y, z)
    return np.sum(Exyz[0, :] * ax.x + Exyz[1, :] * ax.y + Exyz[2, :] * ax.z)


class PortBC(RobinBC, Saveable):
    Zvac: float = Z0
    _color: str = "#e1bd1c"
    _texture: str = "tex5.png"
    _name: str = "PortBC"
    dim: int = 2

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
        self.port_modes: dict[int, tuple[complex, ...]] = {1: (1.0 + 0.0j,)}
        self.Z0: complex | float | None = None
        self.active: bool | None = False
        self.driven: bool = True
        self.power: float = 1.0
        self.v_integration: bool = False
        self.voltage_integration_line: list[Line] = []
        self.current_integration_line: list[Line] = []

    @property
    def voltage(self) -> complex | None:
        return None

    def _gen_port_number(self, mode_nr: int) -> int | float:
        if len(self.port_modes) == 1:
            return self.port_number
        else:
            return float(f"{self.port_number}.{mode_nr}")

    def _iter_port_numbers(self) -> Generator[tuple[int | float, int], None, None]:
        """Iterates over all the mode/port numbers to include.

        Returns:
            _type_: _description_

        Yields:
            Generator[int | float, None, None]: _description_
        """
        if len(self.port_modes) == 1:
            yield self.port_number, 1
        else:
            for number in sorted(self.port_modes.keys()):
                yield self._gen_port_number(number), number

    def _iter_modes(
        self, k0: float
    ) -> Generator[tuple[int | float, Callable], None, None]:
        if len(self.port_modes) == 1:

            def Ufunc(x, y, z):
                return self.get_Uinc(x, y, z, k0)

            yield self.port_number, Ufunc
        else:
            for mode_nr in sorted(self.port_modes.keys()):

                def Ufunc(x, y, z):
                    return self.get_Uinc(x, y, z, k0, mode_nr=mode_nr)

                yield self._gen_port_number(mode_nr), Ufunc

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

    @property
    def mode_number(self) -> int:
        return self.selected_mode + 1

    def get_beta(self, k0) -> float:
        """Return the out of plane propagation constant. βz."""
        return k0

    def get_gamma(self, k0) -> complex:
        """Computes the γ-constant for matrix assembly. This constant is required for the Robin boundary condition.

        Args:
            k0 (float): The free space propagation constant.

        Returns:
            complex: The γ-constant
        """
        return 1j * self.get_beta(k0)

    def port_mode_3d(
        self,
        xs: np.ndarray,
        ys: np.ndarray,
        k0: float,
        which: Literal["E", "H"] = "E",
        mode_nr: int = 1,
    ) -> np.ndarray:
        raise NotImplementedError("port_mode_3d not implemented for Port class")

    def port_mode_3d_global(
        self,
        x_global: np.ndarray,
        y_global: np.ndarray,
        z_global: np.ndarray,
        k0: float,
        which: Literal["E", "H"] = "E",
        mode_nr: int = 1,
    ) -> np.ndarray:
        xl, yl, _ = self.cs.in_local_cs(x_global, y_global, z_global)
        Ex, Ey, Ez = self.port_mode_3d(xl, yl, k0, mode_nr=mode_nr)
        Exg, Eyg, Ezg = self.cs.in_global_basis(Ex, Ey, Ez)
        return np.array([Exg, Eyg, Ezg])


class ModalPort(PortBC, Saveable):
    _include_stiff: bool = True
    _include_mass: bool = False
    _include_force: bool = True
    _color: str = "#e1bd1c"
    _texture: str = "tex5.png"
    _name: str = "ModalPort"
    dim: int = 2

    def __init__(
        self,
        face: FaceSelection | GeoSurface,
        port_number: int,
        cs: CoordinateSystem | None = None,
        power: float = 1,
        modetype: Literal["TE", "TM", "TEM","any"] | None = None,
        number_of_modes: int = 1,
        mixed_materials: bool = False,
        impedance_definition: Literal["PV", "PI", "VI"] = "PV",
    ):
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
            cs (CoordinateSystem, optional): The local coordinate system of the port face. Defaults to None.
            power (float, optional): The radiated power. Defaults to 1.
            modetype (str[TE, TM, TEM], optional): Wether the mode should be considered as a TEM mode. Defaults to False
            mixed_materials (bool, optional): Wether the port consists of multiple different dielectrics. This requires
                A recalculation of the port mode at every frequency
        """
        super().__init__(face)

        self.port_number: int = port_number
        self.active: bool = False
        self.power: float = power
        self.alignment_vectors: list[Axis] = []

        self.selected_mode: int = 0
        self.available_modes: dict[float, list[PortMode]] = defaultdict(list)
        self.port_modes: dict[int | float, tuple[complex, ...]] = dict()

        for i in range(1, number_of_modes + 1):
            vec = [0.0 + 0.0j for _ in range(number_of_modes)]
            vec[i - 1] = 1.0 + 0.0j
            self.port_modes[i] = vec

        self._desired_number_of_modes: int = number_of_modes
        self.forced_modetype: Literal["TE", "TM", "TEM","any"] | None = modetype
        self.mixed_materials: bool = mixed_materials
        self.initialized: bool = False
        self._first_k0: float | None = None
        self._last_k0: float | None = None

        self.plus_terminal: list[tuple[int, int]] = []
        self.minus_terminal: list[tuple[int, int]] = []
        self.N_mesh_tris: int = 50
        self.impedance_definition: Literal["PV", "PI", "VI"] = impedance_definition

        if cs is None:
            logger.info("Constructing coordinate system from port normal")
            self.cs = Axis(self.selection.normal).construct_cs()  # type: ignore
        else:
            raise ValueError("No Coordinate System could be derived.")
        # self._er: np.ndarray | None = None
        # self._ur: np.ndarray | None = None

        self.voltage_integration_line: list[Line] = []
        self.current_integration_line: list[Line] = []

    def neff_estimate(self, k0: float) -> float:
        if len(self.available_modes) == 0:
            return None
        else:
            return self.get_modes(k0)[0].neff

    @property
    def _size_constraint(self) -> float:
        area = self.selection.area
        return np.sqrt(area / self.N_mesh_tris * 4 / np.sqrt(3))

    def set_integration_line(
        self,
        c1: tuple[float, float, float],
        c2: tuple[float, float, float],
        N: int = 51,
    ) -> None:
        """Define the integration line start and end point

        Args:
            c1 (tuple[float, float, float]): The start coordinate
            c2 (tuple[float, float, float]): The end coordinate
            N (int, optional): The number of integration points. Defaults to 21.
        """
        self.voltage_integration_line.append(Line.from_points(c1, c2, N))

    def reset(self) -> None:
        self.available_modes: dict[float, list[PortMode]] = defaultdict(list)
        self.initialized: bool = False
        self.plus_terminal: list[tuple[int, int]] = []
        self.minus_terminal: list[tuple[int, int]] = []

    def portZ0(self, k0: float) -> complex | float | None:
        return self.get_modes(k0)[0].Z0

    def modetype(self, k0: float) -> Literal["TEM", "TE", "TM"]:
        return self.get_modes(k0)[0].modetype

    def get_mode_EH(self, k0: float) -> tuple[Callable, Callable, np.ndarray, float]:
        mode = self.get_modes(k0)[0]
        return mode.E_function, mode.H_function, 1.0

    def align_modes(self, *axes: tuple | np.ndarray | Axis) -> None:
        """Set a reriees of Axis objects that define a sequence of mode field
        alignments.

        The modes will be sorted to maximize the inner product: |∬ E(x,y) · ax dS|

        Args:
            *axes (tuple, np.ndarray, Axis): The alignment vectors.
        """
        self.alignment_vectors = [_parse_axis(ax) for ax in axes]

    def _get_alignment_vector(self, index: int) -> np.ndarray | None:
        if len(self.alignment_vectors) > index:
            return self.alignment_vectors[index].np
        return None

    def set_terminals(
        self,
        positive: Selection | GeoObject | None = None,
        negative: Selection | GeoObject | None = None,
        ground: Selection | GeoObject | None = None,
    ) -> None:
        """Define which objects/faces/selection should be assigned the positive terminal
        and which one the negative terminal.

        The terminal assignment will be used to find an integration line for the impedance calculation.

        Note: Ground is currently unused.

        Args:
            positive (Selection | GeoObject | None, optional): The postive terminal. Defaults to None.
            negative (Selection | GeoObject | None, optional): The negative terminal. Defaults to None.
            ground (Selection | GeoObject | None, optional): _description_. Defaults to None.
        """
        if positive is not None:
            self.plus_terminal = positive.dimtags
        if negative is not None:
            self.minus_terminal = negative.dimtags

    @property
    def nmodes(self) -> int:
        if self._last_k0 is None:
            _GlobalHandler.active().debugcollector.add_report(
                "The modal analysis turned up with no solutions. This can be because:\n"
                " - You assigned the wrong materials to geometries.\n"
                + " - You simulate at a frequency that is too low.\n"
                + " - Your mode face is not appropriately supporting a modal solution."
            )
            raise ValueError(
                "ModalPort is not properly configured. No modes are defined."
            )
        return len(self.available_modes[self._last_k0])

    @property
    def voltage(self) -> complex:
        mode = self.get_modes(0)[0]
        return np.sqrt(mode.Z0)

    def _check_mode_betas(self) -> None:
        """Performs a check if the port mode vectors are properly configured"""
        if len(self.port_modes) <= 1:
            return
        for k0, modes in self.available_modes.items():
            if len(modes) == 1:
                continue
            betas = [np.real(mode.beta) for mode in modes]
            mean_beta = sum(betas) / len(betas)
            std_beta = (
                sum([(beta - mean_beta) ** 2 for beta in betas]) / len(betas)
            ) ** 0.5
            str_betas = ",".join([f"{beta:.1f}" for beta in betas])
            if std_beta / mean_beta > 1e-2:
                logger.warning(
                    f"A variation in port mode propagation constants (k0={k0:.1f}) of {std_beta / mean_beta * 100:.2f}% is detected. Only multi mode ports with similar"
                    + f"propagation constants are suppored. Betas are {str_betas}"
                )
                _GlobalHandler.active().debugcollector.add_report(
                    "A variation in port mode propagation constants (k0={k0:.1f}) is detected. Only multi mode ports with similar"
                    + f"propagation constants are suppored. Betas are {str_betas}"
                )

    def sort_modes(self) -> None:
        """Sorts the port modes based on propagation constant"""

        if len(self.alignment_vectors) > 0:
            logger.trace(
                f"Sorting modes based on alignment vectors: {self.alignment_vectors}"
            )
            X, Y, Z = self.selection.sample(5)
            X = X.flatten()
            Y = Y.flatten()
            Z = Z.flatten()
            for k0, modes in self.available_modes.items():
                logger.trace(f"Aligning modes for k0={k0:.3f} rad/m")
                new_modes = []
                for ax in self.alignment_vectors:
                    logger.trace(f".mode vector {ax}")
                    integrals = [
                        _inner_product(m.E_function, X, Y, Z, ax) for m in modes
                    ]
                    integral, opt_mode = sorted(
                        [pair for pair in zip(integrals, modes)],
                        key=lambda x: abs(x[0]),
                        reverse=True,
                    )[0]
                    opt_mode.polarity = np.sign(integral.real)
                    logger.trace(
                        f"Optimal mode = {opt_mode} ({integral}), polarization alignment = {opt_mode.polarity}"
                    )
                    new_modes.append(opt_mode)

                self.available_modes[k0] = new_modes
            return
        for k0, modes in self.available_modes.items():
            self.available_modes[k0] = sorted(modes, key=lambda m: m.beta, reverse=True)

    def get_modes(self, k0: float) -> list[PortMode]:
        """Returns a list of mode solution in the form of a PortMode object for a specific k0.

        Args:
            k0 (float): The propagation constant

        Returns:
            PortMode: The requested PortMode object
        """
        options = self.available_modes[
            min(self.available_modes.keys(), key=lambda k: abs(k - k0))
        ]
        return options

    def global_field_function(
        self, k0: float = 0, which: Literal["E", "H"] = "E", mode_nr: int = 1
    ) -> Callable:
        """The field function used to compute the E-field.
        This field-function is defined in global coordinates (not local coordinates)."""
        modes = self.get_modes(k0)

        if which == "E":

            def modef(x, y, z):
                amplitudes = self.port_modes[mode_nr]
                out = np.zeros((3, x.shape[0]), dtype=np.complex128)
                for amp, mode in zip(amplitudes, modes):
                    if amp == 0:
                        continue
                    out += amp * (mode.E_function(x, y, z))
                return out

            return modef
        elif which == "Exy":

            def modef(x, y, z):
                amplitudes = self.port_modes[mode_nr]
                out = np.zeros((3, x.shape[0]), dtype=np.complex128)
                for amp, mode in zip(amplitudes, modes):
                    if amp == 0:
                        continue
                    out += amp * (mode.E_function.calcExy(x, y, z))
                return out

            return modef
        elif which == "gradE":

            def modef(x, y, z):
                amplitudes = self.port_modes[mode_nr]
                out = np.zeros((3, x.shape[0]), dtype=np.complex128)
                for amp, mode in zip(amplitudes, modes):
                    if amp == 0:
                        continue
                    out += amp * (mode.E_function.calcEzGrad(x, y, z))
                return out

            return modef
        elif which == "modprof":

            def modef(x, y, z):
                amplitudes = self.port_modes[mode_nr]
                out = np.zeros((3, x.shape[0]), dtype=np.complex128)
                for amp, mode in zip(amplitudes, modes):
                    if amp == 0:
                        continue
                    out += amp * (mode.E_function.calc_eff_modeprofile(x, y, z))
                return out

            return modef
        elif which == "Ez":

            def modef(x, y, z):
                amplitudes = self.port_modes[mode_nr]
                out = np.zeros((3, x.shape[0]), dtype=np.complex128)
                for amp, mode in zip(amplitudes, modes):
                    if amp == 0:
                        continue
                    out += amp * (mode.E_function.calcEz(x, y, z))
                return out

            return modef
        else:

            def modef(x, y, z):
                amplitudes = self.port_modes[mode_nr]
                out = np.zeros((3, x.shape[0]), dtype=np.complex128)
                for amp, mode in zip(amplitudes, modes):
                    if amp == 0:
                        continue
                    out += amp * (mode.H_function(x, y, z))
                return out

            return modef

    def clear_modes(self) -> None:
        """Clear all port mode data"""
        self.available_modes = defaultdict(list)
        self.initialized = False

    def try_add_mode(
        self,
        field: np.ndarray,
        E_function: Callable,
        H_function: Callable,
        beta: float,
        k0: float,
        freq: float,
    ) -> PortMode | None:
        """Add a mode function to the ModalPort

        Args:
            field (np.ndarray): The field value array
            E_function (Callable): The E-field callable
            H_function (Callable): The H-field callable
            beta (float): The out-of-plane propagation constant
            k0 (float): The free space phase constant
            freq (float): The frequency of the port mode

        Returns:
            PortMode: The port mode object.
        """
        mode = PortMode(field, E_function, H_function, k0, beta, freq=freq)
        if mode.energy * beta < 1e-8:
            logger.debug(f"Ignoring mode due to a low mode energy: {mode.energy}")
            return None

        self.available_modes[k0].append(mode)
        self.initialized = True

        self._last_k0 = k0
        if self._first_k0 is None:
            self._first_k0 = k0
        else:
            ref_field = self.get_modes(self._first_k0)[-1].modefield
            polarity = np.sign(np.sum(field * ref_field).real)
            logger.debug(f".. Mode polarity relative to other modes = {polarity}")
            mode.polarity = polarity

        return mode

    def get_basis(self) -> np.ndarray:
        return self.cs._basis

    def get_inv_basis(self) -> np.ndarray:
        return self.cs._basis_inv

    def get_beta(self, k0: float) -> float:
        mode = self.get_modes(k0)[0]
        if mode.modetype == "TEM":
            beta = mode.beta / mode.k0 * k0
        else:
            freq = k0 * 299792458 / (2 * np.pi)
            beta = np.sqrt(mode.beta**2 + k0**2 * (1 - ((mode.freq / freq) ** 2)))
        return beta

    def get_gamma(self, k0: float) -> complex:
        return 1j * self.get_beta(k0)

    def get_Uinc(
        self,
        x_global: np.ndarray,
        y_global: np.ndarray,
        z_global: np.ndarray,
        k0,
        mode_nr: int = 1,
    ) -> np.ndarray:

        return (
            -2
            * 1j
            * self.get_beta(k0)
            * self.port_mode_3d_global(
                x_global, y_global, z_global, k0, mode_nr=mode_nr
            )
        )

    def port_mode_3d(
        self,
        x_local: np.ndarray,
        y_local: np.ndarray,
        k0: float,
        which: Literal["E", "H"] = "E",
        mode_nr: int = 1,
    ) -> np.ndarray:
        x_global, y_global, z_global = self.cs.in_global_cs(
            x_local, y_local, 0 * x_local
        )

        Egxyz = self.port_mode_3d_global(
            x_global, y_global, z_global, k0, which=which, mode_nr=mode_nr
        )

        Ex, Ey, Ez = self.cs.in_local_basis(Egxyz[0, :], Egxyz[1, :], Egxyz[2, :])

        Exyz = np.array([Ex, Ey, Ez])
        return Exyz

    def port_mode_3d_global(
        self,
        x_global: np.ndarray,
        y_global: np.ndarray,
        z_global: np.ndarray,
        k0: float,
        which: Literal["E", "H", "gradE"] = "E",
        mode_nr: int = 1,
    ) -> np.ndarray:
        Ex, Ey, Ez = self.global_field_function(k0, which, mode_nr)(
            x_global, y_global, z_global
        )
        Exyz = np.array([Ex, Ey, Ez])
        return Exyz


class WavePortIH(ModalPort, Saveable):
    _include_stiff: bool = True
    _include_mass: bool = False
    _include_force: bool = True
    _color: str = "#e1bd1c"
    _texture: str = "tex5.png"
    _name: str = "WavePortIH"
    dim: int = 2

    def __init__(
        self,
        face: FaceSelection | GeoSurface,
        port_number: int,
        cs: CoordinateSystem | None = None,
        power: float = 1,
        number_of_modes: int = 1,
        impedance_definition: Literal["PV", "PI", "VI"] = "PV",
    ):
        """Generes a Wave Port Inhomogeneous boundary condition for a port that requires eigenmode solutions for the mode.

        The boundary condition requires a FaceSelection (or GeoSurface related) object for the face and a port
        number.
        If the face coordinate system is not provided a local coordinate system will be derived automatically
        by finding the plane that spans the face nodes with minimial out-of-plane error.

        All modal ports require the execution of a .modal_analysis() by the physics class to define
        the port mode.

        Args:
            face (FaceSelection, GeoSurface): The port mode face
            port_number (int): The port number as an integer
            cs (CoordinateSystem, optional): The local coordinate system of the port face. Defaults to None.
            power (float, optional): The radiated power. Defaults to 1.
            modetype (str[TE, TM, TEM], optional): Wether the mode should be considered as a TEM mode. Defaults to False
            mixed_materials (bool, optional): Wether the port consists of multiple different dielectrics. This requires
                A recalculation of the port mode at every frequency
        """
        super(ModalPort, self).__init__(face)

        self.port_number: int = port_number
        self.active: bool = False
        self.power: float = power
        self.alignment_vectors: list[Axis] = []

        self.selected_mode: int = 0
        self.available_modes: dict[float, list[PortMode]] = defaultdict(list)
        self.port_modes: dict[int | float, tuple[complex, ...]] = dict()
        self.forced_modetype = "TEM"
        for i in range(1, number_of_modes + 1):
            vec = [0.0 + 0.0j for _ in range(number_of_modes)]
            vec[i - 1] = 1.0 + 0.0j
            self.port_modes[i] = vec

        self._desired_number_of_modes: int = number_of_modes
        self.mixed_materials: bool = True
        self.initialized: bool = False
        self._first_k0: float | None = None
        self._last_k0: float | None = None

        self.plus_terminal: list[tuple[int, int]] = []
        self.minus_terminal: list[tuple[int, int]] = []
        self.N_mesh_tris: int = 50
        self.impedance_definition: Literal["PV", "PI", "VI"] = impedance_definition

        if cs is None:
            logger.info("Constructing coordinate system from port normal")
            self.cs = Axis(self.selection.normal).construct_cs()  # type: ignore
        else:
            raise ValueError("No Coordinate System could be derived.")

        self.voltage_integration_line: list[Line] = []
        self.current_integration_line: list[Line] = []

    def get_modepf_kappa(
        self, k0: float, nodes: np.ndarray, tris: np.ndarray
    ) -> tuple[Callable, complex]:
        """Return the modeprofile function γetm - ∇ez

        Args:
            k0 (_type_): _description_
        """
        mode = self.get_modes(k0)[0]

        def fmp(x, y, z):
            return mode.E_function.calc_eff_modeprofile(x, y, z)

        def fxy(x, y, z):
            return mode.E_function.calcExy(x, y, z)

        return (
            fmp,
            fxy,
            mode.compute_kappa(nodes, tris),
        )


class FloquetPort(PortBC, Saveable):
    _include_stiff: bool = True
    _include_mass: bool = False
    _include_force: bool = True
    _color: str = "#e1bd1c"
    _texture: str = "tex5.png"
    _name: str = "FloquetPort"
    dim: int = 2

    def __init__(
        self,
        face: FaceSelection | GeoSurface,
        port_number: int,
        cs: CoordinateSystem | None = None,
        power: float = 1.0,
        er: float = 1.0,
    ):
        super().__init__(face)
        if cs is None:
            cs = GCS
        self.port_number: int = port_number
        self.active: bool = False
        self.power: float = power
        self.type: str = "TEM"
        self.er: float = er
        self.cs: CoordinateSystem = cs
        self.scan_theta: float = 0
        self.scan_phi: float = 0
        self.port_modes: dict[int, tuple[complex, complex]] = {
            1: (1.0 + 0.0j, 0.0 + 0.0j),
            2: (0.0 + 0.0j, 1.0 + 0.0j),
        }
        self.area: float = 1
        self.width: float | None = None
        self.height: float | None = None

        if self.cs is None:
            self.cs = GCS

    def portZ0(self, k0: float | None = None) -> complex | float | None:
        return Z0

    def get_amplitude(self, k0: float) -> float:
        amplitude = np.sqrt(2 * Z0 * self.power / (self.area * np.cos(self.scan_theta)))
        return amplitude

    def get_beta(self, k0: float) -> float:
        """Return the out of plane propagation constant. βz."""
        return k0 * np.cos(self.scan_theta)

    def get_gamma(self, k0: float) -> complex:
        """Computes the γ-constant for matrix assembly. This constant is required for the Robin boundary condition.

        Args:
            k0 (float): The free space propagation constant.

        Returns:
            complex: The γ-constant
        """
        return 1j * self.get_beta(k0)

    def get_Uinc(
        self,
        x_global: np.ndarray,
        y_global: np.ndarray,
        z_global: np.ndarray,
        k0: float,
        mode_nr: int = 1,
    ) -> np.ndarray:
        return (
            -2
            * 1j
            * self.get_beta(k0)
            * self.port_mode_3d_global(
                x_global, y_global, z_global, k0, mode_nr=mode_nr
            )
        )

    def port_mode_3d(
        self,
        x_local: np.ndarray,
        y_local: np.ndarray,
        k0: float,
        which: Literal["E", "H"] = "E",
        mode_nr: int = 1,
    ) -> np.ndarray:
        """Compute the port mode E-field in local coordinates (XY) + Z out of plane."""

        S, P = self.port_modes[mode_nr]
        kx = k0 * np.sin(self.scan_theta) * np.cos(self.scan_phi)
        ky = k0 * np.sin(self.scan_theta) * np.sin(self.scan_phi)
        phi = np.exp(-1j * (x_local * kx + y_local * ky))

        E0 = self.get_amplitude(k0)

        Ex = (
            E0
            * (
                -S * np.sin(self.scan_phi)
                - P * np.cos(self.scan_theta) * np.cos(self.scan_phi)
            )
            * phi
        )
        Ey = (
            E0
            * (
                S * np.cos(self.scan_phi)
                - P * np.cos(self.scan_theta) * np.sin(self.scan_phi)
            )
            * phi
        )
        Ez = (P * E0 * np.sin(self.scan_theta)) * phi

        if which == "E":
            Exyz = np.array([Ex, Ey, Ez])
            return Exyz
        if which == "Exy":
            Exyz = np.array([Ex, Ey, 0 * Ez])
            return Exyz
        elif which == "H":
            Z_space = Z0 / np.sqrt(self.er)

            xu = np.sin(self.scan_theta) * np.cos(self.scan_phi)
            yu = np.sin(self.scan_theta) * np.sin(self.scan_phi)
            zu = np.cos(self.scan_theta)

            Hx = (Ey * zu - Ez * yu) / Z_space
            Hy = (Ez * xu - Ex * zu) / Z_space
            Hz = (Ex * yu - Ey * xu) / Z_space

            Hxyz = np.array([Hx, Hy, Hz])
            return Hxyz
        else:
            raise ValueError("Field parameter 'which' must be either 'E' or 'H'.")

    def port_mode_3d_global(
        self,
        x_global: np.ndarray,
        y_global: np.ndarray,
        z_global: np.ndarray,
        k0: float,
        which: Literal["E", "H"] = "E",
        mode_nr: int = 1,
    ) -> np.ndarray:
        """Compute the port mode field for global xyz coordinates."""
        if self.cs is None:
            raise ValueError("No coordinate system is defined for this FloquetPort")
        xl, yl, _ = self.cs.in_local_cs(x_global, y_global, z_global)
        Ex, Ey, Ez = self.port_mode_3d(xl, yl, k0, mode_nr=mode_nr, which=which)
        Exg, Eyg, Ezg = self.cs.in_global_basis(Ex, Ey, Ez)
        return np.array([Exg, Eyg, Ezg])


class RectangularWaveguide(PortBC, Saveable):
    _include_stiff: bool = True
    _include_mass: bool = False
    _include_force: bool = True
    _color: str = "#e1bd1c"
    _name: str = "RectWG"
    _texture: str = "tex5.png"

    def __init__(
        self,
        face: FaceSelection | GeoSurface,
        port_number: int,
        mode: tuple[int, int] = (1, 0),
        er: float = 1.0,
        cs: CoordinateSystem | None = None,
        dims: tuple[float, float] | None = None,
        power: float = 1,
    ):
        """Creates a rectangular waveguide as a port boundary condition.

        Currently the Rectangular waveguide only supports TE0n modes. The mode field
        is derived analytically. The local face coordinate system and dimensions can be provided
        manually. If not provided the class will attempt to derive the local coordinate system and
        face dimensions itself. It always orients the longest edge along the local X-direction.
        The information on the derived coordiante system will be shown in the DEBUG level logs.

        Args:
            face (FaceSelection, GeoSurface): The port boundary face selection
            port_number (int): The port number
            mode: (tuple[int, int], optional): The TE mode number. Defaults to (1,0).
            er: (float, optional): The Dielectric constant. Defaults to 1.0.
            cs (CoordinateSystem, optional): The local coordinate system. Defaults to None.
            dims (tuple[float, float], optional): The port face. Defaults to None.
            power (float): The port power. Default to 1.
        """
        super().__init__(face)

        self.port_number: int = port_number
        self.active: bool = False
        self.power: float = power
        self.type: str = "TE"
        self.mode: tuple[int, int] = mode
        self.mode_axis: Axis | None = None
        self.er: float = er
        self._polarization: float = 1.0

        if dims is None:
            logger.debug(
                f" - Establishing RectangularWaveguide port face based on selection {self.selection}"
            )
            cs, (width, height) = self.selection.rect_basis()  # type: ignore
            self.cs = cs  # type: ignore
            self.dims = (width, height)
            logger.debug(f" - Port CS: {self.cs}")
            logger.debug(
                f" - Detected port {self.port_number} size = {width * 1000:.1f} mm x {height * 1000:.1f} mm"
            )
        else:
            self.dims = dims
            self.cs = cs

        if self.cs is None:
            logger.info(" - Constructing coordinate system from port normal")
            self.cs = Axis(self.selection.normal).construct_cs()
            logger.debug(f" - Port CS: {self.cs}")

    def align_modes(self, axis: Axis) -> None:
        """Defines the positive E-field direction for the fundamental TE mode.

        Args:
            axis (Axis): The alignment vector for the mode
        """

        self.mode_axis = axis
        self._polarization: float = float(np.sign(self.cs.yax.dot(self.mode_axis)))
        if self._polarization == 0.0:
            self._polarization = 1.0

    def get_basis(self) -> np.ndarray:
        return self.cs._basis

    def get_inv_basis(self) -> np.ndarray:
        return self.cs._basis_inv

    def portZ0(self, k0: float) -> complex:
        return k0 * 299792458 * MU0 / self.get_beta(k0)

    def modetype(self, k0):
        return self.type

    def get_amplitude(self, k0: float) -> float:
        Zte = k0 * 299792458 * MU0 / self.get_beta(k0)
        width = self.dims[0]
        height = self.dims[1]
        m, n = self.mode
        scale_m = 2.0 if m == 0 else 1.0
        scale_n = 2.0 if n == 0 else 1.0
        multiplier = 8.0 / (scale_m * scale_n)
        amplitude = np.sqrt(self.power * multiplier * Zte / (width * height))
        return amplitude

    def get_beta(self, k0: float) -> float:
        """Return the out of plane propagation constant. βz."""
        width = self.dims[0]
        height = self.dims[1]
        beta = np.sqrt(
            self.er * k0**2
            - (np.pi * self.mode[0] / width) ** 2
            - (np.pi * self.mode[1] / height) ** 2
        )
        return beta

    def get_gamma(self, k0: float) -> complex:
        """Computes the γ-constant for matrix assembly. This constant is required for the Robin boundary condition.

        Args:
            k0 (float): The free space propagation constant.

        Returns:
            complex: The γ-constant
        """
        return 1j * self.get_beta(k0)

    def get_Uinc(
        self,
        x_global: np.ndarray,
        y_global: np.ndarray,
        z_global: np.ndarray,
        k0: float,
        mode_nr: int = None,
    ) -> np.ndarray:
        return (
            -2
            * 1j
            * self.get_beta(k0)
            * self.port_mode_3d_global(
                x_global, y_global, z_global, k0, mode_nr=mode_nr
            )
        )

    def port_mode_3d(
        self,
        x_local: np.ndarray,
        y_local: np.ndarray,
        k0: float,
        mode_nr: int = None,
        which: Literal["E", "H"] = "E",
    ) -> np.ndarray:
        """Compute the port mode E-field in local coordinates (XY) + Z out of plane."""
        width = self.dims[0]
        height = self.dims[1]
        m, n = self.mode
        Ev = (
            self._polarization
            * self.get_amplitude(k0)
            * np.cos(np.pi * m * (x_local) / width)
            * np.cos(np.pi * n * (y_local) / height)
        )
        Eh = (
            self._polarization
            * self.get_amplitude(k0)
            * np.sin(np.pi * m * (x_local) / width)
            * np.sin(np.pi * n * (y_local) / height)
        )
        Ex = Eh
        Ey = Ev
        Ez = 0 * Eh

        if which == "E":
            return np.array([Ex, Ey, Ez])
        elif which == "Exy":
            return np.array([Ex, Ey, 0 * Ez])
        elif which == "H":
            Z_te = self.portZ0(k0)
            omega_mu = k0 * 299792458 * 1.25663706212e-6

            Hx = Ey / Z_te
            Hy = -Ex / Z_te

            dEy_dx = (
                self._polarization
                * self.get_amplitude(k0)
                * (-np.pi * m / width)
                * np.sin(np.pi * m * x_local / width)
                * np.cos(np.pi * n * y_local / height)
            )
            dEx_y = (
                self._polarization
                * self.get_amplitude(k0)
                * (np.pi * n / height)
                * np.sin(np.pi * m * x_local / width)
                * np.cos(np.pi * n * y_local / height)
            )
            Hz = -1j * (dEy_dx - dEx_y) / omega_mu

            Hxyz = np.array([Hx, Hy, Hz])
            return Hxyz
        else:
            raise ValueError(
                f"Field parameter 'which' must be either 'E' or 'H', not {which}"
            )

    def port_mode_3d_global(
        self,
        x_global: np.ndarray,
        y_global: np.ndarray,
        z_global: np.ndarray,
        k0: float,
        mode_nr: int = None,
        which: Literal["E", "H"] = "E",
    ) -> np.ndarray:
        """Compute the port mode field for global xyz coordinates."""
        xl, yl, _ = self.cs.in_local_cs(x_global, y_global, z_global)
        Ex, Ey, Ez = self.port_mode_3d(xl, yl, k0, which=which)
        Exg, Eyg, Ezg = self.cs.in_global_basis(Ex, Ey, Ez)
        return np.array([Exg, Eyg, Ezg])


class CoaxPort(PortBC, Saveable):
    _include_stiff: bool = True
    _include_mass: bool = False
    _include_force: bool = True
    _color: str = "#e1bd1c"
    _name: str = "Coax"
    _texture: str = "tex5.png"

    def __init__(
        self,
        face: FaceSelection | GeoSurface,
        port_number: int,
        rad_in_out: tuple[float, float],
        cs: CoordinateSystem,
        er: float = 1.0,
        power: float = 1,
    ):
        """Creates a rectangular waveguide as a port boundary condition.

        Currently the Rectangular waveguide only supports TE0n modes. The mode field
        is derived analytically. The local face coordinate system and dimensions can be provided
        manually. If not provided the class will attempt to derive the local coordinate system and
        face dimensions itself. It always orients the longest edge along the local X-direction.
        The information on the derived coordiante system will be shown in the DEBUG level logs.

        Args:
            face (FaceSelection, GeoSurface): The port boundary face selection
            port_number (int): The port number
            mode: (tuple[int, int], optional): The TE mode number. Defaults to (1,0).
            er: (float, optional): The Dielectric constant. Defaults to 1.0.
            cs (CoordinateSystem, optional): The local coordinate system. Defaults to None.
            dims (tuple[float, float], optional): The port face. Defaults to None.
            power (float): The port power. Default to 1.
        """
        super().__init__(face)

        self.port_number: int = port_number
        self.active: bool = False
        self.power: float = power
        self.type: str = "TEM"
        self.er: float = er
        self._polarization: float = 1.0
        self.rad_in_out: tuple[float, float] = rad_in_out
        self.cs: CoordinateSystem = cs
        self.center: np.ndarray = self.cs.origin
        self.normal: Axis = self.cs.zax
        self.N_mesh_tris: int = 50
        # points = self.selection.points

    @property
    def _size_constraint(self) -> float:
        area = self.selection.area
        return np.sqrt(area / self.N_mesh_tris * 4 / np.sqrt(3))

    def get_basis(self) -> np.ndarray:
        return self.cs._basis

    def get_inv_basis(self) -> np.ndarray:
        return self.cs._basis_inv

    def portZ0(self, k0: float) -> complex:
        return k0 * 299792458 * MU0 / self.get_beta(k0)

    def modetype(self, k0):
        return self.type

    def get_amplitude(self, k0: float) -> float:
        Zte = Z0
        amplitude = np.sqrt(self.power * 4 * Zte / (self.dims[0] * self.dims[1]))
        return amplitude

    def get_beta(self, k0: float) -> float:
        """Return the out of plane propagation constant. βz."""
        return np.sqrt(self.er) * k0

    def get_gamma(self, k0: float) -> complex:
        """Computes the γ-constant for matrix assembly. This constant is required for the Robin boundary condition.

        Args:
            k0 (float): The free space propagation constant.

        Returns:
            complex: The γ-constant
        """
        return 1j * self.get_beta(k0)

    def get_Uinc(
        self,
        x_global: np.ndarray,
        y_global: np.ndarray,
        z_global: np.ndarray,
        k0: float,
        mode_nr: int = None,
    ) -> np.ndarray:
        return (
            -2
            * 1j
            * self.get_beta(k0)
            * self.port_mode_3d_global(
                x_global, y_global, z_global, k0, mode_nr=mode_nr
            )
        )

    def port_mode_3d(
        self,
        x_local: np.ndarray,
        y_local: np.ndarray,
        k0: float,
        mode_nr: int = None,
        which: Literal["E", "H"] = "E",
    ) -> np.ndarray:
        """Compute the port mode E-field in local coordinates (XY) + Z out of plane."""
        # Constants
        eta = Z0 / np.sqrt(self.er)

        Ri, Ro = self.rad_in_out

        pZ0 = (eta / (2 * np.pi)) * np.log(Ro / Ri)

        # 2. Relate Power to Voltage Amplitude: Pout = |V0|^2 / (2 * Z0)
        V0 = np.sqrt(2 * pZ0 * self.power)

        # 3. Geometric calculations
        rho = np.sqrt(x_local**2 + y_local**2)
        phi = np.arctan2(y_local, x_local)

        # 4. Magnitude of Eradial
        E_rho_mag = V0 / (rho * np.log(Ro / Ri))

        # 5. Account for Phase (Propagation along z)
        E_rho = E_rho_mag

        # 6. Convert Cylindrical (rho) to Cartesian (x, y)
        Ex = (E_rho * np.cos(phi)).astype(np.complex128)
        Ey = (E_rho * np.sin(phi)).astype(np.complex128)
        Ez = 0.0 * Ex  # TEM mode has no longitudinal component

        Exyz = np.array([Ex, Ey, Ez])
        if which == "E":
            Exyz = np.array([Ex, Ey, Ez])
            return Exyz
        if which == "Exy":
            Exyz = np.array([Ex, Ey, Ez])
            return Exyz
        elif which == "H":
            Hx = -Ey / eta
            Hy = Ex / eta
            Hz = 0.0 * Hx

            Hxyz = np.array([Hx, Hy, Hz])
            return Hxyz
        else:
            raise ValueError("Field parameter 'which' must be either 'E' or 'H'.")

    def port_mode_3d_global(
        self,
        x_global: np.ndarray,
        y_global: np.ndarray,
        z_global: np.ndarray,
        k0: float,
        mode_nr: int = None,
        which: Literal["E", "H"] = "E",
    ) -> np.ndarray:
        """Compute the port mode field for global xyz coordinates."""
        xl, yl, _ = self.cs.in_local_cs(x_global, y_global, z_global)
        Ex, Ey, Ez = self.port_mode_3d(xl, yl, k0, which=which)
        Exg, Eyg, Ezg = self.cs.in_global_basis(Ex, Ey, Ez)
        return np.array([Exg, Eyg, Ezg])


def _f_zero(k0, x, y, z):
    "Zero field function"
    return np.zeros_like(x, dtype=np.complex128)


class UserDefinedPort(PortBC, Saveable):
    _include_stiff: bool = True
    _include_mass: bool = False
    _include_force: bool = True
    _color: str = "#be9f11"
    _name: str = "UserDefined"
    _texture: str = "tex5.png"
    skip_fields = ("_fex", "_fey", "_fez", "_fkz")
    dim: int = 2

    def __init__(
        self,
        face: FaceSelection | GeoSurface,
        port_number: int,
        Ex: Callable | None = None,
        Ey: Callable | None = None,
        Ez: Callable | None = None,
        kz: Callable | None = None,
        power: float = 1.0,
        modetype: Literal["TEM", "TE", "TM"] = "TEM",
        cs: CoordinateSystem | None = None,
    ):
        """Creates a user defined port field

        The UserDefinedPort is defined based on user defined field callables. All undefined callables will default to 0 field or k0.

        All spatial field functions should be defined using the template:
        >>> def Ec(k0: float, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray
        >>>     return #shape like x

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
        self.port_number: int = port_number
        self.active: bool = False
        self.power: float = power
        self.type: str = "TE"
        if Ex is None:
            Ex = _f_zero
        if Ey is None:
            Ey = _f_zero
        if Ez is None:
            Ez = _f_zero
        if kz is None:
            kz = lambda k0: k0

        self._fex: Callable = Ex
        self._fey: Callable = Ey
        self._fez: Callable = Ez
        self._fkz: Callable = kz
        self.type = modetype

    def get_basis(self) -> np.ndarray:
        return self.cs._basis

    def get_inv_basis(self) -> np.ndarray:
        return self.cs._basis_inv

    def modetype(self, k0):
        return self.type

    def get_amplitude(self, k0: float) -> float:
        return np.sqrt(self.power)

    def get_beta(self, k0: float) -> float:
        """Return the out of plane propagation constant. βz."""
        return self._fkz(k0)

    def get_gamma(self, k0: float) -> complex:
        """Computes the γ-constant for matrix assembly. This constant is required for the Robin boundary condition.

        Args:
            k0 (float): The free space propagation constant.

        Returns:
            complex: The γ-constant
        """
        return 1j * self.get_beta(k0)

    def get_Uinc(
        self,
        x_global: np.ndarray,
        y_global: np.ndarray,
        z_global: np.ndarray,
        k0: float,
        mode_nr: int = 1,
    ) -> np.ndarray:
        return (
            -2
            * 1j
            * self.get_beta(k0)
            * self.port_mode_3d_global(x_global, y_global, z_global, k0)
        )

    def port_mode_3d(
        self,
        x_local: np.ndarray,
        y_local: np.ndarray,
        k0: float,
        which: Literal["E", "H"] = "E",
        mode_nr: int = 1,
    ) -> np.ndarray:
        x_global, y_global, z_global = self.cs.in_global_cs(
            x_local, y_local, 0 * x_local
        )

        Egxyz = self.port_mode_3d_global(x_global, y_global, z_global, k0, which=which)

        Ex, Ey, Ez = self.cs.in_local_basis(Egxyz[0, :], Egxyz[1, :], Egxyz[2, :])

        Exyz = np.array([Ex, Ey, Ez])
        return Exyz

    def port_mode_3d_global(
        self,
        x_global: np.ndarray,
        y_global: np.ndarray,
        z_global: np.ndarray,
        k0: float,
        which: Literal["E", "H"] = "E",
        mode_nr: int = 1,
    ) -> np.ndarray:
        """Compute the port mode field for global xyz coordinates."""
        xl, yl, _ = self.cs.in_local_cs(x_global, y_global, z_global)
        Ex = self._fex(k0, x_global, y_global, z_global)
        Ey = self._fey(k0, x_global, y_global, z_global)
        Ez = self._fez(k0, x_global, y_global, z_global)
        Exg, Eyg, Ezg = self.cs.in_global_basis(Ex, Ey, Ez)
        return np.array([Exg, Eyg, Ezg])


class LumpedPort(PortBC, Saveable):
    _include_stiff: bool = True
    _include_mass: bool = False
    _include_force: bool = True
    _color: str = "#e1851c"
    _name: str = "LumpedPort"
    _texture: str = "tex5.png"
    dim: int = 2

    def __init__(
        self,
        face: FaceSelection | GeoSurface,
        port_number: int,
        width: float | None = None,
        height: float | None = None,
        direction: Axis | tuple[float, float, float] | None = None,
        power: float = 1,
        Z0: float = 50,
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
            power (float, optional): The port output power. Defaults to 1.
            Z0 (float, optional): The port impedance. Defaults to 50.
        """
        super().__init__(face)

        if width is None:
            if not isinstance(face, GeoObject):
                raise ValueError(
                    f"The width, height and direction must be defined. Information cannot be extracted from {face}"
                )
            if attr := face.properties.get(LumpedPortAttribute):
                width = attr.width
                height = attr.height
                direction = attr.direction

        if width is None or height is None or direction is None:
            raise ValueError(
                f"The width, height and direction could not be extracted from {face}"
            )

        logger.debug(
            f"Lumped port: width={1000 * width:.1f}mm, height={1000 * height:.1f}mm, direction={direction}"
        )  # type: ignore
        self.port_number: int = port_number
        self.active: bool = False
        self.power: float = power
        self.Z0: float = Z0

        self.width: float = abs(width)
        self.height: float = abs(height)  # type: ignore
        self.Vdirection: Axis = _parse_axis(direction)  # type: ignore
        self.type = "TEM"

        # self.cs = Axis(self.selection.normal).construct_cs()  # type: ignore
        self.cs = GCS
        self.voltage_integration_line: list[Line] = []
        self.v_integration = True

        # Sanity checks
        if self.width > 0.5 or self.height > 0.5:
            _GlobalHandler.active().debugcollector.add_report(
                f"{self}: A lumped port width/height larger than 0.5m has been detected: width={self.width:.3f}m. Height={self.height:.3f}.m. Perhaps you forgot a unit like mm, um, or mil"
            )

    @property
    def _size_constraint(self) -> float:
        return min(self.width, self.height) / 4

    @property
    def surfZ(self) -> float:
        """The surface sheet impedance for the lumped port

        Returns:
            float: The surface sheet impedance
        """
        return self.Z0 * self.width / self.height

    @property
    def voltage(self) -> float:
        """The Port voltage required for the provided output power (time average)

        Returns:
            float: The port voltage
        """
        return np.sqrt(2 * self.power * self.Z0)

    def get_basis(self) -> np.ndarray:
        return self.cs._basis

    def get_inv_basis(self) -> np.ndarray:
        return self.cs._basis_inv

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
        return 1j * k0 * Z0 / self.surfZ

    def get_Uinc(
        self,
        x_global: np.ndarray,
        y_global: np.ndarray,
        z_global: np.ndarray,
        k0,
        mode_nr: int = 1,
    ) -> np.ndarray:
        Emag = -1j * 2 * k0 * self.voltage / self.height * (Z0 / self.surfZ)
        return Emag * self.port_mode_3d_global(x_global, y_global, z_global, k0)

    def port_mode_3d(
        self,
        x_local: np.ndarray,
        y_local: np.ndarray,
        k0: float,
        which: Literal["E", "H"] = "E",
        mode_nr: int = 1,
    ) -> np.ndarray:
        """Compute the port mode E-field in local coordinates (XY) + Z out of plane."""
        raise RuntimeError("This function should never be called in this context.")

    def port_mode_3d_global(
        self,
        x_global: np.ndarray,
        y_global: np.ndarray,
        z_global: np.ndarray,
        k0: float,
        which: Literal["E", "H"] = "E",
        mode_nr: int = 1,
    ) -> np.ndarray:
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
        ON = np.ones_like(x_global)
        Ex, Ey, Ez = self.Vdirection.np
        return np.array([Ex * ON, Ey * ON, Ez * ON])
