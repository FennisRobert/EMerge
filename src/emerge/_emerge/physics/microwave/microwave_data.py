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
from ...simulation_data import BaseDataset, DataContainer
from ...elements.femdata import FEMBasis
from dataclasses import dataclass
import numpy as np
from typing import Literal, Callable
from loguru import logger
from .adaptive_freq import SparamModel
from ...cs import Axis, _parse_axis
from ...selection import FaceSelection, DomainSelection
from ...geometry import GeoSurface
from ...mesh3d import Mesh3D
from ...const import MU0
from ...coord import Line
from emsutil.emdata import EHField, EHFieldFF, DataStructure
from ...file import Saveable
from .bcs import background_field as bf
from .bcs import LumpedElement

EMField = Literal[
    "er",
    "ur",
    "freq",
    "k0",
    "_Spdata",
    "_Spmapping",
    "_field",
    "_basis",
    "Nports",
    "Ex",
    "Ey",
    "Ez",
    "Hx",
    "Hy",
    "Hz",
    "mode",
    "beta",
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
    ux, uy, uz = vectors[:, 0], vectors[:, 1], vectors[:, 2]

    theta = np.arccos(uz)  # theta = arcsin(z)
    phi = np.arctan2(uy, ux)  # phi = atan2(y, x)

    return theta, phi


def renormalise_s(
    S: np.ndarray,
    Zn: np.ndarray | float | complex,
    Z0: np.ndarray | float | complex = 50,
) -> np.ndarray:
    """
    Renormalise S-parameters to a new reference impedance.

    Implements the renormalisation formula based on power wave theory from:
    K. Kurokawa, "Power Waves and the Scattering Matrix,"
    IEEE MTT, vol. 13, no. 2, pp. 194-202, March 1965

    Parameters
    ----------
    S : np.ndarray
        S-parameters with shape (M, N, N) where M is number of frequency points
        and N is number of ports
    Zn : np.ndarray | float | complex
        Original reference impedance(s). Can be:
        - scalar: same impedance for all ports and frequencies
        - 1D array with shape (N,): different impedance per port (same for all frequencies)
        - 1D array with shape (M,): different impedance per frequency (same for all ports)
          Note: When M == N, 1D arrays are ambiguous and not allowed. Use 2D arrays instead.
        - 2D array with shape (M, N): different impedance per frequency and port
    Z0 : np.ndarray | float | complex
        New reference impedance(s). Same shape options as Zn.
        Default is 50.

    Returns
    -------
    np.ndarray
        Renormalised S-parameters with same shape as input S
    """
    # Input validation
    S = np.asarray(S, dtype=complex)

    N = S.shape[1]
    if S.shape[1:3] != (N, N):
        raise ValueError("S must have shape (M, N, N) with same N on both axes")

    M = S.shape[0]

    # Broadcast Zn to shape (M, N)
    Zn = np.asarray(Zn, dtype=complex)
    if Zn.ndim == 0:  # scalar
        Zn = np.full((M, N), Zn)
    elif Zn.ndim == 1:
        if M == N:
            raise ValueError(
                f"When M == N ({M}), 1D Zn arrays are ambiguous. "
                f"Use a 2D array with shape ({M}, {N}) instead."
            )
        elif len(Zn) == N:  # 1D array with shape (N,) - per port
            Zn = np.tile(Zn, (M, 1))
        elif len(Zn) == M:  # 1D array with shape (M,) - per frequency
            Zn = np.tile(Zn.reshape(-1, 1), (1, N))
        else:
            raise ValueError(
                f"1D Zn must have length {N} (ports) or {M} (frequencies), "
                f"got length {len(Zn)}"
            )
    elif Zn.ndim == 2:
        if Zn.shape != (M, N):
            raise ValueError(f"2D Zn must have shape ({M}, {N}), got {Zn.shape}")
    else:
        raise ValueError(f"Zn must be scalar, 1D, or 2D array, got shape {Zn.shape}")

    # Broadcast Z0 to shape (M, N)
    Z0 = np.asarray(Z0, dtype=complex)
    if Z0.ndim == 0:  # scalar
        Z0 = np.full((M, N), Z0)
    elif Z0.ndim == 1:
        if M == N:
            raise ValueError(
                f"When M == N ({M}), 1D Z0 arrays are ambiguous. "
                f"Use a 2D array with shape ({M}, {N}) instead."
            )
        elif len(Z0) == N:  # 1D array with shape (N,) - per port
            Z0 = np.tile(Z0, (M, 1))
        elif len(Z0) == M:  # 1D array with shape (M,) - per frequency
            Z0 = np.tile(Z0.reshape(-1, 1), (1, N))
        else:
            raise ValueError(
                f"1D Z0 must have length {N} (ports) or {M} (frequencies), "
                f"got length {len(Z0)}"
            )
    elif Z0.ndim == 2:
        if Z0.shape != (M, N):
            raise ValueError(f"2D Z0 must have shape ({M}, {N}), got {Z0.shape}")
    else:
        raise ValueError(f"Z0 must be scalar, 1D, or 2D array, got shape {Z0.shape}")

    # Constant matrices
    I_N = np.eye(N, dtype=complex)
    S0 = np.empty_like(S)

    for k in range(M):
        # Extract data for this frequency point
        Znk = Zn[k, :]
        Z0k = Z0[k, :]
        Sk = S[k, :, :]

        # Diagonal matrices related to original reference impedance Zn
        # Fᵢ = 1 / (2 √|Re Zᵢ|)
        F = np.diag(0.5 / np.sqrt(np.abs(np.real(Znk))))
        # Gᵢ = Zᵢ
        G = np.diag(Znk)
        # same for target Z₀ for F' and G'
        Fp = np.diag(0.5 / np.sqrt(np.abs(np.real(Z0k))))
        Gp = np.diag(Z0k)

        # Renormalise S-parameters
        # Γ = (G' - G) (G' + G⁺)⁻¹
        Gamma = (Gp - G) @ np.linalg.inv(Gp + G.conj().T)
        # A = (F')⁻¹ F (I - Γ⁺)
        A = np.linalg.inv(Fp) @ F @ (I_N - Gamma.conj().T)
        # S' = A⁻¹ (S - Γ⁺) (I - Γ S)⁻¹ A⁺
        S0[k, :, :] = (
            np.linalg.inv(A)
            @ (Sk - Gamma.conj().T)
            @ np.linalg.inv(I_N - Gamma @ Sk)
            @ A.conj().T
        )

    return S0


def generate_ndim(
    outer_data: dict[str, list[float]],
    inner_data: list[float],
    outer_labels: tuple[str, ...],
) -> tuple[np.ndarray, ...]:
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
        self.map: dict[int | float, int] = {
            label: idx for idx, label in enumerate(port_nrs)
        }
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
    def __setitem__(self, key: tuple[int | float, int | float], value: complex) -> None:
        port1, port2 = key
        self.set(port1, port2, value)


@dataclass
class PortProperties(Saveable):
    port_number: int = -1
    k0: float | None = None
    beta: float | None = None
    Z0: float | complex | None = None
    Pout: float | None = None
    mode_number: int = 1
    smat_index: int | float = 1


class MWData(Saveable):
    scalar: BaseDataset[MWScalar, MWScalarNdim]
    field: BaseDataset[MWField, None]

    def __init__(self):
        self.scalar = BaseDataset[MWScalar, MWScalarNdim](MWScalar, MWScalarNdim, True)
        self.field = BaseDataset[MWField, None](MWField, None, False)
        self.sim: DataContainer = DataContainer()

    def merge_with(self, *others: MWData) -> MWData:
        """Merges this dataset with other datasets

        Returns:
            MWData: the merged dataset
        """
        self.sim.merge_with(*[other.sim for other in others])
        self.scalar.merge_with(*[other.scalar for other in others])
        self.field.merge_with(*[other.field for other in others])
        return self

    def setreport(self, report, **vars):
        self.sim.new(**vars)["report"] = report

    def export_farfields(
        self,
        filename: str,
        face: FaceSelection | GeoSurface,
        thetas: np.ndarray,
        phis: np.ndarray,
        origin: tuple[float, float, float] | None = None,
        syms: list[Literal["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"]] | None = None,
        precision: int = 4,
        frequencies: list[float] | None = None,
        **parameters,
    ) -> None:
        """Exports all farfield data to a file.

        Args:
            filename (str): The filename to export to
            face (FaceSelection | GeoSurface): The integration surface for the farfield calculation.
            thetas (np.ndarray): An optional array of theta angles
            phis (np.ndarray): An optional array of phi angles
            origin (tuple[float, float, float] | None, optional): An optional array for a radiation origin 
                used to determine the normal vectors of farfield boundaries if inside vs. outside 
                is not well defined. Defaults to None.
            syms ("Ex","Ey","Ez","Hx","Hy","Hz" | None), optional): Optional simulation domain symmetries. Defaults to None.
            precision (int, optional): The number of decimals for the output file. Defaults to 4.
            frequencies (list[float] | None, optional): The frequencies to pick for the output. Defaults to None.
        """
        from emsutil.inexport.ffdata import export_ffdata

        if frequencies is None:
            frequencies = self.scalar.axis("freq")

        ffsets = []
        freq_data = []
        for freq in frequencies:
            field = self.field.find(freq=freq, **parameters)
            freq_data.append(field.freq)
            ffsets.append(field.farfield_3d(face, thetas, phis, origin, syms))

        export_ffdata(
            filename, thetas, phis, np.array(freq_data), ffsets, precision=precision
        )


class _EHSign(Saveable):
    """A small class to manage the sign of field components when computing the far-field with Stratton-Chu"""

    def __init__(self):
        self.Ex = 1
        self.Ey = 1
        self.Ez = 1
        self.Hx = 1
        self.Hy = 1
        self.Hz = 1

    def fE(self):
        self.Ex = -1 * self.Ex
        self.Ey = -1 * self.Ey
        self.Ez = -1 * self.Ez

    def fH(self):
        self.Hx = -1 * self.Hx
        self.Hy = -1 * self.Hy
        self.Hz = -1 * self.Hz

    def fX(self):
        self.Ex = -1 * self.Ex
        self.Hx = -1 * self.Hx

    def fY(self):
        self.Ey = -1 * self.Ey
        self.Hy = -1 * self.Hy

    def fZ(self):
        self.Ez = -1 * self.Ez
        self.Hz = -1 * self.Hz

    def apply(self, symmetry: str):
        f, c = symmetry
        if f == "E":
            self.fE()
        elif f == "H":
            self.fH()

        if c == "x":
            self.fX()
        elif c == "y":
            self.fY()
        elif c == "z":
            self.fZ()

    def flip_field(self, E: tuple, H: tuple):
        Ex, Ey, Ez = E
        Hx, Hy, Hz = H
        return (Ex * self.Ex, Ey * self.Ey, Ez * self.Ez), (
            Hx * self.Hx,
            Hy * self.Hy,
            Hz * self.Hz,
        )


class MWField(Saveable):
    def __init__(self):
        self._der: np.ndarray = None
        self._dur: np.ndarray = None
        self._dsig: np.ndarray = None
        self.freq: float = None
        self.Q: float = None
        self.basis: FEMBasis = None
        self._fields: dict[int | int, np.ndarray] = dict()
        self._mode_field: np.ndarray = None
        self.excitation: dict[int | float, complex] = dict()
        self.Nports: int = None
        self.port_modes: list[PortProperties] = []
        self.background_fields: list[bf.BackgroundField] = []
        self.Ex: np.ndarray = None
        self.Ey: np.ndarray = None
        self.Ez: np.ndarray = None
        self.Hx: np.ndarray = None
        self.Hy: np.ndarray = None
        self.Hz: np.ndarray = None
        self.er: np.ndarray = None
        self.ur: np.ndarray = None
        self.sig: np.ndarray = None

        self._rel: bool = False
        self._Sp: np.ndarray | None = None
        self._Texcite: np.ndarray = 1.0
        self._silent: bool = False

        # Ports embedded via embed_external_component are marked inactive
        # here rather than removed -- the port list, _Sp, and _Texcite
        # always keep their original full size, so port labels are
        # permanent and never need renumbering or remapping.
        self._active_ports: set[int | float] | None = None

        self._bstags = None
        self._bssurf = None

    def add_port_properties(
        self,
        port_number: int,
        mode_number: int,
        smat_index: int | float,
        k0: float,
        beta: float,
        Z0: float | complex | None,
        Pout: float,
    ) -> None:
        self.port_modes.append(
            PortProperties(
                port_number=port_number,
                mode_number=mode_number,
                smat_index=smat_index,
                k0=k0,
                beta=beta,
                Z0=Z0,
                Pout=Pout,
            )
        )

    def add_field_properties(self, field: bf.BackgroundField):
        self.background_fields.append(field)

    @property
    def mesh(self) -> Mesh3D:
        return self.basis.mesh

    @property
    def k0(self) -> float:
        return self.freq * 2 * np.pi / 299792458

    @property
    def _field(self) -> np.ndarray:
        if self._mode_field is not None:
            return self._mode_field
        
        if not isinstance(self._Texcite, np.ndarray):
            self._Texcite = np.eye(len(self.port_modes), dtype=np.complex128)
        if len(self.port_modes) > 0:
            avec = np.array([self.excitation[i.smat_index] for i in self.port_modes])
            avec = self._Texcite @ avec
            return sum(
                [
                    avec[i] * self._fields[mode.smat_index]
                    for i, mode in enumerate(self.port_modes)
                ]
            )  # type: ignore

        elif len(self.background_fields) > 0:
            return sum(
                [
                    self.excitation[mode] * self._fields[mode]
                    for mode in self.background_fields
                ]
            )


    @property
    def relative(self) -> MWField:
        """ Returns the same MWField object but with the relative flag turned on
        so that all fields are the relative field instead of the total field.
        """
        self._rel = True
        return self

    def backE(self, x: np.ndarray, y: np.ndarray, z: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Compute the background E-field at the provided coordinates.

        Args:
            x (np.ndarray): An array of X-coordinates
            y (np.ndarray): An array of X-coordinates
            z (np.ndarray): An array of X-coordinates
            mask (np.ndarray): A binary mask array that tells this funtion on which coordinates to evaluate the field.

        Returns:
            np.ndarray: _description_
        """
        out = np.zeros((3, x.shape[0]), dtype=np.complex128)
        out[:, ~mask] = np.nan
        for field in self.background_fields:
            out[:, mask] += self.excitation[field] * field.E(x, y, z)[:, mask]
        return out

    def backH(self, x: np.ndarray, y: np.ndarray, z: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Compute the background H-field at the provided coordinates.

        Args:
            x (np.ndarray): An array of X-coordinates
            y (np.ndarray): An array of X-coordinates
            z (np.ndarray): An array of X-coordinates
            mask (np.ndarray): A binary mask array that tells this funtion on which coordinates to evaluate the field.

        Returns:
            np.ndarray: _description_
        """
        out = np.zeros((3, x.shape[0]), dtype=np.complex128)
        out[:, ~mask] = np.nan
        for field in self.background_fields:
            out[:, mask] += self.excitation[field] * field.H(x, y, z)[:, mask]
        return out

    def set_field_vector(self) -> None:
        """Defines the default excitation coefficients for the current dataset as an excitation of only port 1."""
        # Freq sweep with ports
        if len(self.port_modes) > 0:
            self.excite_port(self.port_modes[0].port_number)
        elif len(self.background_fields) > 0:
            self.set_backgroundfield(self.background_fields[0])

    def excite_port(
        self, number: int | float, excitation: complex = 1.0 + 0.0j
    ) -> None:
        """Excite a single port provided by a given port number

        Args:
            number (int): The port number to excite
            coefficient (complex): The port excitation. Defaults to 1.0 + 0.0j
        """
        if self._active_ports is not None and number not in self._active_ports:
            raise KeyError(
                f"Port {number} has been embedded via embed_external_component "
                f"and can no longer be excited directly. Active ports: "
                f"{sorted(self._active_ports)}."
            )
        self.excitation = {key: 0.0 for key in self._fields.keys()}
        self.excitation[number] = excitation

    def set_excitations(self, *excitations: complex) -> None:
        """Set bulk port excitations by an ordered array of excitation coefficients.

        Returns:
            *complex: A sequence of complex numbers
        """
        self.excitation = {key: 0.0 for key in self._fields.keys()}
        for imode, coeff in enumerate(excitations):
            self.excitation[self.port_modes[imode].smat_index] = coeff

    def set_backgroundfield(self, bf) -> None:
        """Activate a specific background field

        Args:
            index (int): _description_
        """
        self.excitation = {key: 0.0 for key in self._fields.keys()}
        self.excitation[bf] = 1.0

    def combine_ports(self, p1: int, p2: int) -> MWField:
        """Combines ports p1 and p2 into a cifferential and common mode port respectively.

        The p1 index becomes the differential mode port
        The p2 index becomes the common mode port

        Args:
            p1 (int): The first port number
            p2 (int): The second port number

        Returns:
            MWField: _description_
        """

        fp1 = self._fields[p1]
        fp2 = self._fields[p2]

        self._fields[p1] = (fp1 - fp2) / np.sqrt(2)
        self._fields[p2] = (fp1 + fp2) / np.sqrt(2)
        return self

    def interpolate(
        self, xs: np.ndarray, ys: np.ndarray, zs: np.ndarray, usenan: bool = True
    ) -> EHField:
        """Interpolate the dataset in the provided xs, ys, zs values"""
        # fmt: off
        if isinstance(xs, (float, int, complex)):
            xs = np.array([xs,])
            ys = np.array([ys,])
            zs = np.array([zs,])

        shp = xs.shape
        xf = xs.flatten()
        yf = ys.flatten()
        zf = zs.flatten()

        constants = 1 / (-1j * 2 * np.pi * self.freq * (self._dur * MU0))

        if not self._silent:
            logger.info(f"Interpolating {xf.shape[0]} field points")
        logger.debug('Finding tet_mapping')

        mapping = self.basis.interpolate_index(
            xf, yf, zf, usenan=usenan
        )
        logger.debug("Index Interpolation complete")
        Ex, Ey, Ez = self.basis.interpolate(
            self._field, xf, yf, zf, mapping, usenan=usenan
        )
        logger.debug("E Interpolation complete")

        Hx, Hy, Hz = self.basis.interpolate_curl(
            self._field, xf, yf, zf, constants, mapping, usenan=usenan
        )
        logger.debug("H Interpolation complete")

        mask = ~np.isnan(Ex)
        if self._rel:
            Eb = self.backE(xf, yf, zf, mask)
            Ex = Ex - Eb[0, :]
            Ey = Ey - Eb[1, :]
            Ez = Ez - Eb[2, :]

        self.Ex = Ex.reshape(shp)
        self.Ey = Ey.reshape(shp)
        self.Ez = Ez.reshape(shp)

        if self._rel:
            Hb = self.backH(xf, yf, zf, mask)
            Hx = Hx - Hb[0, :]
            Hy = Hy - Hb[1, :]
            Hz = Hz - Hb[2, :]

        self.er = self._der[mapping].reshape(shp)
        self.ur = self._dur[mapping].reshape(shp)
        self.sig = self._dsig[mapping].reshape(shp)

        self.Hx = Hx.reshape(shp)
        self.Hy = Hy.reshape(shp)
        self.Hz = Hz.reshape(shp)

        self._x = xs
        self._y = ys
        self._z = zs

        ehfield = EHField(
            _E=np.array([self.Ex, self.Ey, self.Ez]),
            _H=np.array([self.Hx, self.Hy, self.Hz]),
            x=xs,
            y=ys,
            z=zs,
            freq=self.freq,
            er=self.er,
            ur=self.ur,
            sig=self.sig,
        )
        self._rel = False
        return ehfield

    def _solution_quality(self, solve_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        from .adaptive_mesh import compute_error_estimate

        error_tet, max_elem_size = compute_error_estimate(self, solve_ids)
        return error_tet, max_elem_size

    def integrate(self, surface: FaceSelection, gqo: int = 4) -> EHField:
        from ...mth.optimized import generate_int_data_tri
        from ...mth.integrals import gaus_quad_tri

        logger.warning("Use int_surf instead!")
        DPTS = gaus_quad_tri(gqo)
        tris = self.mesh.get_triangles(surface.tags)

        X, Y, Z, W, A, shape = generate_int_data_tri(
            self.mesh.nodes, self.mesh.tris[:, tris], DPTS
        )

        ehfield = self.interpolate(X, Y, Z, False)
        ehfield.aux["areas"] = A
        ehfield.aux["weights"] = W

        return ehfield

    def int_surf(
        self, surface: FaceSelection, argument: Callable, gqo: int = 4
    ) -> EHField:
        """Performs a surface integral on the provided surface object.

        Args:
            surface (FaceSelection): The surface to integrate
            quantity (Callable): A function that takes an EH field as argument
            gqo (int, optional): Gauss Quadrature order. Defaults to 4.

        Returns:
            EHField: _description_
        """
        from ...mth.optimized import generate_int_data_tri
        from ...mth.integrals import gaus_quad_tri

        DPTS = gaus_quad_tri(gqo)
        tris = self.mesh.get_triangles(surface.tags)

        X, Y, Z, W, A, shape = generate_int_data_tri(
            self.mesh.nodes, self.mesh.tris[:, tris], DPTS
        )

        ehfield = self.interpolate(X, Y, Z, False)

        output = argument(ehfield)

        if len(output.shape) == 2:
            axis = 1
        else:
            axis = 0

        return np.sum(output * A * W, axis=axis)

    def int_vol(
        self, domain: DomainSelection, argument: Callable, gqo: int = 4
    ) -> EHField:
        """Performs a surface integral on the provided surface object.

        Args:
            domain (DomainSelection): The surface to integrate
            quantity (Callable): A function that takes an EH field as argument
            gqo (int, optional): Gauss Quadrature order. Defaults to 4.

        Returns:
            EHField: _description_
        """
        from ...mth.optimized import gaus_quad_tet, generate_int_data_tet
        # from ...mth.integrals import gaus_quad_tet

        DPTS = gaus_quad_tet(gqo)
        tets = self.mesh.get_tetrahedra(domain.tags)

        X, Y, Z, W, A, shape = generate_int_data_tet(
            self.mesh.nodes, self.mesh.tets[:, tets], DPTS
        )

        ehfield = self.interpolate(X, Y, Z, False)

        output = argument(ehfield)

        if len(output.shape) == 2:
            axis = 1
        else:
            axis = 0

        return np.sum(output * A * W, axis=axis)

    def int_line(
        self, line: Line | list[tuple[float, float, float]], argument: Callable
    ) -> EHField:
        """Performs a line integral on the provided line with the an integral argument.

        Args:
            line (Line | list[tuple[float, float, float]]): Either an emerge.Line object or a list of points that define a discrete integration path.
            argument (Callable): a function that takes an EMfield and returns a scalar argument. Example lambda x: x.Ex

        Returns:
            EHField: _description_
        """
        if not isinstance(line, Line):
            x, y, z = zip(*line)
            line = Line(x, y, z)

        nint = self.interpolate(*line.cpoint)
        dx = np.append(line.dxs, line.dxs[-1])
        dy = np.append(line.dys, line.dys[-1])
        dz = np.append(line.dzs, line.dzs[-1])
        nint.dl = np.array([dx, dy, dz])
        nint.dlx = dx
        nint.dly = dy
        nint.dlz = dz

        return line._integrate(argument(nint))

    def int_lumped_element(
        self,
        lumped_element: LumpedElement,
        axis: Axis | tuple[float, float, float] | np.ndarray,
        quantity: Literal["E", "H", "S"] = "E",
    ) -> float:
        """Performs a voltage integration of a lumped element.
        It needs an integration direction axis to work.

        Args:
            lumped_element (LumpedElement): The lumped element object.
            axis (Axis | tuple[float,float,float]): The integration axis direction

        Returns:
            float: _description_
        """
        logger.debug(" - Finding Lumped Element integration points")
        field_axis = _parse_axis(axis).np

        points = self.mesh.get_nodes(lumped_element.tags)

        if points.size == 0:
            raise ValueError(
                f"The lumped port {LumpedElement} has no nodes associated with it"
            )

        xs = self.mesh.nodes[0, points]
        ys = self.mesh.nodes[1, points]
        zs = self.mesh.nodes[2, points]

        dotprod = xs * field_axis[0] + ys * field_axis[1] + zs * field_axis[2]

        start_id = np.argwhere(dotprod == np.min(dotprod)).flatten()

        xs = xs[start_id]
        ys = ys[start_id]
        zs = zs[start_id]

        voltages = []
        for x, y, z in zip(xs, ys, zs):
            start = np.array([x, y, z])
            end = start + field_axis * lumped_element.height
            line = Line.from_points(start, end, 51)
            logger.debug(f" - Integration Line {start} -> {end}.")
            V = line.line_integral(
                lambda x, y, z: getattr(self.interpolate(x, y, z), quantity)
            )
            voltages.append(V)
        return sum(voltages) / len(voltages)

    def boundary(self, selection: FaceSelection) -> EHField:
        """Interpolate the field on the node coordinates of the surface."""
        boundary = self.mesh.boundary_surface(selection.tags)
        x = boundary.nodes[0, :]
        y = boundary.nodes[1, :]
        z = boundary.nodes[2, :]
        ehfield = self.interpolate(x, y, z, False)
        ehfield.aux["tris"] = boundary.tris
        ehfield.aux["boundary"] = True
        ehfield.structure = DataStructure.TRISURF
        return ehfield

    def current_boundary(self, selection: FaceSelection) -> EHField:
        """Interpolate the field on the node coordinates of the surface."""
        boundary = self.mesh.boundary_surface(selection.tags)
        ns = boundary.normals
        cs = (
            boundary.nodes[:, boundary.tris[0, :]]
            + boundary.nodes[:, boundary.tris[1, :]]
            + boundary.nodes[:, boundary.tris[2, :]]
        ) / 3

        nx = ns[0, :]
        ny = ns[1, :]
        nz = ns[2, :]
        cx = cs[0, :]
        cy = cs[1, :]
        cz = cs[2, :]

        eps = 1e-6

        ehfield_1 = self.interpolate(cx - nx * eps, cy - ny * eps, cz - nz * eps, False)
        ehfield_2 = self.interpolate(cx + nx * eps, cy + ny * eps, cz + nz * eps, False)

        dHx = ehfield_2.Hx - ehfield_1.Hx
        dHy = ehfield_2.Hy - ehfield_1.Hy
        dHz = ehfield_2.Hz - ehfield_1.Hz

        Jsx = ny * dHz - nz * dHy
        Jsy = nz * dHx - nx * dHz
        Jsz = nx * dHy - ny * dHx

        Jst = np.array([Jsx, Jsy, Jsz])

        Js = np.zeros_like(boundary.nodes, dtype=np.complex128)
        Js_counter = np.zeros((boundary.n_nodes,), dtype=np.int8)

        ehfield = self.interpolate(
            boundary.nodes[0, :], boundary.nodes[1, :], boundary.nodes[2, :], False
        )

        for i in range(boundary.n_tris):
            nids = boundary.tris[:, i]
            Js[:, nids] += Jst[:, i]
            Js_counter[nids] += 1

        Js_counter[Js_counter == 0] = 1

        Js = Js / Js_counter

        ehfield._Js = Js
        ehfield.aux["tris"] = boundary.tris
        ehfield.aux["boundary"] = True
        ehfield.structure = DataStructure.TRISURF
        return ehfield

    def cutplane(
        self,
        ds: float,
        x: float | None = None,
        y: float | None = None,
        z: float | None = None,
        usenan: bool = True,
    ) -> EHField:
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
        xs = np.linspace(xb[0], xb[1], int((xb[1] - xb[0]) / ds))
        ys = np.linspace(yb[0], yb[1], int((yb[1] - yb[0]) / ds))
        zs = np.linspace(zb[0], zb[1], int((zb[1] - zb[0]) / ds))

        if x is not None:
            Y, Z = np.meshgrid(ys, zs)
            X = x * np.ones_like(Y)
        if y is not None:
            X, Z = np.meshgrid(xs, zs)
            Y = y * np.ones_like(X)
        if z is not None:
            X, Y = np.meshgrid(xs, ys)
            Z = z * np.ones_like(Y)
        field = self.interpolate(X, Y, Z, usenan=usenan)
        field.structure = DataStructure.GRID2D
        return field

    def cutplane_normal(
        self, 
        point: tuple[float, float, float] = (0, 0, 0), 
        normal: tuple[float, float, float] = (0, 0, 1), 
        npoints: int = 300, 
        usenan: bool = True
    ) -> EHField:
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

        tmp = np.array([1, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1, 0])
        u = np.cross(n, tmp)
        u /= np.linalg.norm(u)
        v = np.cross(n, u)

        xb, yb, zb = self.basis.bounds
        nx, ny, nz = 5, 5, 5
        Xg = np.linspace(xb[0], xb[1], nx)
        Yg = np.linspace(yb[0], yb[1], ny)
        Zg = np.linspace(zb[0], zb[1], nz)
        Xg, Yg, Zg = np.meshgrid(Xg, Yg, Zg, indexing="ij")
        geometry = np.vstack([Xg.ravel(), Yg.ravel(), Zg.ravel()]).T  # Nx3

        rel_pts = geometry - point
        S = rel_pts @ u
        T = rel_pts @ v

        margin = 0.01
        s_min, s_max = S.min(), S.max()
        t_min, t_max = T.min(), T.max()
        s_bounds = (s_min - margin * (s_max - s_min), s_max + margin * (s_max - s_min))
        t_bounds = (t_min - margin * (t_max - t_min), t_max + margin * (t_max - t_min))

        S_grid = np.linspace(s_bounds[0], s_bounds[1], npoints)
        T_grid = np.linspace(t_bounds[0], t_bounds[1], npoints)
        S_mesh, T_mesh = np.meshgrid(S_grid, T_grid)

        X = point[0] + S_mesh * u[0] + T_mesh * v[0]
        Y = point[1] + S_mesh * u[1] + T_mesh * v[1]
        Z = point[2] + S_mesh * u[2] + T_mesh * v[2]

        field = self.interpolate(X, Y, Z, usenan=usenan)
        field.structure = DataStructure.GRID2D
        return field

    def grid(
        self,
        ds: float | None = None,
        N: int = 10_000,
        usenan: bool = True,
        x_range: tuple[float, float] | None = None,
        y_range: tuple[float, float] | None = None,
        z_range: tuple[float, float] | None = None,
    ) -> EHField:
        """Interpolate a uniform grid sampled at ds

        Args:
            ds (float, optional): the sampling grid size. Defaults to None (uses N)
            N (int, optional): The approximate total number of sample points. Defaults to 10,000

        Returns:
            EHField: Storage container for data
        """
        xb, yb, zb = self.basis.bounds
        if x_range is not None:
            xb = x_range
        if y_range is not None:
            yb = y_range
        if z_range is not None:
            zb = z_range
        DX = xb[1] - xb[0]
        DY = yb[1] - yb[0]
        DZ = zb[1] - zb[0]
        if ds is None:
            ds = ((DX * DY * DZ) / N) ** (1 / 3)

        xs = np.linspace(xb[0], xb[1], int(DX / ds) + 1)
        ys = np.linspace(yb[0], yb[1], int(DY / ds) + 1)
        zs = np.linspace(zb[0], zb[1], int(DZ / ds) + 1)
        X, Y, Z = np.meshgrid(xs, ys, zs)
        field = self.interpolate(X, Y, Z, usenan=usenan)
        field.structure = DataStructure.GRID3D
        return field

    def vector(
        self,
        field: Literal["E", "H"],
        metric: Literal["real", "imag", "complex"] = "real",
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Returns the X,Y,Z,Fx,Fy,Fz data to be directly cast into plot functions.

        The field can be selected by a string literal. The metric of the complex vector field by the metric.
        For animations, make sure to always use the complex metric.

        Args:
            field ('E','H'): The field to return
            metric ([]'real','imag','complex'], optional): the metric to impose on the field. Defaults to 'real'.

        Returns:
            tuple[np.ndarray,...]: The X,Y,Z,Fx,Fy,Fz arrays
        """
        if field == "E":
            Fx, Fy, Fz = self.Ex, self.Ey, self.Ez
        elif field == "H":
            Fx, Fy, Fz = self.Hx, self.Hy, self.Hz

        if metric == "real":
            Fx, Fy, Fz = Fx.real, Fy.real, Fz.real
        elif metric == "imag":
            Fx, Fy, Fz = Fx.imag, Fy.imag, Fz.imag

        return self._x, self._y, self._z, Fx, Fy, Fz

    def scalar(
        self,
        field: Literal["Ex", "Ey", "Ez", "Hx", "Hy", "Hz", "normE", "normH"],
        metric: Literal["abs", "real", "imag", "complex"] = "real",
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Returns the data X, Y, Z, Field based on the interpolation

        For animations, make sure to select the complex metric.

        Args:
            field (str): The field to plot
            metric (str, optional): The metric to impose on the plot. Defaults to 'real'.

        Returns:
            (X,Y,Z,Field): The coordinates plus field scalar
        """
        field = getattr(self, field)
        if metric == "abs":
            field = np.abs(field)
        elif metric == "real":
            field = field.real
        elif metric == "imag":
            field = field.imag
        elif metric == "complex":
            field = field
        return self._x, self._y, self._z, field

    def farfield_2d(
        self,
        ref_direction: tuple[float, float, float] | Axis,
        plane_normal: tuple[float, float, float] | Axis,
        faces: FaceSelection | GeoSurface,
        ang_range: tuple[float, float] = (-180, 180),
        Npoints: int = 201,
        origin: tuple[float, float, float] | None = None,
        syms: list[Literal["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"]] | None = None,
    ) -> EHFieldFF:  # tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        E, H, Ptot = self._farfield(theta, phi, faces, origin, syms=syms)
        angs = np.linspace(*ang_range, Npoints) * np.pi / 180
        return EHFieldFF(
            _E=E, _H=H, theta=theta, phi=phi, Ptot=Ptot, ang=angs, freq=self.freq
        )

    def farfield_3d(
        self,
        faces: FaceSelection | GeoSurface,
        thetas: np.ndarray | None = None,
        phis: np.ndarray | None = None,
        origin: tuple[float, float, float] | None = None,
        syms: list[Literal["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"]] | None = None,
    ) -> EHFieldFF:
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
            thetas = np.linspace(0, np.pi, 91)
        if phis is None:
            phis = np.linspace(-np.pi, np.pi, 181)

        T, P = np.meshgrid(thetas, phis, indexing='ij')

        E, H, Ptot = self._farfield(
            T.flatten(), P.flatten(), faces, origin=origin, syms=syms
        )
        E = E.reshape((3,) + T.shape)
        H = H.reshape((3,) + T.shape)

        return EHFieldFF(E, H, T, P, Ptot, freq=self.freq)

    def embed_external_component(
        self, touchstone_file: str, port_indices: list[int | float]
    ) -> None:
        """Embed an external N-port network (given as a Touchstone file) onto a
        subset of this field's ports.

        This does NOT remove ports -- all port_modes/fields remain present
        at their original size forever. Ports are marked inactive in
        `_active_ports` instead of removed, and both `_Sp` and `_Texcite`
        keep their original full shape. This means port numbers never
        change and never need to be remembered/remapped -- calling this
        multiple times with different, still-active port numbers just
        works, and each call correctly folds in the effect of any prior
        embedding.

        Args:
            touchstone_file (str): The filename of the touchstone file to import
            port_indices (list[int | float]): The port numbers (smat_index) of
                this field to connect the touchstone file to. Must all
                currently be active (not already embedded by a previous call).
        """
        from ....read import TouchstoneData

        Nports_total = len(self.port_modes)
        portmap = {p.smat_index: i for i, p in enumerate(self.port_modes)}

        if self._active_ports is None:
            self._active_ports = set(portmap.keys())
        if not isinstance(self._Texcite, np.ndarray):
            self._Texcite = np.eye(Nports_total, dtype=np.complex128)

        missing = [p for p in port_indices if p not in self._active_ports]
        if missing:
            raise KeyError(
                f"Port(s) {missing} are not available (already embedded, or "
                f"don't exist). Active ports: {sorted(self._active_ports)}"
            )
        if len(set(port_indices)) != len(port_indices):
            raise ValueError(f"Duplicate port numbers in port_indices: {port_indices}")

        td = TouchstoneData(touchstone_file)
        S_ext = td.interp_S(self.freq)  # (M, M) at this field's single frequency

        m_pos = [portmap[p] for p in port_indices]
        n_labels = sorted(self._active_ports - set(port_indices), key=lambda l: portmap[l])
        n_pos = [portmap[l] for l in n_labels]

        S_nn = self._Sp[np.ix_(n_pos, n_pos)]
        S_nm = self._Sp[np.ix_(n_pos, m_pos)]
        S_mn = self._Sp[np.ix_(m_pos, n_pos)]
        S_mm = self._Sp[np.ix_(m_pos, m_pos)]

        I_m = np.eye(len(m_pos), dtype=np.complex128)
        term = np.linalg.inv(I_m - S_mm @ S_ext)
        coupling = S_ext @ term @ S_mn  # (M, N)

        # Fold the embedding into the effective S among still-active ports,
        # so a SUBSEQUENT embedding call sees the correct effective network.
        self._Sp[np.ix_(n_pos, n_pos)] = S_nn + S_nm @ coupling
        # Mark embedded ports as no longer meaningful.
        self._Sp[m_pos, :] = np.nan
        self._Sp[:, m_pos] = np.nan

        # Excitation transform stays a fixed NxN throughout -- no resizing
        # needed since dimensions never change.
        local_T = np.zeros((Nports_total, Nports_total), dtype=np.complex128)
        local_T[np.ix_(n_pos, n_pos)] = np.eye(len(n_pos))
        local_T[np.ix_(m_pos, n_pos)] = coupling
        self._Texcite = self._Texcite @ local_T

        self._active_ports -= set(port_indices)

        logger.info(
            f"Embedded {touchstone_file!r} on ports {port_indices}. "
            f"Active ports remaining: {sorted(self._active_ports)}"
        )

    def farfield(
        self,
        theta: np.ndarray,
        phi: np.ndarray,
        faces: FaceSelection | GeoSurface,
        origin: tuple[float, float, float] | None = None,
        syms: list[Literal["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"]] | None = None,
    ) -> EHFieldFF:
        E, H, Ptot = self._farfield(
            theta.flatten(), phi.flatten(), faces, origin=origin, syms=syms
        )
        E = E.reshape((3,) + theta.shape)
        H = H.reshape((3,) + theta.shape)

        return EHFieldFF(E, H, theta, phi, Ptot, freq=self.freq)
        
    def _farfield(
        self,
        theta: np.ndarray,
        phi: np.ndarray,
        faces: FaceSelection | GeoSurface,
        origin: tuple[float, float, float] | None = None,
        syms: list[Literal["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"]] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Compute the farfield at the provided theta/phi coordinates

        Args:
            theta (np.ndarray): The Theta coordinates as (N,) 1D Array
            phi (np.ndarray): The Phi coordinates as (N,) 1D Array
            faces (FaceSelection | GeoSurface): the faces to use as integration boundary
            origin (tuple[float, float, float], optional): A normal alignment origin. Optional use in cases where the "inside" is not clear.
            syms (list[Literal['Ex','Ey','Ez','Hx','Hy','Hz']], optional): E and H-plane symmetry planes where Ex is E-symmetry in x=0. Defaults to []

        Returns:
            tuple[np.ndarray, np.ndarray, float]: The E and H field as (3,N) arrays and the total radiated power
        """
        if syms is None:
            syms = []

        from .sc import stratton_chu


        surface = self.basis.mesh.boundary_surface(
            faces.tags, inward_normal=False, origin=origin
        )
        ehfield = self.interpolate(*surface.exyz)
        Eff, Hff, wns = stratton_chu(ehfield.E, ehfield.H, surface, theta, phi, self.k0)

        Ptot = np.sum(
            ehfield.Smx * wns[0, :] + ehfield.Smy * wns[1, :] + ehfield.Smz * wns[2, :]
        ).real

        if len(syms) == 0:
            return Eff, Hff, Ptot

        # fmt: off
        factor = 1.0
        if len(syms) == 1:
            factor = (0.5) ** 0.5
            flip_sets = ((syms[0], ),)

        elif len(syms) == 2:
            factor = (0.25) ** 0.5
            s1, s2 = syms
            flip_sets = ((s1,), (s2,), (s1, s2, ))

        elif len(syms) == 3:
            factor = (0.125) ** 0.5
            s1, s2, s3 = syms
            flip_sets = (
                (s1, ),
                (s2, ),
                (s3, ),
                (s1, s2,),
                (s1, s3,),
                (s2, s3,),
                (s1, s2, s3),
            )

        for flips in flip_sets:
            surf = surface.copy()
            ehf = _EHSign()
            Ef, Hf = ehfield.E.copy(), ehfield.H.copy()
            for flip in flips:
                ehf.apply(flip)
                surf.flip(flip[1])
            Ef, Hf = ehf.flip_field(Ef, Hf)

            E2, H2, wns = stratton_chu(Ef, Hf, surf, theta, phi, self.k0)
            Eff = Eff + E2
            Hff = Hff + H2

        # fmt: on
        return Eff * factor, Hff * factor, Ptot * (factor**2)

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

    def optycal_antenna(
        self,
        faces: FaceSelection | GeoSurface | None = None,
        origin: tuple[float, float, float] | None = None,
        syms: list[Literal["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"]] | None = None,
    ) -> dict:
        """Export this models exterior to an Optical acceptable dataset

        Args:
            faces (FaceSelection | GeoSurface): The faces to export. Defaults to None

        Returns:
            tuple: _description_
        """
        freq = self.freq

        def function(theta: np.ndarray, phi: np.ndarray, k0: float):
            E, H, _ = self._farfield(theta, phi, faces, origin, syms)
            return E[0, :], E[1, :], E[2, :], H[0, :], H[1, :], H[2, :]

        return dict(freq=freq, ff_function=function)

    def _field_weight(self, ehfield, weight_by: str) -> np.ndarray:
        """Evaluate a scalar 'energy density' proxy from an interpolated EHField,
        used to importance-weight seed-point sampling.

        Args:
            ehfield: An EHField (e.g. from self.interpolate(...)).
            weight_by: 'E' -> |E|, 'H' -> |H|, 'EH' -> |E|*|H|.

        Returns:
            np.ndarray: weight per sample point.
        """
        if weight_by == "E":
            return np.sqrt(np.abs(ehfield.Ex)**2 + np.abs(ehfield.Ey)**2 + np.abs(ehfield.Ez)**2)
        if weight_by == "H":
            return np.sqrt(np.abs(ehfield.Hx)**2 + np.abs(ehfield.Hy)**2 + np.abs(ehfield.Hz)**2)
        if weight_by == "EH":
            Emag = np.sqrt(np.abs(ehfield.Ex)**2 + np.abs(ehfield.Ey)**2 + np.abs(ehfield.Ez)**2)
            Hmag = np.sqrt(np.abs(ehfield.Hx)**2 + np.abs(ehfield.Hy)**2 + np.abs(ehfield.Hz)**2)
            return Emag * Hmag
        raise ValueError(f"weight_by must be 'E', 'H', or 'EH', got {weight_by!r}")


    def _normalize_weights(self, w: np.ndarray) -> np.ndarray:
        """Turn a nonnegative weight array into a probability distribution,
        falling back to uniform if the total weight is degenerate."""
        w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
        w = np.clip(w, 0.0, None)
        total = w.sum()
        if not np.isfinite(total) or total <= 0:
            return np.full(w.shape, 1.0 / w.shape[0])
        return w / total


    def _weighted_seed_points_surface(self, surface, n_particles: int, weight_by: str, quad_order: int = 4):
        """Importance-sample seed points on a FaceSelection, concentrated on
        triangles (and sub-triangle locations) with high |E|, |H|, or |E|*|H|.
        Uses the same Gauss-quadrature machinery as int_surf."""
        from ...mth.optimized import generate_int_data_tri
        from ...mth.integrals import gaus_quad_tri

        DPTS = gaus_quad_tri(quad_order)
        tris = self.mesh.get_triangles(surface.tags)
        X, Y, Z, W, A, shape = generate_int_data_tri(self.mesh.nodes, self.mesh.tris[:, tris], DPTS)

        ehfield = self.interpolate(X, Y, Z, usenan=True)
        w = self._field_weight(ehfield, weight_by) * np.abs(W) * np.abs(A)
        probs = self._normalize_weights(w)

        M = X.shape[0]
        replace = n_particles > M
        sel = np.random.choice(M, size=n_particles, replace=replace, p=probs)
        return X[sel].copy(), Y[sel].copy(), Z[sel].copy()


    def _weighted_seed_points_domain(self, domain, n_particles: int, weight_by: str, quad_order: int = 4):
        """Importance-sample seed points in a DomainSelection, concentrated on
        tets (and sub-tet locations) with high |E|, |H|, or |E|*|H|.
        Uses the same Gauss-quadrature machinery as int_vol."""
        from ...mth.optimized import gaus_quad_tet, generate_int_data_tet

        DPTS = gaus_quad_tet(quad_order)
        tets = self.mesh.get_tetrahedra(domain.tags)
        X, Y, Z, W, A, shape = generate_int_data_tet(self.mesh.nodes, self.mesh.tets[:, tets], DPTS)

        ehfield = self.interpolate(X, Y, Z, usenan=True)
        w = self._field_weight(ehfield, weight_by) * np.abs(W) * np.abs(A)
        probs = self._normalize_weights(w)

        M = X.shape[0]
        replace = n_particles > M
        sel = np.random.choice(M, size=n_particles, replace=replace, p=probs)
        return X[sel].copy(), Y[sel].copy(), Z[sel].copy()


    def _seed_candidates(self, selection, is_domain: bool, density: str) -> np.ndarray:
        """Build a (3, M) candidate seed-point cloud from a FaceSelection or
        DomainSelection at a given density level, by progressively unioning
        mesh entity centers:

            'sparse' -> nodes only
            'medium' -> nodes + edge centers
            'dense'  -> nodes + edge centers + triangle centers
                        (or + tet centroids, for a DomainSelection)

        Args:
            selection: FaceSelection or DomainSelection to seed from.
            is_domain: True if `selection` is a DomainSelection (uses tet
                centroids at 'dense'), False if it's a FaceSelection (uses
                triangle centers at 'dense').
            density: 'sparse' | 'medium' | 'dense'.

        Returns:
            np.ndarray: (3, M) candidate seed points.
        """
        if density not in ("sparse", "medium", "dense"):
            raise ValueError(f"density must be 'sparse', 'medium', or 'dense', got {density!r}")

        tags = selection.tags
        node_ids = self.mesh.get_nodes(tags)
        parts = [self.mesh.nodes[:, node_ids]]

        if density in ("medium", "dense"):
            edge_ids = self.mesh.get_edges(tags)
            parts.append(self.mesh.edge_centers[:, edge_ids])

        if density == "dense":
            if is_domain:
                tet_ids = self.mesh.get_tetrahedra(tags)
                parts.append(self.mesh.centroids[:, tet_ids])
            else:
                tri_ids = self.mesh.get_triangles(tags)
                parts.append(self.mesh.tri_centers[:, tri_ids])

        return np.hstack(parts)


    def trace_poynting_lines(
        self,
        seed_points=None,          # (3,N) array or (xs, ys, zs) tuple of user coordinates
        seed_surface=None,         # FaceSelection -> seeds from nodes/edges/tris
        seed_domain=None,          # DomainSelection -> seeds from nodes/edges/tets
        density: str = "medium",   # 'sparse' | 'medium' | 'dense' -- unweighted seeding only
        weight_by: Literal['E','H','EH'] | None = None,   # None | 'E' | 'H' | 'EH' -- importance-weighted seeding
        quad_order: int = 4,       # Gauss quadrature order used when weight_by is set
        n_particles: int | None = None,   # cap on seed count; required if weight_by is set
        max_steps: int = 5000,
        ds_init: float | None = None,
        ds_min: float | None = None,
        ds_max: float | None = None,
        rtol: float = 1e-4,
        atol: float = 1e-9,
        normalize: bool = True,
        direction: int = 1,        # +1 along S, -1 against S
        stagnation_eps: float = 1e-12,
        safety: float = 0.9,
        verbose: bool = False,
        dx: float = 0.0,
        dy: float = 0.0,
        dz: float = 0.0,
    ) -> list[np.ndarray]:
        """Trace particle/field-line trajectories through the time-averaged Poynting
        vector field using an adaptive-step embedded RK4(5) (Dormand-Prince) integrator.

        Each returned trajectory is a (4, n_i) array: rows 0-2 are x,y,z and row 3
        is the local wave impedance Z = |E|/|H| at that point.

        Args:
            seed_points: Explicit start coordinates, either shape (3,N) or a tuple
                (xs, ys, zs) of 1D arrays. Takes priority over everything else.
            seed_surface: FaceSelection to seed particles from.
            seed_domain: DomainSelection to seed particles from.
            density: 'sparse' (just nodes), 'medium' (+ edge centers), or
                'dense' (+ triangle centers / tet centroids -- everything).
                Only used for *unweighted* seeding (weight_by is None), with
                seed_surface or seed_domain.
            weight_by: If set to 'E', 'H', or 'EH', seed points are instead
                importance-sampled from Gauss-quadrature points across the
                surface/domain, weighted by |E|, |H|, or |E|*|H| (times the
                local area/volume element) -- i.e. more seeds land on
                triangles/tets with higher local field energy, with
                sub-triangle/sub-tet placement precision. Requires
                seed_surface or seed_domain, and an explicit n_particles.
            quad_order: Gauss quadrature order used to generate the candidate
                points when weight_by is set (higher = more candidate points
                per triangle/tet = finer sub-element resolution, at the cost
                of more field evaluations). Defaults to 4, matching int_surf/int_vol.
            n_particles: Cap on the number of seed points. Default None means
                "use every candidate point at the chosen density" for
                unweighted seeding -- only when a limit is given does it
                randomly subsample down to that many points. Required
                (must not be None) when weight_by is set, since importance
                sampling needs an explicit target count. For the fallback
                random-bounding-box seeding (no seed source given at all),
                None falls back to 200 points.
            max_steps: Hard cap on integrator iterations (safety stop).
            ds_init/ds_min/ds_max: Arc-length step-size bounds. Defaults are
                derived from the model's bounding-box diagonal.
            rtol/atol: Relative/absolute local error tolerance for step control.
            normalize: If True (recommended), integrate the unit tangent dr/ds =
                S/|S| (stable arc-length parameterization). If False, integrates
                the raw field as a literal velocity field (dr/dt = S).
            direction: +1 to trace along the Poynting vector, -1 to trace against it.
            stagnation_eps: |S| below this is treated as a stagnation point (terminate).
            safety: Safety factor (<1) applied to the adaptive step-size update.
            verbose: Print periodic progress.
            dx, dy, dz: Constant offset applied to all generated seed points.

        Returns:
            list[np.ndarray]: one (4, n_i) array per particle: [x, y, z, Z_local].
        """
        self._silent = True
        # -----------------------------------------------------------------
        # 1. Generate the initial particle coordinates (seed points)
        # -----------------------------------------------------------------
        if seed_points is not None:
            pts = np.asarray(seed_points, dtype=float)
            if pts.shape[0] != 3:
                pts = pts.T
            x0, y0, z0 = pts[0].copy(), pts[1].copy(), pts[2].copy()

        elif weight_by is not None:
            if n_particles is None:
                raise ValueError("weight_by requires an explicit n_particles (target seed count).")
            if seed_surface is not None:
                x0, y0, z0 = self._weighted_seed_points_surface(seed_surface, n_particles, weight_by, quad_order)
            elif seed_domain is not None:
                x0, y0, z0 = self._weighted_seed_points_domain(seed_domain, n_particles, weight_by, quad_order)
            else:
                raise ValueError("weight_by requires seed_surface or seed_domain to sample from.")

        elif seed_surface is not None:
            candidates = self._seed_candidates(seed_surface, is_domain=False, density=density)
            if n_particles is not None and candidates.shape[1] > n_particles:
                sel = np.random.choice(candidates.shape[1], n_particles, replace=False)
                candidates = candidates[:, sel]
            x0, y0, z0 = candidates[0], candidates[1], candidates[2]

        elif seed_domain is not None:
            candidates = self._seed_candidates(seed_domain, is_domain=True, density=density)
            if n_particles is not None and candidates.shape[1] > n_particles:
                sel = np.random.choice(candidates.shape[1], n_particles, replace=False)
                candidates = candidates[:, sel]
            x0, y0, z0 = candidates[0], candidates[1], candidates[2]

        else:
            # --- placeholder: default seeding strategy ---
            n_fallback = n_particles if n_particles is not None else 200
            xb, yb, zb = self.basis.bounds
            x0 = np.random.uniform(xb[0], xb[1], n_fallback)
            y0 = np.random.uniform(yb[0], yb[1], n_fallback)
            z0 = np.random.uniform(zb[0], zb[1], n_fallback)

        x0 = x0 + dx
        y0 = y0 + dy
        z0 = z0 + dz
        N = x0.shape[0]
        pos = np.vstack([x0, y0, z0]).astype(float)          # (3, N) current positions

        # -----------------------------------------------------------------
        # 2. Step-size bounds, derived from the model's bounding-box diagonal
        # -----------------------------------------------------------------
        xb, yb, zb = self.basis.bounds
        diag = np.sqrt((xb[1] - xb[0])**2 + (yb[1] - yb[0])**2 + (zb[1] - zb[0])**2)
        if ds_init is None:
            ds_init = diag / 500
        if ds_min is None:
            ds_min = diag / 1e6
        if ds_max is None:
            ds_max = diag / 20

        h = np.full(N, ds_init, dtype=float)     # per-particle current step size
        active = np.ones(N, dtype=bool)          # per-particle still-running mask

        # --- Dormand-Prince RK45 tableau ---
        a21 = 1 / 5
        a31, a32 = 3 / 40, 9 / 40
        a41, a42, a43 = 44 / 45, -56 / 15, 32 / 9
        a51, a52, a53, a54 = 19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729
        a61, a62, a63, a64, a65 = 9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656
        a71, a72, a73, a74, a75, a76 = 35 / 384, 0.0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84
        b5 = np.array([35 / 384, 0.0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0.0])
        b4 = np.array([5179 / 57600, 0.0, 7571 / 16695, 393 / 640, -92097 / 339200, 187 / 2100, 1 / 40])

        imp_eps = 1e-30  # guards |E|/|H| against division by ~0

        def field(p: np.ndarray) -> np.ndarray:
            """Evaluate the (optionally normalized) time-averaged Poynting vector
            at positions p, shape (3, M). NaN in -> NaN out (outside domain)."""
            ehfield = self.interpolate(p[0], p[1], p[2], usenan=True)
            S = np.vstack([ehfield.Smx, ehfield.Smy, ehfield.Smz]).real
            if normalize:
                mag = np.linalg.norm(S, axis=0)
                with np.errstate(invalid="ignore", divide="ignore"):
                    S = np.where(mag > stagnation_eps, S / mag, np.nan)
            return direction * S

        def field_with_impedance(p: np.ndarray):
            """Same as field(), but also returns the local wave impedance
            Z = |E|/|H| at p (shape (M,)). Reuses the same interpolate() call,
            so this costs nothing extra beyond a normal field evaluation."""
            ehfield = self.interpolate(p[0], p[1], p[2], usenan=True)
            S = np.vstack([ehfield.Smx, ehfield.Smy, ehfield.Smz]).real
            Emag = np.sqrt(np.abs(ehfield.Ex)**2 + np.abs(ehfield.Ey)**2 + np.abs(ehfield.Ez)**2)
            Hmag = np.sqrt(np.abs(ehfield.Hx)**2 + np.abs(ehfield.Hy)**2 + np.abs(ehfield.Hz)**2)
            with np.errstate(invalid="ignore", divide="ignore"):
                Z = np.where(Hmag > imp_eps, Emag / Hmag, np.nan)
            if normalize:
                mag = np.linalg.norm(S, axis=0)
                with np.errstate(invalid="ignore", divide="ignore"):
                    S = np.where(mag > stagnation_eps, S / mag, np.nan)
            return direction * S, Z

        # Initial impedance sample at each seed point
        _, Z0 = field_with_impedance(pos)
        paths = [[np.array([pos[0, i], pos[1, i], pos[2, i], Z0[i]])] for i in range(N)]

        # -----------------------------------------------------------------
        # 3. Main adaptive integration loop
        # -----------------------------------------------------------------
        step_count = 0
        while active.any() and step_count < max_steps:
            step_count += 1
            idx = np.where(active)[0]
            p0 = pos[:, idx]
            hs = h[idx]

            k1 = field(p0)
            k2 = field(p0 + hs * a21 * k1)
            k3 = field(p0 + hs * (a31 * k1 + a32 * k2))
            k4 = field(p0 + hs * (a41 * k1 + a42 * k2 + a43 * k3))
            k5 = field(p0 + hs * (a51 * k1 + a52 * k2 + a53 * k3 + a54 * k4))
            k6 = field(p0 + hs * (a61 * k1 + a62 * k2 + a63 * k3 + a64 * k4 + a65 * k5))
            p5 = p0 + hs * (a71 * k1 + a72 * k2 + a73 * k3 + a74 * k4 + a75 * k5 + a76 * k6)
            # k7 is evaluated at p5, which (by the FSAL property of Dormand-Prince)
            # is exactly the accepted 5th-order next position -- so we grab the
            # local impedance there for free.
            k7, Z_next = field_with_impedance(p5)

            ks = np.stack([k1, k2, k3, k4, k5, k6, k7], axis=0)     # (7, 3, M)
            sol5 = p0 + hs * np.einsum('i,ijk->jk', b5, ks)
            sol4 = p0 + hs * np.einsum('i,ijk->jk', b4, ks)

            stage_invalid = np.any(np.isnan(ks), axis=(0, 1)) | np.any(np.isnan(sol5), axis=0)

            err = np.linalg.norm(np.nan_to_num(sol5 - sol4, nan=np.inf), axis=0)
            scale = atol + rtol * np.maximum(np.linalg.norm(p0, axis=0), np.linalg.norm(sol5, axis=0))
            err_ratio = err / np.maximum(scale, 1e-300)

            accept = (~stage_invalid) & (err_ratio <= 1.0)

            # PI-style step-size update (standard RK45 exponent)
            factor = safety * np.power(np.maximum(err_ratio, 1e-12), -0.2)
            factor = np.clip(factor, 0.2, 5.0)
            new_h = np.clip(hs * factor, ds_min, ds_max)

            for j, i in enumerate(idx):
                if stage_invalid[j]:
                    # A trial stage (or the final point) left the domain, or the
                    # field vanished. Shrink and retry; terminate only once we
                    # can't shrink any further.
                    if hs[j] <= ds_min * 1.0001:
                        active[i] = False
                    else:
                        h[i] = max(hs[j] * 0.25, ds_min)
                    continue

                if accept[j]:
                    pos[:, i] = sol5[:, j]
                    paths[i].append(np.array([pos[0, i], pos[1, i], pos[2, i], Z_next[j]]))
                    h[i] = new_h[j]
                else:
                    h[i] = new_h[j]   # rejected step: same position, smaller h next try

            if verbose and step_count % 100 == 0:
                print(f"step {step_count}: {active.sum()} / {N} particles still active")

        self._silent = False
        return [np.array(p).T for p in paths]   # each entry: (4, n_i) -> x, y, z, Z_local

class MWScalar(Saveable):
    """The MWDataSet class stores solution data of FEM Time Harmonic simulations."""

    _fields: list[str] = ["freq", "k0", "Sp", "beta", "Pout", "Z0"]
    _copy: list[str] = ["_portmap", "_portnumbers", "port_modes"]

    def __init__(self):
        self.freq: float = None
        self.k0: float = None
        self.Q: float = None
        self.Sp: np.ndarray = None
        self.beta: np.ndarray = None
        self.Z0: np.ndarray = None
        self.Pout: np.ndarray = None
        self._portmap: dict[int | float, int] = dict()
        self._portnumbers: list[int | float] = []
        self.port_modes: list[PortProperties] = []

    def init_sp(self, portnumbers: list[int | float]) -> None:
        """Initialize the S-parameter dataset with the given number of ports."""
        self._portnumbers = portnumbers
        i = 0
        for n in portnumbers:
            self._portmap[n] = i
            i += 1

        self.Sp = np.zeros((i, i), dtype=np.complex128)
        self.Z0 = np.zeros((i,), dtype=np.complex128)
        self.Pout = np.zeros((i,), dtype=np.float64)
        self.beta = np.zeros((i,), dtype=np.complex128)

    def write_S(self, i: int | float, j: int | float, value: complex) -> None:
        self.Sp[self._portmap[i], self._portmap[j]] = value

    def S(self, i: int | float, j: int | float) -> complex:
        """Return the S-parameter corresponding to the given set of indices:

        S11 = obj.S(1,1)

        Args:
            i (int | float): The first port index
            j (int | float): The second port index

        Returns:
            complex: The S-parameter
        """
        return self.Sp[self._portmap[i], self._portmap[j]]

    def add_port_properties(
        self,
        port_number: int,
        mode_number: int,
        smat_index: int | float,
        k0: float,
        beta: float,
        Z0: float | complex,
        Pout: float,
    ) -> None:
        i = self._portmap[smat_index]
        self.beta[i] = beta
        self.Z0[i] = Z0
        self.Pout[i] = Pout


class MWScalarNdim(Saveable):
    _fields: list[str] = ["freq", "k0", "Sp", "beta", "Pout", "Z0"]
    _copy: list[str] = ["_portmap", "_portnumbers"]

    def __init__(self):
        self.freq: np.ndarray = None
        self.k0: np.ndarray = None
        self.Sp: np.ndarray = None
        self.Q: np.ndarray = None
        self.beta: np.ndarray = None
        self.Z0: np.ndarray = None
        self.Pout: np.ndarray = None
        self._portmap: dict[int | float, int] = dict()
        self._portnumbers: list[int | float] = []
        self._dense_frequencies: np.ndarray = None

        # Ports embedded via embed_external_component are marked inactive
        # here rather than removed -- Sp, _portmap and _portnumbers always
        # keep their original full size, so port labels/positions are
        # permanent and never need renumbering or remapping.
        self._active_ports: set[int | float] | None = None

    def renormalize(self, Z0ref: np.ndarray | float | complex) -> MWScalarNdim:
        if isinstance(Z0ref, (float, complex, int)):
            Z0ref = np.ones_like(self.Z0) * Z0ref

        # Shape is (..., M, N, N) — last 3 axes are the core S-parameter array
        leading_shape = self.Sp.shape[:-3]

        if leading_shape:
            Sout = np.empty_like(self.Sp)
            for idx in np.ndindex(leading_shape):
                Sout[idx] = renormalise_s(self.Sp[idx], self.Z0[idx], Z0ref[idx])
        else:
            # Simple (M, N, N) case — no sweep dimensions
            Sout = renormalise_s(self.Sp, self.Z0, Z0ref)

        newndim = MWScalarNdim()
        newndim.freq = self.freq
        newndim.k0 = self.k0
        newndim.Sp = Sout
        newndim.beta = self.beta
        newndim.Z0 = Z0ref
        newndim.Pout = self.Pout
        newndim._portmap = self._portmap
        newndim._portnumbers = self._portnumbers
        newndim._dense_frequencies = self._dense_frequencies
        newndim._active_ports = (
            set(self._active_ports) if self._active_ports is not None else None
        )
        return newndim

    def dense_f(
        self, N: int | None = None, frequencies: list[float] | np.ndarray | None = None
    ) -> np.ndarray:
        """Specify a frequency subsample point density or provide a list of denser frequency points.

        Args:
            N (int): The number of frequency points
            frequencies (list[float] | np.ndarray | None, optional): A list of frequency points. Defaults to None.

        Returns:
            np.ndarray: The new list of frequency points
        """
        if frequencies is not None:
            self._dense_frequencies = np.array(frequencies)
            return frequencies
        self._dense_frequencies = np.linspace(np.min(self.freq), np.max(self.freq), N)
        return self._dense_frequencies

    def S(self, i: int | float, j: int | float) -> np.ndarray:
        """Get the S-parameter for the given port(port mode) index.

        Single mode ports are numbered like: 1, 2, 3 etc
        Ports with multiple modes are numbered. 1.1, 1.2, 1.3 etc

        Args:
            i (int | float): The i-index
            j (int | float): The j-index

        Returns:
            np.ndarray: The resultant S-parameters
        """
        return self.Sp[..., self._portmap[i], self._portmap[j]]

    def combine_ports(
        self, p1: int, p2: int, Z0renorm: np.ndarray | float | complex | None = None
    ) -> MWScalarNdim:
        """Combine ports p1 and p2 into a differential and common mode port respectively.

        The p1 index becomes the differential mode port
        The p2 index becomes the common mode port

        Args:
            p1 (int): The first port number
            p2 (int): The second port number

        Returns:
            MWScalarNdim: _description_
        """
        if p1 == p2:
            raise ValueError("p1 and p2 must be different port numbers")
        if p1 not in self._portmap or p2 not in self._portmap:
            raise KeyError(
                f"Port(s) {p1}, {p2} not found. Available ports: "
                f"{sorted(self._portmap)}"
            )

        F, N, _ = self.Sp.shape
        # Resolve via the port map instead of assuming array position == port
        # number - 1, so this stays correct even after embed_external_component
        # has marked some ports inactive.
        ii = self._portmap[p1]
        jj = self._portmap[p2]

        Sout = self.Sp.copy()
        if Z0renorm is not None:
            Sout = renormalise_s(Sout, Z0renorm, self.Z0)

        idx = np.ones(N, dtype=np.bool)
        idx[[ii, jj]] = False
        others = np.nonzero(idx)[0]
        isqrt2 = 1.0 / np.sqrt(2.0)

        Sout[:, others, ii] = (self.Sp[:, others, ii] - self.Sp[:, others, jj]) * isqrt2
        Sout[:, others, jj] = (self.Sp[:, others, ii] + self.Sp[:, others, jj]) * isqrt2
        Sout[:, ii, others] = (self.Sp[:, ii, others] - self.Sp[:, jj, others]) * isqrt2
        Sout[:, jj, others] = (self.Sp[:, ii, others] + self.Sp[:, jj, others]) * isqrt2

        Sii = self.Sp[:, ii, ii]
        Sij = self.Sp[:, ii, jj]
        Sji = self.Sp[:, jj, ii]
        Sjj = self.Sp[:, jj, jj]

        Sout[:, ii, ii] = 0.5 * (Sii - Sij - Sji + Sjj)
        Sout[:, ii, jj] = 0.5 * (Sii + Sij - Sji - Sjj)
        Sout[:, jj, ii] = 0.5 * (Sii - Sij + Sji - Sjj)
        Sout[:, jj, jj] = 0.5 * (Sii + Sij + Sji + Sjj)

        self.Sp = Sout

        return self

    def embed_external_component(
        self, touchstone_file: str, port_indices: list[int]
    ) -> "MWScalarNdim":
        """Embed an external N-port network (given as a Touchstone file) onto
        a subset of this dataset's ports.

        This does NOT remove ports -- `Sp`, `_portmap`, and `_portnumbers`
        keep their original full size forever. Eliminated ports are marked
        inactive (their rows/cols in `Sp` become NaN) instead of being
        deleted. This means port labels and array positions are permanent
        for the lifetime of the object: calling this repeatedly with
        different, still-active port numbers just works, with no
        renumbering, remapping, or caller-side bookkeeping required, and
        each call correctly folds in the effect of any prior embedding.

        Args:
            touchstone_file (str): The filename of the touchstone file to import
            port_indices (list[int]): The port numbers of this dataset to
                connect the touchstone file to.

        Returns:
            MWScalarNdim: self, with the embedded ports marked inactive.
        """
        from ....read import TouchstoneData

        if self._active_ports is None:
            self._active_ports = set(self._portnumbers)

        missing = [p for p in port_indices if p not in self._active_ports]
        if missing:
            raise KeyError(
                f"Port(s) {missing} are not available (already embedded, or "
                f"don't exist). Active ports: {sorted(self._active_ports)}"
            )
        if len(set(port_indices)) != len(port_indices):
            raise ValueError(f"Duplicate port numbers in port_indices: {port_indices}")

        td = TouchstoneData(touchstone_file)
        S_ext = td.interp_S(self.freq)  # (n_freq, M, M)

        m_pos = [self._portmap[p] for p in port_indices]
        n_labels = sorted(
            self._active_ports - set(port_indices), key=lambda l: self._portmap[l]
        )
        n_pos = [self._portmap[l] for l in n_labels]

        S_nn = self.Sp[:, n_pos][:, :, n_pos]
        S_nm = self.Sp[:, n_pos][:, :, m_pos]
        S_mn = self.Sp[:, m_pos][:, :, n_pos]
        S_mm = self.Sp[:, m_pos][:, :, m_pos]

        n_freq = self.Sp.shape[0]
        I = np.eye(len(m_pos))
        S_red = np.zeros_like(S_nn)
        for i in range(n_freq):
            term = np.linalg.inv(I - S_mm[i] @ S_ext[i])
            S_red[i] = S_nn[i] + S_nm[i] @ S_ext[i] @ term @ S_mn[i]

        # Fold the embedding into the effective S among still-active ports,
        # so a SUBSEQUENT embedding call sees the correct effective network.
        self.Sp[np.ix_(range(n_freq), n_pos, n_pos)] = S_red
        # Mark eliminated ports as no longer meaningful.
        self.Sp[:, m_pos, :] = np.nan
        self.Sp[:, :, m_pos] = np.nan

        self._active_ports -= set(port_indices)

        logger.info(
            f"Embedded {touchstone_file!r} on ports {port_indices}. "
            f"Active ports remaining: {sorted(self._active_ports)}"
        )
        return self

    @property
    def Smat(self) -> np.ndarray:
        """Returns the full S-matrix

        Returns:
            np.ndarray: The S-matrix with shape (nF, nP, nP). Includes all
                original ports; ports eliminated by embed_external_component
                are NaN.
        """
        return self.Sp

    def emmodel(
        self, f_sample: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
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

    def model_S(
        self,
        i: int,
        j: int,
        freq: np.ndarray | None = None,
        Npoles: int | Literal["auto"] = "auto",
        inc_real: bool = True,
        maxpoles: int = 30,
        minpoles: int = 1,
        _warn: bool = True,
    ) -> np.ndarray:
        """Returns an S-parameter model object at a dense frequency range.
        This method uses vector fitting inside the datasets frequency points to determine a model for the linear system.
        If no frequency array is provided the .dense_f(NF) method should have been called.

        Args:
            i (int): The first S-parameter index
            j (int): The second S-parameter index
            freq (np.ndarray | optional): The frequency sample points. Defaults to None
            Npoles (int | 'auto', optional): The number of poles to use (approx 2x divice order). Defaults to 10.
            inc_real (bool, optional): Wether to allow for a real-pole. Defaults to False.

        Returns:
            SparamModel: The SparamModel object
        """
        if freq is None:
            if self._dense_frequencies is None:
                raise ValueError(
                    "No dense frequency space is defined. Either provide a dense frequency grid or call the .dense_f() method."
                )
            else:
                freq = self._dense_frequencies

        shape = np.squeeze(self.S(i, j)).shape
        if len(shape) > 1:
            *dims, nf = self.S(i, j).shape
            nf = len(freq)
            Sarray = np.zeros(tuple(dims) + (nf,), dtype=np.complex128)
            for ids in np.ndindex(*dims):
                Sarray[ids, :] = SparamModel(
                    np.squeeze(self.freq[(*ids, slice(None))]),
                    np.squeeze(self.S(i, j)[(*ids, slice(None))]),
                    n_poles=Npoles,
                    inc_real=inc_real,
                    maxpoles=maxpoles,
                    minpoles=minpoles,
                    _warn=_warn,
                )(freq)
            return Sarray
        else:
            return SparamModel(
                np.squeeze(self.freq),
                np.squeeze(self.S(i, j)),
                n_poles=Npoles,
                inc_real=inc_real,
                maxpoles=maxpoles,
                minpoles=minpoles,
                _warn=_warn,
            )(freq)

    def model_Smat(
        self,
        frequencies: np.ndarray | None = None,
        Npoles: int = 10,
        inc_real: bool = True,
        _warn: bool = True,
    ) -> np.ndarray:
        """Generates a full S-parameter matrix on the provided frequency points using the Vector Fitting algorithm.

        This function output can be used directly with the .save_matrix() method.

        Args:
            frequencies (np.ndarray): The sample frequencies
            Npoles (int, optional): The number of poles to fit. Defaults to 10.
            inc_real (bool, optional): Wether allow for a real pole. Defaults to False.

        Returns:
            np.ndarray: The (Nf,Np,Np) S-parameter matrix. Entries for
                ports eliminated by embed_external_component are NaN
                (they are not vector-fit since their data is NaN).
        """
        if frequencies is None:
            if self._dense_frequencies is None:
                raise ValueError(
                    "No dense frequency space is defined. Either provide a dense frequency grid or call the .dense_f() method."
                )
            else:
                frequencies = self._dense_frequencies

        Nports = len(self._portmap)
        nfreq = frequencies.shape[0]

        Smat = np.full((nfreq, Nports, Nports), np.nan, dtype=np.complex128)
        active = self._active_ports if self._active_ports is not None else set(self._portnumbers)

        for i in active:
            for j in active:
                S = self.model_S(
                    i, j, frequencies, Npoles=Npoles, inc_real=inc_real, _warn=_warn
                )
                Smat[:, self._portmap[i], self._portmap[j]] = S
        return Smat

    def export_touchstone(
        self,
        filename: str,
        Z0ref: float | None = None,
        format: Literal["RI", "MA", "DB"] = "RI",
        custom_comments: list[str] | None = None,
        funit: Literal["Hz", "KHz", "MHz", "GHz"] = "GHz",
        dense_freq: np.ndarray | None = None,
    ):
        """Export the S-parameter data to a touchstone file

        Only ports still active (not eliminated via embed_external_component)
        are exported -- a Touchstone file can't represent internal/embedded
        ports.

        Additionally, one may provide a reference impedance. If this argument is provided, a port impedance renormalization
        will be performed to that common impedance.

        Args:
            filename (str): The File name
            Z0ref (float): The reference impedance to normalize to. Defaults to None
            format (Literal[DB, RI, MA]): The dataformat used in the touchstone file.
            custom_comments : list[str], optional. List of custom comment strings to add to the touchstone file header.
                                                    Each string will be prefixed with "! " automatically.
            dense_freq (np.ndarray | optional): An optional dense interpolation frequency range
        """

        logger.info(f"Exporting S-data to {filename}")

        active = self._active_ports if self._active_ports is not None else set(self._portnumbers)
        active_ports = sorted(active, key=lambda p: self._portmap[p])
        active_pos = [self._portmap[p] for p in active_ports]

        if dense_freq is None:
            freqs = self.freq
            Smat = self.Sp[:, active_pos][:, :, active_pos]
        else:
            freqs = dense_freq
            full_dense = self.model_Smat(dense_freq)
            Smat = full_dense[:, active_pos][:, :, active_pos]

        Z0_active = self.Z0[:, active_pos]

        self.save_smatrix(
            filename,
            Smat,
            freqs,
            format=format,
            Z0ref=Z0ref,
            custom_comments=custom_comments,
            funit=funit,
            Z0_override=Z0_active,
        )

    def save_smatrix(
        self,
        filename: str,
        Smatrix: np.ndarray,
        frequencies: np.ndarray,
        Z0ref: float | None = None,
        format: Literal["RI", "MA", "DB"] = "RI",
        custom_comments: list[str] | None = None,
        funit: Literal["Hz", "KHz", "MHz", "GHz"] = "GHz",
        Z0_override: np.ndarray | None = None,
    ) -> None:
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
            Z0_override (np.ndarray | optional): Reference impedance array
                matching Smatrix's port axis, for when it's a subset of the
                full self.Z0 (e.g. active ports only). Defaults to self.Z0.
        """
        from .touchstone import generate_touchstone

        if Z0ref is not None:
            Z0s = Z0_override if Z0_override is not None else self.Z0
            logger.debug(f"Renormalizing impedances {Z0s}Ω to {Z0ref}Ω")
            # This can be the case if the S-matrix data is interpolated with vectorfitting
            nz, nport = Z0s.shape
            ns = Smatrix.shape[0]
            if Z0s.shape[0] != Smatrix.shape[0]:
                Z0s_out = np.empty((ns, nport), dtype=np.complex128)
                sparse = np.linspace(0, 1, nz)
                dense = np.linspace(0, 1, ns)
                for i in range(nport):
                    Z0s_out[:, i] = np.interp(dense, sparse, Z0s[:, i])
                Z0s = Z0s_out
            Smatrix = renormalise_s(Smatrix, Z0s, Z0ref)

        generate_touchstone(
            filename, frequencies, Smatrix, format, custom_comments, funit
        )

        logger.info("Export complete!")