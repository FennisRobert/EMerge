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

        self._Texcite: np.ndarray = 1.0

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
    def relative(self) -> MWData:
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
        self.excitation = {key: 0.0 for key in self._fields.keys()}

        # Freq sweep with ports
        if len(self.port_modes) > 0:
            self.excitation[self.port_modes[0].smat_index] = 1.0 + 0.0j
        elif len(self.background_fields) > 0:
            self.excitation[self.background_fields[0]] = 1.0 + 0.0j

    def excite_port(
        self, number: int | float, excitation: complex = 1.0 + 0.0j
    ) -> None:
        """Excite a single port provided by a given port number

        Args:
            number (int): The port number to excite
            coefficient (complex): The port excitation. Defaults to 1.0 + 0.0j
        """
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
        E, H, Ptot = self.farfield(theta, phi, faces, origin, syms=syms)
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

        T, P = np.meshgrid(thetas, phis)

        E, H, Ptot = self.farfield(
            T.flatten(), P.flatten(), faces, origin=origin, syms=syms
        )
        E = E.reshape((3,) + T.shape)
        H = H.reshape((3,) + T.shape)

        return EHFieldFF(E, H, T, P, Ptot, freq=self.freq)

    def embed_external_component(self, touchstone_file: str, port_indices: list[int], Smat: np.ndarray) -> MWScalarNdim:
        # 1. Load the external component
        from ....read import TouchstoneData

        td = TouchstoneData(touchstone_file)
        fem_s_matrix = Smat
        S_ext = td.interp_S(self.freq)  # (n_freq, M, M)
        port_indices = [ind-1 for ind in port_indices]
        all_ports = np.arange(fem_s_matrix.shape[1])
        n_ports = np.setdiff1d(all_ports, port_indices)
        
        # 2. Partition the block matrix
        # S_nn, S_nm, S_mn, S_mm
        S_nn = fem_s_matrix[n_ports][:, n_ports]
        S_nm = fem_s_matrix[n_ports][:, port_indices]
        S_mn = fem_s_matrix[port_indices][:, n_ports]
        S_mm = fem_s_matrix[port_indices][:, port_indices]
        
        # 3. Compute the reduction: S_red = S_nn + S_nm @ S_ext @ inv(I - S_mm @ S_ext) @ S_mn
        # Use np.linalg.solve for stability instead of direct inv
        N = len(n_ports)
        M = len(port_indices)
        I_n = np.eye(N)
        I_m = np.eye(M)
        # Pre-compute the coupling term: S_ext * (I - S_mm * S_ext)^-1 * S_mn
        coupling = S_ext @ np.linalg.inv(I_m - S_mm @ S_ext) @ S_mn
        zero_nm = np.zeros((N, M))
        zero_mn = np.zeros((M, N))
        self._Texcite = np.block([
            [I_n,      zero_nm],
            [coupling, zero_mn]
        ]).squeeze()
    
    def farfield(
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
            E, H, _ = self.farfield(theta, phi, faces, origin, syms)
            return E[0, :], E[1, :], E[2, :], H[0, :], H[1, :], H[2, :]

        return dict(freq=freq, ff_function=function)


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

        F, N, _ = self.Sp.shape
        p1 = p1 - 1
        p2 = p2 - 1

        if not (0 <= p1 < N and 0 <= p2 < N):
            raise IndexError(f"Ports {p1 + 1} or {p2 + 1} are out of range {N}")

        Sout = self.Sp.copy()
        if Z0renorm is not None:
            Sout = renormalise_s(Sout, Z0renorm, self.Z0)

        ii, jj = p1, p2
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

    def embed_external_component(self, touchstone_file: str, port_indices: list[int]) -> MWScalarNdim:
        # 1. Load the external component
        from ....read import TouchstoneData

        td = TouchstoneData(touchstone_file)
        fem_s_matrix = self.Smat
        S_ext = td.interp_S(self.freq)  # (n_freq, M, M)
        port_indices = [ind-1 for ind in port_indices]
        n_freq = fem_s_matrix.shape[0]
        all_ports = np.arange(fem_s_matrix.shape[1])
        n_ports = np.setdiff1d(all_ports, port_indices)
        
        # 2. Partition the block matrix
        # S_nn, S_nm, S_mn, S_mm
        S_nn = fem_s_matrix[:, n_ports][:, :, n_ports]
        S_nm = fem_s_matrix[:, n_ports][:, :, port_indices]
        S_mn = fem_s_matrix[:, port_indices][:, :, n_ports]
        S_mm = fem_s_matrix[:, port_indices][:, :, port_indices]
        
        # 3. Compute the reduction: S_red = S_nn + S_nm @ S_ext @ inv(I - S_mm @ S_ext) @ S_mn
        # Use np.linalg.solve for stability instead of direct inv
        I = np.eye(len(port_indices))
        
        S_red = np.zeros_like(S_nn)
        for i in range(n_freq):
            # Term = (I - S_mm * S_ext)^-1
            term = np.linalg.inv(I - S_mm[i] @ S_ext[i])
            S_red[i] = S_nn[i] + S_nm[i] @ S_ext[i] @ term @ S_mn[i]

        new_ndim = MWScalarNdim()
        new_ndim.freq = self.freq
        new_ndim.k0 = self.k0
        new_ndim.Sp  = S_red
        new_ndim.Q = None
        new_ndim.beta = self.beta[:,n_ports]
        new_ndim.Z0 = self.Z0[:,n_ports]
        new_ndim._portmap = self._portmap
        new_ndim._portnumbers = self._portnumbers
        return new_ndim
        
    @property
    def Smat(self) -> np.ndarray:
        """Returns the full S-matrix

        Returns:
            np.ndarray: The S-matrix with shape (nF, nP, nP)
        """
        Nports = len(self._portmap)
        nfreq = self.freq.shape[0]

        Smat = np.zeros((nfreq, Nports, Nports), dtype=np.complex128)

        for i in self._portnumbers:
            for j in self._portnumbers:
                Smat[:, i - 1, j - 1] = self.S(i, j)

        return Smat

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
            np.ndarray: The (Nf,Np,Np) S-parameter matrix
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

        Smat = np.zeros((nfreq, Nports, Nports), dtype=np.complex128)

        for i in self._portnumbers:
            for j in self._portnumbers:
                S = self.model_S(
                    i, j, frequencies, Npoles=Npoles, inc_real=inc_real, _warn=_warn
                )
                Smat[:, i - 1, j - 1] = S
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
            dense_freq (np.ndarray | optional): An optional dense interpolation frequency range
        """

        logger.info(f"Exporting S-data to {filename}")
        Nports = len(self._portmap)

        if dense_freq is None:
            freqs = self.freq
            Smat = np.zeros((len(freqs), Nports, Nports), dtype=np.complex128)

            for i in range(1, Nports + 1):
                for j in range(1, Nports + 1):
                    S = self.S(i, j)
                    Smat[:, i - 1, j - 1] = S
        else:
            freqs = dense_freq
            Smat = np.zeros((len(freqs), Nports, Nports), dtype=np.complex128)

            for i in range(1, Nports + 1):
                for j in range(1, Nports + 1):
                    S = self.model_S(i, j, dense_freq)
                    Smat[:, i - 1, j - 1] = S

        self.save_smatrix(
            filename,
            Smat,
            freqs,
            format=format,
            Z0ref=Z0ref,
            custom_comments=custom_comments,
            funit=funit,
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
        """
        from .touchstone import generate_touchstone

        if Z0ref is not None:
            Z0s = self.Z0
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
