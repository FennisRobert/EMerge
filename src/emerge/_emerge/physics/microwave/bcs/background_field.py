from typing import Literal, get_args, Callable
from dataclasses import dataclass
from emsutil import Saveable
from ....const import MU0
import numpy as np


############################################################
#               ELEVATION AZIMUTH DEFINITION               #
############################################################


def fK_EA(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    theta: float,
    phi: float,
    psi: float,
    k0: float,
    origin: tuple[float, float, float],
) -> np.ndarray:
    kx = k0 * np.cos(theta) * np.cos(phi)
    ky = k0 * np.cos(theta) * np.sin(phi)
    kz = -k0 * np.sin(theta)
    return np.array([kx, ky, kz])


def fE_EA(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    theta: float,
    phi: float,
    psi: float,
    k0: float,
    origin: tuple[float, float, float],
) -> np.ndarray:
    kx = k0 * np.cos(theta) * np.cos(phi)
    ky = k0 * np.cos(theta) * np.sin(phi)
    kz = -k0 * np.sin(theta)
    xp = x - origin[0]
    yp = y - origin[1]
    zp = z - origin[2]
    Phi = np.exp(-1j * (kx * xp + ky * yp + kz * zp))
    Ex = (np.sin(theta) * np.cos(phi) * np.cos(psi) - np.sin(phi) * np.sin(psi)) * Phi
    Ey = (np.sin(theta) * np.sin(phi) * np.cos(psi) + np.cos(phi) * np.sin(psi)) * Phi
    Ez = (np.cos(theta) * np.cos(psi)) * Phi
    return np.array([Ex, Ey, Ez])


def fH_EA(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    theta: float,
    phi: float,
    psi: float,
    k0: float,
    origin: tuple[float, float, float],
) -> np.ndarray:
    w0 = k0 * 299792458
    kx = k0 * np.cos(theta) * np.cos(phi)
    ky = k0 * np.cos(theta) * np.sin(phi)
    kz = -k0 * np.sin(theta)
    xp = x - origin[0]
    yp = y - origin[1]
    zp = z - origin[2]
    Phi = np.exp(-1j * (kx * xp + ky * yp + kz * zp))
    Ex = (np.sin(theta) * np.cos(phi) * np.cos(psi) - np.sin(phi) * np.sin(psi)) * Phi
    Ey = (np.sin(theta) * np.sin(phi) * np.cos(psi) + np.cos(phi) * np.sin(psi)) * Phi
    Ez = (np.cos(theta) * np.cos(psi)) * Phi

    CEx = -1j * (ky * Ez - kz * Ey) * 1j / (w0 * MU0)
    CEy = -1j * (kz * Ex - kx * Ez) * 1j / (w0 * MU0)
    CEz = -1j * (kx * Ey - ky * Ex) * 1j / (w0 * MU0)
    return np.array([CEx, CEy, CEz])


def fEcurl_EA(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    theta: float,
    phi: float,
    psi: float,
    k0: float,
    origin: tuple[float, float, float],
) -> np.ndarray:
    kx = k0 * np.cos(theta) * np.cos(phi)
    ky = k0 * np.cos(theta) * np.sin(phi)
    kz = -k0 * np.sin(theta)
    xp = x - origin[0]
    yp = y - origin[1]
    zp = z - origin[2]
    Phi = np.exp(-1j * (kx * xp + ky * yp + kz * zp))
    Ex = (np.sin(theta) * np.cos(phi) * np.cos(psi) - np.sin(phi) * np.sin(psi)) * Phi
    Ey = (np.sin(theta) * np.sin(phi) * np.cos(psi) + np.cos(phi) * np.sin(psi)) * Phi
    Ez = (np.cos(theta) * np.cos(psi)) * Phi

    CEx = 1j * (ky * Ez - kz * Ey)
    CEy = 1j * (kz * Ex - kx * Ez)
    CEz = 1j * (kx * Ey - ky * Ex)
    return np.array([CEx, CEy, CEz])


############################################################
#                STANDARD SPHERICAL DEFINITION              #
############################################################

"""Standard (physics/IEEE) spherical convention:
    theta is the polar angle measured from the +z axis (0 <= theta <= pi)
    phi is the azimuthal angle measured from +x towards +y
    k_hat = (sin(theta)cos(phi), sin(theta)sin(phi), cos(theta))
    theta_hat = (cos(theta)cos(phi), cos(theta)sin(phi), -sin(theta))
    phi_hat = (-sin(phi), cos(phi), 0)

The E-field polarization is decomposed as E = cos(psi)*theta_hat + sin(psi)*phi_hat,
i.e. psi=0 corresponds to pure theta-polarization and psi=90 deg corresponds to
pure phi-polarization, consistent with standard far-field E_theta/E_phi convention.
"""


def fK_SPH(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    theta: float,
    phi: float,
    psi: float,
    k0: float,
    origin: tuple[float, float, float],
) -> np.ndarray:
    kx = k0 * np.sin(theta) * np.cos(phi)
    ky = k0 * np.sin(theta) * np.sin(phi)
    kz = k0 * np.cos(theta)
    return np.array([kx, ky, kz])


def fE_SPH(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    theta: float,
    phi: float,
    psi: float,
    k0: float,
    origin: tuple[float, float, float],
) -> np.ndarray:
    kx = k0 * np.sin(theta) * np.cos(phi)
    ky = k0 * np.sin(theta) * np.sin(phi)
    kz = k0 * np.cos(theta)
    xp = x - origin[0]
    yp = y - origin[1]
    zp = z - origin[2]
    Phi = np.exp(-1j * (kx * xp + ky * yp + kz * zp))
    Ex = (np.cos(theta) * np.cos(phi) * np.cos(psi) - np.sin(phi) * np.sin(psi)) * Phi
    Ey = (np.cos(theta) * np.sin(phi) * np.cos(psi) + np.cos(phi) * np.sin(psi)) * Phi
    Ez = (-np.sin(theta) * np.cos(psi)) * Phi
    return np.array([Ex, Ey, Ez])


def fH_SPH(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    theta: float,
    phi: float,
    psi: float,
    k0: float,
    origin: tuple[float, float, float],
) -> np.ndarray:
    w0 = k0 * 299792458
    kx = k0 * np.sin(theta) * np.cos(phi)
    ky = k0 * np.sin(theta) * np.sin(phi)
    kz = k0 * np.cos(theta)
    xp = x - origin[0]
    yp = y - origin[1]
    zp = z - origin[2]
    Phi = np.exp(-1j * (kx * xp + ky * yp + kz * zp))
    Ex = (np.cos(theta) * np.cos(phi) * np.cos(psi) - np.sin(phi) * np.sin(psi)) * Phi
    Ey = (np.cos(theta) * np.sin(phi) * np.cos(psi) + np.cos(phi) * np.sin(psi)) * Phi
    Ez = (-np.sin(theta) * np.cos(psi)) * Phi

    CEx = -1j * (ky * Ez - kz * Ey) * 1j / (w0 * MU0)
    CEy = -1j * (kz * Ex - kx * Ez) * 1j / (w0 * MU0)
    CEz = -1j * (kx * Ey - ky * Ex) * 1j / (w0 * MU0)
    return np.array([CEx, CEy, CEz])


def fEcurl_SPH(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    theta: float,
    phi: float,
    psi: float,
    k0: float,
    origin: tuple[float, float, float],
) -> np.ndarray:
    kx = k0 * np.sin(theta) * np.cos(phi)
    ky = k0 * np.sin(theta) * np.sin(phi)
    kz = k0 * np.cos(theta)
    xp = x - origin[0]
    yp = y - origin[1]
    zp = z - origin[2]
    Phi = np.exp(-1j * (kx * xp + ky * yp + kz * zp))
    Ex = (np.cos(theta) * np.cos(phi) * np.cos(psi) - np.sin(phi) * np.sin(psi)) * Phi
    Ey = (np.cos(theta) * np.sin(phi) * np.cos(psi) + np.cos(phi) * np.sin(psi)) * Phi
    Ez = (-np.sin(theta) * np.cos(psi)) * Phi

    CEx = 1j * (ky * Ez - kz * Ey)
    CEy = 1j * (kz * Ex - kx * Ez)
    CEz = 1j * (kx * Ey - ky * Ex)
    return np.array([CEx, CEy, CEz])


r2d = 180 / np.pi
DEFINITIONS = Literal["EA","SPH"]

"""The background field class abstracts out the methematics of a background EM field.

The two functions that are used by the assembler of the Scattered Field formulation are:
    - BackgroundField.Uinc
    - BackgroundField.Uinc_curl

For the computation of the relative E and H field in post processing, the following methods are used:
    - BackgroundField.E
    - BackgroundField.H

For any user defined BackgroundField object, these functions must be defined.

Two coordinate/angle conventions for the incident-wave definition are supported:
    - "EA": Comsol's elevation-azimuth convention.
    - "SPH": the standard (physics/IEEE) spherical convention, with theta as the
      polar angle from +z and phi as the azimuthal angle from +x.

Custom BackgroundField classes may support these at will.
"""


@dataclass
class BackgroundField(Saveable):
    k0: float
    theta: float
    phi: float
    psi: float
    origin: tuple[float, float, float]
    E0: complex = 1.0 + 0.0j
    definition: DEFINITIONS = "EA"

    def __post_init__(self):
        allowed = get_args(DEFINITIONS)
        if self.definition not in allowed:
            return ValueError(
                f"Cannot define a background field of definition {self.definition}. Please choose from: {allowed}"
            )
        self.origin = tuple(self.origin)

    @property
    def angleset(self) -> tuple[float, float, float]:
        """Returns a tuple of the theta, phi and polarization angle psi.

        Returns:
            tuple[float, float, float]: Tuple of (theta, phi, psi)
        """
        return (self.theta, self.phi, self.psi)

    def __str__(self) -> str:
        r2d = 180 / np.pi
        return f"BackgroundField[amp={self.E0:.3f}V/m, k0={self.k0:.3f}, θ={self.theta * r2d:.1f}°, φ={self.phi * r2d:.1f}°, Ψ={self.psi * r2d:.1f}°, {self.definition}]"

    def Uinc(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Computes the background E-field times -jk0 (for assembly of the forcing vector)

        Args:
            x (np.ndarray): X-coordinate in meters
            y (np.ndarray): Y-coordinate in meters
            z (np.ndarray): Z-coordinate in meters

        Returns:
            np.ndarray: (3,N) complex array
        """
        return -1j * self.k0 * self.E(x, y, z)

    def Uinc_curl(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Computes the background Curl(E)-field times -jk0 (for assembly of the forcing vector)

        Args:
            x (np.ndarray): X-coordinate in meters
            y (np.ndarray): Y-coordinate in meters
            z (np.ndarray): Z-coordinate in meters

        Returns:
            np.ndarray: (3,N) complex array
        """
        return self.curlE(x, y, z)

    def k(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        if self.definition == "EA":
            return fK_EA(x, y, z, self.theta, self.phi, self.psi, self.k0, self.origin)
        elif self.definition == "SPH":
            return fK_SPH(x, y, z, self.theta, self.phi, self.psi, self.k0, self.origin)
        else:
            raise ValueError(
                f"Unsupported spherical coordinate definition {self.definition}"
            )

    def E(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Computes the background E-field

        Args:
            x (np.ndarray): X-coordinate in meters
            y (np.ndarray): Y-coordinate in meters
            z (np.ndarray): Z-coordinate in meters

        Returns:
            np.ndarray: (3,N) complex array
        """
        if self.definition == "EA":
            return self.E0 * fE_EA(
                x, y, z, self.theta, self.phi, self.psi, self.k0, self.origin
            )
        elif self.definition == "SPH":
            return self.E0 * fE_SPH(
                x, y, z, self.theta, self.phi, self.psi, self.k0, self.origin
            )
        else:
            raise ValueError(
                f"Unsupported spherical coordinate definition {self.definition}"
            )

    def H(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Computes the background H-field

        Args:
            x (np.ndarray): X-coordinate in meters
            y (np.ndarray): Y-coordinate in meters
            z (np.ndarray): Z-coordinate in meters

        Returns:
            np.ndarray: (3,N) complex array
        """
        if self.definition == "EA":
            return self.E0 * fH_EA(
                x, y, z, self.theta, self.phi, self.psi, self.k0, self.origin
            )
        elif self.definition == "SPH":
            return self.E0 * fH_SPH(
                x, y, z, self.theta, self.phi, self.psi, self.k0, self.origin
            )
        else:
            raise ValueError(
                f"Unsupported spherical coordinate definition {self.definition}"
            )

    def curlE(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Computes the background Curl(E)-field

        Args:
            x (np.ndarray): X-coordinate in meters
            y (np.ndarray): Y-coordinate in meters
            z (np.ndarray): Z-coordinate in meters

        Returns:
            np.ndarray: (3,N) complex array
        """
        if self.definition == "EA":
            return self.E0 * fEcurl_EA(
                x, y, z, self.theta, self.phi, self.psi, self.k0, self.origin
            )
        elif self.definition == "SPH":
            return self.E0 * fEcurl_SPH(
                x, y, z, self.theta, self.phi, self.psi, self.k0, self.origin
            )
        else:
            raise ValueError(
                f"Unsupported spherical coordinate definition {self.definition}"
            )
    
    def msrcs_angles_rad(self) -> tuple[float, float]:
        """Computes the farfield (theta, phi) angle corresponding to the monostatic
        RCS reflection of this incident plane wave, i.e. the observation direction
        that looks directly back along the incidence direction (-k_hat).

        Returns:
            tuple[float, float]: Tuple of (theta, phi) in the same angle
                convention as self.definition.
        """
        if self.definition == "EA":
            kx = -np.cos(self.theta) * np.cos(self.phi)
            ky = -np.cos(self.theta) * np.sin(self.phi)
            kz = np.sin(self.theta)
            kxy = ((kx**2)+(ky**2))**0.5
            theta = np.atan2(kz, kxy)
            phi = np.atan2(ky,kx)
        elif self.definition == "SPH":
            # -k_hat_SPH(theta,phi) = (-sin(theta)cos(phi), -sin(theta)sin(phi), -cos(theta))
            # which inverts (via the standard spherical-to-Cartesian relations) to
            # theta -> pi - theta, phi -> phi + pi.
            kx = -np.sin(self.theta) * np.cos(self.phi)
            ky = -np.sin(self.theta) * np.sin(self.phi)
            kz = -np.cos(self.theta)
            kxy = ((kx**2)+(ky**2))**0.5
            theta = np.atan2(kxy, kz)
            phi = np.atan2(ky, kx)
        else:
            raise ValueError(
                f"Unsupported spherical coordinate definition {self.definition}"
            )
        return theta, phi
        
    def __hash__(self) -> int:
        return hash(
            (
                self.theta,
                self.phi,
                self.psi,
                self.k0,
                self.origin,
                self.E0,
                self.definition,
            )
        )

    def is_about(self, theta: float, phi: float, psi: float) -> bool:
        return abs(self.theta-theta)< 1e-6 and abs(self.phi-phi)< 1e-6 and abs(self.psi-psi)< 1e-6
    
    def EHK(self, x: float, y: float, z: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Computes the background K-vector, E-field and H-field at a single point.

        Args:
            x (float): X-coordinate in meters
            y (float): Y-coordinate in meters
            z (float): Z-coordinate in meters

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: (E, H, K), each a (3,) complex array
        """
        theta, phi, psi, k0 = self.theta, self.phi, self.psi, self.k0

        if self.definition == "EA":
            kx = k0 * np.cos(theta) * np.cos(phi)
            ky = k0 * np.cos(theta) * np.sin(phi)
            kz = -k0 * np.sin(theta)
        elif self.definition == "SPH":
            kx = k0 * np.sin(theta) * np.cos(phi)
            ky = k0 * np.sin(theta) * np.sin(phi)
            kz = k0 * np.cos(theta)
        else:
            raise ValueError(
                f"Unsupported spherical coordinate definition {self.definition}"
            )

        xp = x - self.origin[0]
        yp = y - self.origin[1]
        zp = z - self.origin[2]
        Phi = 1.0#np.exp(-1j * (kx * xp + ky * yp + kz * zp))

        if self.definition == "EA":
            Ex = (np.sin(theta) * np.cos(phi) * np.cos(psi) - np.sin(phi) * np.sin(psi)) * Phi
            Ey = (np.sin(theta) * np.sin(phi) * np.cos(psi) + np.cos(phi) * np.sin(psi)) * Phi
            Ez = (np.cos(theta) * np.cos(psi)) * Phi
        else:  # SPH
            Ex = (np.cos(theta) * np.cos(phi) * np.cos(psi) - np.sin(phi) * np.sin(psi)) * Phi
            Ey = (np.cos(theta) * np.sin(phi) * np.cos(psi) + np.cos(phi) * np.sin(psi)) * Phi
            Ez = (-np.sin(theta) * np.cos(psi)) * Phi

        w0 = k0 * 299792458
        Hx = -1j * (ky * Ez - kz * Ey) * 1j / (w0 * MU0)
        Hy = -1j * (kz * Ex - kx * Ez) * 1j / (w0 * MU0)
        Hz = -1j * (kx * Ey - ky * Ex) * 1j / (w0 * MU0)

        K = np.array([kx, ky, kz])
        E = self.E0 * np.array([Ex, Ey, Ez])
        H = self.E0 * np.array([Hx, Hy, Hz])

        return E, H, K