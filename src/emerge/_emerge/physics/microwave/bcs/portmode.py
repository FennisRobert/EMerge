from dataclasses import dataclass
from ....elements.nedleg2 import FieldFunctionClass
from emsutil import Saveable
import numpy as np
from typing import Literal
from loguru import logger
from ....const import C0, MU0, Z0, EPS0


@dataclass
class PortMode(Saveable):
    modefield: np.ndarray
    E_function: FieldFunctionClass
    H_function: FieldFunctionClass
    k0: float
    beta: float
    energy: float = 0
    norm_factor: float = 1
    freq: float = 0
    neff: float = 1
    Z0: float = 50.0
    modetype: Literal["TEM", "TE", "TM"] = "TEM"

    @property
    def w0(self) -> float:
        return C0 * self.k0

    def __post_init__(self):
        self.neff = self.beta / self.k0
        self.energy = np.mean(np.abs(self.modefield) ** 2)
        self.E_function.beta = self.beta
        self.H_function.beta = self.beta

    def __str__(self):
        return f"PortMode(k0={self.k0}, beta={self.beta}({self.neff:.3f}))"

    def flip_polarity(self) -> None:
        """Flips the polarity of the port mode (180 degree phase shift)"""
        self.E_function.flip_polarity()
        self.H_function.flip_polarity()

    def normalize_power(self, power: complex) -> None:
        """Applies a correction factor to normalize the power to 1W

        Args:
            power (complex): The current port mode power
        """
        self.norm_factor = np.sqrt(1 / np.abs(power))
        self.E_function.constant *= self.norm_factor
        self.H_function.constant *= self.norm_factor
        logger.debug(f".. setting port mode amplitude to: {self.norm_factor:.2f} ")

    def compute_kappa(self, nodes: np.ndarray, tris: np.ndarray) -> complex:
        """Computes the kappa_m coefficient for Wave Port Boundary Conditions"""
        const = 1j / (self.w0 * MU0)
        from ....mth.integrals import surface_integral

        efunc = self.E_function
        gamma_m = 1j * self.beta
        # print(
        #     f"Computing Kappa for mode k0={self.k0:.3f} and beta={self.beta:.3f}, norm_factor = {self.norm_factor:.3f}"
        # )

        def integrand(xs: np.ndarray, ys: np.ndarray, zs: np.ndarray):
            Etm = efunc.calcExy(xs, ys, zs)
            gEzm = efunc.calcEzGrad(xs, ys, zs)
            term1 = gamma_m * (
                Etm[0, :] * Etm[0, :] + Etm[1, :] * Etm[1, :] + Etm[2, :] * Etm[2, :]
            )
            term2 = (
                Etm[0, :] * gEzm[0, :] + Etm[1, :] * gEzm[1, :] + Etm[2, :] * gEzm[2, :]
            )
            return const * (term1 - term2)

        kappa_m = surface_integral(
            nodes,
            tris,
            integrand,
            np.ones((tris.shape[1],), dtype=np.complex128),
            gq_order=5,
        )
        return kappa_m
