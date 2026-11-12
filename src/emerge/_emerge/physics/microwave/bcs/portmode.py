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
    # [Written by Claude] Cache of robin_gamma()
    _robin_gamma: complex | None = None

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
        # [Written by Claude] E and H usually interpolate the same coefficient array: flipping both flipped it back.
        if self.H_function.field is not self.E_function.field:
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

    # [Written by Claude]
    def urinv_t(self) -> complex:
        """The transverse inverse relative permeability 1/μr of the port face (area weighted mean).

        The port boundary condition sets (1/μr) n×∇×E, the tangential magnetic field, so every port term
        carries this factor. It is 1 for a port face without magnetic material.
        """
        hf = self.H_function
        d = hf.diadic
        if d is None:
            return 1.0
        p = hf.nodes[:2, hf.tris]
        area = 0.5 * np.abs((p[0, 1] - p[0, 0]) * (p[1, 2] - p[1, 0]) - (p[0, 2] - p[0, 0]) * (p[1, 1] - p[1, 0]))
        return complex(np.sum(area * 0.5 * (d[0, 0, :] + d[1, 1, :])) / np.sum(area))

    # [Written by Claude]
    def robin_gamma(self) -> complex:
        """The Robin coefficient γ of the mode: n×∇×E_t = -γ E_t for the mode leaving through the port.

        γ = ∫ (jβ e_t + ∇e_z)·e_t dS / ∫ e_t·e_t dS. It is jβ for TE and TEM modes (e_z = 0) and j k²/β for TM
        modes (∇e_z = -j kc²/β e_t, k² = εr μr k0²), so a Robin port terminates a TM mode without reflection
        too. For a hybrid mode it is the best single coefficient.
        """
        if self._robin_gamma is not None:
            return self._robin_gamma
        from ....mth.integrals import surface_integral

        efunc = self.E_function
        nodes = np.array(efunc.cs.in_global_cs(efunc.nodes[0], efunc.nodes[1], efunc.nodes[2]))
        const = np.ones((efunc.tris.shape[1],), dtype=np.complex128)

        def w_dot_e(xs, ys, zs):
            w = efunc.calc_eff_modeprofile(xs, ys, zs)
            e = efunc.calcExy(xs, ys, zs)
            return w[0] * e[0] + w[1] * e[1] + w[2] * e[2]

        def e_dot_e(xs, ys, zs):
            e = efunc.calcExy(xs, ys, zs)
            return e[0] * e[0] + e[1] * e[1] + e[2] * e[2]

        num = surface_integral(nodes, efunc.tris, w_dot_e, const, gq_order=5)
        den = surface_integral(nodes, efunc.tris, e_dot_e, const, gq_order=5)
        self._robin_gamma = complex(num / den) if abs(den) > 0 else 1j * self.beta
        return self._robin_gamma

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
            return const * (term1 + term2)

        kappa_m = surface_integral(
            nodes,
            tris,
            integrand,
            np.ones((tris.shape[1],), dtype=np.complex128),
            gq_order=5,
        )
        return kappa_m
