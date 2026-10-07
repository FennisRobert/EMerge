# This file was written by Claude Opus 5.5 (Anthropic) for the EMerge project.
# No copyright is claimed on this file.

"""Sample selection for the adaptive frequency sweep.

The S-matrix is modelled with a set-valued AAA rational approximation
(Nakatsukasa, Sete & Trefethen, SIAM J. Sci. Comput. 2018; Lietaert et al.
2022): every S_ij shares one barycentric denominator, i.e. one set of poles,
as it physically should.

Error estimate: the model fitted at the previous iteration is compared with
the model fitted at the current one. Their difference is an a-posteriori
estimate of the previous model's error, so the current model is in general
considerably better than the estimate suggests. New samples go where the
two models disagree most.
"""

from __future__ import annotations

import numpy as np
from loguru import logger


class AAAModel:
    """Barycentric rational model r(x) = sum_j w_j F_j/(x-x_j) / sum_j w_j/(x-x_j)."""

    def __init__(self, x: np.ndarray, F: np.ndarray, w: np.ndarray, residual: float):
        self.x = x  # (m,) support points
        self.F = F  # (m, K) support values
        self.w = w  # (m,) barycentric weights
        self.residual = residual  # max fit error at the non-support samples

    def __call__(self, x: np.ndarray) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            C = 1.0 / (x[:, None] - self.x[None, :])
            R = (C @ (self.w[:, None] * self.F)) / (C @ self.w)[:, None]
        # Evaluating exactly on a support point gives inf/inf
        ix, jx = np.nonzero(x[:, None] == self.x[None, :])
        R[ix] = self.F[jx]
        return R


def aaa(x: np.ndarray, F: np.ndarray, tol: float) -> AAAModel:
    """Set-valued AAA fit of the columns of F (n, K) sampled at x (n,).

    Support points are added greedily at the sample with the largest residual
    until every sample is reproduced to within tol. The number of support
    points is capped such that the Loewner system stays overdetermined (rank
    of F counts, since e.g. S12 = S21 adds no information). Otherwise the
    weights are not unique and a support point can get a ~zero weight, which
    makes the model only pretend to interpolate it.
    """
    n = x.shape[0]
    rank = max(1, np.linalg.matrix_rank(F, tol=tol))
    m_max = max(1, rank * n // (rank + 1))

    free = np.ones(n, dtype=bool)
    err = np.max(np.abs(F - F.mean(axis=0)), axis=1)
    support: list[int] = []
    w = np.ones(1)

    for _ in range(m_max):
        j = int(np.argmax(err))
        if support and err[j] <= tol:
            break
        support.append(j)
        free[j] = False

        xs, Fs = x[support], F[support]
        C = 1.0 / (x[free, None] - xs[None, :])
        L = (F[free, None, :] - Fs[None, :, :]) * C[:, :, None]
        L = L.transpose(0, 2, 1).reshape(-1, len(support))
        w = np.linalg.svd(L)[2][-1].conj()

        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            R = (C @ (w[:, None] * Fs)) / (C @ w)[:, None]
        err = np.zeros(n)
        err[free] = np.nan_to_num(np.max(np.abs(F[free] - R), axis=1), nan=np.inf)

    return AAAModel(x[support], F[support], w, float(err.max()))


class RationalModel:
    """AAA model of sampled S-parameters, evaluated directly in frequency.

    Args:
        f (np.ndarray): Sample frequencies, shape (n,).
        S (np.ndarray): Samples, shape (n, ...), e.g. (n, P, P).
        tol (float): Absolute fit tolerance.
    """

    def __init__(self, f: np.ndarray, S: np.ndarray, tol: float):
        f = np.asarray(f, dtype=float)
        self.fc = 0.5 * (f.max() + f.min())
        self.hw = 0.5 * (f.max() - f.min()) or 1.0
        self.shape = S.shape[1:]
        self._aaa = aaa(self._x(f), S.reshape(f.shape[0], -1), tol)
        self.residual = self._aaa.residual

    def _x(self, f: np.ndarray) -> np.ndarray:
        """Map frequencies to [-1, 1] for a well conditioned fit."""
        return (np.atleast_1d(np.asarray(f, dtype=float)) - self.fc) / self.hw

    def __call__(self, f: np.ndarray) -> np.ndarray:
        x = self._x(f)
        return self._aaa(x).reshape(x.shape + self.shape)


class AdaptiveFrequencySampler:
    """Chooses where to sample next in an adaptive frequency sweep.

    Usage:
        sampler = AdaptiveFrequencySampler(fmin, fmax, tol, batch_size)
        freqs = sampler.initial(n)
        while freqs:
            sampler.add(freqs, solve(freqs))
            freqs = sampler.propose()
    """

    N_EVAL = 33  # evaluation points per interval for the error estimate
    MARGIN = 0.2  # new samples stay this fraction of an interval away from its ends

    def __init__(
        self,
        fmin: float,
        fmax: float,
        tol: float,
        batch_size: int = 1,
        max_samples: int = 100,
    ):
        self.fmin, self.fmax = fmin, fmax
        self.tol = tol
        self.batch_size = max(1, batch_size)
        self.max_samples = max_samples

        self.f = np.empty(0)
        self.S: np.ndarray | None = None  # (n, P, P)
        self.model: RationalModel | None = None
        self.prev_model: RationalModel | None = None
        self.n_passed = 0  # consecutive iterations with error < tol
        self.error = np.inf

    def initial(self, n: int) -> list[float]:
        return list(np.linspace(self.fmin, self.fmax, max(n, 3)))

    @property
    def n(self) -> int:
        return self.f.shape[0]

    @property
    def converged(self) -> bool:
        return self.n_passed >= 2

    @property
    def fit_tol(self) -> float:
        """Tolerance of the AAA fits; well below tol so fit error does not mask model error."""
        return 0.1 * self.tol

    def _fit(self, f: np.ndarray, S: np.ndarray) -> RationalModel:
        return RationalModel(f, S, self.fit_tol)

    def add(self, freqs: list[float], S: np.ndarray) -> None:
        """Adds solved samples, S with shape (len(freqs), P, P), and refits."""
        f = np.concatenate([self.f, freqs])
        S = np.asarray(S) if self.S is None else np.concatenate([self.S, S])
        order = np.argsort(f)
        self.f, self.S = f[order], S[order]

        # The first comparison needs a predecessor: fit every other sample.
        self.prev_model = self.model or self._fit(self.f[::2], self.S[::2])
        self.model = self._fit(self.f, self.S)

    def propose(self) -> list[float]:
        """Returns the next batch of frequencies, empty once converged."""
        f1, f2 = self.f[:-1], self.f[1:]
        t = np.linspace(0, 1, self.N_EVAL)[1:-1]
        fe = f1[:, None] + (f2 - f1)[:, None] * t[None, :]  # (n-1, N_EVAL-2)
        diff = np.abs(self.model(fe.ravel()) - self.prev_model(fe.ravel()))
        diff = diff.reshape(fe.shape + (-1,)).max(axis=2)

        # A model that cannot reproduce its own samples has not resolved the response
        interval_err = diff.max(axis=1)
        self.error = max(float(interval_err.max()), self.model.residual)
        self.n_passed = self.n_passed + 1 if self.error < self.tol else 0
        logger.info(
            f"Adaptive sweep: {self.n} samples, estimated error {self.error:.2e} (tol {self.tol:.1e})"
        )

        if self.converged:
            logger.info(f"Adaptive sweep converged with {self.n} samples.")
            return []
        if self.n >= self.max_samples:
            logger.warning(
                f"Adaptive sweep stopped at max_samples={self.max_samples} with estimated error {self.error:.2e}."
            )
            return []

        # Every interval whose error exceeds tol must be refined regardless, so
        # those can be solved concurrently. Below tol only a single confirming
        # sample is taken at the most uncertain point.
        ranked = np.argsort(interval_err)[::-1]
        n_new = int(np.sum(interval_err >= self.tol)) or 1
        n_new = min(n_new, self.batch_size, self.max_samples - self.n)

        tclip = (t >= self.MARGIN) & (t <= 1 - self.MARGIN)
        new = []
        for i in ranked[:n_new]:
            k = np.argmax(np.where(tclip, diff[i], -1.0))
            new.append(float(fe[i, k]))
        return sorted(new)
