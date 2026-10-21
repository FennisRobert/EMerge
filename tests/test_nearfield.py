import numpy as np

from emr.sc import stratton_chu_ff, stratton_chu_nf

C0 = 299792458.0
EPS0 = 8.854187818814e-12


def _dipole(p: np.ndarray, xyz: np.ndarray, k0: float) -> tuple[np.ndarray, np.ndarray]:
    """Exact E and H (3, N) of an electric dipole with moment p at the origin (exp(+jwt))."""
    r = np.linalg.norm(xyz, axis=0)
    n = xyz / r
    g = np.exp(-1j * k0 * r)
    ndp = p @ n
    nxp = np.cross(n.T, p).T
    nxpxn = np.cross(nxp.T, n.T).T
    E = (k0**2 * nxpxn * g / r + (3 * n * ndp - p[:, None]) * (1 / r**3 + 1j * k0 / r**2) * g) / (4 * np.pi * EPS0)
    H = C0 * k0**2 / (4 * np.pi) * nxp * g / r * (1 + 1 / (1j * k0 * r))
    return E, H


def _sphere(R: float, N: int) -> tuple[np.ndarray, np.ndarray]:
    """Fibonacci sampling of a sphere: points and area weighted outward normals (3, N)."""
    i = np.arange(N) + 0.5
    th = np.arccos(1 - 2 * i / N)
    ph = np.pi * (1 + 5**0.5) * i
    n = np.array([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)])
    return R * n, n * (4 * np.pi * R**2 / N)


K0 = 2 * np.pi
P = np.array([0.3, -0.5, 1.0])
VIS, WNS = _sphere(0.5, 20000)
EIN, HIN = _dipole(P, VIS, K0)


def _targets(r: float, N: int = 50, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    d = rng.standard_normal((3, N))
    return r * d / np.linalg.norm(d, axis=0)


def test_nearfield_reproduces_dipole_outside_surface():
    for r in (0.8, 2.0, 7.0):
        xyz = _targets(r)
        E, H = stratton_chu_nf(EIN, HIN, VIS, WNS, xyz, K0)
        Eref, Href = _dipole(P, xyz, K0)
        assert np.max(np.linalg.norm(E - Eref, axis=0)) < 1e-3 * np.max(np.linalg.norm(Eref, axis=0))
        assert np.max(np.linalg.norm(H - Href, axis=0)) < 1e-3 * np.max(np.linalg.norm(Href, axis=0))


def test_nearfield_vanishes_inside_surface():
    xyz = _targets(0.25)
    E, H = stratton_chu_nf(EIN, HIN, VIS, WNS, xyz, K0)
    Esurf = np.max(np.linalg.norm(EIN, axis=0))
    Hsurf = np.max(np.linalg.norm(HIN, axis=0))
    assert np.max(np.linalg.norm(E, axis=0)) < 1e-3 * Esurf
    assert np.max(np.linalg.norm(H, axis=0)) < 1e-3 * Hsurf


def test_nearfield_tends_to_farfield_kernel():
    r = 1e5
    xyz = _targets(r)
    theta = np.arccos(xyz[2] / r)
    phi = np.arctan2(xyz[1], xyz[0])
    Enf, Hnf = stratton_chu_nf(EIN, HIN, VIS, WNS, xyz, K0)
    Eff, Hff = stratton_chu_ff(EIN, HIN, VIS, WNS, np.array([theta, phi]), K0)
    scale = r * np.exp(1j * K0 * r)
    assert np.allclose(Enf * scale, Eff, rtol=0, atol=1e-4 * np.abs(Eff).max())
    assert np.allclose(Hnf * scale, Hff, rtol=0, atol=1e-4 * np.abs(Hff).max())
