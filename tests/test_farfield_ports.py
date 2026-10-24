# [Written by Claude] All code in this module was written by Claude (AI).
"""Far field of several ports in one pass (emr.sc_multi, MWField.farfield_3d_ports) against the per port path."""
import numpy as np
import pytest

import emerge as em
from emr.sc import stratton_chu_ff
from emr.sc_multi import stratton_chu_ff_multi

MM = 1e-3
C0 = 299792458.0
EPS0 = 8.854187818814e-12


def _dipole(p: np.ndarray, xyz: np.ndarray, k0: float) -> tuple[np.ndarray, np.ndarray]:
    """Exact E and H (3, N) of an electric dipole with moment p at the origin (as in test_nearfield)."""
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


def test_multi_kernel_matches_per_excitation_kernel():
    k0 = 2 * np.pi
    vis, wns = _sphere(0.5, 5000)
    moments = [np.array([0.3, -0.5, 1.0]), np.array([1.0, 0.2, 0.0]), np.array([0.0, 0.7j, 0.1])]
    fields = [_dipole(p, vis, k0) for p in moments]
    th, ph = np.meshgrid(np.linspace(0, np.pi, 19), np.linspace(-np.pi, np.pi, 37), indexing="ij")
    tp = np.array([th.ravel(), ph.ravel()])
    Ein = np.concatenate([E for E, _ in fields])
    Hin = np.concatenate([H for _, H in fields])
    Em, Hm = stratton_chu_ff_multi(Ein, Hin, vis, wns, tp, k0)
    for k, (E, H) in enumerate(fields):
        Eref, Href = stratton_chu_ff(E, H, vis, wns, tp, k0)
        assert np.allclose(Em[3 * k : 3 * k + 3], Eref, rtol=0, atol=1e-12 * np.abs(Eref).max())
        assert np.allclose(Hm[3 * k : 3 * k + 3], Href, rtol=0, atol=1e-12 * np.abs(Href).max())


def _two_dipoles():
    L, g, w, R, d = 46 * MM, 2 * MM, 2 * MM, 40 * MM, 20 * MM
    m = em.Simulation("FarfieldPorts")
    air = em.geo.Box(2 * R, 2 * R, 2 * R, position=(-R, -R, -R))
    plates, ports = [], []
    for x0 in (-d / 2, d / 2):
        plates += [em.geo.Plate(np.array([x0 - w / 2, 0, z]), np.array([w, 0, 0]), np.array([0, 0, L / 2 - g / 2]))
                   for z in (g / 2, -L / 2)]
        ports.append(em.geo.Plate(np.array([x0 - w / 2, 0, -g / 2]), np.array([w, 0, 0]), np.array([0, 0, g])))
    for p in plates:
        p.set_material(em.lib.PEC)
    m.mw.set_resolution(0.2)
    m.mw.set_frequency(3e9)
    m.commit_geometry()
    m.generate_mesh()
    for i, p in enumerate(ports):
        m.mw.bc.LumpedPort(p, i + 1, width=w, height=g, direction=em.ZAX, Z0=75)
    rad = air.boundary()
    m.mw.bc.AbsorbingBoundary(rad)
    return m.mw.run_sweep().field[0], rad


@pytest.mark.filterwarnings("ignore")
def test_farfield_3d_ports_matches_farfield_3d():
    field, rad = _two_dipoles()
    thetas, phis = np.linspace(0, np.pi, 19), np.linspace(-np.pi, np.pi, 37)
    for syms in (None, ["Hx"]):
        field.excite_port(2)
        multi = field.farfield_3d_ports(rad, thetas, phis, syms=syms)
        assert field.excitation[2] == 1.0 and field.excitation[1] == 0.0
        assert list(multi) == [1, 2]
        for p, ff in multi.items():
            field.excite_port(p)
            ref = field.farfield_3d(rad, thetas, phis, syms=syms)
            assert np.allclose(ff._E, ref._E, rtol=0, atol=1e-10 * np.abs(ref._E).max())
            assert np.allclose(ff._H, ref._H, rtol=0, atol=1e-10 * np.abs(ref._H).max())
            assert ff.Ptot == pytest.approx(ref.Ptot, rel=1e-10)
