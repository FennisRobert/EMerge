# [Written by Claude] All code in this module was written by Claude (AI).
import numpy as np

import emerge as em

""" FloquetPort: TE and TM Floquet modes, grating modes and the scan angle of the cell."""

mm = 0.001
C0 = 299792458.0
Z0 = 376.730313668


def _slab(k0, theta, er, d, pol):
    """Reflection and transmission of the tangential E of a slab in air (at its faces)."""
    kt = k0 * np.sin(theta)
    kz1, kz2 = np.sqrt(complex(k0**2 - kt**2)), np.sqrt(complex(er * k0**2 - kt**2))
    Z1, Z2 = (k0 * Z0 / kz1, k0 * Z0 / kz2) if pol == "TE" else (kz1 * Z0 / k0, kz2 * Z0 / (k0 * er))
    r = (Z2 - Z1) / (Z2 + Z1)
    p = np.exp(-2j * kz2 * d)
    return r * (1 - p) / (1 - r**2 * p), (1 - r**2) * np.exp(-1j * kz2 * d) / (1 - r**2 * p)


def _slab_cell(cell, er, d, L1, freq, resolution=0.1):
    em.cleanup()
    sim = em.Simulation("floquet", loglevel="ERROR")
    sim.set_periodic_cell(cell)
    H = 2 * L1 + d
    cell.volume(0, H)
    cell.volume(L1, L1 + d).set_material(em.Material(er=er, name="slab"))
    top, bot = cell.port_face(H), cell.port_face(0)
    sim.commit_geometry()
    sim.mw.set_frequency(freq)
    sim.mw.set_resolution(resolution)
    sim.generate_mesh()
    return sim, top, bot


def test_floquet_port_te_tm_oblique_slab():
    # Uniform slab at θ = 40°, φ = 30° with TE00 and TM00 in both ports: both polarizations against the
    # analytic slab, no cross polarization, power conserved.
    er, d, L1, f, th = 4.0, 6 * mm, 12 * mm, 10e9, 40.0
    cell = em.RectCell(10 * mm, 10 * mm)
    sim, top, bot = _slab_cell(cell, er, d, L1, f)
    sim.mw.bc.floquet_port(top, 1, number_of_modes=2)
    sim.mw.bc.floquet_port(bot, 2, number_of_modes=2)
    cell.set_scanangle(th, 30)
    Sp = sim.mw.run_sweep().scalar.grid.Sp[0]  # indices 1.1 (TE), 1.2 (TM), 2.1, 2.2
    em.cleanup()
    k0 = 2 * np.pi * f / C0
    kz = k0 * np.cos(th * np.pi / 180)
    for i, pol in enumerate(("TE", "TM")):
        G, T = _slab(k0, th * np.pi / 180, er, d, pol)
        assert abs(Sp[i, i] - G * np.exp(-2j * kz * L1)) < 3e-3, (pol, Sp[i, i])
        assert abs(Sp[2 + i, i] - T * np.exp(-2j * kz * L1)) < 3e-3, (pol, Sp[2 + i, i])
    assert abs(Sp[1, 0]) < 1e-3 and abs(Sp[3, 0]) < 1e-3
    assert np.abs(np.sum(np.abs(Sp) ** 2, axis=0) - 1).max() < 1e-3


def test_floquet_port_scan_angle_before_ports():
    # The scan angle set before the ports (and before the periodic BCs) are created also applies to them.
    er, d, L1, f, th = 4.0, 5 * mm, 10 * mm, 8e9, 20.0
    cell = em.HexCell((-10 * mm, 6 * mm, 0), (-10 * mm, -6 * mm, 0), (0, -6 * mm, 0))
    sim, top, bot = _slab_cell(cell, er, d, L1, f)
    cell.set_scanangle(th, 30)
    p1 = sim.mw.bc.floquet_port(top, 1, number_of_modes=2)
    sim.mw.bc.floquet_port(bot, 2, number_of_modes=2)
    Sp = sim.mw.run_sweep().scalar.grid.Sp[0]
    em.cleanup()
    assert abs(p1.scan_theta - th * np.pi / 180) < 1e-12
    k0 = 2 * np.pi * f / C0
    G, _ = _slab(k0, th * np.pi / 180, er, d, "TE")
    assert abs(Sp[0, 0] - G * np.exp(-2j * k0 * np.cos(th * np.pi / 180) * L1)) < 3e-3


def test_floquet_port_grating_modes():
    # Lamellar grating (period 30 mm, 50 % εr = 4 bars, 6 mm thick), normal incidence, TE (Ey). Above 10 GHz the
    # (±1, 0) orders propagate. Diffraction efficiencies against RCWA (Moharam & Gaylord, 81 orders).
    period, L1, d = 30 * mm, 12 * mm, 6 * mm
    freqs = np.array([9.25e9, 12.25e9])
    rcwa = {9.25e9: (0.3212, 0.0, 0.6788, 0.0), 12.25e9: (0.0001, 0.0134, 0.6852, 0.1440)}  # R0, R1, T0, T1
    em.cleanup()
    sim = em.Simulation("grating", loglevel="ERROR")
    cell = em.RectCell(period, 5 * mm)
    sim.set_periodic_cell(cell)
    cell.volume(0, 2 * L1 + d)
    em.geo.Box(period / 2, 5 * mm, d, position=(-period / 4, -2.5 * mm, L1)).set_material(em.Material(er=4, name="bar"))
    top, bot = cell.port_face(2 * L1 + d), cell.port_face(0)
    sim.commit_geometry()
    sim.mw.set_frequency(freqs)
    sim.mw.set_resolution(0.12)
    sim.generate_mesh()
    p1 = sim.mw.bc.floquet_port(top, 1, number_of_modes=None)
    p2 = sim.mw.bc.floquet_port(bot, 2, number_of_modes=None)
    g = sim.mw.run_sweep().scalar.grid
    em.cleanup()
    modes = p1.floquet_modes()
    assert [(m, n, pol) for _, m, n, pol in modes] == [
        (0, 0, "TE"), (0, 0, "TM"), (-1, 0, "TE"), (-1, 0, "TM"), (1, 0, "TE"), (1, 0, "TM")]
    assert [index for index, *_ in modes] == [1.1, 1.2, 1.3, 1.4, 1.5, 1.6]
    i1 = {(m, n, pol): index for index, m, n, pol in modes}
    i2 = {(m, n, pol): index for index, m, n, pol in p2.floquet_modes()}
    for fi, f in enumerate(freqs):
        R0 = abs(g.S(i1[(0, 0, "TE")], 1.1).ravel()[fi]) ** 2
        R1 = abs(g.S(i1[(1, 0, "TE")], 1.1).ravel()[fi]) ** 2
        T0 = abs(g.S(i2[(0, 0, "TE")], 1.1).ravel()[fi]) ** 2
        T1 = abs(g.S(i2[(1, 0, "TE")], 1.1).ravel()[fi]) ** 2
        direction = p2.order_direction(f, 1, 0)
        if f < C0 / period:  # the (±1, 0) modes are evanescent: amplitude ratios, not efficiencies
            assert direction is None
            R1 = T1 = 0.0
        else:
            assert abs(direction[0] - np.degrees(np.arcsin(C0 / (f * period)))) < 1e-9
        np.testing.assert_allclose([R0, R1, T0, T1], rcwa[f], atol=3e-3)
