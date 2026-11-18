# [Written by Claude] All code in this module was written by Claude (AI).
"""Lumped port S-parameters are power waves, also for a complex reference impedance.

A strip dipole is driven by a lumped port with a real and a complex Z0. The input impedance of the dipole
does not depend on the port: Zin from the power wave S11, Zin = (Z0* + S11 Z0) / (1 - S11), must be the same
for every Z0. The source of a lumped port delivers its port power P as available power, so 1 - |S11|² is the
power the dipole accepts: the power the port sheet delivers, integrated over the port face. The gain is
relative to the incident power, so it does not change with the port power.
"""
import numpy as np
import pytest
import emerge as em

mm = 0.001
F0 = 1.5e9
WL = 299792458 / F0
L, W, G = 0.47 * WL, 4 * mm, 2 * mm


def _dipole(Z0, power=1.0):
    em.cleanup()
    m = em.Simulation("dipole", loglevel="ERROR", reset_settings=True)
    a1 = em.geo.Plate(np.array([-W / 2, 0, G / 2]), np.array([W, 0, 0]), np.array([0, 0, L / 2 - G / 2]))
    a2 = em.geo.Plate(np.array([-W / 2, 0, -L / 2]), np.array([W, 0, 0]), np.array([0, 0, L / 2 - G / 2]))
    a1.set_material(em.lib.PEC)
    a2.set_material(em.lib.PEC)
    port = em.geo.Plate(np.array([-W / 2, 0, -G / 2]), np.array([W, 0, 0]), np.array([0, 0, G]))
    air = em.geo.Sphere(0.6 * WL).background()
    m.mw.set_frequency([F0])
    m.mw.set_resolution(0.2)
    m.commit_geometry()
    m.mesher.set_face_size(port, 0.5 * mm)
    m.mesher.set_boundary_size(a1, 2 * mm)
    m.mesher.set_boundary_size(a2, 2 * mm)
    m.generate_mesh()
    p = m.mw.bc.LumpedPort(port, 1, width=W, height=G, direction=em.ZAX, Z0=Z0, power=power)
    m.mw.bc.AbsorbingBoundary(air.boundary())
    data = m.mw.run_sweep()
    S11 = complex(data.scalar.grid.S(1, 1)[0])
    field = data.field.find(freq=F0)

    # The power the port sheet delivers: (1/2) Re ∫ E · J* over the port face, J = (Vs/h - Ez) / Zs
    xs = np.linspace(-W / 2 + 1e-7, W / 2 - 1e-7, 41)
    zs = np.linspace(-G / 2 + 1e-7, G / 2 - 1e-7, 21)
    X, Z = np.meshgrid(xs, zs, indexing="ij")
    Ez = np.asarray(field.interpolate(X.ravel(), np.zeros(X.size), Z.ravel()).E)[2].reshape(X.shape)
    J = (2 * p.voltage / G - Ez) / p.surfZ
    P_sheet = 0.5 * np.real(np.trapezoid(np.trapezoid(Ez * np.conj(J), zs, axis=1), xs))

    th = np.linspace(0.05, np.pi - 0.05, 15)
    ff = field.farfield(th, np.zeros_like(th), air.boundary())
    gain = np.max(np.abs(ff.gain.norm))
    em.cleanup()
    Zc = complex(Z0)
    Zin = (np.conj(Zc) + S11 * Zc) / (1 - S11)
    return Zin, S11, P_sheet, gain, ff


@pytest.fixture(scope="module")
def dipoles():
    return {Z0: _dipole(Z0) for Z0 in (50, 73 + 40j, 73 - 40j)}


def test_input_impedance_does_not_depend_on_the_reference(dipoles):
    zins = np.array([d[0] for d in dipoles.values()])
    assert np.abs(zins - zins[0]).max() < 0.01 * abs(zins[0]), zins


def test_accepted_power_is_the_power_of_the_port_sheet(dipoles):
    for Z0, (_, S11, P_sheet, *_) in dipoles.items():
        assert abs((1 - abs(S11) ** 2) - P_sheet) < 0.01, (Z0, S11, P_sheet)


def test_gain_is_relative_to_the_incident_power(dipoles):
    gain1 = dipoles[50][3]
    _, S11, P_sheet, gain2, ff = _dipole(50, power=2.0)
    assert ff.Pinc == pytest.approx(2.0)
    assert P_sheet == pytest.approx(2 * (1 - abs(S11) ** 2), rel=0.01)
    assert gain2 == pytest.approx(gain1, rel=1e-6)


def test_complex_reference_needs_a_positive_resistance():
    em.cleanup()
    m = em.Simulation("zcheck", loglevel="ERROR", reset_settings=True)
    port = em.geo.Plate(np.zeros(3), np.array([W, 0, 0]), np.array([0, 0, G]))
    em.geo.Box(10 * mm, 10 * mm, 10 * mm, position=(-5 * mm, -5 * mm, -5 * mm)).background()
    m.commit_geometry()
    with pytest.raises(ValueError):
        m.mw.bc.LumpedPort(port, 1, width=W, height=G, direction=em.ZAX, Z0=-1j * 50)
    em.cleanup()


def test_lumped_port_takes_the_impedance_of_a_pcb_port_face():
    from emerge._emerge.attributes import LumpedPortAttribute

    em.cleanup()
    m = em.Simulation("zattr", loglevel="ERROR", reset_settings=True)
    port = em.geo.Plate(np.zeros(3), np.array([W, 0, 0]), np.array([0, 0, G]))
    port.properties += LumpedPortAttribute(port_number=1, direction=em.ZAX, width=W, height=G, impedance=75.0)
    em.geo.Box(10 * mm, 10 * mm, 10 * mm, position=(-5 * mm, -5 * mm, -5 * mm)).background()
    m.commit_geometry()
    assert m.mw.bc.LumpedPort(port, 1).Z0 == 75.0
    em.cleanup()
