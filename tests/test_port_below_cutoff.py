# [Written by Claude] All code in this module was written by Claude (AI).
import numpy as np
import pytest

import emerge as em

""" Wave ports below the cutoff of their mode: the mode decays, gamma = alpha > 0 at every port."""

mm = 0.001
A, B, L = 22.86 * mm, 10.16 * mm, 40 * mm      # WR-90, TE10 cutoff 6.557 GHz (vacuum)


def _guide(er: float, f_mesh: float, freqs):
    em.cleanup()
    sim = em.Simulation("guide_below_cutoff", reset_settings=True)
    box = em.geo.Box(A, B, L)
    if er != 1.0:
        box.set_material(em.Material(er=er, name="fill"))
    sim.commit_geometry()
    sim.mw.set_frequency(f_mesh)
    sim.generate_mesh()
    sim.mw.bc.ModalPort(box.face("-z"), 1, modetype="TE")
    sim.mw.bc.ModalPort(box.face("+z"), 2, modetype="TE")
    sim.mw.set_frequency(list(freqs))
    Sp = sim.mw.run_sweep().scalar.grid.Sp
    em.cleanup()
    return Sp


@pytest.mark.parametrize("er", [1.0, 2.1])
def test_empty_guide_below_cutoff_attenuates_as_exp_minus_alpha_l(er):
    # Below cutoff S21 = exp(-alpha L), alpha = sqrt(kc^2 - er k0^2); above it the guide passes everything.
    # er = 2.1: the dispersion of the mode solved at another frequency has to include the filling.
    fc = 299792458 / (2 * A * np.sqrt(er))
    freqs = np.array([0.6, 0.75, 0.9, 1.2, 1.5]) * fc
    Sp = _guide(er, 1.5 * fc, freqs)
    for f, S in zip(freqs, Sp):
        k0 = 2 * np.pi * f / 299792458
        if f < fc:
            alpha = np.sqrt((np.pi / A) ** 2 - er * k0**2)
            assert 20 * np.log10(abs(S[1, 0])) == pytest.approx(-8.686 * alpha * L, abs=0.3), (f, S)
            assert abs(S[0, 0]) < 0.03, (f, S)
        else:
            assert abs(abs(S[1, 0]) - 1) < 0.01, (f, S)


def test_solution_frequency_below_cutoff_uses_the_mode_of_a_higher_frequency():
    # Only a frequency below cutoff: the mode analysis finds no propagating mode there, the port takes the
    # shape of a higher frequency and absorbs the decaying mode instead of failing.
    fc = 299792458 / (2 * A)
    Sp = _guide(1.0, 0.75 * fc, [0.75 * fc])
    alpha = np.sqrt((np.pi / A) ** 2 - (2 * np.pi * 0.75 * fc / 299792458) ** 2)
    assert 20 * np.log10(abs(Sp[0][1, 0])) == pytest.approx(-8.686 * alpha * L, abs=0.3)


def test_lossy_guide_solved_per_frequency_below_cutoff():
    # A conducting filling is not scaled: the modes are solved at every frequency, and the frequencies below cutoff
    # keep the modes of the nearest one. S21 = exp(-Re(γ) L) with γ = sqrt(kc² - εr,c k0²).
    er, sigma = 2.1, 0.02
    fc = 299792458 / (2 * A * np.sqrt(er))
    freqs = np.array([0.75, 0.9, 1.2, 1.5]) * fc
    em.cleanup()
    sim = em.Simulation("lossy_below_cutoff", reset_settings=True)
    box = em.geo.Box(A, B, L).set_material(em.Material(er=er, cond=sigma, name="lossy"))
    sim.commit_geometry()
    sim.mw.set_frequency(1.5 * fc)
    sim.generate_mesh()
    p1 = sim.mw.bc.ModalPort(box.face("-z"), 1, modetype="TE")
    sim.mw.bc.ModalPort(box.face("+z"), 2, modetype="TE")
    sim.mw.set_frequency(list(freqs))
    Sp = sim.mw.run_sweep().scalar.grid.Sp
    assert p1._scalable is False
    for f, S in zip(freqs, Sp):
        k0 = 2 * np.pi * f / 299792458
        erc = er - 1j * sigma / (2 * np.pi * f * 8.8541878128e-12)
        gamma = np.sqrt((np.pi / A) ** 2 - erc * k0**2 + 0j)
        expected = -8.686 * abs(gamma.real) * L
        assert 20 * np.log10(abs(S[1, 0])) == pytest.approx(expected, abs=0.3), (f, S)
    em.cleanup()


def test_rectangular_waveguide_port_below_cutoff_decays():
    em.cleanup()
    sim = em.Simulation("rect_below_cutoff", reset_settings=True)
    box = em.geo.Box(A, B, L)
    sim.commit_geometry()
    sim.mw.set_frequency(10e9)
    sim.generate_mesh()
    port = sim.mw.bc.RectangularWaveguide(box.face("-z"), 1, dims=(A, B))
    k0 = 2 * np.pi * 4e9 / 299792458
    alpha = np.sqrt((np.pi / A) ** 2 - k0**2)
    assert port.get_beta(k0) == pytest.approx(-1j * alpha)
    assert np.real(port.get_gamma(k0)) > 0
    em.cleanup()
