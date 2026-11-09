# [Written by Claude] All code in this module was written by Claude (AI).
import numpy as np
import pytest

import emerge as em
from emerge._emerge.physics.microwave.bcs.port_bcs import _decaying_beta

""" Wave ports below the cutoff of their mode: the mode decays, gamma = alpha > 0 at every port."""

mm = 0.001
A, B, L = 22.86 * mm, 10.16 * mm, 40 * mm      # WR-90, TE10 cutoff 6.557 GHz (vacuum)


def test_decaying_beta_takes_the_branch_of_a_wave_leaving_the_port():
    # np.sqrt alone gives +2j or -2j for -4 depending on the sign of the zero imaginary part
    assert _decaying_beta(complex(-4.0, 0.0)) == -2j
    assert _decaying_beta(complex(-4.0, -0.0)) == -2j
    assert _decaying_beta(complex(9.0, 0.0)) == 3.0
    assert _decaying_beta((3.0 - 0.5j) ** 2) == pytest.approx(3.0 - 0.5j)


def _guide(er: float, f_mesh: float, freqs):
    em.cleanup()
    sim = em.Simulation("guide_below_cutoff")
    box = em.geo.Box(A, B, L)
    if er != 1.0:
        box.set_material(em.Material(er=er, name="fill"))
    sim.commit_geometry()
    sim.mw.set_frequency(f_mesh)
    sim.generate_mesh()
    sim.mw.bc.ModalPort(box.face("-z"), 1, modetype="TE", mixed_materials=True)
    sim.mw.bc.ModalPort(box.face("+z"), 2, modetype="TE", mixed_materials=True)
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
