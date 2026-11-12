# [Written by Claude] All code in this module was written by Claude (AI).
import numpy as np
import pytest

import emerge as em

""" Wave ports with several modes in an overmoded waveguide: the modes are the dominant ones (largest
propagation constants), and modes below their cutoff decay instead of being dropped from the S-matrix."""

mm = 0.001
A, B = 22.86 * mm, 10.16 * mm      # WR-90: TE10 6.557 GHz, TE20 13.11 GHz, TE01 14.75 GHz (vacuum)
C0 = 299792458


def _kc(m, n):
    return np.hypot(m * np.pi / A, n * np.pi / B)


def test_mode_analysis_finds_the_dominant_modes_of_an_overmoded_guide():
    # At 40 GHz eight or more modes propagate in WR-90; the two requested ones are TE10 and TE20
    em.cleanup()
    f = 40e9
    sim = em.Simulation("overmoded_port")
    box = em.geo.Box(A, 30 * mm, B)
    sim.commit_geometry()
    sim.mw.set_frequency(f)
    sim.mw.set_resolution(0.2)
    sim.generate_mesh()
    port = sim.mw.bc.WavePort(box.face("-y"), 1, number_of_modes=2, modetype=None)
    sim.mw.modal_analysis(port, 2, direct=False, freq=f)
    k0 = 2 * np.pi * f / C0
    betas = [m.beta.real for m in port.get_modes(k0)]
    expected = [np.sqrt(k0**2 - _kc(1, 0) ** 2), np.sqrt(k0**2 - _kc(2, 0) ** 2)]
    assert betas == pytest.approx(expected, rel=2e-3)
    em.cleanup()


def test_mode_below_cutoff_of_a_two_mode_port_decays():
    # Below the TE20 cutoff the analysis finds TE10 only; TE20 is kept as a decaying mode:
    # S(TE20 -> TE20 at the other port) = exp(-alpha L), and the sweep does not fail
    em.cleanup()
    L = 40 * mm
    freqs = [8e9, 10e9, 12e9, 16e9]
    sim = em.Simulation("two_mode_cutoff")
    box = em.geo.Box(A, B, L)
    sim.commit_geometry()
    sim.mw.set_frequency(max(freqs))
    sim.mw.set_resolution(0.2)
    sim.generate_mesh()
    sim.mw.bc.WavePort(box.face("-z"), 1, number_of_modes=2, modetype=None)
    sim.mw.bc.WavePort(box.face("+z"), 2, number_of_modes=2, modetype=None)
    sim.mw.set_frequency(freqs)
    Sp = sim.mw.run_sweep().scalar.grid.Sp
    em.cleanup()
    fc20 = C0 * _kc(2, 0) / (2 * np.pi)
    for f, S in zip(freqs, Sp):
        assert S.shape == (4, 4)
        assert abs(S[2, 0]) == pytest.approx(1.0, abs=0.01), (f, S)       # TE10 passes
        k0 = 2 * np.pi * f / C0
        if f < fc20:
            alpha = np.sqrt(_kc(2, 0) ** 2 - k0**2)
            assert 20 * np.log10(abs(S[3, 1])) == pytest.approx(-8.686 * alpha * L, abs=0.5), (f, S)
            assert abs(S[1, 1]) < 0.03, (f, S)
        else:
            assert abs(S[3, 1]) == pytest.approx(1.0, abs=0.02), (f, S)
