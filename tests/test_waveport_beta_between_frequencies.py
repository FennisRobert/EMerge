# [Written by Claude] All code in this module was written by Claude (AI).
import numpy as np
import pytest

import emerge as em

""" WavePort.get_beta at a frequency where the modes were not solved: extrapolated from the nearest solved
frequency (as ModalPort.get_beta), so is_propagating() sees a cutoff between solved frequencies."""

mm = 0.001
A, B = 22.86 * mm, 10.16 * mm      # WR-90: TE10 6.557 GHz, TE20 13.11 GHz
C0 = 299792458


def test_beta_between_solved_frequencies_follows_the_dispersion():
    em.cleanup()
    sim = em.Simulation("wp_beta")
    box = em.geo.Box(A, B, 40 * mm)
    sim.commit_geometry()
    sim.mw.set_frequency(30e9)
    sim.mw.set_resolution(0.2)
    sim.generate_mesh()
    p1 = sim.mw.bc.WavePort(box.face("-z"), 1, number_of_modes=2, modetype=None)
    sim.mw.bc.WavePort(box.face("+z"), 2, number_of_modes=2, modetype=None)
    sim.mw.set_frequency([20e9, 30e9])
    sim.mw.run_sweep()
    em.cleanup()
    kc = [np.pi / A, 2 * np.pi / A]
    for f in (12e9, 14e9, 25e9):
        k0 = 2 * np.pi * f / C0
        for nr in (1, 2):
            beta = complex(p1.get_beta(k0, nr))
            exact = np.emath.sqrt(k0**2 - kc[nr - 1] ** 2)
            exact = np.conj(exact) if np.imag(exact) > 0 else exact
            assert beta == pytest.approx(exact, rel=5e-3, abs=0.5), (f, nr, beta, exact)
    assert not p1.is_propagating(2 * np.pi * 12e9 / C0, 2)
    assert p1.is_propagating(2 * np.pi * 14e9 / C0, 2)
