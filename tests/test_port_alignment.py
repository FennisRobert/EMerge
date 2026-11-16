# [Written by Claude] All code in this module was written by Claude (AI).
"""align_modes on ports with a degenerate pair of modes: the alignment vectors set the polarization of the pair.

A square guide carries TE10 and TE01 with the same β. The eigen solver returns any basis of the pair, so picking
modes by alignment alone left both port modes at some angle to the axes: a y-polarized wave was split over both
modes of the other port. The pair is now rotated onto the alignment vectors. A dielectric block that is symmetric
in x does not couple the polarizations, so S between the y and the x mode is zero.
"""
import numpy as np
import pytest
import emerge as em

mm = 0.001


@pytest.mark.parametrize("port", ["ModalPort", "WavePort"])
def test_degenerate_pair_follows_alignment_vectors(port):
    a = 22.86 * mm
    em.cleanup()
    sim = em.Simulation("square", loglevel="ERROR", reset_settings=True)
    box = em.geo.Box(a, a, 30 * mm, (-a / 2, -a / 2, 0))
    blk = em.geo.Box(a / 3, a / 4, 6 * mm, (-a / 6, 0, 12 * mm)).set_material(em.Material(er=3.0, name="blk"))
    box = em.geo.subtract(box, blk, remove_tool=False)
    sim.commit_geometry()
    sim.mw.set_frequency([8e9])
    sim.mw.set_resolution(0.15)
    sim.generate_mesh()
    P = getattr(sim.mw.bc, port)
    ports = [P(box.face(side), i + 1, number_of_modes=2) for i, side in enumerate(("-z", "+z"))]
    for p in ports:
        p.align_modes(em.YAX, em.XAX)  # S-matrix index 1, 3: y-polarized; 2, 4: x-polarized
    S = np.abs(sim.mw.run_sweep().scalar.grid.Sp[0])
    em.cleanup()
    assert S[2, 0] > 0.95 and S[3, 1] > 0.95, S  # each polarization passes to the same polarization
    assert S[3, 0] < 0.02 and S[2, 1] < 0.02, S  # and not to the other one
    assert np.abs((S**2).sum(axis=0) - 1).max() < 0.01, S
