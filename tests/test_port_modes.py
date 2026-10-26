# [Written by Claude] All code in this module was written by Claude (AI).
import numpy as np

import emerge as em

""" Modal analysis of ports with several modes."""

mm = 0.001


def _two_wire_ports(mixed_materials: bool):
    """Two PEC wires in an air-filled PEC box: a TEM line with two modes, a two-mode port at both ends."""
    sim = em.Simulation("twowire")
    box = em.geo.Box(10 * mm, 10 * mm, 20 * mm, position=(-5 * mm, -5 * mm, 0))
    w1 = em.geo.Cylinder(0.6 * mm, 20 * mm, cs=em.GCS.displace(-2 * mm, 0, 0))
    w2 = em.geo.Cylinder(0.6 * mm, 20 * mm, cs=em.GCS.displace(2 * mm, 0, 0))
    air = em.geo.subtract(em.geo.subtract(box, w1), w2)
    sim.commit_geometry()
    sim.mw.set_frequency(3e9)
    sim.generate_mesh()
    p1 = sim.mw.bc.ModalPort(air.face("-z"), 1, number_of_modes=2, mixed_materials=mixed_materials)
    p2 = sim.mw.bc.ModalPort(air.face("+z"), 2, number_of_modes=2, mixed_materials=mixed_materials)
    return sim, p1, p2


def test_repeated_modal_analysis_replaces_modes():
    sim, p1, _ = _two_wire_ports(mixed_materials=False)
    sim.mw.modal_analysis(p1, 2, direct=False, TEM=True, freq=3e9)
    sim.mw.modal_analysis(p1, 2, direct=False, TEM=True, freq=3e9)
    assert [len(modes) for modes in p1.available_modes.values()] == [2]
    em.cleanup()


def test_sweep_keeps_one_set_of_modes_per_frequency():
    # run_sweep analyses the ports at the mean frequency and then at every frequency of the sweep, which is the
    # same frequency here.
    sim, p1, p2 = _two_wire_ports(mixed_materials=True)
    sim.mw.run_sweep()
    for port in (p1, p2):
        assert [len(modes) for modes in port.available_modes.values()] == [2]
        assert port.nmodes == 2
    em.cleanup()


if __name__ == "__main__":
    test_repeated_modal_analysis_replaces_modes()
    test_sweep_keeps_one_set_of_modes_per_frequency()
