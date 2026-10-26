# [Written by Claude] All code in this module was written by Claude (AI).
import numpy as np

import emerge as em

""" Wave ports with several modes of different propagation constant (MultiModeWavePort)."""

mm = 0.001


def _coupled_microstrip(port: str):
    # Two coupled microstrip lines on FR4 in a PEC box: an even and an odd mode with about 9 % different
    # propagation constants at each port.
    W, L, h, H, w, gap = 8 * mm, 20 * mm, 0.5 * mm, 4 * mm, 0.8 * mm, 0.4 * mm
    em.cleanup()
    sim = em.Simulation("coupled_microstrip")
    sub = em.geo.Box(W, L, h, position=(-W / 2, 0, 0)).set_material(em.Material(er=4.4, name="FR4"))
    air = em.geo.Box(W, L, H, position=(-W / 2, 0, h))
    s1 = em.geo.XYPlate(w, L, position=(-gap / 2 - w, 0, h)).set_material(em.lib.PEC)
    s2 = em.geo.XYPlate(w, L, position=(gap / 2, 0, h)).set_material(em.lib.PEC)
    sim.commit_geometry()
    sim.mw.set_frequency_range(4e9, 8e9, 3)
    sim.mesher.set_boundary_size(em.select(s1, s2), 0.3 * mm)
    sim.generate_mesh()
    faces = [em.select(sub.face("-y"), air.face("-y")), em.select(sub.face("+y"), air.face("+y"))]
    if port == "MultiModeWavePort":
        ports = [sim.mw.bc.MultiModeWavePort(f, i + 1, number_of_modes=2) for i, f in enumerate(faces)]
    else:
        ports = [sim.mw.bc.ModalPort(f, i + 1, number_of_modes=2, mixed_materials=True) for i, f in enumerate(faces)]
    data = sim.mw.run_sweep()
    k0 = 2 * np.pi * 4e9 / 299792458
    betas = [np.real(m.beta) for m in ports[0].get_modes(k0)]
    Sp = data.scalar.grid.Sp
    em.cleanup()
    return Sp, betas


def test_coupled_microstrip_even_and_odd_mode_without_reflection():
    Sp, betas = _coupled_microstrip("MultiModeWavePort")
    assert len(betas) == 2 and abs(betas[0] - betas[1]) / betas[0] > 0.05, betas
    for S in Sp:
        # Uniform lines: every mode passes without reflection and without conversion into the other mode.
        assert 20 * np.log10(np.abs(S[:2, :2]).max()) < -28, S
        assert 20 * np.log10(abs(S[2, 1])) < -45 and 20 * np.log10(abs(S[3, 0])) < -45, S
        # The model is lossless: S is unitary.
        assert np.abs(S.conj().T @ S - np.eye(4)).max() < 1e-5


def test_modal_port_reflects_the_mode_of_the_other_propagation_constant():
    # Reference for the test above: ModalPort uses the propagation constant of the first mode for both.
    Sp, _ = _coupled_microstrip("ModalPort")
    assert max(20 * np.log10(abs(S[1, 1])) for S in Sp) > -28


def test_two_wires_degenerate_modes():
    # Two PEC wires in an air-filled PEC box: two TEM modes with the same propagation constant.
    em.cleanup()
    sim = em.Simulation("twowire")
    box = em.geo.Box(10 * mm, 10 * mm, 20 * mm, position=(-5 * mm, -5 * mm, 0))
    w1 = em.geo.Cylinder(0.6 * mm, 20 * mm, cs=em.GCS.displace(-2 * mm, 0, 0))
    w2 = em.geo.Cylinder(0.6 * mm, 20 * mm, cs=em.GCS.displace(2 * mm, 0, 0))
    air = em.geo.subtract(em.geo.subtract(box, w1), w2)
    sim.commit_geometry()
    sim.mw.set_frequency(3e9)
    sim.generate_mesh()
    sim.mw.bc.MultiModeWavePort(air.face("-z"), 1, number_of_modes=2)
    sim.mw.bc.MultiModeWavePort(air.face("+z"), 2, number_of_modes=2)
    S = sim.mw.run_sweep().scalar.grid.Sp[0]
    em.cleanup()
    assert np.abs(S.conj().T @ S - np.eye(4)).max() < 1e-5
    assert np.abs(S[:2, :2]).max() < 0.03, S  # mesh error of the faceted wires


if __name__ == "__main__":
    test_two_wires_degenerate_modes()
    test_coupled_microstrip_even_and_odd_mode_without_reflection()
    test_modal_port_reflects_the_mode_of_the_other_propagation_constant()
