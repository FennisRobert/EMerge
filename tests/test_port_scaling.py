# [Written by Claude] All code in this module was written by Claude (AI).
""" How often the modes of ModalPorts and WavePorts are solved: homogeneous ports are solved once and scaled."""
from collections import Counter

import numpy as np
import pytest

import emerge as em
from emerge._emerge.physics.microwave.bcs.port_bcs import ModalPort
from emerge._emerge.physics.microwave.microwave_3d import Microwave3D

mm = 0.001
C0 = 299792458


@pytest.fixture
def solves(monkeypatch):
    """Counts the modal analyses per port number."""
    counts: Counter = Counter()
    original = Microwave3D.modal_analysis

    def counted(self, port, *args, **kwargs):
        counts[port.port_number] += 1
        return original(self, port, *args, **kwargs)

    monkeypatch.setattr(Microwave3D, "modal_analysis", counted)
    return counts


def _guide(freqs, slab_er=None):
    """A WR-90 guide of 30 mm. slab_er fills the lower half with a dielectric (an inhomogeneous port face)."""
    a, b, L = 22.86 * mm, 10.16 * mm, 30 * mm
    sim = em.Simulation("scaling")
    box = em.geo.Box(a, b, L, (-a / 2, -b / 2, 0))
    if slab_er is not None:
        slab = em.geo.Box(a, b / 2, L, (-a / 2, -b / 2, 0)).set_material(em.Material(er=slab_er, name="slab"))
        box = em.geo.subtract(box, slab, remove_tool=False)
    sim.commit_geometry()
    sim.mw.set_frequency(freqs)
    sim.mw.set_resolution(0.2)
    sim.generate_mesh()
    faces = [em.select(*(v.face("-z") for v in sim.state.current_geo_state if v.dim == 3)),
             em.select(*(v.face("+z") for v in sim.state.current_geo_state if v.dim == 3))]
    return sim, faces, a, L


def test_rectangular_modal_port_is_solved_once(solves):
    freqs = np.linspace(8e9, 12e9, 5)
    sim, faces, a, L = _guide(freqs)
    p1 = sim.mw.bc.ModalPort(faces[0], 1)
    sim.mw.bc.ModalPort(faces[1], 2)
    g = sim.mw.run_sweep().scalar.grid
    assert solves == {1: 1, 2: 1}
    assert p1._scalable
    k0 = 2 * np.pi * freqs / C0
    beta = np.sqrt(k0**2 - (np.pi / a) ** 2)
    assert np.abs(g.S(1, 1).ravel()).max() < 0.02
    dphase = np.angle(g.S(2, 1).ravel() * np.exp(1j * beta * L), deg=True)
    assert np.abs(dphase).max() < 2, dphase
    em.cleanup()


def test_scaled_modes_match_solving_every_frequency(solves):
    freqs = np.array([8e9, 10e9, 12e9])
    results = []
    for scale in (True, False):
        sim, faces, _, _ = _guide(freqs)
        if not scale:
            sim.mw._port_scalable = lambda port, k0: False
        sim.mw.bc.ModalPort(faces[0], 1)
        sim.mw.bc.ModalPort(faces[1], 2)
        results.append(sim.mw.run_sweep().scalar.grid.Sp)
        em.cleanup()
    # Scaled: one solve per port. Solved: every frequency (the mean frequency of the sweep, 10 GHz, is one of them).
    assert solves == {1: 1 + 3, 2: 1 + 3}
    assert np.abs(results[0] - results[1]).max() < 5e-3


@pytest.mark.parametrize("how", ["none", "constructor", "method"])
def test_assume_homogeneous(solves, how):
    freqs = np.array([8e9, 10e9, 12e9])
    sim, faces, _, _ = _guide(freqs, slab_er=2.2)
    ports = [sim.mw.bc.ModalPort(f, i + 1, assume_homogeneous=(how == "constructor")) for i, f in enumerate(faces)]
    if how == "method":
        assert all(p.assume_homogeneous() is p for p in ports)
    sim.mw.run_sweep()
    expected = 3 if how == "none" else 1  # every frequency, or once
    assert solves == {1: expected, 2: expected}
    assert all(p._scalable == (how != "none") for p in ports)
    em.cleanup()


def test_changing_assume_homogeneous_clears_modes():
    sim, faces, _, _ = _guide(np.array([10e9]))
    p1 = sim.mw.bc.ModalPort(faces[0], 1)
    sim.mw.modal_analysis(p1, 1, direct=False, freq=10e9)
    assert p1.available_modes
    p1.assume_homogeneous(False)  # unchanged: the modes stay
    assert p1.available_modes
    p1.assume_homogeneous()
    assert not p1.available_modes and p1._scalable is None
    em.cleanup()


def test_pcb_modal_port_assume_homogeneous(solves):
    sim = em.Simulation("pcbports")
    pcb = em.geo.PCB(0.5, unit=mm, material=em.lib.DIEL_RO4003C, layers=2)
    w0 = pcb.calc.z0(50)
    pcb.new(0, 0, w0, (1, 0)).store("p1").straight(10).store("p2")
    pcb.compile_paths(merge=True)
    pcb.determine_bounds(leftmargin=0, topmargin=4, rightmargin=0, bottommargin=4)  # wider than the 5·w0 port
    pcb.modal_port(pcb.load("p1"), 1, 2.0, assume_homogeneous=True)
    pcb.modal_port(pcb.load("p2"), 2, 2.0)
    pcb.generate_pcb()
    pcb.generate_air(2.0)
    sim.commit_geometry()
    sim.mw.set_frequency_range(1e9, 3e9, 3)
    sim.generate_mesh()
    p1, p2 = sorted(sim.mw.bc.oftype(ModalPort), key=lambda p: p.port_number)
    assert p1._assume_homogeneous and not p2._assume_homogeneous
    sim.mw.run_sweep()
    assert solves == {1: 1, 2: 3}  # port 2 is solved at every frequency (the mean, 2 GHz, is one of them)
    em.cleanup()
