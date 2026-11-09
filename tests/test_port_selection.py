# [Written by Claude] All code in this module was written by Claude (AI).
"""Which mode a ModalPort selects (target_neff), quasi-TEM modes with modetype='TEM', and the sign convention of
the voltage path between the port conductors."""
import numpy as np

import emerge as em
from emerge._emerge.physics.microwave.bcs.port_bcs import ModalPort

mm = 0.001
C0 = 299792458


def test_target_neff_selects_the_closest_mode():
    """WR-90 at 16 GHz: TE10 (n_eff 0.913) by default, TE20 (n_eff 0.573) with target_neff=0.55."""
    a, b, f = 22.86 * mm, 10.16 * mm, 16e9
    k0 = 2 * np.pi * f / C0
    neffs = {}
    for target in (None, 0.55):
        sim = em.Simulation("targetneff", loglevel="ERROR", reset_settings=True)
        box = em.geo.Box(a, b, 20 * mm, (-a / 2, -b / 2, 0))
        sim.commit_geometry()
        sim.mw.set_frequency(f)
        sim.mw.set_resolution(0.2)
        sim.generate_mesh()
        port = sim.mw.bc.ModalPort(box.face("-z"), 1, target_neff=target)
        sim.mw.modal_analysis(port, 4, direct=False, freq=f)
        neffs[target] = np.real(port.get_modes(k0)[0].beta) / k0
        em.cleanup()
    assert abs(neffs[None] - np.sqrt(1 - (np.pi / a / k0) ** 2)) < 0.01, neffs
    assert abs(neffs[0.55] - np.sqrt(1 - (2 * np.pi / a / k0) ** 2)) < 0.01, neffs


def _microstrip_port(mode_type: str, swap: bool = False) -> tuple[ModalPort, float]:
    """A 50 Ω microstrip on 0.5 mm RO4003C at 20 GHz, where Hz/Hxy of the quasi-TEM mode (≈ 0.11) exceeds the
    default qtem_limit of 0.1. swap makes the trace the negative terminal."""
    f = 20e9
    sim = em.Simulation("qtem", loglevel="ERROR", reset_settings=True)
    pcb = em.geo.PCB(0.5, unit=mm, material=em.lib.DIEL_RO4003C, layers=2)
    pcb.new(0, 0, pcb.calc.z0(50), (1, 0)).store("p1").straight(10)
    trace = pcb.compile_paths(merge=True)
    pcb.determine_bounds(leftmargin=0, topmargin=4, rightmargin=0, bottommargin=4)
    pcb.modal_port(pcb.load("p1"), 1, 2.0, mode_type=mode_type)
    pcb.generate_pcb()
    pcb.generate_air(2.0)
    sim.commit_geometry()
    sim.mw.set_frequency(f)
    sim.generate_mesh()
    port = sim.mw.bc.oftype(ModalPort)[0]
    if swap:
        port.set_terminals(negative=trace)
    sim.mw.modal_analysis(port, 1, direct=False, freq=f)
    return port, 2 * np.pi * f / C0


def test_quasi_tem_mode_with_forced_tem():
    port, k0 = _microstrip_port("any")
    assert port.get_modes(k0)[0].modetype == "TE"  # classified by qtem_limit: no Z0 from a voltage path
    assert not port.voltage_integration_line
    em.cleanup()

    port, k0 = _microstrip_port("TEM")
    mode = port.get_modes(k0)[0]
    assert mode.modetype == "TEM"
    assert 45 < mode.Z0 < 56, mode.Z0
    # The voltage path runs from the trace (z = 0) to the ground (z = -0.5 mm)
    line = np.array(port.voltage_integration_line[0].cmid)
    assert abs(line[2, 0]) < 0.05 * mm and abs(line[2, -1] + 0.5 * mm) < 0.05 * mm, line[:, [0, -1]]
    em.cleanup()


def test_set_terminals_sets_the_sign_with_two_conductors():
    port, k0 = _microstrip_port("TEM")
    z0 = port.get_modes(k0)[0].Z0
    em.cleanup()

    port, k0 = _microstrip_port("TEM", swap=True)
    line = np.array(port.voltage_integration_line[0].cmid)
    assert abs(line[2, 0] + 0.5 * mm) < 0.05 * mm and abs(line[2, -1]) < 0.05 * mm, line[:, [0, -1]]
    assert abs(port.get_modes(k0)[0].Z0 - z0) < 1e-6 * z0
    em.cleanup()


def test_reset_keeps_the_terminals():
    sim = em.Simulation("terminals", loglevel="ERROR", reset_settings=True)
    box = em.geo.Box(10 * mm, 10 * mm, 10 * mm)
    sim.commit_geometry()
    sim.mw.set_frequency(1e9)
    sim.generate_mesh()
    port = sim.mw.bc.ModalPort(box.face("-z"), 1)
    port.set_terminals(positive=box.face("-x"), negative=box.face("+x"))
    plus, minus = list(port.plus_terminal), list(port.minus_terminal)
    port.reset()
    assert port.plus_terminal == plus and port.minus_terminal == minus
    em.cleanup()
