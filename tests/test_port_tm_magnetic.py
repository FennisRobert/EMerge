# [Written by Claude] All code in this module was written by Claude (AI).
import numpy as np
import pytest

import emerge as em

"""Port boundary conditions for TM modes (waveguide with PMC walls) and for ports in magnetic material.

The Robin term of a ModalPort used γ = jβ for every mode and left out 1/μr: an empty waveguide with PMC side walls
(TM mode) reflected |S11| = 0.18, and a waveguide filled with μr = 4 reflected |S11| = 0.86 (also with a WavePort).
"""

mm = 0.001
C0 = 299792458.0


def _run(f, boxes_materials, ports, pmc=False, resolution=0.2, neffs=({}, {})):
    em.cleanup()
    sim = em.Simulation("tm_port", loglevel="ERROR")
    boxes = []
    for (size, pos), mat in boxes_materials:
        box = em.geo.Box(*size, position=pos)
        if mat is not None:
            box.set_material(mat)
        boxes.append(box)
    sim.commit_geometry()
    sim.mw.set_frequency(f)
    sim.mw.set_resolution(resolution)
    sim.generate_mesh()
    P = getattr(sim.mw.bc, ports)
    P(boxes[0].face("-z"), 1, **neffs[0])
    P(boxes[-1].face("+z"), 2, **neffs[1])
    if pmc:
        walls = [box.face(s) for box in boxes for s in ("-x", "+x", "-y", "+y")]
        sim.mw.bc.PMC(sum(walls[1:], walls[0]))
    g = sim.mw.run_sweep().scalar.grid
    em.cleanup()
    return np.ravel(g.S(1, 1)), np.ravel(g.S(2, 1))


@pytest.mark.parametrize("ports", ["ModalPort", "WavePort"])
def test_empty_guide_pmc_walls(ports):
    # PMC on all side walls: the first mode is TM-like (H_z = 0) with kc = π/a (cutoff 6.4 GHz).
    a, b, L = 23.3 * mm, 4 * mm, 100 * mm
    f = np.array([8e9, 10e9, 12e9])
    S11, S21 = _run(f, [(((a, b, L), (0, 0, 0)), None)], ports, pmc=True)
    beta = np.sqrt((2 * np.pi * f / C0) ** 2 - (np.pi / a) ** 2)
    assert np.abs(S11).max() < 0.01, S11
    assert np.abs(np.angle(S21 * np.exp(1j * beta * L), deg=True)).max() < 2, S21


@pytest.mark.parametrize("ports", ["ModalPort", "WavePort"])
def test_magnetic_filled_guide(ports):
    # PEC guide filled with μr = 4: only TE10 propagates at 6 GHz (TE20 cutoff 7.5 GHz).
    a, b, L, f, ur = 20 * mm, 5 * mm, 60 * mm, 6e9, 4.0
    S11, S21 = _run(f, [(((a, b, L), (0, 0, 0)), em.Material(ur=ur, name="ferrite"))], ports)
    beta = np.sqrt(ur * (2 * np.pi * f / C0) ** 2 - (np.pi / a) ** 2)
    assert np.abs(S11).max() < 0.01, S11
    assert np.abs(np.angle(S21 * np.exp(1j * beta * L), deg=True)).max() < 2, S21


def test_tm_reflection_at_dielectric_interface():
    # PMC guide, εr = 4 for 0 < z < L1 and air after it: the TM mode is a p-polarized plane wave pair at 20°
    # incidence. S11 is the Fresnel reflection with the TM wave impedance β/(ωε), shifted to port 1.
    f, er, theta = 10e9, 4.0, np.radians(20)
    k0 = 2 * np.pi * f / C0
    a = C0 / f / (2 * np.sqrt(er) * np.sin(theta))
    b, L1, L2 = 4 * mm, 40 * mm, 40 * mm
    kc = np.pi / a
    b1, b2 = np.sqrt(er * k0**2 - kc**2), np.sqrt(k0**2 - kc**2)
    # TM20 also propagates in the dielectric: select the TM10 modes by their effective index
    (S11,), _ = _run(
        f,
        [(((a, b, L1), (0, 0, 0)), em.Material(er=er, name="diel")), (((a, b, L2), (0, 0, L1)), None)],
        "ModalPort",
        pmc=True,
        resolution=0.15,
        neffs=({"target_neff": b1 / k0}, {"target_neff": b2 / k0}),
    )
    Z1, Z2 = b1 / er, b2
    expected = (Z2 - Z1) / (Z2 + Z1) * np.exp(-2j * b1 * L1)
    assert abs(abs(S11) - abs(expected)) < 0.01, (S11, expected)
    assert abs(np.angle(S11 / expected, deg=True)) < 2, (S11, expected)
