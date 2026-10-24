# [Written by Claude] All code in this module was written by Claude (AI).
"""Radiated power of the far field with a symmetry plane.

A strip dipole along z is solved once in full and once as the half x >= 0 with a PMC plane at x = 0
(half-width lumped port with twice the impedance). With 1 W into each model, the far field of the half
model mirrored with syms=["Hx"] must give the same radiated power and gain as the full model.
"""
import numpy as np
import pytest

import emerge as em

MM = 1e-3


def _dipole(half: bool):
    L, g, w, R = 46 * MM, 2 * MM, 2 * MM, 40 * MM
    x0, ww = (0.0, w / 2) if half else (-w / 2, w)
    m = em.Simulation("SymmetryDipoleHalf" if half else "SymmetryDipoleFull")
    air = em.geo.Box(R if half else 2 * R, 2 * R, 2 * R, position=(0 if half else -R, -R, -R))
    arms = [em.geo.Plate(np.array([x0, 0, z]), np.array([ww, 0, 0]), np.array([0, 0, L / 2 - g / 2]))
            for z in (g / 2, -L / 2)]
    port = em.geo.Plate(np.array([x0, 0, -g / 2]), np.array([ww, 0, 0]), np.array([0, 0, g]))
    for a in arms:
        a.set_material(em.lib.PEC)
    m.mw.set_resolution(0.15)
    m.mw.set_frequency(3e9)
    m.commit_geometry()
    for a in arms:
        m.mesher.set_boundary_size(a, 1.5 * MM)
    m.mesher.set_face_size(port, 0.5 * MM)
    m.generate_mesh()
    m.mw.bc.LumpedPort(port, 1, width=ww, height=g, direction=em.ZAX, Z0=150 if half else 75)
    if half:
        rad = air.face(no="left")
        m.mw.bc.PMC(air.face("left"))
    else:
        rad = air.boundary()
    m.mw.bc.AbsorbingBoundary(rad)
    field = m.mw.run_sweep().field[0]
    ff = field.farfield_2d((1, 0, 0), (0, 1, 0), rad, syms=["Hx"] if half else None, Npoints=181)
    return ff.Ptot, np.max(ff.gain.norm)


@pytest.mark.filterwarnings("ignore")
def test_symmetry_plane_keeps_radiated_power():
    p_full, g_full = _dipole(half=False)
    p_half, g_half = _dipole(half=True)
    assert abs(g_half - g_full) / g_full < 0.03
    assert abs(p_half - p_full) / p_full < 0.05
