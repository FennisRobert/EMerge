# [Written by Claude]
"""Conductor loss of a ThinConductor stripline in air against Wheeler's incremental inductance rule.

The trace is a zero-thickness copper sheet centred between PEC ground planes, so only the strip is lossy.
Wheeler's formulas describe a strip of the real copper thickness. The edge current of a zero-thickness
sheet is singular, so the simulated loss depends on the edge mesh size. For exactly this geometry and
mesh (domain size 1 mm, trace boundary size 0.5 mm) the theory loss is 0.81 times the simulated loss.
This is a regression test of the ThinConductor loss for that fixed setup, not a general validation.
"""
import numpy as np

ETA0 = 376.730313
MU0 = 4e-7 * np.pi
CORRECTION_FACTOR = 0.81  # theory loss / simulated loss for this geometry and mesh


def z0_stripline(w: float, b: float, t: float) -> float:
    """Characteristic impedance of a centred stripline in air (Wheeler 1978, about 0.5 % accurate)."""
    x = t / b
    m = 2 / (1 + 2 / 3 * x / (1 - x))
    dw = (b - t) * x / (np.pi * (1 - x)) * (
        1 - 0.5 * np.log((x / (2 - x)) ** 2 + (0.0796 * x / (w / b + 1.1 * x)) ** m)
    )
    r = (b - t) / (w + dw)
    return ETA0 / (4 * np.pi) * np.log(1 + 4 / np.pi * r * (8 / np.pi * r + np.sqrt((8 / np.pi * r) ** 2 + 6.27)))


def strip_conductor_loss(f: np.ndarray, w: float, b: float, t: float, sigma: float) -> np.ndarray:
    """Conductor attenuation (Np/m) of the strip only (Wheeler's incremental inductance rule)."""
    rs = np.sqrt(np.pi * f * MU0 / sigma)
    z0 = z0_stripline(w, b, t)
    dn = 1e-9
    dz_dn = (z0_stripline(w - 2 * dn, b, t - 2 * dn) - z0) / dn
    return rs / (2 * ETA0 * z0) * dz_dn


def test_thin_conductor_stripline_loss():
    import emerge as em

    mm = 0.001
    th, w0, ds, L = 1.0 * mm, 1.3 * mm, 3 * mm, 100 * mm

    em.cleanup()
    sim = em.Simulation("ThinConductorLoss", loglevel="ERROR")
    pcb = em.geo.PCB(th, unit=1.0, layers=3, thick_traces=False, trace_material=em.lib.COPPER)
    pcb.new(0, 0, w0, (1, 0), 1)[0].straight(L)[1]
    trace = pcb.compile_paths(True)
    pcb.modal_port(0, 1, 0, width=2 * ds)
    pcb.modal_port(1, 2, 0, width=2 * ds)
    pcb.determine_bounds(0, ds, 0, ds)
    diels = pcb.generate_pcb(True)
    sim.commit_geometry()

    sim.mw.set_frequency_range(1e9, 10e9, 3)
    sim.mesher.set_boundary_size(trace, 0.5 * mm)
    sim.mesher.set_domain_size(diels, 1 * mm)
    sim.generate_mesh()
    data = sim.mw.run_sweep()

    g = data.scalar.grid
    p_loss_sim = 1 - np.abs(g.S(1, 1)) ** 2 - np.abs(g.S(2, 1)) ** 2

    alpha = strip_conductor_loss(g.freq, w0, th, pcb.trace_thickness, em.lib.COPPER.cond.scalar(1e9))
    p_loss_theory = (1 - np.exp(-2 * alpha * L)) / CORRECTION_FACTOR

    rel_error = np.abs(p_loss_sim / p_loss_theory - 1)
    assert np.all(rel_error < 0.05), dict(freq=g.freq, sim=p_loss_sim, theory=p_loss_theory, rel_error=rel_error)
    em.cleanup()
